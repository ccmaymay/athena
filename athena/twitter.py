from concrete.util.tokenization import get_tagged_tokens, get_tokens
from concrete.util.redis_io import read_communication_from_buffer
from concrete.util.unnone import lun

from calendar import timegm
from time import strptime
from concrete.util.twitter import CREATED_AT_FORMAT
from redis import Redis

from multiprocessing import Pool, Process
import logging

from time import sleep


def _find_lid(lid_list, tool):
    for lid in lid_list:
        if lid.metadata.tool.lower() == tool.lower():
            return lid
    raise KeyError(u'no probability one LID from tool %s' % tool)


def _find_lang(lid):
    for (lang, prob) in lid.languageToProbabilityMap.items():
        if prob == 1:
            return lang
    raise KeyError(u'no language with probability 1 in LID')


def _comm_hashtags(comm):
    return lun(comm.communicationMetadata.tweetInfo.entities.hashtagList)


def _comm_tokens(comm):
    for section in lun(comm.sectionList):
        for sentence in lun(section.sentenceList):
            bad_ids = set(
                tag.tokenIndex
                for tag in get_tagged_tokens(sentence.tokenization, 'twitter')
                # if tag.tag != 'HASHTAG'
            )
            for token in get_tokens(sentence.tokenization):
                if token.tokenIndex not in bad_ids:
                    yield token


def _comm_is_tokenized(comm):
    return True if (
        comm.sectionList and
        comm.sectionList[0].sentenceList and
        comm.sectionList[0].sentenceList[0].tokenization and
        comm.sectionList[0].sentenceList[0].tokenization.tokenList and
        comm.sectionList[0].sentenceList[0].tokenization.tokenTaggingList
    ) else False


def _comm_timestamp(comm):
    return unicode(timegm(strptime(
        comm.communicationMetadata.tweetInfo.createdAt,
        CREATED_AT_FORMAT
    )))


def _comm_id(comm):
    return comm.id


def _parse_comm_buf(buf, context_size):
    comm = read_communication_from_buffer(buf)
    if comm.lidList is None:
        return None
    if not _comm_is_tokenized(comm):
        return None
    if _find_lang(_find_lid(comm.lidList, 'Twitter LID')) != 'eng':
        return None
    tokens = list(_comm_tokens(comm))
    if len(tokens) < context_size:
        return None
    return dict(
        id=_comm_id(comm),
        tokens=tuple(t.text.lower() for t in tokens),
        hashtags=tuple(t.text.lower() for t in _comm_hashtags(comm)),
        timestamp=_comm_timestamp(comm),
    )


def _batch_key(data_key, batch_num):
    return '%s/batch:%i' % (data_key, batch_num)


def _rpop_batch(args):
    try:
        (
            host, port, data_key, context_size, batch_size, sleep_interval,
            batch_num
        ) = args[:7]
        db = Redis(host, port)
        batch_data_key = _batch_key(data_key, batch_num)
        while not db.exists(batch_data_key):
            sleep(sleep_interval)
        p = db.pipeline()
        for i in xrange(batch_size):
            p.rpop(batch_data_key)
        batch = p.execute()
        return tuple(
            _parse_comm_buf(b, context_size)
            for b in batch if b is not None
        )
    except KeyboardInterrupt:
        return None


def _map_batches(host, port, data_key, batch_size, sleep_interval,
                 max_batch_lag):
    db = Redis(host, port)
    batch_num = 0
    while True:
        while (db.llen(data_key) < batch_size or (
                   batch_num >= max_batch_lag and
                   db.exists(_batch_key(data_key, batch_num - max_batch_lag))
               )):
            sleep(sleep_interval)
        p = db.pipeline()
        for i in xrange(batch_size):
            p.rpoplpush(data_key, _batch_key(data_key, batch_num))
        p.execute()
        batch_num += 1


def _loop(*args):
    i = 0
    while True:
        yield args + (i,)
        i += 1


def extract_redis_tweet_text(host, port, input_key, num_processes=16,
                             batch_size=10000, sleep_interval=0.1,
                             context_size=5, max_batch_lag=100):
    '''
    Extract text and hashtags from English tweets in redis (as concrete
    communications), returning a generator of dictionaries with the
    following keys:

        id: communication id (unicode)
        tokens: list of tokens (list of unicodes)
        hashtags: list of hashtags (list of unicodes)
        timestamp: tweet time as number of seconds since epoch (int)

    Tweets are popped off the right side of the queue (list) in
    key input_key in the redis database at address host:port

    Skip tweets that do not have LID annotations, do not have
    populated tokenizations, or do not have as many tokens as
    context_size (for the purpose of downstream embedding
    training/inference).

    Use multiprocessing to improve throughput.  A single mapper
    process pops batches of batch_size communications from input_key and
    pushes them onto a temporary key formed by concatenating input_key
    with the batch index.  num_processes child processes concurrently
    pop communications from respective temporary batch keys, parse, and
    filter them.  If the mapper sees less than batch_size tweets in its
    input queue or a temporary key from max_batch_lag batches ago exists,
    it sleeps for sleep_interval seconds.  The other processes also
    sleep for sleep_interval seconds until their inputs are created.
    Thus num_processes, batch_size, sleep_interval, and max_batch_lag
    interact in improving---or degrading---throughput over the naive
    serial implementation: it is important to set batch_size large
    enough that process runtime is not dominated by overhead, set
    sleep_interval large enough that the redis server is not DDOSed by
    rpops from our child processes, set num_processes large enough to
    take advantage of the multiprocessing facilities of the host, and
    set max_batch_lag small enough that the redis server does not blow
    up in memory usage.  The defaults perform well empirically.
    '''

    map_process = Process(
        target=_map_batches,
        args=(host, port, input_key, batch_size, sleep_interval,
              max_batch_lag),
    )
    map_process.start()

    pool = Pool(num_processes)
    batches = pool.imap(
        _rpop_batch,
        _loop(host, port, input_key, context_size, batch_size, sleep_interval)
    )

    try:
        for b in batches:
            if not map_process.is_alive():
                logging.warning('mapper is not alive, exiting')
                break
            if b is None:
                logging.warning('caught exception from child, exiting')
                break
            for d in b:
                if d is not None:
                    yield d

    except KeyboardInterrupt:
        logging.info('caught keyboard interrupt, exiting')

    logging.info('terminating and joining worker pool')
    pool.terminate()
    pool.join()

    logging.info('terminating and joining mapper process')
    map_process.terminate()
    map_process.join()

    logging.info('done')
