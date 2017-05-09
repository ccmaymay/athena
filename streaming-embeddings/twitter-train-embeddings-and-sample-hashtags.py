#!/usr/bin/env python2.7


import logging

from athena.core import SGNSModel, LMMultiLabelDocStore
from athena.util import URLCommunicationReader, mkdirp_parent
from athena.twitter import extract_redis_tweet_text
from multiprocessing import Process, Pipe
from time import gmtime, strftime


TIME_FORMAT = '%Y-%m-%d %H:%M:%S GMT'


def sample_hashtags(ns, conn):
    logging.info('checking doc store parent directory ...')
    mkdirp_parent(ns.doc_store_path)

    if ns.load_doc_store:
        logging.info('loading doc store from disk ...')
        doc_store = LMMultiLabelDocStore.load(ns.doc_store_path)
    else:
        logging.info('initializing doc store ...')
        doc_store = LMMultiLabelDocStore.from_namespace(ns)

    logging.info('sampling hashtags ...')

    i = 0

    try:
        d = conn.recv()

        while d is not None:
            if d['message_type'] == 'ping':
                conn.send(dict(message_type='pong'))

            elif d['message_type'] == 'doc':
                d['labels'] = d['hashtags']

                i += 1

                doc_store.increment(d)

                if i % ns.save_interval == 0:
                    logging.info(u'saving after %d docs...' % i)
                    doc_store.dump(ns.doc_store_path)

            else:
                logging.warning(
                    u'got unknown message type %s from main, saving' %
                    d['message_type'])

                break

            d = conn.recv()

    except KeyboardInterrupt:
        logging.info(
            'caught keyboard interrupt, saving; interrupt again to cancel'
        )

    logging.info(u'saving after %d docs...' % i)
    doc_store.dump(ns.doc_store_path)

    logging.info('done')


def train_embeddings(ns, conn):
    logging.info('checking model parent directory ...')
    mkdirp_parent(ns.model_path)

    if ns.load_model:
        logging.info('loading model from disk ...')
        model = SGNSModel.load(ns.model_path)
    else:
        logging.info('initializing model ...')
        model = SGNSModel.from_namespace(ns)

    logging.info('training embeddings ...')

    i = 0
    token_count = 0

    try:
        d = conn.recv()

        while d is not None:
            if d['message_type'] == 'ping':
                conn.send(dict(message_type='pong'))

            elif d['message_type'] == 'doc':
                tokens = list(d['tokens'])

                i += 1
                token_count += len(tokens)

                if i % ns.log_interval == 0:
                    logging.info(
                        u'training on doc %d (%s) (seen %d tokens) ...' %
                        (i, strftime(TIME_FORMAT,
                         gmtime(float(d['timestamp'])))
                            if 'timestamp' in d else 'no timestamp',
                         token_count)
                    )

                model.sentence_train(tokens)

                if i % ns.save_interval == 0:
                    logging.info(
                        u'saving after %d docs (%d tokens)...' %
                        (i, token_count))
                    model.dump(ns.model_path)

            else:
                logging.warning(
                    u'got unknown message type %s from main, saving' %
                    d['message_type'])

                break

            d = conn.recv()

        logging.info('received signal from parent, exiting')

    except KeyboardInterrupt:
        logging.info(
            'caught keyboard interrupt, saving; interrupt again to cancel'
        )

    logging.info(
        u'saving after %d docs (%d tokens)...' %
        (i, token_count))
    model.dump(ns.model_path)

    logging.info('done')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'train SpaceSaving word2vec model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_url', type=str,
                        help='URL to redis dataset ('
                             'redis://r8n00:6000/data-key')
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    parser.add_argument('doc_store_path', type=str,
                        help='path of hashtag-example doc store on disk')
    parser.add_argument('--load-model', action='store_true',
                        help='load existing model from disk instead of '
                             'creating new')
    parser.add_argument('--load-doc-store', action='store_true',
                        help='load existing doc store from disk instead of '
                             'creating new')
    parser.add_argument('--save-interval', type=int,
                        help='interval (in number of documents) at which '
                             'to save model',
                        default=1000000)
    parser.add_argument('--log-interval', type=int,
                        help='interval (in number of documents) at which '
                             'to log status',
                        default=1000)
    parser.add_argument('--num-processes', type=int,
                        help='number of processes to use for preprocessing',
                        default=6)
    parser.add_argument('--batch-size', type=int,
                        help='document batch size for preprocessing',
                        default=10000)
    parser.add_argument('--sleep-interval', type=float,
                        help='preprocessing I/O sleep interval (in seconds)',
                        default=0.1)
    parser.add_argument('--max-batch-lag', type=int,
                        help='maximum number of batches to cache in redis',
                        default=100)
    parser.add_argument('--stop-year', type=int,
                        help='stop processing at this year')
    parser.add_argument('--stop-month', type=int,
                        help='stop processing at this month')
    SGNSModel.add_arguments(parser)
    LMMultiLabelDocStore.add_arguments(parser)
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('initializing data source ...')
    (host, port, key, params) = URLCommunicationReader.parse_redis_url(
        ns.data_url
    )
    if not params.get('block', False):
        raise ValueError('require block=True for twitter data')
    if not params.get('pop', False):
        raise ValueError('require pop=True for twitter data')

    context_size = 2 * ns.symm_context + 1

    logging.info('starting training subprocess ...')
    (train_embeddings_conn, child_conn) = Pipe()
    embedding_process = Process(target=train_embeddings, args=(ns, child_conn))
    embedding_process.start()

    logging.info('starting hashtag sampling subprocess ...')
    (sample_hashtags_conn, child_conn) = Pipe()
    hashtag_process = Process(target=sample_hashtags, args=(ns, child_conn))
    hashtag_process.start()

    logging.info('sending data to learner ...')
    try:
        for d in extract_redis_tweet_text(host, port, key,
                                          num_processes=ns.num_processes,
                                          batch_size=ns.batch_size,
                                          sleep_interval=ns.sleep_interval,
                                          context_size=context_size,
                                          max_batch_lag=ns.max_batch_lag):
            if not embedding_process.is_alive():
                logging.warning('embedding child process not alive')
                break
            if not hashtag_process.is_alive():
                logging.warning('hashtag child process not alive')
                break

            time_struct = gmtime(float(d['timestamp']))
            if ((ns.stop_year is not None or ns.stop_month is not None) and
                    (ns.stop_year is None or
                        time_struct.tm_year == ns.stop_year) and
                    (ns.stop_month is None or
                        time_struct.tm_mon == ns.stop_month)):
                logging.info('stream has reached year %d, month %d' %
                             (ns.stop_year, ns.stop_month))
                break

            d['message_type'] = 'doc'

            train_embeddings_conn.send(d)
            sample_hashtags_conn.send(d)

            train_embeddings_conn.send(dict(message_type='ping'))
            if train_embeddings_conn.recv().get('message_type') != 'pong':
                logging.warning('embedding child process did not pong to ping')
                break

            sample_hashtags_conn.send(dict(message_type='ping'))
            if sample_hashtags_conn.recv().get('message_type') != 'pong':
                logging.warning('hashtag child process did not pong to ping')
                break

        logging.info('signaling children to exit')

        train_embeddings_conn.send(None)
        sample_hashtags_conn.send(None)

    except KeyboardInterrupt:
        logging.info(
            'caught keyboard interrupt, signaling children to save; '
            'interrupt again to cancel'
        )
        if embedding_process.is_alive():
            train_embeddings_conn.send(None)
        else:
            logging.warning('embedding child process not alive')
        if hashtag_process.is_alive():
            sample_hashtags_conn.send(None)
        else:
            logging.warning('hashtag child process not alive')

    logging.info('joining children (may wait for saves)')

    train_embeddings_conn.close()
    sample_hashtags_conn.close()

    embedding_process.join()
    hashtag_process.join()

    logging.info('done')


if __name__ == '__main__':
    main()
