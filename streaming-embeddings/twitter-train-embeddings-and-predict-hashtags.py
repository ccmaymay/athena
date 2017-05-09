#!/usr/bin/env python2.7


import logging

from athena.core import SGNSModel
from athena.util import URLCommunicationReader, mkdirp_parent
from athena.twitter import extract_redis_tweet_text
from multiprocessing import Process, Pipe
from time import gmtime, strftime
import cPickle as pickle
import numpy as np


TIME_FORMAT = '%Y-%m-%d %H:%M:%S GMT'


def train_and_predict(ns, conn):
    logging.info('checking model parent directory ...')
    mkdirp_parent(ns.model_path)

    logging.info('loading model from disk ...')
    model = SGNSModel.load(ns.model_path)

    logging.info('loading classifier from disk ...')
    with open(ns.classifier_path, 'rb') as f:
        d = pickle.load(f)
        classifier = d['classifier']
        scaler = d['scaler']
        top_hashtags = d['top_hashtags']
    top_hashtags_index = dict(
        (hashtag, i) for (i, hashtag) in enumerate(top_hashtags))

    logging.info('training embeddings / predicting hashtags ...')

    i = 0
    token_count = 0

    with open(ns.gold_path, 'w') as gold_f:
        with open(ns.prediction_path, 'w') as prediction_f:
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
                                u'training on doc '
                                u'%d (%s) (seen %d tokens) ...' % (
                                    i,
                                    strftime(
                                        TIME_FORMAT,
                                        gmtime(float(d['timestamp']))
                                    ) if 'timestamp' in d else 'no timestamp',
                                    token_count
                                )
                            )

                        model.sentence_train(tokens)

                        doc_top_hashtags = set()
                        for hashtag in d['hashtags']:
                            hashtag_idx = top_hashtags_index.get(hashtag)
                            if hashtag_idx is not None:
                                doc_top_hashtags.add(hashtag_idx)
                        if doc_top_hashtags:
                            logging.info('predicting hashtags for doc %s ...' %
                                         d['id'])
                            model.add_doc_embeddings(d)

                            # this needs to be replaced with a real classifier
                            if d['embeddings']:
                                x = sum(d['embeddings']) / len(d['embeddings'])
                            else:
                                x = np.zeros((model.get_embedding_dim(),))
                            y = classifier.predict(
                                scaler.transform(x[np.newaxis, :]))

                            prediction_f.write(
                                '%s\t%s\n' %
                                (d['id'], '\t'.join(
                                    str(int(y[0, j]))
                                    for j in xrange(y.shape[1]))))
                            gold_f.write(
                                '%s\t%s\n' %
                                (d['id'], '\t'.join(
                                    ('1' if j in doc_top_hashtags else '0')
                                    for j in xrange(y.shape[1]))))

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
                    'caught keyboard interrupt, saving; interrupt again to '
                    'cancel'
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
    parser.add_argument('classifier_path', type=str,
                        help='path of classifier on disk')
    parser.add_argument('gold_path', type=str,
                        help='path to which to write gold standard classes')
    parser.add_argument('prediction_path', type=str,
                        help='path to which to write predicted classes')
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

    logging.info('starting training/prediction subprocess ...')
    (train_and_predict_conn, child_conn) = Pipe()
    embedding_process = Process(target=train_and_predict,
                                args=(ns, child_conn))
    embedding_process.start()

    logging.info('sending data to learner ...')
    try:
        for d in extract_redis_tweet_text(host, port, key,
                                          num_processes=ns.num_processes,
                                          batch_size=ns.batch_size,
                                          sleep_interval=ns.sleep_interval,
                                          context_size=context_size,
                                          max_batch_lag=ns.max_batch_lag):
            if not embedding_process.is_alive():
                logging.warning('embedding/hashtag child process not alive')
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

            train_and_predict_conn.send(d)

            train_and_predict_conn.send(dict(message_type='ping'))
            if train_and_predict_conn.recv().get('message_type') != 'pong':
                logging.warning(
                    'embedding/hashtag child process did not pong to ping')
                break

        logging.info('signaling children to exit')

        train_and_predict_conn.send(None)

    except KeyboardInterrupt:
        logging.info(
            'caught keyboard interrupt, signaling children to save; '
            'interrupt again to cancel'
        )
        if embedding_process.is_alive():
            train_and_predict_conn.send(None)
        else:
            logging.warning('embedding/hashtag child process not alive')

    logging.info('joining children (may wait for saves)')

    train_and_predict_conn.close()

    embedding_process.join()

    logging.info('done')


if __name__ == '__main__':
    main()
