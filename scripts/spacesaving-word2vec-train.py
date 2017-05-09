#!/usr/bin/env python2.7


import logging

from athena.core import SGNSModel
from athena.util import URLCommunicationReader, mkdirp_parent
from concrete.util.unnone import lun
from concrete.util.tokenization import get_tokens, get_lemmas


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'train SpaceSaving word2vec model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_url', type=str,
                        help='URL to dataset (/home/me/concrete.tar.gz or '
                             'file:///home/me/concrete.tar.gz or '
                             'redis://r8n00:6000/data-key')
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    parser.add_argument('--load', action='store_true',
                        help='load existing model from disk instead of '
                             'creating new')
    parser.add_argument('--num-sweeps', type=int,
                        help='number of passes to make over dataset '
                             '(should be one if popping from redis)',
                        default=1)
    parser.add_argument('--save-interval', type=int,
                        help='interval (in number of documents) at which '
                             'to save model',
                        default=1000)
    parser.add_argument('--log-interval', type=int,
                        help='interval (in number of documents) at which '
                             'to log status',
                        default=100)
    SGNSModel.add_arguments(parser)
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('checking model parent directory ...')
    mkdirp_parent(ns.model_path)

    if ns.load:
        logging.info('loading model from disk ...')
        model = SGNSModel.load(ns.model_path)
    else:
        logging.info('initializing model ...')
        model = SGNSModel.from_namespace(ns)

    logging.info('training ...')
    for s in xrange(ns.num_sweeps):
        logging.info('starting sweep %d ...' % (s+1))

        logging.info('initializing data source ...')
        reader = URLCommunicationReader(ns.data_url)

        token_count = 0
        for (i, comm) in enumerate(reader):
            if (i+1) % ns.log_interval == 0:
                logging.info(
                    u'sweep %d: training on doc %d (seen %d tokens) ...' %
                    (s+1, i+1, token_count)
                )

            for section in lun(comm.sectionList):
                for sentence in lun(section.sentenceList):
                    tokenization = sentence.tokenization
                    try:
                        tokens = [
                            tt.tag.lower()
                            for tt in get_lemmas(tokenization)
                        ]
                    except:
                        tokens = [
                            token.text.lower()
                            for token in get_tokens(tokenization)
                        ]
                    model.sentence_train(tokens)
                    token_count += len(tokens)

            if (i+1) % ns.save_interval == 0:
                logging.info(
                    u'sweep %d: saving after %d docs (%d tokens)...' %
                    (s+1, i+1, token_count))
                model.dump(ns.model_path)

    logging.info(u'saving after %d sweeps...' % ns.num_sweeps)
    model.dump(ns.model_path)

    logging.info('done')


if __name__ == '__main__':
    main()
