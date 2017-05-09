#!/usr/bin/env python2.7


import logging
import sys
import codecs

from athena.core import SGNSModel


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'probe near neighbors of words in SpaceSaving word2vec model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    parser.add_argument('probe_words', type=str, nargs='+',
                        metavar='probe_word',
                        help='words to probe every log-interval (diagnostic)')
    SGNSModel.add_arguments(parser)
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading model from disk ...')
    model = SGNSModel.load(ns.model_path)

    logging.info('probing nearest neighbors ...')
    writer = codecs.getwriter('utf8')(sys.stdout)
    for probe_word in ns.probe_words:
        writer.write(u'%s : %s\n' % (
            probe_word,
            model.find_nearest_neighbor(probe_word)
        ))


if __name__ == '__main__':
    main()
