#!/usr/bin/env python2.7


import logging
import codecs
import sys

from athena.core import SGNSModel


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'print language model in SpaceSaving word2vec model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    SGNSModel.add_arguments(parser)
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading model from disk ...')
    model = SGNSModel.load(ns.model_path)

    logging.info('printing language model ... ')
    writer = codecs.getwriter('utf8')(sys.stdout)
    for (word, count) in sorted(model.get_word_counts().items(),
                                key=lambda wc: wc[1], reverse=True):
        writer.write(u'%s : %d\n' % (word, count))


if __name__ == '__main__':
    main()
