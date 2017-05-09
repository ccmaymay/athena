#!/usr/bin/env python2.7


import logging

from athena.core import SGNSModel


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'print model parameters',
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

    print 'Vocab dim: %d' % model.get_vocab_dim()
    print 'Vocab used: %d' % model.get_vocab_used()
    print 'Embedding dim: %d' % model.get_embedding_dim()


if __name__ == '__main__':
    main()
