#!/usr/bin/env python2.7


import logging
import codecs

from athena.core import SGNSModel


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'dump pairwise cosine similarity between words to disk',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    parser.add_argument('word_list_path', type=str,
                        help='path of word list to compute similarity on')
    parser.add_argument('output_path', type=str,
                        help='path where similarities will be written')
    SGNSModel.add_arguments(parser)
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading model from disk ...')
    model = SGNSModel.load(ns.model_path)

    logging.info('loading word list from disk ...')
    words = []
    with codecs.open(ns.word_list_path, encoding='utf-8') as f:
        for line in f:
            words.append(line.strip())

    logging.info('writing similarities ... ')
    model.dump_similarity(words, ns.output_path)


if __name__ == '__main__':
    main()
