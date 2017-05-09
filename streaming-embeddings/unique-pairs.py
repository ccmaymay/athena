#!/usr/bin/env python2.7


import codecs
import logging


def unique_pairs(input_path, output_path):
    logging.info('loading lines ...')
    with codecs.open(input_path, encoding='utf-8') as f:
        words = [line.strip() for line in f]

    logging.info('writing pairs ...')
    with codecs.open(output_path, mode='w', encoding='utf-8') as f:
        for j in xrange(1, len(words)):
            for i in xrange(j):
                f.write(u'%s\t%s\n' % (words[i], words[j]))

    logging.info('done')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='read words from input file (one word per line) and emit '
                    'unique word pairs to output file',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_path', type=str,
                        help='path input file (one word per line)')
    parser.add_argument('output_path', type=str,
                        help='path output file (two words per line)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    unique_pairs(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
