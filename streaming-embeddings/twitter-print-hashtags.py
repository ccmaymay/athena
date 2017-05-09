#!/usr/bin/env python2.7


import logging
import codecs
import sys

from athena.core import LMMultiLabelDocStore


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'print doc store parameters',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('doc_store_path', type=str,
                        help='path of doc store on disk')
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading doc store from disk ...')
    doc_store = LMMultiLabelDocStore.load(ns.doc_store_path)

    logging.info('number of hashtags: %d' % doc_store.get_vocab_used())

    logging.info('printing hashtags ... ')
    word_counts = doc_store.get_word_counts()
    sorted_word_counts = sorted(word_counts.items(), key=lambda p: p[1],
                                reverse=True)

    writer = codecs.getwriter('utf8')(sys.stdout)
    for (word, count) in sorted_word_counts:
        docs = doc_store.get_docs(word)
        writer.write(u'%s : %d : %d\n' % (word, count, len(docs)))
        for doc in docs:
            writer.write(u'* %s\n' % u' '.join(doc['tokens']))


if __name__ == '__main__':
    main()
