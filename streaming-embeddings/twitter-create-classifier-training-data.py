#!/usr/bin/env python2.7


import logging

from athena.core import SGNSModel, LMMultiLabelDocStore

from heapq import nlargest

import numpy as np
import cPickle as pickle


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'compute classifier training data and dump to disk',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', type=str,
                        help='path of model on disk')
    parser.add_argument('doc_store_path', type=str,
                        help='path of hashtag-example doc store on disk')
    parser.add_argument('training_data_path', type=str,
                        help='path to which to write training data')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes (top hashtags) to filter to',
                        default=100)
    parser.add_argument('--num-examples-per-class', type=int,
                        help='max. no. examples to keep per class (None: all)')
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading model from disk ...')
    model = SGNSModel.load(ns.model_path)

    logging.info('loading doc store from disk ...')
    doc_store = LMMultiLabelDocStore.load(ns.doc_store_path)

    logging.info('computing top %d hashtags ..' % ns.num_classes)
    hashtag_counts = doc_store.get_word_counts()
    top_hashtag_counts = nlargest(ns.num_classes, hashtag_counts.items(),
                                  key=lambda p: p[1])
    top_hashtags = map(lambda p: p[0], top_hashtag_counts)

    logging.info('fetching embeddings for top hashtag examples ...')
    docs = dict()
    for hashtag_idx in xrange(len(top_hashtags)):
        hashtag = top_hashtags[hashtag_idx]
        for (i, doc) in enumerate(doc_store.get_docs(hashtag)):
            model.add_doc_embeddings(doc)
            if doc['id'] in docs:
                docs[doc['id']]['y'].append(hashtag_idx)
            else:
                doc['y'] = [hashtag_idx]
                docs[doc['id']] = doc
            if (ns.num_examples_per_class is not None and
                    i + 1 == ns.num_examples_per_class):
                break

    num_tokens = sum(
        len(doc['embeddings']) for doc in docs.values()
    )
    logging.info('have %d docs spanning %d tokens' % (len(docs), num_tokens))

    logging.info('computing training matrices ...')
    token_x_doc_offsets = []
    token_x = np.zeros((num_tokens, model.get_embedding_dim()))
    y = np.zeros((len(docs), len(top_hashtags)), dtype=np.uint)
    doc_ids = sorted(docs.keys())
    tok_idx = 0
    for doc_idx in xrange(len(doc_ids)):
        doc = docs[doc_ids[doc_idx]]
        token_x_doc_offsets.append(tok_idx)
        for j in xrange(len(doc['embeddings'])):
            token_x[tok_idx] = doc['embeddings'][j]
            tok_idx += 1
        for hashtag_idx in doc['y']:
            y[doc_idx, hashtag_idx] = 1
    token_x_doc_offsets.append(tok_idx)

    logging.info('saving to disk ...')
    with open(ns.training_data_path, 'wb') as f:
        pickle.dump(dict(doc_ids=doc_ids,
                         token_x_doc_offsets=token_x_doc_offsets,
                         token_x=token_x,
                         y=y,
                         top_hashtags=top_hashtags), f)

    logging.info('done')


if __name__ == '__main__':
    main()
