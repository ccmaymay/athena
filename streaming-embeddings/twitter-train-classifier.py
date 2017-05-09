#!/usr/bin/env python2.7


import logging

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import cPickle as pickle


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'train stub hashtag classifier on pre-compiled dataset',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('training_data_path', type=str,
                        help='path from which to read training data')
    parser.add_argument('classifier_path', type=str,
                        help='path to which to write classifier')
    ns = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    logging.info('loading training data ...')
    with open(ns.training_data_path, 'rb') as f:
        d = pickle.load(f)
        token_x = d['token_x']
        token_x_doc_offsets = d['token_x_doc_offsets']
        y = d['y']
        top_hashtags = d['top_hashtags']

    logging.info('compiling training matrices ...')
    x = np.zeros((y.shape[0], token_x.shape[1]))
    for doc_idx in xrange(y.shape[0]):
        x[doc_idx] = token_x[
            token_x_doc_offsets[doc_idx]:token_x_doc_offsets[doc_idx+1]
        ].mean(axis=0)

    logging.info('computing standardization ...')
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)

    logging.info('training classifier ...')
    classifier = OneVsRestClassifier(SVC(kernel='rbf'))
    classifier.fit(x, y)

    logging.info('saving to disk ...')
    with open(ns.classifier_path, 'wb') as f:
        pickle.dump(dict(scaler=scaler,
                         classifier=classifier,
                         top_hashtags=top_hashtags), f)

    logging.info('done')


if __name__ == '__main__':
    main()
