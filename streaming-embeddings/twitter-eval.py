#!/usr/bin/env python2.7


from sklearn.metrics import classification_report
import numpy as np
import logging
import itertools as it
import cPickle as pickle


def twitter_eval(classifier_path, gold_path, predict_path,
                 alpha=0.01, size=1, log_interval=1000000, header=False,
                 max_num_points=1000,
                 gold_name='gold',
                 predict_name='ssw2v',
                 title='twitter hashtag prediction performance',
                 colour='red',
                 output_path='twitter-hashtag-prediction-eval.pdf'):
    logging.info('loading classifier from disk ...')
    with open(classifier_path, 'rb') as f:
        d = pickle.load(f)
        top_hashtags = d['top_hashtags']

    num_classes = None
    doc_ids = []
    gold_indicators = []
    predict_indicators = []

    logging.info('loading gold labels and predictions ...')
    with open(gold_path) as gold_f:
        with open(predict_path) as predict_f:
            first_line = True
            for (line_num, (gold_line, predict_line)) in enumerate(
                    it.izip(gold_f, predict_f)):
                if header and first_line:
                    first_line = False
                    continue
                gold_pieces = gold_line.strip().split('\t')
                predict_pieces = predict_line.strip().split('\t')
                if gold_pieces[0] != predict_pieces[0]:
                    logging.warning(
                        'doc ID order disagrees at line %d, stopping' %
                        (line_num + 1))
                    break
                doc_id = gold_pieces[0]
                doc_gold_indicators = map(int, gold_pieces[1:])
                doc_predict_indicators = map(int, predict_pieces[1:])
                if not (len(doc_gold_indicators) ==
                        len(doc_predict_indicators) and
                        (num_classes is None or
                            num_classes == len(doc_gold_indicators))):
                    logging.warning(
                        'no. columns disagrees at line %d, stopping' %
                        (line_num + 1))
                    break
                if num_classes is None:
                    num_classes = len(doc_gold_indicators)
                doc_ids.append(doc_id)
                gold_indicators.append(doc_gold_indicators)
                predict_indicators.append(doc_predict_indicators)

    logging.info('computing metrics ...')
    gold_indicators = np.array(gold_indicators)
    predict_indicators = np.array(predict_indicators)
    print classification_report(gold_indicators, predict_indicators,
                                target_names=top_hashtags)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='evaluate twitter hashtag prediction results',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('classifier_path', type=str,
                        help='path of classifier on disk')
    parser.add_argument('gold_path', type=str,
                        help='path to gold-standard tab file')
    parser.add_argument('predict_path', type=str,
                        help='path to prediction tab file')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    twitter_eval(args.classifier_path, args.gold_path, args.predict_path)


if __name__ == '__main__':
    main()
