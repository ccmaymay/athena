#!/usr/bin/env python2.7


import readline  # noqa
from rpy2.robjects import globalenv, r
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FloatVector, DataFrame, IntVector

import random
import codecs
import logging


def plot_counts_scatter(baseline_lm_path, lm_path,
                        alpha=0.01, size=1, log_interval=1000000, header=True,
                        max_num_points=None,
                        title='spacesaving counts',
                        colour='red', output_path='counts.pdf',
                        filter_missing=False):
    count_pairs = dict()

    logging.info('loading baseline language model ...')
    with codecs.open(baseline_lm_path, encoding='utf-8') as f:
        for (line_num, line) in enumerate(f):
            if header and line_num == 0:
                continue

            if (line_num + 1) % log_interval == 0:
                logging.info('processing line %d ...' % (line_num + 1,))

            (word, count) = line.strip().split('\t')
            count_pairs[word] = (int(count), None)

    logging.info('loaded %d baseline words' % len(count_pairs))

    logging.info('loading test language model ...')
    with codecs.open(lm_path, encoding='utf-8') as f:
        for (line_num, line) in enumerate(f):
            if header and line_num == 0:
                continue

            if (line_num + 1) % log_interval == 0:
                logging.info('processing line %d ...' % (line_num + 1,))

            (word, count) = line.strip().split('\t')
            if word in count_pairs:
                count_pairs[word] = (count_pairs[word][0], int(count))

    logging.info('sorting words ...')
    keys = [
        k
        for (k, v)
        in sorted(count_pairs.items(), key=lambda p: p[1], reverse=True)
    ]
    key_indices = dict((k, i + 1) for (i, k) in enumerate(keys))

    if filter_missing:
        logging.info('filtering missing words ...')
        keys = [k for k in keys if count_pairs[k][1] is not None]

    logging.info('estimating missing counts ...')
    missing_test_count_estimate = min(
        c for (_, c) in count_pairs.values() if c is not None
    )
    count_pairs = dict(
        (k, (v[0], missing_test_count_estimate if v[1] is None else v[1]))
        for (k, v) in count_pairs.items()
    )

    logging.info('have %d words' % len(keys))
    if max_num_points is not None and len(keys) > max_num_points:
        keys = random.sample(keys, max_num_points)
        logging.info('have %d words after thresholding' % len(keys))

    logging.info('creating data frame ...')
    globalenv['d'] = DataFrame({
        'index': IntVector(tuple(key_indices[k] for k in keys)),
        'lm.baseline': FloatVector(tuple(count_pairs[k][0] for k in keys)),
        'lm': FloatVector(tuple(count_pairs[k][1] for k in keys)),
        'word': StrVector(tuple(keys)),
    })

    logging.info('plotting ...')
    importr('ggplot2')
    globalenv['alpha'] = alpha
    globalenv['size'] = size
    globalenv['title'] = title
    globalenv['colour'] = colour
    globalenv['path.output'] = output_path
    r('''
    d$rel.error <- (d$lm - d$lm.baseline) / d$lm.baseline
    ggplot(d, aes(x=index, y=rel.error)) +
        geom_point(alpha=alpha, size=size, colour=colour) +
        theme_bw() +
        xlab('word rank (ground truth)') +
        ylab('relative error in count') +
        ggtitle(title)
    ggsave(path.output, width=3.5, height=3.5, units='in')
    ''')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='make scatter-plot of estimated word counts versus true '
                    'word counts',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('baseline_lm_path', type=str,
                        help='path to baseline word language model')
    parser.add_argument('lm_path', type=str,
                        help='path to test word language model')
    parser.add_argument('--size', type=float, default=0.3,
                        help='size of points')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='transparency of points')
    parser.add_argument('--max-num-points', type=int,
                        help='maximum number of points to plot')
    parser.add_argument('--output-path', type=str, default='counts.pdf',
                        help='file path to which to write plot')
    parser.add_argument('--title', type=str, default='spacesaving counts',
                        help='plot title')
    parser.add_argument('--header', action='store_true',
                        help='input files have headers (skip first line)')
    parser.add_argument('--filter-missing', action='store_true',
                        help='remove words that are missing in test LM')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d %(funcName)s:'
               ' %(message)s'
    )

    plot_counts_scatter(args.baseline_lm_path, args.lm_path,
                        size=args.size, alpha=args.alpha,
                        max_num_points=args.max_num_points,
                        output_path=args.output_path,
                        title=args.title,
                        filter_missing=args.filter_missing, header=args.header)


if __name__ == '__main__':
    main()
