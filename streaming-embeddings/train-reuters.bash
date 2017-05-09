#!/bin/bash

set -e

OMP_NUM_THREADS=1 /usr/bin/time spacesaving-word2vec-train.py textfile:///export/projects/cmay/rcv1v2-train.text.split /export/projects/cmay/rcv1v2-train.ssw2v.model --embedding-dim 100 --vocab-dim 200000 --log-interval 1 --save-interval 1 --num-sweeps 10 --tau 6.2e7
