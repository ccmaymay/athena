#!/bin/bash

set -e

OMP_NUM_THREADS=1 /usr/bin/time python spacesaving-word2vec-train-twitter.py 'redis://localhost:34958/twitter/comms?block=True&pop=True' /export/projects/cmay/twitter.ssw2v.model --embedding-dim 100 --vocab-dim 10000000 --log-interval 10000 --reservoir-size 1000000000 --save-interval 10000000 --tau 6.2e7 --symm-context 2
