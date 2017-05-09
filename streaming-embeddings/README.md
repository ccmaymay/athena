# Streaming embeddings

This directory contains scripts for running experiments under the
"streaming embeddings" project, in which word2vec (skip-gram
negative sampling) embeddings
are updated online, in one pass, with bounded memory, in order to
effect adaptation of an expensive-to-train classifier or other kind of
downstream model on a changing data stream, in particular a changing
vocabulary, without incurring the cost of updating or re-training the
downstream model itself.

We implement a streaming word embedding model by taking word2vec and
replacing the fixed vocabulary, learned from the data offline, with an
online-updated bounded-memory vocabulary defined on a space-saving
counter array.  We call this model "space-saving word2vec."

## Cosine similarity experiment

In this experiment we train word2vec and space-saving word2vec models
on the text8 dataset provided with/linked from the original word2vec
C code and compare the cosine distance of random word pairs between the
two models.

### Dependencies

Before running the cosine similarity experiment
the athena main routines must be built.  To do so, go to the top of the
repository and run `make main`.

Additionally,
the `text8` dataset must be downloaded.  This can be performed
by running the `download.sh` script in the
[original word2vec source distribution](https://code.google.com/archive/p/word2vec/)
or with the following command:

```
curl http://mattmahoney.net/dc/text8.zip | gunzip > text8
```

### Configuration

The experiment is organized using `make`; see `Makefile` in this
directory for the configuration.  In particular, edit the values of
parameters at the top of the file or override them on the command line
as necessary.

### Running the experiment

To run the experiment, run `make` from this directory.
There are several models to train
and other processor-intensive tasks; if running on a multicore machine,
it is recommended to use parallel task execution, e.g., `make -j 32`.

## Twitter hashtag prediction

In this experiment we train space-saving word2vec on a Twitter stream
over a preset date range, then train a multi-label classifier over
hashtags on the fixed embedding model and a training set sampled from
that date range, then resume training the embedding model on the
remainder of the Twitter stream, periodically computing hashtag
predictions on Tweets in the stream using the latest embedding model.

We compare the space-saving word2vec approach against a baseline
in which vanilla word2vec is used as the embedding model.  In the
baseline, once the classifier is trained, the embedding model is held
fixed for the purpose of subsequent hashtag predictions (it is not
re-trained on tweets from the second half of the stream).

The experiment comprises five steps:
1. Train LM for word2vec
2. Train embeddings and sample hashtag examples
3. Train classifier
4. Continue training embeddings and predict hashtags
5. Evaluate predictions

After a brief description of dependencies, each step will be
explained in turn.  Note important general information is given in the
earlier steps and not repeated, so it is recommended the steps be read
in order.

### Dependencies

Reading the twitter stream depends on a redis server and scion support.
See [the tube README](https://gitlab.hltcoe.jhu.edu/concrete/tube) for
information about installing both redis and scion and starting a redis
server.  For the remainder of this walkthrough we assume a redis server
is running at `localhost:33211`.

We also assume that this branch of athena (`streaming-embeddings`)
is built and installed.  This can be accomplished by running
`python setup.py install --user` from the parent directory.

### Train LM for word2vec

First, to be safe, we empty the redis server:

```
redis-cli -h localhost -p 33211 flushdb
```

Now we start the ingest.  This process loads tweets starting at
midnight on Jan 1, 2016, sequentially into the redis server at key
`twitter/comms` (hardcoded) as concrete.  Once `twitter/comms` has
reached a certain size, the process sleeps until the size drops again,
in order to prevent an out-of-memory error.  Also note the process
starts *many* subprocess to achieve higher throughput, so it should be
run on appropriate resources (an idle eight-core machine is
recommended).  Finally, note the process will block, so you will want
to run it in a separate terminal session, or redirect
its output and send it to the background:

```
python twitter-stream-load-2016-01.py localhost:33211
```

With the ingest running we train the static language model used by the
word2vec baseline embeddings.  The script we call takes a large number
of optional arguments; the defaults are set to reasonable values.
As with the ingest, the top-level process starts a number of
subprocesses to increase throughput, and it is suggested to run the
script on an idle eight-core machine:

```
python twitter-train-lm.py \
    'redis://localhost:33211/twitter/comms?block=True&pop=True' \
    /export/projects/$USER/twitter.2016-01.w2v-lm \
    --language-model naive \
    --stop-year 2016 --stop-month 2
```

Note this script will save and quit when the stream reaches February.
At that point, kill the ingest process, and then check `ps` or `top`
and make sure it's actually killed.

### Train embeddings and sample hashtag examples

First, to be safe, we empty the redis server:

```
redis-cli -h localhost -p 33211 flushdb
```

Now we start the ingest:

```
python twitter-stream-load-2016-01.py localhost:33211
```

With the ingest running we train the space-saving word2vec embedding
model, simultaneously computing the hashtag examples for classifier
training.  The script we call takes a large number
of optional arguments; the defaults are set to reasonable values.
As with the ingest, the top-level process starts a number of
subprocesses to increase throughput, and it is suggested to run the
script on an idle eight-core machine:

```
python twitter-train-embeddings-and-sample-hashtags.py \
    'redis://localhost:33211/twitter/comms?block=True&pop=True' \
    /export/projects/$USER/twitter.2016-01.ssw2v-model \
    /export/projects/$USER/twitter.2016-01.hashtag-store \
    --doc-store-vocab-dim 10000 \
    --propagate-retained \
    --stop-year 2016 --stop-month 2
```

Note this script will save and quit when the stream reaches February.
At that point, kill the ingest process, and then check `ps` or `top`
and make sure it's actually killed.

To be safe, we now empty the redis server:

```
redis-cli -h localhost -p 33211 flushdb
```

Now we start the ingest again:

```
python twitter-stream-load-2016-01.py localhost:33211
```

With the ingest running we train the baseline word2vec embedding
model using the static language model computed earlier.  The script we
call takes a large number of optional arguments; the defaults are set
to reasonable values.  As with the ingest, the top-level process starts
a number of subprocesses to increase throughput, and it is suggested to
run the script on an idle eight-core machine:

```
cp /export/projects/$USER/twitter.2016-01.w2v-lm \
    /export/projects/$USER/twitter.2016-01.w2v-model
python twitter-train-embeddings.py \
    'redis://localhost:33211/twitter/comms?block=True&pop=True' \
    /export/projects/$USER/twitter.2016-01.w2v-model \
    --load-model \
    --stop-year 2016 --stop-month 2
```

Note this script will save and quit when the stream reaches February.
At that point, kill the ingest process, and then check `ps` or `top`
and make sure it's actually killed.

### Train classifier

We now train the multi-label hashtag classifiers for the space-saving
word2vec and baseline word2vec embedding models, respectively.  First
we generate compact representations of the training data:

```
python twitter-create-classifier-training-data.py \
    /export/projects/$USER/twitter.2016-01.ssw2v-model \
    /export/projects/$USER/twitter.2016-01.hashtag-store \
    /export/projects/$USER/twitter.2016-01.ssw2v-classifier-data \
    --num-examples-per-class 1000
python twitter-create-classifier-training-data.py \
    /export/projects/$USER/twitter.2016-01.w2v-model \
    /export/projects/$USER/twitter.2016-01.hashtag-store \
    /export/projects/$USER/twitter.2016-01.w2v-classifier-data \
    --num-examples-per-class 1000
```

Next we run the classifiers themselves:

```
python twitter-train-classifier.py \
    /export/projects/$USER/twitter.2016-01.ssw2v-classifier-data \
    /export/projects/$USER/twitter.2016-01.ssw2v-classifier
python twitter-train-classifier.py \
    /export/projects/$USER/twitter.2016-01.w2v-classifier-data \
    /export/projects/$USER/twitter.2016-01.w2v-classifier
```

### Continue training embeddings and predict hashtags

First, to be safe, we empty the redis server:

```
redis-cli -h localhost -p 33211 flushdb
```

Now we start the ingest, this time starting at midnight on Feb 1,
2016.  (The script is otherwise identical to the January ingest.)

```
python twitter-stream-load-2016-02.py localhost:33211
```

With the ingest running we continue training the space-saving word2vec
embedding model while simultaneously writing out gold-standard hashtags
and classifier predictions.  The script we call takes a large number
of optional arguments; the defaults are set to reasonable values.
As with the ingest, the top-level process starts a number of
subprocesses to increase throughput, and it is suggested to run the
script on an idle eight-core machine:

```
cp /export/projects/$USER/twitter.2016-01.ssw2v-model \
    /export/projects/$USER/twitter.2016-02.ssw2v-model
python twitter-train-embeddings-and-predict-hashtags.py \
    'redis://localhost:33211/twitter/comms?block=True&pop=True' \
    /export/projects/$USER/twitter.2016-02.ssw2v-model \
    /export/projects/$USER/twitter.2016-01.ssw2v-classifier \
    /export/projects/$USER/twitter.2016-02.ssw2v-gold-labels \
    /export/projects/$USER/twitter.2016-02.ssw2v-predictions \
    --propagate-retained \
    --stop-year 2016 --stop-month 3
```

This script will not save and quit on its own.  The ingest process will
terminate when it reaches the end of the stream.  When this happens,
quit the embedding training/prediction process by sending *one*
interrupt signal (Ctrl-C if it is in the foreground).

To be safe, we now empty the redis server:

```
redis-cli -h localhost -p 33211 flushdb
```

Now we start the ingest again:

```
python twitter-stream-load-2016-02.py localhost:33211
```

With the ingest running we load the word2vec
embedding model and write out gold-standard hashtags
and classifier predictions:

```
cp /export/projects/$USER/twitter.2016-01.w2v-model \
    /export/projects/$USER/twitter.2016-02.w2v-model
python twitter-train-embeddings-and-predict-hashtags.py \
    'redis://localhost:33211/twitter/comms?block=True&pop=True' \
    /export/projects/$USER/twitter.2016-02.w2v-model \
    /export/projects/$USER/twitter.2016-01.w2v-classifier \
    /export/projects/$USER/twitter.2016-02.w2v-gold-labels \
    /export/projects/$USER/twitter.2016-02.w2v-predictions \
    --stop-year 2016 --stop-month 3
```

This script will not save and quit on its own.  The ingest process will
terminate when it reaches the end of the stream.  When this happens,
quit the embedding training/prediction process by sending *one*
interrupt signal (Ctrl-C if it is in the foreground).

### Evaluate predictions

We now evaluate classifier predictions from the space-saving
word2vec and baseline word2vec embedding models, respectively:

```
python twitter-eval.py \
    /export/projects/$USER/twitter.2016-01.ssw2v-classifier \
    /export/projects/$USER/twitter.2016-02.ssw2v-gold-labels \
    /export/projects/$USER/twitter.2016-02.ssw2v-predictions
python twitter-eval.py \
    /export/projects/$USER/twitter.2016-01.w2v-classifier \
    /export/projects/$USER/twitter.2016-02.w2v-gold-labels \
    /export/projects/$USER/twitter.2016-02.w2v-predictions
```
