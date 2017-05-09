#include "_core.h"
#include "_math.h"
#include "_sgns.h"
#include "_log.h"
#include "_io.h"
#include "_math.h"
#include "_serialization.h"

#include <ctime>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>


#define SENTENCE_LIMIT 1000
#define VOCAB_DIM 7000
#define RESERVOIR_SIZE 100000000
#define EMBEDDING_DIM 100
#define SYMM_CONTEXT 5
#define NEG_SAMPLES 5
#define SUBSAMPLE_THRESHOLD 1e-3
#define TAU 1.7e7
#define KAPPA 2.5e-2
#define RHO_LOWER_BOUND 2.5e-6
#define PROPAGATE_DISCARDED false
#define PROPAGATE_RETAINED false
#define SMOOTHING_EXPONENT 0.75
#define SMOOTHING_OFFSET 0


using namespace std;

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "usage: " << argv[0] << " <input-path> <output-path>\n";
    exit(1);
  }
  const char *input_path = argv[1];
  const char *output_path = argv[2];

  info(__func__, "seeding random number generator ...\n");
  seed_default();

  info(__func__, "initializing model ...\n");
  auto language_model(make_shared<NaiveLanguageModel>(SUBSAMPLE_THRESHOLD));
  auto model(make_shared<SGNSModel>(
    make_shared<WordContextFactorization>(VOCAB_DIM, EMBEDDING_DIM),
    make_shared<ReservoirSamplingStrategy>(
      make_shared<ReservoirSampler<long> >(RESERVOIR_SIZE)),
    language_model,
    make_shared<SGD>(VOCAB_DIM, TAU, KAPPA, RHO_LOWER_BOUND),
    make_shared<DynamicContextStrategy>(SYMM_CONTEXT),
    make_shared<SGNSTokenLearner>(),
    make_shared<SGNSSentenceLearner>(NEG_SAMPLES, PROPAGATE_RETAINED),
    make_shared<SubsamplingSGNSSentenceLearner>(PROPAGATE_DISCARDED)
  ));
  model->token_learner->set_model(model);
  model->sentence_learner->set_model(model);
  model->subsampling_sentence_learner->set_model(model);

  info(__func__, "loading words into vocabulary ...\n");
  string word;
  ifstream f;
  f.open(input_path);
  stream_ready_or_throw(f);
  while (f) {
    const char c = f.get();
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      if (! word.empty()) {
        language_model->increment(word);
        word.clear();
      }
    } else {
      word.push_back(c);
    }
  }

  info(__func__, "truncating language model ...\n");
  language_model->truncate(VOCAB_DIM);

  info(__func__, "initializing reservoir ...\n");
  CountNormalizer normalizer(SMOOTHING_EXPONENT, SMOOTHING_OFFSET);
  model->neg_sampling_strategy->reset(*language_model, normalizer);

  info(__func__, "training ...\n");
  vector<string> sentence;
  size_t words_seen = 0;
  f.clear();
  f.seekg(0);
  time_t start = time(NULL);
  while (f) {
    sentence.clear();
    while (f) {
      const char c = f.get();
      if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
        if (! word.empty()) {
          sentence.push_back(word);
          ++words_seen;
          word.clear();
        }
        if (c == '\n') {
          break;
        }
      } else {
        word.push_back(c);
      }
      if (sentence.size() == SENTENCE_LIMIT) {
        break;
      }
    }

    time_t now = time(NULL);
    if (difftime(now, start) >= 10) {
      start = now;
      info(__func__, "loaded sentence of " << sentence.size() <<
        " new words (" << words_seen << " total), training ...\n");
    }
    model->subsampling_sentence_learner->sentence_train(sentence);
  }

  f.close();

  info(__func__, "saving ...\n");
  FileSerializer<SGNSModel>(output_path).dump(*model);

  info(__func__, "done\n");
}
