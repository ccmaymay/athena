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
#include <unistd.h>


#define SENTENCE_LIMIT 1000
#define RHO_LOWER_BOUND_FACTOR 1e-4
#define PROPAGATE_DISCARDED false
#define PROPAGATE_RETAINED false
#define SMOOTHING_EXPONENT 0.75
#define SMOOTHING_OFFSET 0
#define RESERVOIR_SIZE 100000000

#define DEFAULT_SYMM_CONTEXT 5
#define DEFAULT_NEG_SAMPLES 5
#define DEFAULT_TAU 1.7e7
#define DEFAULT_KAPPA 2.5e-2


using namespace std;

void usage(ostream& s, const string& program) {
  s << "Train word2vec (SGNS) model from text file.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (training text).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (serialized model).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -v <vocab-dim>\n";
  s << "     Set vocabulary dimension.\n";
  s << "     Default: " << DEFAULT_VOCAB_DIM << "\n";
  s << "  -e <embedding-dim>\n";
  s << "     Set embedding dimension.\n";
  s << "     Default: " << DEFAULT_EMBEDDING_DIM << "\n";
  s << "  -s <subsample-threshold>\n";
  s << "     Set word subsampling threshold.\n";
  s << "     Default: " << DEFAULT_SUBSAMPLE_THRESHOLD << "\n";
  s << "  -n <neg-samples>\n";
  s << "     Set number of negative samples to draw.\n";
  s << "     Default: " << DEFAULT_NEG_SAMPLES << "\n";
  s << "  -c <context>\n";
  s << "     Set number of words on each side to use as context.\n";
  s << "     Default: " << DEFAULT_SYMM_CONTEXT << "\n";
  s << "  -t <tau>\n";
  s << "     Set learning rate iteration divisor.\n";
  s << "     Default: " << DEFAULT_TAU << "\n";
  s << "  -k <kappa>\n";
  s << "     Set learning rate overall multiplier.\n";
  s << "     Default: " << DEFAULT_KAPPA << "\n";
  s << "  -x <eos-symbol>\n";
  s << "     Set explicit end-of-sentence symbol.\n";
  s << "     Default: none (sentences delimited by newlines only)\n";
  s << "  -l <lm-path>\n";
  s << "     Load language model from file (rather than learning from data).\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  string lm_path, eos_symbol;
  size_t
    vocab_dim(DEFAULT_VOCAB_DIM),
    embedding_dim(DEFAULT_EMBEDDING_DIM),
    neg_samples(DEFAULT_NEG_SAMPLES),
    symm_context(DEFAULT_SYMM_CONTEXT);
  float
    subsample_threshold(DEFAULT_SUBSAMPLE_THRESHOLD),
    tau(DEFAULT_TAU),
    kappa(DEFAULT_KAPPA);

  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "v:e:s:n:c:t:k:l:x:h");
    switch (ret) {
      case 'v':
        vocab_dim = stoull(string(optarg));
        break;
      case 'e':
        embedding_dim = stoull(string(optarg));
        break;
      case 's':
        subsample_threshold = stof(string(optarg));
        break;
      case 'n':
        neg_samples = stoull(string(optarg));
        break;
      case 'c':
        symm_context = stoull(string(optarg));
        break;
      case 't':
        tau = stof(string(optarg));
        break;
      case 'k':
        kappa = stof(string(optarg));
        break;
      case 'l':
        lm_path = string(optarg);
        break;
      case 'x':
        eos_symbol = string(optarg);
        break;
      case 'h':
        usage(cout, program);
        exit(0);
      case '?':
        usage(cerr, program);
        exit(1);
      case -1:
        break;
    }
  }
  if (optind + 2 != argc) {
    usage(cerr, program);
    exit(1);
  }
  const char *input_path = argv[optind];
  const char *output_path = argv[optind + 1];

  info(__func__, "seeding random number generator ...\n");
  seed_default();

  shared_ptr<LanguageModel> language_model;

  if (lm_path.empty()) {
    info(__func__, "initializing language model ...\n");
    language_model = make_shared<NaiveLanguageModel>(subsample_threshold);

    info(__func__, "loading words into vocabulary ...\n");
    string word;
    ifstream f;
    f.open(input_path);
    stream_ready_or_throw(f);
    while (f) {
      const char c = f.get();
      if (c == '\r') {
        continue;
      }
      if (c == ' ' || c == '\n' || c == '\t') {
        if (! word.empty() && word != eos_symbol) {
          language_model->increment(word);
          word.clear();
        }
      } else {
        word.push_back(c);
      }
    }
    f.close();

    info(__func__, "truncating language model ...\n");
    language_model->truncate(vocab_dim);
  } else {
    info(__func__, "loading language model ...\n");
    language_model = FileSerializer<LanguageModel>(lm_path).load();

    info(__func__, "setting vocab dim to language model size " <<
                     language_model->size() <<
                     " ...\n");
    vocab_dim = language_model->size();
  }

  info(__func__, "initializing SGNS model ...\n");
  auto model(make_shared<SGNSModel>(
    make_shared<WordContextFactorization>(vocab_dim, embedding_dim),
    make_shared<ReservoirSamplingStrategy>(
      make_shared<ReservoirSampler<long> >(RESERVOIR_SIZE)),
    language_model,
    make_shared<SGD>(vocab_dim, tau, kappa, RHO_LOWER_BOUND_FACTOR * kappa),
    make_shared<DynamicContextStrategy>(symm_context),
    make_shared<SGNSTokenLearner>(),
    make_shared<SGNSSentenceLearner>(neg_samples, PROPAGATE_RETAINED),
    make_shared<SubsamplingSGNSSentenceLearner>(PROPAGATE_DISCARDED)
  ));
  model->token_learner->set_model(model);
  model->sentence_learner->set_model(model);
  model->subsampling_sentence_learner->set_model(model);

  info(__func__, "initializing reservoir ...\n");
  CountNormalizer normalizer(SMOOTHING_EXPONENT, SMOOTHING_OFFSET);
  model->neg_sampling_strategy->reset(*language_model, normalizer);

  info(__func__, "training ...\n");
  vector<string> sentence;
  size_t words_seen = 0;
  string word;
  ifstream f;
  f.open(input_path);
  stream_ready_or_throw(f);
  time_t start = time(NULL);
  while (f) {
    sentence.clear();
    while (f) {
      const char c = f.get();
      if (c == '\r') {
        continue;
      }
      if (c == ' ' || c == '\n' || c == '\t') {
        if (! word.empty() && word != eos_symbol) {
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
