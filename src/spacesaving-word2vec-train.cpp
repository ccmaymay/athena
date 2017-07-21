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
#include <utility>
#include <iostream>
#include <fstream>
#include <unistd.h>


#define SENTENCE_LIMIT 1000
#define RHO_LOWER_BOUND_FACTOR 1e-4
#define NEG_SAMPLING_TABLE_SIZE 100000000

#define DEFAULT_SYMM_CONTEXT 5
#define DEFAULT_NEG_SAMPLES 5
#define DEFAULT_TAU 1.7e7
#define DEFAULT_KAPPA 2.5e-2


typedef ReservoirSamplingStrategy<SpaceSavingLanguageModel, ReservoirSampler<long> > NegSamplingStrategy;
typedef SGNSTokenLearner<SpaceSavingLanguageModel, NegSamplingStrategy> SGNSTokenLearnerType;
typedef SGNSSentenceLearner<SGNSTokenLearnerType, DynamicContextStrategy> SGNSSentenceLearnerType;

using namespace std;

void usage(ostream& s, const string& program) {
  s << "Train Space-Saving word2vec (SGNS) model from text file.\n";
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
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  string eos_symbol;
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
    ret = getopt(argc, argv, "v:e:s:n:c:t:k:h");
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

  info(__func__, "initializing SGNS model ...\n");
  auto sgd(make_shared<SGD>(
    vocab_dim,
    tau,
    kappa,
    RHO_LOWER_BOUND_FACTOR * kappa));
  auto language_model(make_shared<SpaceSavingLanguageModel>(
    vocab_dim, subsample_threshold));
  auto neg_sampling_strategy(make_shared<NegSamplingStrategy>(
    make_shared<ReservoirSampler<long> >(NEG_SAMPLING_TABLE_SIZE)));
  auto token_learner(make_shared<SGNSTokenLearnerType>(
    make_shared<WordContextFactorization>(vocab_dim, embedding_dim),
    neg_sampling_strategy,
    language_model,
    sgd
  ));
  SGNSSentenceLearnerType sentence_learner(
    token_learner,
    make_shared<DynamicContextStrategy>(symm_context),
    neg_samples
  );

  info(__func__, "training ...\n");
  size_t words_seen = 0, prev_words_seen = 0;
  ifstream f;
  f.open(input_path);
  stream_ready_or_throw(f);
  SentenceReader reader(f, SENTENCE_LIMIT);
  time_t start = time(NULL), prev_now = time(NULL);
  while (reader.has_next()) {
    vector<string> sentence(reader.next());

    for (auto it = sentence.begin(); it != sentence.end(); ++it) {
      const pair<long,string> ejectee = language_model->increment(*it);
      const long ejectee_idx = ejectee.first;
      if (ejectee_idx >= 0) {
        token_learner->reset_word(ejectee_idx);
      }
      neg_sampling_strategy->step(
        *language_model, language_model->lookup(*it));
    }

    vector<long> word_ids;
    word_ids.reserve(sentence.size());
    for (auto it = sentence.begin(); it != sentence.end(); ++it) {
      long word_id = language_model->lookup(*it);
      if (word_id >= 0) {
        if (language_model->subsample(word_id)) {
          word_ids.push_back(word_id);
        }
        ++words_seen;
      }
    }

    sentence_learner.sentence_train(word_ids);

    for (size_t input_word_pos = 0; input_word_pos < word_ids.size();
         ++input_word_pos) {
      sgd->step(word_ids[input_word_pos]);
    }

    time_t now = time(NULL);
    if (difftime(now, prev_now) >= 5) {
      info(__func__, "loaded " << (words_seen / 1000) << " kwords total, " <<
          round(
            (words_seen - prev_words_seen) / difftime(now, prev_now) / 1000
          ) << " kwords/sec; training ...\n");
      prev_words_seen = words_seen;
      prev_now = now;
    }
  }

  time_t now = time(NULL);
  info(__func__, "loaded " << (words_seen / 1000) << " kwords total, " <<
      round(words_seen / difftime(now, start) / 1000) <<
      " kwords/sec overall, " << difftime(now, start) << " sec\n");
  prev_words_seen = words_seen;
  prev_now = now;

  f.close();

  info(__func__, "saving ...\n");
  FileSerializer<SGNSSentenceLearnerType>(output_path).dump(sentence_learner);

  info(__func__, "done\n");
}
