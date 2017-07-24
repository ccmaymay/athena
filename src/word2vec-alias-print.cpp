#include "_core.h"
#include "_sgns.h"
#include "_log.h"
#include "_io.h"
#include "_serialization.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unistd.h>


typedef SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > SGNSTokenLearnerType;
typedef SGNSSentenceLearner<SGNSTokenLearnerType, DynamicContextStrategy> SGNSSentenceLearnerType;

using namespace std;

void usage(ostream& s, const string& program) {
  s << "Load serialized word2vec (SGNS) model and print to a file.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (serialized word2vec model).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (text representation of word2vec model).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -w\n";
  s << "     Print word before embedding vector (separated by a space).\n";
  s << "  -d\n";
  s << "     Print vocab and embedding dimension before embeddings\n";
  s << "     (separated by a newline).\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  bool with_words(false), with_dims(false);

  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "wdh");
    switch (ret) {
      case 'w':
        with_words = true;
        break;
      case 'd':
        with_dims = true;
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

  info(__func__, "loading model ...\n");
  auto sentence_learner(FileSerializer<SGNSSentenceLearnerType>(input_path).load());
  WordContextFactorization& factorization(sentence_learner.token_learner.factorization);
  NaiveLanguageModel& language_model(sentence_learner.token_learner.language_model);

  ofstream f;
  f.open(output_path);
  stream_ready_or_throw(f);
  if (with_dims) {
    info(__func__, "printing dimensions to file ...\n");
    f << factorization.get_vocab_dim();
    f.write(" ", 1);
    f << factorization.get_embedding_dim();
    f.write("\n", 1);
  }
  info(__func__, "printing embeddings to file ...\n");
  for (size_t i = 0; i < factorization.get_vocab_dim(); ++i) {
    if (with_words) {
      string word(language_model.reverse_lookup(i));
      f.write(word.c_str(), word.size());
      f.write(" ", 1);
    }
    for (size_t j = 0; j < factorization.get_embedding_dim(); ++j) {
      f << factorization.get_word_embedding(i)[j];
      f.write(" ", 1); // write extra space at end, just like word2vec
    }
    f.write("\n", 1);
  }
  f.close();

  info(__func__, "done\n");
}
