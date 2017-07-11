#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_math.h"
#include "_serialization.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>


using namespace std;

void usage(ostream& s, const string& program) {
  s << "Train naive language model from text file.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (training text).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (serialized language model).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -v <vocab-dim>\n";
  s << "     Default: " << DEFAULT_VOCAB_DIM << "\n";
  s << "  -s <subsample-threshold>\n";
  s << "     Default: " << DEFAULT_SUBSAMPLE_THRESHOLD << "\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  size_t vocab_dim(DEFAULT_VOCAB_DIM);
  float subsample_threshold(DEFAULT_SUBSAMPLE_THRESHOLD);

  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "v:s:h");
    switch (ret) {
      case 'v':
        vocab_dim = stoull(string(optarg));
        break;
      case 's':
        subsample_threshold = stof(string(optarg));
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

  info(__func__, "initializing model ...\n");
  auto language_model(make_shared<NaiveLanguageModel>(subsample_threshold));

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
    if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
      if (! word.empty()) {
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

  info(__func__, "saving ...\n");
  FileSerializer<LanguageModel>(output_path).dump(*language_model);

  info(__func__, "done\n");
}
