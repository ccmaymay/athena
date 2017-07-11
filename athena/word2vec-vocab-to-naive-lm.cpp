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
  s << "Build naive language model from Google word2vec vocabulary.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (Google word2vec vocabulary).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (serialized language model).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -s <subsample-threshold>\n";
  s << "     Default: " << DEFAULT_SUBSAMPLE_THRESHOLD << "\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  float subsample_threshold(DEFAULT_SUBSAMPLE_THRESHOLD);

  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "v:s:h");
    switch (ret) {
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
    if (c == ' ') {
      unsigned long count = 1;
      f >> count;
      for (size_t i = 0; i < count; ++i) {
        language_model->increment(word);
      }
      if (f.get() != '\n') {
        throw runtime_error(string("malformatted vocabulary"));
      }
      word.clear();
    } else {
      word.push_back(c);
    }
  }
  f.close();

  info(__func__, "saving ...\n");
  FileSerializer<LanguageModel>(output_path).dump(*language_model);

  info(__func__, "done\n");
}
