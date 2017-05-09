#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_math.h"
#include "_serialization.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>


#define VOCAB_DIM 7000
#define SUBSAMPLE_THRESHOLD 1e-3


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
  auto language_model(
    make_shared<SpaceSavingLanguageModel>(VOCAB_DIM, SUBSAMPLE_THRESHOLD));

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
  f.close();

  info(__func__, "saving ...\n");
  FileSerializer<LanguageModel>(output_path).dump(*language_model);

  info(__func__, "done\n");
}
