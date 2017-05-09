#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_serialization.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>


using namespace std;

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    cerr << "usage: " << argv[0] << " [--with-counts] <input-path> <output-path>\n";
    exit(1);
  }
  char *input_path = NULL,
       *output_path = NULL;
  bool with_counts = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--with-counts") == 0) {
      with_counts = true;
    } else if (input_path == NULL) {
      input_path = argv[i];
    } else if (output_path == NULL) {
      output_path = argv[i];
    } else {
      throw runtime_error(string("main: unexpected argument list"));
    }
  }
  if (input_path == NULL || output_path == NULL) {
    throw runtime_error(string("main: input-path and output-path must be specified"));
  }

  info(__func__, "loading model ...\n");
  auto language_model(FileSerializer<LanguageModel>(input_path).load());

  info(__func__, "printing words in vocabulary to file ...\n");
  ofstream f;
  f.open(output_path);
  stream_ready_or_throw(f);
  for (size_t w = 0; w < language_model->size(); ++w) {
    string word(language_model->reverse_lookup(w));
    f.write(word.c_str(), word.size());
    if (with_counts) {
      f.write("\t", 1);
      f << language_model->count(w);
    }
    f.write("\n", 1);
  }
  f.close();

  info(__func__, "done\n");
}
