#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_serialization.h"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unistd.h>


using namespace std;

void usage(ostream& s, const string& program) {
  s << "Load serialized language model and print its words to a file.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (serialized language model).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (text representation of language model).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -c\n";
  s << "     Print word count next to each word (separated by a space).\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  bool with_counts(false);

  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "ch");
    switch (ret) {
      case 'c':
        with_counts = true;
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
  auto language_model(FileSerializer<NaiveLanguageModel>(input_path).load());

  info(__func__, "printing words in vocabulary to file ...\n");
  ofstream f;
  f.open(output_path);
  stream_ready_or_throw(f);
  for (size_t w = 0; w < language_model.size(); ++w) {
    string word(language_model.reverse_lookup(w));
    f.write(word.c_str(), word.size());
    if (with_counts) {
      f.write(" ", 1);
      f << language_model.count(w);
    }
    f.write("\n", 1);
  }
  f.close();

  info(__func__, "done\n");
}
