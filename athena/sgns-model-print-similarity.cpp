#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_sgns.h"
#include "_serialization.h"

#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>


using namespace std;

void usage(ostream& s, const string& program) {
  s << "Load SGNS model and word pair list and print similarities of word pairs to a file.\n";
  s << "\n";
  s << "Usage: " << program << " [...] <input-path> <word-pair-list-path> <output-path>\n";
  s << "\n";
  s << "Required arguments:\n";
  s << "  <input-path>\n";
  s << "     Path to input file (serialized SGNS model).\n";
  s << "  <word-pair-list-path>\n";
  s << "     Path to word pair list (one pair per line: two words separated by a tab).\n";
  s << "  <output-path>\n";
  s << "     Path to output file (one triple per line: two words and a similarity score separated by tabs).\n";
  s << "\n";
  s << "Optional arguments:\n";
  s << "  -h\n";
  s << "     Print this help and exit.\n";
}

int main(int argc, char **argv) {
  const string program(argv[0]);

  int ret = 0;
  while (ret != -1) {
    ret = getopt(argc, argv, "h");
    switch (ret) {
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
  if (optind + 3 != argc) {
    usage(cerr, program);
    exit(1);
  }
  const char *input_path = argv[optind];
  const char *word_pair_list_path = argv[optind + 1];
  const char *output_path = argv[optind + 2];

  info(__func__, "loading model ...\n");
  auto model(FileSerializer<SGNSModel>(input_path).load());

  info(__func__, "loading word-pair list ...\n");
  string word_1, word_2;
  vector<pair<string,string> > word_pair_list;
  bool in_word_2 = false;
  ifstream f;
  f.open(word_pair_list_path);
  stream_ready_or_throw(f);
  while (f) {
    const char c = f.get();
    if (c == '\n') {
      word_pair_list.push_back(make_pair(word_1, word_2));
      word_1.clear();
      word_2.clear();
      in_word_2 = false;
    } else if (c == '\t') {
      in_word_2 = true;
    } else if (in_word_2) {
      word_2.push_back(c);
    } else {
      word_1.push_back(c);
    }
  }
  f.close();

  info(__func__, "printing similarity to file ...\n");
  ofstream of;
  of.open(output_path);
  stream_ready_or_throw(of);
  for (size_t w = 0; w < word_pair_list.size(); ++w) {
    string word_1(word_pair_list[w].first);
    string word_2(word_pair_list[w].second);
    of.write(word_1.c_str(), word_1.size());
    of.write("\t", 1);
    of.write(word_2.c_str(), word_2.size());
    of.write("\t", 1);

    long word_1_idx(model->language_model->lookup(word_1));
    long word_2_idx(model->language_model->lookup(word_2));
    float sim =
      (word_1_idx >= 0 && word_2_idx >= 0) ?
      model->token_learner->compute_similarity(word_1_idx, word_2_idx) :
      NAN;

    of << sim;
    of.write("\n", 1);
  }
  f.close();

  info(__func__, "done\n");
}
