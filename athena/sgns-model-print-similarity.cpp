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


using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
    cerr << "usage: " << argv[0] << " <input-path> <word-pair-list-path> <output-path>\n";
    exit(1);
  }
  const char *input_path = argv[1];
  const char *word_pair_list_path = argv[2];
  const char *output_path = argv[3];

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
