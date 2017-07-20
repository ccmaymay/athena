#include "_core.h"
#include "_log.h"
#include "_io.h"
#include "_math.h"
#include "_serialization.h"

#include <ctime>
#include <vector>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>


#define SENTENCE_LIMIT 1000


using namespace std;

void usage(ostream& s, const string& program) {
  s << "Train Space-Saving language model from text file.\n";
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
  SpaceSavingLanguageModel language_model(vocab_dim, subsample_threshold);

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
      language_model.increment(*it);
      ++words_seen;
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
  FileSerializer<SpaceSavingLanguageModel>(output_path).dump(language_model);

  info(__func__, "done\n");
}
