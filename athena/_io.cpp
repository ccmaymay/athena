#include "_io.h"


#include <string>
#include <stdexcept>
#include <istream>
#include <cstddef>
#include <vector>


using namespace std;


void stream_ready_or_throw(ios& stream) {
  if (! stream) {
    throw runtime_error(string("stream_ready_or_throw: stream not ready"));
  }
}


void SentenceReader::_load_next_sentence() {
  _has_next_sentence = false;
  _next_sentence.clear();

  string word;
  while (_f) {
    const char c = _f.get();
    if (c == '\r') {
      continue;
    }
    if (c == ' ' || c == '\n' || c == '\t') {
      if (! word.empty()) {
        _next_sentence.push_back(word);
        word.clear();
        _has_next_sentence = true;
        if (_next_sentence.size() == _sentence_limit) {
          break;
        }
      }
      if (c == '\n') { _has_next_sentence = true;
        break;
      }
    } else {
      word.push_back(c);
    }
  }

  _initialized = true;
}

bool SentenceReader::has_next() {
  // if this is the first call, load a sentence
  if (! _initialized) {
    _load_next_sentence();
  }
  // return whether previous sentence load was successful
  return _has_next_sentence;
}

vector<string> SentenceReader::next() {
  // if this is the first call, load a sentence
  if (! _initialized) {
    _load_next_sentence();
  }
  // copy previously-loaded sentence
  vector<string> sentence = _next_sentence;
  // load a sentence for the next call
  _load_next_sentence();
  // return previously-loaded sentence
  return sentence;
}

void SentenceReader::reset() {
  _f.seekg(0, _f.beg);
  _initialized = false;
}
