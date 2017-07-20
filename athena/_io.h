#ifndef ATHENA__IO_H
#define ATHENA__IO_H


#include <ios>
#include <istream>
#include <cstddef>
#include <string>
#include <vector>


void stream_ready_or_throw(std::ios& stream);


class SentenceReader {
  std::istream& _f;
  size_t _sentence_limit;
  bool _initialized, _has_next_sentence;
  std::vector<std::string> _next_sentence;

  public:
    SentenceReader(std::istream& f, size_t sentence_limit = 1000):
      _f(f),
      _sentence_limit(sentence_limit),
      _initialized(false),
      _has_next_sentence(false),
      _next_sentence() { }

    bool has_next();
    std::vector<std::string> next();
    void reset();

  private:
    void _load_next_sentence();
};


#endif
