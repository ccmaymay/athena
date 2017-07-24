#ifndef ATHENA__WORD2VEC_H
#define ATHENA__WORD2VEC_H


#include <vector>
#include <string>
#include <iostream>
#include <memory>


struct Word2VecModel;


// word2vec model as defined in Google C code

struct Word2VecModel {
  long long vocab_dim;
  long long embedding_dim;
  std::vector<std::string> vocab;
  std::vector<float> word_embeddings;

  void serialize(std::ostream& stream) const;
  static std::shared_ptr<Word2VecModel> deserialize(std::istream& stream);
};


#endif
