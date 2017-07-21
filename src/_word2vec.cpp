#include "_word2vec.h"
#include "_cblas.h"
#include "_serialization.h"
#include "_log.h"


using namespace std;


void Word2VecModel::serialize(ostream& stream) const {
  throw logic_error(string("Word2VecModel::serialize: not implemented"));
}

shared_ptr<Word2VecModel> Word2VecModel::deserialize(istream& stream) {
  auto model(make_shared<Word2VecModel>());
  char c;

  stream >> model->vocab_dim;
  debug(__func__, "vocab_dim: " << model->vocab_dim << "\n");
  c = stream.get();
  if (c != ' ') {
    throw runtime_error(
      string("Word2VecModel::deserialize: expected space but got ") +
      string(1, c));
  }

  stream >> model->embedding_dim;
  debug(__func__, "embedding_dim: " << model->embedding_dim << "\n");
  c = stream.get();
  if (c != '\n') {
    throw runtime_error(
      string("Word2VecModel::deserialize: expected newline but got ") +
      string(1, c));
  }

  debug(__func__, "allocating space for model\n");
  model->word_embeddings.resize(model->vocab_dim * model->embedding_dim);
  model->vocab.resize(model->vocab_dim);

  for (long long word_num = 0; word_num < model->vocab_dim; ++word_num) {
    /* read word */
    if (word_num % 1000 == 0) {
      debug(__func__, "reading word " << word_num << "\n");
    }
    stream >> model->vocab[word_num];
    c = stream.get();
    if (c != ' ') {
      throw runtime_error(
        string("Word2VecModel::deserialize: expected space but got ") +
        string(1, c));
    }

    /* read word vector and l2-normalize */
    if (word_num % 1000 == 0) {
      debug(__func__,
        "reading embedding for " << model->vocab[word_num] << "\n");
    }
    stream.read(
      reinterpret_cast<char*>(
        model->word_embeddings.data() + word_num * model->embedding_dim),
      model->embedding_dim * sizeof(float));
    float norm = cblas_snrm2(
      model->embedding_dim,
      model->word_embeddings.data() + word_num * model->embedding_dim,
      1);
    cblas_sscal(
      model->embedding_dim,
      1. / norm,
      model->word_embeddings.data() + word_num * model->embedding_dim,
      1);
  }

  return model;
}
