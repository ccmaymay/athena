#include "_core.h"
#include "_log.h"
#include "_math.h"
#include "_serialization.h"
#include "_cblas.h"
#include "_sgns.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
#include <cstring>
#include <utility>


using namespace std;


//
// SGNSModel
//


void SGNSModel::serialize(ostream& stream) const {
  Serializer<WordContextFactorization>::serialize(*factorization, stream);
  Serializer<SamplingStrategy>::serialize(*neg_sampling_strategy, stream);
  Serializer<LanguageModel>::serialize(*language_model, stream);
  Serializer<SGD>::serialize(*sgd, stream);
  Serializer<ContextStrategy>::serialize(*ctx_strategy, stream);
  Serializer<SGNSTokenLearner>::serialize(*token_learner, stream);
  Serializer<SGNSSentenceLearner>::serialize(*sentence_learner, stream);
  Serializer<SubsamplingSGNSSentenceLearner>::serialize(
    *subsampling_sentence_learner, stream);
}

shared_ptr<SGNSModel> SGNSModel::deserialize(istream& stream) {
  auto factorization_(
      Serializer<WordContextFactorization>::deserialize(stream));
  auto neg_sampling_strategy_(
      Serializer<SamplingStrategy>::deserialize(stream));
  auto language_model_(
      Serializer<LanguageModel>::deserialize(stream));
  auto sgd_(Serializer<SGD>::deserialize(stream));
  auto ctx_strategy_(Serializer<ContextStrategy>::deserialize(stream));
  auto token_learner_(Serializer<SGNSTokenLearner>::deserialize(stream));
  auto sentence_learner_(Serializer<SGNSSentenceLearner>::deserialize(stream));
  auto subsampling_sentence_learner_(
      Serializer<SubsamplingSGNSSentenceLearner>::deserialize(stream));
  auto model(make_shared<SGNSModel>(
    factorization_,
    neg_sampling_strategy_,
    language_model_,
    sgd_,
    ctx_strategy_,
    token_learner_,
    sentence_learner_,
    subsampling_sentence_learner_
  ));
  token_learner_->set_model(model);
  sentence_learner_->set_model(model);
  subsampling_sentence_learner_->set_model(model);
  return model;
}

bool SGNSModel::equals(const SGNSModel& other) const {
  return
    factorization->equals(*(other.factorization)) &&
    neg_sampling_strategy->equals(*(other.neg_sampling_strategy)) &&
    language_model->equals(*(other.language_model)) &&
    sgd->equals(*(other.sgd)) &&
    ctx_strategy->equals(*(other.ctx_strategy)) &&
    token_learner->equals(*(other.token_learner)) &&
    sentence_learner->equals(*(other.sentence_learner)) &&
    subsampling_sentence_learner->equals(
      *(other.subsampling_sentence_learner));
}


//
// SGNSTokenLearner
//


void SGNSTokenLearner::reset_word(long word_idx) {
  auto model = _model.lock();
  sample_centered_uniform_vector(
    model->factorization->get_embedding_dim(),
    model->factorization->get_word_embedding(word_idx)
  );
  memset(model->factorization->get_context_embedding(word_idx), 0,
         model->factorization->get_embedding_dim() * sizeof(float));
}

long SGNSTokenLearner::find_context_nearest_neighbor_idx(size_t left_context,
                                                    size_t right_context,
                                                    const long *word_ids) {
  auto model = _model.lock();

  long best_candidate_word_idx = -1;
  float best_score = -numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < model->language_model->size();
       ++candidate_word_idx) {
    // should we try to take a MAP estimate here?
    float log_prob_ctx_given_candidate = 0;
    for (size_t i = 0; i < left_context + 1 + right_context; ++i) {
      // for all context (output) words...
      if (i != left_context) {
        const long context_word_idx = word_ids[i];
        if (context_word_idx >= 0) {
          log_prob_ctx_given_candidate += fast_sigmoid(
            cblas_sdot(
              model->factorization->get_embedding_dim(),
              model->factorization->get_word_embedding(candidate_word_idx), 1,
              model->factorization->get_context_embedding(context_word_idx), 1
            )
          );
        }
      }
    }

    if (log_prob_ctx_given_candidate > best_score) {
      best_candidate_word_idx = (long) candidate_word_idx;
      best_score = log_prob_ctx_given_candidate;
    }
  }

  return best_candidate_word_idx;
}

float SGNSTokenLearner::compute_similarity(size_t word1_idx,
                                           size_t word2_idx) const {
  auto model = _model.lock();

  return cblas_sdot(
    model->factorization->get_embedding_dim(),
    model->factorization->get_word_embedding(word1_idx), 1,
    model->factorization->get_word_embedding(word2_idx), 1
  ) / (
    cblas_snrm2(
      model->factorization->get_embedding_dim(),
      model->factorization->get_word_embedding(word1_idx), 1
    ) * cblas_snrm2(
      model->factorization->get_embedding_dim(),
      model->factorization->get_word_embedding(word2_idx), 1
    )
  );
}

long SGNSTokenLearner::find_nearest_neighbor_idx(size_t word_idx) {
  auto model = _model.lock();

  long best_candidate_word_idx = -1;
  float best_score = -numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < model->language_model->size();
       ++candidate_word_idx) {
    if (candidate_word_idx != word_idx) {
      const float score = compute_similarity(candidate_word_idx, word_idx);
      if (score > best_score) {
        best_candidate_word_idx = (long) candidate_word_idx;
        best_score = score;
      }
    }
  }

  return best_candidate_word_idx;
}

float SGNSTokenLearner::compute_gradient_coeff(long target_word_idx,
                                           long context_word_idx,
                                           bool negative_sample) const {
  auto model = _model.lock();
  return (negative_sample ? 0 : 1) - fast_sigmoid(cblas_sdot(
    model->factorization->get_embedding_dim(),
    model->factorization->get_word_embedding(target_word_idx), 1,
    model->factorization->get_context_embedding(context_word_idx), 1
  ));
}

bool SGNSTokenLearner::context_contains_oov(const long* ctx_word_ids,
                                              size_t ctx_size) const {
  for (size_t i = 0; i < ctx_size; ++i) {
    if (ctx_word_ids[i] < 0) {
      return true;
    }
  }
  return false;
}

void SGNSTokenLearner::token_train(size_t target_word_idx,
                                     size_t context_word_idx,
                                     size_t neg_samples) {
  auto model = _model.lock();

  // initialize target (input) word gradient
  AlignedVector target_word_gradient(
    model->factorization->get_embedding_dim());
  memset(target_word_gradient.data(), 0,
    sizeof(float) * model->factorization->get_embedding_dim());

  // compute contribution of context (output) word to target (input)
  // word gradient, take context word gradient step
  const float coeff = compute_gradient_coeff(target_word_idx,
                                              context_word_idx, false);
  cblas_saxpy(
    model->factorization->get_embedding_dim(),
    coeff,
    model->factorization->get_context_embedding(context_word_idx), 1,
    target_word_gradient.data(), 1
  );
  model->sgd->scaled_gradient_update(
    model->factorization->get_embedding_dim(),
    model->factorization->get_word_embedding(target_word_idx),
    model->factorization->get_context_embedding(context_word_idx),
    coeff
  );

  for (size_t j = 0; j < neg_samples; ++j) {
    // compute contribution of neg-sample word to target (input) word
    // gradient, take neg-sample word gradient step
    const long neg_sample_word_idx =
      model->neg_sampling_strategy->sample_idx(*(model->language_model));

    const float coeff = compute_gradient_coeff(target_word_idx,
                                               neg_sample_word_idx, true);
    cblas_saxpy(
      model->factorization->get_embedding_dim(),
      coeff,
      model->factorization->get_context_embedding(neg_sample_word_idx), 1,
      target_word_gradient.data(), 1
    );
    model->sgd->scaled_gradient_update(
      model->factorization->get_embedding_dim(),
      model->factorization->get_word_embedding(target_word_idx),
      model->factorization->get_context_embedding(neg_sample_word_idx),
      coeff
    );
  }

  // take target (input) word gradient step
  model->sgd->gradient_update(
    model->factorization->get_embedding_dim(),
    target_word_gradient.data(),
    model->factorization->get_word_embedding(target_word_idx)
  );
}

void SGNSTokenLearner::serialize(ostream& stream) const { }

shared_ptr<SGNSTokenLearner> SGNSTokenLearner::deserialize(istream& stream) {
  return make_shared<SGNSTokenLearner>();
}

bool SGNSTokenLearner::equals(const SGNSTokenLearner& other) const {
  return true;
}


//
// SGNSSentenceLearner
//


void SGNSSentenceLearner::increment(const string& word) {
  auto model = _model.lock();

  const pair<long,string> ejectee =
    model->language_model->increment(word);
  const long ejectee_idx = ejectee.first;
  if (ejectee_idx >= 0) {
    model->token_learner->reset_word(ejectee_idx);
  }
  model->neg_sampling_strategy->step(*(model->language_model),
                                      model->language_model->lookup(word));
}

void SGNSSentenceLearner::sentence_train(const vector<string>& words) {
  auto model = _model.lock();

  // (optionally) add all words in sentence to language model
  if (_propagate_retained)
    for (auto it = words.begin(); it != words.end(); ++it)
      increment(*it);

  // compute in-vocabulary subsequence (dirty: effective context size
  // grows to skip over out-of-vocabulary words)
  vector<long> word_ids;
  word_ids.reserve(words.size());
  size_t num_oov = 0;
  for (auto it = words.begin(); it != words.end(); ++it) {
    long word_id = model->language_model->lookup(*it);
    if (word_id >= 0) {
      word_ids.push_back(word_id);
    } else {
      ++num_oov;
    }
  }
  debug(__func__, "skipping " << num_oov << " OOV words in sentence\n");

  // loop over all contexts, training on each non-empty one
  for (size_t target_word_pos = 0; target_word_pos < word_ids.size();
       ++target_word_pos) {
    // compute context size
    const pair<size_t,size_t> ctx_size = model->ctx_strategy->size(
      target_word_pos, (word_ids.size() - 1) - target_word_pos
    );
    const size_t left_ctx = ctx_size.first;
    const size_t right_ctx = ctx_size.second;

    // if computed context is non-empty, train
    if (left_ctx > 0 || right_ctx > 0) {
      const size_t ctx_size = left_ctx + 1 + right_ctx;
      const size_t ctx_start = target_word_pos - left_ctx;
      const size_t ctx_end = ctx_start + ctx_size;

      // train on current context
      for (size_t i = ctx_start; i < ctx_end; ++i) {
        // for all context (output) words...
        if (i != target_word_pos) {
          model->token_learner->token_train(word_ids[target_word_pos],
                                            word_ids[i], _neg_samples);
        }
      }

      model->sgd->step();
    }
  }
}

void SGNSSentenceLearner::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_neg_samples, stream);
  Serializer<bool>::serialize(_propagate_retained, stream);
}

shared_ptr<SGNSSentenceLearner> SGNSSentenceLearner::deserialize(istream& stream) {
  auto neg_samples(*Serializer<size_t>::deserialize(stream));
  auto propagate_retained(*Serializer<bool>::deserialize(stream));
  return make_shared<SGNSSentenceLearner>(
    neg_samples,
    propagate_retained
  );
}

bool SGNSSentenceLearner::equals(const SGNSSentenceLearner& other) const {
  return
    _neg_samples == other._neg_samples &&
    _propagate_retained == other._propagate_retained;
}


//
// SubsamplingSGNSSentenceLearner
//


void SubsamplingSGNSSentenceLearner::sentence_train(const vector<string>&
                                                      words) {
  auto model = _model.lock();

  vector<string> subsampled_words;
  for (auto it = words.begin(); it != words.end(); ++it) {
    const string& word(*it);
    const long word_id = model->language_model->lookup(word);
    if (word_id < 0 || model->language_model->subsample(word_id)) {
      subsampled_words.push_back(word);
    } else if (_propagate_discarded) {
      model->sentence_learner->increment(word);
    }
  }
  model->sentence_learner->sentence_train(subsampled_words);
}

void SubsamplingSGNSSentenceLearner::serialize(ostream& stream) const {
  Serializer<bool>::serialize(_propagate_discarded, stream);
}

shared_ptr<SubsamplingSGNSSentenceLearner>
    SubsamplingSGNSSentenceLearner::deserialize(istream& stream) {
  auto propagate_discarded(*Serializer<bool>::deserialize(stream));
  return make_shared<SubsamplingSGNSSentenceLearner>(
    propagate_discarded
  );
}

bool SubsamplingSGNSSentenceLearner::equals(
    const SubsamplingSGNSSentenceLearner& other) const {
  return
    _propagate_discarded == other._propagate_discarded;
}
