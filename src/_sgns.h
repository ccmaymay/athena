#ifndef ATHENA__SGNS_H
#define ATHENA__SGNS_H


#include "_core.h"
#include "_cblas.h"
#include "_log.h"
#include "_math.h"
#include "_serialization.h"

#include <fstream>
#include <cstring>
#include <utility>
#include <limits>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>


// Core SGNS implementation.  At training time takes a single input-output
// word pair (and a specification of the desired number of negative
// samples).  Not intended to be called directly (see SGNSSentenceLearner
// instead).

template <class LanguageModel, class SamplingStrategy, class SGDType = SGD>
class SGNSTokenLearner;

template <class LanguageModel, class SamplingStrategy, class SGDType>
class SGNSTokenLearner {
  public:
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<LanguageModel> language_model;
    std::shared_ptr<SGDType> sgd;

    SGNSTokenLearner(
        std::shared_ptr<WordContextFactorization> factorization_,
        std::shared_ptr<SamplingStrategy> neg_sampling_strategy_,
        std::shared_ptr<LanguageModel> language_model_,
        std::shared_ptr<SGDType> sgd_):
            factorization(factorization_),
            neg_sampling_strategy(neg_sampling_strategy_),
            language_model(language_model_),
            sgd(sgd_) { }
    void reset_word(long word_idx);
    void token_train(size_t target_word_idx, size_t context_word_idx,
                     size_t neg_samples);
    float compute_gradient_coeff(long target_word_idx,
                                 long context_word_idx,
                                 bool negative_sample);
    float compute_similarity(size_t word1_idx, size_t word2_idx);
    long find_nearest_neighbor_idx(size_t word_idx);
    long find_context_nearest_neighbor_idx(size_t left_context,
                                           size_t right_context,
                                           const long *word_ids);
    bool context_contains_oov(const long* ctx_word_ids, size_t ctx_size) const;
    ~SGNSTokenLearner() { }

    bool equals(const SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>& other) const;
    void serialize(std::ostream& stream) const;
    static SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType> deserialize(std::istream& stream);

    SGNSTokenLearner(SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>&& other):
        factorization(other.factorization),
        neg_sampling_strategy(other.neg_sampling_strategy),
        language_model(other.language_model),
        sgd(other.sgd) { }
    SGNSTokenLearner(const SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>& other):
        factorization(other.factorization),
        neg_sampling_strategy(other.neg_sampling_strategy),
        language_model(other.language_model),
        sgd(other.sgd) { }

  private:
    void _context_word_gradient(long target_word, long context_word);
    void _neg_sample_word_gradient(long target_word, long neg_sample_word);
    void _predicted_word_gradient(long target_word, long predicted_word);
};


// Wraps SGNSTokenLearner, providing the logic for training over
// sentences (sequences of overlapping contexts) and also for looping
// over the words within each context.

template <class SGNSTokenLearnerType, class ContextStrategy = DynamicContextStrategy>
class SGNSSentenceLearner;

template <class SGNSTokenLearnerType, class ContextStrategy>
class SGNSSentenceLearner {
  public:
    std::shared_ptr<SGNSTokenLearnerType> token_learner;
    std::shared_ptr<ContextStrategy> ctx_strategy;
    size_t neg_samples;

    SGNSSentenceLearner(
        std::shared_ptr<SGNSTokenLearnerType> token_learner_,
        std::shared_ptr<ContextStrategy> ctx_strategy_,
        size_t neg_samples_):
      token_learner(token_learner_),
      ctx_strategy(ctx_strategy_),
      neg_samples(neg_samples_) { }
    void sentence_train(const std::vector<long>& word_ids);
    ~SGNSSentenceLearner() { }

    virtual bool
      equals(const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>
      deserialize(std::istream& stream);

    SGNSSentenceLearner(SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>&& other):
        token_learner(other.token_learner),
        ctx_strategy(other.ctx_strategy),
        neg_samples(other.neg_samples) { }
    SGNSSentenceLearner(const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other):
        token_learner(other.token_learner),
        ctx_strategy(other.ctx_strategy),
        neg_samples(other.neg_samples) { }
};


//
// SGNSTokenLearner
//


template <class LanguageModel, class SamplingStrategy, class SGDType>
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::reset_word(long word_idx) {
  sgd->reset(word_idx);
  sample_centered_uniform_vector(
    factorization->get_embedding_dim(),
    factorization->get_word_embedding(word_idx)
  );
  memset(factorization->get_context_embedding(word_idx), 0,
         factorization->get_embedding_dim() * sizeof(float));
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
long SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::find_context_nearest_neighbor_idx(size_t left_context,
                                                    size_t right_context,
                                                    const long *word_ids) {
  long best_candidate_word_idx = -1;
  float best_score = -std::numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < language_model->size();
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
              factorization->get_embedding_dim(),
              factorization->get_word_embedding(candidate_word_idx), 1,
              factorization->get_context_embedding(context_word_idx), 1
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

template <class LanguageModel, class SamplingStrategy, class SGDType>
float SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::compute_similarity(size_t word1_idx,
                                           size_t word2_idx) {
  return cblas_sdot(
    factorization->get_embedding_dim(),
    factorization->get_word_embedding(word1_idx), 1,
    factorization->get_word_embedding(word2_idx), 1
  ) / (
    cblas_snrm2(
      factorization->get_embedding_dim(),
      factorization->get_word_embedding(word1_idx), 1
    ) * cblas_snrm2(
      factorization->get_embedding_dim(),
      factorization->get_word_embedding(word2_idx), 1
    )
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
long SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::find_nearest_neighbor_idx(size_t word_idx) {
  long best_candidate_word_idx = -1;
  float best_score = -std::numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < language_model->size();
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

template <class LanguageModel, class SamplingStrategy, class SGDType>
float SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::compute_gradient_coeff(long target_word_idx,
                                           long context_word_idx,
                                           bool negative_sample) {
  return (negative_sample ? 0 : 1) - fast_sigmoid(cblas_sdot(
    factorization->get_embedding_dim(),
    factorization->get_word_embedding(target_word_idx), 1,
    factorization->get_context_embedding(context_word_idx), 1
  ));
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
bool SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::context_contains_oov(const long* ctx_word_ids,
                                              size_t ctx_size) const {
  for (size_t i = 0; i < ctx_size; ++i) {
    if (ctx_word_ids[i] < 0) {
      return true;
    }
  }
  return false;
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::token_train(size_t target_word_idx,
                                     size_t context_word_idx,
                                     size_t neg_samples) {
  // initialize target (input) word gradient
  AlignedVector target_word_gradient(
    factorization->get_embedding_dim());
  memset(target_word_gradient.data(), 0,
    sizeof(float) * factorization->get_embedding_dim());

  // compute contribution of context (output) word to target (input)
  // word gradient, take context word gradient step
  const float coeff = compute_gradient_coeff(target_word_idx,
                                              context_word_idx, false);
  cblas_saxpy(
    factorization->get_embedding_dim(),
    coeff,
    factorization->get_context_embedding(context_word_idx), 1,
    target_word_gradient.data(), 1
  );
  sgd->scaled_gradient_update(
    context_word_idx,
    factorization->get_embedding_dim(),
    factorization->get_word_embedding(target_word_idx),
    factorization->get_context_embedding(context_word_idx),
    coeff
  );

  for (size_t j = 0; j < neg_samples; ++j) {
    // compute contribution of neg-sample word to target (input) word
    // gradient, take neg-sample word gradient step
    const long neg_sample_word_idx =
      neg_sampling_strategy->sample_idx(*language_model);

    const float coeff = compute_gradient_coeff(target_word_idx,
                                               neg_sample_word_idx, true);
    cblas_saxpy(
      factorization->get_embedding_dim(),
      coeff,
      factorization->get_context_embedding(neg_sample_word_idx), 1,
      target_word_gradient.data(), 1
    );
    sgd->scaled_gradient_update(
      neg_sample_word_idx,
      factorization->get_embedding_dim(),
      factorization->get_word_embedding(target_word_idx),
      factorization->get_context_embedding(neg_sample_word_idx),
      coeff
    );
  }

  // take target (input) word gradient step
  sgd->gradient_update(
    target_word_idx,
    factorization->get_embedding_dim(),
    target_word_gradient.data(),
    factorization->get_word_embedding(target_word_idx)
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::serialize(std::ostream& stream) const {
  Serializer<WordContextFactorization>::serialize(*factorization, stream);
  Serializer<SamplingStrategy>::serialize(*neg_sampling_strategy, stream);
  Serializer<LanguageModel>::serialize(*language_model, stream);
  Serializer<SGDType>::serialize(*sgd, stream);
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>
SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::deserialize(std::istream& stream) {
  auto factorization_(Serializer<WordContextFactorization>::deserialize(stream));
  auto neg_sampling_strategy_(Serializer<SamplingStrategy>::deserialize(stream));
  auto language_model_(Serializer<LanguageModel>::deserialize(stream));
  auto sgd_(Serializer<SGDType>::deserialize(stream));
  return SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>(
    std::make_shared<WordContextFactorization>(std::move(factorization_)),
    std::make_shared<SamplingStrategy>(std::move(neg_sampling_strategy_)),
    std::make_shared<LanguageModel>(std::move(language_model_)),
    std::make_shared<SGDType>(std::move(sgd_))
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
bool SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::equals(const SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>& other) const {
  return factorization->equals(*(other.factorization)) &&
    neg_sampling_strategy->equals(*(other.neg_sampling_strategy)) &&
    language_model->equals(*(other.language_model)) &&
    sgd->equals(*(other.sgd));
}


//
// SGNSSentenceLearner
//


template <class SGNSTokenLearnerType, class ContextStrategy>
void SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::sentence_train(
    const std::vector<long>& word_ids) {
  // loop over all contexts, training on each non-empty one
  for (size_t input_word_pos = 0; input_word_pos < word_ids.size();
       ++input_word_pos) {
    // compute context size
    const std::pair<size_t,size_t> ctx_sizes = ctx_strategy->size(
      input_word_pos, (word_ids.size() - 1) - input_word_pos
    );
    const size_t left_ctx = ctx_sizes.first;
    const size_t right_ctx = ctx_sizes.second;

    const size_t ctx_start = input_word_pos - left_ctx;
    const size_t ctx_end = ctx_start + left_ctx + 1 + right_ctx;

    // train on current context
    for (size_t output_word_pos = ctx_start; output_word_pos < ctx_end; ++output_word_pos) {
      // for all context (output) words...
      if (output_word_pos != input_word_pos) {
        token_learner->token_train(word_ids[input_word_pos],
                                    word_ids[output_word_pos], neg_samples);
      }
    }
  }
}

template <class SGNSTokenLearnerType, class ContextStrategy>
void SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::serialize(std::ostream& stream) const {
  Serializer<SGNSTokenLearnerType>::serialize(*token_learner, stream);
  Serializer<ContextStrategy>::serialize(*ctx_strategy, stream);
  Serializer<size_t>::serialize(neg_samples, stream);
}

template <class SGNSTokenLearnerType, class ContextStrategy>
SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>
SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::deserialize(std::istream& stream) {
  auto token_learner_(Serializer<SGNSTokenLearnerType>::deserialize(stream));
  auto ctx_strategy_(Serializer<ContextStrategy>::deserialize(stream));
  auto neg_samples_(Serializer<size_t>::deserialize(stream));
  return SGNSSentenceLearner(
    std::make_shared<SGNSTokenLearnerType>(std::move(token_learner_)),
    std::make_shared<ContextStrategy>(std::move(ctx_strategy_)),
    neg_samples_
  );
}

template <class SGNSTokenLearnerType, class ContextStrategy>
bool SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::equals(
    const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other) const {
  return
    token_learner->equals(*(other.token_learner)) &&
    ctx_strategy->equals(*(other.ctx_strategy)) &&
    neg_samples == other.neg_samples;
}


#endif
