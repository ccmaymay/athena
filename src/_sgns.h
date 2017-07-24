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
class SGNSTokenLearner final {
  public:
    WordContextFactorization factorization;
    SamplingStrategy neg_sampling_strategy;
    LanguageModel language_model;
    SGDType sgd;

    SGNSTokenLearner(
        WordContextFactorization&& factorization_,
        SamplingStrategy&& neg_sampling_strategy_,
        LanguageModel&& language_model_,
        SGDType&& sgd_):
            factorization(std::move(factorization_)),
            neg_sampling_strategy(std::move(neg_sampling_strategy_)),
            language_model(std::move(language_model_)),
            sgd(std::move(sgd_)) { }
    void reset_word(long word_idx);
    void token_train(size_t input_word_idx, size_t output_word_idx,
                     size_t neg_samples);
    float compute_gradient_coeff(long input_word_idx,
                                 long output_word_idx,
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

    SGNSTokenLearner(SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>&& other) = default;
    SGNSTokenLearner(const SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>& other) = default;
};


// Wraps SGNSTokenLearner, providing the logic for training over
// sentences (sequences of overlapping contexts) and also for looping
// over the words within each context.

template <class SGNSTokenLearnerType, class ContextStrategy = DynamicContextStrategy>
class SGNSSentenceLearner;

template <class SGNSTokenLearnerType, class ContextStrategy>
class SGNSSentenceLearner final {
  public:
    SGNSTokenLearnerType token_learner;
    ContextStrategy ctx_strategy;
    size_t neg_samples;

    SGNSSentenceLearner(
        SGNSTokenLearnerType&& token_learner_,
        ContextStrategy&& ctx_strategy_,
        size_t neg_samples_):
      token_learner(std::move(token_learner_)),
      ctx_strategy(std::move(ctx_strategy_)),
      neg_samples(neg_samples_) { }
    void sentence_train(const std::vector<long>& word_ids);
    ~SGNSSentenceLearner() { }

    bool equals(const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other) const;
    void serialize(std::ostream& stream) const;
    static SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>
      deserialize(std::istream& stream);

    SGNSSentenceLearner(SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>&& other) = default;
    SGNSSentenceLearner(const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other) = default;
};


//
// SGNSTokenLearner
//


template <class LanguageModel, class SamplingStrategy, class SGDType>
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::reset_word(long word_idx) {
  sgd.reset(word_idx);
  sample_centered_uniform_vector(
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(word_idx)
  );
  memset(factorization.get_context_embedding(word_idx), 0,
         factorization.get_embedding_dim() * sizeof(float));
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
long SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::find_context_nearest_neighbor_idx(size_t left_context,
                                                    size_t right_context,
                                                    const long *word_ids) {
  long best_candidate_word_idx = -1;
  float best_score = -std::numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < language_model.size();
       ++candidate_word_idx) {
    // should we try to take a MAP estimate here?
    float log_prob_ctx_given_candidate = 0;
    for (size_t i = 0; i < left_context + 1 + right_context; ++i) {
      // for all output words...
      if (i != left_context) {
        const long output_word_idx = word_ids[i];
        if (output_word_idx >= 0) {
          log_prob_ctx_given_candidate += fast_sigmoid(
            cblas_sdot(
              factorization.get_embedding_dim(),
              factorization.get_word_embedding(candidate_word_idx), 1,
              factorization.get_context_embedding(output_word_idx), 1
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
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(word1_idx), 1,
    factorization.get_word_embedding(word2_idx), 1
  ) / (
    cblas_snrm2(
      factorization.get_embedding_dim(),
      factorization.get_word_embedding(word1_idx), 1
    ) * cblas_snrm2(
      factorization.get_embedding_dim(),
      factorization.get_word_embedding(word2_idx), 1
    )
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
long SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::find_nearest_neighbor_idx(size_t word_idx) {
  long best_candidate_word_idx = -1;
  float best_score = -std::numeric_limits<float>::max();

  for (size_t candidate_word_idx = 0;
       candidate_word_idx < language_model.size();
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
float SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::compute_gradient_coeff(long input_word_idx,
                                           long output_word_idx,
                                           bool negative_sample) {
  return (negative_sample ? 0 : 1) - fast_sigmoid(cblas_sdot(
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(input_word_idx), 1,
    factorization.get_context_embedding(output_word_idx), 1
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
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::token_train(size_t input_word_idx,
                                     size_t output_word_idx,
                                     size_t neg_samples) {
  // initialize input word gradient
  AlignedVector input_word_gradient(
    factorization.get_embedding_dim());
  memset(input_word_gradient.data(), 0,
    sizeof(float) * factorization.get_embedding_dim());

  // compute contribution of output word to input
  // word gradient, take output word gradient step
  const float coeff = compute_gradient_coeff(input_word_idx,
                                              output_word_idx, false);
  cblas_saxpy(
    factorization.get_embedding_dim(),
    coeff,
    factorization.get_context_embedding(output_word_idx), 1,
    input_word_gradient.data(), 1
  );
  sgd.scaled_gradient_update(
    output_word_idx,
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(input_word_idx),
    factorization.get_context_embedding(output_word_idx),
    coeff
  );

  for (size_t j = 0; j < neg_samples; ++j) {
    // compute contribution of neg-sample word to input word
    // gradient, take neg-sample word gradient step
    const long neg_sample_word_idx =
      neg_sampling_strategy.sample_idx(language_model);

    const float coeff = compute_gradient_coeff(input_word_idx,
                                               neg_sample_word_idx, true);
    cblas_saxpy(
      factorization.get_embedding_dim(),
      coeff,
      factorization.get_context_embedding(neg_sample_word_idx), 1,
      input_word_gradient.data(), 1
    );
    sgd.scaled_gradient_update(
      neg_sample_word_idx,
      factorization.get_embedding_dim(),
      factorization.get_word_embedding(input_word_idx),
      factorization.get_context_embedding(neg_sample_word_idx),
      coeff
    );
  }

  // take input word gradient step
  sgd.gradient_update(
    input_word_idx,
    factorization.get_embedding_dim(),
    input_word_gradient.data(),
    factorization.get_word_embedding(input_word_idx)
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
void SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::serialize(std::ostream& stream) const {
  Serializer<WordContextFactorization>::serialize(factorization, stream);
  Serializer<SamplingStrategy>::serialize(neg_sampling_strategy, stream);
  Serializer<LanguageModel>::serialize(language_model, stream);
  Serializer<SGDType>::serialize(sgd, stream);
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>
SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::deserialize(std::istream& stream) {
  auto factorization_(Serializer<WordContextFactorization>::deserialize(stream));
  auto neg_sampling_strategy_(Serializer<SamplingStrategy>::deserialize(stream));
  auto language_model_(Serializer<LanguageModel>::deserialize(stream));
  auto sgd_(Serializer<SGDType>::deserialize(stream));
  return SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>(
    std::move(factorization_),
    std::move(neg_sampling_strategy_),
    std::move(language_model_),
    std::move(sgd_)
  );
}

template <class LanguageModel, class SamplingStrategy, class SGDType>
bool SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>::equals(const SGNSTokenLearner<LanguageModel,SamplingStrategy,SGDType>& other) const {
  return factorization.equals(other.factorization) &&
    neg_sampling_strategy.equals(other.neg_sampling_strategy) &&
    language_model.equals(other.language_model) &&
    sgd.equals(other.sgd);
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
    const std::pair<size_t,size_t> ctx_sizes = ctx_strategy.size(
      input_word_pos, (word_ids.size() - 1) - input_word_pos
    );
    const size_t left_ctx = ctx_sizes.first;
    const size_t right_ctx = ctx_sizes.second;

    const size_t ctx_start = input_word_pos - left_ctx;
    const size_t ctx_end = ctx_start + left_ctx + 1 + right_ctx;

    // train on current context
    for (size_t output_word_pos = ctx_start; output_word_pos < ctx_end; ++output_word_pos) {
      if (output_word_pos != input_word_pos) {
        token_learner.token_train(word_ids[input_word_pos],
                                    word_ids[output_word_pos], neg_samples);
      }
    }
  }
}

template <class SGNSTokenLearnerType, class ContextStrategy>
void SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::serialize(std::ostream& stream) const {
  Serializer<SGNSTokenLearnerType>::serialize(token_learner, stream);
  Serializer<ContextStrategy>::serialize(ctx_strategy, stream);
  Serializer<size_t>::serialize(neg_samples, stream);
}

template <class SGNSTokenLearnerType, class ContextStrategy>
SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>
SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::deserialize(std::istream& stream) {
  auto token_learner_(Serializer<SGNSTokenLearnerType>::deserialize(stream));
  auto ctx_strategy_(Serializer<ContextStrategy>::deserialize(stream));
  auto neg_samples_(Serializer<size_t>::deserialize(stream));
  return SGNSSentenceLearner(
    std::move(token_learner_),
    std::move(ctx_strategy_),
    neg_samples_
  );
}

template <class SGNSTokenLearnerType, class ContextStrategy>
bool SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>::equals(
    const SGNSSentenceLearner<SGNSTokenLearnerType,ContextStrategy>& other) const {
  return
    token_learner.equals(other.token_learner) &&
    ctx_strategy.equals(other.ctx_strategy) &&
    neg_samples == other.neg_samples;
}


#endif
