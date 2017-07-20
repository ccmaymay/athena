#ifndef ATHENA__SGNS_H
#define ATHENA__SGNS_H


#include "_core.h"
#include "_cblas.h"
#include <limits>
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>
#include <memory>


/*
class SGNSSentenceLearner;
class SubsamplingSGNSSentenceLearner;
*/


// Core SGNS implementation.  At training time takes a single input-output
// word pair (and a specification of the desired number of negative
// samples).  Not intended to be called directly (see SGNSSentenceLearner
// instead).

template <class SamplingStrategy, class LanguageModel>
class SGNSTokenLearner;

template <class SamplingStrategy, class LanguageModel>
class SGNSTokenLearner {
  public:
    WordContextFactorization factorization;
    SamplingStrategy neg_sampling_strategy;
    LanguageModel language_model;
    SGD sgd;

    SGNSTokenLearner(
        WordContextFactorization&& factorization_,
        SamplingStrategy&& neg_sampling_strategy_,
        LanguageModel&& language_model_,
        SGD&& sgd_):
            factorization(std::move(factorization_)),
            neg_sampling_strategy(std::move(neg_sampling_strategy_)),
            language_model(std::move(language_model_)),
            sgd(std::move(sgd_)) { }
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
    bool context_contains_oov(const long* ctx_word_ids,
                                      size_t ctx_size) const;
    ~SGNSTokenLearner() { }

    /*
    virtual bool equals(const SGNSTokenLearner& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SGNSTokenLearner&& deserialize(std::istream& stream);
    */

  private:
    void _context_word_gradient(long target_word, long context_word);
    void _neg_sample_word_gradient(long target_word, long neg_sample_word);
    void _predicted_word_gradient(long target_word, long predicted_word);
    SGNSTokenLearner(const SGNSTokenLearner& s);
};


/*
// Wraps SGNSTokenLearner, providing the logic for training over
// sentences (sequences of overlapping contexts) and also for looping
// over the words within each context.

class SGNSSentenceLearner {
  public:
    SGNSTokenLearner* token_learner;
    ContextStrategy* ctx_strategy;

  private:
    size_t _neg_samples;
    bool _propagate_retained;

  public:
    SGNSSentenceLearner(SGNSTokenLearner* token_learner_,
        ContextStrategy* ctx_strategy_,
        size_t neg_samples, bool propagate_retained):
      token_learner(token_learner_),
      ctx_strategy(ctx_strategy_),
      _neg_samples(neg_samples),
      _propagate_retained(propagate_retained) { }
    void increment(const std::string& word);
    void sentence_train(const std::vector<std::string>& words);
    ~SGNSSentenceLearner() { }

    //virtual bool equals(const SGNSSentenceLearner& other) const;
    //virtual void serialize(std::ostream& stream) const;
    //static SGNSSentenceLearner* deserialize(std::istream& stream);

  private:
    SGNSSentenceLearner(const SGNSSentenceLearner& s);
};


// Wraps SGNSSentenceLearner, subsampling words by frequency
// (as in word2vec) before training.

class SubsamplingSGNSSentenceLearner {
  public:
    SGNSSentenceLearner* sentence_learner;

  private:
    bool _propagate_discarded;

  public:
    SubsamplingSGNSSentenceLearner(
        SGNSSentenceLearner* sentence_learner_,
        bool propagate_discarded):
      sentence_learner(sentence_learner_),
      _propagate_discarded(propagate_discarded) { }
    void sentence_train(const std::vector<std::string>& words);
    ~SubsamplingSGNSSentenceLearner() { }

    //virtual bool equals(const SubsamplingSGNSSentenceLearner& other) const;
    //virtual void serialize(std::ostream& stream) const;
    //static SubsamplingSGNSSentenceLearner* deserialize(std::istream& stream);

  private:
    SubsamplingSGNSSentenceLearner(const SubsamplingSGNSSentenceLearner& s);
};
*/


//
// SGNSTokenLearner
//


template <class SamplingStrategy, class LanguageModel>
void SGNSTokenLearner<SamplingStrategy, LanguageModel>::reset_word(long word_idx) {
  sgd.reset(word_idx);
  sample_centered_uniform_vector(
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(word_idx)
  );
  memset(factorization.get_context_embedding(word_idx), 0,
         factorization.get_embedding_dim() * sizeof(float));
}

template <class SamplingStrategy, class LanguageModel>
long SGNSTokenLearner<SamplingStrategy, LanguageModel>::find_context_nearest_neighbor_idx(size_t left_context,
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
      // for all context (output) words...
      if (i != left_context) {
        const long context_word_idx = word_ids[i];
        if (context_word_idx >= 0) {
          log_prob_ctx_given_candidate += fast_sigmoid(
            cblas_sdot(
              factorization.get_embedding_dim(),
              factorization.get_word_embedding(candidate_word_idx), 1,
              factorization.get_context_embedding(context_word_idx), 1
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

template <class SamplingStrategy, class LanguageModel>
float SGNSTokenLearner<SamplingStrategy, LanguageModel>::compute_similarity(size_t word1_idx,
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

template <class SamplingStrategy, class LanguageModel>
long SGNSTokenLearner<SamplingStrategy, LanguageModel>::find_nearest_neighbor_idx(size_t word_idx) {
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

template <class SamplingStrategy, class LanguageModel>
float SGNSTokenLearner<SamplingStrategy, LanguageModel>::compute_gradient_coeff(long target_word_idx,
                                           long context_word_idx,
                                           bool negative_sample) {
  return (negative_sample ? 0 : 1) - fast_sigmoid(cblas_sdot(
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(target_word_idx), 1,
    factorization.get_context_embedding(context_word_idx), 1
  ));
}

template <class SamplingStrategy, class LanguageModel>
bool SGNSTokenLearner<SamplingStrategy, LanguageModel>::context_contains_oov(const long* ctx_word_ids,
                                              size_t ctx_size) const {
  for (size_t i = 0; i < ctx_size; ++i) {
    if (ctx_word_ids[i] < 0) {
      return true;
    }
  }
  return false;
}

template <class SamplingStrategy, class LanguageModel>
void SGNSTokenLearner<SamplingStrategy, LanguageModel>::token_train(size_t target_word_idx,
                                     size_t context_word_idx,
                                     size_t neg_samples) {
  // initialize target (input) word gradient
  AlignedVector target_word_gradient(
    factorization.get_embedding_dim());
  memset(target_word_gradient.data(), 0,
    sizeof(float) * factorization.get_embedding_dim());

  // compute contribution of context (output) word to target (input)
  // word gradient, take context word gradient step
  const float coeff = compute_gradient_coeff(target_word_idx,
                                              context_word_idx, false);
  cblas_saxpy(
    factorization.get_embedding_dim(),
    coeff,
    factorization.get_context_embedding(context_word_idx), 1,
    target_word_gradient.data(), 1
  );
  sgd.scaled_gradient_update(
    context_word_idx,
    factorization.get_embedding_dim(),
    factorization.get_word_embedding(target_word_idx),
    factorization.get_context_embedding(context_word_idx),
    coeff
  );

  for (size_t j = 0; j < neg_samples; ++j) {
    // compute contribution of neg-sample word to target (input) word
    // gradient, take neg-sample word gradient step
    const long neg_sample_word_idx =
      neg_sampling_strategy.sample_idx(language_model);

    const float coeff = compute_gradient_coeff(target_word_idx,
                                               neg_sample_word_idx, true);
    cblas_saxpy(
      factorization.get_embedding_dim(),
      coeff,
      factorization.get_context_embedding(neg_sample_word_idx), 1,
      target_word_gradient.data(), 1
    );
    sgd.scaled_gradient_update(
      neg_sample_word_idx,
      factorization.get_embedding_dim(),
      factorization.get_word_embedding(target_word_idx),
      factorization.get_context_embedding(neg_sample_word_idx),
      coeff
    );
  }

  // take target (input) word gradient step
  sgd.gradient_update(
    target_word_idx,
    factorization.get_embedding_dim(),
    target_word_gradient.data(),
    factorization.get_word_embedding(target_word_idx)
  );
}
#endif
