#ifndef ATHENA__SGNS_H
#define ATHENA__SGNS_H


#include "_core.h"
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>
#include <memory>


class SGNSTokenLearner;
class SGNSSentenceLearner;
class SubsamplingSGNSSentenceLearner;


// Core SGNS implementation.  At training time takes a single input-output
// word pair (and a specification of the desired number of negative
// samples).  Not intended to be called directly (see SGNSSentenceLearner
// instead).

class SGNSTokenLearner {
  public:
    WordContextFactorization* factorization;
    ReservoirSamplingStrategy* neg_sampling_strategy;
    NaiveLanguageModel* language_model;
    SGD* sgd;

    SGNSTokenLearner(
        WordContextFactorization* factorization_,
        ReservoirSamplingStrategy* neg_sampling_strategy_,
        NaiveLanguageModel* language_model_,
        SGD* sgd_):
            factorization(factorization_),
            neg_sampling_strategy(neg_sampling_strategy_),
            language_model(language_model_),
            sgd(sgd_) { }
    virtual void reset_word(long word_idx);
    virtual void token_train(size_t target_word_idx, size_t context_word_idx,
                             size_t neg_samples);
    virtual float compute_gradient_coeff(long target_word_idx,
                                         long context_word_idx,
                                         bool negative_sample) const;
    virtual float compute_similarity(size_t word1_idx, size_t word2_idx) const;
    virtual long find_nearest_neighbor_idx(size_t word_idx);
    virtual long find_context_nearest_neighbor_idx(size_t left_context,
                                                   size_t right_context,
                                                   const long *word_ids);
    virtual bool context_contains_oov(const long* ctx_word_ids,
                                      size_t ctx_size) const;
    virtual ~SGNSTokenLearner() { }

    virtual bool equals(const SGNSTokenLearner& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SGNSTokenLearner* deserialize(std::istream& stream);

  private:
    void _context_word_gradient(long target_word, long context_word);
    void _neg_sample_word_gradient(long target_word, long neg_sample_word);
    void _predicted_word_gradient(long target_word, long predicted_word);
    SGNSTokenLearner(const SGNSTokenLearner& s);
};


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
    virtual void increment(const std::string& word);
    virtual void sentence_train(const std::vector<std::string>& words);
    virtual ~SGNSSentenceLearner() { }

    virtual bool equals(const SGNSSentenceLearner& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SGNSSentenceLearner* deserialize(std::istream& stream);

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
    virtual void sentence_train(const std::vector<std::string>& words);
    virtual ~SubsamplingSGNSSentenceLearner() { }

    virtual bool equals(const SubsamplingSGNSSentenceLearner& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SubsamplingSGNSSentenceLearner* deserialize(std::istream& stream);

  private:
    SubsamplingSGNSSentenceLearner(const SubsamplingSGNSSentenceLearner& s);
};


#endif
