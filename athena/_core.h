#ifndef ATHENA__CORE_H
#define ATHENA__CORE_H


#include <cstddef>
#include <cstring>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "_math.h"


// frequent-word subsampling threshold as defined in word2vec.
#define DEFAULT_SUBSAMPLE_THRESHOLD 1e-3
#define DEFAULT_VOCAB_DIM 16000
#define DEFAULT_EMBEDDING_DIM 200
#define DEFAULT_REFRESH_INTERVAL 64000
#define DEFAULT_REFRESH_BURN_IN 32000
#define DEFAULT_RESERVOIR_SIZE 100000000

#define ALIGN_EACH_EMBEDDING 1


// pair comparators

template <typename T, typename U>
bool pair_first_cmp(std::pair<T,U> x, std::pair<T,U> y) {
  return x.first < y.first;
}

template <typename T, typename U>
bool pair_second_cmp(std::pair<T,U> x, std::pair<T,U> y) {
  return x.second < y.second;
}


// Language model abstract base class

enum language_model_t {
  naive_lm,
  space_saving_lm
};

class LanguageModel {
  public:
    // return ejected (index, word) pair
    // (index is -1 if nothing was ejected)
    virtual std::pair<long,std::string> increment(const std::string& word) = 0;
    // return index of word (-1 if does not exist)
    virtual long lookup(const std::string& word) const = 0;
    // return word at index (raise exception if does not exist)
    virtual std::string reverse_lookup(long word_idx) const = 0;
    // return count at word index
    virtual size_t count(long word_idx) const = 0;
    // return counts of all word indices
    virtual std::vector<size_t> counts() const = 0;
    // return ordered (descending) counts of all word indices
    virtual std::vector<size_t> ordered_counts() const = 0;
    // return number of word types present in language model
    virtual size_t size() const = 0;
    // return total number of word tokens seen by language model
    virtual size_t total() const = 0;
    // return true if word should be kept after subsampling
    // (return true with probability
    // sqrt(subsample_threshold / f(word_idx)) where f(word_idx) is the
    // normalized frequency corresponding to word_idx)
    virtual bool subsample(long word_idx) const = 0;
    // truncate language model to top max_size types; do nothing if
    // language model has at most max_size types already
    virtual void truncate(size_t max_size) = 0;
    virtual ~LanguageModel() { }

    virtual bool equals(const LanguageModel& other) const;
    virtual void serialize(std::ostream& stream) const = 0;
    static LanguageModel* deserialize(std::istream& stream);

  protected:
    LanguageModel() { }

  private:
    LanguageModel(const LanguageModel& lm);
};


// Language model implemented naively

class NaiveLanguageModel : public LanguageModel {
  float _subsample_threshold;
  size_t _size;
  size_t _total;
  std::vector<size_t> _counters;
  std::unordered_map<std::string,long> _word_ids;
  std::vector<std::string> _words;

  public:
    NaiveLanguageModel(float subsample_threshold = DEFAULT_SUBSAMPLE_THRESHOLD);
    // return ejected (index, word) pair
    // (index is -1 if nothing was ejected)
    virtual std::pair<long,std::string> increment(const std::string& word);
    // return index of word (-1 if does not exist)
    virtual long lookup(const std::string& word) const;
    // return word at index (raise exception if does not exist)
    virtual std::string reverse_lookup(long word_idx) const;
    // return count at word index
    virtual size_t count(long word_idx) const;
    // return counts of all word indices
    virtual std::vector<size_t> counts() const;
    // return ordered (descending) counts of all word indices
    virtual std::vector<size_t> ordered_counts() const;
    // return number of word types present in language model
    virtual size_t size() const;
    // return total number of word tokens seen by language model
    virtual size_t total() const;
    // return true if word should be kept after subsampling
    // (return true with probability
    // sqrt(subsample_threshold / f(word_idx)) where f(word_idx) is the
    // normalized frequency corresponding to word_idx)
    virtual bool subsample(long word_idx) const;
    virtual void truncate(size_t max_size);
    // sort language model words by count (descending)
    virtual void sort();
    virtual ~NaiveLanguageModel() { }

    virtual bool equals(const LanguageModel& other) const;
    virtual void serialize(std::ostream& stream) const;
    static NaiveLanguageModel* deserialize(std::istream& stream);

    NaiveLanguageModel(float subsample_threshold,
                  size_t size,
                  size_t total,
                  std::vector<size_t>&& counters,
                  std::unordered_map<std::string,long>&& word_ids,
                  std::vector<std::string>&& words):
        _subsample_threshold(subsample_threshold),
        _size(size),
        _total(total),
        _counters(std::forward<std::vector<size_t> >(counters)),
        _word_ids(
          std::forward<std::unordered_map<std::string,long> >(word_ids)),
        _words(std::forward<std::vector<std::string> >(words)) { }

  private:
    NaiveLanguageModel(const NaiveLanguageModel& lm);
};


// Language model implemented on SpaceSaving approximate counter.

class SpaceSavingLanguageModel : public LanguageModel {
  float _subsample_threshold;
  size_t _num_counters;
  size_t _size;
  size_t _total;
  size_t _min_idx;
  std::vector<size_t> _counters;
  std::unordered_map<std::string,long> _word_ids;
  std::vector<long> _internal_ids;
  std::vector<long> _external_ids;
  std::vector<std::string> _words;

  public:
    SpaceSavingLanguageModel(
      size_t num_counters = DEFAULT_VOCAB_DIM,
      float subsample_threshold = DEFAULT_SUBSAMPLE_THRESHOLD);
    // return ejected (index, word) pair
    // (index is -1 if nothing was ejected)
    virtual std::pair<long,std::string> increment(const std::string& word);
    // return index of word (-1 if does not exist)
    virtual long lookup(const std::string& word) const;
    // return word at index (raise exception if does not exist)
    virtual std::string reverse_lookup(long ext_word_idx) const;
    // return count at word index
    virtual size_t count(long ext_word_idx) const;
    // return counts of all word indices
    virtual std::vector<size_t> counts() const;
    // return ordered (descending) counts of all word indices
    virtual std::vector<size_t> ordered_counts() const;
    // return number of word types present in language model
    virtual size_t size() const;
    // return number of word types possible language model
    virtual size_t capacity() const;
    // return total number of word tokens seen by language model
    virtual size_t total() const;
    // return true if word should be kept after subsampling
    // (return true with probability
    // sqrt(subsample_threshold / f(word_idx)) where f(word_idx) is the
    // normalized frequency corresponding to word_idx)
    virtual bool subsample(long ext_word_idx) const;
    virtual void truncate(size_t max_size);
    virtual ~SpaceSavingLanguageModel() { }

    virtual bool equals(const LanguageModel& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SpaceSavingLanguageModel* deserialize(std::istream& stream);

    SpaceSavingLanguageModel(float subsample_threshold,
                             size_t num_counters,
                             size_t size,
                             size_t total,
                             size_t min_idx,
                             std::vector<size_t>&& counters,
                             std::unordered_map<std::string,long>&& word_ids,
                             std::vector<long>&& internal_ids,
                             std::vector<long>&& external_ids,
                             std::vector<std::string>&& words):
        _subsample_threshold(subsample_threshold),
        _num_counters(num_counters),
        _size(size),
        _total(total),
        _min_idx(min_idx),
        _counters(std::forward<std::vector<size_t> >(counters)),
        _word_ids(
          std::forward<std::unordered_map<std::string,long> >(word_ids)),
        _internal_ids(std::forward<std::vector<long> >(internal_ids)),
        _external_ids(std::forward<std::vector<long> >(external_ids)),
        _words(std::forward<std::vector<std::string> >(words)) { }

  private:
    void _update_min_idx();
    std::pair<long,std::string> _unfull_append(const std::string& word);
    std::pair<long,std::string> _full_replace(const std::string& word);
    std::pair<long,std::string> _full_increment(long ext_idx);
    SpaceSavingLanguageModel(const SpaceSavingLanguageModel& sslm);
};


// Word-context matrix factorization model.

class WordContextFactorization {
  size_t _vocab_dim, _embedding_dim, _actual_embedding_dim;
  AlignedVector _word_embeddings, _context_embeddings;

  public:
    WordContextFactorization(size_t vocab_dim = DEFAULT_VOCAB_DIM,
                             size_t embedding_dim = DEFAULT_EMBEDDING_DIM);
    virtual size_t get_embedding_dim();
    virtual size_t get_vocab_dim();
    virtual float* get_word_embedding(size_t word_idx);
    virtual float* get_context_embedding(size_t word_idx);
    virtual ~WordContextFactorization() { }

    virtual bool equals(const WordContextFactorization& other) const;
    virtual void serialize(std::ostream& stream) const;
    static WordContextFactorization*
      deserialize(std::istream& stream);

    WordContextFactorization(size_t vocab_dim,
                             size_t embedding_dim,
                             size_t actual_embedding_dim,
                             AlignedVector&& word_embeddings,
                             AlignedVector&& context_embeddings):
        _vocab_dim(vocab_dim),
        _embedding_dim(embedding_dim),
        _actual_embedding_dim(actual_embedding_dim),
        _word_embeddings(std::forward<AlignedVector>(word_embeddings)),
        _context_embeddings(
          std::forward<AlignedVector>(context_embeddings)) { }

  private:
    WordContextFactorization(const WordContextFactorization& wcf);
};


// Stochastic gradient descent parametrization and state.

class SGD {
  size_t _dimension;
  float _tau, _kappa, _rho_lower_bound;
  std::vector<float> _rho;
  std::vector<size_t> _t;

  public:
    SGD(size_t dimension = 1, float tau = 0, float kappa = 0.6,
        float rho_lower_bound = 0);
    virtual void step(size_t dim);
    virtual float get_rho(size_t dim) const;
    virtual void gradient_update(size_t dim, size_t n, const float *g,
                                 float *x);
    virtual void scaled_gradient_update(size_t dim, size_t n, const float *g,
                                        float *x, float alpha);
    virtual void reset(size_t dim);
    virtual ~SGD() { };

    virtual bool equals(const SGD& other) const;
    virtual void serialize(std::ostream& stream) const;
    static SGD* deserialize(std::istream& stream);

    SGD(size_t dimension,
        float tau,
        float kappa,
        float rho_lower_bound,
        std::vector<float>&& rho,
        std::vector<size_t>&& t):
          _dimension(dimension),
          _tau(tau),
          _kappa(kappa),
          _rho_lower_bound(rho_lower_bound),
          _rho(std::forward<std::vector<float> >(rho)),
          _t(std::forward<std::vector<size_t> >(t)) { }

  private:
    void _compute_rho(size_t dimension);
    SGD(const SGD& sgd);
};


// Discrete sampling strategy for language model (abstract base class).

enum sampling_strategy_t {
  uniform,
  empirical,
  reservoir
};

class SamplingStrategy {
  public:
    // return
    // word idx distributed according to implementation-defined
    // distribution
    virtual long
      sample_idx(const LanguageModel& language_model) = 0;
    // update internal state according to language model
    // (this should be called incrementally as the language model
    // is updated)
    virtual void
      step(const LanguageModel& language_model, size_t word_idx) { }
    virtual void
      reset(const LanguageModel& language_model,
            const CountNormalizer& normalizer) { }
    virtual ~SamplingStrategy() { }

    virtual bool equals(const SamplingStrategy& other) const { return true; }
    virtual void serialize(std::ostream& stream) const = 0;
    static SamplingStrategy* deserialize(std::istream& stream);

  protected:
    SamplingStrategy() { }

  private:
    SamplingStrategy(const SamplingStrategy& sampling_strategy);
};


// Uniform sampling strategy for language model.

class UniformSamplingStrategy : public SamplingStrategy {
  public:
    using SamplingStrategy::SamplingStrategy;
    // sample from uniform distribution
    virtual long sample_idx(const LanguageModel& language_model);

    virtual void serialize(std::ostream& stream) const;
    static UniformSamplingStrategy*
      deserialize(std::istream& stream) {
        return new UniformSamplingStrategy();
      }
};


// Empirical sampling strategy for language model.

class EmpiricalSamplingStrategy : public SamplingStrategy {
  size_t _refresh_interval;
  size_t _refresh_burn_in;
  CountNormalizer* _normalizer;
  AliasSampler* _alias_sampler;
  size_t _t;
  bool _initialized;

  public:
    EmpiricalSamplingStrategy(CountNormalizer* normalizer,
                              size_t
                                refresh_interval = DEFAULT_REFRESH_INTERVAL,
                              size_t
                                refresh_burn_in = DEFAULT_REFRESH_BURN_IN);
    // if we have taken no more than refresh_burn_in steps
    // or the number of steps since then is a multiple of
    // refresh_interval, refresh (recompute) distribution
    // based on current counts
    virtual void
      step(const LanguageModel& language_model, size_t word_idx);
    // reset distribution according to specified language model, using
    // specified count normalizer (ignore normalizer provided to ctor)
    virtual void
      reset(const LanguageModel& language_model,
            const CountNormalizer& normalizer);
    // sample from (potentially stale) empirical distribution
    // computed by transforming counts via normalizer
    virtual long sample_idx(const LanguageModel& language_model);
    virtual ~EmpiricalSamplingStrategy() { }

    virtual bool equals(const SamplingStrategy& other) const;
    virtual void serialize(std::ostream& stream) const;
    static EmpiricalSamplingStrategy*
      deserialize(std::istream& stream);

    EmpiricalSamplingStrategy(size_t refresh_interval,
                              size_t refresh_burn_in,
                              CountNormalizer* normalizer,
                              AliasSampler* alias_sampler,
                              size_t t,
                              bool initialized):
        _refresh_interval(refresh_interval),
        _refresh_burn_in(refresh_burn_in),
        _normalizer(normalizer),
        _alias_sampler(alias_sampler),
        _t(t),
        _initialized(initialized) { }
};


// Reservoir sampling strategy for language model.

class ReservoirSamplingStrategy : public SamplingStrategy {
  ReservoirSampler<long>* _reservoir_sampler;

  public:
    ReservoirSamplingStrategy(
      ReservoirSampler<long>* reservoir_sampler):
        _reservoir_sampler(reservoir_sampler) { }
    // (randomly) add word to reservoir
    virtual void
      step(const LanguageModel& language_model, size_t word_idx) {
        _reservoir_sampler->insert(word_idx);
      }
    // re-populate reservoir according to language model
    virtual void
      reset(const LanguageModel& language_model,
            const CountNormalizer& normalizer);
    virtual long sample_idx(const LanguageModel& language_model) {
      return _reservoir_sampler->sample();
    }
    virtual ~ReservoirSamplingStrategy() { }
    virtual bool equals(const SamplingStrategy& other) const;
    virtual void serialize(std::ostream& stream) const;
    static ReservoirSamplingStrategy*
      deserialize(std::istream& stream);
};


enum context_strategy_t {
  static_ctx,
  dynamic_ctx
};


// Context size strategy (abstract base class).

class ContextStrategy {
  public:
    // return number of words in left and right context (respectively)
    // given there are at most avail_left and avail_right words to the
    // left and right (respectively); return pair (0,0) if no context
    virtual std::pair<size_t,size_t> size(size_t avail_left,
                                          size_t avail_right) const = 0;
    virtual ~ContextStrategy() { }

    virtual bool equals(const ContextStrategy& other) const { return true; }
    virtual void serialize(std::ostream& stream) const = 0;
    static ContextStrategy* deserialize(std::istream& stream);

  protected:
    ContextStrategy() { }

  private:
    ContextStrategy(const ContextStrategy& context_strategy);
};


// Static context strategy

class StaticContextStrategy : public ContextStrategy {
  size_t _symm_context;

  public:
    StaticContextStrategy(size_t symm_context):
      ContextStrategy(), _symm_context(symm_context) { }
    // return static (fixed) thresholded context
    virtual std::pair<size_t,size_t> size(size_t avail_left,
                                          size_t avail_right) const;
    virtual bool equals(const ContextStrategy& other) const;
    virtual void serialize(std::ostream& stream) const;
    static StaticContextStrategy*
      deserialize(std::istream& stream);
};


// Dynamic context strategy

class DynamicContextStrategy : public ContextStrategy {
  size_t _symm_context;

  public:
    DynamicContextStrategy(size_t symm_context):
      ContextStrategy(), _symm_context(symm_context) { }
    // return dynamic (sampled) thresholded context
    virtual std::pair<size_t,size_t> size(size_t avail_left,
                                          size_t avail_right) const;
    virtual bool equals(const ContextStrategy& other) const;
    virtual void serialize(std::ostream& stream) const;
    static DynamicContextStrategy*
      deserialize(std::istream& stream);
};


// Space-saving LM with reservoirs attached to words
// (Idea: heavy-hitting word types and usage examples)

template <typename T>
class LanguageModelExampleStore;

template <typename T>
class LanguageModelExampleStore {
  LanguageModel* _language_model;
  size_t _num_examples_per_word;
  std::vector<ReservoirSampler<T> > _examples;

  public:
    LanguageModelExampleStore(LanguageModel* language_model,
                              size_t num_examples_per_word):
        _language_model(language_model),
        _num_examples_per_word(num_examples_per_word),
        _examples() { }
    virtual std::pair<long,std::string> increment(const std::string& word,
                                                  const T& example);
    virtual const LanguageModel& get_language_model() const {
      return *_language_model;
    }
    virtual const ReservoirSampler<T>& get_examples(long idx) {
      return _examples[idx];
    }
    virtual ~LanguageModelExampleStore() { }

    virtual bool equals(const LanguageModelExampleStore<T>& other) const;
    virtual void serialize(std::ostream& stream) const;
    static LanguageModelExampleStore<T>*
      deserialize(std::istream& stream);

    LanguageModelExampleStore(LanguageModel* language_model,
                              size_t num_examples_per_word,
                              std::vector<ReservoirSampler<T> >&& examples):
      _language_model(language_model),
      _num_examples_per_word(num_examples_per_word),
      _examples(std::forward<std::vector<ReservoirSampler<T> > >(examples)) { }

  private:
    LanguageModelExampleStore(const LanguageModelExampleStore<T>&
                                lm_example_store);
};


//
// LanguageModelExampleStore
//


template <typename T>
std::pair<long,std::string> LanguageModelExampleStore<T>::increment(
    const std::string& word, const T& example) {
  // if ejection, clear corresponding reservoir
  std::pair<long,std::string> ejectee(_language_model->increment(word));
  if (ejectee.first >= 0 &&
      static_cast<size_t>(ejectee.first) < _examples.size()) {
    _examples[ejectee.first].clear();
  }

  // pad reservoir vector as needed and insert example
  const long idx = _language_model->lookup(word);
  if (idx >= 0) {
    for (size_t i = _examples.size(); i <= static_cast<size_t>(idx); ++i) {
      _examples.emplace_back(_num_examples_per_word);
    }
    _examples[idx].insert(example);
  }

  return ejectee;
}

template <typename T>
bool LanguageModelExampleStore<T>::equals(
    const LanguageModelExampleStore<T>& other) const {
  return
    _language_model->equals(*(other._language_model)) &&
    _num_examples_per_word == other._num_examples_per_word &&
    _examples == other._examples;
}

template <typename T>
void LanguageModelExampleStore<T>::serialize(std::ostream& stream) const {
  Serializer<LanguageModel>::serialize(*_language_model, stream);
  Serializer<size_t>::serialize(_num_examples_per_word, stream);
  Serializer<std::vector<ReservoirSampler<T> > >::serialize(_examples, stream);
}

template <typename T>
LanguageModelExampleStore<T>*
    LanguageModelExampleStore<T>::deserialize(std::istream& stream) {
  auto language_model(Serializer<LanguageModel>::deserialize(stream));
  auto num_examples_per_word(*Serializer<size_t>::deserialize(stream));
  auto examples(Serializer<std::vector<ReservoirSampler<T> > >::deserialize(stream));
  return new LanguageModelExampleStore<T>(
    language_model,
    num_examples_per_word,
    std::move(*examples)
  );
}


#endif
