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
#include <random>
#include <memory>
#include <algorithm>

#include "_math.h"


// frequent-word subsampling threshold as defined in word2vec.
#define DEFAULT_SUBSAMPLE_THRESHOLD 1e-3
#define DEFAULT_VOCAB_DIM 16000
#define DEFAULT_EMBEDDING_DIM 100
#define DEFAULT_REFRESH_INTERVAL 0
#define DEFAULT_REFRESH_BURN_IN 1000
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


// Language model implemented naively

class NaiveLanguageModel {
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
    std::pair<long,std::string> increment(const std::string& word);
    // return index of word (-1 if does not exist)
    long lookup(const std::string& word) const;
    // return word at index (raise exception if does not exist)
    std::string reverse_lookup(long word_idx) const;
    // return count at word index
    size_t count(long word_idx) const;
    // return counts of all word indices
    std::vector<size_t> counts() const;
    // return ordered (descending) counts of all word indices
    std::vector<size_t> ordered_counts() const;
    // return number of word types present in language model
    size_t size() const;
    // return total number of word tokens seen by language model
    size_t total() const;
    // return true if word should be kept after subsampling
    // (return true with probability
    // sqrt(subsample_threshold / f(word_idx)) where f(word_idx) is the
    // normalized frequency corresponding to word_idx)
    bool subsample(long word_idx) const;
    void truncate(size_t max_size);
    // sort language model words by count (descending)
    void sort();
    virtual ~NaiveLanguageModel() { }

    bool equals(const NaiveLanguageModel& other) const;
    void serialize(std::ostream& stream) const;
    static NaiveLanguageModel deserialize(std::istream& stream);

    NaiveLanguageModel(float subsample_threshold,
                  size_t size,
                  size_t total,
                  std::vector<size_t>&& counters,
                  std::unordered_map<std::string,long>&& word_ids,
                  std::vector<std::string>&& words):
        _subsample_threshold(subsample_threshold),
        _size(size),
        _total(total),
        _counters(std::move(counters)),
        _word_ids(std::move(word_ids)),
        _words(std::move(words)) { }
    NaiveLanguageModel(NaiveLanguageModel&& other) = default;
    NaiveLanguageModel(const NaiveLanguageModel& other) = default;
};


// Language model implemented on SpaceSaving approximate counter.

class SpaceSavingLanguageModel {
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
    std::pair<long,std::string> increment(const std::string& word);
    // return index of word (-1 if does not exist)
    long lookup(const std::string& word) const;
    // return word at index (raise exception if does not exist)
    std::string reverse_lookup(long ext_word_idx) const;
    // return count at word index
    size_t count(long ext_word_idx) const;
    // return counts of all word indices
    std::vector<size_t> counts() const;
    // return ordered (descending) counts of all word indices
    std::vector<size_t> ordered_counts() const;
    // return number of word types present in language model
    size_t size() const;
    // return number of word types possible language model
    size_t capacity() const;
    // return total number of word tokens seen by language model
    size_t total() const;
    // return true if word should be kept after subsampling
    // (return true with probability
    // sqrt(subsample_threshold / f(word_idx)) where f(word_idx) is the
    // normalized frequency corresponding to word_idx)
    bool subsample(long ext_word_idx) const;
    void truncate(size_t max_size);
    virtual ~SpaceSavingLanguageModel() { }

    bool equals(const SpaceSavingLanguageModel& other) const;
    void serialize(std::ostream& stream) const;
    static SpaceSavingLanguageModel deserialize(std::istream& stream);

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
        _counters(std::move(counters)),
        _word_ids(std::move(word_ids)),
        _internal_ids(std::move(internal_ids)),
        _external_ids(std::move(external_ids)),
        _words(std::move(words)) { }
    SpaceSavingLanguageModel(SpaceSavingLanguageModel&& other) = default;
    SpaceSavingLanguageModel(const SpaceSavingLanguageModel& other) = default;

  private:
    void _update_min_idx();
    std::pair<long,std::string> _unfull_append(const std::string& word);
    std::pair<long,std::string> _full_replace(const std::string& word);
    std::pair<long,std::string> _full_increment(long ext_idx);
};


// Word-context matrix factorization model.

class WordContextFactorization {
  size_t _vocab_dim, _embedding_dim, _actual_embedding_dim;
  AlignedVector _word_embeddings, _context_embeddings;

  public:
    WordContextFactorization(size_t vocab_dim = DEFAULT_VOCAB_DIM,
                             size_t embedding_dim = DEFAULT_EMBEDDING_DIM);
    size_t get_embedding_dim() const;
    size_t get_vocab_dim() const;
    float* get_word_embedding(size_t word_idx);
    float* get_context_embedding(size_t word_idx);
    virtual ~WordContextFactorization() { }

    bool equals(const WordContextFactorization& other) const;
    void serialize(std::ostream& stream) const;
    static WordContextFactorization deserialize(std::istream& stream);

    WordContextFactorization(size_t vocab_dim,
                             size_t embedding_dim,
                             size_t actual_embedding_dim,
                             AlignedVector&& word_embeddings,
                             AlignedVector&& context_embeddings):
        _vocab_dim(vocab_dim),
        _embedding_dim(embedding_dim),
        _actual_embedding_dim(actual_embedding_dim),
        _word_embeddings(std::move(word_embeddings)),
        _context_embeddings(std::move(context_embeddings)) { }
    WordContextFactorization(WordContextFactorization&& other) = default;
    WordContextFactorization(const WordContextFactorization& other) = default;
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
    void step(size_t dim);
    float get_rho(size_t dim) const;
    void gradient_update(size_t dim, size_t n, const float *g,
                                 float *x);
    void scaled_gradient_update(size_t dim, size_t n, const float *g,
                                        float *x, float alpha);
    void reset(size_t dim);
    virtual ~SGD() { }

    bool equals(const SGD& other) const;
    void serialize(std::ostream& stream) const;
    static SGD deserialize(std::istream& stream);

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
          _rho(std::move(rho)),
          _t(std::move(t)) { }
    SGD(SGD&& other) = default;
    SGD(const SGD& other) = default;

  private:
    void _compute_rho(size_t dimension);
};


// Uniform sampling strategy for language model.

template <class LanguageModel>
class UniformSamplingStrategy;

template <class LanguageModel>
class UniformSamplingStrategy {
  public:
    UniformSamplingStrategy() { }
    // sample from uniform distribution
    long sample_idx(const LanguageModel& language_model);
    void
      step(const LanguageModel& language_model, size_t word_idx) { }

    UniformSamplingStrategy(UniformSamplingStrategy&& other) = default;
    UniformSamplingStrategy(const UniformSamplingStrategy& other) = default;

    virtual ~UniformSamplingStrategy() { }

    bool equals(const UniformSamplingStrategy& other) const { return true; }
    void serialize(std::ostream& stream) const { }
    static UniformSamplingStrategy<LanguageModel> deserialize(std::istream& stream) {
      return UniformSamplingStrategy<LanguageModel>();
    }
};


// Empirical sampling strategy for language model.

template <class LanguageModel, class CountNormalizer = ExponentCountNormalizer>
class EmpiricalSamplingStrategy;

template <class LanguageModel, class CountNormalizer>
class EmpiricalSamplingStrategy {
  size_t _refresh_interval;
  size_t _refresh_burn_in;

  public:
    CountNormalizer normalizer;
    AliasSampler alias_sampler;

  private:
    size_t _t;
    bool _initialized;

  public:
    EmpiricalSamplingStrategy(CountNormalizer&& normalizer_,
                              size_t
                                refresh_interval = DEFAULT_REFRESH_INTERVAL,
                              size_t
                                refresh_burn_in = DEFAULT_REFRESH_BURN_IN);
    // if we have taken no more than refresh_burn_in steps
    // or the number of steps since then is a multiple of
    // refresh_interval, refresh (recompute) distribution
    // based on current counts
    void
      step(const LanguageModel& language_model, size_t word_idx);
    long sample_idx(const LanguageModel& language_model);
    virtual ~EmpiricalSamplingStrategy() { }

    bool equals(const EmpiricalSamplingStrategy& other) const;
    void serialize(std::ostream& stream) const;
    static EmpiricalSamplingStrategy deserialize(std::istream& stream);

    EmpiricalSamplingStrategy(size_t refresh_interval,
                              size_t refresh_burn_in,
                              CountNormalizer&& normalizer_,
                              AliasSampler&& alias_sampler_,
                              size_t t,
                              bool initialized):
        _refresh_interval(refresh_interval),
        _refresh_burn_in(refresh_burn_in),
        normalizer(std::move(normalizer_)),
        alias_sampler(std::move(alias_sampler_)),
        _t(t),
        _initialized(initialized) { }
    EmpiricalSamplingStrategy(EmpiricalSamplingStrategy&& other) = default;
    EmpiricalSamplingStrategy(const EmpiricalSamplingStrategy& other) = default;
};


// Reservoir sampling strategy for language model.

template <class LanguageModel, class ReservoirSamplerType = ReservoirSampler<long> >
class ReservoirSamplingStrategy;

template <class LanguageModel, class ReservoirSamplerType>
class ReservoirSamplingStrategy {
  public:
    ReservoirSamplerType reservoir_sampler;

    ReservoirSamplingStrategy(
      ReservoirSamplerType&& reservoir_sampler_):
        reservoir_sampler(std::move(reservoir_sampler_)) { }
    // (randomly) add word to reservoir
    void
      step(const LanguageModel& language_model, size_t word_idx) {
        reservoir_sampler.insert(word_idx);
      }
    long sample_idx(const LanguageModel& language_model) {
      return reservoir_sampler.sample();
    }
    virtual ~ReservoirSamplingStrategy() { }

    bool equals(const ReservoirSamplingStrategy& other) const;
    void serialize(std::ostream& stream) const;
    static ReservoirSamplingStrategy deserialize(std::istream& stream);

    ReservoirSamplingStrategy(ReservoirSamplingStrategy&& other) = default;
    ReservoirSamplingStrategy(const ReservoirSamplingStrategy& other) = default;
};


// Fixed discrete sampling strategy for language model.

template <class LanguageModel, class DiscretizationType = Discretization>
class DiscreteSamplingStrategy;

template <class LanguageModel, class DiscretizationType>
class DiscreteSamplingStrategy {
  public:
    DiscretizationType discretization;

    DiscreteSamplingStrategy(DiscretizationType&& discretization_):
      discretization(std::move(discretization_)) { }
    void step(const LanguageModel& language_model, size_t word_idx) { }
    long sample_idx(const LanguageModel& language_model) {
      return discretization.sample();
    }

    DiscreteSamplingStrategy(DiscreteSamplingStrategy&& other) = default;
    DiscreteSamplingStrategy(const DiscreteSamplingStrategy& other) = default;

    virtual ~DiscreteSamplingStrategy() { }

    bool equals(const DiscreteSamplingStrategy& other) const;
    void serialize(std::ostream& stream) const;
    static DiscreteSamplingStrategy<LanguageModel, DiscretizationType> deserialize(std::istream& stream);
};


// Static context strategy

class StaticContextStrategy {
  size_t _symm_context;

  public:
    StaticContextStrategy(size_t symm_context):
      _symm_context(symm_context) { }
    // return static (fixed) thresholded context
    // return number of words in left and right context (respectively)
    // given there are at most avail_left and avail_right words to the
    // left and right (respectively); return pair (0,0) if no context
    std::pair<size_t,size_t> size(size_t avail_left,
                                  size_t avail_right) const;
    virtual ~StaticContextStrategy() { }
    bool equals(const StaticContextStrategy& other) const;
    void serialize(std::ostream& stream) const;
    static StaticContextStrategy deserialize(std::istream& stream);
    StaticContextStrategy(StaticContextStrategy&& other) = default;
    StaticContextStrategy(const StaticContextStrategy& other) = default;
};


// Dynamic context strategy

class DynamicContextStrategy {
  size_t _symm_context;

  public:
    DynamicContextStrategy(size_t symm_context):
      _symm_context(symm_context) { }
    // return dynamic (sampled) thresholded context
    // return number of words in left and right context (respectively)
    // given there are at most avail_left and avail_right words to the
    // left and right (respectively); return pair (0,0) if no context
    std::pair<size_t,size_t> size(size_t avail_left,
                                  size_t avail_right) const;
    virtual ~DynamicContextStrategy() { }
    bool equals(const DynamicContextStrategy& other) const;
    void serialize(std::ostream& stream) const;
    static DynamicContextStrategy deserialize(std::istream& stream);

    DynamicContextStrategy(DynamicContextStrategy&& other) = default;
    DynamicContextStrategy(const DynamicContextStrategy& other) = default;
};


//
// UniformSamplingStrategy
//


template <class LanguageModel>
long UniformSamplingStrategy<LanguageModel>::sample_idx(const LanguageModel& language_model) {
  std::uniform_int_distribution<long> d(0, language_model.size() - 1);
  return d(get_urng());
}


//
// EmpiricalSamplingStrategy
//


template <class LanguageModel, class CountNormalizer>
EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::EmpiricalSamplingStrategy(
    CountNormalizer&& normalizer_,
    size_t refresh_interval,
    size_t refresh_burn_in):
  _refresh_interval(refresh_interval),
  _refresh_burn_in(refresh_burn_in),
  normalizer(std::move(normalizer_)),
  alias_sampler(std::vector<float>()),
  _t(0),
  _initialized(false) {
}

template <class LanguageModel, class CountNormalizer>
long EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::sample_idx(
    const LanguageModel& language_model) {
  if (! _initialized) {
    alias_sampler = AliasSampler(
      normalizer.normalize(language_model.counts())
    );
    _initialized = true;
  }
  return alias_sampler.sample();
}

template <class LanguageModel, class CountNormalizer>
void EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::step(
    const LanguageModel& language_model, size_t word_idx) {
  ++_t;
  if ((! _initialized) ||
      (_refresh_interval > 0 &&
       (_t < _refresh_burn_in ||
        (_t - _refresh_burn_in) % _refresh_interval == 0))) {
    alias_sampler = AliasSampler(
      normalizer.normalize(language_model.counts())
    );
    _initialized = true;
  }
}

template <class LanguageModel, class CountNormalizer>
void EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::serialize(std::ostream& stream) const {
  Serializer<size_t>::serialize(_refresh_interval, stream);
  Serializer<size_t>::serialize(_refresh_burn_in, stream);
  Serializer<CountNormalizer>::serialize(normalizer, stream);
  Serializer<AliasSampler>::serialize(alias_sampler, stream);
  Serializer<size_t>::serialize(_t, stream);
  Serializer<bool>::serialize(_initialized, stream);
}

template <class LanguageModel, class CountNormalizer>
EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>
    EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::deserialize(std::istream& stream) {
  auto refresh_interval(Serializer<size_t>::deserialize(stream));
  auto refresh_burn_in(Serializer<size_t>::deserialize(stream));
  auto normalizer_(Serializer<CountNormalizer>::deserialize(stream));
  auto alias_sampler_(Serializer<AliasSampler>::deserialize(stream));
  auto t(Serializer<size_t>::deserialize(stream));
  auto initialized(Serializer<bool>::deserialize(stream));
  return EmpiricalSamplingStrategy(
    refresh_interval,
    refresh_burn_in,
    std::move(normalizer_),
    std::move(alias_sampler_),
    t,
    initialized
  );
}

template <class LanguageModel, class CountNormalizer>
bool EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>::equals(const EmpiricalSamplingStrategy<LanguageModel, CountNormalizer>& other) const {
  return
    _refresh_interval == other._refresh_interval &&
    _refresh_burn_in == other._refresh_burn_in &&
    normalizer.equals(other.normalizer) &&
    alias_sampler.equals(other.alias_sampler) &&
    _t == other._t &&
    _initialized == other._initialized;
}


//
// ReservoirSamplingStrategy
//


template <class LanguageModel, class ReservoirSamplerType>
void ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>::serialize(std::ostream& stream) const {
  Serializer<ReservoirSamplerType>::serialize(reservoir_sampler, stream);
}

template <class LanguageModel, class ReservoirSamplerType>
ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>
    ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>::deserialize(std::istream& stream) {
  auto reservoir_sampler(Serializer<ReservoirSamplerType>::deserialize(stream));
  return ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>(
    std::move(reservoir_sampler)
  );
}

template <class LanguageModel, class ReservoirSamplerType>
bool ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>::equals(
    const ReservoirSamplingStrategy<LanguageModel, ReservoirSamplerType>& other) const {
  return reservoir_sampler.equals(other.reservoir_sampler);
}


//
// DiscreteSamplingStrategy
//


template <class LanguageModel, class DiscretizationType>
void DiscreteSamplingStrategy<LanguageModel, DiscretizationType>::serialize(std::ostream& stream) const {
  Serializer<DiscretizationType>::serialize(discretization, stream);
}

template <class LanguageModel, class DiscretizationType>
DiscreteSamplingStrategy<LanguageModel, DiscretizationType>
    DiscreteSamplingStrategy<LanguageModel, DiscretizationType>::deserialize(std::istream& stream) {
  auto discretization(Serializer<DiscretizationType>::deserialize(stream));
  return DiscreteSamplingStrategy<LanguageModel, DiscretizationType>(
    std::move(discretization)
  );
}

template <class LanguageModel, class DiscretizationType>
bool DiscreteSamplingStrategy<LanguageModel, DiscretizationType>::equals(
    const DiscreteSamplingStrategy<LanguageModel, DiscretizationType>& other) const {
  return discretization.equals(other.discretization);
}


#endif
