#include "_core.h"
#include "_math.h"
#include "_serialization.h"
#include "_cblas.h"
#include "_log.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <utility>
#include <exception>


using namespace std;


//
// LanguageModel
//


shared_ptr<LanguageModel> LanguageModel::deserialize(istream& stream) {
  auto derived(*Serializer<int>::deserialize(stream));
  switch (derived) {
    case naive_lm:
      return NaiveLanguageModel::deserialize(stream);
    case space_saving_lm:
      return SpaceSavingLanguageModel::deserialize(stream);
    default:
      throw runtime_error(
        string("LanguageModel::deserialize: invalid derived type ") +
        to_string(derived));
  }
}

bool LanguageModel::equals(const LanguageModel& other) const {
  return true;
}


//
// NaiveLanguageModel
//


NaiveLanguageModel::NaiveLanguageModel(float subsample_threshold):
    LanguageModel(),
    _subsample_threshold(subsample_threshold),
    _size(0),
    _total(0),
    _counters(),
    _word_ids(),
    _words() { }

pair<long,string> NaiveLanguageModel::increment(const string& word) {
  long idx = lookup(word);
  if (idx < 0) {
    // word not in language model
    idx = (long) _size;
    _word_ids[word] = idx;
    _words.push_back(word);
    _counters.push_back(1);
    ++_size;
    ++_total;
  } else {
    // word in language model: increment counter
    ++_counters[idx];
    ++_total;
  }
  return make_pair(-1L, string());
}

long NaiveLanguageModel::lookup(const string& word) const {
  auto it = _word_ids.find(word);
  return (it == _word_ids.end()) ? -1 : it->second;
}

string NaiveLanguageModel::reverse_lookup(long word_idx) const {
  return _words.at(word_idx);
}

size_t NaiveLanguageModel::count(long word_idx) const {
  return _counters[word_idx];
}

vector<size_t> NaiveLanguageModel::counts() const {
  return _counters;
}

vector<size_t> NaiveLanguageModel::ordered_counts() const {
  vector<size_t> c(_counters);
  ::sort(c.begin(), c.end());
  reverse(c.begin(), c.end());
  return c;
}

size_t NaiveLanguageModel::size() const {
  return _size;
}

size_t NaiveLanguageModel::total() const {
  return _total;
}

bool NaiveLanguageModel::subsample(long word_idx) const {
  const float normalized_freq = count(word_idx) / (float) total();
  uniform_real_distribution<float> d;
  const float random_unif = d(get_urng());
  return random_unif > 1 - sqrt(_subsample_threshold / normalized_freq);
}

void NaiveLanguageModel::truncate(size_t max_size) {
  vector<pair<string,size_t> > _sorted_words(_size);
  for (size_t i = 0; i < _size; ++i) {
    _sorted_words[i] = make_pair(_words[i], _counters[i]);
  }
  ::sort(_sorted_words.rbegin(), _sorted_words.rend(),
         pair_second_cmp<string,size_t>);

  _size = min(_size, max_size);

  _sorted_words.resize(_size);

  _word_ids.clear();
  _word_ids.reserve(_size);
  _words.clear();
  _words.reserve(_size);
  _counters.clear();
  _counters.reserve(_size);
  _total = 0;

  size_t i = 0;
  for (auto it = _sorted_words.begin(); it != _sorted_words.end(); ++it, ++i) {
    _word_ids[it->first] = i;
    _words.push_back(it->first);
    _counters.push_back(it->second);
    _total += it->second;
  }
}

void NaiveLanguageModel::sort() {
  truncate(_size);
}

void NaiveLanguageModel::serialize(ostream& stream) const {
  Serializer<int>::serialize(naive_lm, stream);
  Serializer<float>::serialize(_subsample_threshold, stream);
  Serializer<size_t>::serialize(_size, stream);
  Serializer<size_t>::serialize(_total, stream);
  Serializer<vector<size_t> >::serialize(_counters, stream);
  Serializer<unordered_map<string,long> >::serialize(_word_ids, stream);
  Serializer<vector<string> >::serialize(_words, stream);
}

shared_ptr<NaiveLanguageModel> NaiveLanguageModel::deserialize(istream& stream) {
  auto subsample_threshold(*Serializer<float>::deserialize(stream));
  auto size(*Serializer<size_t>::deserialize(stream));
  auto total(*Serializer<size_t>::deserialize(stream));
  auto counters(Serializer<vector<size_t> >::deserialize(stream));
  auto word_ids(Serializer<unordered_map<string,long> >::deserialize(stream));
  auto words(Serializer<vector<string> >::deserialize(stream));
  return make_shared<NaiveLanguageModel>(
    subsample_threshold,
    size,
    total,
    move(*counters),
    move(*word_ids),
    move(*words)
  );
}

bool NaiveLanguageModel::equals(const LanguageModel& other) const {
  const auto& cast_other(
      dynamic_cast<const NaiveLanguageModel&>(other));
  return
    near(_subsample_threshold, cast_other._subsample_threshold) &&
    _size == cast_other._size &&
    _total == cast_other._total &&
    _counters == cast_other._counters &&
    _word_ids == cast_other._word_ids &&
    _words == cast_other._words;
}


//
// SpaceSavingLanguageModel
//


SpaceSavingLanguageModel::SpaceSavingLanguageModel(size_t num_counters,
                                                   float subsample_threshold):
    LanguageModel(),
    _subsample_threshold(subsample_threshold),
    _num_counters(num_counters),
    _size(0),
    _total(0),
    _min_idx(0),
    _counters(),
    _word_ids(),
    _internal_ids(),
    _external_ids(),
    _words(num_counters, string()) {
  _counters.reserve(_num_counters);
  _internal_ids.reserve(_num_counters);
  _external_ids.reserve(_num_counters);
}

pair<long,string> SpaceSavingLanguageModel::increment(const string& word) {
  ++_total;

  long ext_idx = lookup(word);
  if (ext_idx < 0) {
    // word not in language model
    if (_size < _num_counters) {
      // at least one unused counter: insert this word and increment
      return _unfull_append(word);
    } else {
      // all counters in use: eject word, insert this word, and increment
      return _full_replace(word);
    }
  } else {
    // word in language model: increment counter
    return _full_increment(ext_idx);
  }
}

long SpaceSavingLanguageModel::lookup(const string& word) const {
  auto it = _word_ids.find(word);
  return (it == _word_ids.end()) ? -1 : _external_ids[it->second];
}

string SpaceSavingLanguageModel::reverse_lookup(long ext_word_idx) const {
  return _words.at(_internal_ids[ext_word_idx]);
}

size_t SpaceSavingLanguageModel::count(long ext_word_idx) const {
  return _counters[_internal_ids[ext_word_idx]];
}

vector<size_t> SpaceSavingLanguageModel::counts() const {
  vector<size_t> c(_size, 0);
  for (size_t int_idx = 0; int_idx < _size; ++int_idx) {
    c[_external_ids[int_idx]] = _counters[int_idx];
  }
  return c;
}

vector<size_t> SpaceSavingLanguageModel::ordered_counts() const {
  return _counters;
}

size_t SpaceSavingLanguageModel::size() const {
  return _size;
}

size_t SpaceSavingLanguageModel::capacity() const {
  return _num_counters;
}

size_t SpaceSavingLanguageModel::total() const {
  return _total;
}

bool SpaceSavingLanguageModel::subsample(long ext_word_idx) const {
  const float normalized_freq = count(ext_word_idx) / (float) total();
  uniform_real_distribution<float> d;
  const float random_unif = d(get_urng());
  return random_unif > 1 - sqrt(_subsample_threshold / normalized_freq);
}

void SpaceSavingLanguageModel::truncate(size_t max_size) {
  throw logic_error(
    string("SpaceSavingLanguageModel::truncate: not implemented"));
}

void SpaceSavingLanguageModel::serialize(ostream& stream) const {
  Serializer<int>::serialize(space_saving_lm, stream);
  Serializer<float>::serialize(_subsample_threshold, stream);
  Serializer<size_t>::serialize(_num_counters, stream);
  Serializer<size_t>::serialize(_size, stream);
  Serializer<size_t>::serialize(_total, stream);
  Serializer<size_t>::serialize(_min_idx, stream);
  Serializer<vector<size_t> >::serialize(_counters, stream);
  Serializer<unordered_map<string,long> >::serialize(_word_ids, stream);
  Serializer<vector<long> >::serialize(_internal_ids, stream);
  Serializer<vector<long> >::serialize(_external_ids, stream);
  Serializer<vector<string> >::serialize(_words, stream);
}

shared_ptr<SpaceSavingLanguageModel>
    SpaceSavingLanguageModel::deserialize(istream& stream) {
  auto subsample_threshold(*Serializer<float>::deserialize(stream));
  auto num_counters(*Serializer<size_t>::deserialize(stream));
  auto size(*Serializer<size_t>::deserialize(stream));
  auto total(*Serializer<size_t>::deserialize(stream));
  auto min_idx(*Serializer<size_t>::deserialize(stream));
  auto counters(Serializer<vector<size_t> >::deserialize(stream));
  auto word_ids(Serializer<unordered_map<string,long> >::deserialize(stream));
  auto internal_ids(Serializer<vector<long> >::deserialize(stream));
  auto external_ids(Serializer<vector<long> >::deserialize(stream));
  auto words(Serializer<vector<string> >::deserialize(stream));
  return make_shared<SpaceSavingLanguageModel>(
    subsample_threshold,
    num_counters,
    size,
    total,
    min_idx,
    move(*counters),
    move(*word_ids),
    move(*internal_ids),
    move(*external_ids),
    move(*words)
  );
}

bool SpaceSavingLanguageModel::equals(const LanguageModel& other) const {
  const auto& cast_other(
      dynamic_cast<const SpaceSavingLanguageModel&>(other));
  return
    near(_subsample_threshold, cast_other._subsample_threshold) &&
    _num_counters == cast_other._num_counters &&
    _size == cast_other._size &&
    _total == cast_other._total &&
    _min_idx == cast_other._min_idx &&
    _counters == cast_other._counters &&
    _word_ids == cast_other._word_ids &&
    _internal_ids == cast_other._internal_ids &&
    _external_ids == cast_other._external_ids &&
    _words == cast_other._words;
}

void SpaceSavingLanguageModel::_update_min_idx() {
  if (_min_idx + 1 == _size) {
    const size_t min_count = _counters[_min_idx];
    while (_min_idx > 0 && _counters[_min_idx - 1] == min_count) {
      --_min_idx;
    }
  } else {
    ++_min_idx;
  }
}

pair<long,string> SpaceSavingLanguageModel::_unfull_append(const string& word) {
  // int_idx == ext_idx
  const long ext_idx = (long) _size;
  _word_ids[word] = ext_idx;
  _internal_ids.push_back(ext_idx);
  _external_ids.push_back(ext_idx);
  _words[ext_idx] = word;
  ++_size;
  _counters.push_back(1);
  if (ext_idx == 0 || _counters[_min_idx] > 1) {
    _min_idx = ext_idx;
  }
  return make_pair(-1L, string());
}

pair<long,string> SpaceSavingLanguageModel::_full_replace(const string& word) {
  // int_idx == _min_idx
  string ejectee_word(_words[_min_idx]);
  const long ext_idx = _external_ids[_min_idx];
  _word_ids.erase(ejectee_word);
  _word_ids[word] = _min_idx;
  _words[_min_idx] = word;
  ++_counters[_min_idx];
  _update_min_idx();
  return make_pair(ext_idx, ejectee_word);
}

pair<long,string> SpaceSavingLanguageModel::_full_increment(long ext_idx) {
  long int_idx = _internal_ids[ext_idx];
  ++_counters[int_idx];
  if ((size_t) int_idx == _min_idx) {
    _update_min_idx();
  } else {
    if ((size_t) int_idx > _min_idx) {
      ++_min_idx;
    }
    size_t new_int_idx;
    const size_t new_count = _counters[int_idx];
    for (new_int_idx = int_idx;
         new_int_idx > 0 && new_count > _counters[new_int_idx - 1];
         --new_int_idx) { }
    swap(_word_ids.at(_words[int_idx]),
         _word_ids.at(_words[new_int_idx]));
    swap(_counters[int_idx],
         _counters[new_int_idx]);
    swap(_words[int_idx],
         _words[new_int_idx]);
    swap(_internal_ids[_external_ids[int_idx]],
         _internal_ids[_external_ids[new_int_idx]]);
    swap(_external_ids[int_idx],
         _external_ids[new_int_idx]);
  }
  return make_pair(-1L, string());
}


//
// WordContextFactorization
//


WordContextFactorization::WordContextFactorization(size_t vocab_dim,
                                                   size_t embedding_dim):
    _vocab_dim(vocab_dim),
    _embedding_dim(embedding_dim),
    _actual_embedding_dim(embedding_dim),
    _word_embeddings(vocab_dim * embedding_dim),
    _context_embeddings(vocab_dim * embedding_dim) {
  // resize embedding matrices so that each vector is aligned
  if (_actual_embedding_dim % VECTOR_ALIGNMENT != 0) {
    _actual_embedding_dim = INCREASE_TO_MULTIPLE(_actual_embedding_dim, VECTOR_ALIGNMENT);
    _word_embeddings.resize(vocab_dim * _actual_embedding_dim);
    _context_embeddings.resize(vocab_dim * _actual_embedding_dim);
  }
  // initialize embedding matrices randomly
  sample_centered_uniform_vector(vocab_dim * _actual_embedding_dim,
                                 _word_embeddings.data());
  memset(_context_embeddings.data(), 0,
         vocab_dim * _actual_embedding_dim * sizeof(float));
  // we have randomly initialized the padding for vector alignment,
  // reset it to zero
  for (size_t i = 0; i < _vocab_dim; ++i) {
    for (size_t j = _embedding_dim; j < _actual_embedding_dim; ++j) {
      _word_embeddings[j + i * _actual_embedding_dim] = 0;
      _context_embeddings[j + i * _actual_embedding_dim] = 0;
    }
  }
}

size_t WordContextFactorization::get_embedding_dim() {
  return _embedding_dim;
}

size_t WordContextFactorization::get_vocab_dim() {
  return _vocab_dim;
}

float* WordContextFactorization::get_word_embedding(size_t word_idx) {
  return _word_embeddings.data() + word_idx * _actual_embedding_dim;
}

float* WordContextFactorization::get_context_embedding(size_t word_idx) {
  return _context_embeddings.data() + word_idx * _actual_embedding_dim;
}

void WordContextFactorization::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_vocab_dim, stream);
  Serializer<size_t>::serialize(_embedding_dim, stream);
  Serializer<size_t>::serialize(_actual_embedding_dim, stream);
  Serializer<AlignedVector>::serialize(_word_embeddings, stream);
  Serializer<AlignedVector>::serialize(_context_embeddings, stream);
}

shared_ptr<WordContextFactorization>
    WordContextFactorization::deserialize(istream& stream) {
  auto vocab_dim(*Serializer<size_t>::deserialize(stream));
  auto embedding_dim(*Serializer<size_t>::deserialize(stream));
  auto actual_embedding_dim(*Serializer<size_t>::deserialize(stream));
  auto word_embeddings(Serializer<AlignedVector>::deserialize(stream));
  auto context_embeddings(Serializer<AlignedVector>::deserialize(stream));
  return make_shared<WordContextFactorization>(
    vocab_dim,
    embedding_dim,
    actual_embedding_dim,
    move(*word_embeddings),
    move(*context_embeddings)
  );
}

bool WordContextFactorization::equals(const WordContextFactorization& other) const {
  return
    _vocab_dim == other._vocab_dim &&
    _embedding_dim == other._embedding_dim &&
    _actual_embedding_dim == other._actual_embedding_dim &&
    near(_word_embeddings, other._word_embeddings) &&
    near(_context_embeddings, other._context_embeddings);
}


//
// SGD
//


SGD::SGD(size_t dimension, float tau, float kappa, float rho_lower_bound):
    _dimension(dimension),
    _tau(tau),
    _kappa(kappa),
    _rho_lower_bound(rho_lower_bound),
    _t(dimension, 0) {
  _rho.resize(dimension);
  for (size_t dim = 0; dim < dimension; ++dim) {
    _compute_rho(dim);
  }
}

void SGD::step(size_t dim) {
  ++_t[dim];
  _compute_rho(dim);
}

float SGD::get_rho(size_t dim) const {
  return _rho[dim];
}

void SGD::gradient_update(size_t dim, size_t n, const float *g, float *x) {
  scaled_gradient_update(dim, n, g, x, 1);
}

void SGD::scaled_gradient_update(size_t dim, size_t n, const float *g, float *x,
                                 float alpha) {
  cblas_saxpy(n, _rho[dim] * alpha, g, 1, x, 1);
}

void SGD::reset(size_t dim) {
  _t[dim] = 0;
  _compute_rho(dim);
}

void SGD::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_dimension, stream);
  Serializer<float>::serialize(_tau, stream);
  Serializer<float>::serialize(_kappa, stream);
  Serializer<float>::serialize(_rho_lower_bound, stream);
  Serializer<vector<float> >::serialize(_rho, stream);
  Serializer<vector<size_t> >::serialize(_t, stream);
}

shared_ptr<SGD> SGD::deserialize(istream& stream) {
  auto dimension(*Serializer<size_t>::deserialize(stream));
  auto tau(*Serializer<float>::deserialize(stream));
  auto kappa(*Serializer<float>::deserialize(stream));
  auto rho_lower_bound(*Serializer<float>::deserialize(stream));
  auto rho(Serializer<vector<float> >::deserialize(stream));
  auto t(Serializer<vector<size_t> >::deserialize(stream));
  return make_shared<SGD>(
    dimension,
    tau,
    kappa,
    rho_lower_bound,
    move(*rho),
    move(*t)
  );
}

bool SGD::equals(const SGD& other) const {
  return
    _dimension == other._dimension &&
    near(_tau, other._tau) &&
    near(_kappa, other._kappa) &&
    near(_rho_lower_bound, other._rho_lower_bound) &&
    near(_rho, other._rho) &&
    _t == other._t;
}

void SGD::_compute_rho(size_t dim) {
  _rho[dim] = max(_rho_lower_bound, _kappa * (1.f - (float) _t[dim] / _tau));
}


//
// SamplingStrategy
//


shared_ptr<SamplingStrategy> SamplingStrategy::deserialize(istream& stream) {
  auto derived(*Serializer<int>::deserialize(stream));
  switch (derived) {
    case uniform:
      return UniformSamplingStrategy::deserialize(stream);
    case empirical:
      return EmpiricalSamplingStrategy::deserialize(stream);
    case reservoir:
      return ReservoirSamplingStrategy::deserialize(stream);
    default:
      throw runtime_error(
        string("SamplingStrategy::deserialize: invalid derived type ") +
        to_string(derived));
  }
}


//
// UniformSamplingStrategy
//


void UniformSamplingStrategy::serialize(ostream& stream) const {
  Serializer<int>::serialize(uniform, stream);
}

long UniformSamplingStrategy::sample_idx(const LanguageModel& language_model) {
  uniform_int_distribution<long> d(0, language_model.size() - 1);
  return d(get_urng());
}


//
// EmpiricalSamplingStrategy
//


EmpiricalSamplingStrategy::EmpiricalSamplingStrategy(
  shared_ptr<CountNormalizer> normalizer,
  size_t refresh_interval,
  size_t refresh_burn_in):
    SamplingStrategy(),
    _refresh_interval(refresh_interval),
    _refresh_burn_in(refresh_burn_in),
    _normalizer(normalizer),
    _alias_sampler(make_shared<AliasSampler>(vector<float>())),
    _t(0),
    _initialized(false) {
}

long EmpiricalSamplingStrategy::sample_idx(
    const LanguageModel& language_model) {
  if (! _initialized) {
    _alias_sampler = make_shared<AliasSampler>(
      _normalizer->normalize(language_model.counts())
    );
    _initialized = true;
  }
  return _alias_sampler->sample();
}

void EmpiricalSamplingStrategy::step(
    const LanguageModel& language_model, size_t word_idx) {
  ++_t;
  if ((! _initialized) ||
      _t < _refresh_burn_in ||
      (_t - _refresh_burn_in) % _refresh_interval == 0) {
    _alias_sampler = make_shared<AliasSampler>(
      _normalizer->normalize(language_model.counts())
    );
    _initialized = true;
  }
}

void EmpiricalSamplingStrategy::reset(const LanguageModel& language_model,
                                      const CountNormalizer& normalizer) {
  _alias_sampler = make_shared<AliasSampler>(
    normalizer.normalize(language_model.counts())
  );
  _initialized = true;
}

void EmpiricalSamplingStrategy::serialize(ostream& stream) const {
  Serializer<int>::serialize(empirical, stream);
  Serializer<size_t>::serialize(_refresh_interval, stream);
  Serializer<size_t>::serialize(_refresh_burn_in, stream);
  Serializer<CountNormalizer>::serialize(*_normalizer, stream);
  Serializer<AliasSampler>::serialize(*_alias_sampler, stream);
  Serializer<size_t>::serialize(_t, stream);
  Serializer<bool>::serialize(_initialized, stream);
}

shared_ptr<EmpiricalSamplingStrategy>
    EmpiricalSamplingStrategy::deserialize(istream& stream) {
  auto refresh_interval(*Serializer<size_t>::deserialize(stream));
  auto refresh_burn_in(*Serializer<size_t>::deserialize(stream));
  auto normalizer(Serializer<CountNormalizer>::deserialize(stream));
  auto alias_sampler(Serializer<AliasSampler>::deserialize(stream));
  auto t(*Serializer<size_t>::deserialize(stream));
  auto initialized(*Serializer<bool>::deserialize(stream));
  return make_shared<EmpiricalSamplingStrategy>(
    refresh_interval,
    refresh_burn_in,
    normalizer,
    alias_sampler,
    t,
    initialized
  );
}

bool EmpiricalSamplingStrategy::equals(const SamplingStrategy& other) const {
  const auto& cast_other(
      dynamic_cast<const EmpiricalSamplingStrategy&>(other));
  return
    _refresh_interval == cast_other._refresh_interval &&
    _refresh_burn_in == cast_other._refresh_burn_in &&
    _normalizer->equals(*(cast_other._normalizer)) &&
    _alias_sampler->equals(*(cast_other._alias_sampler)) &&
    _t == cast_other._t &&
    _initialized == cast_other._initialized;
}


//
// ReservoirSamplingStrategy
//


void ReservoirSamplingStrategy::reset(const LanguageModel& language_model,
                                      const CountNormalizer& normalizer) {
  const vector<float> weights(normalizer.normalize(language_model.counts()));
  discrete_distribution<size_t> d(weights.begin(), weights.end());
  _reservoir_sampler->clear();
  for (size_t i = 0; i < _reservoir_sampler->size(); ++i) {
    step(language_model, d(get_urng()));
  }
}

void ReservoirSamplingStrategy::serialize(ostream& stream) const {
  Serializer<int>::serialize(reservoir, stream);
  Serializer<ReservoirSampler<long> >::serialize(*_reservoir_sampler, stream);
}

shared_ptr<ReservoirSamplingStrategy>
    ReservoirSamplingStrategy::deserialize(istream& stream) {
  auto reservoir_sampler(Serializer<ReservoirSampler<long> >::deserialize(stream));
  return make_shared<ReservoirSamplingStrategy>(reservoir_sampler);
}

bool ReservoirSamplingStrategy::equals(const SamplingStrategy& other) const {
  const auto& cast_other(
      dynamic_cast<const ReservoirSamplingStrategy&>(other));
  return _reservoir_sampler->equals(*(cast_other._reservoir_sampler));
}


//
// ContextStrategy
//


shared_ptr<ContextStrategy> ContextStrategy::deserialize(istream& stream) {
  auto derived(*Serializer<int>::deserialize(stream));
  switch (derived) {
    case static_ctx:
      return StaticContextStrategy::deserialize(stream);
    case dynamic_ctx:
      return DynamicContextStrategy::deserialize(stream);
    default:
      throw runtime_error(
        string("ContextStrategy::deserialize: invalid derived type ") +
        to_string(derived));
  }
}


//
// StaticContextStrategy
//


pair<size_t,size_t> StaticContextStrategy::size(size_t avail_left,
                                               size_t avail_right) const {
  return make_pair(min(avail_left, _symm_context),
                   min(avail_right, _symm_context));
}

void StaticContextStrategy::serialize(ostream& stream) const {
  Serializer<int>::serialize(static_ctx, stream);
  Serializer<size_t>::serialize(_symm_context, stream);
}

shared_ptr<StaticContextStrategy>
    StaticContextStrategy::deserialize(istream& stream) {
  auto symm_context(*Serializer<size_t>::deserialize(stream));
  return make_shared<StaticContextStrategy>(symm_context);
}

bool StaticContextStrategy::equals(const ContextStrategy& other) const {
  const auto& cast_other(
      dynamic_cast<const StaticContextStrategy&>(other));
  return _symm_context == cast_other._symm_context;
}


//
// DynamicContextStrategy
//


pair<size_t,size_t> DynamicContextStrategy::size(size_t avail_left,
                                                 size_t avail_right) const {
  uniform_int_distribution<size_t> d(1, _symm_context);
  const size_t ctx_size = d(get_urng());
  return make_pair(min(avail_left, ctx_size),
                   min(avail_right, ctx_size));
}

void DynamicContextStrategy::serialize(ostream& stream) const {
  Serializer<int>::serialize(dynamic_ctx, stream);
  Serializer<size_t>::serialize(_symm_context, stream);
}

shared_ptr<DynamicContextStrategy>
    DynamicContextStrategy::deserialize(istream& stream) {
  auto symm_context(*Serializer<size_t>::deserialize(stream));
  return make_shared<DynamicContextStrategy>(symm_context);
}

bool DynamicContextStrategy::equals(const ContextStrategy& other) const {
  const auto& cast_other(
      dynamic_cast<const DynamicContextStrategy&>(other));
  return _symm_context == cast_other._symm_context;
}
