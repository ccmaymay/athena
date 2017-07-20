#include "_math.h"
#include "_serialization.h"
#include <cmath>
#include <cstdlib>
#include <new>
#include <climits>
#include <vector>
#include <utility>
#include <random>
#include <unordered_set>
#include <cstring>

#ifdef __APPLE__
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#else
extern "C" {
#include <omp.h>
}
#endif


using namespace std;


static vector<PRNG> prngs;
static vector<float> fast_sigmoid_grid;


//
// AlignedVector
//


AlignedVector::AlignedVector(size_t size): _data(0), _size(0) {
  resize(size);
}

void AlignedVector::resize(size_t size) {
  float* temp = _data;
  const int ret = posix_memalign(
    reinterpret_cast<void**>(&_data),
    VECTOR_ALIGNMENT,
    INCREASE_TO_MULTIPLE(size, VECTOR_ALIGNMENT) * sizeof(float)
  );
  if (ret == 0) {
    if (temp != 0) {
      memcpy(_data, temp, min(size, _size) * sizeof(float));
      free(temp);
    }
    _size = size;
  } else {
    if (temp != 0) {
      free(temp);
    }
    throw bad_alloc();
  }
}

AlignedVector::AlignedVector(AlignedVector&& other):
    _data(other._data),
    _size(other._size) {
  other._data = 0;
}

AlignedVector::AlignedVector(const AlignedVector& other): _data(0), _size(0) {
  resize(other._size);
  memcpy(_data, other._data, other._size * sizeof(float));
}

AlignedVector::~AlignedVector() {
  if (_data != 0) {
    free(_data);
  }
}

bool operator==(const AlignedVector& lhs, const AlignedVector& rhs) {
  return lhs.equals(rhs);
}

bool AlignedVector::equals(const AlignedVector& other) const {
  return _size == other._size &&
    memcmp(
      reinterpret_cast<void*>(_data),
      reinterpret_cast<void*>(other._data),
      _size * sizeof(float)
    ) == 0;
}

void AlignedVector::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_size, stream);
  stream.write(reinterpret_cast<const char*>(_data), _size * sizeof(float));
}

AlignedVector AlignedVector::deserialize(istream& stream) {
  auto size(Serializer<size_t>::deserialize(stream));
  AlignedVector container(size);
  stream.read(reinterpret_cast<char*>(container.data()),
              size * sizeof(float));
  return container;
}


//
// near
//


template <>
bool near(const float& x, const float& y) {
  return fabs(x - y) < FLOAT_NEAR_THRESHOLD;
}

template <>
bool near(const double& x, const double& y) {
  return fabs(x - y) < DOUBLE_NEAR_THRESHOLD;
}

template <>
bool near(const AlignedVector& x, const AlignedVector& y) {
  if (x.size() != y.size()) {
    return false;
  }

  for (size_t i = 0; i < x.size(); ++i) {
    if (! near(x[i], y[i])) {
      return false;
    }
  }

  return true;
}


//
// other
//


float sigmoid(float x) {
  if (x > SIGMOID_ARG_THRESHOLD) {
    return 1.f;
  } else if (x < -SIGMOID_ARG_THRESHOLD) {
    return 0.f;
  } else {
    return 1.f / (1.f + exp(-x));
  }
}


float fast_sigmoid(float x, size_t grid_size) {
  const size_t max_i = grid_size - 1;
  if (fast_sigmoid_grid.size() != grid_size) {
    #pragma omp critical(fast_sigmoid_grid)
    {
      fast_sigmoid_grid.resize(grid_size, 1.f);
      for (size_t i = 0; i < grid_size; ++i) {
        const float i_x =
          SIGMOID_ARG_THRESHOLD * (i / (float) max_i - 0.5f) * 2.f;
        fast_sigmoid_grid[i] = 1.f / (1.f + exp(-i_x));
      }
    }
  }
  const long long x_i = max_i * ((x / SIGMOID_ARG_THRESHOLD) + 1.f) / 2.f;
  if (x_i < 0) {
    return 0.f;
  } else if (x_i >= (long long) grid_size) {
    return 1.f;
  } else {
    return fast_sigmoid_grid[x_i];
  }
}


void seed(unsigned int s) {
  prngs.clear();
  int num_threads = 1;
  #pragma omp parallel default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
  }
  for (int t = 0; t < num_threads; ++t) {
    prngs.push_back(PRNG(s + t));
  }
}

void seed_default() {
  seed(random_device()());
}

PRNG& get_urng() {
  if (prngs.empty()) {
    seed(0);
  }
  return prngs[omp_get_thread_num()];
}


//
// ExponentCountNormalizer
//


ExponentCountNormalizer::ExponentCountNormalizer(float exponent, float offset):
    _exponent(exponent),
    _offset(offset) { }

vector<float> ExponentCountNormalizer::normalize(const vector<size_t>& counts) const {
  vector<float> probabilities(counts.size(), 0);
  float normalizer = 0;
  for (size_t i = 0; i < probabilities.size(); ++i) {
    probabilities[i] = pow(counts[i] + _offset, _exponent);
    normalizer += probabilities[i];
  }
  for (size_t i = 0; i < probabilities.size(); ++i) {
    probabilities[i] /= normalizer;
  }
  return probabilities;
}

void ExponentCountNormalizer::serialize(ostream& stream) const {
  Serializer<float>::serialize(_exponent, stream);
  Serializer<float>::serialize(_offset, stream);
}

ExponentCountNormalizer ExponentCountNormalizer::deserialize(istream& stream) {
  auto exponent(Serializer<float>::deserialize(stream));
  auto offset(Serializer<float>::deserialize(stream));
  return ExponentCountNormalizer(exponent, offset);
}

bool ExponentCountNormalizer::equals(const ExponentCountNormalizer& other) const {
  return
    near(_exponent, other._exponent) &&
    near(_offset, other._offset);
}


//
// NaiveSampler
//


NaiveSampler::NaiveSampler(const vector<float>& probabilities):
    _size(probabilities.size()),
    _probability_table(probabilities.size(), 0.) {
  if (_size > 0) {
    _probability_table[0] = probabilities[0];
  }
  for (size_t i = 1; i < _size; ++i) {
    _probability_table[i] = _probability_table[i - 1] + probabilities[i];
  }
}

size_t NaiveSampler::sample() const {
  uniform_real_distribution<float> d(0, 1);
  const float target_p = d(get_urng());
  size_t low = 0, high = _size - 1;
  while (high > low) {
    const size_t mid = (low + high) / 2;
    if (target_p > _probability_table[mid]) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return high;
}

void NaiveSampler::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_size, stream);
  Serializer<vector<float> >::serialize(_probability_table, stream);
}

NaiveSampler NaiveSampler::deserialize(istream& stream) {
  auto size(Serializer<size_t>::deserialize(stream));
  auto probability_table(Serializer<vector<float> >::deserialize(stream));
  return NaiveSampler(
    size,
    move(probability_table)
  );
}

bool NaiveSampler::equals(const NaiveSampler& other) const {
  return
    _size == other._size &&
    near(_probability_table, other._probability_table);
}


//
// AliasSampler
//


AliasSampler::AliasSampler(const vector<float>& probabilities):
    _size(probabilities.size()),
    _alias_table(probabilities.size(), 0),
    _probability_table(probabilities.size(), 0.) {
  unordered_set<size_t> underfull, overfull;
  for (size_t i = 0; i < _size; ++i) {
    const float mass = _size * probabilities[i];
    _alias_table[i] = i;
    _probability_table[i] = mass;
    if (mass < 1.) {
      underfull.insert(i);
    } else if (mass > 1.) {
      overfull.insert(i);
    }
  }

  while (! (overfull.empty() || underfull.empty())) {
    auto underfull_it = underfull.begin();
    const size_t underfull_idx = *underfull_it;

    auto overfull_it = overfull.begin();
    const size_t overfull_idx = *overfull_it;

    _alias_table[underfull_idx] = overfull_idx;
    _probability_table[overfull_idx] -= 1. - _probability_table[underfull_idx];

    // set underfull_idx to exactly full
    underfull.erase(underfull_it);

    // set overfull_idx to underfull or exactly full, if appropriate
    if (_probability_table[overfull_idx] < 1.) {
      overfull.erase(overfull_it);
      underfull.insert(overfull_idx);
    } else if (_probability_table[overfull_idx] == 1.) {
      overfull.erase(overfull_it);
    }
  }

  // overfull and underfull may contain masses negligibly close to 1
  // due to floating-point error
  for (auto it = overfull.cbegin(); it != overfull.cend(); ++it) {
    _alias_table[*it] = *it;
    _probability_table[*it] = 1.;
  }
  for (auto it = underfull.cbegin(); it != underfull.cend(); ++it) {
    _alias_table[*it] = *it;
    _probability_table[*it] = 1.;
  }
}

size_t AliasSampler::sample() const {
  uniform_int_distribution<size_t> d(0, _size - 1);
  const size_t i = d(get_urng());
  if (_probability_table[i] == 1) {
    return i;
  } else {
    bernoulli_distribution alias_d(_probability_table[i]);
    return (alias_d(get_urng()) ? i : _alias_table[i]);
  }
}

void AliasSampler::serialize(ostream& stream) const {
  Serializer<size_t>::serialize(_size, stream);
  Serializer<vector<size_t> >::serialize(_alias_table, stream);
  Serializer<vector<float> >::serialize(_probability_table, stream);
}

AliasSampler AliasSampler::deserialize(istream& stream) {
  auto size(Serializer<size_t>::deserialize(stream));
  auto alias_table(Serializer<vector<size_t> >::deserialize(stream));
  auto probability_table(Serializer<vector<float> >::deserialize(stream));
  return AliasSampler(
    size,
    move(alias_table),
    move(probability_table)
  );
}

bool AliasSampler::equals(const AliasSampler& other) const {
  return
    _size == other._size &&
    _alias_table == other._alias_table &&
    near(_probability_table, other._probability_table);
}

AliasSampler& AliasSampler::operator=(AliasSampler const & other) {
  _size = other._size;
  _alias_table = other._alias_table;
  _probability_table = other._probability_table;
  return *this;
}

AliasSampler& AliasSampler::operator=(AliasSampler && other) {
  _size = other._size;
  _alias_table = std::move(other._alias_table);
  _probability_table = std::move(other._probability_table);
  return *this;
}


//
// Discretization
//


Discretization::Discretization(const vector<float>& probabilities,
                               size_t num_samples):
    _samples(num_samples, -1) {
  if (! probabilities.empty()) {
    size_t i = 0, j = 0;
    float cum_mass = probabilities[j];

    while (i < num_samples) {
      // add sample before checking bounds: favor weights near beginning
      // of input
      _samples[i] = j;
      ++i;

      if (i / (float) num_samples > cum_mass) {
        ++j;
        if (j == probabilities.size())
          break;

        cum_mass += probabilities[j];
      }
    }

    for (; i < num_samples; ++i)
      _samples[i] = probabilities.size() - 1;
  }
}

void Discretization::serialize(ostream& stream) const {
  Serializer<vector<long> >::serialize(_samples, stream);
}

Discretization Discretization::deserialize(istream& stream) {
  auto samples(Serializer<vector<long> >::deserialize(stream));
  return Discretization(
    move(samples)
  );
}

bool Discretization::equals(const Discretization& other) const {
  return _samples == other._samples;
}
