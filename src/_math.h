#ifndef ATHENA__MATH_H
#define ATHENA__MATH_H


#include "_serialization.h"

#include <cstddef>
#include <cmath>
#include <vector>
#include <utility>
#include <random>
#include <iostream>
#include <memory>


#define PI 3.14159265358979323846

// sigmoid is hard-coded to 1 (or 0) beyond this threshold (two-sided).
#define SIGMOID_ARG_THRESHOLD 11.f

// precision of equals on doubles (vectors of doubles)
#define DOUBLE_NEAR_THRESHOLD 1e-8

// precision of equals on floats (vectors of floats)
#define FLOAT_NEAR_THRESHOLD 1e-4f

#define INCREASE_TO_MULTIPLE(n, k) \
  ((n) + ((n) % (k) == 0 ? 0 : (k) - ((n) % (k))))

#define VECTOR_ALIGNMENT 64

// must be at least two
#define DEFAULT_FAST_SIGMOID_GRID_SIZE 1000


typedef std::linear_congruential_engine<size_t,25214903917ull,11ull,1ull<<48>
        PRNG;


// Wrapper on cache-aligned floating-point vector.

class AlignedVector final {
  float* _data;
  size_t _size;

  public:
    AlignedVector(size_t size);
    size_t size() const { return _size; }
    float* data() { return _data; }
    const float& operator[](size_t i) const { return _data[i]; }
    float& operator[](size_t i) { return _data[i]; }
    void resize(size_t size);
    ~AlignedVector();

    AlignedVector(AlignedVector&& other);
    AlignedVector(const AlignedVector& other);

    bool equals(const AlignedVector& other) const;
    void serialize(std::ostream& stream) const;
    static AlignedVector deserialize(std::istream& stream);
};

bool operator==(const AlignedVector& lhs, const AlignedVector& rhs);



// Return true iff x is approximately equal to y.
template <class T>
bool near(const T& x, const T& y);

template <class T>
bool near(const std::vector<T>& x, const std::vector<T>& y) {
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


// Compute 1 / (1 + exp(-x)) .
float sigmoid(float x);
float fast_sigmoid(float x, size_t grid_size = DEFAULT_FAST_SIGMOID_GRID_SIZE);


// Seed the random number generator(s).
void seed(unsigned int s);


// Seed the random number generator(s) randomly.
void seed_default();


// Get thread's random number generator.
PRNG& get_urng();


// Sample vector of i.i.d. Gaussian random variables.
template <class T>
void sample_gaussian_vector(size_t n, T *z) {
  std::normal_distribution<T> d;
  for (size_t i = 0; i < n; ++i) {
    z[i] = d(get_urng());
  }
}

// Sample vector of i.i.d. centered uniform(0, 1) random variables.
template <class T>
void sample_centered_uniform_vector(size_t n, T *z) {
  std::uniform_real_distribution<T> d(-0.5, 0.5);
  for (size_t i = 0; i < n; ++i) {
    z[i] = d(get_urng());
  }
}


class ExponentCountNormalizer final {
  float _exponent;
  float _offset;

  public:
    ExponentCountNormalizer(float exponent = 1, float offset = 0);
    std::vector<float> normalize(const std::vector<size_t>&
                                            counts) const;

    ExponentCountNormalizer(ExponentCountNormalizer&& other) = default;
    ExponentCountNormalizer(const ExponentCountNormalizer& other) = default;

    bool equals(const ExponentCountNormalizer& other) const;
    void serialize(std::ostream& stream) const;
    static ExponentCountNormalizer deserialize(std::istream& stream);
};


class NaiveSampler final {
  size_t _size;
  std::vector<float> _probability_table;

  public:
    NaiveSampler(const std::vector<float>& probabilities);
    size_t sample() const;

    bool equals(const NaiveSampler& other) const;
    void serialize(std::ostream& stream) const;
    static NaiveSampler deserialize(std::istream& stream);

    NaiveSampler(size_t size,
                 std::vector<float>&& probability_table):
        _size(size),
        _probability_table(std::move(probability_table)) { }
    NaiveSampler(NaiveSampler&& other) = default;
    NaiveSampler(const NaiveSampler& other) = default;
};


class AliasSampler final {
  size_t _size;
  std::vector<size_t> _alias_table;
  std::vector<float> _probability_table;

  public:
    AliasSampler(const std::vector<float>& probabilities);
    size_t sample() const;

    bool equals(const AliasSampler& other) const;
    void serialize(std::ostream& stream) const;
    static AliasSampler deserialize(std::istream& stream);

    AliasSampler(size_t size,
                 std::vector<size_t>&& alias_table,
                 std::vector<float>&& probability_table):
        _size(size),
        _alias_table(std::move(alias_table)),
        _probability_table(std::move(probability_table)) { }
    AliasSampler(AliasSampler&& other) = default;
    AliasSampler(const AliasSampler& other) = default;

    AliasSampler& operator=(AliasSampler const & other);
    AliasSampler& operator=(AliasSampler && other);
};


template <typename T>
class ReservoirSampler;

template <typename T>
class ReservoirSampler final {
  size_t _size, _filled_size, _count;
  std::vector<T> _reservoir;

  public:
    ReservoirSampler(size_t size);
    T sample() const {
      std::uniform_int_distribution<size_t> d(0, _filled_size - 1);
      return _reservoir[d(get_urng())];
    }
    const T& operator[](size_t idx) const {
      return _reservoir[idx];
    }
    size_t size() const { return _size; }
    size_t filled_size() const { return _filled_size; }
    T insert(T val);
    void clear();

    bool equals(const ReservoirSampler<T>& other) const;
    void serialize(std::ostream& stream) const;
    static ReservoirSampler<T> deserialize(std::istream& stream);

    ReservoirSampler(size_t size, size_t filled_size, size_t count,
                     std::vector<T>&& reservoir):
        _size(size),
        _filled_size(filled_size),
        _count(count),
        _reservoir(std::move(reservoir)) { }
    ReservoirSampler(ReservoirSampler<T>&& other) = default;
    ReservoirSampler(const ReservoirSampler<T>& other) = default;
};


class Discretization final {
  std::vector<long> _samples;

  public:
    Discretization(const std::vector<float>& probabilities,
                   size_t num_samples);
    long sample() const {
      std::uniform_int_distribution<size_t> d(0, _samples.size() - 1);
      return _samples[d(get_urng())];
    }
    const long& operator[](size_t idx) const { return _samples[idx]; }
    size_t num_samples() const { return _samples.size(); }

    bool equals(const Discretization& other) const;
    void serialize(std::ostream& stream) const;
    static Discretization deserialize(std::istream& stream);

    Discretization(std::vector<long>&& samples):
      _samples(std::move(samples)) { }
    Discretization(Discretization&& other) = default;
    Discretization(const Discretization& other) = default;
};


//
// ReservoirSampler
//


template <typename T>
ReservoirSampler<T>::ReservoirSampler(size_t size):
    _size(size),
    _filled_size(0),
    _count(0),
    _reservoir(size) { }

template <typename T>
T ReservoirSampler<T>::insert(T val) {
  if (_filled_size < _size) {
    // reservoir not yet at capacity, insert val
    _reservoir[_filled_size] = val;
    ++_filled_size;
    ++_count;
    return val;
  } else {
    // reservoir at capacity, insert val w.p. _size/(_count+1)
    // (+1 is for val)
    std::uniform_int_distribution<size_t> d(0, _count);
    const size_t idx = d(get_urng());
    if (idx < _size) {
      const T prev_val = _reservoir[idx];
      _reservoir[idx] = val;
      ++_count;
      return prev_val;
    } else {
      ++_count;
      return val;
    }
  }
}

template <typename T>
void ReservoirSampler<T>::clear() {
  _filled_size = 0;
  _count = 0;
}

template <typename T>
void ReservoirSampler<T>::serialize(std::ostream& stream) const {
  Serializer<size_t>::serialize(_size, stream);
  Serializer<size_t>::serialize(_filled_size, stream);
  Serializer<size_t>::serialize(_count, stream);
  Serializer<std::vector<T> >::serialize(_reservoir, stream);
}

template <typename T>
ReservoirSampler<T> ReservoirSampler<T>::deserialize(std::istream& stream) {
  auto size(Serializer<size_t>::deserialize(stream));
  auto filled_size(Serializer<size_t>::deserialize(stream));
  auto count(Serializer<size_t>::deserialize(stream));
  auto reservoir(Serializer<std::vector<T> >::deserialize(stream));
  return ReservoirSampler<T>(
    size,
    filled_size,
    count,
    std::move(reservoir)
  );
}

template <typename T>
bool ReservoirSampler<T>::equals(const ReservoirSampler<T>& other) const {
  return
    _size == other._size &&
    _filled_size == other._filled_size &&
    _count == other._count &&
    _reservoir == other._reservoir;
}

template <typename T>
bool operator==(const ReservoirSampler<T>& lhs,
                const ReservoirSampler<T>& rhs) {
  return lhs.equals(rhs);
}


#endif
