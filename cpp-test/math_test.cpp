#include "math_test.h"
#include "test_util.h"
#include "_math.h"

#include <vector>

#include <gtest/gtest.h>


using namespace std;


TEST(aligned_vector_test, ctor_dtor_size_set_get) {
  AlignedVector v(7);
  EXPECT_EQ(7, v.size());
  for (size_t i = 0; i < 7; ++i) {
    v[i] = -(float)i;
  }
  EXPECT_EQ(-0, v[0]); EXPECT_EQ(-0, v.data()[0]);
  EXPECT_EQ(-1, v[1]); EXPECT_EQ(-1, v.data()[1]);
  EXPECT_EQ(-2, v[2]); EXPECT_EQ(-2, v.data()[2]);
  EXPECT_EQ(-3, v[3]); EXPECT_EQ(-3, v.data()[3]);
  EXPECT_EQ(-4, v[4]); EXPECT_EQ(-4, v.data()[4]);
  EXPECT_EQ(-5, v[5]); EXPECT_EQ(-5, v.data()[5]);
  EXPECT_EQ(-6, v[6]); EXPECT_EQ(-6, v.data()[6]);
  EXPECT_EQ(7, v.size());
}

TEST(aligned_vector_test, resize_smaller) {
  AlignedVector v(7);
  for (size_t i = 0; i < 7; ++i) {
    v[i] = -(float)i;
  }
  v.resize(3);
  EXPECT_EQ(3, v.size());
  EXPECT_EQ(-0, v[0]);
  EXPECT_EQ(-1, v[1]);
  EXPECT_EQ(-2, v[2]);
  for (size_t i = 0; i < 3; ++i) {
    v[i] = -2.*(float)i;
  }
  EXPECT_EQ(-0, v[0]);
  EXPECT_EQ(-2, v[1]);
  EXPECT_EQ(-4, v[2]);
}

TEST(aligned_vector_test, resize_larger) {
  AlignedVector v(3);
  for (size_t i = 0; i < 3; ++i) {
    v[i] = -(float)i;
  }
  v.resize(7);
  EXPECT_EQ(7, v.size());
  EXPECT_EQ(-0, v[0]);
  EXPECT_EQ(-1, v[1]);
  EXPECT_EQ(-2, v[2]);
  for (size_t i = 0; i < 7; ++i) {
    v[i] = -2.*(float)i;
  }
  EXPECT_EQ(-0, v[0]);
  EXPECT_EQ(-2, v[1]);
  EXPECT_EQ(-4, v[2]);
  EXPECT_EQ(-6, v[3]);
  EXPECT_EQ(-8, v[4]);
  EXPECT_EQ(-10, v[5]);
  EXPECT_EQ(-12, v[6]);
}

TEST(aligned_vector_test, equal) {
  AlignedVector v1(3), v2(3);
  for (size_t i = 0; i < 3; ++i) {
    v1[i] = -(float)i;
    v2[i] = -(float)i;
  }
  EXPECT_TRUE(v1 == v2);
  EXPECT_TRUE(v2 == v1);
  v1[1] = 1;
  EXPECT_FALSE(v1 == v2);
  EXPECT_FALSE(v2 == v1);
}

TEST(aligned_vector_test, equal_size_mismatch) {
  AlignedVector v1(3), v2(4);
  for (size_t i = 0; i < 3; ++i) {
    v1[i] = -(float)i;
    v2[i] = -(float)i;
  }
  v2[3] = 0;
  EXPECT_FALSE(v1 == v2);
  EXPECT_FALSE(v2 == v1);
}

TEST(sigmoid_test, positive) {
  EXPECT_NEAR(0.8807971, sigmoid(2), EPS);
}

TEST(sigmoid_test, negative) {
  EXPECT_NEAR(0.1192029, sigmoid(-2), EPS);
}

TEST(sigmoid_test, threshold_positive) {
  EXPECT_LT(sigmoid(SIGMOID_ARG_THRESHOLD - 1), 1);
  EXPECT_NEAR(1, sigmoid(SIGMOID_ARG_THRESHOLD + 1), EPS);
}

TEST(sigmoid_test, threshold_negative) {
  EXPECT_GT(sigmoid(-(SIGMOID_ARG_THRESHOLD - 1)), 0);
  EXPECT_NEAR(0, sigmoid(-(SIGMOID_ARG_THRESHOLD + 1)), EPS);
}

TEST(seed_test, seed) {
  seed(7);
  uniform_real_distribution<float> d;
  float u1 = d(get_urng());
  float u2 = d(get_urng());
  EXPECT_NE(u1, u2);
}

TEST(seed_test, seed_reset) {
  seed(7);
  uniform_real_distribution<float> d1;
  float u1 = d1(get_urng());
  float u2 = d1(get_urng());
  EXPECT_NE(u1, u2);
  seed(7);
  uniform_real_distribution<float> d3;
  float u3 = d3(get_urng());
  EXPECT_EQ(u1, u3);
}

TEST(sample_gaussian_vector_test, moments) {
  const size_t num_samples = 100000;
  vector<float> x(num_samples, 0);
  sample_gaussian_vector(num_samples, x.data());
  float sum = 0;
  float sumsq = 0;
  for (size_t t = 0; t < num_samples; ++t) {
    sum += x[t];
    sumsq += x[t] * x[t];
  }
  const float mean_sigma = sqrt(
    1. / num_samples
  );
  EXPECT_NEAR(0., sum / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) / pow(num_samples, 2)
  );
  EXPECT_NEAR(1., sumsq / (num_samples), 6. * variance_sigma);
}

TEST(sample_centered_uniform_vector_test, moments) {
  const size_t num_samples = 100000;
  vector<float> x(num_samples, 0);
  sample_centered_uniform_vector(num_samples, x.data());
  float sum = 0;
  float sumsq = 0;
  for (size_t t = 0; t < num_samples; ++t) {
    sum += x[t];
    sumsq += x[t] * x[t];
  }
  const float mean_sigma = sqrt(
    1. / num_samples
  );
  EXPECT_NEAR(0., sum / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) / pow(num_samples, 2)
  );
  EXPECT_NEAR(0.25 / 3., sumsq / (num_samples), 6. * variance_sigma);
}

TEST(double_near_test, general) {
  EXPECT_TRUE(near(7., 7. + DOUBLE_NEAR_THRESHOLD / 2.));
  EXPECT_TRUE(near(7., 7. - DOUBLE_NEAR_THRESHOLD / 2.));
  EXPECT_FALSE(near(7., 7. + DOUBLE_NEAR_THRESHOLD * 2.));
  EXPECT_FALSE(near(7., 7. - DOUBLE_NEAR_THRESHOLD * 2.));
}

TEST_F(DoubleVectorNearTest, exact) {
  EXPECT_TRUE(near(x, y));
}

TEST_F(DoubleVectorNearTest, too_large) {
  y.push_back(0);
  EXPECT_FALSE(near(x, y));
}

TEST_F(DoubleVectorNearTest, too_small) {
  x.push_back(0);
  EXPECT_FALSE(near(x, y));
}

TEST_F(DoubleVectorNearTest, near_0) {
  x[0] += DOUBLE_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(DoubleVectorNearTest, near_neg_0) {
  x[0] -= DOUBLE_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(DoubleVectorNearTest, near_0_2) {
  x[0] += DOUBLE_NEAR_THRESHOLD / 2;
  x[2] -= DOUBLE_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(DoubleVectorNearTest, far_0) {
  x[0] += DOUBLE_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(DoubleVectorNearTest, far_neg_0) {
  x[0] -= DOUBLE_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(DoubleVectorNearTest, far_2_0) {
  x[0] += DOUBLE_NEAR_THRESHOLD / 2;
  x[2] -= DOUBLE_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(DoubleVectorNearTest, far_0_2) {
  x[0] += DOUBLE_NEAR_THRESHOLD * 2;
  x[2] -= DOUBLE_NEAR_THRESHOLD / 2;
  EXPECT_FALSE(near(x, y));
}

TEST(float_near_test, general) {
  EXPECT_TRUE(near(7.f, 7.f + FLOAT_NEAR_THRESHOLD / 2.f));
  EXPECT_TRUE(near(7.f, 7.f - FLOAT_NEAR_THRESHOLD / 2.f));
  EXPECT_FALSE(near(7.f, 7.f + FLOAT_NEAR_THRESHOLD * 2.f));
  EXPECT_FALSE(near(7.f, 7.f - FLOAT_NEAR_THRESHOLD * 2.f));
}

TEST_F(FloatVectorNearTest, exact) {
  EXPECT_TRUE(near(x, y));
}

TEST_F(FloatVectorNearTest, too_large) {
  y.push_back(0);
  EXPECT_FALSE(near(x, y));
}

TEST_F(FloatVectorNearTest, too_small) {
  x.push_back(0);
  EXPECT_FALSE(near(x, y));
}

TEST_F(FloatVectorNearTest, near_0) {
  x[0] += FLOAT_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(FloatVectorNearTest, near_neg_0) {
  x[0] -= FLOAT_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(FloatVectorNearTest, near_0_2) {
  x[0] += FLOAT_NEAR_THRESHOLD / 2;
  x[2] -= FLOAT_NEAR_THRESHOLD / 2;
  EXPECT_TRUE(near(x, y));
}

TEST_F(FloatVectorNearTest, far_0) {
  x[0] += FLOAT_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(FloatVectorNearTest, far_neg_0) {
  x[0] -= FLOAT_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(FloatVectorNearTest, far_2_0) {
  x[0] += FLOAT_NEAR_THRESHOLD / 2;
  x[2] -= FLOAT_NEAR_THRESHOLD * 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(FloatVectorNearTest, far_0_2) {
  x[0] += FLOAT_NEAR_THRESHOLD * 2;
  x[2] -= FLOAT_NEAR_THRESHOLD / 2;
  EXPECT_FALSE(near(x, y));
}

TEST_F(AliasSamplerTest, sample) {
  const size_t num_samples = 100000;
  const float p_0 = 0.1,
               p_1 = 0.5,
               p_2 = 0.4;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = alias_sampler->sample();
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) -
                      (w == 2 ? p_2 : (w == 1 ? p_1 : p_0)), 2);
    }
  }
  const float sigma_0 = p_0 * (1 - p_0);
  const float sigma_1 = p_1 * (1 - p_1);
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_0, sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(p_1, sum[1] / num_samples, 6. * mean_sigma_1);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_samples - 1.) * sigma_1 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_samples, 6. * variance_sigma_1);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST(null_alias_sampler_test, constructor) {
  vector<float> probabilities;
  AliasSampler alias_sampler(probabilities);
  EXPECT_TRUE(true);
}

TEST_F(OneAtomAliasSamplerTest, sample) {
  const size_t num_samples = 100000;
  const float p_2 = 1;
  float sum[] = {0, 0, 0, 0};
  float sumsq[] = {0, 0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = alias_sampler->sample();
    sum[word_idx] += 1;
    for (size_t w = 0; w < 4; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) -
                      (w == 2 ? p_2 : 0), 2);
    }
  }
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_EQ(0, sum[0]);
  EXPECT_EQ(0, sum[1]);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  EXPECT_EQ(0, sum[3]);
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(TwoAtomAliasSamplerTest, sample) {
  const size_t num_samples = 100000;
  const float p_0 = 0.6,
               p_2 = 0.4;
  float sum[] = {0, 0, 0, 0};
  float sumsq[] = {0, 0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = alias_sampler->sample();
    sum[word_idx] += 1;
    for (size_t w = 0; w < 4; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) -
                      (w == 2 ? p_2 : (w == 0 ? p_0 : 0)), 2);
    }
  }
  const float sigma_0 = p_0 * (1 - p_0);
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_0, sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_EQ(0, sum[1]);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  EXPECT_EQ(0, sum[3]);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(UniformAliasSamplerTest, sample) {
  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0, 0};
  float sumsq[] = {0, 0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = alias_sampler->sample();
    sum[word_idx] += 1;
    for (size_t w = 0; w < 4; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - 0.25, 2);
    }
  }
  const float sigma = 0.25 * (1 - 0.25);
  const float mean_sigma = sqrt(
    sigma / num_samples
  );
  EXPECT_NEAR(0.25, sum[0] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(0.25, sum[1] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(0.25, sum[2] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(0.25, sum[3] / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) * sigma / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma, sumsq[0] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(sigma, sumsq[1] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(sigma, sumsq[2] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(sigma, sumsq[3] / num_samples, 6. * variance_sigma);
}

TEST_F(CountNormalizerTest, normalize) {
  vector<size_t> counts = {7, 0, 12};
  vector<float> normalized = count_normalizer->normalize(counts);
  const float z = pow(7 + 4.2, 0.8) + pow(0 + 4.2, 0.8) + pow(12 + 4.2, 0.8);
  EXPECT_NEAR(pow(7 + 4.2, 0.8) / z, normalized[0], EPS);
  EXPECT_NEAR(pow(0 + 4.2, 0.8) / z, normalized[1], EPS);
  EXPECT_NEAR(pow(12 + 4.2, 0.8) / z, normalized[2], EPS);
}

TEST_F(CountNormalizerTest, serialization_fixed_point) {
  stringstream ostream;
  count_normalizer->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(CountNormalizer::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(count_normalizer->equals(*from_stream));
}

TEST_F(ReservoirSamplerTest, sample) {
  const size_t num_samples = 100000;
  const float p[] = {2./3., 1./3., 0.};
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const long val = sampler->sample();
    const size_t idx = (val == -1 ? 0 : (val == 7 ? 1 : 2));
    sum[idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == idx ? 1. : 0.) - p[w], 2);
    }
  }
  const float sigma_0 = p[0] * (1 - p[0]);
  const float sigma_1 = p[1] * (1 - p[1]);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_samples
  );
  EXPECT_NEAR(p[0], sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(p[1], sum[1] / num_samples, 6. * mean_sigma_1);
  EXPECT_EQ(0, sum[2]);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_samples - 1.) * sigma_1 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_samples, 6. * variance_sigma_1);
}

TEST_F(ReservoirSamplerTest, move_ctor) {
  ReservoirSampler<long> other(move(*sampler));
  EXPECT_EQ(3, other.size());
  EXPECT_EQ(3, other.filled_size());
  EXPECT_EQ(-1, other[0]);
  EXPECT_EQ(7, other[1]);
  EXPECT_EQ(-1, other[2]);
  EXPECT_EQ(0, sampler->size());
  EXPECT_EQ(0, sampler->filled_size());
}

TEST(reservoir_sampler_test, accessors_mutators) {
  ReservoirSampler<long> sampler(3);

  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(0, sampler.filled_size());

  sampler.insert(-1);
  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(1, sampler.filled_size());
  EXPECT_EQ(-1, sampler[0]);

  sampler.insert(7);
  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(2, sampler.filled_size());
  EXPECT_EQ(-1, sampler[0]);
  EXPECT_EQ(7, sampler[1]);

  sampler.insert(-1);
  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(3, sampler.filled_size());
  EXPECT_EQ(-1, sampler[0]);
  EXPECT_EQ(7, sampler[1]);
  EXPECT_EQ(-1, sampler[2]);

  sampler.insert(-1);
  sampler.insert(4);
  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(3, sampler.filled_size());

  sampler.clear();
  EXPECT_EQ(3, sampler.size());
  EXPECT_EQ(0, sampler.filled_size());
  sampler.insert(7);
  sampler.insert(-1);
  sampler.insert(4);
  EXPECT_EQ(7, sampler[0]);
  EXPECT_EQ(-1, sampler[1]);
  EXPECT_EQ(4, sampler[2]);
}

TEST(reservoir_sampler_test, sample_with_ejections) {
  const size_t num_samples = 100000;
  const float p[] = {8./15., 4./15., 2./15., 1./15., 0};
  float sum[] = {0, 0, 0, 0, 0};
  float sumsq[] = {0, 0, 0, 0, 0};
  bool all_distinct = false;
  bool all_same = false;
  for (size_t t = 0; t < num_samples; ++t) {
    ReservoirSampler<long> sampler(3);
    // create distribution with weights 8, 4, 2, 1
    // for points -1, 7, 3, 0
    sampler.insert(-1);
    sampler.insert(7);
    sampler.insert(-1);
    sampler.insert(3);
    sampler.insert(-1);
    sampler.insert(-1);
    sampler.insert(-1);
    sampler.insert(3);
    sampler.insert(-1);
    sampler.insert(-1);
    sampler.insert(-1);
    sampler.insert(0);
    sampler.insert(7);
    sampler.insert(7);
    sampler.insert(7);
    const long val = sampler.sample();
    const size_t idx = (val == -1 ?
                        0 :
                        (val == 7 ?
                         1 :
                         (val == 3 ?
                          2 :
                          (val == 0 ?
                           3 :
                           4))));
    sum[idx] += 1;
    for (size_t w = 0; w < 5; ++w) {
      sumsq[w] += pow((w == idx ? 1. : 0.) - p[w], 2);
    }
    if (sampler[0] != sampler[1] && sampler[1] != sampler[2]) {
      // probability: 8/15 * 4/15 * 2/13
      all_distinct = true;
    }
    if (sampler[0] == sampler[1] && sampler[1] == sampler[2]) {
      // probability: 8/15 * 7/14 * 6/13
      all_same = true;
    }
  }
  const float sigma[] = {
    p[0] * (1 - p[0]),
    p[1] * (1 - p[1]),
    p[2] * (1 - p[2]),
    p[3] * (1 - p[3])
  };
  const float mean_sigma[] = {
    sqrt(sigma[0] / num_samples),
    sqrt(sigma[1] / num_samples),
    sqrt(sigma[2] / num_samples),
    sqrt(sigma[3] / num_samples),
  };
  EXPECT_NEAR(p[0], sum[0] / num_samples, 6. * mean_sigma[0]);
  EXPECT_NEAR(p[1], sum[1] / num_samples, 6. * mean_sigma[1]);
  EXPECT_NEAR(p[2], sum[2] / num_samples, 6. * mean_sigma[2]);
  EXPECT_NEAR(p[3], sum[3] / num_samples, 6. * mean_sigma[3]);
  EXPECT_EQ(0, sum[4]);
  const float variance_sigma[] = {
    (float) sqrt(2. * (num_samples - 1.) * sigma[0] / pow(num_samples, 2)),
    (float) sqrt(2. * (num_samples - 1.) * sigma[1] / pow(num_samples, 2)),
    (float) sqrt(2. * (num_samples - 1.) * sigma[2] / pow(num_samples, 2)),
    (float) sqrt(2. * (num_samples - 1.) * sigma[3] / pow(num_samples, 2))
  };
  EXPECT_NEAR(sigma[0], sumsq[0] / num_samples, 6. * variance_sigma[0]);
  EXPECT_NEAR(sigma[1], sumsq[1] / num_samples, 6. * variance_sigma[1]);
  EXPECT_NEAR(sigma[2], sumsq[2] / num_samples, 6. * variance_sigma[2]);
  EXPECT_NEAR(sigma[3], sumsq[3] / num_samples, 6. * variance_sigma[3]);
  EXPECT_TRUE(all_distinct);
  EXPECT_TRUE(all_same);
}

TEST_F(ReservoirSamplerTest, serialization_fixed_point) {
  stringstream ostream;
  sampler->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(ReservoirSampler<long>::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sampler->equals(*from_stream));
}

TEST_F(ReservoirSamplerTest, serialization_fixed_point_overfull) {
  sampler->insert(47);
  sampler->insert(47);
  sampler->insert(47);
  sampler->insert(47);

  stringstream ostream;
  sampler->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(ReservoirSampler<long>::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sampler->equals(*from_stream));
}

TEST(reservoir_sampler_test, serialization_fixed_point_underfull) {
  ReservoirSampler<long> sampler(3);
  sampler.insert(-1);
  sampler.insert(7);

  stringstream ostream;
  sampler.serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(ReservoirSampler<long>::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sampler.equals(*from_stream));
}

TEST_F(DiscretizationTest, sample) {
  const size_t num_samples = 100000;
  const float p_0 = 1./9.,
               p_1 = 5./9.,
               p_2 = 3./9.;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const long word_idx = sampler->sample();
    ASSERT_GE(word_idx, 0);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == size_t(word_idx) ? 1. : 0.) -
                      (w == 2 ? p_2 : (w == 1 ? p_1 : p_0)), 2);
    }
  }
  const float sigma_0 = p_0 * (1 - p_0);
  const float sigma_1 = p_1 * (1 - p_1);
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_0, sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(p_1, sum[1] / num_samples, 6. * mean_sigma_1);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_samples - 1.) * sigma_1 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_samples, 6. * variance_sigma_1);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(SubProbabilityDiscretizationTest, sample) {
  const size_t num_samples = 100000;
  const float p_0 = 1./9.,
               p_1 = 2./9.,
               p_2 = 6./9.;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const long word_idx = sampler->sample();
    ASSERT_GE(word_idx, 0);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == size_t(word_idx) ? 1. : 0.) -
                      (w == 2 ? p_2 : (w == 1 ? p_1 : p_0)), 2);
    }
  }
  const float sigma_0 = p_0 * (1 - p_0);
  const float sigma_1 = p_1 * (1 - p_1);
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_0, sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(p_1, sum[1] / num_samples, 6. * mean_sigma_1);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_samples - 1.) * sigma_1 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_samples, 6. * variance_sigma_1);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(OneAtomDiscretizationTest, sample) {
  const size_t num_samples = 100000;
  for (size_t t = 0; t < num_samples; ++t) {
    EXPECT_EQ(0, sampler->sample());
  }
}

TEST_F(NearlyTwoAtomDiscretizationTest, sample) {
  const size_t num_samples = 100000;
  const float p_0 = 1./9.,
               p_1 = 1./9.,
               p_2 = 7./9.;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const long word_idx = sampler->sample();
    ASSERT_GE(word_idx, 0);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == size_t(word_idx) ? 1. : 0.) -
                      (w == 2 ? p_2 : (w == 1 ? p_1 : p_0)), 2);
    }
  }
  const float sigma_0 = p_0 * (1 - p_0);
  const float sigma_1 = p_1 * (1 - p_1);
  const float sigma_2 = p_2 * (1 - p_2);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_0, sum[0] / num_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(p_1, sum[1] / num_samples, 6. * mean_sigma_1);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_0 = sqrt(
    2. * (num_samples - 1.) * sigma_0 / pow(num_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_samples - 1.) * sigma_1 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_samples, 6. * variance_sigma_1);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST(null_discretization_test, constructor) {
  vector<float> probabilities;
  Discretization sampler(probabilities, 7);
  EXPECT_TRUE(true);
}
