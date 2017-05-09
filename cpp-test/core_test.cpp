#include "core_test.h"
#include "test_util.h"
#include "_core.h"
#include "_serialization.h"
#include "_math.h"

#include <gtest/gtest.h>
#include <utility>
#include <sstream>


using namespace std;
using ::testing::Return;
using ::testing::Ref;
using ::testing::_;
using ::testing::InSequence;
using ::testing::Assign;
using ::testing::DoAll;


TEST(pair_cmp_test, individual) {
  // along first: d < a, b, c, e < f
  // along second: a < b, d, e, f < c
  pair<int,int>
    a = make_pair(2, 1),
    b = make_pair(2, 2),
    c = make_pair(2, 3),
    d = make_pair(1, 2),
    e = make_pair(2, 2),
    f = make_pair(3, 2);

  EXPECT_FALSE(pair_first_cmp(a, a)); EXPECT_FALSE(pair_second_cmp(a, a));
  EXPECT_FALSE(pair_first_cmp(a, b)); EXPECT_TRUE(pair_second_cmp(a, b));
  EXPECT_FALSE(pair_first_cmp(a, c)); EXPECT_TRUE(pair_second_cmp(a, c));
  EXPECT_FALSE(pair_first_cmp(a, d)); EXPECT_TRUE(pair_second_cmp(a, d));
  EXPECT_FALSE(pair_first_cmp(a, e)); EXPECT_TRUE(pair_second_cmp(a, e));
  EXPECT_TRUE(pair_first_cmp(a, f)); EXPECT_TRUE(pair_second_cmp(a, f));

  EXPECT_FALSE(pair_first_cmp(b, a)); EXPECT_FALSE(pair_second_cmp(b, a));
  EXPECT_FALSE(pair_first_cmp(b, b)); EXPECT_FALSE(pair_second_cmp(b, b));
  EXPECT_FALSE(pair_first_cmp(b, c)); EXPECT_TRUE(pair_second_cmp(b, c));
  EXPECT_FALSE(pair_first_cmp(b, d)); EXPECT_FALSE(pair_second_cmp(b, d));
  EXPECT_FALSE(pair_first_cmp(b, e)); EXPECT_FALSE(pair_second_cmp(b, e));
  EXPECT_TRUE(pair_first_cmp(b, f)); EXPECT_FALSE(pair_second_cmp(b, f));

  EXPECT_FALSE(pair_first_cmp(c, a)); EXPECT_FALSE(pair_second_cmp(c, a));
  EXPECT_FALSE(pair_first_cmp(c, b)); EXPECT_FALSE(pair_second_cmp(c, b));
  EXPECT_FALSE(pair_first_cmp(c, c)); EXPECT_FALSE(pair_second_cmp(c, c));
  EXPECT_FALSE(pair_first_cmp(c, d)); EXPECT_FALSE(pair_second_cmp(c, d));
  EXPECT_FALSE(pair_first_cmp(c, e)); EXPECT_FALSE(pair_second_cmp(c, e));
  EXPECT_TRUE(pair_first_cmp(c, f)); EXPECT_FALSE(pair_second_cmp(c, f));

  EXPECT_TRUE(pair_first_cmp(d, a)); EXPECT_FALSE(pair_second_cmp(d, a));
  EXPECT_TRUE(pair_first_cmp(d, b)); EXPECT_FALSE(pair_second_cmp(d, b));
  EXPECT_TRUE(pair_first_cmp(d, c)); EXPECT_TRUE(pair_second_cmp(d, c));
  EXPECT_FALSE(pair_first_cmp(d, d)); EXPECT_FALSE(pair_second_cmp(d, d));
  EXPECT_TRUE(pair_first_cmp(d, e)); EXPECT_FALSE(pair_second_cmp(d, e));
  EXPECT_TRUE(pair_first_cmp(d, f)); EXPECT_FALSE(pair_second_cmp(d, f));

  EXPECT_FALSE(pair_first_cmp(e, a)); EXPECT_FALSE(pair_second_cmp(e, a));
  EXPECT_FALSE(pair_first_cmp(e, b)); EXPECT_FALSE(pair_second_cmp(e, b));
  EXPECT_FALSE(pair_first_cmp(e, c)); EXPECT_TRUE(pair_second_cmp(e, c));
  EXPECT_FALSE(pair_first_cmp(e, d)); EXPECT_FALSE(pair_second_cmp(e, d));
  EXPECT_FALSE(pair_first_cmp(e, e)); EXPECT_FALSE(pair_second_cmp(e, e));
  EXPECT_TRUE(pair_first_cmp(e, f)); EXPECT_FALSE(pair_second_cmp(e, f));

  EXPECT_FALSE(pair_first_cmp(f, a)); EXPECT_FALSE(pair_second_cmp(f, a));
  EXPECT_FALSE(pair_first_cmp(f, b)); EXPECT_FALSE(pair_second_cmp(f, b));
  EXPECT_FALSE(pair_first_cmp(f, c)); EXPECT_TRUE(pair_second_cmp(f, c));
  EXPECT_FALSE(pair_first_cmp(f, d)); EXPECT_FALSE(pair_second_cmp(f, d));
  EXPECT_FALSE(pair_first_cmp(f, e)); EXPECT_FALSE(pair_second_cmp(f, e));
  EXPECT_FALSE(pair_first_cmp(f, f)); EXPECT_FALSE(pair_second_cmp(f, f));
}

TEST(sampling_strategy_test, serialization_exception) {
  stringstream ostream;
  Serializer<int>::serialize(-1, ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  EXPECT_THROW(SamplingStrategy::deserialize(istream), runtime_error);
}

TEST_F(UniformSamplingStrategyTest, sample_idx) {
  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - 1./3., 2);
    }
  }
  const float mean_sigma = sqrt(
    (1./3. * 2./3.) / num_samples
  );
  EXPECT_NEAR(1./3., sum[0] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[1] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[2] / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) * (1./3. * 2./3.) / pow(num_samples, 2)
  );
  EXPECT_NEAR(1./3. * 2./3., sumsq[0] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[1] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[2] / num_samples, 6. * variance_sigma);
}

TEST_F(UniformSamplingStrategyTest, step) {
  strategy->step(*lm, 42);

  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - 1./3., 2);
    }
  }
  const float mean_sigma = sqrt(
    (1./3. * 2./3.) / num_samples
  );
  EXPECT_NEAR(1./3., sum[0] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[1] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[2] / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) * (1./3. * 2./3.) / pow(num_samples, 2)
  );
  EXPECT_NEAR(1./3. * 2./3., sumsq[0] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[1] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[2] / num_samples, 6. * variance_sigma);
}

TEST_F(UniformSamplingStrategyTest, reset) {
  strategy->step(*lm, 42);
  strategy->reset(*lm, *reset_normalizer);

  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - 1./3., 2);
    }
  }
  const float mean_sigma = sqrt(
    (1./3. * 2./3.) / num_samples
  );
  EXPECT_NEAR(1./3., sum[0] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[1] / num_samples, 6. * mean_sigma);
  EXPECT_NEAR(1./3., sum[2] / num_samples, 6. * mean_sigma);
  const float variance_sigma = sqrt(
    2. * (num_samples - 1.) * (1./3. * 2./3.) / pow(num_samples, 2)
  );
  EXPECT_NEAR(1./3. * 2./3., sumsq[0] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[1] / num_samples, 6. * variance_sigma);
  EXPECT_NEAR(1./3. * 2./3., sumsq[2] / num_samples, 6. * variance_sigma);
}

TEST_F(UniformSamplingStrategyTest, serialization_fixed_point) {
  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SamplingStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
        dynamic_cast<const UniformSamplingStrategy&>(*from_stream)));
}

TEST_F(SmoothedEmpiricalSamplingStrategyTest, sample_idx) {
  const size_t num_samples = 100000;
  const float z = 2 * pow(smoothing_offset + 2., smoothing_exponent) +
                   pow(smoothing_offset + 3., smoothing_exponent);
  const float p_01 = pow(smoothing_offset + 2., smoothing_exponent) / z,
               p_2 = pow(smoothing_offset + 3., smoothing_exponent) / z;

  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - (w == 2 ? p_2 : p_01), 2);
    }
  }

  const float sigma_01 = p_01 * (1. - p_01);
  const float sigma_2 = p_2 * (1. - p_2);
  const float mean_sigma_01 = sqrt(
    sigma_01 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_01, sum[0] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_01, sum[1] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_01 = sqrt(
    2. * (num_samples - 1.) * sigma_01 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_01, sumsq[0] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_01, sumsq[1] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(EmpiricalSamplingStrategyTest, sample_idx) {
  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - (w == 2 ? 3./7. : 2./7.), 2);
    }
  }
  const float sigma_01 = 2./7. * 5./7.;
  const float sigma_2 = 3./7. * 4./7.;
  const float mean_sigma_01 = sqrt(
    sigma_01 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(2./7., sum[0] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(2./7., sum[1] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(3./7., sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_01 = sqrt(
    2. * (num_samples - 1.) * sigma_01 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_01, sumsq[0] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_01, sumsq[1] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(EmpiricalSamplingStrategyTest, step) {
  size_t num_trials = 100;
  vector<size_t> _counts = {7, 47, 9};
  EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));
  vector<float> _normalized_counts = {0, 0, 1};
  EXPECT_CALL(*count_normalizer, normalize(_counts)).
    WillRepeatedly(Return(_normalized_counts));

  // recompute during burn-in

  _normalized_counts = {1, 0, 0};
  EXPECT_CALL(*count_normalizer, normalize(_counts)).
    WillRepeatedly(Return(_normalized_counts));
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }

  _normalized_counts = {0, 0, 1};
  EXPECT_CALL(*count_normalizer, normalize(_counts)).
    WillRepeatedly(Return(_normalized_counts));
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(2, strategy->sample_idx(*lm)); }

  // recompute every refresh_interval steps

  _normalized_counts = {1, 0, 0};
  EXPECT_CALL(*count_normalizer, normalize(_counts)).
    WillRepeatedly(Return(_normalized_counts));
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(2, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(2, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(2, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(2, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }

  _normalized_counts = {0, 1, 0};
  EXPECT_CALL(*count_normalizer, normalize(_counts)).
    WillRepeatedly(Return(_normalized_counts));
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(0, strategy->sample_idx(*lm)); }
  strategy->step(*lm, 42);
  for (size_t i = 0; i < num_trials; ++i) { EXPECT_EQ(1, strategy->sample_idx(*lm)); }
}

TEST_F(SmoothedEmpiricalSamplingStrategyTest, reset) {
  strategy->step(*lm, 42);

  {
    InSequence in_sequence;
    EXPECT_CALL(*reset_lm, size()).WillRepeatedly(Return(3));
    EXPECT_CALL(*reset_lm, counts()).WillRepeatedly(Return(reset_counts));
    EXPECT_CALL(*reset_normalizer, normalize(reset_counts)).
      WillRepeatedly(Return(normalized_reset_counts));
  }

  strategy->reset(*reset_lm, *reset_normalizer);

  const size_t num_samples = 100000;
  const float p_01 = 7./16.,
               p_2 = 2./16.;

  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - (w == 2 ? p_2 : p_01), 2);
    }
  }

  const float sigma_01 = p_01 * (1. - p_01);
  const float sigma_2 = p_2 * (1. - p_2);
  const float mean_sigma_01 = sqrt(
    sigma_01 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(p_01, sum[0] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_01, sum[1] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_2, sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_01 = sqrt(
    2. * (num_samples - 1.) * sigma_01 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_01, sumsq[0] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_01, sumsq[1] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(EmpiricalSamplingStrategyTest, reset) {
  strategy->step(*lm, 42);

  {
    InSequence in_sequence;
    EXPECT_CALL(*reset_lm, size()).WillRepeatedly(Return(3));
    EXPECT_CALL(*reset_lm, counts()).WillRepeatedly(Return(reset_counts));
    EXPECT_CALL(*reset_normalizer, normalize(reset_counts)).
      WillRepeatedly(Return(normalized_reset_counts));
  }

  strategy->reset(*reset_lm, *reset_normalizer);

  const size_t num_samples = 100000;
  float sum[] = {0, 0, 0};
  float sumsq[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    sum[word_idx] += 1;
    for (size_t w = 0; w < 3; ++w) {
      sumsq[w] += pow((w == word_idx ? 1. : 0.) - (w == 2 ? 2./16. : 7./16.), 2);
    }
  }
  const float sigma_01 = 7./16. * 9./16.;
  const float sigma_2 = 2./16. * 14./16.;
  const float mean_sigma_01 = sqrt(
    sigma_01 / num_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_samples
  );
  EXPECT_NEAR(7./16., sum[0] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(7./16., sum[1] / num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(2./16., sum[2] / num_samples, 6. * mean_sigma_2);
  const float variance_sigma_01 = sqrt(
    2. * (num_samples - 1.) * sigma_01 / pow(num_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_samples - 1.) * sigma_2 / pow(num_samples, 2)
  );
  EXPECT_NEAR(sigma_01, sumsq[0] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_01, sumsq[1] / num_samples, 6. * variance_sigma_01);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_samples, 6. * variance_sigma_2);
}

TEST_F(EmpiricalSamplingStrategySerializationTest, serialization_fixed_point) {
  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SamplingStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
        dynamic_cast<const EmpiricalSamplingStrategy&>(*from_stream)));
}

TEST(language_model_test, serialization_exception) {
  stringstream ostream;
  Serializer<int>::serialize(-1, ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  EXPECT_THROW(LanguageModel::deserialize(istream), runtime_error);
}

TEST_F(EmpiricalSamplingStrategySerializationTest, initialized_serialization_fixed_point) {
  auto lm = std::make_shared<MockLanguageModel>();
  const std::vector<size_t> _counts = {2, 2, 3};
  EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
  EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));
  strategy->sample_idx(*lm);

  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SamplingStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
        dynamic_cast<const EmpiricalSamplingStrategy&>(*from_stream)));
}

TEST_F(ReservoirSamplingStrategyTest, sample_idx) {
  EXPECT_CALL(*sampler, sample()).WillRepeatedly(Return(47));

  const size_t num_samples = 100000;
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    EXPECT_EQ(47, word_idx);
  }
}

TEST_F(ReservoirSamplingStrategyTest, step) {
  EXPECT_CALL(*sampler, insert(47));
  strategy->step(*lm, 47);
}

TEST_F(ReservoirSamplingStrategyTest, reset) {
  const size_t num_reset_samples = 100000;

  EXPECT_CALL(*sampler, insert(47));
  strategy->step(*lm, 47);

  {
    InSequence in_sequence;
    EXPECT_CALL(*reset_lm, size()).WillRepeatedly(Return(3));
    EXPECT_CALL(*reset_lm, counts()).WillRepeatedly(Return(reset_counts));
    EXPECT_CALL(*reset_normalizer, normalize(reset_counts)).
      WillRepeatedly(Return(normalized_reset_counts));
    EXPECT_CALL(*sampler, clear());
    EXPECT_CALL(*sampler, size()).WillRepeatedly(Return(num_reset_samples));
  }

  vector<Counter> counters(3);
  // count calls over the domain (we use Assign with an overloaded
  // assignment op instead of something intuitive like Invoke because
  // with an overloaded call op because of misc. technical limitations
  // in gmock)
  EXPECT_CALL(*sampler, insert(0)).WillRepeatedly(DoAll(Assign(&(counters[0]), 1), Return(-1L)));
  EXPECT_CALL(*sampler, insert(1)).WillRepeatedly(DoAll(Assign(&(counters[1]), 1), Return(-1L)));
  EXPECT_CALL(*sampler, insert(2)).WillRepeatedly(DoAll(Assign(&(counters[2]), 1), Return(-1L)));

  strategy->reset(*reset_lm, *reset_normalizer);

  // insert was called correct number of times

  vector<float> sumsq(3);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t t = 0; t < counters[i].get(); ++t) {
      for (size_t j = 0; j < 3; ++j) {
        sumsq[j] += pow((i == j ? 1 : 0) - normalized_reset_counts[j], 2);
      }
    }
  }

  const float sigma_0 = normalized_reset_counts[0] * (1. - normalized_reset_counts[0]);
  const float sigma_1 = normalized_reset_counts[1] * (1. - normalized_reset_counts[1]);
  const float sigma_2 = normalized_reset_counts[2] * (1. - normalized_reset_counts[2]);
  const float mean_sigma_0 = sqrt(
    sigma_0 / num_reset_samples
  );
  const float mean_sigma_1 = sqrt(
    sigma_1 / num_reset_samples
  );
  const float mean_sigma_2 = sqrt(
    sigma_2 / num_reset_samples
  );
  EXPECT_NEAR(normalized_reset_counts[0], counters[0].get() / (float) num_reset_samples, 6. * mean_sigma_0);
  EXPECT_NEAR(normalized_reset_counts[1], counters[1].get() / (float) num_reset_samples, 6. * mean_sigma_1);
  EXPECT_NEAR(normalized_reset_counts[2], counters[2].get() / (float) num_reset_samples, 6. * mean_sigma_2);
  const float variance_sigma_0 = sqrt(
    2. * (num_reset_samples - 1.) * sigma_0 / pow(num_reset_samples, 2)
  );
  const float variance_sigma_1 = sqrt(
    2. * (num_reset_samples - 1.) * sigma_1 / pow(num_reset_samples, 2)
  );
  const float variance_sigma_2 = sqrt(
    2. * (num_reset_samples - 1.) * sigma_2 / pow(num_reset_samples, 2)
  );
  EXPECT_NEAR(sigma_0, sumsq[0] / num_reset_samples, 6. * variance_sigma_0);
  EXPECT_NEAR(sigma_1, sumsq[1] / num_reset_samples, 6. * variance_sigma_1);
  EXPECT_NEAR(sigma_2, sumsq[2] / num_reset_samples, 6. * variance_sigma_2);

  // sampler behaves according to spec

  EXPECT_CALL(*sampler, sample()).WillRepeatedly(Return(48));

  const size_t num_samples = 100000;
  for (size_t t = 0; t < num_samples; ++t) {
    const size_t word_idx = strategy->sample_idx(*lm);
    EXPECT_EQ(48, word_idx);
  }
}

TEST_F(ReservoirSamplingStrategySerializationTest, serialization_fixed_point) {
  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SamplingStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
      dynamic_cast<const ReservoirSamplingStrategy&>(*from_stream)));
}

TEST(context_strategy_test, serialization_exception) {
  stringstream ostream;
  Serializer<int>::serialize(-1, ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  EXPECT_THROW(ContextStrategy::deserialize(istream), runtime_error);
}

TEST_F(StaticContextStrategyTest, size) {
  const size_t num_trials = 1000;

  size_t t;
  for (t = 0; t < num_trials; ++t) {
    EXPECT_EQ(make_pair(size_t(3), size_t(3)), strategy->size(4, 4)); // LR loose
    EXPECT_EQ(make_pair(size_t(3), size_t(3)), strategy->size(3, 4)); // L tight
    EXPECT_EQ(make_pair(size_t(3), size_t(3)), strategy->size(4, 3)); // R tight
    EXPECT_EQ(make_pair(size_t(3), size_t(3)), strategy->size(3, 3)); // LR tight
    EXPECT_EQ(make_pair(size_t(2), size_t(3)), strategy->size(2, 3)); // L thresh
    EXPECT_EQ(make_pair(size_t(3), size_t(2)), strategy->size(3, 2)); // R thresh
    EXPECT_EQ(make_pair(size_t(2), size_t(2)), strategy->size(2, 2)); // LR thresh
    EXPECT_EQ(make_pair(size_t(3), size_t(0)), strategy->size(3, 0)); // R zero
    EXPECT_EQ(make_pair(size_t(0), size_t(3)), strategy->size(0, 3)); // L zero
    EXPECT_EQ(make_pair(size_t(0), size_t(0)), strategy->size(0, 0)); // LR zero
  }
}

TEST_F(StaticContextStrategyTest, serialization_fixed_point) {
  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(ContextStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
        dynamic_cast<const StaticContextStrategy&>(*from_stream)));
}

TEST_F(DynamicContextStrategyTest, size) {
  const size_t num_trials = 1000;

  size_t t;
  pair<size_t,size_t> ctx;
  size_t min_left, min_right, max_left, max_right;

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // LR loose
    ctx = strategy->size(4, 4);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 3); EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // L tight
    ctx = strategy->size(3, 4);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 3); EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // R tight
    ctx = strategy->size(4, 3);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 3); EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // LR tight
    ctx = strategy->size(3, 3);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 3); EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // L thresh
    ctx = strategy->size(2, 3);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 2);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 2); EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // R thresh
    ctx = strategy->size(3, 2);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 2);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 3); EXPECT_EQ(max_right, 2);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // LR thresh
    ctx = strategy->size(2, 2);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 2);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 2);
  }
  EXPECT_EQ(min_left, 1); EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_left, 2); EXPECT_EQ(max_right, 2);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // L zero
    ctx = strategy->size(0, 3);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
    EXPECT_EQ(ctx.first, 0);
    EXPECT_GE(ctx.second, 1); EXPECT_LE(ctx.second, 3);
  }
  EXPECT_EQ(min_right, 1);
  EXPECT_EQ(max_right, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // R zero
    ctx = strategy->size(3, 0);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
     EXPECT_GE(ctx.first, 1);  EXPECT_LE(ctx.first, 3);
    EXPECT_EQ(ctx.second, 0);
  }
  EXPECT_EQ(min_left, 1);
  EXPECT_EQ(max_left, 3);

  min_left = 5; min_right = 5; max_left = 0; max_right = 0;
  for (t = 0; t < num_trials; ++t) { // LR zero
    ctx = strategy->size(0, 0);
    min_left = min(min_left, ctx.first); min_right = min(min_right, ctx.second);
    max_left = max(max_left, ctx.first); max_right = max(max_right, ctx.second);
    EXPECT_EQ(ctx.first, 0);
    EXPECT_EQ(ctx.second, 0);
  }
}

TEST_F(DynamicContextStrategyTest, serialization_fixed_point) {
  stringstream ostream;
  strategy->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(ContextStrategy::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(strategy->equals(
        dynamic_cast<const DynamicContextStrategy&>(*from_stream)));
}

TEST(space_saving_language_model_test, subsample_none) {
  const float threshold = 4./7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  EXPECT_EQ(num_samples, sum[0]);
  EXPECT_EQ(num_samples, sum[1]);
  EXPECT_EQ(num_samples, sum[2]);
}

TEST(space_saving_language_model_test, subsample_one) {
  const float threshold = 2.5/7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  const float p_2 = sqrt(threshold / (3./7.));
  const float mean_sigma_2 = sqrt(
    p_2 * (1. - p_2) / num_samples
  );
  EXPECT_EQ(num_samples, sum[0]);
  EXPECT_EQ(num_samples, sum[1]);
  EXPECT_NEAR(p_2, sum[2] / (float) num_samples, 6. * mean_sigma_2);
}

TEST(space_saving_language_model_test, subsample_all) {
  const float threshold = 1./7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  const float p_01 = sqrt(threshold / (2./7.));
  const float p_2 = sqrt(threshold / (3./7.));
  const float mean_sigma_01 = sqrt(
    p_01 * (1. - p_01) / num_samples
  );
  const float mean_sigma_2 = sqrt(
    p_2 * (1. - p_2) / num_samples
  );
  EXPECT_NEAR(p_01, sum[0] / (float) num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_01, sum[1] / (float) num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_2, sum[2] / (float) num_samples, 6. * mean_sigma_2);
}

TEST(space_saving_language_model_test, truncate_trivial_loose) {
  const float threshold = 2.5/7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  EXPECT_THROW(lm.truncate(4), logic_error);
}

TEST(space_saving_language_model_test, truncate_trivial_tight) {
  const float threshold = 2.5/7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  EXPECT_THROW(lm.truncate(3), logic_error);
}

TEST(space_saving_language_model_test, truncate_nontrivial) {
  const float threshold = 2.5/7.;

  SpaceSavingLanguageModel lm(3, threshold);
  lm.increment("foo");
  lm.increment("bar");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");
  lm.increment("baz");

  EXPECT_THROW(lm.truncate(2), logic_error);
}

TEST_F(SpaceSavingLanguageModelTest, algorithm) {
  EXPECT_EQ(0, lm->size());
  EXPECT_EQ(0, lm->total());

  pair<long,string> ret;

  ret = lm->increment("foo");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(1, lm->size());
  EXPECT_EQ(1, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(1, lm->count(0));
  EXPECT_EQ((vector<size_t> {1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bar"));
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));

  ret = lm->increment("bar");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(2, lm->size());
  EXPECT_EQ(2, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(1, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ((vector<size_t> {1, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));

  ret = lm->increment("foo");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(2, lm->size());
  EXPECT_EQ(3, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ((vector<size_t> {2, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(4, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(1, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 1, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(5, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 1, 2}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));

  ret = lm->increment("bbq");
  EXPECT_EQ(1, ret.first);
  EXPECT_EQ("bar", ret.second);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(6, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bbq"));   EXPECT_EQ(2, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 2, 2}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bar"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bbq", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->capacity());
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(7, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bbq"));   EXPECT_EQ(2, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(3, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 2, 3}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bar"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bbq", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));
}

TEST_F(SpaceSavingLanguageModelTest, ordered_counts) {
  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {1}), lm->ordered_counts());

  lm->increment("bar");
  EXPECT_EQ((vector<size_t> {1, 1}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {2, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {2, 1, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {2, 2, 1}), lm->ordered_counts());

  lm->increment("bbq");
  EXPECT_EQ((vector<size_t> {2, 2, 2}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {3, 2, 2}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {4, 2, 2}), lm->ordered_counts());

  lm->increment("bbq");
  EXPECT_EQ((vector<size_t> {4, 3, 2}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {4, 3, 3}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {4, 4, 3}), lm->ordered_counts());
}

TEST_F(SpaceSavingLanguageModelUnfullSerializationTest, fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(SpaceSavingLanguageModelUnfullSerializationTest,
       dimension_fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(SpaceSavingLanguageModelSerializationTest, fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(SpaceSavingLanguageModelSerializationTest, dimension_fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(SpaceSavingLanguageModelTest,
       serialization_subsample_threshold_fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST(naive_language_model_test, subsample_none) {
  const float threshold = 4./7.;

  NaiveLanguageModel lm(threshold);
  lm.increment("foo");
  lm.increment("bbq");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  EXPECT_EQ(num_samples, sum[0]);
  EXPECT_EQ(num_samples, sum[1]);
  EXPECT_EQ(num_samples, sum[2]);
}

TEST(naive_language_model_test, subsample_one) {
  const float threshold = 2.5/7.;

  NaiveLanguageModel lm(threshold);
  lm.increment("foo");
  lm.increment("bbq");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  const float p_2 = sqrt(threshold / (3./7.));
  const float mean_sigma_2 = sqrt(
    p_2 * (1. - p_2) / num_samples
  );
  EXPECT_EQ(num_samples, sum[0]);
  EXPECT_EQ(num_samples, sum[1]);
  EXPECT_NEAR(p_2, sum[2] / (float) num_samples, 6. * mean_sigma_2);
}

TEST(naive_language_model_test, subsample_all) {
  const float threshold = 1./7.;

  NaiveLanguageModel lm(threshold);
  lm.increment("foo");
  lm.increment("bbq");
  lm.increment("foo");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("baz");
  lm.increment("bbq");

  const size_t num_samples = 100000;
  size_t sum[] = {0, 0, 0};
  for (size_t t = 0; t < num_samples; ++t) {
    for (size_t word_idx = 0; word_idx < 3; ++word_idx) {
      if (lm.subsample(word_idx)) {
        ++(sum[word_idx]);
      }
    }
  }
  const float p_01 = sqrt(threshold / (2./7.));
  const float p_2 = sqrt(threshold / (3./7.));
  const float mean_sigma_01 = sqrt(
    p_01 * (1. - p_01) / num_samples
  );
  const float mean_sigma_2 = sqrt(
    p_2 * (1. - p_2) / num_samples
  );
  EXPECT_NEAR(p_01, sum[0] / (float) num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_01, sum[1] / (float) num_samples, 6. * mean_sigma_01);
  EXPECT_NEAR(p_2, sum[2] / (float) num_samples, 6. * mean_sigma_2);
}

TEST_F(NaiveLanguageModelTest, truncate_trivial_loose) {
  // foo: 5, bbq: 2, baz: 4, bar: 1
  lm->increment("bbq");
  lm->increment("bar");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("bbq");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");

  lm->truncate(5);

  EXPECT_EQ(4, lm->size());
  EXPECT_EQ(12, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(5, lm->count(0));
  EXPECT_EQ(1, lm->lookup("baz"));   EXPECT_EQ(4, lm->count(1));
  EXPECT_EQ(2, lm->lookup("bbq"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ(3, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(3));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("baz", lm->reverse_lookup(1));
  EXPECT_EQ("bbq", lm->reverse_lookup(2));
  EXPECT_EQ("bar", lm->reverse_lookup(3));
}

TEST_F(NaiveLanguageModelTest, truncate_trivial_tight) {
  // foo: 5, bbq: 2, baz: 4, bar: 1
  lm->increment("bbq");
  lm->increment("bar");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("bbq");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");

  lm->truncate(4);

  EXPECT_EQ(4, lm->size());
  EXPECT_EQ(12, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(5, lm->count(0));
  EXPECT_EQ(1, lm->lookup("baz"));   EXPECT_EQ(4, lm->count(1));
  EXPECT_EQ(2, lm->lookup("bbq"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ(3, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(3));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("baz", lm->reverse_lookup(1));
  EXPECT_EQ("bbq", lm->reverse_lookup(2));
  EXPECT_EQ("bar", lm->reverse_lookup(3));
}

TEST_F(NaiveLanguageModelTest, truncate_nontrivial) {
  // foo: 5, bbq: 2, baz: 4, bar: 1
  lm->increment("bbq");
  lm->increment("bar");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("bbq");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");

  lm->truncate(2);

  EXPECT_EQ(2, lm->size());
  EXPECT_EQ(9, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(5, lm->count(0));
  EXPECT_EQ(1, lm->lookup("baz"));   EXPECT_EQ(4, lm->count(1));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("baz", lm->reverse_lookup(1));
}

TEST_F(NaiveLanguageModelTest, truncate_nontrivial_ties) {
  // foo: 5, bbq: 3, baz: 3, bar: 1
  lm->increment("bbq");
  lm->increment("bar");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("baz");
  lm->increment("bbq");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("foo");
  lm->increment("bbq");

  lm->truncate(3);

  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(11, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(5, lm->count(0));
  EXPECT_EQ(3, lm->count(1));
  EXPECT_EQ(3, lm->count(2));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  switch (lm->lookup("baz")) {
    case 1:
      EXPECT_EQ(2, lm->lookup("bbq"));
      EXPECT_EQ("baz", lm->reverse_lookup(1));
      EXPECT_EQ("bbq", lm->reverse_lookup(2));
      break;
    case 2:
      EXPECT_EQ(2, lm->lookup("baz"));
      EXPECT_EQ("bbq", lm->reverse_lookup(1));
      EXPECT_EQ("baz", lm->reverse_lookup(2));
      break;
    default:
      FAIL();
      break;
  }
}

TEST_F(NaiveLanguageModelTest, algorithm) {
  EXPECT_EQ(0, lm->size());
  EXPECT_EQ(0, lm->total());

  pair<long,string> ret;

  ret = lm->increment("foo");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(1, lm->size());
  EXPECT_EQ(1, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(1, lm->count(0));
  EXPECT_EQ((vector<size_t> {1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bar"));
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));

  ret = lm->increment("bar");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(2, lm->size());
  EXPECT_EQ(2, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(1, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ((vector<size_t> {1, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));

  ret = lm->increment("foo");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(2, lm->size());
  EXPECT_EQ(3, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ((vector<size_t> {2, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("baz"));
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(4, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(1, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 1, 1}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(3, lm->size());
  EXPECT_EQ(5, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ((vector<size_t> {2, 1, 2}), lm->counts());
  EXPECT_EQ(-1, lm->lookup("bbq"));
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));

  ret = lm->increment("bbq");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(4, lm->size());
  EXPECT_EQ(6, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(2, lm->count(2));
  EXPECT_EQ(3, lm->lookup("bbq"));   EXPECT_EQ(1, lm->count(3));
  EXPECT_EQ((vector<size_t> {2, 1, 2, 1}), lm->counts());
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));
  EXPECT_EQ("bbq", lm->reverse_lookup(3));

  ret = lm->increment("baz");
  EXPECT_EQ(-1, ret.first);
  EXPECT_EQ(4, lm->size());
  EXPECT_EQ(7, lm->total());
  EXPECT_EQ(0, lm->lookup("foo"));   EXPECT_EQ(2, lm->count(0));
  EXPECT_EQ(1, lm->lookup("bar"));   EXPECT_EQ(1, lm->count(1));
  EXPECT_EQ(2, lm->lookup("baz"));   EXPECT_EQ(3, lm->count(2));
  EXPECT_EQ(3, lm->lookup("bbq"));   EXPECT_EQ(1, lm->count(3));
  EXPECT_EQ((vector<size_t> {2, 1, 3, 1}), lm->counts());
  EXPECT_EQ("foo", lm->reverse_lookup(0));
  EXPECT_EQ("bar", lm->reverse_lookup(1));
  EXPECT_EQ("baz", lm->reverse_lookup(2));
  EXPECT_EQ("bbq", lm->reverse_lookup(3));
}

TEST_F(NaiveLanguageModelTest, ordered_counts) {
  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {1}), lm->ordered_counts());

  lm->increment("bar");
  EXPECT_EQ((vector<size_t> {1, 1}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {2, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {2, 1, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {2, 2, 1}), lm->ordered_counts());

  lm->increment("bbq");
  EXPECT_EQ((vector<size_t> {2, 2, 1, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {3, 2, 1, 1}), lm->ordered_counts());

  lm->increment("baz");
  EXPECT_EQ((vector<size_t> {4, 2, 1, 1}), lm->ordered_counts());

  lm->increment("bbq");
  EXPECT_EQ((vector<size_t> {4, 2, 2, 1}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {4, 3, 2, 1}), lm->ordered_counts());

  lm->increment("foo");
  EXPECT_EQ((vector<size_t> {4, 4, 2, 1}), lm->ordered_counts());
}

TEST_F(NaiveLanguageModelSerializationTest, fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(NaiveLanguageModelSerializationTest, dimension_fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(NaiveLanguageModelTest,
       serialization_subsample_threshold_fixed_point) {
  stringstream ostream;
  lm->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(lm->equals(*from_stream));
}

TEST_F(WordContextFactorizationTest, dimension) {
  EXPECT_EQ(3, factorization->get_vocab_dim());
  EXPECT_EQ(2, factorization->get_embedding_dim());
}

TEST_F(WordContextFactorizationTest, get_embedding) {
  const float
    *word_0 = factorization->get_word_embedding(0),
    *word_1 = factorization->get_word_embedding(1),
    *word_2 = factorization->get_word_embedding(2),
    *context_0 = factorization->get_context_embedding(0),
    *context_1 = factorization->get_context_embedding(1),
    *context_2 = factorization->get_context_embedding(2);

  EXPECT_NE(0, word_0[0]); EXPECT_NE(0, word_0[1]);
  EXPECT_NE(0, word_1[0]); EXPECT_NE(0, word_1[1]);
  EXPECT_NE(0, word_2[0]); EXPECT_NE(0, word_2[1]);

  EXPECT_EQ(0, context_0[0]); EXPECT_EQ(0, context_0[1]);
  EXPECT_EQ(0, context_1[0]); EXPECT_EQ(0, context_1[1]);
  EXPECT_EQ(0, context_2[0]); EXPECT_EQ(0, context_2[1]);

  factorization->get_context_embedding(1)[0] = 7;

  EXPECT_NE(0, word_0[0]); EXPECT_NE(0, word_0[1]);
  EXPECT_NE(0, word_1[0]); EXPECT_NE(0, word_1[1]);
  EXPECT_NE(0, word_2[0]); EXPECT_NE(0, word_2[1]);

  EXPECT_EQ(0, context_0[0]); EXPECT_EQ(0, context_0[1]);
  EXPECT_NE(0, context_1[0]); EXPECT_EQ(0, context_1[1]);
  EXPECT_EQ(0, context_2[0]); EXPECT_EQ(0, context_2[1]);

  factorization->get_word_embedding(0)[1] = 0;

  EXPECT_NE(0, word_0[0]); EXPECT_EQ(0, word_0[1]);
  EXPECT_NE(0, word_1[0]); EXPECT_NE(0, word_1[1]);
  EXPECT_NE(0, word_2[0]); EXPECT_NE(0, word_2[1]);

  EXPECT_EQ(0, context_0[0]); EXPECT_EQ(0, context_0[1]);
  EXPECT_NE(0, context_1[0]); EXPECT_EQ(0, context_1[1]);
  EXPECT_EQ(0, context_2[0]); EXPECT_EQ(0, context_2[1]);
}

TEST_F(WordContextFactorizationSerializationTest, fixed_point) {
  stringstream ostream;
  factorization->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(WordContextFactorization::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(factorization->equals(*from_stream));
}

TEST_F(WordContextFactorizationSerializationTest, dimension_fixed_point) {
  stringstream ostream;
  factorization->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(WordContextFactorization::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(factorization->equals(*from_stream));
}

TEST_F(OneDimSGDTest, step) {
  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  sgd->step(0);
  EXPECT_NEAR(0.495, sgd->get_rho(0), EPS);
  sgd->step(0);
  EXPECT_NEAR(0.49, sgd->get_rho(0), EPS);
  for (size_t t = 0; t < 200; ++t) {
    sgd->step(0);
  }
  EXPECT_NEAR(0.1, sgd->get_rho(0), EPS);
}

TEST_F(OneDimSGDTest, reset) {
  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  sgd->step(0);
  EXPECT_NEAR(0.495, sgd->get_rho(0), EPS);

  sgd->reset(0);

  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  sgd->step(0);
  EXPECT_NEAR(0.495, sgd->get_rho(0), EPS);
}

TEST_F(OneDimSGDTest, gradient_update) {
  float x[] = {0.5, -1, -2};
  const float g[] = {-5, 0, -3};

  sgd->step(0);

  sgd->gradient_update(0, 3, g, x);
  EXPECT_NEAR(-1.975, x[0], EPS);
  EXPECT_NEAR(-1,        x[1], EPS);
  EXPECT_NEAR(-3.485, x[2], EPS);
}

TEST_F(OneDimSGDTest, scaled_gradient_update) {
  float x[] = {0.5, -1, -2};
  const float g[] = {-5, 0, -3};

  sgd->step(0);

  sgd->scaled_gradient_update(0, 3, g, x, 2);
  EXPECT_NEAR(-4.45, x[0], EPS);
  EXPECT_NEAR(-1,        x[1], EPS);
  EXPECT_NEAR(-4.97, x[2], EPS);
}

TEST_F(ThreeDimSGDTest, step) {
  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(1), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(2), EPS);

  sgd->step(1);

  EXPECT_LT(sgd->get_rho(1), sgd->get_rho(0));
  EXPECT_EQ(sgd->get_rho(0), sgd->get_rho(2));

  sgd->step(2);

  EXPECT_LT(sgd->get_rho(1), sgd->get_rho(0));
  EXPECT_EQ(sgd->get_rho(1), sgd->get_rho(2));

  sgd->step(2);

  EXPECT_LT(sgd->get_rho(2), sgd->get_rho(1));
  EXPECT_LT(sgd->get_rho(1), sgd->get_rho(0));
}

TEST_F(ThreeDimSGDTest, reset) {
  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(1), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(2), EPS);

  sgd->step(1);
  sgd->step(2);
  sgd->step(2);

  sgd->reset(2);

  EXPECT_LT(sgd->get_rho(1), sgd->get_rho(0));
  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(2), EPS);

  sgd->reset(1);

  EXPECT_NEAR(0.5, sgd->get_rho(0), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(1), EPS);
  EXPECT_NEAR(0.5, sgd->get_rho(2), EPS);
}

TEST_F(ThreeDimSGDTest, gradient_update) {
  float x[] = {0.5, -1, -2};
  const float g[] = {-5, 0, -3};

  sgd->step(0);
  sgd->step(1);
  sgd->step(1);
  sgd->step(2);

  sgd->gradient_update(1, 3, g, x);
  EXPECT_NEAR(-1.95,     x[0], EPS);
  EXPECT_NEAR(-1,        x[1], EPS);
  EXPECT_NEAR(-3.47,     x[2], EPS);
}

TEST_F(ThreeDimSGDTest, scaled_gradient_update) {
  float x[] = {0.5, -1, -2};
  const float g[] = {-5, 0, -3};

  sgd->step(0);
  sgd->step(1);
  sgd->step(1);
  sgd->step(2);

  sgd->scaled_gradient_update(1, 3, g, x, 2);
  EXPECT_NEAR(-4.4,      x[0], EPS);
  EXPECT_NEAR(-1,        x[1], EPS);
  EXPECT_NEAR(-4.94,     x[2], EPS);
}

TEST_F(SGDSerializationTest, fixed_point) {
  stringstream ostream;
  sgd->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGD::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sgd->equals(*from_stream));
}

TEST_F(SGDSerializationTest, schedule_fixed_point) {
  stringstream ostream;
  sgd->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGD::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sgd->equals(*from_stream));
}

TEST_F(SGDSerializationTest, dimension_fixed_point) {
  stringstream ostream;
  sgd->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGD::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sgd->equals(*from_stream));
}

TEST_F(LanguageModelExampleStoreTest, get_language_model) {
  EXPECT_CALL(*lm, equals(Ref(*lm))).WillOnce(Return(true));
  EXPECT_TRUE(lm->equals(example_store->get_language_model()));
}

TEST_F(LanguageModelExampleStoreTest, increment) {
  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(3L));

  example_store->increment("foo", 42L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(0, example_store->get_examples(1).filled_size());
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(1, example_store->get_examples(3).filled_size());
  EXPECT_EQ(42L, example_store->get_examples(3)[0]);

  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(3L));

  example_store->increment("foo", 47L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(0, example_store->get_examples(1).filled_size());
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(2, example_store->get_examples(3).filled_size());
  EXPECT_EQ(42L, example_store->get_examples(3)[0]);
  EXPECT_EQ(47L, example_store->get_examples(3)[1]);

  EXPECT_CALL(*lm, increment("bar")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1L));

  example_store->increment("bar", 7L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(1, example_store->get_examples(1).filled_size());
  EXPECT_EQ(7L, example_store->get_examples(1)[0]);
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(2, example_store->get_examples(3).filled_size());
  EXPECT_EQ(42L, example_store->get_examples(3)[0]);
  EXPECT_EQ(47L, example_store->get_examples(3)[1]);

  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(3L));

  example_store->increment("foo", 7L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(1, example_store->get_examples(1).filled_size());
  EXPECT_EQ(7L, example_store->get_examples(1)[0]);
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(2, example_store->get_examples(3).filled_size());

  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(5L));

  example_store->increment("foo", 9L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(1, example_store->get_examples(1).filled_size());
  EXPECT_EQ(7L, example_store->get_examples(1)[0]);
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(2, example_store->get_examples(3).filled_size());
  EXPECT_EQ(2, example_store->get_examples(4).size());
  EXPECT_EQ(0, example_store->get_examples(4).filled_size());
  EXPECT_EQ(2, example_store->get_examples(5).size());
  EXPECT_EQ(1, example_store->get_examples(5).filled_size());
  EXPECT_EQ(9L, example_store->get_examples(5)[0]);

  EXPECT_CALL(*lm, increment("baz")).
    WillOnce(Return(make_pair(3L, string())));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(3L));

  example_store->increment("baz", 5L);

  EXPECT_EQ(2, example_store->get_examples(0).size());
  EXPECT_EQ(0, example_store->get_examples(0).filled_size());
  EXPECT_EQ(2, example_store->get_examples(1).size());
  EXPECT_EQ(1, example_store->get_examples(1).filled_size());
  EXPECT_EQ(7L, example_store->get_examples(1)[0]);
  EXPECT_EQ(2, example_store->get_examples(2).size());
  EXPECT_EQ(0, example_store->get_examples(2).filled_size());
  EXPECT_EQ(2, example_store->get_examples(3).size());
  EXPECT_EQ(1, example_store->get_examples(3).filled_size());
  EXPECT_EQ(5L, example_store->get_examples(3)[0]);
  EXPECT_EQ(2, example_store->get_examples(4).size());
  EXPECT_EQ(0, example_store->get_examples(4).filled_size());
  EXPECT_EQ(2, example_store->get_examples(5).size());
  EXPECT_EQ(1, example_store->get_examples(5).filled_size());
  EXPECT_EQ(9L, example_store->get_examples(5)[0]);
}

TEST_F(LanguageModelExampleStoreSerializationTest, serialization_fixed_point) {
  stringstream ostream;
  example_store->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(LanguageModelExampleStore<long>::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(example_store->equals(
        dynamic_cast<const LanguageModelExampleStore<long>&>(*from_stream)));
}
