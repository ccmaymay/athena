#ifndef ATHENA_CORE_TEST_H
#define ATHENA_CORE_TEST_H


#include "_core.h"
#include <cmath>
#include "core_mock.h"
#include "math_mock.h"
#include "test_util.h"

#include <gtest/gtest.h>


using ::testing::Return;


// workaround for ByRef not exposing operator() ... or anything else
// (if it did we'd use InvokeWithoutArgs and ByRef)
class Counter {
  size_t _i;
  public:
    Counter(): _i(0) { }
    Counter& operator=(size_t inc) { _i += inc; return *this; }
    size_t get() const { return _i; }
  private:
    Counter(const Counter& other);
};


class SpaceSavingLanguageModelTest: public ::testing::Test {
  protected:
    std::shared_ptr<SpaceSavingLanguageModel> lm;

    virtual void SetUp() {
      lm = std::make_shared<SpaceSavingLanguageModel>(3);
    }

    virtual void TearDown() { }
};

class SpaceSavingLanguageModelUnfullSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SpaceSavingLanguageModel> lm;

    virtual void SetUp() {
      lm = std::make_shared<SpaceSavingLanguageModel>(3);
      lm->increment("foo");
      lm->increment("bar");
      lm->increment("foo");
    }

    virtual void TearDown() { }
};

class SpaceSavingLanguageModelSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SpaceSavingLanguageModel> lm;

    virtual void SetUp() {
      lm = std::make_shared<SpaceSavingLanguageModel>(3);
      lm->increment("foo");
      lm->increment("bar");
      lm->increment("foo");
      lm->increment("baz");
      lm->increment("baz");
      lm->increment("bbq");
      lm->increment("baz");
    }

    virtual void TearDown() { }
};

class NaiveLanguageModelTest: public ::testing::Test {
  protected:
    std::shared_ptr<NaiveLanguageModel> lm;

    virtual void SetUp() {
      lm = std::make_shared<NaiveLanguageModel>();
    }

    virtual void TearDown() { }
};

class NaiveLanguageModelSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<NaiveLanguageModel> lm;

    virtual void SetUp() {
      lm = std::make_shared<NaiveLanguageModel>();
      lm->increment("foo");
      lm->increment("bar");
      lm->increment("foo");
      lm->increment("baz");
      lm->increment("baz");
      lm->increment("bbq");
      lm->increment("baz");
    }

    virtual void TearDown() { }
};

class WordContextFactorizationTest: public ::testing::Test {
  protected:
    std::shared_ptr<WordContextFactorization> factorization;

    virtual void SetUp() {
      factorization = std::make_shared<WordContextFactorization>(3, 2);
    }

    virtual void TearDown() { }
};

class WordContextFactorizationSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<WordContextFactorization> factorization;

    virtual void SetUp() {
      factorization = std::make_shared<WordContextFactorization>(3, 2);

      factorization->get_word_embedding(0)[0] = 1;
      factorization->get_word_embedding(0)[1] = -2;

      factorization->get_word_embedding(1)[0] = -3;
      factorization->get_word_embedding(1)[1] = 2;

      factorization->get_word_embedding(2)[0] = 4;
      factorization->get_word_embedding(2)[1] = 0;

      factorization->get_context_embedding(0)[0] = 4;
      factorization->get_context_embedding(0)[1] = 0;

      factorization->get_context_embedding(1)[0] = -3;
      factorization->get_context_embedding(1)[1] = 2;

      factorization->get_context_embedding(2)[0] = 1;
      factorization->get_context_embedding(2)[1] = -2;
    }

    virtual void TearDown() { }
};

class SGDTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(100, 0.5, 0.1);
    }

    virtual void TearDown() { }
};

class SGDSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(100, 0.5, 0.1);
      sgd->step();
      sgd->step();
      sgd->step();
    }

    virtual void TearDown() { }
};

class UniformSamplingStrategyTest: public ::testing::Test {
  protected:
    std::vector<size_t> reset_counts;
    std::vector<float> normalized_reset_counts;
    std::shared_ptr<MockLanguageModel> reset_lm;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<UniformSamplingStrategy> strategy;
    std::shared_ptr<MockCountNormalizer> reset_normalizer;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      strategy = std::make_shared<UniformSamplingStrategy>();

      reset_lm = std::make_shared<MockLanguageModel>();

      reset_counts = {5, 5, 2};
      normalized_reset_counts = {7./16., 7./16., 2./16.};

      reset_normalizer = std::make_shared<MockCountNormalizer>();
    }

    virtual void TearDown() { }
};

class EmpiricalSamplingStrategyTest: public ::testing::Test {
  protected:
    std::vector<size_t> reset_counts;
    std::vector<float> normalized_reset_counts;
    std::shared_ptr<MockLanguageModel> reset_lm;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<EmpiricalSamplingStrategy> strategy;
    std::shared_ptr<MockCountNormalizer> count_normalizer;
    std::shared_ptr<MockCountNormalizer> reset_normalizer;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      const std::vector<size_t> _counts = {2, 2, 3};
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));

      count_normalizer = std::make_shared<MockCountNormalizer>();
      const std::vector<float> _normalized_counts = {2./7., 2./7., 3./7.};
      EXPECT_CALL(*count_normalizer, normalize(_counts)).
        WillRepeatedly(Return(_normalized_counts));

      strategy = std::make_shared<EmpiricalSamplingStrategy>(count_normalizer,
                                                             5, 2);

      reset_lm = std::make_shared<MockLanguageModel>();

      reset_counts = {5, 5, 2};
      normalized_reset_counts = {7./16., 7./16., 2./16.};

      reset_normalizer = std::make_shared<MockCountNormalizer>();
    }

    virtual void TearDown() { }
};

class EmpiricalSamplingStrategySerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<EmpiricalSamplingStrategy> strategy;
    std::shared_ptr<CountNormalizer> count_normalizer;

    virtual void SetUp() {
      count_normalizer = std::make_shared<CountNormalizer>();
      strategy = std::make_shared<EmpiricalSamplingStrategy>(count_normalizer,
                                                             5, 2);
    }

    virtual void TearDown() { }
};

class SmoothedEmpiricalSamplingStrategyTest: public ::testing::Test {
  protected:
    float smoothing_exponent, smoothing_offset;
    std::vector<size_t> reset_counts;
    std::vector<float> normalized_reset_counts;
    std::shared_ptr<MockLanguageModel> reset_lm;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<EmpiricalSamplingStrategy> strategy;
    std::shared_ptr<MockCountNormalizer> count_normalizer;
    std::shared_ptr<MockCountNormalizer> reset_normalizer;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      const std::vector<size_t> _counts = {2, 2, 3};
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));

      smoothing_exponent = 0.75;
      smoothing_offset = 2.33;

      count_normalizer = std::make_shared<MockCountNormalizer>();
      const float w_0 = pow(smoothing_offset + 2, smoothing_exponent),
                   w_1 = pow(smoothing_offset + 2, smoothing_exponent),
                   w_2 = pow(smoothing_offset + 3, smoothing_exponent);
      const float z = w_0 + w_1 + w_2;
      const std::vector<float> _normalized_counts = {
        w_0 / z,
        w_1 / z,
        w_2 / z,
      };
      EXPECT_CALL(*count_normalizer, normalize(_counts)).
        WillRepeatedly(Return(_normalized_counts));

      strategy = std::make_shared<EmpiricalSamplingStrategy>(count_normalizer);

      reset_lm = std::make_shared<MockLanguageModel>();

      reset_counts = {5, 5, 2};
      normalized_reset_counts = {7./16., 7./16., 2./16.};

      reset_normalizer = std::make_shared<MockCountNormalizer>();
    }

    virtual void TearDown() { }
};

class ReservoirSamplingStrategyTest: public ::testing::Test {
  protected:
    std::vector<size_t> reset_counts;
    std::vector<float> normalized_reset_counts;
    std::shared_ptr<MockLanguageModel> reset_lm;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<ReservoirSamplingStrategy> strategy;
    std::shared_ptr<MockLongReservoirSampler> sampler;
    std::shared_ptr<MockCountNormalizer> reset_normalizer;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();

      sampler = std::make_shared<MockLongReservoirSampler>(9);

      strategy = std::make_shared<ReservoirSamplingStrategy>(sampler);

      reset_lm = std::make_shared<MockLanguageModel>();

      reset_counts = {5, 5, 2};
      normalized_reset_counts = {7./16., 7./16., 2./16.};

      reset_normalizer = std::make_shared<MockCountNormalizer>();
    }

    virtual void TearDown() { }
};

class ReservoirSamplingStrategySerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<ReservoirSamplingStrategy> strategy;
    std::shared_ptr<ReservoirSampler<long> > sampler;

    virtual void SetUp() {
      sampler = std::make_shared<ReservoirSampler<long> >(9);
      strategy = std::make_shared<ReservoirSamplingStrategy>(sampler);
    }

    virtual void TearDown() { }
};

class StaticContextStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<StaticContextStrategy> strategy;

    virtual void SetUp() {
      strategy = std::make_shared<StaticContextStrategy>(3);
    }

    virtual void TearDown() { }
};

class DynamicContextStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<DynamicContextStrategy> strategy;

    virtual void SetUp() {
      strategy = std::make_shared<DynamicContextStrategy>(3);
    }

    virtual void TearDown() { }
};

class LanguageModelExampleStoreTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<LanguageModelExampleStore<long> > example_store;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      example_store = std::make_shared<LanguageModelExampleStore<long> >(lm, 2);
    }

    virtual void TearDown() { }
};

class LanguageModelExampleStoreSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SpaceSavingLanguageModel> lm;
    std::shared_ptr<LanguageModelExampleStore<long> > example_store;

    virtual void SetUp() {
      lm = std::make_shared<SpaceSavingLanguageModel>(3);
      example_store = std::make_shared<LanguageModelExampleStore<long> >(lm, 2);
      example_store->increment("foo", 42L);
      example_store->increment("foo", 47L);
      example_store->increment("bar", 7L);
      example_store->increment("foo", 7L);
      example_store->increment("foo", 9L);
      example_store->increment("baz", 5L);
    }

    virtual void TearDown() { }
};


#endif
