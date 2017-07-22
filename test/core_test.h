#ifndef ATHENA_CORE_TEST_H
#define ATHENA_CORE_TEST_H


#include "_core.h"
#include <cmath>
#include <memory>
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

class OneDimSGDTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(1, 100, 0.5, 0.1);
    }

    virtual void TearDown() { }
};

class ThreeDimSGDTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(3, 100, 0.5, 0.1);
    }

    virtual void TearDown() { }
};

class SGDSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(3, 100, 0.5, 0.1);
      sgd->step(0);
      sgd->step(2);
      sgd->step(2);
    }

    virtual void TearDown() { }
};

class UniformSamplingStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<UniformSamplingStrategy<MockLanguageModel> > strategy;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      strategy = std::make_shared<UniformSamplingStrategy<MockLanguageModel> >();
    }

    virtual void TearDown() { }
};

class EmpiricalSamplingStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<EmpiricalSamplingStrategy<MockLanguageModel, MockCountNormalizer> > strategy;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      const std::vector<size_t> _counts = {2, 2, 3};
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));

      strategy = std::make_shared<EmpiricalSamplingStrategy<MockLanguageModel, MockCountNormalizer> >(MockCountNormalizer(), 5, 2);

      const std::vector<float> _normalized_counts = {2./7., 2./7., 3./7.};
      EXPECT_CALL(strategy->normalizer, normalize(_counts)).
        WillRepeatedly(Return(_normalized_counts));
    }

    virtual void TearDown() { }
};

class EmpiricalSamplingStrategySerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<EmpiricalSamplingStrategy<MockLanguageModel> > strategy;

    virtual void SetUp() {
      strategy = std::make_shared<EmpiricalSamplingStrategy<MockLanguageModel> >(ExponentCountNormalizer(0.8, 8), 5, 2);
    }

    virtual void TearDown() { }
};

class SmoothedEmpiricalSamplingStrategyTest: public ::testing::Test {
  protected:
    float smoothing_exponent, smoothing_offset;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<EmpiricalSamplingStrategy<MockLanguageModel, MockCountNormalizer> > strategy;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();
      const std::vector<size_t> _counts = {2, 2, 3};
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      EXPECT_CALL(*lm, counts()).WillRepeatedly(Return(_counts));

      smoothing_exponent = 0.75;
      smoothing_offset = 2.33;

      const float w_0 = pow(smoothing_offset + 2, smoothing_exponent),
                   w_1 = pow(smoothing_offset + 2, smoothing_exponent),
                   w_2 = pow(smoothing_offset + 3, smoothing_exponent);
      const float z = w_0 + w_1 + w_2;
      const std::vector<float> _normalized_counts = {
        w_0 / z,
        w_1 / z,
        w_2 / z,
      };

      strategy = std::make_shared<EmpiricalSamplingStrategy<MockLanguageModel, MockCountNormalizer> >(MockCountNormalizer());
      EXPECT_CALL(strategy->normalizer, normalize(_counts)).
        WillRepeatedly(Return(_normalized_counts));
    }

    virtual void TearDown() { }
};

class ReservoirSamplingStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<ReservoirSamplingStrategy<MockLanguageModel, MockLongReservoirSampler> > strategy;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();

      strategy = std::make_shared<ReservoirSamplingStrategy<MockLanguageModel, MockLongReservoirSampler> >(MockLongReservoirSampler());
    }

    virtual void TearDown() { }
};

class ReservoirSamplingStrategySerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<ReservoirSamplingStrategy<MockLanguageModel> > strategy;

    virtual void SetUp() {
      strategy = std::make_shared<ReservoirSamplingStrategy<MockLanguageModel> >(ReservoirSampler<long>(9));
    }

    virtual void TearDown() { }
};

class DiscreteSamplingStrategyTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<DiscreteSamplingStrategy<MockLanguageModel, MockDiscretization> > strategy;

    virtual void SetUp() {
      lm = std::make_shared<MockLanguageModel>();

      strategy = std::make_shared<DiscreteSamplingStrategy<MockLanguageModel, MockDiscretization> >(MockDiscretization());
    }

    virtual void TearDown() { }
};

class DiscreteSamplingStrategySerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<DiscreteSamplingStrategy<MockLanguageModel> > strategy;

    virtual void SetUp() {
      const std::vector<float> _normalized_counts = {2./7., 2./7., 3./7.};
      strategy = std::make_shared<DiscreteSamplingStrategy<MockLanguageModel> >(Discretization(_normalized_counts, 9));
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


#endif
