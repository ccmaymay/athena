#ifndef ATHENA_SGNS_TEST_H
#define ATHENA_SGNS_TEST_H


#include "_core.h"
#include "_sgns.h"
#include "core_mock.h"
#include "sgns_mock.h"
#include "test_util.h"

#include <gtest/gtest.h>


using ::testing::Return;
using ::testing::ReturnRef;


class SGNSMockSGDTokenLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockSGD> sgd;
    std::shared_ptr<MockSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy, MockSGD> > token_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;

    virtual void SetUp() {
      sgd = std::make_shared<MockSGD>();
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      token_learner = std::make_shared<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy, MockSGD> >(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd);

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

class SGNSTokenLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;
    std::shared_ptr<MockSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy> > token_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(3, 2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      token_learner = std::make_shared<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy> >(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd);

      factorization->get_word_embedding(0)[0] = .1;
      factorization->get_word_embedding(0)[1] = -.2;

      factorization->get_word_embedding(1)[0] = -.3;
      factorization->get_word_embedding(1)[1] = .2;

      factorization->get_word_embedding(2)[0] = .4;
      factorization->get_word_embedding(2)[1] = 0;

      factorization->get_context_embedding(0)[0] = .4;
      factorization->get_context_embedding(0)[1] = 0;

      factorization->get_context_embedding(1)[0] = -.3;
      factorization->get_context_embedding(1)[1] = .2;

      factorization->get_context_embedding(2)[0] = .1;
      factorization->get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};

class SGNSTokenLearnerSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;
    std::shared_ptr<EmpiricalSamplingStrategy<NaiveLanguageModel> > neg_sampling_strategy;
    std::shared_ptr<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > token_learner;
    std::shared_ptr<NaiveLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<ExponentCountNormalizer> count_normalizer;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(19, 23, 0.5, 0.1);
      lm = std::make_shared<NaiveLanguageModel>();
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      count_normalizer = std::make_shared<ExponentCountNormalizer>();
      neg_sampling_strategy = std::make_shared<EmpiricalSamplingStrategy<NaiveLanguageModel> >(count_normalizer, 7, 11);
      token_learner = std::make_shared<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > >(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd);

      factorization->get_word_embedding(0)[0] = .1;
      factorization->get_word_embedding(0)[1] = -.2;

      factorization->get_word_embedding(1)[0] = -.3;
      factorization->get_word_embedding(1)[1] = .2;

      factorization->get_word_embedding(2)[0] = .4;
      factorization->get_word_embedding(2)[1] = 0;

      factorization->get_context_embedding(0)[0] = .4;
      factorization->get_context_embedding(0)[1] = 0;

      factorization->get_context_embedding(1)[0] = -.3;
      factorization->get_context_embedding(1)[1] = .2;

      factorization->get_context_embedding(2)[0] = .1;
      factorization->get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};

class SGNSSentenceLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<MockSGNSTokenLearner> token_learner;
    std::shared_ptr<SGNSSentenceLearner<MockSGNSTokenLearner, MockContextStrategy> > sentence_learner;

    virtual void SetUp() {
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<MockSGNSTokenLearner>();
      sentence_learner = std::make_shared<SGNSSentenceLearner<MockSGNSTokenLearner, MockContextStrategy> >(token_learner, ctx_strategy, 5);
    }

    virtual void TearDown() { }
};

class SGNSSentenceLearnerSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;
    std::shared_ptr<EmpiricalSamplingStrategy<NaiveLanguageModel> > neg_sampling_strategy;
    std::shared_ptr<DynamicContextStrategy> ctx_strategy;
    std::shared_ptr<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > token_learner;
    std::shared_ptr<NaiveLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<ExponentCountNormalizer> count_normalizer;
    std::shared_ptr<SGNSSentenceLearner<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > > sentence_learner;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(19, 23, 0.5, 0.1);
      lm = std::make_shared<NaiveLanguageModel>();
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      count_normalizer = std::make_shared<ExponentCountNormalizer>();
      neg_sampling_strategy = std::make_shared<EmpiricalSamplingStrategy<NaiveLanguageModel> >(count_normalizer, 7, 11);
      ctx_strategy = std::make_shared<DynamicContextStrategy>(13);
      token_learner = std::make_shared<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > >(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd);
      sentence_learner = std::make_shared<SGNSSentenceLearner<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > >(token_learner, ctx_strategy, 5);

      factorization->get_word_embedding(0)[0] = .1;
      factorization->get_word_embedding(0)[1] = -.2;

      factorization->get_word_embedding(1)[0] = -.3;
      factorization->get_word_embedding(1)[1] = .2;

      factorization->get_word_embedding(2)[0] = .4;
      factorization->get_word_embedding(2)[1] = 0;

      factorization->get_context_embedding(0)[0] = .4;
      factorization->get_context_embedding(0)[1] = 0;

      factorization->get_context_embedding(1)[0] = -.3;
      factorization->get_context_embedding(1)[1] = .2;

      factorization->get_context_embedding(2)[0] = .1;
      factorization->get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};


#endif
