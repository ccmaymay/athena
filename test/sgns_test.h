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
    std::shared_ptr<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy, MockSGD> > token_learner;

    virtual void SetUp() {
      token_learner = std::make_shared<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy, MockSGD> >(
        WordContextFactorization(3, 2),
        MockSamplingStrategy(),
        MockLanguageModel(),
        MockSGD());
      EXPECT_CALL(token_learner->language_model, size()).WillRepeatedly(Return(3));

      token_learner->factorization.get_word_embedding(0)[0] = 1;
      token_learner->factorization.get_word_embedding(0)[1] = -2;

      token_learner->factorization.get_word_embedding(1)[0] = -3;
      token_learner->factorization.get_word_embedding(1)[1] = 2;

      token_learner->factorization.get_word_embedding(2)[0] = 4;
      token_learner->factorization.get_word_embedding(2)[1] = 0;

      token_learner->factorization.get_context_embedding(0)[0] = 4;
      token_learner->factorization.get_context_embedding(0)[1] = 0;

      token_learner->factorization.get_context_embedding(1)[0] = -3;
      token_learner->factorization.get_context_embedding(1)[1] = 2;

      token_learner->factorization.get_context_embedding(2)[0] = 1;
      token_learner->factorization.get_context_embedding(2)[1] = -2;
    }

    virtual void TearDown() { }
};

class SGNSTokenLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy> > token_learner;

    virtual void SetUp() {
      token_learner = std::make_shared<SGNSTokenLearner<MockLanguageModel, MockSamplingStrategy> >(
        WordContextFactorization(3, 2),
        MockSamplingStrategy(),
        MockLanguageModel(),
        SGD(3, 2, 0.5, 0.1));
      EXPECT_CALL(token_learner->language_model, size()).WillRepeatedly(Return(3));

      token_learner->factorization.get_word_embedding(0)[0] = .1;
      token_learner->factorization.get_word_embedding(0)[1] = -.2;

      token_learner->factorization.get_word_embedding(1)[0] = -.3;
      token_learner->factorization.get_word_embedding(1)[1] = .2;

      token_learner->factorization.get_word_embedding(2)[0] = .4;
      token_learner->factorization.get_word_embedding(2)[1] = 0;

      token_learner->factorization.get_context_embedding(0)[0] = .4;
      token_learner->factorization.get_context_embedding(0)[1] = 0;

      token_learner->factorization.get_context_embedding(1)[0] = -.3;
      token_learner->factorization.get_context_embedding(1)[1] = .2;

      token_learner->factorization.get_context_embedding(2)[0] = .1;
      token_learner->factorization.get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};

class SGNSTokenLearnerSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > token_learner;

    virtual void SetUp() {
      token_learner = std::make_shared<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > >(
        WordContextFactorization(3, 2),
        EmpiricalSamplingStrategy<NaiveLanguageModel>(ExponentCountNormalizer(), 7, 11),
        NaiveLanguageModel(),
        SGD(19, 23, 0.5, 0.1));

      token_learner->factorization.get_word_embedding(0)[0] = .1;
      token_learner->factorization.get_word_embedding(0)[1] = -.2;

      token_learner->factorization.get_word_embedding(1)[0] = -.3;
      token_learner->factorization.get_word_embedding(1)[1] = .2;

      token_learner->factorization.get_word_embedding(2)[0] = .4;
      token_learner->factorization.get_word_embedding(2)[1] = 0;

      token_learner->factorization.get_context_embedding(0)[0] = .4;
      token_learner->factorization.get_context_embedding(0)[1] = 0;

      token_learner->factorization.get_context_embedding(1)[0] = -.3;
      token_learner->factorization.get_context_embedding(1)[1] = .2;

      token_learner->factorization.get_context_embedding(2)[0] = .1;
      token_learner->factorization.get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};

class SGNSSentenceLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGNSSentenceLearner<MockSGNSTokenLearner, MockContextStrategy> > sentence_learner;

    virtual void SetUp() {
      sentence_learner = std::make_shared<SGNSSentenceLearner<MockSGNSTokenLearner, MockContextStrategy> >(MockSGNSTokenLearner(), MockContextStrategy(), 5);
    }

    virtual void TearDown() { }
};

class SGNSSentenceLearnerSerializationTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGNSSentenceLearner<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > > sentence_learner;

    virtual void SetUp() {
      sentence_learner = std::make_shared<SGNSSentenceLearner<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > > >(
        SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> >(
          WordContextFactorization(3, 2),
          EmpiricalSamplingStrategy<NaiveLanguageModel>(ExponentCountNormalizer(), 7, 11),
          NaiveLanguageModel(),
          SGD(19, 23, 0.5, 0.1)),
        DynamicContextStrategy(13),
        5);

      sentence_learner->token_learner.factorization.get_word_embedding(0)[0] = .1;
      sentence_learner->token_learner.factorization.get_word_embedding(0)[1] = -.2;

      sentence_learner->token_learner.factorization.get_word_embedding(1)[0] = -.3;
      sentence_learner->token_learner.factorization.get_word_embedding(1)[1] = .2;

      sentence_learner->token_learner.factorization.get_word_embedding(2)[0] = .4;
      sentence_learner->token_learner.factorization.get_word_embedding(2)[1] = 0;

      sentence_learner->token_learner.factorization.get_context_embedding(0)[0] = .4;
      sentence_learner->token_learner.factorization.get_context_embedding(0)[1] = 0;

      sentence_learner->token_learner.factorization.get_context_embedding(1)[0] = -.3;
      sentence_learner->token_learner.factorization.get_context_embedding(1)[1] = .2;

      sentence_learner->token_learner.factorization.get_context_embedding(2)[0] = .1;
      sentence_learner->token_learner.factorization.get_context_embedding(2)[1] = -.2;
    }

    virtual void TearDown() { }
};


#endif
