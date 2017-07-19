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
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<SGNSTokenLearner> token_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<MockSGD>(2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<SGNSTokenLearner>();
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        std::shared_ptr<SGNSSentenceLearner>(),
        std::shared_ptr<SubsamplingSGNSSentenceLearner>());
      token_learner->set_model(model);

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
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<SGNSTokenLearner> token_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<SGNSTokenLearner>();
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        std::shared_ptr<SGNSSentenceLearner>(),
        std::shared_ptr<SubsamplingSGNSSentenceLearner>());
      token_learner->set_model(model);

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
    std::shared_ptr<MockSGD> sgd;
    std::shared_ptr<MockSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<MockSGNSTokenLearner> token_learner;
    std::shared_ptr<SGNSSentenceLearner> sentence_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<MockSGD>(2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<MockSGNSTokenLearner>();
      sentence_learner = std::make_shared<SGNSSentenceLearner>(5, true);
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        sentence_learner,
        std::shared_ptr<SubsamplingSGNSSentenceLearner>());
      sentence_learner->set_model(model);

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

class NonPropagatingSubsamplingSGNSSentenceLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockSGD> sgd;
    std::shared_ptr<MockSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<MockSGNSTokenLearner> token_learner;
    std::shared_ptr<MockSGNSSentenceLearner> sentence_learner;
    std::shared_ptr<SubsamplingSGNSSentenceLearner>
      subsampling_sentence_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<MockSGD>(2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<MockSGNSTokenLearner>();
      sentence_learner = std::make_shared<MockSGNSSentenceLearner>();
      subsampling_sentence_learner = std::make_shared<SubsamplingSGNSSentenceLearner>(false);
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        sentence_learner,
        subsampling_sentence_learner);
      subsampling_sentence_learner->set_model(model);
    }

    virtual void TearDown() { }
};

class SubsamplingSGNSSentenceLearnerTest: public ::testing::Test {
  protected:
    std::shared_ptr<MockSGD> sgd;
    std::shared_ptr<MockSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<MockContextStrategy> ctx_strategy;
    std::shared_ptr<MockSGNSTokenLearner> token_learner;
    std::shared_ptr<MockSGNSSentenceLearner> sentence_learner;
    std::shared_ptr<SubsamplingSGNSSentenceLearner>
      subsampling_sentence_learner;
    std::shared_ptr<MockLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<MockSGD>(2, 0.5, 0.1);
      lm = std::make_shared<MockLanguageModel>();
      EXPECT_CALL(*lm, size()).WillRepeatedly(Return(3));
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      neg_sampling_strategy = std::make_shared<MockSamplingStrategy>();
      ctx_strategy = std::make_shared<MockContextStrategy>();
      token_learner = std::make_shared<MockSGNSTokenLearner>();
      sentence_learner = std::make_shared<MockSGNSSentenceLearner>();
      subsampling_sentence_learner = std::make_shared<SubsamplingSGNSSentenceLearner>(true);
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        sentence_learner,
        subsampling_sentence_learner);
      subsampling_sentence_learner->set_model(model);
    }

    virtual void TearDown() { }
};

class SGNSModelTest: public ::testing::Test {
  protected:
    std::shared_ptr<SGD> sgd;
    std::shared_ptr<ReservoirSamplingStrategy> neg_sampling_strategy;
    std::shared_ptr<ReservoirSampler<long> > reservoir_sampler;
    std::shared_ptr<DynamicContextStrategy> ctx_strategy;
    std::shared_ptr<SGNSTokenLearner> token_learner;
    std::shared_ptr<SGNSSentenceLearner> sentence_learner;
    std::shared_ptr<SubsamplingSGNSSentenceLearner>
      subsampling_sentence_learner;
    std::shared_ptr<SpaceSavingLanguageModel> lm;
    std::shared_ptr<WordContextFactorization> factorization;
    std::shared_ptr<SGNSModel> model;

    virtual void SetUp() {
      sgd = std::make_shared<SGD>(2, 0.5, 0.1);
      lm = std::make_shared<SpaceSavingLanguageModel>();
      factorization = std::make_shared<WordContextFactorization>(3, 2);
      reservoir_sampler = std::make_shared<ReservoirSampler<long> >(7);
      neg_sampling_strategy =
        std::make_shared<ReservoirSamplingStrategy>(reservoir_sampler);
      ctx_strategy = std::make_shared<DynamicContextStrategy>(3);
      token_learner = std::make_shared<SGNSTokenLearner>();
      sentence_learner = std::make_shared<SGNSSentenceLearner>(5, true);
      subsampling_sentence_learner =
        std::make_shared<SubsamplingSGNSSentenceLearner>(true);
      model = std::make_shared<SGNSModel>(
        factorization,
        neg_sampling_strategy,
        lm,
        sgd,
        ctx_strategy,
        token_learner,
        sentence_learner,
        subsampling_sentence_learner);
      token_learner->set_model(model);
      sentence_learner->set_model(model);
      subsampling_sentence_learner->set_model(model);
    }

    virtual void TearDown() { }
};


#endif
