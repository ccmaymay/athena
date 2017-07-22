#include "sgns_test.h"
#include "test_util.h"
#include "sgns_mock.h"
#include "core_mock.h"
#include "_core.h"
#include "_sgns.h"
#include "_math.h"

#include <gtest/gtest.h>
#include <utility>
#include <sstream>


using namespace std;
using ::testing::Return;
using ::testing::Ref;
using ::testing::InSequence;
using ::testing::_;


TEST_F(SGNSMockSGDTokenLearnerTest, reset_word) {
  EXPECT_CALL(token_learner->sgd, reset(1));

  token_learner->reset_word(1);

  EXPECT_EQ(token_learner->factorization.get_word_embedding(0)[0], 1);
  EXPECT_EQ(token_learner->factorization.get_word_embedding(0)[1], -2);

  EXPECT_NE(token_learner->factorization.get_word_embedding(1)[0], -3);
  EXPECT_NE(token_learner->factorization.get_word_embedding(1)[1], 2);

  EXPECT_EQ(token_learner->factorization.get_word_embedding(2)[0], 4);
  EXPECT_EQ(token_learner->factorization.get_word_embedding(2)[1], 0);

  EXPECT_EQ(token_learner->factorization.get_context_embedding(0)[0], 4);
  EXPECT_EQ(token_learner->factorization.get_context_embedding(0)[1], 0);

  EXPECT_NE(token_learner->factorization.get_context_embedding(1)[0], -3);
  EXPECT_NE(token_learner->factorization.get_context_embedding(1)[1], 2);

  EXPECT_EQ(token_learner->factorization.get_context_embedding(2)[0], 1);
  EXPECT_EQ(token_learner->factorization.get_context_embedding(2)[1], -2);
}

TEST_F(SGNSTokenLearnerTest, context_contains_oov) {
  const long word_ids[] = {0, 1, 1, 2, -1, 1, 2, 1, 2};
  EXPECT_FALSE(token_learner->context_contains_oov(word_ids, 4));
  EXPECT_TRUE(token_learner->context_contains_oov(word_ids + 1, 4));
  EXPECT_TRUE(token_learner->context_contains_oov(word_ids + 2, 4));
  EXPECT_TRUE(token_learner->context_contains_oov(word_ids + 3, 4));
  EXPECT_TRUE(token_learner->context_contains_oov(word_ids + 4, 4));
  EXPECT_FALSE(token_learner->context_contains_oov(word_ids + 5, 4));
}

TEST_F(SGNSTokenLearnerTest, compute_gradient_coeff) {
  EXPECT_NEAR(1 - sigmoid(.1 * .4 + (-.2) * 0),
              token_learner->compute_gradient_coeff(0, 0, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid(.1 * (-.3) + (-.2) * .2),
              token_learner->compute_gradient_coeff(0, 1, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid(.1 * .1 + (-.2) * (-.2)),
              token_learner->compute_gradient_coeff(0, 2, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid((-.3) * .4 + .2 * 0),
              token_learner->compute_gradient_coeff(1, 0, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid((-.3) * (-.3) + .2 * .2),
              token_learner->compute_gradient_coeff(1, 1, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid((-.3) * .1 + .2 * (-.2)),
              token_learner->compute_gradient_coeff(1, 2, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid(.4 * .4 + 0 * 0),
              token_learner->compute_gradient_coeff(2, 0, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid(.4 * (-.3) + 0 * .2),
              token_learner->compute_gradient_coeff(2, 1, false), FAST_EPS);
  EXPECT_NEAR(1 - sigmoid(.4 * .1 + 0 * (-.2)),
              token_learner->compute_gradient_coeff(2, 2, false), FAST_EPS);
}

TEST_F(SGNSTokenLearnerTest, compute_gradient_coeff_negative) {
  EXPECT_NEAR(0 - sigmoid(.1 * .4 + (-.2) * 0),
              token_learner->compute_gradient_coeff(0, 0, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid(.1 * (-.3) + (-.2) * .2),
              token_learner->compute_gradient_coeff(0, 1, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid(.1 * .1 + (-.2) * (-.2)),
              token_learner->compute_gradient_coeff(0, 2, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid((-.3) * .4 + .2 * 0),
              token_learner->compute_gradient_coeff(1, 0, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid((-.3) * (-.3) + .2 * .2),
              token_learner->compute_gradient_coeff(1, 1, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid((-.3) * .1 + .2 * (-.2)),
              token_learner->compute_gradient_coeff(1, 2, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid(.4 * .4 + 0 * 0),
              token_learner->compute_gradient_coeff(2, 0, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid(.4 * (-.3) + 0 * .2),
              token_learner->compute_gradient_coeff(2, 1, true), FAST_EPS);
  EXPECT_NEAR(0 - sigmoid(.4 * .1 + 0 * (-.2)),
              token_learner->compute_gradient_coeff(2, 2, true), FAST_EPS);
}

TEST_F(SGNSTokenLearnerTest, token_train_neg1) {
  const float
    rho0 = token_learner->sgd.get_rho(0),
    rho1 = token_learner->sgd.get_rho(1),
    rho2 = token_learner->sgd.get_rho(2);
  const float rho = rho0;
  const float adj0 = 1;

  InSequence in_sequence;
  EXPECT_CALL(token_learner->neg_sampling_strategy,
    sample_idx(Ref(token_learner->language_model))).WillOnce(Return(0l));

  token_learner->token_train(2, 1, 1);

  EXPECT_NEAR(.1, token_learner->factorization.get_word_embedding(0)[0], EPS);
  EXPECT_NEAR(-.2, token_learner->factorization.get_word_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3, token_learner->factorization.get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, token_learner->factorization.get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * (-.3) +
                  rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * .4,
              token_learner->factorization.get_word_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * .2 +
                  rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * 0,
              token_learner->factorization.get_word_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(.4 + rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * .4,
              token_learner->factorization.get_context_embedding(0)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * 0,
              token_learner->factorization.get_context_embedding(0)[1], FAST_EPS);
  EXPECT_NEAR(-.3 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * .4,
              token_learner->factorization.get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * 0,
              token_learner->factorization.get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1, token_learner->factorization.get_context_embedding(2)[0], EPS);
  EXPECT_NEAR(-.2, token_learner->factorization.get_context_embedding(2)[1], EPS);

  EXPECT_NEAR(token_learner->sgd.get_rho(0), rho0, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(1), rho1, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(2), rho2, EPS);
}

TEST_F(SGNSTokenLearnerTest, token_train_neg1_partial_vocab_coverage) {
  const float
    rho0 = token_learner->sgd.get_rho(0),
    rho1 = token_learner->sgd.get_rho(1),
    rho2 = token_learner->sgd.get_rho(2);
  const float rho = rho0;
  const float adj2 = 1;

  InSequence in_sequence;
  EXPECT_CALL(token_learner->neg_sampling_strategy,
    sample_idx(Ref(token_learner->language_model))).WillOnce(Return(2l));

  token_learner->token_train(2, 1, 1);

  EXPECT_NEAR(.1, token_learner->factorization.get_word_embedding(0)[0], EPS);
  EXPECT_NEAR(-.2, token_learner->factorization.get_word_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3, token_learner->factorization.get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, token_learner->factorization.get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * (-.3) +
                  rho * adj2 * (0 - sigmoid(.4 * .1 + 0 * (-.2))) * .1,
              token_learner->factorization.get_word_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * .2 +
                  rho * adj2 * (0 - sigmoid(.4 * .1 + 0 * (-.2))) * (-.2),
              token_learner->factorization.get_word_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(.4, token_learner->factorization.get_context_embedding(0)[0], EPS);
  EXPECT_NEAR(0, token_learner->factorization.get_context_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * .4,
              token_learner->factorization.get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * 0,
              token_learner->factorization.get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1 + rho * adj2 * (0 - sigmoid(.1 * .4 + (-.2) * 0)) * .4,
              token_learner->factorization.get_context_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * adj2 * (0 - sigmoid(.1 * .4 + (-.2) * 0)) * 0,
              token_learner->factorization.get_context_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(token_learner->sgd.get_rho(0), rho0, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(1), rho1, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(2), rho2, EPS);
}

TEST_F(SGNSTokenLearnerTest, token_train_neg2) {
  const float
    rho0 = token_learner->sgd.get_rho(0),
    rho1 = token_learner->sgd.get_rho(1),
    rho2 = token_learner->sgd.get_rho(2);
  const float rho = rho0;
  const float adj2 = 1, adj1 = 1;

  InSequence in_sequence;
  EXPECT_CALL(token_learner->neg_sampling_strategy,
    sample_idx(Ref(token_learner->language_model))).WillOnce(Return(2l)).
                  WillOnce(Return(1l));

  token_learner->token_train(0, 1, 2);

  const float w0c1_coeff = (1 - sigmoid(.1 * (-.3) + (-.2) * .2));

  EXPECT_NEAR(.1 + rho * w0c1_coeff * (-.3) +
                  rho * adj2 * (0 - sigmoid(.1 * .1 + (-.2) * (-.2))) * .1 +
                  rho * adj1 * (0 - sigmoid(
                    .1 * (-.3 + rho * w0c1_coeff * .1) +
                    (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                  )) * (-.3 + rho * w0c1_coeff * .1),
              token_learner->factorization.get_word_embedding(0)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * w0c1_coeff * .2 +
                   rho * adj2 * (0 - sigmoid(
                     .1 * .1 + (-.2) * (-.2)
                   )) * (-.2) +
                   rho * adj1 * (0 - sigmoid(
                     .1 * (-.3 + rho * w0c1_coeff * .1) +
                     (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                   )) * (.2 + rho * w0c1_coeff * (-.2)),
              token_learner->factorization.get_word_embedding(0)[1], FAST_EPS);
  EXPECT_NEAR(-.3, token_learner->factorization.get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, token_learner->factorization.get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4, token_learner->factorization.get_word_embedding(2)[0], EPS);
  EXPECT_NEAR(0, token_learner->factorization.get_word_embedding(2)[1], EPS);

  EXPECT_NEAR(.4, token_learner->factorization.get_context_embedding(0)[0], EPS);
  EXPECT_NEAR(0, token_learner->factorization.get_context_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3 + rho * w0c1_coeff * .1 +
                   rho * adj1 * (0 - sigmoid(
                     .1 * (-.3 + rho * w0c1_coeff * .1) +
                     (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                   )) * .1,
              token_learner->factorization.get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * w0c1_coeff * (-.2) +
                  rho * adj1 * (0 - sigmoid(
                    .1 * (-.3 + rho * w0c1_coeff * .1) +
                    (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                  )) * (-.2),
              token_learner->factorization.get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1 + rho * adj2 * (0 - sigmoid(.1 * .1 + (-.2) * (-.2))) * .1,
              token_learner->factorization.get_context_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * adj2 * (0 - sigmoid(
                .1 * .1 + (-.2) * (-.2)
              )) * (-.2),
              token_learner->factorization.get_context_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(token_learner->sgd.get_rho(0), rho0, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(1), rho1, EPS);
  EXPECT_NEAR(token_learner->sgd.get_rho(2), rho2, EPS);
}

TEST_F(SGNSTokenLearnerTest, compute_similarity) {
  EXPECT_NEAR(1, token_learner->compute_similarity(0, 0), EPS);
  EXPECT_NEAR(-0.8682431, token_learner->compute_similarity(0, 1), EPS);
  EXPECT_NEAR(0.4472136, token_learner->compute_similarity(0, 2), EPS);
  EXPECT_NEAR(-0.8682431, token_learner->compute_similarity(1, 0), EPS);
  EXPECT_NEAR(1, token_learner->compute_similarity(1, 1), EPS);
  EXPECT_NEAR(-0.8320503, token_learner->compute_similarity(1, 2), EPS);
  EXPECT_NEAR(0.4472136, token_learner->compute_similarity(2, 0), EPS);
  EXPECT_NEAR(-0.8320503, token_learner->compute_similarity(2, 1), EPS);
  EXPECT_NEAR(1, token_learner->compute_similarity(2, 2), EPS);
}

TEST_F(SGNSTokenLearnerTest, find_nearest_neighbor_idx) {
  EXPECT_EQ(2, token_learner->find_nearest_neighbor_idx(0));
  EXPECT_EQ(2, token_learner->find_nearest_neighbor_idx(1));
  EXPECT_EQ(0, token_learner->find_nearest_neighbor_idx(2));
}

TEST_F(SGNSTokenLearnerTest, find_context_nearest_neighbor_idx_left1_right0) {
  const long context0[] = {0, -1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(1, 0, context0));
  const long context1[] = {1, -1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(1, 0, context1));
  const long context2[] = {2, -1};
  EXPECT_EQ(0, token_learner->find_context_nearest_neighbor_idx(1, 0, context2));
}

TEST_F(SGNSTokenLearnerTest, find_context_nearest_neighbor_idx_left0_right1) {
  const long context0[] = {-1, 0};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(0, 1, context0));
  const long context1[] = {-1, 1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(0, 1, context1));
  const long context2[] = {-1, 2};
  EXPECT_EQ(0, token_learner->find_context_nearest_neighbor_idx(0, 1, context2));
}

TEST_F(SGNSTokenLearnerTest, find_context_nearest_neighbor_idx_left2_right0) {
  const long context0[] = {0, 0, -1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(2, 0, context0));
  const long context1[] = {0, 1, -1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(2, 0, context1));
  const long context2[] = {0, 2, -1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(2, 0, context2));
  const long context3[] = {1, 1, -1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(2, 0, context3));
  const long context4[] = {1, 2, -1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(2, 0, context4));
  const long context5[] = {2, 2, -1};
  EXPECT_EQ(0, token_learner->find_context_nearest_neighbor_idx(2, 0, context5));
}

TEST_F(SGNSTokenLearnerTest, find_context_nearest_neighbor_idx_left1_right1) {
  const long context0[] = {0, -1, 0};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(1, 1, context0));
  const long context1[] = {0, -1, 1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(1, 1, context1));
  const long context2[] = {0, -1, 2};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(1, 1, context2));
  const long context3[] = {1, -1, 1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(1, 1, context3));
  const long context4[] = {1, -1, 2};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(1, 1, context4));
  const long context5[] = {2, -1, 2};
  EXPECT_EQ(0, token_learner->find_context_nearest_neighbor_idx(1, 1, context5));
}

TEST_F(SGNSTokenLearnerTest, find_context_nearest_neighbor_idx_left0_right2) {
  const long context0[] = {-1, 0, 0};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(0, 2, context0));
  const long context1[] = {-1, 0, 1};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(0, 2, context1));
  const long context2[] = {-1, 0, 2};
  EXPECT_EQ(2, token_learner->find_context_nearest_neighbor_idx(0, 2, context2));
  const long context3[] = {-1, 1, 1};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(0, 2, context3));
  const long context4[] = {-1, 1, 2};
  EXPECT_EQ(1, token_learner->find_context_nearest_neighbor_idx(0, 2, context4));
  const long context5[] = {-1, 2, 2};
  EXPECT_EQ(0, token_learner->find_context_nearest_neighbor_idx(0, 2, context5));
}

TEST_F(SGNSTokenLearnerSerializationTest, serialization_fixed_point) {
  stringstream ostream;
  token_learner->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> >::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(token_learner->equals(from_stream));
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_zero) {
  vector<long> words;

  EXPECT_CALL(sentence_learner->ctx_strategy, size(_, _)).Times(0);
  EXPECT_CALL(sentence_learner->token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(sentence_learner->token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_one) {
  vector<long> words;
  words.push_back(2L);

  InSequence in_sequence;

  EXPECT_CALL(sentence_learner->ctx_strategy, size(0, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(sentence_learner->token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_empty_context) {
  vector<long> words;
  words.push_back(0L);
  words.push_back(2L);
  words.push_back(1L);

  InSequence in_sequence;

  EXPECT_CALL(sentence_learner->ctx_strategy, size(0, 2)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(1, 1)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(2, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(sentence_learner->token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_short) {
  vector<long> words;
  words.push_back(0L);
  words.push_back(2L);
  words.push_back(1L);

  InSequence in_sequence;

  EXPECT_CALL(sentence_learner->ctx_strategy, size(0, 2)).
    WillOnce(Return(make_pair(size_t(0), size_t(2))));
  EXPECT_CALL(sentence_learner->token_learner, token_train(0, 2, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(0, 1, 5));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(1, 1)).
    WillOnce(Return(make_pair(size_t(1), size_t(1))));
  EXPECT_CALL(sentence_learner->token_learner, token_train(2, 0, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(2, 1, 5));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(2, 0)).
    WillOnce(Return(make_pair(size_t(2), size_t(0))));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 0, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 2, 5));

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_many) {
  vector<long> words;
  words.push_back(2L);
  words.push_back(1L);
  words.push_back(1L);
  words.push_back(0L);
  words.push_back(0L);

  InSequence in_sequence;

  EXPECT_CALL(sentence_learner->ctx_strategy, size(0, 4)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(1, 3)).
    WillOnce(Return(make_pair(size_t(1), size_t(2))));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 2, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 1, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 0, 5));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(2, 2)).
    WillOnce(Return(make_pair(size_t(1), size_t(2))));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 1, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 0, 5));
  EXPECT_CALL(sentence_learner->token_learner, token_train(1, 0, 5));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(3, 1)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(sentence_learner->ctx_strategy, size(4, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerSerializationTest, serialization_fixed_point) {
  stringstream ostream;
  sentence_learner->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGNSSentenceLearner<SGNSTokenLearner<NaiveLanguageModel, EmpiricalSamplingStrategy<NaiveLanguageModel> > >::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(sentence_learner->equals(from_stream));
}
