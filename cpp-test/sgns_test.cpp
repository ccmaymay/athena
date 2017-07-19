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
  token_learner->reset_word(1);

  EXPECT_EQ(factorization->get_word_embedding(0)[0], 1);
  EXPECT_EQ(factorization->get_word_embedding(0)[1], -2);

  EXPECT_NE(factorization->get_word_embedding(1)[0], -3);
  EXPECT_NE(factorization->get_word_embedding(1)[1], 2);

  EXPECT_EQ(factorization->get_word_embedding(2)[0], 4);
  EXPECT_EQ(factorization->get_word_embedding(2)[1], 0);

  EXPECT_EQ(factorization->get_context_embedding(0)[0], 4);
  EXPECT_EQ(factorization->get_context_embedding(0)[1], 0);

  EXPECT_NE(factorization->get_context_embedding(1)[0], -3);
  EXPECT_NE(factorization->get_context_embedding(1)[1], 2);

  EXPECT_EQ(factorization->get_context_embedding(2)[0], 1);
  EXPECT_EQ(factorization->get_context_embedding(2)[1], -2);
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
    rho = sgd->get_rho();
  const float adj0 = 1;

  InSequence in_sequence;
  EXPECT_CALL(*neg_sampling_strategy,
    sample_idx(Ref(*lm))).WillOnce(Return(0l));

  token_learner->token_train(2, 1, 1);

  EXPECT_NEAR(.1, factorization->get_word_embedding(0)[0], EPS);
  EXPECT_NEAR(-.2, factorization->get_word_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3, factorization->get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, factorization->get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * (-.3) +
                  rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * .4,
              factorization->get_word_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * .2 +
                  rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * 0,
              factorization->get_word_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(.4 + rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * .4,
              factorization->get_context_embedding(0)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * adj0 * (0 - sigmoid(.4 * .4 + 0 * 0)) * 0,
              factorization->get_context_embedding(0)[1], FAST_EPS);
  EXPECT_NEAR(-.3 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * .4,
              factorization->get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * 0,
              factorization->get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1, factorization->get_context_embedding(2)[0], EPS);
  EXPECT_NEAR(-.2, factorization->get_context_embedding(2)[1], EPS);

  EXPECT_NEAR(sgd->get_rho(), rho, EPS);
}

TEST_F(SGNSTokenLearnerTest, token_train_neg1_partial_vocab_coverage) {
  const float
    rho = sgd->get_rho();
  const float adj2 = 1;

  InSequence in_sequence;
  EXPECT_CALL(*neg_sampling_strategy,
    sample_idx(Ref(*lm))).WillOnce(Return(2l));

  token_learner->token_train(2, 1, 1);

  EXPECT_NEAR(.1, factorization->get_word_embedding(0)[0], EPS);
  EXPECT_NEAR(-.2, factorization->get_word_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3, factorization->get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, factorization->get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * (-.3) +
                  rho * adj2 * (0 - sigmoid(.4 * .1 + 0 * (-.2))) * .1,
              factorization->get_word_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(0 + rho * (1 - sigmoid(.4 * (-.3) + 0 * .2)) * .2 +
                  rho * adj2 * (0 - sigmoid(.4 * .1 + 0 * (-.2))) * (-.2),
              factorization->get_word_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(.4, factorization->get_context_embedding(0)[0], EPS);
  EXPECT_NEAR(0, factorization->get_context_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * .4,
              factorization->get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * (1 - sigmoid(-.3 * .4 + .2 * 0)) * 0,
              factorization->get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1 + rho * adj2 * (0 - sigmoid(.1 * .4 + (-.2) * 0)) * .4,
              factorization->get_context_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * adj2 * (0 - sigmoid(.1 * .4 + (-.2) * 0)) * 0,
              factorization->get_context_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(sgd->get_rho(), rho, EPS);
}

TEST_F(SGNSTokenLearnerTest, token_train_neg2) {
  const float
    rho = sgd->get_rho();
  const float adj2 = 1, adj1 = 1;

  InSequence in_sequence;
  EXPECT_CALL(*neg_sampling_strategy,
    sample_idx(Ref(*lm))).WillOnce(Return(2l)).
                  WillOnce(Return(1l));

  token_learner->token_train(0, 1, 2);

  const float w0c1_coeff = (1 - sigmoid(.1 * (-.3) + (-.2) * .2));

  EXPECT_NEAR(.1 + rho * w0c1_coeff * (-.3) +
                  rho * adj2 * (0 - sigmoid(.1 * .1 + (-.2) * (-.2))) * .1 +
                  rho * adj1 * (0 - sigmoid(
                    .1 * (-.3 + rho * w0c1_coeff * .1) +
                    (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                  )) * (-.3 + rho * w0c1_coeff * .1),
              factorization->get_word_embedding(0)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * w0c1_coeff * .2 +
                   rho * adj2 * (0 - sigmoid(
                     .1 * .1 + (-.2) * (-.2)
                   )) * (-.2) +
                   rho * adj1 * (0 - sigmoid(
                     .1 * (-.3 + rho * w0c1_coeff * .1) +
                     (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                   )) * (.2 + rho * w0c1_coeff * (-.2)),
              factorization->get_word_embedding(0)[1], FAST_EPS);
  EXPECT_NEAR(-.3, factorization->get_word_embedding(1)[0], EPS);
  EXPECT_NEAR(.2, factorization->get_word_embedding(1)[1], EPS);
  EXPECT_NEAR(.4, factorization->get_word_embedding(2)[0], EPS);
  EXPECT_NEAR(0, factorization->get_word_embedding(2)[1], EPS);

  EXPECT_NEAR(.4, factorization->get_context_embedding(0)[0], EPS);
  EXPECT_NEAR(0, factorization->get_context_embedding(0)[1], EPS);
  EXPECT_NEAR(-.3 + rho * w0c1_coeff * .1 +
                   rho * adj1 * (0 - sigmoid(
                     .1 * (-.3 + rho * w0c1_coeff * .1) +
                     (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                   )) * .1,
              factorization->get_context_embedding(1)[0], FAST_EPS);
  EXPECT_NEAR(.2 + rho * w0c1_coeff * (-.2) +
                  rho * adj1 * (0 - sigmoid(
                    .1 * (-.3 + rho * w0c1_coeff * .1) +
                    (-.2) * (.2 + rho * w0c1_coeff * (-.2))
                  )) * (-.2),
              factorization->get_context_embedding(1)[1], FAST_EPS);
  EXPECT_NEAR(.1 + rho * adj2 * (0 - sigmoid(.1 * .1 + (-.2) * (-.2))) * .1,
              factorization->get_context_embedding(2)[0], FAST_EPS);
  EXPECT_NEAR(-.2 + rho * adj2 * (0 - sigmoid(
                .1 * .1 + (-.2) * (-.2)
              )) * (-.2),
              factorization->get_context_embedding(2)[1], FAST_EPS);

  EXPECT_NEAR(sgd->get_rho(), rho, EPS);
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

TEST_F(SGNSTokenLearnerTest, serialization_fixed_point) {
  stringstream ostream;
  token_learner->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGNSTokenLearner::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());
  from_stream->set_model(model);

  EXPECT_TRUE(token_learner->equals(*from_stream));
}

TEST_F(SGNSSentenceLearnerTest, increment) {
  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  sentence_learner->increment(string("foo"));

  EXPECT_CALL(*lm, increment("bbq")).
    WillOnce(Return(make_pair(2L, string("baz"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(4));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 4));
  sentence_learner->increment(string("bbq"));

  EXPECT_CALL(*lm, increment("baz")).
    WillOnce(Return(make_pair(2L, string("bbq"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(42));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 42));
  sentence_learner->increment(string("baz"));

  EXPECT_CALL(*lm, increment("bar")).
    WillOnce(Return(make_pair(1L, string("baz"))));
  EXPECT_CALL(*token_learner, reset_word(1));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(2));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 2));
  sentence_learner->increment(string("bar"));
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_zero) {
  vector<string> words;

  InSequence in_sequence;

  EXPECT_CALL(*ctx_strategy, size(_, _)).Times(0);
  EXPECT_CALL(*lm, increment(_)).Times(0);
  EXPECT_CALL(*lm, lookup(_)).Times(0);
  EXPECT_CALL(*token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), _)).Times(0);
  EXPECT_CALL(*token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_one) {
  vector<string> words;
  words.push_back(string("foo"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(2L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 2));

  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(2L));

  EXPECT_CALL(*ctx_strategy, size(0, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(*token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_empty_context) {
  vector<string> words;
  words.push_back(string("foo"));
  words.push_back(string("bbq"));
  words.push_back(string("baz"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(2L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 2));
  EXPECT_CALL(*lm, increment("bbq")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(0L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 0));
  EXPECT_CALL(*lm, increment("baz")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(1L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 1));

  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(0));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(2));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(1));

  EXPECT_CALL(*ctx_strategy, size(0, 2)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*ctx_strategy, size(1, 1)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*ctx_strategy, size(2, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*token_learner, reset_word(_)).Times(0);
  EXPECT_CALL(*token_learner, token_train(_, _, _)).Times(0);

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_short) {
  vector<string> words;
  words.push_back(string("foo"));
  words.push_back(string("bbq"));
  words.push_back(string("baz"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(2L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 2));
  EXPECT_CALL(*lm, increment("bbq")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(0L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 0));
  EXPECT_CALL(*lm, increment("baz")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(1L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 1));

  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(0));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(2));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(1));

  EXPECT_CALL(*ctx_strategy, size(0, 2)).
    WillOnce(Return(make_pair(size_t(0), size_t(2))));
  EXPECT_CALL(*token_learner, token_train(0, 2, 5));
  EXPECT_CALL(*token_learner, token_train(0, 1, 5));
  EXPECT_CALL(*ctx_strategy, size(1, 1)).
    WillOnce(Return(make_pair(size_t(1), size_t(1))));
  EXPECT_CALL(*token_learner, token_train(2, 0, 5));
  EXPECT_CALL(*token_learner, token_train(2, 1, 5));
  EXPECT_CALL(*ctx_strategy, size(2, 0)).
    WillOnce(Return(make_pair(size_t(2), size_t(0))));
  EXPECT_CALL(*token_learner, token_train(1, 0, 5));
  EXPECT_CALL(*token_learner, token_train(1, 2, 5));

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_lm_ejected) {
  vector<string> words;
  words.push_back(string("foo"));
  words.push_back(string("bbq"));
  words.push_back(string("baz"));
  words.push_back(string("bar"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("foo")).WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(2L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 2));
  EXPECT_CALL(*lm, increment("bbq")).WillOnce(Return(make_pair(2L, string("baz"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(0L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 0));
  EXPECT_CALL(*lm, increment("baz")).WillOnce(Return(make_pair(2L, string("bbq"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(1L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 1));
  EXPECT_CALL(*lm, increment("bar")).WillOnce(Return(make_pair(1L, string("baz"))));
  EXPECT_CALL(*token_learner, reset_word(1));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1L));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 1));

  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(0));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(-1));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(2));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1));

  EXPECT_CALL(*ctx_strategy, size(0, 2)).
    WillOnce(Return(make_pair(size_t(0), size_t(2))));
  EXPECT_CALL(*token_learner, token_train(0, 2, 5));
  EXPECT_CALL(*token_learner, token_train(0, 1, 5));
  EXPECT_CALL(*ctx_strategy, size(1, 1)).
    WillOnce(Return(make_pair(size_t(1), size_t(1))));
  EXPECT_CALL(*token_learner, token_train(2, 0, 5));
  EXPECT_CALL(*token_learner, token_train(2, 1, 5));
  EXPECT_CALL(*ctx_strategy, size(2, 0)).
    WillOnce(Return(make_pair(size_t(2), size_t(0))));
  EXPECT_CALL(*token_learner, token_train(1, 0, 5));
  EXPECT_CALL(*token_learner, token_train(1, 2, 5));

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, sentence_train_lm_many_tokens) {
  vector<string> words;
  words.push_back(string("bbq"));
  words.push_back(string("baz"));
  words.push_back(string("bar"));
  words.push_back(string("bar"));
  words.push_back(string("foo"));
  words.push_back(string("foo"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, increment("bbq")).
    WillOnce(Return(make_pair(2L, string("baz"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  EXPECT_CALL(*lm, increment("baz")).
    WillOnce(Return(make_pair(2L, string("bbq"))));
  EXPECT_CALL(*token_learner, reset_word(2));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  EXPECT_CALL(*lm, increment("bar")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  EXPECT_CALL(*lm, increment("bar")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));
  EXPECT_CALL(*lm, increment("foo")).
    WillOnce(Return(make_pair(-1L, string())));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(7));
  EXPECT_CALL(*neg_sampling_strategy, step(Ref(*lm), 7));

  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(-1));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(2));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1)).
                                  WillOnce(Return(1));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(0)).
                                  WillOnce(Return(0));

  EXPECT_CALL(*ctx_strategy, size(0, 4)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*ctx_strategy, size(1, 3)).
    WillOnce(Return(make_pair(size_t(1), size_t(2))));
  EXPECT_CALL(*token_learner, token_train(1, 2, 5));
  EXPECT_CALL(*token_learner, token_train(1, 1, 5));
  EXPECT_CALL(*token_learner, token_train(1, 0, 5));
  EXPECT_CALL(*ctx_strategy, size(2, 2)).
    WillOnce(Return(make_pair(size_t(1), size_t(2))));
  EXPECT_CALL(*token_learner, token_train(1, 1, 5));
  EXPECT_CALL(*token_learner, token_train(1, 0, 5));
  EXPECT_CALL(*token_learner, token_train(1, 0, 5));
  EXPECT_CALL(*ctx_strategy, size(3, 1)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));
  EXPECT_CALL(*ctx_strategy, size(4, 0)).
    WillOnce(Return(make_pair(size_t(0), size_t(0))));

  sentence_learner->sentence_train(words);
}

TEST_F(SGNSSentenceLearnerTest, serialization_fixed_point) {
  stringstream ostream;
  sentence_learner->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGNSSentenceLearner::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());
  from_stream->set_model(model);

  EXPECT_TRUE(sentence_learner->equals(*from_stream));
}

TEST_F(NonPropagatingSubsamplingSGNSSentenceLearnerTest, sentence_train) {
  vector<string> words;
  words.push_back(string("bbq"));
  words.push_back(string("baz"));
  words.push_back(string("bar"));
  words.push_back(string("bar"));
  words.push_back(string("bbq"));
  words.push_back(string("foo"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(2));
  EXPECT_CALL(*lm, subsample(2)).WillOnce(Return(true));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(-1));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1));
  EXPECT_CALL(*lm, subsample(1)).WillOnce(Return(false));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1));
  EXPECT_CALL(*lm, subsample(1)).WillOnce(Return(false));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(0));
  EXPECT_CALL(*lm, subsample(0)).WillOnce(Return(true));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(-1));

  vector<string> expected_subsampled_words;
  expected_subsampled_words.push_back(string("bbq"));
  expected_subsampled_words.push_back(string("baz"));
  expected_subsampled_words.push_back(string("bbq"));
  expected_subsampled_words.push_back(string("foo"));

  EXPECT_CALL(*sentence_learner, sentence_train(expected_subsampled_words));

  subsampling_sentence_learner->sentence_train(words);
}

TEST_F(SubsamplingSGNSSentenceLearnerTest, sentence_train) {
  vector<string> words;
  words.push_back(string("bbq"));
  words.push_back(string("baz"));
  words.push_back(string("bar"));
  words.push_back(string("bar"));
  words.push_back(string("bbq"));
  words.push_back(string("foo"));

  InSequence in_sequence;

  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(2));
  EXPECT_CALL(*lm, subsample(2)).WillOnce(Return(true));
  EXPECT_CALL(*lm, lookup("baz")).WillOnce(Return(-1));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1));
  EXPECT_CALL(*lm, subsample(1)).WillOnce(Return(false));
  EXPECT_CALL(*sentence_learner, increment("bar"));
  EXPECT_CALL(*lm, lookup("bar")).WillOnce(Return(1));
  EXPECT_CALL(*lm, subsample(1)).WillOnce(Return(false));
  EXPECT_CALL(*sentence_learner, increment("bar"));
  EXPECT_CALL(*lm, lookup("bbq")).WillOnce(Return(0));
  EXPECT_CALL(*lm, subsample(0)).WillOnce(Return(true));
  EXPECT_CALL(*lm, lookup("foo")).WillOnce(Return(-1));

  vector<string> expected_subsampled_words;
  expected_subsampled_words.push_back(string("bbq"));
  expected_subsampled_words.push_back(string("baz"));
  expected_subsampled_words.push_back(string("bbq"));
  expected_subsampled_words.push_back(string("foo"));

  EXPECT_CALL(*sentence_learner, sentence_train(expected_subsampled_words));

  subsampling_sentence_learner->sentence_train(words);
}

TEST_F(SubsamplingSGNSSentenceLearnerTest, serialization_fixed_point) {
  stringstream ostream;
  subsampling_sentence_learner->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SubsamplingSGNSSentenceLearner::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());
  from_stream->set_model(model);

  EXPECT_TRUE(subsampling_sentence_learner->equals(*from_stream));
}

TEST_F(SGNSModelTest, serialization_fixed_point) {
  stringstream ostream;
  model->serialize(ostream);
  ostream.flush();

  stringstream istream(ostream.str());
  auto from_stream(SGNSModel::deserialize(istream));
  ASSERT_EQ(EOF, istream.peek());

  EXPECT_TRUE(model->equals(*from_stream));
}
