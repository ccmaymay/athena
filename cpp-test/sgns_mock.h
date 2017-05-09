#ifndef ATHENA_SGNS_MOCK_H
#define ATHENA_SGNS_MOCK_H


#include "_core.h"
#include "_sgns.h"

#include <gmock/gmock.h>


class MockSGNSTokenLearner : public SGNSTokenLearner {
  public:
    using SGNSTokenLearner::SGNSTokenLearner;
    MOCK_METHOD1(reset_word, void (long word_idx));
    MOCK_METHOD3(token_train, void (size_t target_word_idx,
                                    size_t context_word_idx,
                                    size_t neg_samples));
    MOCK_CONST_METHOD3(compute_gradient_coeff,
                       float (long target_word_idx, long context_word_idx,
                               bool negative_sample));
    MOCK_CONST_METHOD2(context_contains_oov,
                       bool (const long *ctx_word_ids, size_t ctx_size));
    MOCK_CONST_METHOD2(compute_similarity,
                       float (size_t word1_idx, size_t word2_idx));
    MOCK_METHOD1(find_nearest_neighbor_idx, long (size_t word_idx));
    MOCK_METHOD3(find_context_nearest_neighbor_idx,
                 long (size_t left_context, size_t right_context,
                       const long *word_ids));
    MOCK_METHOD1(set_model, void (std::shared_ptr<SGNSModel> model));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const SGNSTokenLearner& other));
};


class MockSGNSSentenceLearner : public SGNSSentenceLearner {
  public:
    MockSGNSSentenceLearner():
      SGNSSentenceLearner(5, true) { }
    MOCK_METHOD1(increment, void (const std::string& word));
    MOCK_METHOD1(sentence_train, void (const std::vector<std::string>& words));
    MOCK_METHOD1(set_model, void (std::shared_ptr<SGNSModel> model));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const SGNSSentenceLearner& other));
};


class MockSubsamplingSGNSSentenceLearner : public SubsamplingSGNSSentenceLearner {
  public:
    MockSubsamplingSGNSSentenceLearner():
      SubsamplingSGNSSentenceLearner(true) { }
    MOCK_METHOD1(increment, void (const std::string& word));
    MOCK_METHOD1(sentence_train, void (const std::vector<std::string>& words));
    MOCK_METHOD1(set_model, void (std::shared_ptr<SGNSModel> model));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const SubsamplingSGNSSentenceLearner& other));
};


#endif
