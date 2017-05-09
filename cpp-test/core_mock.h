#ifndef ATHENA_CORE_MOCK_H
#define ATHENA_CORE_MOCK_H


#include "_core.h"

#include <gmock/gmock.h>


class MockLanguageModel : public LanguageModel {
  public:
    MOCK_METHOD1(increment,
                 std::pair<long,std::string> (const std::string& word));
    MOCK_CONST_METHOD1(lookup, long (const std::string& word));
    MOCK_CONST_METHOD1(reverse_lookup, std::string (long word_idx));
    MOCK_CONST_METHOD1(count, size_t (long word_idx));
    MOCK_CONST_METHOD0(counts, std::vector<size_t> ());
    MOCK_CONST_METHOD0(ordered_counts, std::vector<size_t> ());
    MOCK_CONST_METHOD0(size, size_t ());
    MOCK_CONST_METHOD0(total, size_t ());
    MOCK_CONST_METHOD1(subsample, bool (long word_idx));
    MOCK_METHOD1(truncate, void (size_t max_size));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const LanguageModel& other));
};


class MockSGD : public SGD {
  public:
    using SGD::SGD;
    MOCK_METHOD1(step, void (size_t dim));
    MOCK_CONST_METHOD1(get_rho, float (size_t dim));
    MOCK_METHOD4(gradient_update, void (size_t dim, size_t n, const float *g,
                                        float *x));
    MOCK_METHOD5(scaled_gradient_update, void (size_t dim, size_t n,
                                               const float *g, float *x,
                                               float alpha));
    MOCK_METHOD1(reset, void (size_t dim));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const SGD& other));

  private:
    MOCK_METHOD1(_compute_rho, void (size_t dimension));
};


class MockSamplingStrategy : public SamplingStrategy {
  public:
    using SamplingStrategy::SamplingStrategy;
    MOCK_METHOD1(sample_idx,
      long (const LanguageModel& language_model));
    MOCK_METHOD2(step,
      void (const LanguageModel& language_model, size_t word_idx));
    MOCK_METHOD2(reset,
      void (const LanguageModel& language_model,
            const CountNormalizer& normalizer));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const SamplingStrategy& other));
};


class MockContextStrategy : public ContextStrategy {
  public:
    using ContextStrategy::ContextStrategy;
    MOCK_CONST_METHOD2(size,
      std::pair<size_t,size_t> (size_t avail_left, size_t avail_right));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const ContextStrategy& other));
};


#endif
