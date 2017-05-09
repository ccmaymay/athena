#ifndef ATHENA_MATH_MOCK_H
#define ATHENA_MATH_MOCK_H


#include "_math.h"

#include <gmock/gmock.h>


class MockCountNormalizer : public CountNormalizer {
  public:
    using CountNormalizer::CountNormalizer;
    MOCK_CONST_METHOD1(normalize, std::vector<float>
                                    (const std::vector<size_t>& counts));

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const CountNormalizer& other));
};


static const long mock_reservoir_sampler_ret = 99;


class MockLongReservoirSampler : public ReservoirSampler<long> {
  public:
    using ReservoirSampler<long>::ReservoirSampler;
    MOCK_CONST_METHOD0(sample, long ());
    virtual const long& operator[](size_t idx) const {
      return mock_reservoir_sampler_ret;
    }
    MOCK_CONST_METHOD0(size, size_t ());
    MOCK_CONST_METHOD0(filled_size, size_t ());
    MOCK_METHOD1(insert, long (long val));
    MOCK_METHOD0(clear, void ());

    MOCK_CONST_METHOD1(serialize, void (std::ostream& stream));
    MOCK_CONST_METHOD1(equals, bool (const ReservoirSampler<long>& other));
};


#endif
