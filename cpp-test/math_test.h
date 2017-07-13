#ifndef ATHENA_MATH_TEST_H
#define ATHENA_MATH_TEST_H


#include "_math.h"


#include <gtest/gtest.h>
#include <vector>


using namespace std;


class DoubleVectorNearTest: public ::testing::Test {
  protected:
    vector<double> x, y;

    virtual void SetUp() {
      x = {3, -2.5, 0};
      y = {3, -2.5, 0};
    }

    virtual void TearDown() { }
};

class FloatVectorNearTest: public ::testing::Test {
  protected:
    vector<float> x, y;

    virtual void SetUp() {
      x = {3, -2.5, 0};
      y = {3, -2.5, 0};
    }

    virtual void TearDown() { }
};


class SamplerTest: public ::testing::Test {
  protected:
    vector<float> probabilities;

    virtual void SetUp() {
      probabilities = {0.1, 0.5, 0.4};
    }
};


class OneAtomSamplerTest: public ::testing::Test {
  protected:
    vector<float> probabilities;

    virtual void SetUp() {
      probabilities = {0, 0, 1, 0};
    }
};


class TwoAtomSamplerTest: public ::testing::Test {
  protected:
    vector<float> probabilities;

    virtual void SetUp() {
      probabilities = {0.6, 0, 0.4, 0};
    }
};


class UniformSamplerTest: public ::testing::Test {
  protected:
    vector<float> probabilities;

    virtual void SetUp() {
      probabilities = {0.25, 0.25, 0.25, 0.25};
    }
};


class CountNormalizerTest: public ::testing::Test {
  protected:
    CountNormalizer *count_normalizer;

    virtual void SetUp() {
      count_normalizer = new CountNormalizer(0.8, 4.2);
    }

    virtual void TearDown() {
      delete count_normalizer;
    }
};


class ReservoirSamplerTest: public ::testing::Test {
  protected:
    ReservoirSampler<long> *sampler;

    virtual void SetUp() {
      sampler = new ReservoirSampler<long>(3);
      sampler->insert(-1);
      sampler->insert(7);
      sampler->insert(-1);
    }

    virtual void TearDown() {
      delete sampler;
    }
};


class DiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    Discretization *sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.5, 0.4};
      sampler = new Discretization(probabilities, 9);
    }

    virtual void TearDown() {
      delete sampler;
    }
};


class SubProbabilityDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    Discretization *sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.2, 0.1};
      sampler = new Discretization(probabilities, 9);
    }

    virtual void TearDown() {
      delete sampler;
    }
};


class OneAtomDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    Discretization *sampler;

    virtual void SetUp() {
      probabilities = {1.};
      sampler = new Discretization(probabilities, 9);
    }

    virtual void TearDown() {
      delete sampler;
    }
};


class NearlyTwoAtomDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    Discretization *sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.001, 0.899};
      sampler = new Discretization(probabilities, 9);
    }

    virtual void TearDown() {
      delete sampler;
    }
};


#endif
