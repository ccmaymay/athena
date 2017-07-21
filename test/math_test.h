#ifndef ATHENA_MATH_TEST_H
#define ATHENA_MATH_TEST_H


#include "_math.h"


#include <gtest/gtest.h>
#include <vector>
#include <memory>


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


class ExponentCountNormalizerTest: public ::testing::Test {
  protected:
    std::shared_ptr<ExponentCountNormalizer> count_normalizer;

    virtual void SetUp() {
      count_normalizer = std::make_shared<ExponentCountNormalizer>(0.8, 4.2);
    }

    virtual void TearDown() { }
};


class ReservoirSamplerTest: public ::testing::Test {
  protected:
    std::shared_ptr<ReservoirSampler<long> > sampler;

    virtual void SetUp() {
      sampler = std::make_shared<ReservoirSampler<long> >(3);
      sampler->insert(-1);
      sampler->insert(7);
      sampler->insert(-1);
    }

    virtual void TearDown() { }
};


class DiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    std::shared_ptr<Discretization> sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.5, 0.4};
      sampler = std::make_shared<Discretization>(probabilities, 9);
    }

    virtual void TearDown() { }
};


class SubProbabilityDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    std::shared_ptr<Discretization> sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.2, 0.1};
      sampler = std::make_shared<Discretization>(probabilities, 9);
    }

    virtual void TearDown() { }
};


class OneAtomDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    std::shared_ptr<Discretization> sampler;

    virtual void SetUp() {
      probabilities = {1.};
      sampler = std::make_shared<Discretization>(probabilities, 9);
    }

    virtual void TearDown() { }
};


class NearlyTwoAtomDiscretizationTest: public ::testing::Test {
  protected:
    vector<float> probabilities;
    std::shared_ptr<Discretization> sampler;

    virtual void SetUp() {
      probabilities = {0.1, 0.001, 0.899};
      sampler = std::make_shared<Discretization>(probabilities, 9);
    }

    virtual void TearDown() { }
};


#endif
