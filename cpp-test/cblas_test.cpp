#include "cblas_test.h"
#include "test_util.h"
#include "_cblas.h"

#include <gtest/gtest.h>


TEST(cblas_sdot, three) {
  const float x[] = {0.5, -1, -2};
  const float y[] = {-5, 0, -3};
  EXPECT_NEAR(3.5, cblas_sdot(3, x, 1, y, 1), EPS);
}

TEST(cblas_sdot, three_strided_x) {
  const float x[] = {0.5, 4, 4, -1, 5, 5, -2, 6, 6};
  const float y[] = {-5, 0, -3};
  EXPECT_NEAR(3.5, cblas_sdot(3, x, 3, y, 1), EPS);
}

TEST(cblas_sdot, three_strided_y) {
  const float x[] = {0.5, -1, -2};
  const float y[] = {-5, 4, 4, 0, 5, 5, -3, 6, 6};
  EXPECT_NEAR(3.5, cblas_sdot(3, x, 1, y, 3), EPS);
}

TEST(cblas_snrm2, three) {
  const float x[] = {0.5, -1, -2};
  EXPECT_NEAR(2.291288, cblas_snrm2(3, x, 1), EPS);
}

TEST(cblas_snrm2, three_strided) {
  const float x[] = {0.5, 4, 4, -1, 5, 5, -2, 6, 6};
  EXPECT_NEAR(2.291288, cblas_snrm2(3, x, 3), EPS);
}

TEST(cblas_saxpy, three) {
  const float x[] = {0.5, -1, -2};
  float y[] = {-5, 0, -3};
  cblas_saxpy(3, 7, x, 1, y, 1);
  EXPECT_NEAR(-1.5, y[0], EPS);
  EXPECT_NEAR(-7,   y[1], EPS);
  EXPECT_NEAR(-17,  y[2], EPS);
}

TEST(cblas_saxpy, three_strided_x) {
  const float x[] = {0.5, 3, 3, -1, 4, 4, -2, 5, 5};
  float y[] = {-5, 0, -3};
  cblas_saxpy(3, 7, x, 3, y, 1);
  EXPECT_NEAR(-1.5, y[0], EPS);
  EXPECT_NEAR(-7,   y[1], EPS);
  EXPECT_NEAR(-17,  y[2], EPS);
}

TEST(cblas_saxpy, three_strided_y) {
  const float x[] = {0.5, -1, -2};
  float y[] = {-5, 3, 3, 0, 4, 4, -3, 5, 5};
  cblas_saxpy(3, 7, x, 1, y, 3);
  EXPECT_NEAR(-1.5, y[0], EPS);
  EXPECT_NEAR(3,    y[1], EPS);
  EXPECT_NEAR(3,    y[2], EPS);
  EXPECT_NEAR(-7,   y[3], EPS);
  EXPECT_NEAR(4,    y[4], EPS);
  EXPECT_NEAR(4,    y[5], EPS);
  EXPECT_NEAR(-17,  y[6], EPS);
  EXPECT_NEAR(5,    y[7], EPS);
  EXPECT_NEAR(5,    y[8], EPS);
}

TEST(cblas_sscal, three) {
  float x[] = {0.5, -1, -2};
  cblas_sscal(3, 7, x, 1);
  EXPECT_NEAR(3.5, x[0], EPS);
  EXPECT_NEAR(-7,  x[1], EPS);
  EXPECT_NEAR(-14, x[2], EPS);
}

TEST(cblas_sscal, three_strided) {
  float x[] = {0.5, 3, 3, -1, 4, 4, -2, 5, 5};
  cblas_sscal(3, 7, x, 3);
  EXPECT_NEAR(3.5, x[0], EPS);
  EXPECT_NEAR(3,   x[1], EPS);
  EXPECT_NEAR(3,   x[2], EPS);
  EXPECT_NEAR(-7,  x[3], EPS);
  EXPECT_NEAR(4,   x[4], EPS);
  EXPECT_NEAR(4,   x[5], EPS);
  EXPECT_NEAR(-14, x[6], EPS);
  EXPECT_NEAR(5,   x[7], EPS);
  EXPECT_NEAR(5,   x[8], EPS);
}
