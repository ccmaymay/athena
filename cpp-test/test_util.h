#ifndef ATHENA_TEST_UTIL_H
#define ATHENA_TEST_UTIL_H


#include <gtest/gtest.h>
#include <gmock/gmock.h>


#define EPS 1e-4


template <class T>
bool array_eq(const T *x, const T *y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (x[i] != y[i]) {
      return false;
    }
  }
  return true;
}


MATCHER_P2(ArrayEq, x, n, "") {
  return array_eq(x, arg, n);
}


#endif
