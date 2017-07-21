#include "_cblas.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#ifndef HAVE_CBLAS

#include <cmath>

extern "C" {

void cblas_saxpy(const int N, const float alpha, const float* __restrict__ X,
                 const int incX, float* __restrict__ Y, const int incY) {
  int i, x_i = 0, y_i = 0;
  if (incX == 1 && incY == 1) {
    for (i = 0; i < N; ++i) {
      Y[i] += alpha * X[i];
    }
  } else {
    for (i = 0; i < N; ++i) {
      Y[y_i] += alpha * X[x_i];
      x_i += incX;
      y_i += incY;
    }
  }
}

float cblas_sdot(const int N, const float* __restrict__ X, const int incX,
                 const float* __restrict__ Y, const int incY) {
  int i, x_i = 0, y_i = 0;
  float dot = 0;
  if (incX == 1 && incY == 1) {
    for (i = 0; i < N; ++i) {
      dot += X[i] * Y[i];
    }
  } else {
    for (i = 0; i < N; ++i) {
      dot += X[x_i] * Y[y_i];
      x_i += incX;
      y_i += incY;
    }
  }
  return dot;
}

float cblas_snrm2(const int N, const float* __restrict__ X, const int incX) {
  return sqrt(cblas_sdot(N, X, incX, X, incX));
}

void cblas_sscal(const int N, const float alpha, float* __restrict__ X,
                 const int incX) {
  int i, x_i = 0;
  if (incX == 1) {
    for (i = 0; i < N; ++i) {
      X[i] *= alpha;
    }
  } else {
    for (i = 0; i < N; ++i) {
      X[x_i] *= alpha;
      x_i += incX;
    }
  }
}

}

#endif
#endif
