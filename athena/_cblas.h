#ifndef ATHENA__CBLAS_H
#define ATHENA__CBLAS_H


extern "C" {

void cblas_saxpy(const int N, const float alpha, const float* X,
                 const int incX, float* Y, const int incY);
float cblas_sdot(const int N, const float* X, const int incX,
                 const float* Y, const int incY);
float cblas_snrm2(const int N, const float* X, const int incX);
void cblas_sscal(const int N, const float alpha, float* X,
                 const int incX);

}


#endif
