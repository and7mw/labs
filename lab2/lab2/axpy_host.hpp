#pragma once

#include <omp.h>

namespace host {

template <typename dataType>
void axpy(const int& n, const dataType& a, const dataType* x, const int& incx, dataType* y, const int& incy) {
    for (int i = 0; i < n && i * incx < n && i * incy < n; i++)
        y[i * incy] += a * x[i * incx];
}

void saxpy(const int& n, const float& a, const float* x, const int& incx, float* y, const int& incy) {
    int step = std::max(incx, incy);
    int workAmount = n / step;
#pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int lwa = workAmount / nthreads;
        int rem = 0;
        if (tid == (nthreads - 1))
            rem += workAmount % nthreads;
        for (int i = tid * lwa; i < (tid + 1) * (lwa + rem); i++) {
            if (i * incx < n && i * incy < n)
                y[i * incy] += a * x[i * incx];
        }
    }
}

void daxpy(const int& n, const double& a, const double* x, const int& incx, double* y, const int& incy) {
    int step = std::max(incx, incy);
    int workAmount = n / step;
#pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int lwa = workAmount / nthreads;
        int rem = 0;
        if (tid == (nthreads - 1))
            rem += workAmount % nthreads;
        for (int i = tid * lwa; i < (tid + 1) * (lwa + rem); i++) {
            if (i * incx < n && i * incy < n)
                y[i * incy] += a * x[i * incx];
        }
    }
}

}
