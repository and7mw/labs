#pragma once

#include <omp.h>

namespace host {

template <typename dataType>
void axpy(const int& n, const dataType& a, const dataType* x, const int& incx, dataType* y, const int& incy) {
    for (int i = 0; i < n && i * incx < n && i * incy < n; i++)
        y[i * incy] += a * x[i * incx];
}

void saxpy(const int& n, const float& a, const float* x, const int& incx, float* y, const int& incy) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (i * incx < n && i * incy < n)
            y[i * incy] += a * x[i * incx];
    }
}

void daxpy(const int& n, const double &a, const double* x, const int& incx, double* y, const int& incy) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (i * incx < n && i * incy < n)
            y[i * incy] += a * x[i * incx];
    }
}

}