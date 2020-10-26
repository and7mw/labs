__kernel void saxpy(const int n, const float a, __global float* x, const int incx, __global float* y, const int incy) {
    int id = get_global_id(0);

    if (id < n && id * incx < n && id * incy < n)
        y[id * incy] += a * x[id * incx];
}

__kernel void daxpy(const int n, const double a, __global double* x, const int incx, __global double* y, const int incy) {
    int id = get_global_id(0);

    if (id < n && id * incx < n && id * incy < n)
        y[id * incy] += a * x[id * incx];
}