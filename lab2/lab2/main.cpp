#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <omp.h>

#include "axpy_host.hpp"
#include "opencl_utils.hpp"

template <typename dataType>
std::vector<dataType> getVector(const int& size) {
    std::vector<dataType> resVector(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<dataType> dist(-100, 100);

    for (size_t i = 0; i < resVector.size(); i++)
        resVector[i] = dist(gen);

    return resVector;
}

template <typename dataType>
void computeOnDevice(const int& n, const int& incx, const int& incy, const std::vector<dataType>& srcVector, const dataType& a,
    const cl_device_type deviceType, const cl_platform_id& platform, const std::vector<char>& kernelText,
    const size_t& localWorkSize, std::vector<dataType>& result) {
    cl_device_id device{};
    getDevice(platform, deviceType, device);
    cl_context context{};
    createContext(platform, device, context);
    cl_command_queue queue{};
    createQueue(context, device, queue);
    cl_program program{};
    cl_kernel kernel{};

    if (std::is_same<dataType, float>::value)
        createProgramAndKernel(context, device, program, kernel, "saxpy");
    else if (std::is_same<dataType, double>::value)
        createProgramAndKernel(context, device, program, kernel, "daxpy");
    else
        throw std::runtime_error("Unsupported data type to execute");

    cl_mem x{}, y{};
    createMemoryObject(context, x, y, n, sizeof(dataType));
    writeToBuffer(queue, x, srcVector.data(), n, sizeof(dataType));
    writeToBuffer(queue, y, result.data(), n, sizeof(dataType));
    setArguments<dataType>(kernel, n, a, x, incx, y, incy);

    size_t maxLocWorkGroup{};
    if (clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxLocWorkGroup, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get kernel work group info");
    std::cout << "Max kernel work group size: " << maxLocWorkGroup << std::endl;

    execute(queue, kernel, n, localWorkSize);

    if (clEnqueueReadBuffer(queue, y, CL_TRUE, 0, sizeof(dataType) * n, result.data(), 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't write to buffer");
}

template <typename dataType>
void compare(const std::vector<dataType>& ref, const std::vector<dataType>& res) {
    if (ref.size() != res.size())
        throw std::runtime_error("Vectors have different size");
    dataType refVal = ref[0];
    dataType resVal = res[0];
    dataType diff = std::abs(refVal - resVal);
    for (size_t i = 1; i < ref.size(); i++) {
        if (std::abs(ref[i] - res[i]) > diff) {
            diff = std::abs(ref[i] - res[i]);
            refVal = ref[i];
            resVal = res[i];
        }
    }
    std::cout << "Max difference is: " << diff << " on ref: " << refVal << " and res: " << resVal << std::endl;
}

int main() {
    const int n = 1073741824; // 2^30
    std::cout << n << std::endl;
    const int incx = 2;
    const int incy = 1;
    cl_device_type deviceTypeGPU = CL_DEVICE_TYPE_GPU;
    cl_device_type deviceTypeCPU = CL_DEVICE_TYPE_CPU;

    cl_platform_id platform{};
    getPlatform(platform);
    std::vector<char> kernelText;
    readKernelFile(kernelText);

    std::cout << "******************** FLOAT ********************" << std::endl;
    {
        const float a = 0.2f;
        const std::vector<float> x = getVector<float>(n);
        const std::vector<float> y = getVector<float>(n);

        // reference
        std::vector<float> yRef(y.begin(), y.end());
        std::cout << "Reference start" << std::endl;
        double start = omp_get_wtime();
        host::axpy<float>(n, a, x.data(), incx, yRef.data(), incy);
        double end = omp_get_wtime();
        std::cout << "Reference time: " << end - start << " sec" << std::endl;
        std::cout << std::endl;

        // OpenMP
        std::vector<float> yOmp(y.begin(), y.end());
        std::cout << "OpenMP start" << std::endl;
        start = omp_get_wtime();
        host::saxpy(n, a, x.data(), incx, yOmp.data(), incy);
        end = omp_get_wtime();
        std::cout << "OpenMP time: " << end - start << " sec" << std::endl;
        compare<float>(yRef, yOmp);
        yOmp.clear();
        std::cout << std::endl;

        // GPU OpenCL
        std::cout << "OpenCL GPU start" << std::endl;
        for (size_t localWorkSize = 8; localWorkSize <= 256; localWorkSize *= 2) {
            try {
                std::vector<float> yGpu(y.begin(), y.end());
                start = omp_get_wtime();
                computeOnDevice<float>(n, incx, incy, x, a, deviceTypeGPU, platform, kernelText, localWorkSize, yGpu);
                end = omp_get_wtime();
                std::cout << "OpenCL GPU with group size: " << localWorkSize
                    << " has time: " << end - start << " sec" << std::endl;
                compare<float>(yRef, yGpu);
                std::cout << std::endl;
            }
            catch (const std::exception & e) {
                std::cout << e.what() << std::endl;
            }
        }

        // CPU OpenCL
        std::cout << "OpenCL CPU start" << std::endl;
        for (size_t localWorkSize = 8; localWorkSize <= 256; localWorkSize *= 2) {
            try {
                std::vector<float> yCpu(y.begin(), y.end());
                start = omp_get_wtime();
                computeOnDevice<float>(n, incx, incy, x, a, deviceTypeCPU, platform, kernelText, localWorkSize, yCpu);
                end = omp_get_wtime();
                std::cout << "OpenCL CPU with group size: " << localWorkSize
                    << " has time: " << end - start << " sec" << std::endl;
                compare<float>(yRef, yCpu);
                std::cout << std::endl;
            }
            catch (const std::exception & e) {
                std::cout << e.what() << std::endl;
            }
        }
    }

    std::cout << std::endl << "******************** DOUBLE ********************" << std::endl;
    {
        const double a = 0.2;
        const std::vector<double> x = getVector<double>(n);
        const std::vector<double> y = getVector<double>(n);

        // reference
        std::vector<double> yRef(y.begin(), y.end());
        std::cout << "Reference start" << std::endl;
        double start = omp_get_wtime();
        host::axpy<double>(n, a, x.data(), incx, yRef.data(), incy);
        double end = omp_get_wtime();
        std::cout << "Reference time: " << end - start << " sec" << std::endl;
        std::cout << std::endl;

        // OpenMP
        std::vector<double> yOmp(y.begin(), y.end());
        std::cout << "OpenMP start" << std::endl;
        start = omp_get_wtime();
        host::daxpy(n, a, x.data(), incx, yOmp.data(), incy);
        end = omp_get_wtime();
        std::cout << "OpenMP time: " << end - start << " sec" << std::endl;
        compare<double>(yRef, yOmp);
        std::cout << std::endl;

        // GPU OpenCL
        std::cout << "OpenCL GPU start" << std::endl;
        for (size_t localWorkSize = 8; localWorkSize <= 256; localWorkSize *= 2) {
            try {
                std::vector<double> yGpu(y.begin(), y.end());
                start = omp_get_wtime();
                computeOnDevice<double>(n, incx, incy, x, a, deviceTypeGPU, platform, kernelText, localWorkSize, yGpu);
                end = omp_get_wtime();
                std::cout << "OpenCL GPU with group size: " << localWorkSize
                    << " has time: " << end - start << " sec" << std::endl;
                compare<double>(yRef, yGpu);
                std::cout << std::endl;
            }
            catch (const std::exception & e) {
                std::cout << e.what() << std::endl;
            }
        }

        // CPU OpenCL
        std::cout << "OpenCL CPU start" << std::endl;
        for (size_t localWorkSize = 8; localWorkSize <= 256; localWorkSize *= 2) {
            try {
                std::vector<double> yCpu(y.begin(), y.end());
                start = omp_get_wtime();
                computeOnDevice<double>(n, incx, incy, x, a, deviceTypeCPU, platform, kernelText, localWorkSize, yCpu);
                end = omp_get_wtime();
                std::cout << "OpenCL CPU with group size: " << localWorkSize
                    << " has time: " << end - start << " sec" << std::endl;
                compare<double>(yRef, yCpu);
                std::cout << std::endl;
            }
            catch (const std::exception & e) {
                std::cout << e.what() << std::endl;
            }
        }
    }

    return 0;
}
