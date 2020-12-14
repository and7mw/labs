#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

#include "opencl_utils.hpp"

#define BLOCK_SIZE 16

std::vector<float> getMatrix(const int& size) {
    std::vector<float> resVector(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100, 100);
    //std::uniform_real_distribution<float> dist(-100, 100);

    for (size_t i = 0; i < resVector.size(); i++)
        resVector[i] = dist(gen);

    return resVector;
}

std::vector<float> reference(const std::vector<float>& A, const std::vector<float>& B,
                             const unsigned int col1, const unsigned int row1, const unsigned int col2, const unsigned int row2) {
    if (A.size() != B.size()) {
        throw std::runtime_error("Cant mult matrix");
    }

    size_t workAmount = row1 * col2;
    std::vector<float> C(workAmount);
    const float* in1 = A.data();
    const float* in2 = B.data();
    float* out = C.data();
    memset(out, 0, C.size() * sizeof(float));
    double start = omp_get_wtime();
    for (size_t id = 0; id < workAmount; id++) {
        size_t col = id % col2;
        size_t row = id / col2;

        const float* inA = in1 + col1 * row;
        const float* inB = in2 + col;

        for (unsigned int i = 0; i < col1; i++) {
            out[id] += inA[i] * inB[i * col2];
        }
    }
    double end = omp_get_wtime();
    std::cout << "Reference execution time: " << (end - start) << std::endl;
    return C;
}

void computeOMP(const std::vector<float>& _in1, const std::vector<float>& _in2, std::vector<float>& _out,
                const unsigned int col1, const unsigned int row1, const unsigned int col2, const unsigned int row2) {
    const unsigned int workAmount = row1 * col2;
    _out.resize(workAmount);
    double start = omp_get_wtime();
#pragma omp parallel num_threads(8)
    {
        unsigned int tid = static_cast<unsigned int>(omp_get_thread_num());
        unsigned int nthreads = static_cast<unsigned int>(omp_get_num_threads());
        unsigned int lwa = workAmount / nthreads;
        unsigned int rem = 0;
        if (tid == (nthreads - 1))
            rem += workAmount % nthreads;
        unsigned int start = tid * lwa;
        unsigned int end = (tid + 1) * lwa + rem;
        unsigned int startRow = start / col2;
        unsigned int startCol = start % col2;
        unsigned int endRow = end / col2;
        unsigned int endCol = (end % col2) == 0 ? col2 : (end % col2);
        for (unsigned int r = startRow; r < endRow; r++) {
            for (unsigned int c = startCol; c < endCol; c++) {
                float acc = 0.0f;
                for (unsigned int i = 0; i < col1; i++) {
                    acc += _in1[r * col1 + i] * _in2[i * col2 + c];
                }
                _out[r * col2 + c] = acc;
            }
        }
    }
    double end = omp_get_wtime();
    std::cout << "Execution time: " << (end - start) << std::endl;
}

enum bufferType {
    BUFFER,
    IMAGE
};

void computeOnDevice(const cl_platform_id& platform, const cl_device_type deviceType, const std::vector<char>& kernelText,
                     const std::string kernelName, const std::vector<float>& _in1, const std::vector<float>& _in2, std::vector<float>& _out,
                     const unsigned int col1, const unsigned int row1, const unsigned int col2, const unsigned int row2, bufferType bt = bufferType::BUFFER) {
    cl_device_id device{};
    getDevice(platform, deviceType, device);
    cl_context context{};
    createContext(platform, device, context);
    cl_command_queue queue{};
    createQueue(context, device, queue);
    cl_program program{};
    cl_kernel kernel{};
    createProgramAndKernel(context, device, program, kernel, kernelText, kernelName);

    cl_int retCode;

    cl_mem in1{};
    if (bt == bufferType::BUFFER) {
        in1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * _in1.size(), NULL, &retCode);
    } else if (bt == bufferType::IMAGE) {
        cl_image_format format{};
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;
        cl_image_desc desc{};
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = col1;
        desc.image_height = row1;

        in1 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &retCode);
    } else {
        throw std::runtime_error("Unsupported buffer type");
    }
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create in1 buffer");

    cl_mem in2{};
    if (bt == bufferType::BUFFER) {
        in2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * _in2.size(), NULL, &retCode);
    } else if (bt == bufferType::IMAGE) {
        cl_image_format format{};
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;
        cl_image_desc desc{};
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = col2;
        desc.image_height = row2;

        in2 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &retCode);
    } else {
        throw std::runtime_error("Unsupported buffer type");
    }
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create in2 buffer");

    cl_mem out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * row1 * col2, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create out buffer");

    if (bt == bufferType::BUFFER) {
        if (clEnqueueWriteBuffer(queue, in1, CL_TRUE, 0, sizeof(float) * _in1.size(), _in1.data(), 0, NULL, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't write to in1 BUFFER");
        if (clEnqueueWriteBuffer(queue, in2, CL_TRUE, 0, sizeof(float) * _in2.size(), _in2.data(), 0, NULL, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't write to in2 BUFFER");
    } else if (bt == bufferType::IMAGE) {
        const size_t origin[3]{ 0, 0, 0 };
        const size_t region1[3]{col1, row1, 1};
        retCode = clEnqueueWriteImage(queue, in1, CL_TRUE, origin, region1, 0, 0, _in1.data(), 0, nullptr, nullptr);
        if (retCode != CL_SUCCESS)
            throw std::runtime_error("Can't write to in1 IMAGE " + std::to_string(retCode));

        const size_t region2[3]{ col2, row2, 1 };
        retCode = clEnqueueWriteImage(queue, in2, CL_TRUE, origin, region2, 0, 0, _in2.data(), 0, nullptr, nullptr);
        if (retCode != CL_SUCCESS)
            throw std::runtime_error("Can't write to in2 IMAGE " + std::to_string(retCode));
    } else {
        throw std::runtime_error("Unsupported buffer type for writing");
    }

    if (clSetKernelArg(kernel, 0, sizeof(cl_mem), &in1) != CL_SUCCESS)
        throw std::runtime_error("Can't set 0 kernel arg");
    if (clSetKernelArg(kernel, 1, sizeof(cl_mem), &in2) != CL_SUCCESS)
        throw std::runtime_error("Can't set 1 kernel arg");
    if (clSetKernelArg(kernel, 2, sizeof(cl_mem), &out) != CL_SUCCESS)
        throw std::runtime_error("Can't set 2 kernel arg");
    if (clSetKernelArg(kernel, 3, sizeof(unsigned int), &col1) != CL_SUCCESS)
        throw std::runtime_error("Can't set 3 kernel arg");
    if (clSetKernelArg(kernel, 4, sizeof(unsigned int), &row1) != CL_SUCCESS)
        throw std::runtime_error("Can't set 4 kernel arg");
    if (clSetKernelArg(kernel, 5, sizeof(unsigned int), &col2) != CL_SUCCESS)
        throw std::runtime_error("Can't set 5 kernel arg");
    if (clSetKernelArg(kernel, 6, sizeof(unsigned int), &row2) != CL_SUCCESS)
        throw std::runtime_error("Can't set 6 kernel arg");

    size_t globalWorkSize[]{row1, col2};
    size_t localWorkSize[]{ BLOCK_SIZE, BLOCK_SIZE };
    double start = omp_get_wtime();
    retCode = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (retCode != CL_SUCCESS) {
        std::string err = "Can't run kernel execution: " + std::to_string(retCode);
        throw std::runtime_error(err);
    }
        
    cl_int ret = clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "Execution time: " << (end - start) << std::endl;

    _out.resize(row1 * col2);
    if (clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(float) * row1 * col2, _out.data(), 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't read from buffer");


    clReleaseMemObject(in1);
    clReleaseMemObject(in2);
    clReleaseMemObject(out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    const unsigned int col1 = 1024;
    const unsigned int row1 = 1024;
    const unsigned int col2 = 1024;
    const unsigned int row2 = 1024;
    const std::vector<float> in1 = getMatrix(col1 * row1);
    const std::vector<float> in2 = getMatrix(col2 * row2);

    //std::vector<float> ref = reference(in1, in2, col1, row1, col2, row2);

    try {
        cl_device_type deviceTypeGPU = CL_DEVICE_TYPE_GPU;
        cl_device_type deviceTypeCPU = CL_DEVICE_TYPE_CPU;
        cl_platform_id platform{};
        getPlatform(platform);
        std::vector<char> kernelText;
        readKernelFile(kernelText);
        kernelText.push_back(0);

        // Task 1
        // GPU
        {
            std::vector<float> out;
            std::cout << "Slow simple GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "slowSimpleGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        {
            std::vector<float> out;
            std::cout << "Simple GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "simpleGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        // CPU
        {
            std::vector<float> out;
            std::cout << "Simple GEMM CPU" << std::endl;
            computeOnDevice(platform, deviceTypeCPU, kernelText, "simpleGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        std::cout << std::endl << std::endl;
        {
            std::vector<float> out;
            std::cout << "Simple GEMM Open MP" << std::endl;
            computeOMP(in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        std::cout << std::endl << std::endl;

        // Task 2
        // GPU
        {
            std::vector<float> out;
            std::cout << "Slow opt GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "slowOptGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        {
            std::vector<float> out;
            std::cout << "Opt GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "optGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        // CPU
        {
            std::vector<float> out;
            std::cout << "Opt GEMM CPU" << std::endl;
            computeOnDevice(platform, deviceTypeCPU, kernelText, "optGemm", in1, in2, out, col1, row1, col2, row2);
            //compare(ref, out);
        }
        std::cout << std::endl << std::endl;

        // Task 3
        // GPU
        {
            std::vector<float> out;
            std::cout << "Slow image GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "slowImageGemm", in1, in2, out, col1, row1, col2, row2, bufferType::IMAGE);
            //compare(ref, out);
        }
        {
            std::vector<float> out;
            std::cout << "Image GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "imageGemm", in1, in2, out, col1, row1, col2, row2, bufferType::IMAGE);
            //compare(ref, out);
        }
        // CPU
        {
            std::vector<float> out;
            std::cout << "Image GEMM CPU" << std::endl;
            computeOnDevice(platform, deviceTypeCPU, kernelText, "imageGemm", in1, in2, out, col1, row1, col2, row2, bufferType::IMAGE);
            //compare(ref, out);
        }
        

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
