#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

#include "opencl_utils.hpp"

#define BLOCK_SIZE 2

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
    }
    else if (bt == bufferType::IMAGE) {
        cl_image_format format{};
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;
        cl_image_desc desc{};
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = col1;
        desc.image_height = row1;

        in1 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &retCode);
    }
    else {
        throw std::runtime_error("Unsupported buffer type");
    }
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create in1 buffer");

    cl_mem in2{};
    if (bt == bufferType::BUFFER) {
        in2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * _in2.size(), NULL, &retCode);
    }
    else if (bt == bufferType::IMAGE) {
        cl_image_format format{};
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;
        cl_image_desc desc{};
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = col2;
        desc.image_height = row2;

        in2 = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, nullptr, &retCode);
    }
    else {
        throw std::runtime_error("Unsupported buffer type");
    }
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create in2 buffer");

    cl_mem out{};
    if (bt == bufferType::BUFFER) {
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * row1 * col2, NULL, &retCode);
    } else if (bt == bufferType::IMAGE) {
        cl_image_format format{};
        format.image_channel_order = CL_R;
        format.image_channel_data_type = CL_FLOAT;
        cl_image_desc desc{};
        memset(&desc, 0, sizeof(desc));
        desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        desc.image_width = col2;
        desc.image_height = row1;

        out = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, nullptr, &retCode);
    } else {
        throw std::runtime_error("Unsupported buffer type");
    }
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
    if (clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't run kernel execution");
    cl_int ret = clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "Execution time: " << (end - start) << std::endl;

    _out.resize(row1 * col2);
    if (bt == bufferType::BUFFER) {
        if (clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(float) * row1 * col2, _out.data(), 0, NULL, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't write to buffer BUFFER");
    } else if (bt == bufferType::IMAGE) {
        const size_t origin[3]{ 0, 0, 0 };
        const size_t region[3]{ col2, row1, 1 };
        if (clEnqueueReadImage(queue, out, CL_TRUE, origin, region, 0, 0, _out.data(), 0, nullptr, nullptr) != CL_SUCCESS)
            throw std::runtime_error("Can't write to buffer IMAGE");
    } else {
        throw std::runtime_error("Unsupported buffer type for writing");
    }

    clReleaseMemObject(in1);
    clReleaseMemObject(in2);
    clReleaseMemObject(out);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    const unsigned int col1 = 4;
    const unsigned int row1 = 4;
    const unsigned int col2 = 4;
    const unsigned int row2 = 4;
    const std::vector<float> in1 = getMatrix(col1 * row1);
    const std::vector<float> in2 = getMatrix(col2 * row2);

    try {
        cl_device_type deviceTypeGPU = CL_DEVICE_TYPE_GPU;
        cl_device_type deviceTypeCPU = CL_DEVICE_TYPE_CPU;
        cl_platform_id platform{};
        getPlatform(platform);
        std::vector<char> kernelText;
        readKernelFile(kernelText);
        kernelText.push_back(0);

        std::cout << "MATRIX 1" << std::endl;
        for (size_t row = 0; row < row1; row++) {
            for (size_t col = 0; col < col1; col++) {
                std::cout << in1[row * col1 + col] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "MATRIX 2" << std::endl;
        for (size_t row = 0; row < row2; row++) {
            for (size_t col = 0; col < col2; col++) {
                std::cout << in2[row * col2 + col] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Task 1
        /*{
            std::vector<float> out;
            std::cout << "Simple GEMM CPU" << std::endl;
            computeOnDevice(platform, deviceTypeCPU, kernelText, "simpleGemm", in1, in2, out, col1, row1, col2, row2);
        }*/
        //std::vector<float> out1;
        //{
        //    //std::vector<float> out;
        //    std::cout << "Simple GEMM GPU" << std::endl;
        //    computeOnDevice(platform, deviceTypeGPU, kernelText, "simpleGemm", in1, in2, out1, col1, row1, col2, row2);
        //}
        /*{
            std::vector<float> out;
            std::cout << "Simple GEMM Open MP" << std::endl;
            computeOMP(in1, in2, out, col1, row1, col2, row2);
        }
        std::cout << std::endl;*/

        // Task 2
        /*{
            std::vector<float> out;
            std::cout << "Opt GEMM CPU" << std::endl;
            computeOnDevice(platform, deviceTypeCPU, kernelText, "optGemm", in1, in2, out, col1, row1, col2, row2);
        }
        {
            std::vector<float> out;
            std::cout << "Opt GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "optGemm", in1, in2, out, col1, row1, col2, row2);
        }
        std::cout << std::endl;*/

        // Task 3
        std::vector<float> out2;
        {
            //std::vector<float> out;
            std::cout << "Image GEMM GPU" << std::endl;
            computeOnDevice(platform, deviceTypeGPU, kernelText, "imageGemm", in1, in2, out2, col1, row1, col2, row2, bufferType::IMAGE);
        }
        //compare(out1, out2);

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
