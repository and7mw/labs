#pragma once

#include <CL/cl.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <fstream>
#include <string>

void getPlatform(cl_platform_id& platform) {
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS)
        throw std::runtime_error("Can't get number platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get platforms");

    for (cl_uint i = 0; i < numPlatforms; i++) {
        const size_t paramValueSize = 100;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get platforms info");
        name[parValRetSize] = '\0';
        if (strstr(name, "Intel") != nullptr) {
            platform = platforms[i];
            return;
        }
    }

    throw std::runtime_error("Can't find Intel platform");
}

void getDevice(const cl_platform_id& platform, const cl_device_type& dt, cl_device_id& device) {
    cl_uint numDevices = 0;
    if (clGetDeviceIDs(platform, dt, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices > 1)
        throw std::runtime_error("Unsupport more than one device");

    if (clGetDeviceIDs(platform, dt, numDevices, &device, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get devices");
}

void createContext(const cl_platform_id& platform, const cl_device_id& device, cl_context& context) {
    cl_context_properties contextProp[3]{ CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_int retCode = 0;
    context = clCreateContext(contextProp, 1, &device, NULL, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create context");
}

void createQueue(const cl_context& context, const cl_device_id& device, cl_command_queue& queue) {
    cl_int retCode = 0;
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create queue");
}

void readKernelFile(std::vector<char>& kernelText) {
    std::ifstream desc("axpy.cl", std::ios_base::ate | std::ios_base::binary);
    std::streamoff fileSize = desc.tellg();
    if (fileSize == -1)
        throw std::runtime_error("Can't read kernel file");

    desc.seekg(0, std::ios_base::beg);
    kernelText.resize(fileSize);
    desc.read(&kernelText[0], fileSize);
}

void createProgramAndKernel(const cl_context& context, const cl_device_id& device, cl_program& program, cl_kernel& kernel,
    const std::string kernelName) {
    std::vector<char> kernelText;
    readKernelFile(kernelText);
    kernelText.push_back(0);
    const char* rawKernelText = &kernelText[0];

    cl_int retCode;
    program = clCreateProgramWithSource(context, 1, &rawKernelText, 0, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create program with source");
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't build program");

    kernel = clCreateKernel(program, kernelName.c_str(), &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create kernel");
}

void createMemoryObject(const cl_context& context, cl_mem& x, cl_mem& y, const size_t& size, const size_t dataSize) {
    cl_int retCode;
    x = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize * size, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create input buffer");
    y = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize * size, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create output buffer");
}

void writeToBuffer(const cl_command_queue& queue, cl_mem& x, const void* buffer, const size_t& size, const size_t dataSize) {
    if (clEnqueueWriteBuffer(queue, x, CL_TRUE, 0, dataSize * size, buffer, 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't write to buffer");
}

template <typename dataType>
void setArguments(const cl_kernel& kernel, const int& n, const dataType& a, const cl_mem& x, const int& incx, const cl_mem& y, const int& incy) {
    if (clSetKernelArg(kernel, 0, sizeof(int), &n) != CL_SUCCESS)
        throw std::runtime_error("Can't set 0 kernel arg");
    if (clSetKernelArg(kernel, 1, sizeof(dataType), &a) != CL_SUCCESS)
        throw std::runtime_error("Can't set 1 kernel arg");
    if (clSetKernelArg(kernel, 2, sizeof(cl_mem), &x) != CL_SUCCESS)
        throw std::runtime_error("Can't set 2 kernel arg");
    if (clSetKernelArg(kernel, 3, sizeof(int), &incx) != CL_SUCCESS)
        throw std::runtime_error("Can't set 3 kernel arg");
    if (clSetKernelArg(kernel, 4, sizeof(cl_mem), &y) != CL_SUCCESS)
        throw std::runtime_error("Can't set 4 kernel arg");
    if (clSetKernelArg(kernel, 5, sizeof(int), &incy) != CL_SUCCESS)
        throw std::runtime_error("Can't set 5 kernel arg");
}

void execute(const cl_command_queue& queue, cl_kernel& kernel, const size_t& globalWorkSize, const size_t& localWorkSize) {
    if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't run kernel execution");
}

