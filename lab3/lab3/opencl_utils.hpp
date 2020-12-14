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

void readKernelFile(std::vector<char>& kernelText) {
    std::ifstream desc("kernels.cl", std::ios_base::ate | std::ios_base::binary);
    std::streamoff fileSize = desc.tellg();
    if (fileSize == -1)
        throw std::runtime_error("Can't read kernel file");

    desc.seekg(0, std::ios_base::beg);
    kernelText.resize(fileSize);
    desc.read(&kernelText[0], fileSize);
}

void getDevice(const cl_platform_id& platform, const cl_device_type& dt, cl_device_id& device) {
    cl_uint numDevices = 0;
    if (clGetDeviceIDs(platform, dt, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices > 1)
        throw std::runtime_error("Unsupport more than one device");

    if (clGetDeviceIDs(platform, dt, numDevices, &device, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get devices");
    /*size_t ret = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(ret), &ret, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get devices CL_DEVICE_MAX_WORK_GROUP_SIZE");
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << ret << std::endl;*/
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

void createProgramAndKernel(const cl_context& context, const cl_device_id& device, cl_program& program, cl_kernel& kernel,
                            const std::vector<char>& kernelText, const std::string kernelName) {
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

void compare(const std::vector<float>& res1, const std::vector<float>& res2) {
    if (res1.size() != res2.size()) {
        std::cout << "Vectors have different size" << std::endl;
        return;
    }
    for (size_t i = 0; i < res1.size(); i++) {
        if (std::abs(res1[i] - res2[i]) > 0.01f) {
            std::cout << "Different result on res1: " << res1[i] << " and res2: " << res2[i] << " on idx: " << i << std::endl;
            return;
        }
    }
}

