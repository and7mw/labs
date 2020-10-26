#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void platformInfo() {
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS)
        throw std::runtime_error("Can't get number platforms");

    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get platforms");
std::cout << "Platfroms:" << std::endl;
    for (cl_uint i = 0; i < platforms.size(); i++) {
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get platforms info");
        name[parValRetSize] = '\0';
        std::cout << "\t" << std::string(name) << std::endl;
    }
    std::cout << "Devices:" << std::endl;
    cl_uint numDevices = 0;
    if (clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices == 1) {
        cl_device_id device;
        if (clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, numDevices, &device, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't get device");
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetDeviceInfo(device, CL_DEVICE_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get number devices");
        name[parValRetSize] = '\0';
        std::cout << "\t" << std::string(name) << std::endl;
    }
    if (clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices == 1) {
        cl_device_id device;
        if (clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, &device, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't get device");
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetDeviceInfo(device, CL_DEVICE_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get number devices");
        name[parValRetSize] = '\0';
        std::cout << "\t" << std::string(name) << std::endl;
    }
    if (clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices == 1) {
        cl_device_id device;
        if (clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, numDevices, &device, NULL) != CL_SUCCESS)
            throw std::runtime_error("Can't get device");
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetDeviceInfo(device, CL_DEVICE_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get number devices");
        name[parValRetSize] = '\0';
        std::cout << "\t" << std::string(name) << std::endl;
    }
}

void print() {
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS)
        throw std::runtime_error("Can't get number platforms");
    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get platforms");
    cl_platform_id platform{};
    for (cl_uint i = 0; i < platforms.size(); i++) {
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get platforms info");
        name[parValRetSize] = '\0';
        if (std::string(name) == "Intel(R) OpenCL") {
            platform = platforms[i];
            break;
        }
    }

    cl_uint numDevices = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices > 1)
        throw std::runtime_error("Unsupport more than one device");
    cl_device_id device;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, &device, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get devices");
    
    cl_context_properties contextProp[3]{ CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_int retCode = 0;
    cl_context context = clCreateContext(contextProp, 1, &device, NULL, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create context");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create queue");

    std::vector<char> kernelText;
    std::ifstream desc("kernelPrint.cl", std::ios_base::ate | std::ios_base::binary);
    std::streamoff fileSize = desc.tellg();
    if (fileSize == -1)
        throw std::runtime_error("Can't read kernel file");
    desc.seekg(0, std::ios_base::beg);
    kernelText.resize(fileSize);
    desc.read(&kernelText[0], fileSize);
    kernelText.push_back(0);
    const char* rawKernelText = &kernelText[0];

    cl_program program = clCreateProgramWithSource(context, 1, &rawKernelText, 0, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create program with source");
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't build program");

    cl_kernel kernel = clCreateKernel(program, "kernelPrint", &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create kernel");

    size_t globalWorkSize = 100;
    size_t localWorkSize = 20;
    if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't run kernel execution");
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
}

void add() {
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, NULL, &numPlatforms) != CL_SUCCESS)
        throw std::runtime_error("Can't get number platforms");
    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get platforms");
    cl_platform_id platform{};
    for (cl_uint i = 0; i < platforms.size(); i++) {
        const size_t paramValueSize = 200;
        size_t parValRetSize = 0;
        char name[paramValueSize];
        if (clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, paramValueSize, name, &parValRetSize) != CL_SUCCESS)
            throw std::runtime_error("Can't get platforms info");
        name[parValRetSize] = '\0';
        if (std::string(name) == "Intel(R) OpenCL") {
            platform = platforms[i];
            break;
        }
    }

    cl_uint numDevices = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices) != CL_SUCCESS)
        throw std::runtime_error("Can't get number devices");
    if (numDevices > 1)
        throw std::runtime_error("Unsupport more than one device");
    cl_device_id device;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, &device, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't get devices");

    cl_context_properties contextProp[3]{ CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_int retCode = 0;
    cl_context context = clCreateContext(contextProp, 1, &device, NULL, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create context");

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create queue");

    std::vector<char> kernelText;
    std::ifstream desc("kernelAdd.cl", std::ios_base::ate | std::ios_base::binary);
    std::streamoff fileSize = desc.tellg();
    if (fileSize == -1)
        throw std::runtime_error("Can't read kernel file");
    desc.seekg(0, std::ios_base::beg);
    kernelText.resize(fileSize);
    desc.read(&kernelText[0], fileSize);
    kernelText.push_back(0);
    const char* rawKernelText = &kernelText[0];

    cl_program program = clCreateProgramWithSource(context, 1, &rawKernelText, 0, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create program with source");
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't build program");

    cl_kernel kernel = clCreateKernel(program, "kernelAdd", &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create kernel");

    cl_mem buffer;
    const unsigned int vectorSize = 100;
    std::vector<unsigned int> vec(vectorSize, 1);
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * vectorSize, NULL, &retCode);
    if (retCode != CL_SUCCESS)
        throw std::runtime_error("Can't create input buffer");
    if (clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * vectorSize, &vec[0], 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't write to buffer");

    if (clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer) != CL_SUCCESS)
        throw std::runtime_error("Can't set 0 kernel arg");
    if (clSetKernelArg(kernel, 1, sizeof(unsigned int), &vectorSize) != CL_SUCCESS)
        throw std::runtime_error("Can't set 1 kernel arg");

    for (size_t i = 0; i < vec.size(); i++)
        std::cout << vec[i] << " ";
    std::cout << std::endl;

    size_t globalWorkSize = vectorSize;
    size_t localWorkSize = 20;
    if (clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't run kernel execution");

    if (clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * vectorSize, &vec[0], 0, NULL, NULL) != CL_SUCCESS)
        throw std::runtime_error("Can't write to buffer");
    for (size_t i = 0; i < vec.size(); i++)
        std::cout << vec[i] << " ";
    std::cout << std::endl;

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
}

int main() {
    try {
        platformInfo();
        /*print();
        add();*/
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    
    return 0;
}