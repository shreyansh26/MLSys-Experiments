#include "cuda_utils.cuh"

void check_cuda(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void cuda_device_info() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
           deviceId, props.name, props.major, props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.regsPerBlock,
           props.regsPerMultiprocessor,
           props.totalGlobalMem >> 20,
           props.sharedMemPerBlock >> 10,
           props.sharedMemPerMultiprocessor >> 10,
           props.totalConstMem >> 10,
           props.multiProcessorCount,
           props.warpSize);
}

int cdiv(int a, int b) {
    return (a + b - 1) / b;
}