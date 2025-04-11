#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <device_launch_parameters.h>

namespace zyraai {
namespace cuda {

// Check if CUDA is available
bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

// Get CUDA device properties
cudaDeviceProp getDeviceProperties(int deviceId = 0) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    return props;
}

// Print CUDA device information
void printDeviceInfo() {
    if (!isCudaAvailable()) {
        printf("CUDA not available. Using CPU.\n");
        return;
    }
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("CUDA Devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        
        printf("Device %d: %s\n", i, props.name);
        printf("  Compute capability: %d.%d\n", props.major, props.minor);
        printf("  Memory: %.2f GB\n", props.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Multiprocessors: %d\n", props.multiProcessorCount);
        printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
        printf("  Memory Clock Rate (KHz): %d\n", props.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", props.memoryBusWidth);
        printf("  L2 Cache Size (bytes): %d\n\n", props.l2CacheSize);
    }
}

// Initialize CUDA device
bool initCuda(int deviceId = 0) {
    if (!isCudaAvailable()) {
        return false;
    }
    
    cudaError_t error = cudaSetDevice(deviceId);
    return (error == cudaSuccess);
}

// Memory management helpers
template<typename T>
T* allocateDeviceMemory(size_t size) {
    T* devPtr;
    cudaMalloc((void**)&devPtr, size * sizeof(T));
    return devPtr;
}

template<typename T>
void copyToDevice(const T* hostPtr, T* devicePtr, size_t size) {
    cudaMemcpy(devicePtr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void copyToHost(T* hostPtr, const T* devicePtr, size_t size) {
    cudaMemcpy(hostPtr, devicePtr, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void freeDeviceMemory(T* devicePtr) {
    cudaFree(devicePtr);
}

// Simple kernel to multiply matrix elements by a scalar
template <typename T>
__global__ void scalarMultiply(T* matrix, T scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scalar;
    }
}

// Generic kernel launch helper
template <typename T>
void launchScalarMultiply(T* deviceData, T scalar, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Use CUDA syntax only in CUDA mode
    #ifdef USE_CUDA
    scalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(deviceData, scalar, size);
    cudaDeviceSynchronize();
    #endif
}

} // namespace cuda
} // namespace zyraai

#else

// Stub implementations for when CUDA is not available
namespace zyraai {
namespace cuda {

// Check if CUDA is available
inline bool isCudaAvailable() { return false; }

// Print CUDA device information
inline void printDeviceInfo() { 
    printf("CUDA support not compiled. Using CPU.\n"); 
}

// Initialize CUDA device
inline bool initCuda(int deviceId = 0) { return false; }

// Stub memory management helpers
template<typename T>
T* allocateDeviceMemory(size_t size) { return nullptr; }

template<typename T>
void copyToDevice(const T* hostPtr, T* devicePtr, size_t size) {}

template<typename T>
void copyToHost(T* hostPtr, const T* devicePtr, size_t size) {}

template<typename T>
void freeDeviceMemory(T* devicePtr) {}

// Stub kernel launcher
template <typename T>
void launchScalarMultiply(T* deviceData, T scalar, int size) {}

} // namespace cuda
} // namespace zyraai

#endif // USE_CUDA 