#include "metal/metal_kernel.h"
#include "metal/metal_device.h"
#include <stdexcept>

namespace zyraai {
namespace metal {

MetalKernel::MetalKernel(MetalDeviceRef device, const std::string& functionName) {
    NSError* error = nil;
    NSString* name = [NSString stringWithUTF8String:functionName.c_str()];
    auto library = [device newLibraryWithFile:@"/Volumes/ziroCTO/Dev/ZyraAI/build/test_kernel.metallib" error:&error];
    
    if (!library) {
        throw std::runtime_error("Failed to load Metal library");
    }
    
    auto function = [library newFunctionWithName:name];
    
    if (!function) {
        throw std::runtime_error("Failed to create Metal function");
    }
    
    pipelineState_ = [device newComputePipelineStateWithFunction:function error:&error];
    
    if (error) {
        throw std::runtime_error("Failed to create compute pipeline state");
    }
}

MetalKernel::~MetalKernel() {
    if (pipelineState_) {
        [pipelineState_ release];
    }
}

void MetalKernel::setBuffer(MetalBufferRef buffer, uint32_t index) {
    if (index >= buffers_.size()) {
        buffers_.resize(index + 1);
    }
    buffers_[index] = buffer;
}

void MetalKernel::setThreadgroupSize(MetalSize size) {
    threadgroupSize_ = size;
}

void MetalKernel::setGridSize(MetalSize size) {
    gridSize_ = size;
}

void MetalKernel::execute(MetalCommandQueueRef commandQueue) {
    auto commandBuffer = [commandQueue commandBuffer];
    auto computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:pipelineState_];
    
    for (size_t i = 0; i < buffers_.size(); ++i) {
        if (buffers_[i]) {
            [computeEncoder setBuffer:buffers_[i] offset:0 atIndex:i];
        }
    }
    
    MTLSize mtlGridSize = MTLSizeMake(gridSize_.width, gridSize_.height, gridSize_.depth);
    MTLSize mtlThreadgroupSize = MTLSizeMake(threadgroupSize_.width, threadgroupSize_.height, threadgroupSize_.depth);
    
    [computeEncoder dispatchThreads:mtlGridSize threadsPerThreadgroup:mtlThreadgroupSize];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

} // namespace metal
} // namespace zyraai