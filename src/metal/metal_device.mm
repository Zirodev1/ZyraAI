#include "metal/metal_device.h"
#include <stdexcept>

namespace zyraai {
namespace metal {

MetalDevice::MetalDevice()
    : device_(nullptr), commandQueue_(nullptr), defaultLibrary_(nullptr) {}

MetalDevice::~MetalDevice() {
    if (defaultLibrary_) {
        [defaultLibrary_ release];
    }
    
    if (commandQueue_) {
        [commandQueue_ release];
    }
    
    if (device_) {
        [device_ release];
    }
}

bool MetalDevice::initialize() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        return false;
    }
    
    commandQueue_ = [device_ newCommandQueue];
    if (!commandQueue_) {
        return false;
    }
    
    // Load our compiled Metal shader library
    NSString* libraryPath = @"/Volumes/ziroCTO/Dev/ZyraAI/build/test_kernel.metallib";
    NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
    
    NSError* error = nil;
    defaultLibrary_ = [device_ newLibraryWithURL:libraryURL error:&error];
    if (!defaultLibrary_) {
        return false;
    }

    return true;
}

MetalBufferRef MetalDevice::createBuffer(size_t size, uint32_t options) {
    return [device_ newBufferWithLength:size options:options];
}

MetalLibraryRef MetalDevice::loadLibrary(const std::string& libraryPath) {
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:libraryPath.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path];
    MetalLibraryRef library = [device_ newLibraryWithURL:url error:&error];
    
    if (error) {
        return nullptr;
    }
    
    return library;
}

MetalComputePipelineStateRef MetalDevice::createComputePipelineState(const std::string& functionName) {
    NSError* error = nil;
    NSString* name = [NSString stringWithUTF8String:functionName.c_str()];
    MetalFunctionRef function = [defaultLibrary_ newFunctionWithName:name];
    
    if (!function) {
        return nullptr;
    }
    
    MetalComputePipelineStateRef pipelineState = [device_ newComputePipelineStateWithFunction:function error:&error];
    
    if (error) {
        return nullptr;
    }
    
    return pipelineState;
}

} // namespace metal
} // namespace zyraai