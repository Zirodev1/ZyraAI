#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#else
#include <cstdint>
#endif

#include <memory>
#include <string>

namespace zyraai {
namespace metal {

#ifdef __OBJC__
using MetalDeviceRef = id<MTLDevice>;
using MetalCommandQueueRef = id<MTLCommandQueue>;
using MetalLibraryRef = id<MTLLibrary>;
using MetalBufferRef = id<MTLBuffer>;
using MetalComputePipelineStateRef = id<MTLComputePipelineState>;
using MetalFunctionRef = id<MTLFunction>;
#else
using MetalDeviceRef = void *;
using MetalCommandQueueRef = void *;
using MetalLibraryRef = void *;
using MetalBufferRef = void *;
using MetalComputePipelineStateRef = void *;
using MetalFunctionRef = void *;
#endif

class MetalDevice {
public:
  MetalDevice();
  ~MetalDevice();

  // Initialize Metal device and command queue
  bool initialize();

  // Get Metal device and command queue
  MetalDeviceRef getDevice() const { return device_; }
  MetalCommandQueueRef getCommandQueue() const { return commandQueue_; }

  // Create a new Metal buffer
  MetalBufferRef createBuffer(size_t size, uint32_t options = 0);

  // Load Metal shader library
  MetalLibraryRef loadLibrary(const std::string &libraryPath);

  // Create compute pipeline state
  MetalComputePipelineStateRef
  createComputePipelineState(const std::string &functionName);

private:
  MetalDeviceRef device_;
  MetalCommandQueueRef commandQueue_;
  MetalLibraryRef defaultLibrary_;

  // Prevent copying
  MetalDevice(const MetalDevice &) = delete;
  MetalDevice &operator=(const MetalDevice &) = delete;
};

} // namespace metal
} // namespace zyraai