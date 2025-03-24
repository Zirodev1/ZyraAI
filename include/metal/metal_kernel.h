#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
#include <cstdint>
#endif

#include "metal/metal_device.h"
#include <string>
#include <vector>

namespace zyraai {
namespace metal {

struct MetalSize {
  uint32_t width;
  uint32_t height;
  uint32_t depth;
};

class MetalKernel {
public:
  MetalKernel(MetalDeviceRef device, const std::string &functionName);
  ~MetalKernel();

  // Set buffer at index
  void setBuffer(MetalBufferRef buffer, uint32_t index);

  // Set threadgroup size
  void setThreadgroupSize(MetalSize size);

  // Set grid size
  void setGridSize(MetalSize size);

  // Execute kernel
  void execute(MetalCommandQueueRef commandQueue);

  // Get compute pipeline state
  MetalComputePipelineStateRef getPipelineState() const {
    return pipelineState_;
  }

private:
  MetalComputePipelineStateRef pipelineState_;
  MetalSize threadgroupSize_;
  MetalSize gridSize_;
  std::vector<MetalBufferRef> buffers_;

  // Prevent copying
  MetalKernel(const MetalKernel &) = delete;
  MetalKernel &operator=(const MetalKernel &) = delete;
};

} // namespace metal
} // namespace zyraai