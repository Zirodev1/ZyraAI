#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
#include <cstdint>
#endif

#include "metal/metal_device.h"
#include <memory>
#include <vector>

namespace zyraai {
namespace metal {

class MetalBuffer {
public:
  MetalBuffer(MetalDeviceRef device, size_t size, uint32_t options = 0);
  ~MetalBuffer();

  // Copy data to buffer
  void copyData(const void *data, size_t size);

  // Copy data from buffer
  void getData(void *data, size_t size) const;

  // Get underlying Metal buffer
  MetalBufferRef getBuffer() const { return buffer_; }

  // Get buffer size
  size_t getSize() const { return size_; }

  // Get buffer contents
  void *getContents() const;

private:
  MetalBufferRef buffer_;
  size_t size_;

  // Prevent copying
  MetalBuffer(const MetalBuffer &) = delete;
  MetalBuffer &operator=(const MetalBuffer &) = delete;
};

} // namespace metal
} // namespace zyraai