#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include <cstring>

namespace zyraai {
namespace metal {

MetalBuffer::MetalBuffer(MetalDeviceRef device, size_t size, uint32_t options)
    : buffer_(nullptr), size_(size) {
    buffer_ = [device newBufferWithLength:size options:options];
}

MetalBuffer::~MetalBuffer() {
    if (buffer_) {
        [buffer_ release];
    }
}

void MetalBuffer::copyData(const void* data, size_t size) {
    if (!buffer_ || size > size_) {
        return;
    }
    
    void* contents = [buffer_ contents];
    std::memcpy(contents, data, size);
}

void MetalBuffer::getData(void* data, size_t size) const {
    if (!buffer_ || size > size_) {
        return;
    }
    
    const void* contents = [buffer_ contents];
    std::memcpy(data, contents, size);
}

void* MetalBuffer::getContents() const {
    return buffer_ ? [buffer_ contents] : nullptr;
}

} // namespace metal
} // namespace zyraai