#include "metal/metal_buffer.h"
#include "metal/metal_device.h"
#include "metal/metal_kernel.h"
#include <gtest/gtest.h>

using namespace ::std;

namespace zyraai {
namespace metal {
namespace test {

class MetalTest : public ::testing::Test {
protected:
  void SetUp() override {
    device_ = ::std::make_unique<MetalDevice>();
    ASSERT_TRUE(device_->initialize());
  }

  void TearDown() override { device_.reset(); }

  ::std::unique_ptr<MetalDevice> device_;
};

TEST_F(MetalTest, DeviceInitialization) {
  ASSERT_NE(device_->getDevice(), nullptr);
  ASSERT_NE(device_->getCommandQueue(), nullptr);
}

TEST_F(MetalTest, BufferCreation) {
  const size_t size = 1024;
  auto buffer = ::std::make_unique<MetalBuffer>(device_->getDevice(), size, 0);
  ASSERT_NE(buffer->getContents(), nullptr);
}

TEST_F(MetalTest, BufferDataTransfer) {
  const size_t size = sizeof(float) * 4;
  auto buffer = ::std::make_unique<MetalBuffer>(device_->getDevice(), size, 0);

  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  buffer->copyData(data, size);

  float result[4] = {0.0f};
  buffer->getData(result, size);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(data[i], result[i]);
  }
}

TEST_F(MetalTest, KernelCreation) {
  auto kernel =
      ::std::make_unique<MetalKernel>(device_->getDevice(), "test_kernel");
  ASSERT_NE(kernel->getPipelineState(), nullptr);
}

TEST_F(MetalTest, KernelExecution) {
  const size_t size = sizeof(float) * 4;
  auto input_buffer =
      ::std::make_unique<MetalBuffer>(device_->getDevice(), size, 0);
  auto output_buffer =
      ::std::make_unique<MetalBuffer>(device_->getDevice(), size, 0);

  float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  input_buffer->copyData(input_data, size);

  auto kernel =
      ::std::make_unique<MetalKernel>(device_->getDevice(), "test_kernel");
  ASSERT_NE(kernel->getPipelineState(), nullptr);

  // Set buffers
  kernel->setBuffer(input_buffer->getBuffer(), 0);
  kernel->setBuffer(output_buffer->getBuffer(), 1);

  // Set threadgroup size
  MetalSize threadgroup_size = {4, 1, 1};
  kernel->setThreadgroupSize(threadgroup_size);

  // Set grid size
  MetalSize grid_size = {4, 1, 1};
  kernel->setGridSize(grid_size);

  // Execute kernel
  kernel->execute(device_->getCommandQueue());

  float result[4] = {0.0f};
  output_buffer->getData(result, size);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(input_data[i] + 1.0f, result[i]);
  }
}

} // namespace test
} // namespace metal
} // namespace zyraai