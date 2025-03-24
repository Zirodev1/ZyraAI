#include <metal_stdlib>
using namespace metal;

kernel void test_kernel(device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    output[index] = input[index] + 1.0f;
} 