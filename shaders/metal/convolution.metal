#include <metal_stdlib>
using namespace metal;

struct ConvolutionParams {
    uint inputWidth;
    uint inputHeight;
    uint inputChannels;
    uint outputWidth;
    uint outputHeight;
    uint outputChannels;
    uint kernelSize;
    uint stride;
    uint padding;
};

kernel void convolution2d(
    texture2d<float, access::read> input [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant float* weights [[buffer(0)]],
    constant float* biases [[buffer(1)]],
    constant ConvolutionParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (gid.x >= params.outputWidth || gid.y >= params.outputHeight) {
        return;
    }

    float sum = 0.0f;
    
    // For each input channel
    for (uint ic = 0; ic < params.inputChannels; ic++) {
        // For each kernel element
        for (uint ky = 0; ky < params.kernelSize; ky++) {
            for (uint kx = 0; kx < params.kernelSize; kx++) {
                // Calculate input position
                int inX = int(gid.x) * int(params.stride) + int(kx) - int(params.padding);
                int inY = int(gid.y) * int(params.stride) + int(ky) - int(params.padding);
                
                // Check bounds
                if (inX >= 0 && inX < int(params.inputWidth) && inY >= 0 && inY < int(params.inputHeight)) {
                    float inputValue = input.read(uint2(inX, inY), ic).r;
                    float weightValue = weights[ic * params.kernelSize * params.kernelSize + ky * params.kernelSize + kx];
                    sum += inputValue * weightValue;
                }
            }
        }
    }
    
    // Add bias
    sum += biases[tid];
    
    // Write output
    output.write(float4(sum, 0.0f, 0.0f, 1.0f), gid);
} 