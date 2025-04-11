#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

namespace zyraai {
namespace cuda {

// CUDA kernel for forward convolution - optimized for CIFAR-10 sized images
__global__ void convForwardKernel(
    const float* input,     // Input data, shape [C_in, H_in, W_in]
    const float* weights,   // Weights, shape [C_out, C_in, K_h, K_w]
    const float* bias,      // Bias, shape [C_out]
    float* output,          // Output data, shape [C_out, H_out, W_out]
    int inChannels,         // Number of input channels
    int outChannels,        // Number of output channels
    int inHeight,           // Input height
    int inWidth,            // Input width
    int kernelHeight,       // Kernel height
    int kernelWidth,        // Kernel width
    int stride,             // Stride
    int padding,            // Padding
    int outHeight,          // Output height
    int outWidth            // Output width
) {
    // Get thread indices
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate position in the output
    if (outIdx < outChannels * outHeight * outWidth) {
        // Calculate output indices
        int oc = outIdx / (outHeight * outWidth);
        int oh = (outIdx % (outHeight * outWidth)) / outWidth;
        int ow = outIdx % outWidth;
        
        // Initialize output value with bias
        float value = bias[oc];
        
        // Calculate convolution
        for (int ic = 0; ic < inChannels; ++ic) {
            for (int kh = 0; kh < kernelHeight; ++kh) {
                for (int kw = 0; kw < kernelWidth; ++kw) {
                    // Calculate input position with padding
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    // Skip if outside input bounds
                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        // Input index
                        int inIdx = (ic * inHeight + ih) * inWidth + iw;
                        
                        // Weight index
                        int weightIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                        
                        // Accumulate result
                        value += input[inIdx] * weights[weightIdx];
                    }
                }
            }
        }
        
        // Store result
        output[outIdx] = value;
    }
}

// CUDA kernel for backward convolution (input gradient)
__global__ void convBackwardInputKernel(
    const float* outputGrad,   // Output gradient, shape [C_out, H_out, W_out]
    const float* weights,      // Weights, shape [C_out, C_in, K_h, K_w]
    float* inputGrad,          // Input gradient, shape [C_in, H_in, W_in]
    int inChannels,            // Number of input channels
    int outChannels,           // Number of output channels
    int inHeight,              // Input height
    int inWidth,               // Input width
    int kernelHeight,          // Kernel height
    int kernelWidth,           // Kernel width
    int stride,                // Stride
    int padding,               // Padding
    int outHeight,             // Output height
    int outWidth               // Output width
) {
    // Get thread indices
    int inIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate position in the input gradient
    if (inIdx < inChannels * inHeight * inWidth) {
        // Calculate input indices
        int ic = inIdx / (inHeight * inWidth);
        int ih = (inIdx % (inHeight * inWidth)) / inWidth;
        int iw = inIdx % inWidth;
        
        // Initialize input gradient
        float gradient = 0.0f;
        
        // Calculate the gradient contribution from each output element
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int kh = 0; kh < kernelHeight; ++kh) {
                for (int kw = 0; kw < kernelWidth; ++kw) {
                    // Calculate output position
                    int oh = (ih - kh + padding) / stride;
                    int ow = (iw - kw + padding) / stride;
                    
                    // Skip if outside output bounds or not aligned with stride
                    if (oh >= 0 && oh < outHeight && ow >= 0 && ow < outWidth &&
                        (ih - kh + padding) % stride == 0 && (iw - kw + padding) % stride == 0) {
                        
                        // Output gradient index
                        int outGradIdx = (oc * outHeight + oh) * outWidth + ow;
                        
                        // Weight index - need to flip the kernel for gradient computation
                        int weightIdx = ((oc * inChannels + ic) * kernelHeight + kernelHeight - 1 - kh) * kernelWidth + (kernelWidth - 1 - kw);
                        
                        // Accumulate gradient
                        gradient += outputGrad[outGradIdx] * weights[weightIdx];
                    }
                }
            }
        }
        
        // Store result
        inputGrad[inIdx] = gradient;
    }
}

// CUDA kernel for backward convolution (weight gradient)
__global__ void convBackwardWeightKernel(
    const float* input,        // Input data, shape [C_in, H_in, W_in]
    const float* outputGrad,   // Output gradient, shape [C_out, H_out, W_out]
    float* weightGrad,         // Weight gradient, shape [C_out, C_in, K_h, K_w]
    int inChannels,            // Number of input channels
    int outChannels,           // Number of output channels
    int inHeight,              // Input height
    int inWidth,               // Input width
    int kernelHeight,          // Kernel height
    int kernelWidth,           // Kernel width
    int stride,                // Stride
    int padding,               // Padding
    int outHeight,             // Output height
    int outWidth               // Output width
) {
    // Get thread indices
    int weightIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = outChannels * inChannels * kernelHeight * kernelWidth;
    
    // Calculate position in the weight gradient
    if (weightIdx < totalWeights) {
        // Calculate weight indices
        int oc = weightIdx / (inChannels * kernelHeight * kernelWidth);
        int ic = (weightIdx % (inChannels * kernelHeight * kernelWidth)) / (kernelHeight * kernelWidth);
        int kh = (weightIdx % (kernelHeight * kernelWidth)) / kernelWidth;
        int kw = weightIdx % kernelWidth;
        
        // Initialize weight gradient
        float gradient = 0.0f;
        
        // Calculate the gradient for this weight
        for (int oh = 0; oh < outHeight; ++oh) {
            for (int ow = 0; ow < outWidth; ++ow) {
                // Calculate input position
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                
                // Skip if outside input bounds
                if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                    // Input index
                    int inIdx = (ic * inHeight + ih) * inWidth + iw;
                    
                    // Output gradient index
                    int outGradIdx = (oc * outHeight + oh) * outWidth + ow;
                    
                    // Accumulate gradient
                    gradient += input[inIdx] * outputGrad[outGradIdx];
                }
            }
        }
        
        // Store result
        weightGrad[weightIdx] = gradient;
    }
}

// CUDA kernel for backward convolution (bias gradient)
__global__ void convBackwardBiasKernel(
    const float* outputGrad,   // Output gradient, shape [C_out, H_out, W_out]
    float* biasGrad,           // Bias gradient, shape [C_out]
    int outChannels,           // Number of output channels
    int outHeight,             // Output height
    int outWidth               // Output width
) {
    // Get thread indices
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate bias gradient for this output channel
    if (oc < outChannels) {
        float gradient = 0.0f;
        
        // Sum over all output positions for this channel
        for (int oh = 0; oh < outHeight; ++oh) {
            for (int ow = 0; ow < outWidth; ++ow) {
                // Output gradient index
                int outGradIdx = (oc * outHeight + oh) * outWidth + ow;
                
                // Accumulate gradient
                gradient += outputGrad[outGradIdx];
            }
        }
        
        // Store result
        biasGrad[oc] = gradient;
    }
}

// Wrapper function to launch forward convolution kernel
void launchConvForward(
    const float* input, 
    const float* weights, 
    const float* bias, 
    float* output,
    int inChannels, 
    int outChannels, 
    int inHeight, 
    int inWidth,
    int kernelHeight, 
    int kernelWidth, 
    int stride, 
    int padding
) {
    // Calculate output dimensions
    int outHeight = (inHeight + 2 * padding - kernelHeight) / stride + 1;
    int outWidth = (inWidth + 2 * padding - kernelWidth) / stride + 1;
    
    // Configure kernel
    int outputSize = outChannels * outHeight * outWidth;
    int threadsPerBlock = 256;
    int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    convForwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, weights, bias, output,
        inChannels, outChannels, inHeight, inWidth,
        kernelHeight, kernelWidth, stride, padding,
        outHeight, outWidth
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in convForwardKernel: %s\n", cudaGetErrorString(error));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}

// Wrapper function to launch backward convolution kernels
void launchConvBackward(
    const float* input,
    const float* weights,
    const float* outputGrad,
    float* inputGrad,
    float* weightGrad,
    float* biasGrad,
    int inChannels,
    int outChannels,
    int inHeight,
    int inWidth,
    int kernelHeight,
    int kernelWidth,
    int stride,
    int padding
) {
    // Calculate output dimensions
    int outHeight = (inHeight + 2 * padding - kernelHeight) / stride + 1;
    int outWidth = (inWidth + 2 * padding - kernelWidth) / stride + 1;
    
    // Configure and launch input gradient kernel
    if (inputGrad != nullptr) {
        int inputSize = inChannels * inHeight * inWidth;
        int threadsPerBlock = 256;
        int blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;
        
        convBackwardInputKernel<<<blocksPerGrid, threadsPerBlock>>>(
            outputGrad, weights, inputGrad,
            inChannels, outChannels, inHeight, inWidth,
            kernelHeight, kernelWidth, stride, padding,
            outHeight, outWidth
        );
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in convBackwardInputKernel: %s\n", cudaGetErrorString(error));
        }
    }
    
    // Configure and launch weight gradient kernel
    if (weightGrad != nullptr) {
        int totalWeights = outChannels * inChannels * kernelHeight * kernelWidth;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalWeights + threadsPerBlock - 1) / threadsPerBlock;
        
        convBackwardWeightKernel<<<blocksPerGrid, threadsPerBlock>>>(
            input, outputGrad, weightGrad,
            inChannels, outChannels, inHeight, inWidth,
            kernelHeight, kernelWidth, stride, padding,
            outHeight, outWidth
        );
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in convBackwardWeightKernel: %s\n", cudaGetErrorString(error));
        }
    }
    
    // Configure and launch bias gradient kernel
    if (biasGrad != nullptr) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (outChannels + threadsPerBlock - 1) / threadsPerBlock;
        
        convBackwardBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(
            outputGrad, biasGrad,
            outChannels, outHeight, outWidth
        );
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error in convBackwardBiasKernel: %s\n", cudaGetErrorString(error));
        }
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}

} // namespace cuda
} // namespace zyraai

#endif // USE_CUDA 