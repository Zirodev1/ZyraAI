#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace zyraai {

// Helper function to perform the im2col operation
// This transforms the input image patches into columns for efficient matrix
// multiplication
Eigen::MatrixXf im2col(const Eigen::MatrixXf &input, int inputChannels,
                       int inputHeight, int inputWidth, int filterSize,
                       int stride, int padding, int outputHeight,
                       int outputWidth, int batchSize) {
  const int patchSize = filterSize * filterSize * inputChannels;
  Eigen::MatrixXf result(patchSize, outputHeight * outputWidth * batchSize);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < batchSize; ++b) {
    for (int oh = 0; oh < outputHeight; ++oh) {
      for (int ow = 0; ow < outputWidth; ++ow) {
        // For each output position and batch item
        const int colIdx =
            b * (outputHeight * outputWidth) + oh * outputWidth + ow;

        // Extract the patch
        for (int c = 0; c < inputChannels; ++c) {
          for (int fh = 0; fh < filterSize; ++fh) {
            for (int fw = 0; fw < filterSize; ++fw) {
              const int ih = oh * stride + fh - padding;
              const int iw = ow * stride + fw - padding;

              const int rowIdx = (c * filterSize + fh) * filterSize + fw;

              if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                const int inputIdx = (c * inputHeight + ih) * inputWidth + iw;
                result(rowIdx, colIdx) = input(inputIdx, b);
              } else {
                result(rowIdx, colIdx) = 0;
              }
            }
          }
        }
      }
    }
  }

  return result;
}

// Helper function to perform the col2im operation (inverse of im2col)
// Used in the backward pass to reconstruct gradients
Eigen::MatrixXf col2im(const Eigen::MatrixXf &colMat, int inputChannels,
                       int inputHeight, int inputWidth, int filterSize,
                       int stride, int padding, int outputHeight,
                       int outputWidth, int batchSize) {
  Eigen::MatrixXf result = Eigen::MatrixXf::Zero(
      inputChannels * inputHeight * inputWidth, batchSize);

#pragma omp parallel
  {
    // Thread-local result to avoid race conditions
    Eigen::MatrixXf threadResult = Eigen::MatrixXf::Zero(
        inputChannels * inputHeight * inputWidth, batchSize);

#pragma omp for collapse(3)
    for (int b = 0; b < batchSize; ++b) {
      for (int oh = 0; oh < outputHeight; ++oh) {
        for (int ow = 0; ow < outputWidth; ++ow) {
          // For each output position and batch item
          const int colIdx =
              b * (outputHeight * outputWidth) + oh * outputWidth + ow;

          // Distribute the gradient to the input positions
          for (int c = 0; c < inputChannels; ++c) {
            for (int fh = 0; fh < filterSize; ++fh) {
              for (int fw = 0; fw < filterSize; ++fw) {
                const int ih = oh * stride + fh - padding;
                const int iw = ow * stride + fw - padding;

                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                  const int rowIdx = (c * filterSize + fh) * filterSize + fw;
                  const int inputIdx = (c * inputHeight + ih) * inputWidth + iw;
                  threadResult(inputIdx, b) += colMat(rowIdx, colIdx);
                }
              }
            }
          }
        }
      }
    }

// Merge thread-local results
#pragma omp critical
    {
      result += threadResult;
    }
  }

  return result;
}

class OptimizedConvLayer : public Layer {
public:
  OptimizedConvLayer(const ::std::string &name, int inputChannels,
                     int inputHeight, int inputWidth, int numFilters,
                     int filterSize, int stride = 1, int padding = 0)
      : Layer(name, inputChannels * inputHeight * inputWidth,
              numFilters *
                  ((inputHeight - filterSize + 2 * padding) / stride + 1) *
                  ((inputWidth - filterSize + 2 * padding) / stride + 1)),
        inputChannels_(inputChannels), inputHeight_(inputHeight),
        inputWidth_(inputWidth), numFilters_(numFilters),
        filterSize_(filterSize), stride_(stride), padding_(padding) {

    // Calculate output dimensions
    outputHeight_ = (inputHeight_ - filterSize_ + 2 * padding_) / stride_ + 1;
    outputWidth_ = (inputWidth_ - filterSize_ + 2 * padding_) / stride_ + 1;

    // Initialize filters with Xavier/Glorot initialization
    const int fanIn = filterSize_ * filterSize_ * inputChannels_;
    const int fanOut = filterSize_ * filterSize_ * numFilters_;
    float weight_scale = sqrt(6.0f / (fanIn + fanOut));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-weight_scale, weight_scale);

    // Reshape filters for more efficient matrix multiply
    // Shape: [numFilters, filterSize*filterSize*inputChannels]
    filters_ = Eigen::MatrixXf(numFilters_,
                               filterSize_ * filterSize_ * inputChannels_);

    for (int f = 0; f < numFilters_; ++f) {
      for (int i = 0; i < filterSize_ * filterSize_ * inputChannels_; ++i) {
        filters_(f, i) = dis(gen);
      }
    }

    // Initialize biases to zero
    biases_ = Eigen::VectorXf::Zero(numFilters_);

    // Initialize gradients
    gradFilters_ = Eigen::MatrixXf::Zero(
        numFilters_, filterSize_ * filterSize_ * inputChannels_);
    gradBiases_ = Eigen::VectorXf::Zero(numFilters_);
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    input_ = input; // Store for backward pass

    // Step 1: Convert input patches to columns using im2col
    // Shape: [filterSize*filterSize*inputChannels,
    // outputHeight*outputWidth*batchSize]
    inputCols_ =
        im2col(input, inputChannels_, inputHeight_, inputWidth_, filterSize_,
               stride_, padding_, outputHeight_, outputWidth_, batchSize);

    // Step 2: Perform the convolution as a matrix multiplication
    // filters_: [numFilters, filterSize*filterSize*inputChannels]
    // inputCols_: [filterSize*filterSize*inputChannels,
    // outputHeight*outputWidth*batchSize] output after GEMM: [numFilters,
    // outputHeight*outputWidth*batchSize]
    Eigen::MatrixXf outputTmp = filters_ * inputCols_;

    // Step 3: Reshape the output to the proper format
    // [numFilters*outputHeight*outputWidth, batchSize]
    Eigen::MatrixXf output(numFilters_ * outputHeight_ * outputWidth_,
                           batchSize);

// Add biases and reshape
#pragma omp parallel for
    for (int b = 0; b < batchSize; ++b) {
      for (int f = 0; f < numFilters_; ++f) {
        for (int oh = 0; oh < outputHeight_; ++oh) {
          for (int ow = 0; ow < outputWidth_; ++ow) {
            int outputIdx = (f * outputHeight_ + oh) * outputWidth_ + ow;
            int tmpIdx =
                b * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
            output(outputIdx, b) = outputTmp(f, tmpIdx) + biases_(f);
          }
        }
      }
    }

    return output;
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    const int batchSize = gradOutput.cols();

    // Step 1: Reshape gradOutput to match the format of the convolution output
    // [numFilters, outputHeight*outputWidth*batchSize]
    Eigen::MatrixXf gradOutputReshaped(
        numFilters_, outputHeight_ * outputWidth_ * batchSize);

#pragma omp parallel for
    for (int b = 0; b < batchSize; ++b) {
      for (int f = 0; f < numFilters_; ++f) {
        for (int oh = 0; oh < outputHeight_; ++oh) {
          for (int ow = 0; ow < outputWidth_; ++ow) {
            int outputIdx = (f * outputHeight_ + oh) * outputWidth_ + ow;
            int reshapedIdx =
                b * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
            gradOutputReshaped(f, reshapedIdx) = gradOutput(outputIdx, b);
          }
        }
      }
    }

    // Step 2: Compute gradient with respect to filters
    // gradFilters = gradOutputReshaped * inputCols^T
    // [numFilters, outputHeight*outputWidth*batchSize] *
    // [outputHeight*outputWidth*batchSize, filterSize*filterSize*inputChannels]
    // = [numFilters, filterSize*filterSize*inputChannels]
    gradFilters_ = gradOutputReshaped * inputCols_.transpose();
    gradFilters_ /= batchSize; // Average over batch

    // Step 3: Compute gradient with respect to biases
    // Sum over all positions and batch items
    gradBiases_ = gradOutputReshaped.rowwise().sum();
    gradBiases_ /= batchSize; // Average over batch

    // Step 4: Compute gradient with respect to input
    // gradInputCols = filters^T * gradOutputReshaped
    // [filterSize*filterSize*inputChannels, numFilters] * [numFilters,
    // outputHeight*outputWidth*batchSize] =
    // [filterSize*filterSize*inputChannels, outputHeight*outputWidth*batchSize]
    Eigen::MatrixXf gradInputCols = filters_.transpose() * gradOutputReshaped;

    // Step 5: Convert columns back to image format
    Eigen::MatrixXf gradInput = col2im(
        gradInputCols, inputChannels_, inputHeight_, inputWidth_, filterSize_,
        stride_, padding_, outputHeight_, outputWidth_, batchSize);

    // Step 6: Update parameters
    filters_ -= learningRate * gradFilters_;
    biases_ -= learningRate * gradBiases_;

    return gradInput;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    ::std::vector<Eigen::MatrixXf> params;
    params.push_back(filters_);
    params.push_back(biases_);
    return params;
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    ::std::vector<Eigen::MatrixXf> grads;
    grads.push_back(gradFilters_);
    grads.push_back(gradBiases_);
    return grads;
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      filters_ -= update;
    } else if (index == 1) {
      biases_ -= update;
    }
  }

  void setTraining(bool training) override { isTraining_ = training; }

private:
  int inputChannels_;
  int inputHeight_;
  int inputWidth_;
  int numFilters_;
  int filterSize_;
  int stride_;
  int padding_;
  int outputHeight_;
  int outputWidth_;

  Eigen::MatrixXf filters_; // [numFilters, filterSize*filterSize*inputChannels]
  Eigen::VectorXf biases_;  // [numFilters]
  Eigen::MatrixXf
      gradFilters_; // [numFilters, filterSize*filterSize*inputChannels]
  Eigen::VectorXf gradBiases_; // [numFilters]
  Eigen::MatrixXf input_;      // Original input, stored for backward pass
  Eigen::MatrixXf inputCols_;  // Input after im2col, stored for backward pass
  bool isTraining_ = true;
};

} // namespace zyraai