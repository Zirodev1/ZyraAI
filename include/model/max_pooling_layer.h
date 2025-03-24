#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace zyraai {

class MaxPoolingLayer : public Layer {
public:
  MaxPoolingLayer(const ::std::string &name, int inputChannels, int inputHeight,
                  int inputWidth, int poolSize, int stride = 0)
      : Layer(name, inputChannels * inputHeight * inputWidth,
              inputChannels * ((inputHeight + stride - 1) / stride) *
                  ((inputWidth + stride - 1) / stride)),
        inputChannels_(inputChannels), inputHeight_(inputHeight),
        inputWidth_(inputWidth), poolSize_(poolSize),
        stride_(stride > 0 ? stride : poolSize) {

    // Calculate output dimensions
    outputHeight_ = (inputHeight_ - poolSize_) / stride_ + 1;
    outputWidth_ = (inputWidth_ - poolSize_) / stride_ + 1;
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();

    // Initialize output
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(
        outputHeight_ * outputWidth_ * inputChannels_, batchSize);

    // Initialize max indices for backward pass
    maxIndices_.resize(batchSize);
    for (int b = 0; b < batchSize; ++b) {
      maxIndices_[b] =
          ::std::vector<int>(outputHeight_ * outputWidth_ * inputChannels_, -1);
    }

// For each image in the batch
#pragma omp parallel for
    for (int b = 0; b < batchSize; ++b) {
      // For each channel
      for (int c = 0; c < inputChannels_; ++c) {
        // For each output position
        for (int oh = 0; oh < outputHeight_; ++oh) {
          for (int ow = 0; ow < outputWidth_; ++ow) {
            float maxValue = -::std::numeric_limits<float>::max();
            int maxIdx = -1;

            // For each element in the pooling window
            for (int ph = 0; ph < poolSize_; ++ph) {
              for (int pw = 0; pw < poolSize_; ++pw) {
                int ih = oh * stride_ + ph;
                int iw = ow * stride_ + pw;

                if (ih < inputHeight_ && iw < inputWidth_) {
                  int inputIdx = (c * inputHeight_ + ih) * inputWidth_ + iw;
                  float value = input(inputIdx, b);

                  if (value > maxValue) {
                    maxValue = value;
                    maxIdx = inputIdx;
                  }
                }
              }
            }

            int outputIdx = (c * outputHeight_ + oh) * outputWidth_ + ow;
            output(outputIdx, b) = maxValue;
            maxIndices_[b][outputIdx] = maxIdx;
          }
        }
      }
    }

    return output;
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    const int batchSize = gradOutput.cols();

    // Initialize gradient of input
    Eigen::MatrixXf gradInput = Eigen::MatrixXf::Zero(
        inputChannels_ * inputHeight_ * inputWidth_, batchSize);

// Parallelize over batch samples
#pragma omp parallel for
    for (int b = 0; b < batchSize; ++b) {
      // For each output element
      for (int outputIdx = 0; outputIdx < gradOutput.rows(); ++outputIdx) {
        int inputIdx = maxIndices_[b][outputIdx];
        if (inputIdx >= 0) {
          gradInput(inputIdx, b) += gradOutput(outputIdx, b);
        }
      }
    }

    return gradInput;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // No trainable parameters
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // No gradients
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // No parameters to update
  }

  void setTraining(bool training) override { isTraining_ = training; }

private:
  int inputChannels_;
  int inputHeight_;
  int inputWidth_;
  int poolSize_;
  int stride_;
  int outputHeight_;
  int outputWidth_;

  ::std::vector<::std::vector<int>>
      maxIndices_; // Stores indices of max values for backward pass
  bool isTraining_ = true;
};

} // namespace zyraai