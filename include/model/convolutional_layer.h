#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace zyraai {

class ConvolutionalLayer : public Layer {
public:
  ConvolutionalLayer(const ::std::string &name, int inputChannels,
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
    float weight_scale =
        sqrt(6.0f / (filterSize_ * filterSize_ * inputChannels_ +
                     filterSize_ * filterSize_ * numFilters_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-weight_scale, weight_scale);

    filters_.resize(numFilters_);
    for (int i = 0; i < numFilters_; ++i) {
      filters_[i] =
          Eigen::MatrixXf::Zero(filterSize_ * filterSize_ * inputChannels_, 1);
      for (int j = 0; j < filterSize_ * filterSize_ * inputChannels_; ++j) {
        filters_[i](j, 0) = dis(gen);
      }
    }

    // Initialize biases to zero
    biases_ = Eigen::VectorXf::Zero(numFilters_);

    // Initialize gradients
    gradFilters_.resize(numFilters_);
    for (int i = 0; i < numFilters_; ++i) {
      gradFilters_[i] =
          Eigen::MatrixXf::Zero(filterSize_ * filterSize_ * inputChannels_, 1);
    }
    gradBiases_ = Eigen::VectorXf::Zero(numFilters_);
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    input_ = input; // Store for backward pass

    // Initialize output
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(
        outputHeight_ * outputWidth_ * numFilters_, batchSize);

// Parallelize over batch samples and filters
#pragma omp parallel for collapse(2) if (batchSize > 1)
    for (int b = 0; b < batchSize; ++b) {
      for (int f = 0; f < numFilters_; ++f) {
        float bias = biases_(f);

        // For each output position
        for (int oh = 0; oh < outputHeight_; ++oh) {
          for (int ow = 0; ow < outputWidth_; ++ow) {
            float sum = bias;

            // For each input channel
            for (int c = 0; c < inputChannels_; ++c) {
              // For each filter element
              for (int fh = 0; fh < filterSize_; ++fh) {
                for (int fw = 0; fw < filterSize_; ++fw) {
                  int ih = oh * stride_ + fh - padding_;
                  int iw = ow * stride_ + fw - padding_;

                  if (ih >= 0 && ih < inputHeight_ && iw >= 0 &&
                      iw < inputWidth_) {
                    int inputIdx = (c * inputHeight_ + ih) * inputWidth_ + iw;
                    int filterIdx = (c * filterSize_ + fh) * filterSize_ + fw;
                    sum += input(inputIdx, b) * filters_[f](filterIdx, 0);
                  }
                }
              }
            }

            int outputIdx = (f * outputHeight_ + oh) * outputWidth_ + ow;
            output(outputIdx, b) = sum;
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

// Reset gradients
#pragma omp parallel for
    for (int f = 0; f < numFilters_; ++f) {
      gradFilters_[f].setZero();
    }
    gradBiases_.setZero();

// Parallelize over batch samples
#pragma omp parallel
    {
      // Create thread-local storage for gradients to avoid race conditions
      ::std::vector<Eigen::MatrixXf> threadGradFilters(numFilters_);
      Eigen::VectorXf threadGradBiases = Eigen::VectorXf::Zero(numFilters_);

      for (int f = 0; f < numFilters_; ++f) {
        threadGradFilters[f] = Eigen::MatrixXf::Zero(
            filterSize_ * filterSize_ * inputChannels_, 1);
      }

// Each thread processes a subset of the batch
#pragma omp for
      for (int b = 0; b < batchSize; ++b) {
        // For each filter
        for (int f = 0; f < numFilters_; ++f) {
          // For each output position
          for (int oh = 0; oh < outputHeight_; ++oh) {
            for (int ow = 0; ow < outputWidth_; ++ow) {
              int outputIdx = (f * outputHeight_ + oh) * outputWidth_ + ow;
              float gradOutputValue = gradOutput(outputIdx, b);

              // Update bias gradient
              threadGradBiases(f) += gradOutputValue;

              // For each input channel
              for (int c = 0; c < inputChannels_; ++c) {
                // For each filter element
                for (int fh = 0; fh < filterSize_; ++fh) {
                  for (int fw = 0; fw < filterSize_; ++fw) {
                    int ih = oh * stride_ + fh - padding_;
                    int iw = ow * stride_ + fw - padding_;

                    if (ih >= 0 && ih < inputHeight_ && iw >= 0 &&
                        iw < inputWidth_) {
                      int inputIdx = (c * inputHeight_ + ih) * inputWidth_ + iw;
                      int filterIdx = (c * filterSize_ + fh) * filterSize_ + fw;

                      // Update filter gradient
                      threadGradFilters[f](filterIdx, 0) +=
                          input_(inputIdx, b) * gradOutputValue;

// Update input gradient - no need for thread synchronization as each thread
// handles different batch samples
#pragma omp atomic
                      gradInput(inputIdx, b) +=
                          filters_[f](filterIdx, 0) * gradOutputValue;
                    }
                  }
                }
              }
            }
          }
        }
      }

// Merge thread-local gradients
#pragma omp critical
      {
        for (int f = 0; f < numFilters_; ++f) {
          gradFilters_[f] += threadGradFilters[f];
        }
        gradBiases_ += threadGradBiases;
      }
    }

// Update parameters
#pragma omp parallel for
    for (int f = 0; f < numFilters_; ++f) {
      filters_[f] -= learningRate * gradFilters_[f] / batchSize;
    }
    biases_ -= learningRate * gradBiases_ / batchSize;

    return gradInput;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    ::std::vector<Eigen::MatrixXf> params;
    for (const auto &filter : filters_) {
      params.push_back(filter);
    }
    params.push_back(biases_);
    return params;
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    ::std::vector<Eigen::MatrixXf> grads;
    for (const auto &gradFilter : gradFilters_) {
      grads.push_back(gradFilter);
    }
    grads.push_back(gradBiases_);
    return grads;
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index < numFilters_) {
      filters_[index] -= update;
    } else if (index == numFilters_) {
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

  ::std::vector<Eigen::MatrixXf> filters_;
  Eigen::VectorXf biases_;
  ::std::vector<Eigen::MatrixXf> gradFilters_;
  Eigen::VectorXf gradBiases_;
  Eigen::MatrixXf input_; // Stored for backward pass
  bool isTraining_ = true;
};

} // namespace zyraai