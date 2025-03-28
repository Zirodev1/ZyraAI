/**
 * @file convolutional_layer.h
 * @brief Convolutional layer implementation for neural networks
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_CONVOLUTIONAL_LAYER_H
#define ZYRAAI_CONVOLUTIONAL_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace zyraai {

/**
 * @class ConvolutionalLayer
 * @brief Implements a 2D convolutional layer for neural networks
 * 
 * A convolutional layer applies a set of learnable filters to the input
 * by performing a convolution operation. Each filter produces an activation
 * map that detects specific features in the input.
 */
class ConvolutionalLayer : public Layer {
public:
  /**
   * @brief Construct a new Convolutional Layer
   * @param name Layer name
   * @param inputChannels Number of input channels
   * @param inputHeight Height of the input feature map
   * @param inputWidth Width of the input feature map
   * @param numFilters Number of convolutional filters
   * @param filterSize Size of each filter (square filters only)
   * @param stride Stride of the convolution (default: 1)
   * @param padding Zero-padding around the input (default: 0)
   * @throws std::invalid_argument If any parameter is invalid
   */
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

    // Validate parameters
    if (inputChannels <= 0) {
      throw std::invalid_argument("ConvolutionalLayer: inputChannels must be positive");
    }
    if (inputHeight <= 0 || inputWidth <= 0) {
      throw std::invalid_argument("ConvolutionalLayer: input dimensions must be positive");
    }
    if (numFilters <= 0) {
      throw std::invalid_argument("ConvolutionalLayer: numFilters must be positive");
    }
    if (filterSize <= 0) {
      throw std::invalid_argument("ConvolutionalLayer: filterSize must be positive");
    }
    if (stride <= 0) {
      throw std::invalid_argument("ConvolutionalLayer: stride must be positive");
    }
    if (padding < 0) {
      throw std::invalid_argument("ConvolutionalLayer: padding cannot be negative");
    }

    // Calculate output dimensions
    outputHeight_ = (inputHeight_ - filterSize_ + 2 * padding_) / stride_ + 1;
    outputWidth_ = (inputWidth_ - filterSize_ + 2 * padding_) / stride_ + 1;
    
    // Check if output dimensions are valid
    if (outputHeight_ <= 0 || outputWidth_ <= 0) {
      throw std::invalid_argument(
          "ConvolutionalLayer: output dimensions are invalid with current parameters. "
          "Try increasing padding or decreasing stride.");
    }

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

  /**
   * @brief Forward pass of convolution operation
   * @param input Input tensor of shape [inputChannels*inputHeight*inputWidth, batchSize]
   * @return Output tensor of shape [numFilters*outputHeight*outputWidth, batchSize]
   * @throws std::invalid_argument If input dimensions don't match expected size
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    
    // Validate input dimensions
    if (input.rows() != inputChannels_ * inputHeight_ * inputWidth_) {
      throw std::invalid_argument(
          "ConvolutionalLayer::forward: input dimension mismatch. Expected: " +
          std::to_string(inputChannels_ * inputHeight_ * inputWidth_) + 
          ", got: " + std::to_string(input.rows()));
    }
    
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

  /**
   * @brief Backward pass of convolution operation
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate for parameter updates
   * @return Gradient with respect to the input
   * @throws std::invalid_argument If gradient dimensions don't match expected size
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    const int batchSize = gradOutput.cols();
    
    // Validate gradient dimensions
    if (gradOutput.rows() != numFilters_ * outputHeight_ * outputWidth_) {
      throw std::invalid_argument(
          "ConvolutionalLayer::backward: gradient dimension mismatch. Expected: " +
          std::to_string(numFilters_ * outputHeight_ * outputWidth_) + 
          ", got: " + std::to_string(gradOutput.rows()));
    }
    
    if (learningRate <= 0.0f) {
      throw std::invalid_argument("ConvolutionalLayer::backward: learningRate must be positive");
    }

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

  /**
   * @brief Get the layer's parameters
   * @return Vector containing filter weights and biases
   */
  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    ::std::vector<Eigen::MatrixXf> params;
    for (const auto &filter : filters_) {
      params.push_back(filter);
    }
    params.push_back(biases_);
    return params;
  }

  /**
   * @brief Get the parameter gradients
   * @return Vector containing filter gradients and bias gradients
   */
  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    ::std::vector<Eigen::MatrixXf> grads;
    for (const auto &gradFilter : gradFilters_) {
      grads.push_back(gradFilter);
    }
    grads.push_back(gradBiases_);
    return grads;
  }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (0 to numFilters-1: filters, numFilters: biases)
   * @param update Update value to apply
   * @throws std::out_of_range If index is invalid
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index < numFilters_) {
      filters_[index] -= update;
    } else if (index == numFilters_) {
      biases_ -= update;
    } else {
      throw std::out_of_range("ConvolutionalLayer: parameter index out of range");
    }
  }

  /**
   * @brief Set the layer to training or evaluation mode
   * @param training Whether the layer is in training mode
   */
  void setTraining(bool training) override { isTraining_ = training; }

private:
  int inputChannels_;    ///< Number of input channels
  int inputHeight_;      ///< Height of input feature map
  int inputWidth_;       ///< Width of input feature map
  int numFilters_;       ///< Number of convolutional filters
  int filterSize_;       ///< Size of each filter (square)
  int stride_;           ///< Convolution stride
  int padding_;          ///< Zero-padding around input
  int outputHeight_;     ///< Height of output feature map
  int outputWidth_;      ///< Width of output feature map

  ::std::vector<Eigen::MatrixXf> filters_;      ///< Convolutional filters
  Eigen::VectorXf biases_;                      ///< Bias terms
  ::std::vector<Eigen::MatrixXf> gradFilters_;  ///< Filter gradients
  Eigen::VectorXf gradBiases_;                  ///< Bias gradients
  Eigen::MatrixXf input_;                       ///< Stored for backward pass
  bool isTraining_;                             ///< Whether in training mode
};

} // namespace zyraai

#endif // ZYRAAI_CONVOLUTIONAL_LAYER_H