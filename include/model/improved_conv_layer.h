/**
 * @file improved_conv_layer.h
 * @brief Improved convolutional layer implementation with proper channel handling
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_IMPROVED_CONV_LAYER_H
#define ZYRAAI_IMPROVED_CONV_LAYER_H

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
 * @class ImprovedConvLayer
 * @brief Implements a 2D convolutional layer with proper handling of channels and feature maps
 * 
 * This improved implementation correctly handles the channel and spatial dimensions
 * of input and output tensors, making it more suitable for complex architectures.
 */
class ImprovedConvLayer : public Layer {
public:
  /**
   * @brief Construct a new Improved Convolutional Layer
   * @param name Layer name
   * @param inputChannels Number of input channels
   * @param inputHeight Height of the input feature map
   * @param inputWidth Width of the input feature map
   * @param numFilters Number of convolutional filters (output channels)
   * @param filterSize Size of each filter (square filters only)
   * @param stride Stride of the convolution (default: 1)
   * @param padding Zero-padding around the input (default: 0)
   * @throws std::invalid_argument If any parameter is invalid
   */
  ImprovedConvLayer(const std::string &name, int inputChannels,
                    int inputHeight, int inputWidth, int numFilters,
                    int filterSize, int stride = 1, int padding = 0)
      : Layer(name, inputChannels * inputHeight * inputWidth,
              numFilters *
                  ((inputHeight - filterSize + 2 * padding) / stride + 1) *
                  ((inputWidth - filterSize + 2 * padding) / stride + 1)),
        inputChannels_(inputChannels), inputHeight_(inputHeight),
        inputWidth_(inputWidth), numFilters_(numFilters),
        filterSize_(filterSize), stride_(stride), padding_(padding),
        training_(true) {

    // Validate parameters
    if (inputChannels <= 0) {
      throw std::invalid_argument("ImprovedConvLayer: inputChannels must be positive");
    }
    if (inputHeight <= 0 || inputWidth <= 0) {
      throw std::invalid_argument("ImprovedConvLayer: input dimensions must be positive");
    }
    if (numFilters <= 0) {
      throw std::invalid_argument("ImprovedConvLayer: numFilters must be positive");
    }
    if (filterSize <= 0) {
      throw std::invalid_argument("ImprovedConvLayer: filterSize must be positive");
    }
    if (stride <= 0) {
      throw std::invalid_argument("ImprovedConvLayer: stride must be positive");
    }
    if (padding < 0) {
      throw std::invalid_argument("ImprovedConvLayer: padding cannot be negative");
    }

    // Calculate output dimensions
    outputHeight_ = (inputHeight_ - filterSize_ + 2 * padding_) / stride_ + 1;
    outputWidth_ = (inputWidth_ - filterSize_ + 2 * padding_) / stride_ + 1;
    
    // Check if output dimensions are valid
    if (outputHeight_ <= 0 || outputWidth_ <= 0) {
      throw std::invalid_argument(
          "ImprovedConvLayer: output dimensions are invalid with current parameters. "
          "Try increasing padding or decreasing stride.");
    }

    // Initialize filters with Kaiming/He initialization
    // Scale: sqrt(2 / (filterSize^2 * inputChannels)) for ReLU activations
    float weight_scale =
        std::sqrt(2.0f / (filterSize_ * filterSize_ * inputChannels_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, weight_scale);

    // Each filter is a 4D tensor: [outChannels, inChannels, filterHeight, filterWidth]
    filters_.resize(numFilters_);
    for (int i = 0; i < numFilters_; ++i) {
      filters_[i] = Eigen::MatrixXf::Zero(filterSize_ * filterSize_ * inputChannels_, 1);
      for (int j = 0; j < filterSize_ * filterSize_ * inputChannels_; ++j) {
        filters_[i](j, 0) = dist(gen);
      }
    }

    // Initialize biases to zero
    biases_ = Eigen::VectorXf::Zero(numFilters_);

    // Initialize gradients
    gradFilters_.resize(numFilters_);
    for (int i = 0; i < numFilters_; ++i) {
      gradFilters_[i] = Eigen::MatrixXf::Zero(filterSize_ * filterSize_ * inputChannels_, 1);
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
          "ImprovedConvLayer::forward: input dimension mismatch. Expected: " +
          std::to_string(inputChannels_ * inputHeight_ * inputWidth_) + 
          ", got: " + std::to_string(input.rows()));
    }
    
    input_ = input; // Store for backward pass

    // Initialize output
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(
        outputHeight_ * outputWidth_ * numFilters_, batchSize);

    // Process each sample in the batch
    #pragma omp parallel for if (batchSize > 1)
    for (int b = 0; b < batchSize; ++b) {
      // Process each filter (output channel)
      for (int f = 0; f < numFilters_; ++f) {
        const float bias = biases_(f);
        
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
                  
                  if (ih >= 0 && ih < inputHeight_ && iw >= 0 && iw < inputWidth_) {
                    // Calculate input index
                    int inputIdx = c * (inputHeight_ * inputWidth_) + ih * inputWidth_ + iw;
                    
                    // Calculate filter index
                    int filterIdx = c * (filterSize_ * filterSize_) + fh * filterSize_ + fw;
                    
                    // Accumulate convolution
                    sum += input(inputIdx, b) * filters_[f](filterIdx, 0);
                  }
                }
              }
            }
            
            // Calculate output index
            int outputIdx = f * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
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
          "ImprovedConvLayer::backward: gradient dimension mismatch. Expected: " +
          std::to_string(numFilters_ * outputHeight_ * outputWidth_) + 
          ", got: " + std::to_string(gradOutput.rows()));
    }
    
    if (learningRate <= 0.0f) {
      throw std::invalid_argument("ImprovedConvLayer::backward: learningRate must be positive");
    }

    // Initialize gradient of input
    Eigen::MatrixXf gradInput = Eigen::MatrixXf::Zero(
        inputChannels_ * inputHeight_ * inputWidth_, batchSize);

    // Clear parameter gradients
    for (int f = 0; f < numFilters_; ++f) {
      gradFilters_[f].setZero();
    }
    gradBiases_.setZero();

    // Compute gradients
    #pragma omp parallel
    {
      // Thread-local storage for gradients to avoid race conditions
      std::vector<Eigen::MatrixXf> threadGradFilters(numFilters_);
      Eigen::VectorXf threadGradBiases = Eigen::VectorXf::Zero(numFilters_);
      
      for (int f = 0; f < numFilters_; ++f) {
        threadGradFilters[f] = Eigen::MatrixXf::Zero(
            filterSize_ * filterSize_ * inputChannels_, 1);
      }
      
      // Each thread processes a subset of the batch
      #pragma omp for
      for (int b = 0; b < batchSize; ++b) {
        // For each output gradient
        for (int f = 0; f < numFilters_; ++f) {
          // For each position in the output gradient
          for (int oh = 0; oh < outputHeight_; ++oh) {
            for (int ow = 0; ow < outputWidth_; ++ow) {
              // Calculate output gradient index
              int outputIdx = f * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
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
                    
                    if (ih >= 0 && ih < inputHeight_ && iw >= 0 && iw < inputWidth_) {
                      // Calculate input index
                      int inputIdx = c * (inputHeight_ * inputWidth_) + ih * inputWidth_ + iw;
                      
                      // Calculate filter index
                      int filterIdx = c * (filterSize_ * filterSize_) + fh * filterSize_ + fw;
                      
                      // Update filter gradient
                      threadGradFilters[f](filterIdx, 0) += input_(inputIdx, b) * gradOutputValue;
                      
                      // Update input gradient
                      gradInput(inputIdx, b) += filters_[f](filterIdx, 0) * gradOutputValue;
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
    
    // Average gradients over the batch
    const float scale = 1.0f / static_cast<float>(batchSize);
    for (int f = 0; f < numFilters_; ++f) {
      gradFilters_[f] *= scale;
      
      // Update filters with gradient descent
      filters_[f] -= learningRate * gradFilters_[f];
    }
    
    gradBiases_ *= scale;
    biases_ -= learningRate * gradBiases_;

    return gradInput;
  }

  /**
   * @brief Get the output shape for a given filter
   * @param inputShape 3D tensor shape [channels, height, width]
   * @return Output shape as 3D tensor [numFilters, outputHeight, outputWidth]
   */
  std::vector<int> outputShape(const std::vector<int>& inputShape) const {
    if (inputShape.size() != 3) {
      throw std::invalid_argument("ImprovedConvLayer::outputShape: input shape must be 3D");
    }
    
    return {numFilters_, outputHeight_, outputWidth_};
  }
  
  /**
   * @brief Get output dimensions
   * @return pair of (outputHeight, outputWidth)
   */
  std::pair<int, int> getOutputDimensions() const {
    return {outputHeight_, outputWidth_};
  }
  
  /**
   * @brief Get number of output channels
   * @return Number of filters (output channels)
   */
  int getOutputChannels() const {
    return numFilters_;
  }

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  std::string getType() const override {
    return "ImprovedConvLayer";
  }

  /**
   * @brief Get the parameters of the layer
   * @return Vector containing all filter weights and biases
   */
  std::vector<Eigen::MatrixXf> getParameters() const override {
    std::vector<Eigen::MatrixXf> params;
    // Add all filters
    for (const auto& filter : filters_) {
      params.push_back(filter);
    }
    // Add biases as a matrix
    Eigen::MatrixXf biasMatrix = biases_;
    params.push_back(biasMatrix);
    return params;
  }

  /**
   * @brief Get the gradients for the parameters
   * @return Vector containing all filter gradients and bias gradients
   */
  std::vector<Eigen::MatrixXf> getGradients() const override {
    std::vector<Eigen::MatrixXf> grads;
    // Add all filter gradients
    for (const auto& gradFilter : gradFilters_) {
      grads.push_back(gradFilter);
    }
    // Add bias gradients as a matrix
    Eigen::MatrixXf gradBiasMatrix = gradBiases_;
    grads.push_back(gradBiasMatrix);
    return grads;
  }

  /**
   * @brief Set the training mode of the layer
   * @param training True for training mode, false for inference mode
   */
  void setTraining(bool training) override {
    training_ = training;
  }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index
   * @param update Parameter update
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index < numFilters_) {
      // Update filter weights
      filters_[index] -= update;
    } else if (index == numFilters_) {
      // Update biases
      biases_ -= update.col(0);
    } else {
      throw std::invalid_argument("ImprovedConvLayer::updateParameter: invalid parameter index");
    }
  }

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
  bool training_;
  
  std::vector<Eigen::MatrixXf> filters_;     // Weights
  Eigen::VectorXf biases_;                   // Biases
  
  std::vector<Eigen::MatrixXf> gradFilters_; // Weight gradients
  Eigen::VectorXf gradBiases_;               // Bias gradients
  
  Eigen::MatrixXf input_;                    // Saved input for backward pass
};

} // namespace zyraai

#endif // ZYRAAI_IMPROVED_CONV_LAYER_H 