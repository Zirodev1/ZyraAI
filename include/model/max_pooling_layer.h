/**
 * @file max_pooling_layer.h
 * @brief Max pooling layer implementation for neural networks
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_MAX_POOLING_LAYER_H
#define ZYRAAI_MAX_POOLING_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <limits>

namespace zyraai {

/**
 * @class MaxPoolingLayer
 * @brief Implements max pooling for convolutional neural networks
 * 
 * This layer performs downsampling by taking the maximum value in each pooling region,
 * reducing the spatial dimensions while preserving the number of channels.
 */
class MaxPoolingLayer : public Layer {
public:
  /**
   * @brief Construct a new Max Pooling Layer
   * @param name Layer name
   * @param channels Number of input/output channels
   * @param inputHeight Height of the input feature map
   * @param inputWidth Width of the input feature map
   * @param poolSize Size of the pooling window (default: 2)
   * @param stride Stride of the pooling operation (default: 2)
   * @throws std::invalid_argument If any parameter is invalid
   */
  MaxPoolingLayer(const std::string &name, int channels, int inputHeight, int inputWidth,
                  int poolSize = 2, int stride = 2)
      : Layer(name, channels * inputHeight * inputWidth,
              channels * ((inputHeight - poolSize) / stride + 1) * ((inputWidth - poolSize) / stride + 1)),
        channels_(channels), inputHeight_(inputHeight), inputWidth_(inputWidth),
        poolSize_(poolSize), stride_(stride), training_(true) {
    
    if (channels <= 0) {
      throw std::invalid_argument("MaxPoolingLayer: channels must be positive");
    }
    if (inputHeight <= 0 || inputWidth <= 0) {
      throw std::invalid_argument("MaxPoolingLayer: input dimensions must be positive");
    }
    if (poolSize <= 0) {
      throw std::invalid_argument("MaxPoolingLayer: poolSize must be positive");
    }
    if (stride <= 0) {
      throw std::invalid_argument("MaxPoolingLayer: stride must be positive");
    }
    if (inputHeight < poolSize || inputWidth < poolSize) {
      throw std::invalid_argument("MaxPoolingLayer: input dimensions must be >= poolSize");
    }
    
    // Calculate output dimensions
    outputHeight_ = (inputHeight_ - poolSize_) / stride_ + 1;
    outputWidth_ = (inputWidth_ - poolSize_) / stride_ + 1;
    
    if (outputHeight_ <= 0 || outputWidth_ <= 0) {
      throw std::invalid_argument(
          "MaxPoolingLayer: output dimensions are invalid with current parameters");
    }
  }

  /**
   * @brief Forward pass
   * @param input Input tensor of shape [channels*inputHeight*inputWidth, batchSize]
   * @return Output tensor of shape [channels*outputHeight*outputWidth, batchSize]
   * @throws std::invalid_argument If input dimensions don't match expected size
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    const int inputSize = channels_ * inputHeight_ * inputWidth_;
    
    if (input.rows() != inputSize) {
      throw std::invalid_argument(
          "MaxPoolingLayer::forward: input dimension mismatch. Expected: " +
          std::to_string(inputSize) + ", got: " + std::to_string(input.rows()));
    }
    
    // Save input for backward pass
    input_ = input;
    
    // Prepare output tensor
    const int outputSize = channels_ * outputHeight_ * outputWidth_;
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(outputSize, batchSize);
    
    // Prepare indices storage for backward pass
    maxIndices_.resize(outputSize * batchSize, -1);
    
    // Process each batch sample
    for (int b = 0; b < batchSize; ++b) {
      // Process each channel
      for (int c = 0; c < channels_; ++c) {
        // For each output position
        for (int oh = 0; oh < outputHeight_; ++oh) {
          for (int ow = 0; ow < outputWidth_; ++ow) {
            // Calculate the corresponding input region
            int ihStart = oh * stride_;
            int iwStart = ow * stride_;
            
            // Initialize with negative infinity
            float maxVal = -std::numeric_limits<float>::max();
            int maxIdx = -1;
            
            // For each element in the pooling region
            for (int ph = 0; ph < poolSize_; ++ph) {
              for (int pw = 0; pw < poolSize_; ++pw) {
                int ih = ihStart + ph;
                int iw = iwStart + pw;
                
                // Skip if outside input bounds
                if (ih >= inputHeight_ || iw >= inputWidth_) {
                  continue;
                }
                
                // Calculate input index
                int inputIdx = c * (inputHeight_ * inputWidth_) + ih * inputWidth_ + iw;
                float val = input(inputIdx, b);
                
                // Update max if necessary
                if (val > maxVal) {
                  maxVal = val;
                  maxIdx = inputIdx;
                }
              }
            }
            
            // Calculate output index
            int outputIdx = c * (outputHeight_ * outputWidth_) + oh * outputWidth_ + ow;
            output(outputIdx, b) = maxVal;
            
            // Store max index for backward pass
            maxIndices_[outputIdx + b * outputSize] = maxIdx;
          }
        }
      }
    }
    
    return output;
  }

  /**
   * @brief Backward pass
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate (unused for pooling layer)
   * @return Gradient with respect to the input
   * @throws std::invalid_argument If gradient dimensions don't match expected size
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput, float learningRate) override {
    const int batchSize = gradOutput.cols();
    const int outputSize = channels_ * outputHeight_ * outputWidth_;
    const int inputSize = channels_ * inputHeight_ * inputWidth_;
    
    if (gradOutput.rows() != outputSize) {
      throw std::invalid_argument(
          "MaxPoolingLayer::backward: gradient dimension mismatch. Expected: " +
          std::to_string(outputSize) + ", got: " + std::to_string(gradOutput.rows()));
    }
    
    // Initialize input gradient
    Eigen::MatrixXf gradInput = Eigen::MatrixXf::Zero(inputSize, batchSize);
    
    // Process each batch sample
    for (int b = 0; b < batchSize; ++b) {
      // Process each output position
      for (int outputIdx = 0; outputIdx < outputSize; ++outputIdx) {
        // Get the stored max index
        int maxIdx = maxIndices_[outputIdx + b * outputSize];
        
        if (maxIdx >= 0) {
          // Pass the gradient only to the max element
          gradInput(maxIdx, b) += gradOutput(outputIdx, b);
        }
      }
    }
    
    return gradInput;
  }

  /**
   * @brief Get the output dimensions
   * @return Pair of (outputHeight, outputWidth)
   */
  std::pair<int, int> getOutputDimensions() const {
    return {outputHeight_, outputWidth_};
  }

  /**
   * @brief Implementation of getParameters virtual function
   * Pooling layer has no parameters to train
   * @return Empty vector as this layer has no trainable parameters
   */
  std::vector<Eigen::MatrixXf> getParameters() const override {
    return {};
  }

  /**
   * @brief Implementation of getGradients virtual function
   * Pooling layer has no gradients for parameters
   * @return Empty vector as this layer has no parameter gradients
   */
  std::vector<Eigen::MatrixXf> getGradients() const override {
    return {};
  }

  /**
   * @brief Set the training mode of the layer
   * For MaxPoolingLayer, this doesn't change behavior but is required by Layer interface
   * @param training True for training mode, false for inference mode
   */
  void setTraining(bool training) override {
    training_ = training;
  }

  /**
   * @brief Implementation of updateParameter virtual function
   * This is a no-op for MaxPoolingLayer as it has no parameters to update
   * @param index Parameter index (unused)
   * @param update Parameter update (unused)
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // No-op as pooling layer has no parameters
    (void)index;
    (void)update;
  }

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  std::string getType() const override {
    return "MaxPoolingLayer";
  }

private:
  int channels_;     // Number of channels
  int inputHeight_;  // Input height
  int inputWidth_;   // Input width
  int poolSize_;     // Pooling window size
  int stride_;       // Stride of pooling operation
  int outputHeight_; // Output height
  int outputWidth_;  // Output width
  bool training_;    // Training mode flag
  
  Eigen::MatrixXf input_;            // Saved input for backward pass
  std::vector<int> maxIndices_;      // Indices of maximum values for backward pass
};

} // namespace zyraai

#endif // ZYRAAI_MAX_POOLING_LAYER_H