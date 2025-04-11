/**
 * @file channel_batch_norm_layer.h
 * @brief Channel-wise batch normalization layer implementation
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_CHANNEL_BATCH_NORM_LAYER_H
#define ZYRAAI_CHANNEL_BATCH_NORM_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace zyraai {

/**
 * @class ChannelBatchNormLayer
 * @brief Implements channel-wise batch normalization for convolutional neural networks
 *
 * This layer performs batch normalization separately for each channel in a CNN feature map,
 * which is the standard approach in modern CNN architectures.
 */
class ChannelBatchNormLayer : public Layer {
public:
  /**
   * @brief Construct a new Channel Batch Norm Layer
   * @param name Layer name
   * @param channels Number of channels
   * @param height Feature map height
   * @param width Feature map width
   * @param momentum Momentum for running mean and variance updates (default: 0.9)
   * @param epsilon Small value to avoid division by zero (default: 1e-5)
   */
  ChannelBatchNormLayer(const std::string &name, int channels, int height, int width,
                        float momentum = 0.9f, float epsilon = 1e-5f)
      : Layer(name, channels * height * width, channels * height * width),
        channels_(channels), height_(height), width_(width),
        featureSize_(height * width), momentum_(momentum), epsilon_(epsilon),
        training_(true) {
    
    if (channels <= 0) {
      throw std::invalid_argument("ChannelBatchNormLayer: channels must be positive");
    }
    if (height <= 0 || width <= 0) {
      throw std::invalid_argument("ChannelBatchNormLayer: dimensions must be positive");
    }
    if (momentum <= 0.0f || momentum >= 1.0f) {
      throw std::invalid_argument("ChannelBatchNormLayer: momentum must be in range (0, 1)");
    }
    if (epsilon <= 0.0f) {
      throw std::invalid_argument("ChannelBatchNormLayer: epsilon must be positive");
    }
    
    // Initialize gamma (scale) and beta (shift) parameters
    gamma_ = Eigen::VectorXf::Ones(channels_);
    beta_ = Eigen::VectorXf::Zero(channels_);
    
    // Initialize running statistics for inference
    runningMean_ = Eigen::VectorXf::Zero(channels_);
    runningVar_ = Eigen::VectorXf::Ones(channels_);
    
    // Initialize gradients
    gradGamma_ = Eigen::VectorXf::Zero(channels_);
    gradBeta_ = Eigen::VectorXf::Zero(channels_);
  }

  /**
   * @brief Forward pass
   * @param input Input tensor of shape [channels*height*width, batchSize]
   * @return Normalized output tensor of shape [channels*height*width, batchSize]
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    const int totalFeatures = input.rows();
    
    if (totalFeatures != channels_ * height_ * width_) {
      throw std::invalid_argument(
          "ChannelBatchNormLayer::forward: input dimension mismatch. Expected: " +
          std::to_string(channels_ * height_ * width_) + 
          ", got: " + std::to_string(totalFeatures));
    }
    
    // Save input for backward pass
    input_ = input;
    
    // Prepare output tensor
    Eigen::MatrixXf output(totalFeatures, batchSize);
    
    // Process each channel separately
    for (int c = 0; c < channels_; ++c) {
      // Extract this channel's data
      Eigen::MatrixXf channelData = extractChannel(input, c);
      
      if (training_) {
        // Compute mean and variance for this batch
        Eigen::VectorXf batchMean = channelData.rowwise().mean();
        Eigen::MatrixXf centered = channelData.colwise() - batchMean;
        Eigen::VectorXf batchVar = (centered.array().square().rowwise().sum() / batchSize).matrix();
        
        // Update running statistics
        runningMean_(c) = momentum_ * runningMean_(c) + (1.0f - momentum_) * batchMean(0);
        runningVar_(c) = momentum_ * runningVar_(c) + (1.0f - momentum_) * batchVar(0);
        
        // Save for backward pass
        batchMeans_.push_back(batchMean);
        batchVars_.push_back(batchVar);
        
        // Normalize
        float stdDev = std::sqrt(batchVar(0) + epsilon_);
        Eigen::MatrixXf normalized = centered / stdDev;
        
        // Scale and shift
        Eigen::MatrixXf transformed = gamma_(c) * normalized;
        transformed = transformed.colwise() + Eigen::VectorXf::Constant(featureSize_, beta_(c));
        
        // Place the transformed channel back in the output tensor
        insertChannel(output, transformed, c);
      } else {
        // Inference mode: use running statistics
        Eigen::MatrixXf centered = channelData.colwise() - 
            Eigen::VectorXf::Constant(featureSize_, runningMean_(c));
        float stdDev = std::sqrt(runningVar_(c) + epsilon_);
        Eigen::MatrixXf normalized = centered / stdDev;
        
        // Scale and shift
        Eigen::MatrixXf transformed = gamma_(c) * normalized;
        transformed = transformed.colwise() + Eigen::VectorXf::Constant(featureSize_, beta_(c));
        
        // Place the transformed channel back in the output tensor
        insertChannel(output, transformed, c);
      }
    }
    
    return output;
  }

  /**
   * @brief Backward pass
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate for parameter updates
   * @return Gradient with respect to the input
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput, float learningRate) override {
    const int batchSize = gradOutput.cols();
    const int totalFeatures = gradOutput.rows();
    
    if (totalFeatures != channels_ * height_ * width_) {
      throw std::invalid_argument(
          "ChannelBatchNormLayer::backward: gradient dimension mismatch. Expected: " +
          std::to_string(channels_ * height_ * width_) + 
          ", got: " + std::to_string(totalFeatures));
    }
    
    if (learningRate <= 0.0f) {
      throw std::invalid_argument("ChannelBatchNormLayer::backward: learningRate must be positive");
    }
    
    // Initialize input gradient
    Eigen::MatrixXf gradInput(totalFeatures, batchSize);
    
    // Reset parameter gradients
    gradGamma_.setZero();
    gradBeta_.setZero();
    
    // Process each channel separately
    for (int c = 0; c < channels_; ++c) {
      // Extract this channel's data
      Eigen::MatrixXf channelInput = extractChannel(input_, c);
      Eigen::MatrixXf channelGradOutput = extractChannel(gradOutput, c);
      
      // Get saved batch statistics
      Eigen::VectorXf batchMean = batchMeans_[c];
      Eigen::VectorXf batchVar = batchVars_[c];
      
      // Compute intermediate values
      float stdDev = std::sqrt(batchVar(0) + epsilon_);
      Eigen::MatrixXf centered = channelInput.colwise() - batchMean;
      Eigen::MatrixXf normalized = centered / stdDev;
      
      // Compute gradients for gamma and beta
      gradGamma_(c) = (normalized.array() * channelGradOutput.array()).sum();
      gradBeta_(c) = channelGradOutput.sum();
      
      // Compute gradient for normalized input
      Eigen::MatrixXf gradNormalized = channelGradOutput * gamma_(c);
      
      // Compute gradient through the normalization
      float invStdDev = 1.0f / stdDev;
      float invBatchSize = 1.0f / static_cast<float>(batchSize);
      
      // Gradient with respect to the centered input
      Eigen::MatrixXf gradCentered = gradNormalized * invStdDev;
      
      // Gradient with respect to the variance
      float gradVar = (gradNormalized.array() * centered.array()).sum() * -0.5f * 
                      std::pow(batchVar(0) + epsilon_, -1.5f);
      
      // Gradient with respect to the sum of squares
      float gradSumSquares = gradVar * invBatchSize;
      
      // Gradient with respect to the centered input (additional term from variance)
      Eigen::MatrixXf gradCenteredVar = 
          2.0f * centered * gradSumSquares;
      
      // Full gradient with respect to the centered input
      gradCentered += gradCenteredVar;
      
      // Gradient with respect to the mean
      Eigen::VectorXf gradMean = -gradCentered.rowwise().sum();
      
      // Gradient with respect to the input
      Eigen::MatrixXf channelGradInput = gradCentered + 
          gradMean * invBatchSize * Eigen::MatrixXf::Ones(featureSize_, batchSize);
      
      // Place the channel gradient back in the full gradient tensor
      insertChannel(gradInput, channelGradInput, c);
    }
    
    // Update parameters
    gamma_ -= learningRate * gradGamma_;
    beta_ -= learningRate * gradBeta_;
    
    // Clear saved batch statistics
    batchMeans_.clear();
    batchVars_.clear();
    
    return gradInput;
  }

  /**
   * @brief Set layer mode (training or inference)
   * @param training True for training mode, false for inference mode
   */
  void setTraining(bool training) override {
    training_ = training;
  }

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  std::string getType() const override {
    return "ChannelBatchNormLayer";
  }

  /**
   * @brief Implementation of getParameters virtual function
   * @return Vector containing gamma and beta parameters
   */
  std::vector<Eigen::MatrixXf> getParameters() const override {
    std::vector<Eigen::MatrixXf> params;
    // Convert gamma and beta to matrix form
    Eigen::MatrixXf gammaMatrix = gamma_;
    Eigen::MatrixXf betaMatrix = beta_;
    params.push_back(gammaMatrix);
    params.push_back(betaMatrix);
    return params;
  }

  /**
   * @brief Implementation of getGradients virtual function
   * @return Vector containing gradients for gamma and beta
   */
  std::vector<Eigen::MatrixXf> getGradients() const override {
    std::vector<Eigen::MatrixXf> grads;
    // Convert gradients to matrix form
    Eigen::MatrixXf gradGammaMatrix = gradGamma_;
    Eigen::MatrixXf gradBetaMatrix = gradBeta_;
    grads.push_back(gradGammaMatrix);
    grads.push_back(gradBetaMatrix);
    return grads;
  }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (0 for gamma, 1 for beta)
   * @param update Parameter update
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      // Update gamma
      gamma_ -= update.col(0);
    } else if (index == 1) {
      // Update beta
      beta_ -= update.col(0);
    } else {
      throw std::invalid_argument("ChannelBatchNormLayer::updateParameter: invalid parameter index");
    }
  }

private:
  /**
   * @brief Extract a single channel from the input tensor
   * @param input Input tensor
   * @param channel Channel index
   * @return Channel data as matrix
   */
  Eigen::MatrixXf extractChannel(const Eigen::MatrixXf &input, int channel) const {
    int startIdx = channel * featureSize_;
    int endIdx = startIdx + featureSize_ - 1;
    return input.block(startIdx, 0, featureSize_, input.cols());
  }
  
  /**
   * @brief Insert channel data back into the full tensor
   * @param output Output tensor to modify
   * @param channelData Channel data to insert
   * @param channel Channel index
   */
  void insertChannel(Eigen::MatrixXf &output, const Eigen::MatrixXf &channelData, int channel) const {
    int startIdx = channel * featureSize_;
    output.block(startIdx, 0, featureSize_, output.cols()) = channelData;
  }

  int channels_;              // Number of channels
  int height_;                // Feature map height
  int width_;                 // Feature map width
  int featureSize_;           // Size of each channel (height * width)
  float momentum_;            // Momentum for running statistics
  float epsilon_;             // Small value to avoid division by zero
  bool training_;             // Mode flag: training or inference
  
  Eigen::VectorXf gamma_;     // Scale parameter
  Eigen::VectorXf beta_;      // Shift parameter
  Eigen::VectorXf runningMean_; // Running mean for inference
  Eigen::VectorXf runningVar_;  // Running variance for inference
  
  Eigen::VectorXf gradGamma_; // Gradient for gamma
  Eigen::VectorXf gradBeta_;  // Gradient for beta
  
  Eigen::MatrixXf input_;     // Saved input for backward pass
  std::vector<Eigen::VectorXf> batchMeans_; // Saved batch means
  std::vector<Eigen::VectorXf> batchVars_;  // Saved batch variances
};

} // namespace zyraai

#endif // ZYRAAI_CHANNEL_BATCH_NORM_LAYER_H 