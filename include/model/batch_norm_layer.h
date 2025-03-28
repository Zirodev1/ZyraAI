/**
 * @file batch_norm_layer.h
 * @brief Batch normalization layer implementation
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_BATCH_NORM_LAYER_H
#define ZYRAAI_BATCH_NORM_LAYER_H

#include "layer.h"
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

namespace zyraai {

/**
 * @class BatchNormLayer
 * @brief Implements batch normalization for neural network layers
 * 
 * Batch normalization normalizes the input across the batch dimension by
 * subtracting the batch mean and dividing by the batch standard deviation.
 * It then scales and shifts the result using learnable parameters.
 */
class BatchNormLayer : public Layer {
public:
  /**
   * @brief Construct a batch normalization layer
   * @param name Layer name
   * @param inputSize Input feature size
   * @param outputSize Output feature size (must equal inputSize)
   * @throws std::invalid_argument If inputSize != outputSize
   */
  BatchNormLayer(const ::std::string &name, int inputSize, int outputSize)
      : Layer(name, inputSize, outputSize), momentum_(0.9f), epsilon_(1e-5f) {
    
    if (inputSize != outputSize) {
      throw std::invalid_argument("BatchNormLayer: inputSize must equal outputSize");
    }
    
    gamma_ = Eigen::VectorXf::Ones(inputSize);
    beta_ = Eigen::VectorXf::Zero(inputSize);
    runningMean_ = Eigen::VectorXf::Zero(inputSize);
    runningVar_ = Eigen::VectorXf::Ones(inputSize);
  }

  /**
   * @brief Construct a batch normalization layer with equal input/output sizes
   * @param name Layer name
   * @param size Feature size
   */
  BatchNormLayer(const ::std::string &name, int size)
      : Layer(name, size, size), momentum_(0.9f), epsilon_(1e-5f) {
    
    if (size <= 0) {
      throw std::invalid_argument("BatchNormLayer: size must be positive");
    }
    
    gamma_ = Eigen::VectorXf::Ones(size);
    beta_ = Eigen::VectorXf::Zero(size);
    runningMean_ = Eigen::VectorXf::Zero(size);
    runningVar_ = Eigen::VectorXf::Ones(size);
  }

  /**
   * @brief Forward pass of batch normalization
   * @param input Input tensor of shape [features, batch_size]
   * @return Normalized output tensor of same shape
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    if (input.rows() != inputSize_) {
      throw std::invalid_argument(
          "BatchNormLayer: input feature dimension mismatch. Expected: " +
          std::to_string(inputSize_) + ", got: " + std::to_string(input.rows()));
    }
    
    const int batchSize = input.cols();
    const int numFeatures = input.rows();

    input_ = input; // Store for backward pass

    // Compute mean and variance
    mean_ = input.rowwise().mean();
    centered_ = input.array().colwise() - mean_.array();
    var_ = (centered_.array() * centered_.array()).rowwise().mean();

    // Update running statistics during training
    if (isTraining_) {
      runningMean_ = momentum_ * runningMean_ + (1.0f - momentum_) * mean_;
      runningVar_ = momentum_ * runningVar_ + (1.0f - momentum_) * var_;
    }

    // Compute standard deviation and its inverse
    stdInv_ = (var_.array() + epsilon_).sqrt().inverse();

    // Store normalized input for backward pass
    normalized_ = (centered_.array().colwise() * stdInv_.array()).matrix();

    // Scale and shift
    Eigen::MatrixXf result =
        (normalized_.array().colwise() * gamma_.array()).matrix();
    result = (result.array().colwise() + beta_.array()).matrix();

    return result;
  }

  /**
   * @brief Backward pass of batch normalization
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate for parameter updates
   * @return Gradient with respect to the input
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    if (gradOutput.rows() != outputSize_) {
      throw std::invalid_argument(
          "BatchNormLayer: gradient dimension mismatch. Expected: " +
          std::to_string(outputSize_) + ", got: " + std::to_string(gradOutput.rows()));
    }
                               
    int batchSize = gradOutput.cols();
    float invBatchSize = 1.0f / static_cast<float>(batchSize);

    // Compute gradients for gamma and beta
    gradGamma_ = (normalized_.array() * gradOutput.array()).rowwise().sum();
    gradBeta_ = gradOutput.rowwise().sum();

    // Compute gradient with respect to input (simplified)
    Eigen::MatrixXf gradNormalized =
        gradOutput.array().colwise() * gamma_.array();

    // Gradient with respect to input
    gradInput_ = gradNormalized.array().colwise() * stdInv_.array();

    return gradInput_;
  }

  /**
   * @brief Get the model parameters
   * @return Vector of parameters (gamma, beta)
   */
  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    return {gamma_, beta_};
  }

  /**
   * @brief Get the parameter gradients
   * @return Vector of gradients (gradGamma, gradBeta)
   */
  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    return {gradGamma_, gradBeta_};
  }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (0=gamma, 1=beta)
   * @param update Update value to apply
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      gamma_ -= update;
    } else if (index == 1) {
      beta_ -= update;
    } else {
      throw std::out_of_range("BatchNormLayer: parameter index out of range");
    }
  }

  /**
   * @brief Set the layer to training or evaluation mode
   * @param training Whether the layer is in training mode
   */
  void setTraining(bool training) override { isTraining_ = training; }

private:
  float momentum_;    ///< Momentum factor for running statistics
  float epsilon_;     ///< Small constant for numerical stability
  bool isTraining_;   ///< Whether the layer is in training mode

  // Learnable parameters
  Eigen::VectorXf gamma_;        ///< Scale parameter
  Eigen::VectorXf beta_;         ///< Shift parameter
  
  // Running statistics for inference
  Eigen::VectorXf runningMean_;  ///< Running mean for inference
  Eigen::VectorXf runningVar_;   ///< Running variance for inference

  // Temporary variables for forward/backward pass
  Eigen::VectorXf mean_;         ///< Batch mean
  Eigen::VectorXf var_;          ///< Batch variance
  Eigen::VectorXf stdInv_;       ///< Inverse of standard deviation
  Eigen::MatrixXf centered_;     ///< Input centered by mean
  Eigen::MatrixXf normalized_;   ///< Normalized input
  Eigen::MatrixXf input_;        ///< Stored input
  Eigen::VectorXf gradGamma_;    ///< Gradient for gamma
  Eigen::VectorXf gradBeta_;     ///< Gradient for beta
  Eigen::MatrixXf gradInput_;    ///< Gradient with respect to input
};

} // namespace zyraai

#endif // ZYRAAI_BATCH_NORM_LAYER_H