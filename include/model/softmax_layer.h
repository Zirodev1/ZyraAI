/**
 * @file softmax_layer.h
 * @brief Softmax activation layer implementation for neural networks
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_SOFTMAX_LAYER_H
#define ZYRAAI_SOFTMAX_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>
#include <stdexcept>

namespace zyraai {

/**
 * @class SoftmaxLayer
 * @brief Implements the softmax activation function
 * 
 * The softmax function normalizes input values into a probability distribution,
 * where each output value is in the range [0,1] and all outputs sum to 1.
 * It is commonly used as the final activation in classification networks.
 * 
 * softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
 */
class SoftmaxLayer : public Layer {
public:
  /**
   * @brief Construct a new Softmax Layer
   * @param name Layer name
   * @param size Input and output size
   * @throws std::invalid_argument If size is invalid
   */
  SoftmaxLayer(const std::string &name, int size) 
      : Layer(name, size, size) {
    
    if (size <= 0) {
      throw std::invalid_argument(
          "SoftmaxLayer: size must be positive, got: " + 
          std::to_string(size));
    }
  }

  /**
   * @brief Forward pass computation of softmax activation
   * @param input Input tensor of shape [features, batchSize]
   * @return Output tensor of same shape, with softmax applied per column
   * @throws std::invalid_argument If input dimensions don't match expected size
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    if (input.rows() != inputSize_) {
      throw std::invalid_argument(
          "SoftmaxLayer::forward: input dimension mismatch. Expected: " +
          std::to_string(inputSize_) + ", got: " + std::to_string(input.rows()));
    }
    
    lastInput_ = input;
    const int numFeatures = input.rows();
    const int batchSize = input.cols();

    // Compute max for numerical stability
    Eigen::VectorXf maxVals = input.colwise().maxCoeff();
    Eigen::MatrixXf centered = input;
    for (int i = 0; i < batchSize; ++i) {
      centered.col(i).array() -= maxVals(i);
    }

    // Compute exp and sum
    Eigen::MatrixXf expValues = centered.array().exp();
    Eigen::VectorXf sumExp = expValues.colwise().sum();

    // Normalize
    lastOutput_ = Eigen::MatrixXf(numFeatures, batchSize);
    for (int i = 0; i < batchSize; ++i) {
      lastOutput_.col(i) = expValues.col(i) / sumExp(i);
    }

    return lastOutput_;
  }

  /**
   * @brief Backward pass computation
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate (unused for softmax)
   * @return Gradient with respect to the input
   * @throws std::invalid_argument If gradient dimensions don't match expected size
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    if (gradOutput.rows() != outputSize_) {
      throw std::invalid_argument(
          "SoftmaxLayer::backward: gradient dimension mismatch. Expected: " +
          std::to_string(outputSize_) + ", got: " + std::to_string(gradOutput.rows()));
    }
    
    const int numFeatures = gradOutput.rows();
    const int batchSize = gradOutput.cols();
    Eigen::MatrixXf gradInput = Eigen::MatrixXf::Zero(numFeatures, batchSize);

    // Compute Jacobian for each batch sample
    for (int i = 0; i < batchSize; ++i) {
      Eigen::MatrixXf jacobian =
          Eigen::MatrixXf::Zero(numFeatures, numFeatures);
      
      // Jacobian of softmax: J_ij = softmax_i * (delta_ij - softmax_j)
      // Where delta_ij is 1 if i=j, 0 otherwise (Kronecker delta)
      for (int j = 0; j < numFeatures; ++j) {
        for (int k = 0; k < numFeatures; ++k) {
          float kronecker = (j == k) ? 1.0f : 0.0f;
          jacobian(j, k) = lastOutput_(j, i) * (kronecker - lastOutput_(k, i));
        }
      }
      
      // Multiply Jacobian by incoming gradient
      gradInput.col(i) = jacobian * gradOutput.col(i);
    }

    return gradInput;
  }

  /**
   * @brief Get the layer's parameters
   * @return Empty vector as softmax has no parameters
   */
  std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // Softmax has no parameters
  }

  /**
   * @brief Get the parameter gradients
   * @return Empty vector as softmax has no parameters
   */
  std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // Softmax has no parameters
  }

  /**
   * @brief Set the layer to training or evaluation mode
   * @param training Whether the layer is in training mode
   */
  void setTraining(bool training) override { isTraining_ = training; }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (unused as softmax has no parameters)
   * @param update Update value (unused as softmax has no parameters)
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // No parameters to update
  }

  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return inputSize_; }

private:
  Eigen::MatrixXf lastInput_;   ///< Cached input for backward pass
  Eigen::MatrixXf lastOutput_;  ///< Cached output for computing gradients
};

} // namespace zyraai

#endif // ZYRAAI_SOFTMAX_LAYER_H