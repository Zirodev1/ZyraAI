/**
 * @file dense_layer.h
 * @brief Fully connected (dense) layer implementation for neural networks
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_DENSE_LAYER_H
#define ZYRAAI_DENSE_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace zyraai {

/**
 * @class DenseLayer
 * @brief Implements a fully connected (dense) layer for neural networks
 * 
 * A dense layer performs a linear transformation of the input followed by
 * an activation function. By default, it uses ReLU activation but can be
 * extended to use other activations by overriding the activation methods.
 */
class DenseLayer : public Layer {
public:
  /**
   * @brief Construct a new Dense Layer
   * @param name Layer name
   * @param inputSize Number of input features
   * @param outputSize Number of output features
   * @param useBias Whether to include bias terms (default: true)
   * @throws std::invalid_argument If input or output size is invalid
   */
  DenseLayer(const ::std::string &name, int inputSize, int outputSize,
             bool useBias = true);

  /**
   * @brief Forward pass computation
   * @param input Input tensor of shape [inputSize, batchSize]
   * @return Output tensor of shape [outputSize, batchSize]
   * @throws std::invalid_argument If input dimensions don't match expected size
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

  /**
   * @brief Backward pass computation
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate for parameter updates
   * @return Gradient with respect to the input
   * @throws std::invalid_argument If gradient dimensions don't match expected size
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override;

  /**
   * @brief Get the layer's parameters
   * @return Vector containing the weights and optionally bias matrices
   */
  ::std::vector<Eigen::MatrixXf> getParameters() const override;
  
  /**
   * @brief Get the parameter gradients
   * @return Vector containing the weight and optionally bias gradients
   */
  ::std::vector<Eigen::MatrixXf> getGradients() const override;

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  ::std::string getType() const override {
    return "DenseLayer";
  }

  /**
   * @brief Apply activation function to the linear output
   * @param input Linear output tensor
   * @return Activated tensor
   * 
   * Default implementation uses ReLU: f(x) = max(0, x)
   */
  virtual Eigen::MatrixXf activation(const Eigen::MatrixXf &input) {
    return input.array().max(0.0f);
  }

  /**
   * @brief Compute gradient of the activation function
   * @param input Linear output tensor (same as in activation)
   * @return Gradient of activation function at the input points
   * 
   * Default implementation is derivative of ReLU: f'(x) = 1 if x > 0, else 0
   */
  virtual Eigen::MatrixXf activationGradient(const Eigen::MatrixXf &input) {
    return (input.array() > 0.0f).cast<float>();
  }

  /**
   * @brief Set the layer to training or evaluation mode
   * @param training Whether the layer is in training mode
   */
  void setTraining(bool training) override { isTraining_ = training; }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (0=weights, 1=bias if useBias=true)
   * @param update Update value to apply
   * @throws std::out_of_range If index is invalid
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      weights_ -= update;
    } else if (index == 1 && useBias_) {
      bias_ -= update;
    } else {
      throw std::out_of_range("DenseLayer: parameter index out of range");
    }
  }

  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return outputSize_; }

protected:
  Eigen::MatrixXf weights_;           ///< Weight matrix [outputSize x inputSize]
  Eigen::MatrixXf bias_;              ///< Bias vector [outputSize x 1]
  Eigen::MatrixXf gradWeights_;       ///< Weight gradients
  Eigen::MatrixXf gradBias_;          ///< Bias gradients
  Eigen::MatrixXf lastInput_;         ///< Cached input for backpropagation
  Eigen::MatrixXf lastPreActivation_; ///< Cached pre-activation output
  bool useBias_;                      ///< Whether to use bias terms
};

} // namespace zyraai

#endif // ZYRAAI_DENSE_LAYER_H