/**
 * @file relu_layer.h
 * @brief Rectified Linear Unit (ReLU) activation layer implementation
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_RELU_LAYER_H
#define ZYRAAI_RELU_LAYER_H

#include "model/layer.h"
#include "model/error_handler.h"
#include <stdexcept>

namespace zyraai {

/**
 * @class ReLULayer
 * @brief Implements the Rectified Linear Unit (ReLU) activation function
 * 
 * The ReLU activation function is defined as f(x) = max(0, x).
 * It is one of the most commonly used activation functions in neural networks
 * due to its simplicity and effectiveness in deep networks.
 */
class ReLULayer : public Layer {
public:
  /**
   * @brief Construct a new ReLU Layer
   * @param name Layer name
   * @param inputSize Input size (must equal outputSize)
   * @param outputSize Output size (must equal inputSize)
   * @throws error::InvalidArgument If dimensions are invalid
   */
  ReLULayer(const ::std::string &name, int inputSize, int outputSize)
      : Layer(name, inputSize, outputSize) {
    
    // ReLU layer must have equal input and output sizes
    if (inputSize != outputSize) {
      throw error::InvalidArgument(
          "Input and output sizes must be equal, got input: " +
          std::to_string(inputSize) + ", output: " + std::to_string(outputSize),
          "ReLULayer");
    }
  }

  /**
   * @brief Forward pass computation
   * @param input Input tensor
   * @return Output tensor with ReLU activation applied
   * @throws error::DimensionMismatch If input dimensions don't match expected size
   */
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    if (input.rows() != inputSize_) {
      throw error::DimensionMismatch(
          inputSize_, input.rows(), "ReLULayer::forward");
    }
    
    lastInput_ = input;
    return input.array().max(0.0f);
  }

  /**
   * @brief Backward pass computation
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate (unused for ReLU)
   * @return Gradient with respect to the input
   * @throws error::DimensionMismatch If gradient dimensions don't match expected size
   */
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    if (gradOutput.rows() != outputSize_) {
      throw error::DimensionMismatch(
          outputSize_, gradOutput.rows(), "ReLULayer::backward");
    }
    
    // Derivative of ReLU is 1 for x > 0, 0 otherwise
    return gradOutput.array() * (lastInput_.array() > 0.0f).cast<float>();
  }

  /**
   * @brief Get the layer's parameters
   * @return Empty vector as ReLU has no parameters
   */
  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // ReLU has no parameters
  }

  /**
   * @brief Get the parameter gradients
   * @return Empty vector as ReLU has no parameters
   */
  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // ReLU has no parameters
  }

  /**
   * @brief Set the layer to training or evaluation mode
   * @param training Whether the layer is in training mode
   */
  void setTraining(bool training) override { isTraining_ = training; }

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index (unused as ReLU has no parameters)
   * @param update Update value (unused as ReLU has no parameters)
   * @throws error::OutOfRange If index is not valid
   */
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // In debug builds, throw an error for invalid indices
    #ifndef NDEBUG
    if (index > 0) {
      throw error::OutOfRange(
          "Parameter index out of range, ReLU has no parameters", 
          "ReLULayer::updateParameter");
    }
    #endif
    // No parameters to update
  }

  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return outputSize_; }

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  ::std::string getType() const override {
    return "ReLULayer";
  }

private:
  Eigen::MatrixXf lastInput_;  ///< Cached input for computing gradients
};

} // namespace zyraai

#endif // ZYRAAI_RELU_LAYER_H