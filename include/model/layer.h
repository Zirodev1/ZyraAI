/**
 * @file layer.h
 * @brief Base layer class for neural network layers
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_LAYER_H
#define ZYRAAI_LAYER_H

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace zyraai {

/**
 * @class Layer
 * @brief Abstract base class for all neural network layers
 * 
 * This class defines the interface that all layers must implement, including
 * forward and backward passes, parameter management, and training state.
 */
class Layer {
public:
  /**
   * @brief Construct a new Layer
   * @param name The name of the layer
   * @param inputSize The input size of the layer
   * @param outputSize The output size of the layer
   */
  Layer(const ::std::string &name, int inputSize, int outputSize);
  
  /**
   * @brief Virtual destructor
   */
  virtual ~Layer() = default;

  /**
   * @brief Forward pass computation
   * @param input Input tensor
   * @return Output tensor
   */
  virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) = 0;

  /**
   * @brief Backward pass computation
   * @param gradOutput Gradient from the next layer
   * @param learningRate Learning rate for parameter updates
   * @return Gradient with respect to the input
   */
  virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                                   float learningRate) = 0;

  /**
   * @brief Get the layer's learnable parameters
   * @return Vector of parameter matrices
   */
  virtual ::std::vector<Eigen::MatrixXf> getParameters() const = 0;
  
  /**
   * @brief Get the gradients of the layer's parameters
   * @return Vector of gradient matrices
   */
  virtual ::std::vector<Eigen::MatrixXf> getGradients() const = 0;

  /**
   * @brief Get the layer name
   * @return Layer name
   */
  ::std::string getName() const { return name_; }
  
  /**
   * @brief Get the input size
   * @return Input size
   */
  int getInputSize() const { return inputSize_; }
  
  /**
   * @brief Get the output size
   * @return Output size
   */
  int getOutputSize() const { return outputSize_; }

  /**
   * @brief Set the layer's training mode
   * @param training Whether the layer is in training mode
   */
  virtual void setTraining(bool training) = 0;

  /**
   * @brief Update a parameter with the given update
   * @param index Parameter index
   * @param update Update value to apply
   */
  virtual void updateParameter(size_t index, const Eigen::MatrixXf &update) = 0;

  /**
   * @brief Get the type of the layer
   * @return String representing the layer type
   */
  virtual ::std::string getType() const = 0;

protected:
  ::std::string name_;   ///< Layer name
  int inputSize_;        ///< Input size
  int outputSize_;       ///< Output size
  bool isTraining_;      ///< Whether the layer is in training mode
};

} // namespace zyraai

#endif // ZYRAAI_LAYER_H