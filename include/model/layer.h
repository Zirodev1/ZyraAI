#ifndef ZYRAAI_LAYER_H
#define ZYRAAI_LAYER_H

#include <Eigen/Dense>
#include <memory>
#include <string>

namespace zyraai {

class Layer {
public:
  Layer(const std::string &name, int inputSize, int outputSize);
  virtual ~Layer() = default;

  // Forward pass
  virtual Eigen::MatrixXf forward(const Eigen::MatrixXf &input) = 0;

  // Backward pass
  virtual Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                                   float learningRate) = 0;

  // Get layer parameters
  virtual std::vector<Eigen::MatrixXf> getParameters() const = 0;
  virtual std::vector<Eigen::MatrixXf> getGradients() const = 0;

  // Layer information
  std::string getName() const { return name_; }
  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return outputSize_; }

protected:
  std::string name_;
  int inputSize_;
  int outputSize_;
  bool isTraining_;
};

} // namespace zyraai

#endif // ZYRAAI_LAYER_H