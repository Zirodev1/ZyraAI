#ifndef ZYRAAI_RELU_LAYER_H
#define ZYRAAI_RELU_LAYER_H

#include "model/layer.h"

namespace zyraai {

class ReLULayer : public Layer {
public:
  ReLULayer(const std::string &name, int inputSize, int outputSize)
      : Layer(name, inputSize, outputSize) {}

  // Forward pass
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    lastInput_ = input;
    return input.array().max(0.0f);
  }

  // Backward pass
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    return gradOutput.array() * (lastInput_.array() > 0.0f).cast<float>();
  }

  // Get layer parameters
  std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // ReLU has no parameters
  }

  std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // ReLU has no gradients
  }

private:
  Eigen::MatrixXf lastInput_;
};

} // namespace zyraai

#endif // ZYRAAI_RELU_LAYER_H