#ifndef ZYRAAI_DENSE_LAYER_H
#define ZYRAAI_DENSE_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>
#include <vector>

namespace zyraai {

class DenseLayer : public Layer {
public:
  DenseLayer(const std::string &name, int inputSize, int outputSize,
             bool useBias = true);

  // Forward pass
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

  // Backward pass
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override;

  // Get layer parameters
  std::vector<Eigen::MatrixXf> getParameters() const override;
  std::vector<Eigen::MatrixXf> getGradients() const override;

  // Activation function
  virtual Eigen::MatrixXf activation(const Eigen::MatrixXf &input) {
    return input.array().max(0.0f);
  }

  virtual Eigen::MatrixXf activationGradient(const Eigen::MatrixXf &input) {
    return (input.array() > 0.0f).cast<float>();
  }

protected:
  Eigen::MatrixXf weights_;           // Weight matrix
  Eigen::MatrixXf bias_;              // Bias vector
  Eigen::MatrixXf gradWeights_;       // Weight gradients
  Eigen::MatrixXf gradBias_;          // Bias gradients
  Eigen::MatrixXf lastInput_;         // Last input for backpropagation
  Eigen::MatrixXf lastPreActivation_; // Last pre-activation output
  bool useBias_;                      // Whether to use bias terms
};

} // namespace zyraai

#endif // ZYRAAI_DENSE_LAYER_H