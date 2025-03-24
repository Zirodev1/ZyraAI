#ifndef ZYRAAI_DENSE_LAYER_H
#define ZYRAAI_DENSE_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>
#include <vector>

namespace zyraai {

class DenseLayer : public Layer {
public:
  DenseLayer(const ::std::string &name, int inputSize, int outputSize,
             bool useBias = true);

  // Forward pass
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override;

  // Backward pass
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override;

  // Get layer parameters
  ::std::vector<Eigen::MatrixXf> getParameters() const override;
  ::std::vector<Eigen::MatrixXf> getGradients() const override;

  // Activation function
  virtual Eigen::MatrixXf activation(const Eigen::MatrixXf &input) {
    return input.array().max(0.0f);
  }

  virtual Eigen::MatrixXf activationGradient(const Eigen::MatrixXf &input) {
    return (input.array() > 0.0f).cast<float>();
  }

  void setTraining(bool training) override { isTraining_ = training; }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      weights_ -= update;
    } else if (index == 1 && useBias_) {
      bias_ -= update;
    }
  }

  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return outputSize_; }

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