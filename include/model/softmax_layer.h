#ifndef ZYRAAI_SOFTMAX_LAYER_H
#define ZYRAAI_SOFTMAX_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>

namespace zyraai {

class SoftmaxLayer : public Layer {
public:
  SoftmaxLayer(const std::string &name, int size) : Layer(name, size, size) {}

  // Forward pass
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
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

  // Backward pass
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    const int numFeatures = gradOutput.rows();
    const int batchSize = gradOutput.cols();
    Eigen::MatrixXf gradInput = Eigen::MatrixXf::Zero(numFeatures, batchSize);

    for (int i = 0; i < batchSize; ++i) {
      Eigen::MatrixXf jacobian =
          Eigen::MatrixXf::Zero(numFeatures, numFeatures);
      for (int j = 0; j < numFeatures; ++j) {
        for (int k = 0; k < numFeatures; ++k) {
          float kronecker = (j == k) ? 1.0f : 0.0f;
          jacobian(j, k) = lastOutput_(j, i) * (kronecker - lastOutput_(k, i));
        }
      }
      gradInput.col(i) = jacobian * gradOutput.col(i);
    }

    return gradInput;
  }

  // Get layer parameters
  std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // Softmax has no parameters
  }

  std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // Softmax has no parameters
  }

  void setTraining(bool training) override { isTraining_ = training; }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // No parameters to update
  }

  int getInputSize() const { return inputSize_; }
  int getOutputSize() const { return inputSize_; }

private:
  Eigen::MatrixXf lastInput_;
  Eigen::MatrixXf lastOutput_;
};

} // namespace zyraai

#endif // ZYRAAI_SOFTMAX_LAYER_H