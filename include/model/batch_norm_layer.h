#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <iostream>

namespace zyraai {

class BatchNormLayer : public Layer {
public:
  BatchNormLayer(const std::string &name, int inputSize, int outputSize)
      : Layer(name, inputSize, outputSize), momentum_(0.9f), epsilon_(1e-5f) {
    gamma_ = Eigen::VectorXf::Ones(inputSize);
    beta_ = Eigen::VectorXf::Zero(inputSize);
    runningMean_ = Eigen::VectorXf::Zero(inputSize);
    runningVar_ = Eigen::VectorXf::Ones(inputSize);
  }

  BatchNormLayer(const std::string &name, int size)
      : Layer(name, size, size), momentum_(0.9f), epsilon_(1e-5f) {
    gamma_ = Eigen::VectorXf::Ones(size);
    beta_ = Eigen::VectorXf::Zero(size);
    runningMean_ = Eigen::VectorXf::Zero(size);
    runningVar_ = Eigen::VectorXf::Ones(size);
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    const int batchSize = input.cols();
    const int numFeatures = input.rows();

    // Compute mean and variance
    mean_ = input.rowwise().mean();
    centered_ = input.array().colwise() - mean_.array();
    var_ = (centered_.array() * centered_.array()).rowwise().mean();

    // Update running statistics during training
    if (isTraining_) {
      runningMean_ = momentum_ * runningMean_ + (1 - momentum_) * mean_;
      runningVar_ = momentum_ * runningVar_ + (1 - momentum_) * var_;
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

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    int batchSize = gradOutput.cols();
    float invBatchSize = 1.0f / static_cast<float>(batchSize);

    // Compute gradients for gamma and beta
    gradGamma_ = (normalized_.array() * gradOutput.array()).rowwise().sum();
    gradBeta_ = gradOutput.rowwise().sum();

    // Compute gradient with respect to normalized input
    Eigen::MatrixXf gradNormalized =
        gradOutput.array().colwise() * gamma_.array();

    // Compute gradient with respect to variance
    Eigen::VectorXf sumGradNormalized =
        (normalized_.array() * gradNormalized.array()).rowwise().sum();
    Eigen::VectorXf gradVar =
        (-0.5f * stdInv_.array().cube() * sumGradNormalized.array()).matrix();

    // Compute gradient with respect to mean
    Eigen::VectorXf sumGradNormalizedWeighted =
        (gradNormalized.array().colwise() * stdInv_.array()).rowwise().sum();
    Eigen::VectorXf sumCentered = centered_.rowwise().sum();
    Eigen::VectorXf gradMean =
        -sumGradNormalizedWeighted -
        2.0f * invBatchSize * (gradVar.array() * sumCentered.array()).matrix();

    // Compute gradient with respect to input
    float scale = 2.0f * invBatchSize;
    Eigen::MatrixXf term1 = gradNormalized.array().colwise() * stdInv_.array();
    Eigen::MatrixXf term2 =
        centered_.array().colwise() * (scale * gradVar.array());
    Eigen::MatrixXf term3 = gradMean.array().replicate(1, batchSize);
    gradInput_ = term1 + term2 + term3 * invBatchSize;

    // Update parameters
    gamma_ = gamma_.array() - learningRate * gradGamma_.array();
    beta_ = beta_.array() - learningRate * gradBeta_.array();

    return gradInput_;
  }

  std::vector<Eigen::MatrixXf> getParameters() const override {
    return {gamma_, beta_};
  }

  std::vector<Eigen::MatrixXf> getGradients() const override {
    return {gradGamma_, gradBeta_};
  }

  void setTraining(bool training) { isTraining_ = training; }

private:
  float momentum_;
  float epsilon_;
  bool isTraining_ = true;

  Eigen::VectorXf gamma_;
  Eigen::VectorXf beta_;
  Eigen::VectorXf runningMean_;
  Eigen::VectorXf runningVar_;

  Eigen::VectorXf mean_;
  Eigen::VectorXf var_;
  Eigen::VectorXf stdInv_;
  Eigen::MatrixXf centered_;
  Eigen::MatrixXf normalized_;
  Eigen::MatrixXf input_;
  Eigen::VectorXf gradGamma_;
  Eigen::VectorXf gradBeta_;
  Eigen::MatrixXf gradInput_;
};

} // namespace zyraai