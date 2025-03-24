#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <iostream>

namespace zyraai {

class BatchNormLayer : public Layer {
public:
  BatchNormLayer(const ::std::string &name, int inputSize, int outputSize)
      : Layer(name, inputSize, outputSize), momentum_(0.9f), epsilon_(1e-5f) {
    gamma_ = Eigen::VectorXf::Ones(inputSize);
    beta_ = Eigen::VectorXf::Zero(inputSize);
    runningMean_ = Eigen::VectorXf::Zero(inputSize);
    runningVar_ = Eigen::VectorXf::Ones(inputSize);
  }

  BatchNormLayer(const ::std::string &name, int size)
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

  // Replace the entire backward method in include/model/batch_norm_layer.h
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    int batchSize = gradOutput.cols();
    float invBatchSize = 1.0f / static_cast<float>(batchSize);

    // Compute gradients for gamma and beta
    gradGamma_ = (normalized_.array() * gradOutput.array()).rowwise().sum();
    gradBeta_ = gradOutput.rowwise().sum();

    // Compute gradient with respect to input (simplified)
    Eigen::MatrixXf gradNormalized =
        gradOutput.array().colwise() * gamma_.array();

    // Gradient with respect to input
    gradInput_ = gradNormalized.array().colwise() * stdInv_.array();

    return gradInput_;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    return {gamma_, beta_};
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    return {gradGamma_, gradBeta_};
  }

  // Add this method to the public section in include/model/batch_norm_layer.h
  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    if (index == 0) {
      gamma_ -= update;
    } else if (index == 1) {
      beta_ -= update;
    }
  }

  void setTraining(bool training) override { isTraining_ = training; }

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