#pragma once

#include "layer.h"
#include <Eigen/Dense>

namespace zyraai {

class FlattenLayer : public Layer {
public:
  FlattenLayer(const ::std::string &name, int inputChannels, int inputHeight,
               int inputWidth)
      : Layer(name, inputChannels * inputHeight * inputWidth,
              inputChannels * inputHeight * inputWidth),
        inputChannels_(inputChannels), inputHeight_(inputHeight),
        inputWidth_(inputWidth) {}

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    // No transformation needed for forward pass, just pass through
    return input;
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    // No transformation needed for backward pass, just pass through
    return gradOutput;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    return {}; // No trainable parameters
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    return {}; // No gradients
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // No parameters to update
  }

  void setTraining(bool training) override { isTraining_ = training; }

private:
  int inputChannels_;
  int inputHeight_;
  int inputWidth_;
  bool isTraining_ = true;
};

} // namespace zyraai