#pragma once

#include "layer.h"
#include <Eigen/Dense>
#include <vector>

namespace zyraai {

// Simple identity layer that passes input directly to output
// Useful for implementing skip connections
class IdentityLayer : public Layer {
public:
  IdentityLayer(const ::std::string &name, int size)
      : Layer(name, size, size) {}

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    input_ = input; // Store for backward pass
    return input;
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    // Just pass the gradient straight through
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

  /**
   * @brief Get the layer type
   * @return Layer type as string
   */
  ::std::string getType() const override {
    return "IdentityLayer";
  }

private:
  Eigen::MatrixXf input_; // Stored for backward pass
  bool isTraining_ = true;
};

} // namespace zyraai