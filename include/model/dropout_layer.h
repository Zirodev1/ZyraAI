#ifndef ZYRAAI_DROPOUT_LAYER_H
#define ZYRAAI_DROPOUT_LAYER_H

#include "model/layer.h"
#include <Eigen/Dense>
#include <random>

namespace zyraai {

class DropoutLayer : public Layer {
public:
  DropoutLayer(const std::string &name, int size, float dropoutRate = 0.5f)
      : Layer(name, size, size), dropoutRate_(dropoutRate),
        mask_(Eigen::MatrixXf::Ones(size, 1)), isTraining_(true) {
    // Initialize random number generator
    std::random_device rd;
    gen_ = std::mt19937(rd());
    dist_ = std::uniform_real_distribution<float>(0.0f, 1.0f);
  }

  // Forward pass
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    if (isTraining_) {
      // Create dropout mask
      mask_ = Eigen::MatrixXf::Ones(input.rows(), input.cols());
      for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
          if (dist_(gen_) < dropoutRate_) {
            mask_(i, j) = 0.0f;
          }
        }
      }
      // Apply dropout mask and scale
      return (input.array() * mask_.array()) / (1.0f - dropoutRate_);
    } else {
      // During inference, no dropout is applied
      return input;
    }
  }

  // Backward pass
  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    // Apply the same mask to the gradient
    return (gradOutput.array() * mask_.array()) / (1.0f - dropoutRate_);
  }

  // Get layer parameters (dropout has no trainable parameters)
  std::vector<Eigen::MatrixXf> getParameters() const override {
    return std::vector<Eigen::MatrixXf>();
  }

  // Get layer gradients (dropout has no gradients)
  std::vector<Eigen::MatrixXf> getGradients() const override {
    return std::vector<Eigen::MatrixXf>();
  }

  // Set training mode
  void setTraining(bool training) { isTraining_ = training; }

private:
  float dropoutRate_;    // Probability of dropping a neuron
  Eigen::MatrixXf mask_; // Dropout mask
  bool isTraining_;      // Whether we're in training mode
  std::mt19937 gen_;     // Random number generator
  std::uniform_real_distribution<float>
      dist_; // Distribution for random numbers
};

} // namespace zyraai

#endif // ZYRAAI_DROPOUT_LAYER_H