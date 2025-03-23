#include "model/dense_layer.h"
#include <random>

namespace zyraai {

DenseLayer::DenseLayer(const std::string &name, int inputSize, int outputSize,
                       bool useBias)
    : Layer(name, inputSize, outputSize), useBias_(useBias) {

  // Initialize weights with He initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  float scale = std::sqrt(2.0f / inputSize); // He initialization
  std::normal_distribution<float> dist(0.0f, scale);

  weights_ = Eigen::MatrixXf::Zero(outputSize, inputSize);
  for (int i = 0; i < weights_.rows(); ++i) {
    for (int j = 0; j < weights_.cols(); ++j) {
      weights_(i, j) = dist(gen);
    }
  }

  // Initialize bias if needed
  if (useBias_) {
    bias_ = Eigen::MatrixXf::Zero(outputSize, 1);
    for (int i = 0; i < bias_.rows(); ++i) {
      bias_(i, 0) = dist(gen);
    }
  }

  // Initialize gradients
  gradWeights_ = Eigen::MatrixXf::Zero(outputSize, inputSize);
  if (useBias_) {
    gradBias_ = Eigen::MatrixXf::Zero(outputSize, 1);
  }
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf &input) {
  lastInput_ = input;

  // Linear transformation
  lastPreActivation_ = weights_ * input;
  if (useBias_) {
    for (int i = 0; i < input.cols(); ++i) {
      lastPreActivation_.col(i) += bias_;
    }
  }

  // Apply activation function
  return activation(lastPreActivation_);
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf &gradOutput,
                                     float learningRate) {
  // Compute gradients with respect to the activation
  Eigen::MatrixXf gradActivation = activationGradient(lastPreActivation_);
  Eigen::MatrixXf delta = gradOutput.array() * gradActivation.array();

  // Compute gradients with respect to weights and bias
  gradWeights_ = (delta * lastInput_.transpose()) /
                 delta.cols(); // Normalize by batch size
  if (useBias_) {
    gradBias_ = delta.rowwise().sum() / delta.cols(); // Normalize by batch size
  }

  // Gradient clipping
  float maxNorm = 1.0f;
  float gradNorm = gradWeights_.norm();
  if (gradNorm > maxNorm) {
    gradWeights_ *= maxNorm / gradNorm;
  }
  if (useBias_) {
    gradNorm = gradBias_.norm();
    if (gradNorm > maxNorm) {
      gradBias_ *= maxNorm / gradNorm;
    }
  }

  // Update weights and bias
  weights_ -= learningRate * gradWeights_;
  if (useBias_) {
    bias_ -= learningRate * gradBias_;
  }

  // Compute gradients with respect to input
  return weights_.transpose() * delta;
}

std::vector<Eigen::MatrixXf> DenseLayer::getParameters() const {
  std::vector<Eigen::MatrixXf> params;
  params.push_back(weights_);
  if (useBias_) {
    params.push_back(bias_);
  }
  return params;
}

std::vector<Eigen::MatrixXf> DenseLayer::getGradients() const {
  std::vector<Eigen::MatrixXf> grads;
  grads.push_back(gradWeights_);
  if (useBias_) {
    grads.push_back(gradBias_);
  }
  return grads;
}

} // namespace zyraai