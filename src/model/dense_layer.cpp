/**
 * @file dense_layer.cpp
 * @brief Implementation of the fully connected (dense) layer
 * @author ZyraAI Team
 */

#include "model/dense_layer.h"
#include <random>
#include <stdexcept>
#include <cmath>

namespace zyraai {

DenseLayer::DenseLayer(const std::string &name, int inputSize, int outputSize,
                       bool useBias)
    : Layer(name, inputSize, outputSize), useBias_(useBias) {

  // Additional validation beyond base Layer class
  if (outputSize <= 0 && useBias) {
    throw std::invalid_argument("DenseLayer: output size must be positive when using bias");
  }

  // Initialize weights with He initialization (good for ReLU activation)
  std::random_device rd;
  std::mt19937 gen(rd());
  float scale = std::sqrt(2.0f / inputSize); // He initialization for ReLU
  std::normal_distribution<float> dist(0.0f, scale);

  // Allocate and initialize weight matrix [outputSize x inputSize]
  weights_ = Eigen::MatrixXf::Zero(outputSize, inputSize);
  for (int i = 0; i < weights_.rows(); ++i) {
    for (int j = 0; j < weights_.cols(); ++j) {
      weights_(i, j) = dist(gen);
    }
  }

  // Initialize bias vector if needed
  if (useBias_) {
    bias_ = Eigen::MatrixXf::Zero(outputSize, 1);
    // Small initialization for biases
    for (int i = 0; i < bias_.rows(); ++i) {
      bias_(i, 0) = dist(gen) * 0.1f; // Scale bias initialization
    }
  }

  // Initialize gradient matrices/vectors
  gradWeights_ = Eigen::MatrixXf::Zero(outputSize, inputSize);
  if (useBias_) {
    gradBias_ = Eigen::MatrixXf::Zero(outputSize, 1);
  }
}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf &input) {
  // Validate input dimensions
  if (input.rows() != inputSize_) {
    throw std::invalid_argument(
        "DenseLayer::forward: input dimension mismatch. Expected: " +
        std::to_string(inputSize_) + ", got: " + std::to_string(input.rows()));
  }
  
  if (input.cols() <= 0) {
    throw std::invalid_argument(
        "DenseLayer::forward: batch size must be positive, got: " +
        std::to_string(input.cols()));
  }

  // Store input for backpropagation
  lastInput_ = input;

  // Linear transformation: y = Wx + b
  lastPreActivation_ = weights_ * input;
  if (useBias_) {
    // Add bias to each column (each sample in the batch)
    for (int i = 0; i < input.cols(); ++i) {
      lastPreActivation_.col(i) += bias_;
    }
  }

  // Apply activation function and return
  return activation(lastPreActivation_);
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf &gradOutput,
                                     float learningRate) {
  // Validate gradient dimensions
  if (gradOutput.rows() != outputSize_) {
    throw std::invalid_argument(
        "DenseLayer::backward: gradient dimension mismatch. Expected: " +
        std::to_string(outputSize_) + ", got: " + std::to_string(gradOutput.rows()));
  }
  
  if (gradOutput.cols() <= 0) {
    throw std::invalid_argument(
        "DenseLayer::backward: batch size must be positive, got: " +
        std::to_string(gradOutput.cols()));
  }
  
  if (learningRate <= 0.0f) {
    throw std::invalid_argument(
        "DenseLayer::backward: learning rate must be positive, got: " +
        std::to_string(learningRate));
  }

  // Compute gradients with respect to the activation
  Eigen::MatrixXf gradActivation = activationGradient(lastPreActivation_);
  
  // Element-wise multiplication of incoming gradient and activation gradient
  Eigen::MatrixXf delta = gradOutput.array() * gradActivation.array();

  // Compute gradients with respect to weights and bias
  // Normalize by batch size for more stable training
  int batchSize = delta.cols();
  gradWeights_ = (delta * lastInput_.transpose()) / batchSize;
  
  if (useBias_) {
    gradBias_ = delta.rowwise().sum() / batchSize;
  }

  // Gradient clipping to prevent exploding gradients
  const float maxNorm = 1.0f;
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

  // Compute gradients with respect to input for backpropagation
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