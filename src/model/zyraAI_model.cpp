// src/model/ziroai_model.cpp

#include "model/zyraAI_model.h"
#include "model/batch_norm_layer.h"
#include "model/dropout_layer.h"
#include <cmath>
#include <iostream>

namespace zyraai {
ZyraAIModel::ZyraAIModel() {
  std::cout << "ZyraAI Model Initialized" << std::endl;
}

ZyraAIModel::~ZyraAIModel() {
  std::cout << "ZyraAI Model Destroyed" << std::endl;
}

void ZyraAIModel::addLayer(std::shared_ptr<Layer> layer) {
  layers_.push_back(layer);
}

Eigen::MatrixXf ZyraAIModel::forward(const Eigen::MatrixXf &input) {
  activations_.clear();
  activations_.push_back(input);

  Eigen::MatrixXf current = input;
  for (const auto &layer : layers_) {
    current = layer->forward(current);
    activations_.push_back(current);
  }

  return current;
}

void ZyraAIModel::backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) {
  Eigen::MatrixXf currentGrad = gradOutput;

  // Backward pass through layers in reverse order
  for (int i = layers_.size() - 1; i >= 0; --i) {
    currentGrad = layers_[i]->backward(currentGrad, learningRate);
  }
}

float ZyraAIModel::train(const Eigen::MatrixXf &input,
                         const Eigen::MatrixXf &target, float learningRate) {
  // Forward pass
  Eigen::MatrixXf output = forward(input);

  // Compute loss
  float loss = computeLoss(output, target);

  // Compute gradients (using cross-entropy loss with softmax)
  // For cross-entropy loss with softmax, the gradient is (output - target)
  // We also normalize by batch size for stability
  Eigen::MatrixXf gradOutput =
      (output - target) / static_cast<float>(target.cols());

  // Backward pass
  backward(gradOutput, learningRate);

  return loss;
}

float ZyraAIModel::computeLoss(const Eigen::MatrixXf &output,
                               const Eigen::MatrixXf &target) {
  const int batchSize = output.cols();
  float loss = 0.0f;

  // Compute cross-entropy loss more efficiently
  for (int i = 0; i < batchSize; ++i) {
    // Find the true class index
    int trueClass = -1;
    for (int j = 0; j < output.rows(); ++j) {
      if (target(j, i) > 0.5f) {
        trueClass = j;
        break;
      }
    }

    if (trueClass >= 0) {
      // Add small epsilon to avoid log(0)
      loss -= std::log(std::max(output(trueClass, i), 1e-7f));
    }
  }

  // Add L2 regularization
  float l2Lambda = 0.01f; // L2 regularization strength
  float l2Loss = 0.0f;
  for (const auto &layer : layers_) {
    auto params = layer->getParameters();
    for (const auto &param : params) {
      l2Loss += param.squaredNorm();
    }
  }
  l2Loss *= l2Lambda;

  return (loss / batchSize) + l2Loss;
}

std::vector<Eigen::MatrixXf> ZyraAIModel::getParameters() const {
  std::vector<Eigen::MatrixXf> params;
  for (const auto &layer : layers_) {
    auto layerParams = layer->getParameters();
    params.insert(params.end(), layerParams.begin(), layerParams.end());
  }
  return params;
}

std::vector<Eigen::MatrixXf> ZyraAIModel::getGradients() const {
  std::vector<Eigen::MatrixXf> grads;
  for (const auto &layer : layers_) {
    auto layerGrads = layer->getGradients();
    grads.insert(grads.end(), layerGrads.begin(), layerGrads.end());
  }
  return grads;
}

void ZyraAIModel::setTraining(bool training) {
  for (const auto &layer : layers_) {
    if (auto dropoutLayer = std::dynamic_pointer_cast<DropoutLayer>(layer)) {
      dropoutLayer->setTraining(training);
    }
    if (auto batchNormLayer =
            std::dynamic_pointer_cast<BatchNormLayer>(layer)) {
      batchNormLayer->setTraining(training);
    }
  }
}

int ZyraAIModel::getInputSize() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_.front()->getInputSize();
}

int ZyraAIModel::getOutputSize() const {
  if (layers_.empty()) {
    return 0;
  }
  return layers_.back()->getOutputSize();
}
} // namespace zyraai
