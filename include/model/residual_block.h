#pragma once

#include "batch_norm_layer.h"
#include "layer.h"
#include "optimized_conv_layer.h"
#include "relu_layer.h"
#include <memory>
#include <string>
#include <vector>

namespace zyraai {

class ResidualBlock : public Layer {
public:
  ResidualBlock(const ::std::string &name, int inputChannels, int inputHeight,
                int inputWidth, int outputChannels, int kernelSize,
                int stride = 1, int padding = 1)
      : Layer(name, inputChannels * inputHeight * inputWidth,
              outputChannels * (inputHeight / stride) * (inputWidth / stride)),
        inputChannels_(inputChannels), inputHeight_(inputHeight),
        inputWidth_(inputWidth), outputChannels_(outputChannels),
        outputHeight_((inputHeight - kernelSize + 2 * padding) / stride + 1),
        outputWidth_((inputWidth - kernelSize + 2 * padding) / stride + 1),
        useProjection_(inputChannels != outputChannels || stride != 1) {

    // Main path - first convolution
    std::string conv1Name = name + "_conv1";
    mainPath_.push_back(std::make_shared<OptimizedConvLayer>(
        conv1Name, inputChannels, inputHeight, inputWidth, outputChannels,
        kernelSize, stride, padding));

    std::string bn1Name = name + "_bn1";
    mainPath_.push_back(std::make_shared<BatchNormLayer>(
        bn1Name, outputChannels * outputHeight_ * outputWidth_));

    std::string relu1Name = name + "_relu1";
    mainPath_.push_back(std::make_shared<ReLULayer>(
        relu1Name, outputChannels * outputHeight_ * outputWidth_,
        outputChannels * outputHeight_ * outputWidth_));

    // Main path - second convolution
    std::string conv2Name = name + "_conv2";
    mainPath_.push_back(std::make_shared<OptimizedConvLayer>(
        conv2Name, outputChannels, outputHeight_, outputWidth_, outputChannels,
        kernelSize, 1, padding));

    std::string bn2Name = name + "_bn2";
    mainPath_.push_back(std::make_shared<BatchNormLayer>(
        bn2Name, outputChannels * outputHeight_ * outputWidth_));

    // Skip connection - only if dimensions don't match
    if (useProjection_) {
      std::string skipName = name + "_skip";
      skipConnection_ = std::make_shared<OptimizedConvLayer>(
          skipName, inputChannels, inputHeight, inputWidth, outputChannels, 1,
          stride, 0); // 1x1 conv with stride and no padding

      std::string skipBnName = name + "_skipBn";
      skipBnLayer_ = std::make_shared<BatchNormLayer>(
          skipBnName, outputChannels * outputHeight_ * outputWidth_);
    }

    // Final ReLU after residual connection
    std::string finalReluName = name + "_finalRelu";
    finalRelu_ = std::make_shared<ReLULayer>(
        finalReluName, outputChannels * outputHeight_ * outputWidth_,
        outputChannels * outputHeight_ * outputWidth_);
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    input_ = input; // Store for backward pass

    // Forward through main path
    Eigen::MatrixXf x = input;
    mainOutput_.clear();
    mainOutput_.push_back(x);

    for (const auto &layer : mainPath_) {
      x = layer->forward(x);
      mainOutput_.push_back(x);
    }

    // Forward through skip connection if needed
    Eigen::MatrixXf skipOutput;
    if (useProjection_) {
      skipOutput = skipConnection_->forward(input);
      skipOutput = skipBnLayer_->forward(skipOutput);
    } else {
      skipOutput = input;
    }

    // Add the skip connection to the main path output
    x = x + skipOutput;

    // Apply final ReLU activation
    return finalRelu_->forward(x);
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    // Backward through final ReLU
    Eigen::MatrixXf gradMainPath =
        finalRelu_->backward(gradOutput, learningRate);

    // Split gradients between main path and skip connection
    Eigen::MatrixXf gradSkip = gradMainPath;

    // Backward through skip connection if needed
    Eigen::MatrixXf gradInput;
    if (useProjection_) {
      gradSkip = skipBnLayer_->backward(gradSkip, learningRate);
      gradInput = skipConnection_->backward(gradSkip, learningRate);
    } else {
      gradInput = gradSkip;
    }

    // Backward through main path (in reverse order)
    for (int i = mainPath_.size() - 1; i >= 0; --i) {
      gradMainPath = mainPath_[i]->backward(gradMainPath, learningRate);
    }

    // Combine gradients from main path and skip connection
    if (useProjection_) {
      return gradInput + gradMainPath;
    } else {
      return gradMainPath + gradSkip;
    }
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    ::std::vector<Eigen::MatrixXf> params;

    // Main path parameters
    for (const auto &layer : mainPath_) {
      auto layerParams = layer->getParameters();
      params.insert(params.end(), layerParams.begin(), layerParams.end());
    }

    // Skip connection parameters
    if (useProjection_) {
      auto skipParams = skipConnection_->getParameters();
      params.insert(params.end(), skipParams.begin(), skipParams.end());

      auto skipBnParams = skipBnLayer_->getParameters();
      params.insert(params.end(), skipBnParams.begin(), skipBnParams.end());
    }

    // Final ReLU has no parameters

    return params;
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    ::std::vector<Eigen::MatrixXf> grads;

    // Main path gradients
    for (const auto &layer : mainPath_) {
      auto layerGrads = layer->getGradients();
      grads.insert(grads.end(), layerGrads.begin(), layerGrads.end());
    }

    // Skip connection gradients
    if (useProjection_) {
      auto skipGrads = skipConnection_->getGradients();
      grads.insert(grads.end(), skipGrads.begin(), skipGrads.end());

      auto skipBnGrads = skipBnLayer_->getGradients();
      grads.insert(grads.end(), skipBnGrads.begin(), skipBnGrads.end());
    }

    // Final ReLU has no gradients

    return grads;
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    // Count total parameters
    size_t totalParams = 0;

    // Update parameters in main path
    for (const auto &layer : mainPath_) {
      auto layerParams = layer->getParameters();
      if (index < totalParams + layerParams.size()) {
        layer->updateParameter(index - totalParams, update);
        return;
      }
      totalParams += layerParams.size();
    }

    // Update parameters in skip connection
    if (useProjection_) {
      auto skipParams = skipConnection_->getParameters();
      if (index < totalParams + skipParams.size()) {
        skipConnection_->updateParameter(index - totalParams, update);
        return;
      }
      totalParams += skipParams.size();

      auto skipBnParams = skipBnLayer_->getParameters();
      if (index < totalParams + skipBnParams.size()) {
        skipBnLayer_->updateParameter(index - totalParams, update);
        return;
      }
    }
  }

  void setTraining(bool training) override {
    // Set training mode for all layers
    for (auto &layer : mainPath_) {
      layer->setTraining(training);
    }

    if (useProjection_) {
      skipConnection_->setTraining(training);
      skipBnLayer_->setTraining(training);
    }

    finalRelu_->setTraining(training);
    isTraining_ = training;
  }

private:
  int inputChannels_;
  int inputHeight_;
  int inputWidth_;
  int outputChannels_;
  int outputHeight_;
  int outputWidth_;
  bool useProjection_;
  bool isTraining_ = true;

  // Store intermediate outputs for backward pass
  Eigen::MatrixXf input_;
  ::std::vector<Eigen::MatrixXf> mainOutput_;

  // Main path layers
  ::std::vector<std::shared_ptr<Layer>> mainPath_;

  // Skip connection (if dimensions don't match)
  std::shared_ptr<OptimizedConvLayer> skipConnection_;
  std::shared_ptr<BatchNormLayer> skipBnLayer_;

  // Final activation
  std::shared_ptr<ReLULayer> finalRelu_;
};

} // namespace zyraai