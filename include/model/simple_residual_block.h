#pragma once

#include "batch_norm_layer.h"
#include "identity_layer.h"
#include "layer.h"
#include "optimized_conv_layer.h"
#include "relu_layer.h"
#include <memory>
#include <string>
#include <vector>

namespace zyraai {

// A simplified residual block with better initialization
class SimpleResidualBlock : public Layer {
public:
  SimpleResidualBlock(const ::std::string &name, int channels, int height,
                      int width, int kernelSize = 3, int padding = 1)
      : Layer(name, channels * height * width, channels * height * width),
        channels_(channels), height_(height), width_(width),
        kernelSize_(kernelSize), padding_(padding) {

    int inputSize = channels * height * width;

    // Main path - first convolution + bn + relu
    std::string conv1Name = name + "_conv1";
    conv1_ = std::make_shared<OptimizedConvLayer>(
        conv1Name, channels, height, width, channels, kernelSize, 1, padding);

    std::string bn1Name = name + "_bn1";
    bn1_ = std::make_shared<BatchNormLayer>(bn1Name, channels * height * width);

    std::string relu1Name = name + "_relu1";
    relu1_ = std::make_shared<ReLULayer>(relu1Name, channels * height * width,
                                         channels * height * width);

    // Main path - second convolution + bn
    std::string conv2Name = name + "_conv2";
    conv2_ = std::make_shared<OptimizedConvLayer>(
        conv2Name, channels, height, width, channels, kernelSize, 1, padding);

    std::string bn2Name = name + "_bn2";
    bn2_ = std::make_shared<BatchNormLayer>(bn2Name, channels * height * width);

    // Skip connection - direct identity
    std::string skipName = name + "_skip";
    skip_ =
        std::make_shared<IdentityLayer>(skipName, channels * height * width);

    // Final ReLU after residual connection
    std::string finalReluName = name + "_finalRelu";
    finalRelu_ = std::make_shared<ReLULayer>(
        finalReluName, channels * height * width, channels * height * width);
  }

  Eigen::MatrixXf forward(const Eigen::MatrixXf &input) override {
    // Forward through main path
    Eigen::MatrixXf mainPath = conv1_->forward(input);
    mainPath = bn1_->forward(mainPath);
    mainPath = relu1_->forward(mainPath);

    mainPath = conv2_->forward(mainPath);
    mainPath = bn2_->forward(mainPath);

    // Forward through skip connection
    Eigen::MatrixXf skipPath = skip_->forward(input);

    // Add the skip connection to the main path output
    Eigen::MatrixXf output = mainPath + skipPath;

    // Apply final ReLU activation
    return finalRelu_->forward(output);
  }

  Eigen::MatrixXf backward(const Eigen::MatrixXf &gradOutput,
                           float learningRate) override {
    // Backward through final ReLU
    Eigen::MatrixXf gradFinal = finalRelu_->backward(gradOutput, learningRate);

    // Split gradients between main path and skip connection
    Eigen::MatrixXf gradSkip = gradFinal;
    Eigen::MatrixXf gradMain = gradFinal;

    // Backward through skip connection
    gradSkip = skip_->backward(gradSkip, learningRate);

    // Backward through main path (in reverse order)
    gradMain = bn2_->backward(gradMain, learningRate);
    gradMain = conv2_->backward(gradMain, learningRate);

    gradMain = relu1_->backward(gradMain, learningRate);
    gradMain = bn1_->backward(gradMain, learningRate);
    gradMain = conv1_->backward(gradMain, learningRate);

    // Combine gradients from main path and skip connection
    return gradMain + gradSkip;
  }

  ::std::vector<Eigen::MatrixXf> getParameters() const override {
    ::std::vector<Eigen::MatrixXf> params;

    // Gather parameters from all layers
    auto conv1Params = conv1_->getParameters();
    params.insert(params.end(), conv1Params.begin(), conv1Params.end());

    auto bn1Params = bn1_->getParameters();
    params.insert(params.end(), bn1Params.begin(), bn1Params.end());

    auto conv2Params = conv2_->getParameters();
    params.insert(params.end(), conv2Params.begin(), conv2Params.end());

    auto bn2Params = bn2_->getParameters();
    params.insert(params.end(), bn2Params.begin(), bn2Params.end());

    return params;
  }

  ::std::vector<Eigen::MatrixXf> getGradients() const override {
    ::std::vector<Eigen::MatrixXf> grads;

    // Gather gradients from all layers
    auto conv1Grads = conv1_->getGradients();
    grads.insert(grads.end(), conv1Grads.begin(), conv1Grads.end());

    auto bn1Grads = bn1_->getGradients();
    grads.insert(grads.end(), bn1Grads.begin(), bn1Grads.end());

    auto conv2Grads = conv2_->getGradients();
    grads.insert(grads.end(), conv2Grads.begin(), conv2Grads.end());

    auto bn2Grads = bn2_->getGradients();
    grads.insert(grads.end(), bn2Grads.begin(), bn2Grads.end());

    return grads;
  }

  void updateParameter(size_t index, const Eigen::MatrixXf &update) override {
    size_t paramCount = 0;

    // Update convolution parameters
    auto conv1Params = conv1_->getParameters();
    if (index < paramCount + conv1Params.size()) {
      conv1_->updateParameter(index - paramCount, update);
      return;
    }
    paramCount += conv1Params.size();

    // Update batch norm parameters
    auto bn1Params = bn1_->getParameters();
    if (index < paramCount + bn1Params.size()) {
      bn1_->updateParameter(index - paramCount, update);
      return;
    }
    paramCount += bn1Params.size();

    // Update second convolution parameters
    auto conv2Params = conv2_->getParameters();
    if (index < paramCount + conv2Params.size()) {
      conv2_->updateParameter(index - paramCount, update);
      return;
    }
    paramCount += conv2Params.size();

    // Update second batch norm parameters
    auto bn2Params = bn2_->getParameters();
    if (index < paramCount + bn2Params.size()) {
      bn2_->updateParameter(index - paramCount, update);
      return;
    }
  }

  void setTraining(bool training) override {
    conv1_->setTraining(training);
    bn1_->setTraining(training);
    relu1_->setTraining(training);
    conv2_->setTraining(training);
    bn2_->setTraining(training);
    skip_->setTraining(training);
    finalRelu_->setTraining(training);
    isTraining_ = training;
  }

private:
  int channels_;
  int height_;
  int width_;
  int kernelSize_;
  int padding_;
  bool isTraining_ = true;

  // Component layers
  std::shared_ptr<OptimizedConvLayer> conv1_;
  std::shared_ptr<BatchNormLayer> bn1_;
  std::shared_ptr<ReLULayer> relu1_;

  std::shared_ptr<OptimizedConvLayer> conv2_;
  std::shared_ptr<BatchNormLayer> bn2_;

  std::shared_ptr<IdentityLayer> skip_;
  std::shared_ptr<ReLULayer> finalRelu_;
};

} // namespace zyraai