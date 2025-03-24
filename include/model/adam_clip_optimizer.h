#pragma once

#include "zyraAI_model.h"
#include <Eigen/Dense>
#include <vector>

namespace zyraai {

class AdamClipOptimizer {
public:
  AdamClipOptimizer(ZyraAIModel &model, float learningRate = 0.001f,
                    float beta1 = 0.9f, float beta2 = 0.999f,
                    float epsilon = 1e-8f, float clipNorm = 1.0f,
                    float weightDecay = 0.0001f)
      : model_(model), learningRate_(learningRate), beta1_(beta1),
        beta2_(beta2), epsilon_(epsilon), clipNorm_(clipNorm),
        weightDecay_(weightDecay) {

    // Initialize moment vectors for each parameter
    const auto &layers = model_.getLayers();
    for (const auto &layer : layers) {
      const auto &params = layer->getParameters();

      // For each parameter in the layer
      for (size_t i = 0; i < params.size(); ++i) {
        // Initialize first and second moments to zero
        momentVectors_.push_back(
            {Eigen::MatrixXf::Zero(params[i].rows(), params[i].cols()),
             Eigen::MatrixXf::Zero(params[i].rows(), params[i].cols())});
      }
    }
  }

  void step() {
    // Increment time step
    t_++;

    // Get gradients from all layers
    ::std::vector<Eigen::MatrixXf> allGradients;
    const auto &layers = model_.getLayers();
    for (const auto &layer : layers) {
      const auto &grads = layer->getGradients();
      allGradients.insert(allGradients.end(), grads.begin(), grads.end());
    }

    // Apply gradient clipping - find global norm
    float globalSqNorm = 0.0f;
    for (const auto &grad : allGradients) {
      globalSqNorm += grad.squaredNorm();
    }
    float globalNorm = std::sqrt(globalSqNorm);

    // Clipping factor
    float clipFactor = 1.0f;
    if (globalNorm > clipNorm_ && globalNorm > 0) {
      clipFactor = clipNorm_ / globalNorm;
    }

    // Compute bias corrections
    float correctedBeta1 = 1.0f - std::pow(beta1_, t_);
    float correctedBeta2 = 1.0f - std::pow(beta2_, t_);
    float correctedLR =
        learningRate_ * std::sqrt(correctedBeta2) / correctedBeta1;

    // Update all parameters
    size_t paramIdx = 0;
    size_t layerParamIdx = 0;

    for (size_t l = 0; l < layers.size(); ++l) {
      const auto &params = layers[l]->getParameters();
      const auto &grads = layers[l]->getGradients();

      // For each parameter in the layer
      for (size_t i = 0; i < params.size(); ++i) {
        // Get clipped gradient
        Eigen::MatrixXf clippedGrad = grads[i] * clipFactor;

        // Add weight decay (L2 regularization)
        clippedGrad += weightDecay_ * params[i];

        // Update biased first and second moment estimates
        momentVectors_[paramIdx].first =
            beta1_ * momentVectors_[paramIdx].first +
            (1.0f - beta1_) * clippedGrad;

        momentVectors_[paramIdx].second =
            beta2_ * momentVectors_[paramIdx].second +
            (1.0f - beta2_) * clippedGrad.cwiseProduct(clippedGrad);

        // Compute bias-corrected update
        Eigen::MatrixXf update =
            correctedLR *
            momentVectors_[paramIdx].first.cwiseQuotient(
                (momentVectors_[paramIdx].second.cwiseSqrt().array() + epsilon_)
                    .matrix());

        // Apply update to parameter
        layers[l]->updateParameter(i, update);

        paramIdx++;
      }
    }
  }

  void setLearningRate(float learningRate) { learningRate_ = learningRate; }

  float getLearningRate() const { return learningRate_; }

  void setClipNorm(float clipNorm) { clipNorm_ = clipNorm; }

  float getClipNorm() const { return clipNorm_; }

  void setWeightDecay(float weightDecay) { weightDecay_ = weightDecay; }

  float getWeightDecay() const { return weightDecay_; }

private:
  ZyraAIModel &model_;
  float learningRate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  float clipNorm_;
  float weightDecay_;
  int t_ = 0; // Time step

  // First and second moment vectors for each parameter
  ::std::vector<::std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> momentVectors_;
};

} // namespace zyraai