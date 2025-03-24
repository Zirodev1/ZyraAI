// include/model/adam_optimizer.h
#ifndef ZYRAAI_ADAM_OPTIMIZER_H
#define ZYRAAI_ADAM_OPTIMIZER_H

#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <vector>

namespace zyraai {

class AdamOptimizer {
public:
  AdamOptimizer(ZyraAIModel &model, float learningRate = 0.001f,
                float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
      : model_(model), learningRate_(learningRate), beta1_(beta1),
        beta2_(beta2), epsilon_(epsilon) {

    // Initialize momentum and velocity vectors
    const auto &layers = model_.getLayers();
    for (const auto &layer : layers) {
      auto params = layer->getParameters();
      for (size_t i = 0; i < params.size(); ++i) {
        m_.push_back(Eigen::MatrixXf::Zero(params[i].rows(), params[i].cols()));
        v_.push_back(Eigen::MatrixXf::Zero(params[i].rows(), params[i].cols()));
      }
    }
  }

  void step() {
    t_++;
    float alpha_t = learningRate_ * std::sqrt(1.0f - std::pow(beta2_, t_)) /
                    (1.0f - std::pow(beta1_, t_));

    const auto &layers = model_.getLayers();
    const auto &gradients = model_.getGradients();

    size_t paramIdx = 0;
    for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
      auto layerGrads = layers[layerIdx]->getGradients();
      auto layerParams = layers[layerIdx]->getParameters();

      for (size_t i = 0; i < layerGrads.size(); ++i) {
        // Update biased first moment estimate
        m_[paramIdx] = beta1_ * m_[paramIdx] + (1.0f - beta1_) * layerGrads[i];

        // Update biased second raw moment estimate
        v_[paramIdx] =
            beta2_ * v_[paramIdx] +
            (1.0f - beta2_) * layerGrads[i].array().square().matrix();

        // Apply update
        layers[layerIdx]->updateParameter(
            i, (alpha_t * m_[paramIdx].array() /
                (v_[paramIdx].array().sqrt() + epsilon_))
                   .matrix());

        paramIdx++;
      }
    }
  }

  void setLearningRate(float lr) { learningRate_ = lr; }
  float getLearningRate() const { return learningRate_; }

private:
  ZyraAIModel &model_;
  float learningRate_;
  float beta1_;
  float beta2_;
  float epsilon_;
  int t_ = 0;
  std::vector<Eigen::MatrixXf> m_; // First moment vectors
  std::vector<Eigen::MatrixXf> v_; // Second moment vectors
};

} // namespace zyraai

#endif // ZYRAAI_ADAM_OPTIMIZER_H