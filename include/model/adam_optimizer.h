/**
 * @file adam_optimizer.h
 * @brief Adam optimizer implementation
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_ADAM_OPTIMIZER_H
#define ZYRAAI_ADAM_OPTIMIZER_H

#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace zyraai {

/**
 * @class AdamOptimizer
 * @brief Implements the Adam optimization algorithm
 * 
 * Adam (Adaptive Moment Estimation) is an optimization algorithm that computes
 * adaptive learning rates for each parameter. It combines the advantages of
 * AdaGrad and RMSProp algorithms.
 * 
 * Reference: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic
 * optimization. arXiv preprint arXiv:1412.6980.
 */
class AdamOptimizer {
public:
  /**
   * @brief Construct a new Adam Optimizer
   * @param model Reference to the model to optimize
   * @param learningRate Initial learning rate (default: 0.001)
   * @param beta1 Exponential decay rate for first moment estimates (default: 0.9)
   * @param beta2 Exponential decay rate for second moment estimates (default: 0.999)
   * @param epsilon Small constant for numerical stability (default: 1e-8)
   * @throws std::invalid_argument If learning rate or beta values are invalid
   */
  AdamOptimizer(ZyraAIModel &model, float learningRate = 0.001f,
                float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
      : model_(model), learningRate_(learningRate), beta1_(beta1),
        beta2_(beta2), epsilon_(epsilon) {

    // Validate parameters
    if (learningRate <= 0.0f) {
      throw std::invalid_argument("AdamOptimizer: learning rate must be positive");
    }
    
    if (beta1 < 0.0f || beta1 >= 1.0f) {
      throw std::invalid_argument("AdamOptimizer: beta1 must be in range [0, 1)");
    }
    
    if (beta2 < 0.0f || beta2 >= 1.0f) {
      throw std::invalid_argument("AdamOptimizer: beta2 must be in range [0, 1)");
    }
    
    if (epsilon <= 0.0f) {
      throw std::invalid_argument("AdamOptimizer: epsilon must be positive");
    }

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

  /**
   * @brief Perform one optimization step
   * 
   * Updates all model parameters using the Adam algorithm:
   * 1. Update biased first and second moment estimates
   * 2. Compute bias-corrected estimates
   * 3. Update parameters
   */
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
        if (paramIdx >= m_.size() || paramIdx >= v_.size()) {
          throw std::runtime_error("AdamOptimizer: parameter index out of bounds");
        }
        
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

  /**
   * @brief Set the learning rate
   * @param lr New learning rate
   * @throws std::invalid_argument If learning rate is not positive
   */
  void setLearningRate(float lr) { 
    if (lr <= 0.0f) {
      throw std::invalid_argument("AdamOptimizer: learning rate must be positive");
    }
    learningRate_ = lr; 
  }
  
  /**
   * @brief Get the current learning rate
   * @return Current learning rate
   */
  float getLearningRate() const { return learningRate_; }

private:
  ZyraAIModel &model_;     ///< Reference to the model being optimized
  float learningRate_;     ///< Learning rate
  float beta1_;            ///< Exponential decay rate for first moment
  float beta2_;            ///< Exponential decay rate for second moment
  float epsilon_;          ///< Small constant for numerical stability
  int t_ = 0;              ///< Time step counter
  std::vector<Eigen::MatrixXf> m_; ///< First moment vectors
  std::vector<Eigen::MatrixXf> v_; ///< Second moment vectors
};

} // namespace zyraai

#endif // ZYRAAI_ADAM_OPTIMIZER_H