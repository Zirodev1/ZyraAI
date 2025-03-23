// include/model/zyraAI_model.h

#ifndef ZYRAAI_MODEL_H
#define ZYRAAI_MODEL_H

#include "model/layer.h"
#include "model/relu_layer.h"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace zyraai {

class ZyraAIModel {
public:
  ZyraAIModel();
  ~ZyraAIModel();

  // Model operations
  void addLayer(std::shared_ptr<Layer> layer);
  Eigen::MatrixXf forward(const Eigen::MatrixXf &input);
  void backward(const Eigen::MatrixXf &gradOutput, float learningRate);

  // Training operations
  float train(const Eigen::MatrixXf &input, const Eigen::MatrixXf &target,
              float learningRate);
  float computeLoss(const Eigen::MatrixXf &output,
                    const Eigen::MatrixXf &target);

  // Model information
  std::vector<std::shared_ptr<Layer>> getLayers() const { return layers_; }
  int getInputSize() const;
  int getOutputSize() const;

  // Get model parameters
  std::vector<Eigen::MatrixXf> getParameters() const;

  // Get model gradients
  std::vector<Eigen::MatrixXf> getGradients() const;

  // Set training mode
  void setTraining(bool training);

private:
  std::vector<std::shared_ptr<Layer>> layers_;
  std::vector<Eigen::MatrixXf> activations_; // Store intermediate activations
};

} // namespace zyraai

#endif // ZYRAAI_MODEL_H
