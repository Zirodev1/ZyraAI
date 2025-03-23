// tests/model_tests.cpp

#include "model/relu_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

int main() {
  std::cout << "Running model tests..." << std::endl;

  // Create a simple model
  zyraai::ZyraAIModel model;
  model.addLayer(std::make_shared<zyraai::ReLULayer>("hidden1", 2, 2));
  model.addLayer(std::make_shared<zyraai::ReLULayer>("output", 2, 1));

  // Create simple test data
  Eigen::MatrixXf input(2, 1);
  input << 1.0f, 1.0f;

  Eigen::MatrixXf target(1, 1);
  target << 1.0f;

  // Test forward pass
  Eigen::MatrixXf output = model.forward(input);
  std::cout << "Forward pass output shape: " << output.rows() << "x"
            << output.cols() << std::endl;

  if (output.rows() != 1 || output.cols() != 1) {
    std::cerr << "Error: Incorrect output shape" << std::endl;
    return 1;
  }

  // Test training
  model.train(input, target, 0.01f);
  std::cout << "Training successful" << std::endl;

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
