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
  model.addLayer(std::make_shared<zyraai::ReLULayer>("output", 2, 2));

  // Create simple test data - Matrix is (features x batch_size)
  Eigen::MatrixXf input(2, 2);
  input << 1.0f, 0.5f, 1.0f, 0.8f;

  Eigen::MatrixXf target(2, 2);
  target << 1.0f, 0.7f, 0.5f, 0.3f;

  // Test forward pass
  Eigen::MatrixXf output = model.forward(input);
  std::cout << "Forward pass output shape: " << output.rows() << "x"
            << output.cols() << std::endl;

  if (output.rows() != 2 || output.cols() != 2) {
    std::cerr << "Error: Incorrect output shape" << std::endl;
    return 1;
  }

  // Test training
  model.train(input, target, 0.01f);
  std::cout << "Training successful" << std::endl;

  std::cout << "All tests passed!" << std::endl;
  return 0;
}
