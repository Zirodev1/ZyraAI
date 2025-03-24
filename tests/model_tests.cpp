// tests/model_tests.cpp

#include "model/relu_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

using namespace ::std;

int main() {
  cout << "Running model tests..." << endl;

  // Create a simple model
  zyraai::ZyraAIModel model;
  model.addLayer(::std::make_shared<zyraai::ReLULayer>("hidden1", 2, 2));
  model.addLayer(::std::make_shared<zyraai::ReLULayer>("output", 2, 2));

  // Create simple test data - Matrix is (features x batch_size)
  Eigen::MatrixXf input(2, 2);
  input << 1.0f, 0.5f, 1.0f, 0.8f;

  Eigen::MatrixXf target(2, 2);
  target << 1.0f, 0.7f, 0.5f, 0.3f;

  // Test forward pass
  Eigen::MatrixXf output = model.forward(input);
  cout << "Forward pass output shape: " << output.rows() << "x" << output.cols()
       << endl;

  if (output.rows() != 2 || output.cols() != 2) {
    cerr << "Error: Incorrect output shape" << endl;
    return 1;
  }

  // Test training
  model.train(input, target, 0.01f);
  cout << "Training successful" << endl;

  cout << "All tests passed!" << endl;
  return 0;
}
