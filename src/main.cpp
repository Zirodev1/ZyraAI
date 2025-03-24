// src/main.cpp

#include "model/relu_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

using namespace ::std;

int main() {
  cout << "ZyraAI Starting..." << ::std::endl;

  // Create a simple neural network
  zyraai::ZyraAIModel model;
  model.addLayer(::std::make_shared<zyraai::ReLULayer>("hidden1", 2, 4));
  model.addLayer(::std::make_shared<zyraai::ReLULayer>("output", 4, 1));

  // Create XOR training data
  Eigen::MatrixXf input(2, 4);  // 4 samples, 2 features each
  Eigen::MatrixXf target(1, 4); // 4 samples, 1 output each

  // XOR truth table
  input << 0, 0, 1, 1,  // First feature
      0, 1, 0, 1;       // Second feature
  target << 0, 1, 1, 0; // XOR output

  cout << "Training data:" << ::std::endl;
  cout << "Input:\n" << input << ::std::endl;
  cout << "Target:\n" << target << ::std::endl;

  // Training loop
  const int epochs = 10000;
  const float learningRate = 0.01f;

  cout << "\nTraining for " << epochs << " epochs..." << ::std::endl;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    model.train(input, target, learningRate);

    if (epoch % 1000 == 0) {
      cout << "Epoch " << epoch << ::std::endl;
    }
  }

  // Test the model
  cout << "\nTesting the model:" << ::std::endl;
  Eigen::MatrixXf output = model.forward(input);
  cout << "Predicted output:\n" << output << ::std::endl;
  cout << "Expected output:\n" << target << ::std::endl;

  // Calculate accuracy
  float accuracy =
      (output.array().round() == target.array()).cast<float>().mean();
  cout << "\nAccuracy: " << (accuracy * 100) << "%" << ::std::endl;

  return 0;
}
