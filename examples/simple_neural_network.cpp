#include "model/relu_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

using namespace zyraai;

int main() {
  std::cout << "Creating a simple neural network for XOR problem..."
            << std::endl;

  // Create the model
  ZyraAIModel model;

  // Add layers (2 input neurons, 4 hidden neurons, 1 output neuron)
  model.addLayer(std::make_shared<ReLULayer>("hidden1", 2, 4));
  model.addLayer(std::make_shared<ReLULayer>("output", 4, 1));

  // Create XOR training data
  Eigen::MatrixXf input(2, 4);  // 4 samples, 2 features each
  Eigen::MatrixXf target(1, 4); // 4 samples, 1 output each

  // XOR truth table
  input << 0, 0, 1, 1,  // First feature
      0, 1, 0, 1;       // Second feature
  target << 0, 1, 1, 0; // XOR output

  std::cout << "Training data:" << std::endl;
  std::cout << "Input:\n" << input << std::endl;
  std::cout << "Target:\n" << target << std::endl;

  // Training loop
  const int epochs = 10000;
  const float learningRate = 0.01f;

  std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    model.train(input, target, learningRate);

    if (epoch % 1000 == 0) {
      std::cout << "Epoch " << epoch << std::endl;
    }
  }

  // Test the model
  std::cout << "\nTesting the model:" << std::endl;
  Eigen::MatrixXf output = model.forward(input);
  std::cout << "Predicted output:\n" << output << std::endl;
  std::cout << "Expected output:\n" << target << std::endl;

  // Calculate accuracy
  float accuracy =
      (output.array().round() == target.array()).cast<float>().mean();
  std::cout << "\nAccuracy: " << (accuracy * 100) << "%" << std::endl;

  return 0;
}