#include "model/adam_optimizer.h"
#include "model/dense_layer.h"
#include "model/relu_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <ctime>
#include <iostream>
#include <memory>
#include <random>

using namespace zyraai;
using namespace ::std;

int main() {
  cout << "Creating a simple neural network for XOR problem..." << ::std::endl;

  // Use a fixed seed for reproducibility
  srand(42);

  // Create multiple models and train them with different initializations
  const int numModels = 5;
  vector<ZyraAIModel> models(numModels);
  vector<AdamOptimizer> optimizers;

  // Set up the models
  for (int i = 0; i < numModels; i++) {
    // Relatively large network for this simple problem
    models[i].addLayer(::std::make_shared<DenseLayer>("dense1", 2, 16, true));
    models[i].addLayer(::std::make_shared<ReLULayer>("relu1", 16, 16));
    models[i].addLayer(::std::make_shared<DenseLayer>("dense2", 16, 8, true));
    models[i].addLayer(::std::make_shared<ReLULayer>("relu2", 8, 8));
    models[i].addLayer(::std::make_shared<DenseLayer>("dense3", 8, 1, true));

    // Create an optimizer with a slightly different learning rate for each
    // model
    optimizers.emplace_back(models[i], 0.01f + 0.002f * i);
  }

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
  const int epochs = 20000;

  cout << "\nTraining " << numModels << " models for " << epochs
       << " epochs each..." << ::std::endl;

  // Keep track of best model
  int bestModelIdx = 0;
  float bestAccuracy = 0.0f;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (int modelIdx = 0; modelIdx < numModels; modelIdx++) {
      // Forward pass
      Eigen::MatrixXf output = models[modelIdx].forward(input);

      // Compute gradients
      Eigen::MatrixXf gradOutput = output - target;

      // Backward pass
      models[modelIdx].backward(gradOutput, 0.0001f); // Small positive learning rate for backward pass

      // Update parameters
      optimizers[modelIdx].step();
    }

    if (epoch % 1000 == 0) {
      cout << "Epoch " << epoch << ::std::endl;

      // Check which model is performing best
      for (int modelIdx = 0; modelIdx < numModels; modelIdx++) {
        Eigen::MatrixXf output = models[modelIdx].forward(input);
        float accuracy =
            (output.array().round() == target.array()).cast<float>().mean();

        if (accuracy > bestAccuracy) {
          bestAccuracy = accuracy;
          bestModelIdx = modelIdx;

          if (bestAccuracy == 1.0f) {
            cout << "Found perfect solution with model " << bestModelIdx
                 << " at epoch " << epoch << ::std::endl;
          }
        }
      }
    }
  }

  // Test the best model
  cout << "\nTesting the best model (model " << bestModelIdx
       << "):" << ::std::endl;
  Eigen::MatrixXf output = models[bestModelIdx].forward(input);
  cout << "Predicted output (raw):\n" << output << ::std::endl;
  cout << "Predicted output (rounded):\n"
       << output.array().round() << ::std::endl;
  cout << "Expected output:\n" << target << ::std::endl;

  // Calculate accuracy
  float accuracy =
      (output.array().round() == target.array()).cast<float>().mean();
  cout << "\nAccuracy: " << (accuracy * 100) << "%" << ::std::endl;

  return 0;
}