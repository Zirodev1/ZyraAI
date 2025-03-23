#include "model/batch_norm_layer.h"
#include "model/dense_layer.h"
#include "model/dropout_layer.h"
#include "model/relu_layer.h"
#include "model/softmax_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace zyraai;

// Function to read MNIST data from binary files
std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
readMNIST(const std::string &imagesPath, const std::string &labelsPath,
          int numSamples) {
  // Read images
  std::ifstream imageFile(imagesPath, std::ios::binary);
  if (!imageFile.is_open()) {
    throw std::runtime_error("Cannot open images file: " + imagesPath);
  }

  // Skip header
  imageFile.seekg(16);

  // Read image data
  const int imageSize = 28 * 28;
  Eigen::MatrixXf images(imageSize, numSamples);
  std::vector<unsigned char> buffer(imageSize);

  // First pass: compute mean and std
  float sum = 0.0f;
  float sumSq = 0.0f;
  for (int i = 0; i < numSamples; ++i) {
    imageFile.read(reinterpret_cast<char *>(buffer.data()), imageSize);
    for (int j = 0; j < imageSize; ++j) {
      float pixel = static_cast<float>(buffer[j]) / 255.0f;
      sum += pixel;
      sumSq += pixel * pixel;
    }
  }
  float mean = sum / (numSamples * imageSize);
  float std = std::sqrt(sumSq / (numSamples * imageSize) - mean * mean);

  // Second pass: normalize data
  imageFile.seekg(16);
  for (int i = 0; i < numSamples; ++i) {
    imageFile.read(reinterpret_cast<char *>(buffer.data()), imageSize);
    for (int j = 0; j < imageSize; ++j) {
      float pixel = static_cast<float>(buffer[j]) / 255.0f;
      images(j, i) =
          (pixel - mean) / (std + 1e-7f); // Add epsilon for stability
    }
  }

  // Read labels
  std::ifstream labelFile(labelsPath, std::ios::binary);
  if (!labelFile.is_open()) {
    throw std::runtime_error("Cannot open labels file: " + labelsPath);
  }

  // Skip header
  labelFile.seekg(8);

  // Read label data and convert to one-hot encoding
  Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(10, numSamples);
  unsigned char label;
  for (int i = 0; i < numSamples; ++i) {
    labelFile.read(reinterpret_cast<char *>(&label), 1);
    labels(label, i) = 1.0f;
  }

  return {images, labels};
}

int main() {
  std::cout << "Creating MNIST classifier..." << std::endl;

  // Create the model with a more sophisticated architecture
  ZyraAIModel model;

  // Input layer: 784 (28x28) neurons
  // Hidden layer 1: 512 neurons with batch norm, ReLU, and dropout
  // Hidden layer 2: 256 neurons with batch norm, ReLU, and dropout
  // Output layer: 10 neurons (one for each digit)
  model.addLayer(std::make_shared<BatchNormLayer>("bn1", 784));
  model.addLayer(std::make_shared<DenseLayer>("dense1", 784, 512, true));
  model.addLayer(std::make_shared<BatchNormLayer>("bn2", 512));
  model.addLayer(std::make_shared<DropoutLayer>("dropout1", 512, 0.1f));
  model.addLayer(std::make_shared<DenseLayer>("dense2", 512, 256, true));
  model.addLayer(std::make_shared<BatchNormLayer>("bn3", 256));
  model.addLayer(std::make_shared<DropoutLayer>("dropout2", 256, 0.1f));
  model.addLayer(std::make_shared<DenseLayer>("dense3", 256, 10, true));
  model.addLayer(std::make_shared<SoftmaxLayer>("softmax", 10));

  try {
    // Load training data (using 50000 samples)
    const int numTrainSamples = 50000;
    std::cout << "Loading training data..." << std::endl;
    auto [trainImages, trainLabels] =
        readMNIST("data/train-images-idx3-ubyte",
                  "data/train-labels-idx1-ubyte", numTrainSamples);

    // Load test data (using 10000 samples)
    const int numTestSamples = 10000;
    std::cout << "Loading test data..." << std::endl;
    auto [testImages, testLabels] =
        readMNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte",
                  numTestSamples);

    // Training parameters
    const int epochs = 50;
    float learningRate = 0.0005f; // Even lower learning rate for stability
    const int batchSize = 128;    // Medium batch size for balance
    const int numBatches = numTrainSamples / batchSize;

    std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
    std::cout << "Batch size: " << batchSize
              << ", Learning rate: " << learningRate << std::endl;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
      float epochLoss = 0.0f;
      auto epochStartTime = std::chrono::high_resolution_clock::now();

      // Train on mini-batches
      for (int batch = 0; batch < numBatches; ++batch) {
        int startIdx = batch * batchSize;
        int endIdx = startIdx + batchSize;

        // Extract batch data
        Eigen::MatrixXf batchImages =
            trainImages.block(0, startIdx, 784, batchSize);
        Eigen::MatrixXf batchLabels =
            trainLabels.block(0, startIdx, 10, batchSize);

        // Train on batch
        float batchLoss = model.train(batchImages, batchLabels, learningRate);
        epochLoss += batchLoss;
      }

      epochLoss /= numBatches;

      // Learning rate decay (more gradual)
      if ((epoch + 1) % 10 == 0) { // Less frequent decay
        learningRate *= 0.95f;     // Slower decay
      }

      // Calculate epoch duration
      auto epochEndTime = std::chrono::high_resolution_clock::now();
      auto epochDuration =
          std::chrono::duration_cast<std::chrono::milliseconds>(epochEndTime -
                                                                epochStartTime);

      // Print progress every epoch
      std::cout << "Epoch " << epoch + 1 << "/" << epochs
                << ", Loss: " << epochLoss
                << ", Time: " << (epochDuration.count() / 1000.0f) << "s"
                << std::endl;

      // Evaluate on test set every 5 epochs
      if ((epoch + 1) % 5 == 0) {
        // Set dropout layers to evaluation mode
        model.setTraining(false);
        Eigen::MatrixXf predictions = model.forward(testImages);

        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < numTestSamples; ++i) {
          Eigen::MatrixXf::Index maxRow, maxCol;
          predictions.col(i).maxCoeff(&maxRow, &maxCol);
          Eigen::MatrixXf::Index trueRow, trueCol;
          testLabels.col(i).maxCoeff(&trueRow, &trueCol);
          if (maxRow == trueRow)
            correct++;
        }

        float accuracy = static_cast<float>(correct) / numTestSamples;
        std::cout << "Test Accuracy: " << (accuracy * 100) << "%" << std::endl;

        // Set dropout layers back to training mode
        model.setTraining(true);
      }
    }

    // Calculate total training duration
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration =
        std::chrono::duration_cast<std::chrono::minutes>(endTime - startTime);

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Total training time: " << totalDuration.count() << " minutes"
              << std::endl;

    // Final evaluation
    model.setTraining(false);
    Eigen::MatrixXf predictions = model.forward(testImages);
    int correct = 0;
    for (int i = 0; i < numTestSamples; ++i) {
      Eigen::MatrixXf::Index maxRow, maxCol;
      predictions.col(i).maxCoeff(&maxRow, &maxCol);
      Eigen::MatrixXf::Index trueRow, trueCol;
      testLabels.col(i).maxCoeff(&trueRow, &trueCol);
      if (maxRow == trueRow)
        correct++;
    }

    float finalAccuracy = static_cast<float>(correct) / numTestSamples;
    std::cout << "Final Test Accuracy: " << (finalAccuracy * 100) << "%"
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}