#include "model/adam_optimizer.h"
#include "model/batch_norm_layer.h"
#include "model/dense_layer.h"
#include "model/dropout_layer.h"
#include "model/relu_layer.h"
#include "model/softmax_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace zyraai;

// Function to augment MNIST images
Eigen::MatrixXf augmentImage(const Eigen::MatrixXf &image) {
  // Reshape to 28x28
  Eigen::MatrixXf img = Eigen::Map<const Eigen::MatrixXf>(image.data(), 28, 28);

  // Random shift (up to 2 pixels in each direction)
  int shiftX = rand() % 5 - 2;
  int shiftY = rand() % 5 - 2;

  // Create output image
  Eigen::MatrixXf shifted = Eigen::MatrixXf::Zero(28, 28);

  // Apply shift
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      int newY = y + shiftY;
      int newX = x + shiftX;
      if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
        shifted(newY, newX) = img(y, x);
      }
    }
  }

  // Reshape back to vector
  return Eigen::Map<Eigen::MatrixXf>(shifted.data(), 784, 1);
}

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
  // Hidden layer 1: 512 neurons with Dense->BatchNorm->ReLU->Dropout
  // Hidden layer 2: 256 neurons with Dense->BatchNorm->ReLU->Dropout
  // Output layer: 10 neurons (one for each digit)
  model.addLayer(std::make_shared<DenseLayer>("dense1", 784, 512, true));
  model.addLayer(std::make_shared<BatchNormLayer>("bn1", 512));
  model.addLayer(std::make_shared<ReLULayer>("relu1", 512, 512));
  model.addLayer(std::make_shared<DropoutLayer>("dropout1", 512, 0.2f));

  model.addLayer(std::make_shared<DenseLayer>("dense2", 512, 256, true));
  model.addLayer(std::make_shared<BatchNormLayer>("bn2", 256));
  model.addLayer(std::make_shared<ReLULayer>("relu2", 256, 256));
  model.addLayer(std::make_shared<DropoutLayer>("dropout2", 256, 0.2f));

  model.addLayer(std::make_shared<DenseLayer>("dense3", 256, 10, true));
  model.addLayer(std::make_shared<SoftmaxLayer>("softmax", 10));

  try {
    // Load training data (using 50000 samples)
    const int numTrainSamples = 50000;
    std::cout << "Loading training data..." << std::endl;
    auto [trainImages, trainLabels] =
        readMNIST("I:/ZyraAI/data/train-images.idx3-ubyte",
                  "I:/ZyraAI/data/train-labels.idx1-ubyte", numTrainSamples);

    // Load test data (using 10000 samples)
    const int numTestSamples = 10000;
    std::cout << "Loading test data..." << std::endl;
    auto [testImages, testLabels] =
        readMNIST("I:/ZyraAI/data/t10k-images.idx3-ubyte", 
                  "I:/ZyraAI/data/t10k-labels.idx1-ubyte",
                  numTestSamples);

    // Create Adam optimizer
    zyraai::AdamOptimizer optimizer(model, 0.001f); // Higher learning rate

    // Training parameters
    const int epochs = 50; // Increased from 30 to 50 epochs
    const int batchSize = 128;
    const int numBatches = numTrainSamples / batchSize;

    // Learning rate schedule - extended for more epochs
    std::vector<float> lrSchedule = {0.001f,  0.001f,  0.0008f, 0.0008f,
                                     0.0005f, 0.0005f, 0.0003f, 0.0003f,
                                     0.0001f, 0.0001f};

    // For early stopping
    float bestAccuracy = 0.0f;
    int patienceCount = 0;
    const int patience = 8; // Increased patience for longer training

    std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
    std::cout << "Batch size: " << batchSize
              << ", Initial learning rate: " << optimizer.getLearningRate()
              << std::endl;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
      // Set learning rate for this epoch
      if (epoch < lrSchedule.size()) {
        optimizer.setLearningRate(lrSchedule[epoch]);
      }

      // Shuffle training data indices
      std::vector<int> indices(numTrainSamples);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(),
                   std::mt19937(std::random_device()()));

      float epochLoss = 0.0f;
      auto epochStartTime = std::chrono::high_resolution_clock::now();

      // Train on mini-batches
      for (int batch = 0; batch < numBatches; ++batch) {
        // Extract batch data using shuffled indices
        Eigen::MatrixXf batchImages(784, batchSize);
        Eigen::MatrixXf batchLabels(10, batchSize);

        for (int i = 0; i < batchSize; ++i) {
          int idx = indices[batch * batchSize + i];
          // Apply augmentation with 50% probability during training
          if (rand() % 2 == 0) {
            batchImages.col(i) = augmentImage(trainImages.col(idx));
          } else {
            batchImages.col(i) = trainImages.col(idx);
          }
          batchLabels.col(i) = trainLabels.col(idx);
        }

        // Forward pass
        Eigen::MatrixXf output = model.forward(batchImages);

        // Compute loss
        float batchLoss = model.computeLoss(output, batchLabels);
        epochLoss += batchLoss;

        // Compute gradients
        Eigen::MatrixXf gradOutput =
            (output - batchLabels) / static_cast<float>(batchSize);

        // Backward pass
        model.backward(gradOutput, optimizer.getLearningRate());

        // Update parameters with Adam optimizer
        optimizer.step();
      }

      epochLoss /= numBatches;

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

      // Evaluate on test set every 2 epochs
      if ((epoch + 1) % 2 == 0) {
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

        // Early stopping check
        if (accuracy > bestAccuracy) {
          bestAccuracy = accuracy;
          patienceCount = 0;
          std::cout << "New best accuracy!" << std::endl;
        } else {
          patienceCount++;
          if (patienceCount >= patience) {
            std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
            break;
          }
        }

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