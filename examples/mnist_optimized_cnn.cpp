#include "model/adam_optimizer.h"
#include "model/batch_norm_layer.h"
#include "model/dense_layer.h"
#include "model/dropout_layer.h"
#include "model/flatten_layer.h"
#include "model/lr_scheduler.h"
#include "model/max_pooling_layer.h"
#include "model/model_serializer.h"
#include "model/optimized_conv_layer.h"
#include "model/relu_layer.h"
#include "model/softmax_layer.h"
#include "model/zyraAI_model.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip> // For timing output formatting
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace zyraai;

// Function to augment MNIST images with enhanced techniques
Eigen::MatrixXf augmentImage(const Eigen::MatrixXf &image) {
  // Reshape to 28x28
  Eigen::MatrixXf img = Eigen::Map<const Eigen::MatrixXf>(image.data(), 28, 28);

  // Create output image
  Eigen::MatrixXf result = img;

  // Random shift (up to 2 pixels in each direction) - 70% probability
  if (rand() % 10 < 7) {
    int shiftX = rand() % 5 - 2;
    int shiftY = rand() % 5 - 2;

    Eigen::MatrixXf shifted = Eigen::MatrixXf::Zero(28, 28);
    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        int newY = y + shiftY;
        int newX = x + shiftX;
        if (newX >= 0 && newX < 28 && newY >= 0 && newY < 28) {
          shifted(newY, newX) = img(y, x);
        }
      }
    }
    result = shifted;
  }

  // Add small random noise (with 30% probability)
  if (rand() % 10 < 3) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.05f);

    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        result(y, x) += noise(gen);
        // Clamp to valid range
        result(y, x) = std::min(std::max(result(y, x), 0.0f), 1.0f);
      }
    }
  }

  // Reshape back to vector
  return Eigen::Map<Eigen::MatrixXf>(result.data(), 784, 1);
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

// Create directory if it doesn't exist
void createDirectory(const std::string &path) {
  std::filesystem::path dirPath(path);
  if (!std::filesystem::exists(dirPath)) {
    std::filesystem::create_directories(dirPath);
    std::cout << "Created directory: " << path << std::endl;
  }
}

// Format time in a human-readable way
std::string formatTime(std::chrono::milliseconds ms) {
  auto total_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(ms).count();
  auto minutes = total_seconds / 60;
  auto seconds = total_seconds % 60;

  std::stringstream ss;
  ss << minutes << "m " << seconds << "s";
  return ss.str();
}

int main() {
  std::cout << "Creating optimized MNIST CNN classifier..." << std::endl;

  // Create checkpoint directory
  std::string checkpointDir = "checkpoints_optimized";
  createDirectory(checkpointDir);

  // Create the model
  ZyraAIModel model;

  // Define CNN architecture:
  // 1x28x28 (input) -> 16x28x28 (Conv) -> 16x28x28 (BN+ReLU) -> 16x14x14 (Pool)
  // -> 32x14x14 (Conv) -> 32x14x14 (BN+ReLU) -> 32x7x7 (Pool) -> 1568 (Flatten)
  // -> 128 (Dense+ReLU) -> 10 (Dense+Softmax)

  // Input dimensions: 1 channel, 28x28 image
  const int inputChannels = 1;
  const int inputHeight = 28;
  const int inputWidth = 28;

  // First convolutional layer: 16 filters of size 5x5
  model.addLayer(std::make_shared<OptimizedConvLayer>(
      "conv1", inputChannels, inputHeight, inputWidth, 16, 5, 1,
      2)); // padding=2 to keep size same
  model.addLayer(std::make_shared<BatchNormLayer>("bn1", 16 * 28 * 28));
  model.addLayer(
      std::make_shared<ReLULayer>("relu1", 16 * 28 * 28, 16 * 28 * 28));
  model.addLayer(
      std::make_shared<MaxPoolingLayer>("pool1", 16, 28, 28, 2)); // 16x14x14

  // Second convolutional layer: 32 filters of size 3x3
  model.addLayer(std::make_shared<OptimizedConvLayer>(
      "conv2", 16, 14, 14, 32, 3, 1, 1)); // padding=1 to keep size same
  model.addLayer(std::make_shared<BatchNormLayer>("bn2", 32 * 14 * 14));
  model.addLayer(
      std::make_shared<ReLULayer>("relu2", 32 * 14 * 14, 32 * 14 * 14));
  model.addLayer(
      std::make_shared<MaxPoolingLayer>("pool2", 32, 14, 14, 2)); // 32x7x7

  // Flatten layer
  model.addLayer(
      std::make_shared<FlattenLayer>("flatten", 32, 7, 7)); // 32*7*7 = 1568

  // First fully connected layer
  model.addLayer(std::make_shared<DenseLayer>("dense1", 32 * 7 * 7, 128, true));
  model.addLayer(std::make_shared<ReLULayer>("relu3", 128, 128));
  model.addLayer(std::make_shared<DropoutLayer>("dropout", 128, 0.5f));

  // Output layer
  model.addLayer(std::make_shared<DenseLayer>("dense2", 128, 10, true));
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

    // Create Adam optimizer
    zyraai::AdamOptimizer optimizer(model, 0.001f);

    // Training parameters
    const int epochs = 30;
    const int batchSize = 128;
    const int numBatches = numTrainSamples / batchSize;

    // Create learning rate scheduler with warmup
    const int warmupEpochs = 3;
    WarmupCosineScheduler scheduler(0.001f, 0.00001f, warmupEpochs, epochs);

    // For early stopping
    float bestAccuracy = 0.0f;
    int patienceCount = 0;
    const int patience = 7;

    std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
    std::cout << "Batch size: " << batchSize
              << ", Initial learning rate: " << scheduler.getLearningRate(0)
              << std::endl;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
      // Set learning rate for this epoch using scheduler
      float learningRate = scheduler.getLearningRate(epoch);
      optimizer.setLearningRate(learningRate);

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

#pragma omp parallel for
        for (int i = 0; i < batchSize; ++i) {
          int idx = indices[batch * batchSize + i];
          // Apply augmentation during training
          batchImages.col(i) = augmentImage(trainImages.col(idx));
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
        model.backward(gradOutput,
                       0.0f); // Learning rate is handled by optimizer

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
                << ", Loss: " << epochLoss << ", LR: " << learningRate
                << ", Time: " << formatTime(epochDuration) << std::endl;

      // Evaluate on test set every epoch
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

      // Early stopping check and model saving
      if (accuracy > bestAccuracy) {
        bestAccuracy = accuracy;
        patienceCount = 0;
        std::cout << "New best accuracy! Saving model checkpoint..."
                  << std::endl;

        // Save model checkpoint
        std::string checkpointPath = checkpointDir + "/model_best.bin";
        if (ModelSerializer::saveModel(model, checkpointPath)) {
          std::cout << "Model saved to " << checkpointPath << std::endl;
        } else {
          std::cout << "Failed to save model" << std::endl;
        }

        // Also save epoch-specific checkpoint
        std::string epochCheckpointPath = checkpointDir + "/model_epoch_" +
                                          std::to_string(epoch + 1) + ".bin";
        ModelSerializer::saveModel(model, epochCheckpointPath);
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

    // Calculate total training duration
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Total training time: " << formatTime(totalDuration)
              << std::endl;

    // Load the best model for final evaluation
    std::string bestModelPath = checkpointDir + "/model_best.bin";
    if (std::filesystem::exists(bestModelPath)) {
      std::cout << "Loading best model from " << bestModelPath << std::endl;
      if (ModelSerializer::loadModel(model, bestModelPath)) {
        std::cout << "Best model loaded successfully" << std::endl;
      } else {
        std::cout << "Failed to load best model, using current model"
                  << std::endl;
      }
    }

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