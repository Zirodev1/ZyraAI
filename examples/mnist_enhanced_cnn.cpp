#include "model/adam_optimizer.h"
#include "model/batch_norm_layer.h"
#include "model/convolutional_layer.h"
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
#include <iomanip>
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

  // Small rotation (with 20% probability) - simplified approximation
  if (rand() % 10 < 2) {
    float angle = (rand() % 20 - 10) * 3.14159f / 180.0f; // +/- 10 degrees
    float sinA = sin(angle);
    float cosA = cos(angle);

    Eigen::MatrixXf rotated = Eigen::MatrixXf::Zero(28, 28);
    float centerX = 13.5f; // Center of 28x28 image
    float centerY = 13.5f;

    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        // Translate to origin, rotate, translate back
        float xr = cosA * (x - centerX) - sinA * (y - centerY) + centerX;
        float yr = sinA * (x - centerX) + cosA * (y - centerY) + centerY;

        // Bilinear interpolation
        int x0 = static_cast<int>(xr);
        int y0 = static_cast<int>(yr);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        if (x0 >= 0 && x1 < 28 && y0 >= 0 && y1 < 28) {
          float dx = xr - x0;
          float dy = yr - y0;

          rotated(y, x) = (1 - dx) * (1 - dy) * result(y0, x0) +
                          dx * (1 - dy) * result(y0, x1) +
                          (1 - dx) * dy * result(y1, x0) +
                          dx * dy * result(y1, x1);
        }
      }
    }
    result = rotated;
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
  std::cout
      << "Creating Enhanced MNIST CNN classifier with three conv layers..."
      << std::endl;

  // Create checkpoint directory
  std::string checkpointDir = "checkpoints_enhanced_cnn";
  createDirectory(checkpointDir);

  // Create the model
  ZyraAIModel model;

  // Define CNN architecture with three convolutional layers:
  // 1x28x28 (input) -> 16x28x28 (Conv1) -> 16x28x28 (BN+ReLU) -> 16x14x14
  // (Pool)
  // -> 32x14x14 (Conv2) -> 32x14x14 (BN+ReLU) -> 32x7x7 (Pool)
  // -> 64x7x7 (Conv3) -> 64x7x7 (BN+ReLU) -> 64x3x3 (Pool)
  // -> 576 (Flatten) -> 128 (Dense+ReLU) -> 10 (Dense+Softmax)

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

  // Third convolutional layer: 64 filters of size 3x3
  model.addLayer(std::make_shared<OptimizedConvLayer>(
      "conv3", 32, 7, 7, 64, 3, 1, 1)); // padding=1 to keep size same
  model.addLayer(std::make_shared<BatchNormLayer>("bn3", 64 * 7 * 7));
  model.addLayer(std::make_shared<ReLULayer>("relu3", 64 * 7 * 7, 64 * 7 * 7));
  model.addLayer(
      std::make_shared<MaxPoolingLayer>("pool3", 64, 7, 7, 2)); // 64x3x3

  // Flatten layer
  model.addLayer(
      std::make_shared<FlattenLayer>("flatten", 64, 3, 3)); // 64*3*3 = 576

  // First fully connected layer
  model.addLayer(std::make_shared<DenseLayer>("dense1", 64 * 3 * 3, 128, true));
  model.addLayer(std::make_shared<ReLULayer>("relu_dense", 128, 128));
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
    const int epochs = 50;
    const int batchSize = 128;
    const int numBatches = numTrainSamples / batchSize;

    // Create learning rate scheduler with warmup
    const int warmupEpochs = 5;
    WarmupCosineScheduler scheduler(0.001f, 0.00001f, warmupEpochs, epochs);

    // For early stopping
    float bestAccuracy = 0.0f;
    int patienceCount = 0;
    const int patience = 10;

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
        // Create batch
        Eigen::MatrixXf batchImages(784, batchSize);
        Eigen::MatrixXf batchLabels(10, batchSize);

        for (int i = 0; i < batchSize; ++i) {
          int idx = indices[batch * batchSize + i];

          // Apply data augmentation
          batchImages.col(i) = augmentImage(trainImages.col(idx));
          batchLabels.col(i) = trainLabels.col(idx);
        }

        // Forward pass
        Eigen::MatrixXf predictions = model.forward(batchImages);

        // Compute loss (cross-entropy)
        float batchLoss = 0.0f;
        for (int i = 0; i < batchSize; ++i) {
          for (int j = 0; j < 10; ++j) {
            float y = batchLabels(j, i);
            float yHat = predictions(j, i);
            batchLoss -= y * log(yHat + 1e-7f);
          }
        }
        batchLoss /= batchSize;
        epochLoss += batchLoss;

        // Backward pass and optimize
        Eigen::MatrixXf gradOutput = predictions - batchLabels;
        model.backward(gradOutput, learningRate);
        optimizer.step();
      }

      // Calculate average loss for the epoch
      epochLoss /= numBatches;

      // Measure epoch time
      auto epochEndTime = std::chrono::high_resolution_clock::now();
      auto epochDuration =
          std::chrono::duration_cast<std::chrono::milliseconds>(epochEndTime -
                                                                epochStartTime);

      // Evaluate on test set every 2 epochs
      if (epoch % 2 == 0 || epoch == epochs - 1) {
        // Switch to evaluation mode
        model.setTraining(false);

        // Compute test accuracy
        int correct = 0;
        int total = 0;

        // Process in batches to avoid memory issues
        const int testBatchSize = 500;
        const int testBatches = numTestSamples / testBatchSize;

        for (int b = 0; b < testBatches; ++b) {
          Eigen::MatrixXf testBatchImages =
              testImages.block(0, b * testBatchSize, 784, testBatchSize);
          Eigen::MatrixXf testBatchLabels =
              testLabels.block(0, b * testBatchSize, 10, testBatchSize);

          Eigen::MatrixXf predictions = model.forward(testBatchImages);

          for (int i = 0; i < testBatchSize; ++i) {
            int predictedClass = 0;
            float maxVal = predictions(0, i);

            for (int j = 1; j < 10; ++j) {
              if (predictions(j, i) > maxVal) {
                maxVal = predictions(j, i);
                predictedClass = j;
              }
            }

            int actualClass = 0;
            for (int j = 0; j < 10; ++j) {
              if (testBatchLabels(j, i) > 0.5f) {
                actualClass = j;
                break;
              }
            }

            if (predictedClass == actualClass) {
              correct++;
            }
            total++;
          }
        }

        float testAccuracy = 100.0f * correct / total;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << epochLoss << ", Time: " << std::fixed
                  << std::setprecision(3) << epochDuration.count() / 1000.0f
                  << "s" << std::endl;
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2)
                  << testAccuracy << "%" << std::endl;

        // Save checkpoint if accuracy improved
        if (testAccuracy > bestAccuracy) {
          std::cout << "New best accuracy!" << std::endl;
          bestAccuracy = testAccuracy;
          patienceCount = 0;

          // Save model checkpoint
          std::string checkpointPath =
              checkpointDir + "/checkpoint_" + std::to_string(epoch + 1) + "_" +
              std::to_string(static_cast<int>(testAccuracy)) + ".bin";
          // Note: ModelSerializer::save not implemented yet - will be added in
          // Phase 3
          std::cout << "Would save model to: " << checkpointPath << std::endl;
          // ModelSerializer::save(model, checkpointPath);
        } else {
          patienceCount++;
          if (patienceCount >= patience) {
            std::cout << "Early stopping triggered after " << epoch + 1
                      << " epochs" << std::endl;
            break;
          }
        }

        // Switch back to training mode
        model.setTraining(true);
      } else {
        // Just print training progress
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << ", Loss: " << epochLoss << ", Time: " << std::fixed
                  << std::setprecision(3) << epochDuration.count() / 1000.0f
                  << "s" << std::endl;
      }
    }

    // Calculate total training time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto trainingDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime -
                                                              startTime);

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "Total training time: " << formatTime(trainingDuration)
              << std::endl;
    std::cout << "Final Test Accuracy: " << std::fixed << std::setprecision(2)
              << bestAccuracy << "%" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}