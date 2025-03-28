#include "model/adam_optimizer.h"
#include "model/batch_norm_layer.h"
#include "model/convolutional_layer.h"
#include "model/dense_layer.h"
#include "model/dropout_layer.h"
#include "model/flatten_layer.h"
#include "model/max_pooling_layer.h"
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

// Function to read CIFAR-10 data from binary files
std::pair<Eigen::MatrixXf, Eigen::MatrixXf>
readCIFAR10(const std::string &dataPath, int numSamples) {
    std::ifstream file(dataPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + dataPath);
    }

    const int imageSize = 32 * 32 * 3; // 32x32 RGB images
    Eigen::MatrixXf images(imageSize, numSamples);
    Eigen::MatrixXf labels = Eigen::MatrixXf::Zero(10, numSamples);
    
    std::vector<unsigned char> buffer(imageSize + 1); // +1 for label

    // First pass: compute mean and std
    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), imageSize + 1);
        unsigned char label = buffer[0];
        for (int j = 0; j < imageSize; ++j) {
            float pixel = static_cast<float>(buffer[j + 1]) / 255.0f;
            sum += pixel;
            sumSq += pixel * pixel;
        }
    }
    float mean = sum / (numSamples * imageSize);
    float std = std::sqrt(sumSq / (numSamples * imageSize) - mean * mean);

    // Second pass: normalize data
    file.clear();
    file.seekg(0);
    for (int i = 0; i < numSamples; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), imageSize + 1);
        
        // Set one-hot encoded label
        labels(buffer[0], i) = 1.0f;
        
        // Normalize image data
        for (int j = 0; j < imageSize; ++j) {
            float pixel = static_cast<float>(buffer[j + 1]) / 255.0f;
            images(j, i) = (pixel - mean) / (std + 1e-7f);
        }
    }

    return {images, labels};
}

// Function to augment CIFAR-10 images
Eigen::MatrixXf augmentImage(const Eigen::MatrixXf &image) {
    // Reshape to 32x32x3
    Eigen::MatrixXf img = Eigen::Map<const Eigen::MatrixXf>(image.data(), 32 * 32, 3);
    
    // Random horizontal flip
    if (rand() % 2 == 0) {
        for (int c = 0; c < 3; ++c) {
            Eigen::MatrixXf channel = Eigen::Map<const Eigen::MatrixXf>(
                img.col(c).data(), 32, 32);
            channel = channel.rowwise().reverse();
            img.col(c) = Eigen::Map<const Eigen::VectorXf>(channel.data(), 32 * 32);
        }
    }
    
    // Random shift (up to 4 pixels)
    int shiftX = rand() % 9 - 4;
    int shiftY = rand() % 9 - 4;
    
    Eigen::MatrixXf shifted = Eigen::MatrixXf::Zero(32 * 32, 3);
    for (int c = 0; c < 3; ++c) {
        Eigen::MatrixXf channel = Eigen::Map<const Eigen::MatrixXf>(
            img.col(c).data(), 32, 32);
        Eigen::MatrixXf shiftedChannel = Eigen::MatrixXf::Zero(32, 32);
        
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                int newY = y + shiftY;
                int newX = x + shiftX;
                if (newX >= 0 && newX < 32 && newY >= 0 && newY < 32) {
                    shiftedChannel(newY, newX) = channel(y, x);
                }
            }
        }
        
        shifted.col(c) = Eigen::Map<const Eigen::VectorXf>(
            shiftedChannel.data(), 32 * 32);
    }
    
    // Reshape back to vector
    return Eigen::Map<Eigen::MatrixXf>(shifted.data(), 32 * 32 * 3, 1);
}

// Function to check CIFAR-10 data files
bool checkCIFAR10Files() {
    std::vector<std::string> requiredFiles = {
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_1.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_2.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_3.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_4.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_5.bin"
    };
    
    bool allFilesExist = true;
    
    // Check training batches
    for (const auto& file : requiredFiles) {
        std::ifstream checkFile(file);
        if (!checkFile.good()) {
            std::cerr << "Missing file: " << file << std::endl;
            allFilesExist = false;
        }
        checkFile.close();
    }
    
    // Test file is optional (we'll split from training if needed)
    std::string testPath = "I:/ZyraAI/data/cifar-10-batches-bin/test_batch.bin";
    std::ifstream checkTestFile(testPath);
    bool hasTestFile = checkTestFile.good();
    checkTestFile.close();
    
    if (!hasTestFile) {
        std::cout << "Note: Test batch file not found. Will split from training data instead." << std::endl;
    }
    
    if (!allFilesExist) {
        std::cerr << "Please download the CIFAR-10 dataset from:\n"
                  << "https://www.cs.toronto.edu/~kriz/cifar.html\n"
                  << "Extract the binary version to I:/ZyraAI/data/cifar-10-batches-bin/" << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    std::cout << "Creating CIFAR-10 classifier..." << std::endl;
    
    // Check if dataset is available
    if (!checkCIFAR10Files()) {
        return 1;
    }

    // Create the model
    ZyraAIModel model;
    
    // Input: 32x32x3 (RGB images)
    // Conv1: 32 filters, 3x3, stride 1, padding 1
    model.addLayer(std::make_shared<ConvolutionalLayer>(
        "conv1", 3, 32, 32, 32, 3, 1, 1));
    model.addLayer(std::make_shared<BatchNormLayer>("bn1", 32 * 32 * 32));
    model.addLayer(std::make_shared<ReLULayer>("relu1", 32 * 32 * 32, 32 * 32 * 32));
    model.addLayer(std::make_shared<DropoutLayer>("dropout1", 32 * 32 * 32, 0.2f));
    
    // Conv2: 32 filters, 3x3, stride 1, padding 1
    model.addLayer(std::make_shared<ConvolutionalLayer>(
        "conv2", 32, 32, 32, 32, 3, 1, 1));
    model.addLayer(std::make_shared<BatchNormLayer>("bn2", 32 * 32 * 32));
    model.addLayer(std::make_shared<ReLULayer>("relu2", 32 * 32 * 32, 32 * 32 * 32));
    model.addLayer(std::make_shared<DropoutLayer>("dropout2", 32 * 32 * 32, 0.2f));
    
    // MaxPool1: 2x2, stride 2
    model.addLayer(std::make_shared<MaxPoolingLayer>("pool1", 32, 32, 32, 2, 2));
    
    // Conv3: 64 filters, 3x3, stride 1, padding 1
    model.addLayer(std::make_shared<ConvolutionalLayer>(
        "conv3", 32, 16, 16, 64, 3, 1, 1));
    model.addLayer(std::make_shared<BatchNormLayer>("bn3", 16 * 16 * 64));
    model.addLayer(std::make_shared<ReLULayer>("relu3", 16 * 16 * 64, 16 * 16 * 64));
    model.addLayer(std::make_shared<DropoutLayer>("dropout3", 16 * 16 * 64, 0.3f));
    
    // Conv4: 64 filters, 3x3, stride 1, padding 1
    model.addLayer(std::make_shared<ConvolutionalLayer>(
        "conv4", 64, 16, 16, 64, 3, 1, 1));
    model.addLayer(std::make_shared<BatchNormLayer>("bn4", 16 * 16 * 64));
    model.addLayer(std::make_shared<ReLULayer>("relu4", 16 * 16 * 64, 16 * 16 * 64));
    model.addLayer(std::make_shared<DropoutLayer>("dropout4", 16 * 16 * 64, 0.3f));
    
    // MaxPool2: 2x2, stride 2
    model.addLayer(std::make_shared<MaxPoolingLayer>("pool2", 16, 16, 64, 2, 2));
    
    // Conv5: 128 filters, 3x3, stride 1, padding 1
    model.addLayer(std::make_shared<ConvolutionalLayer>(
        "conv5", 64, 8, 8, 128, 3, 1, 1));
    model.addLayer(std::make_shared<BatchNormLayer>("bn5", 8 * 8 * 128));
    model.addLayer(std::make_shared<ReLULayer>("relu5", 8 * 8 * 128, 8 * 8 * 128));
    model.addLayer(std::make_shared<DropoutLayer>("dropout5", 8 * 8 * 128, 0.4f));
    
    // MaxPool3: 2x2, stride 2
    model.addLayer(std::make_shared<MaxPoolingLayer>("pool3", 8, 8, 128, 2, 2));
    
    // Flatten layer
    model.addLayer(std::make_shared<FlattenLayer>("flatten", 128, 4, 4));
    
    // Dense layers with L2 regularization
    model.addLayer(std::make_shared<DenseLayer>("dense1", 2048, 512, 0.01f));
    model.addLayer(std::make_shared<BatchNormLayer>("bn6", 512));
    model.addLayer(std::make_shared<ReLULayer>("relu6", 512, 512));
    model.addLayer(std::make_shared<DropoutLayer>("dropout6", 512, 0.5f));
    
    model.addLayer(std::make_shared<DenseLayer>("dense2", 512, 256, 0.01f));
    model.addLayer(std::make_shared<BatchNormLayer>("bn7", 256));
    model.addLayer(std::make_shared<ReLULayer>("relu7", 256, 256));
    model.addLayer(std::make_shared<DropoutLayer>("dropout7", 256, 0.5f));
    
    // Output layer
    model.addLayer(std::make_shared<DenseLayer>("output", 256, 10, 0.01f));
    model.addLayer(std::make_shared<SoftmaxLayer>("softmax", 10));

    try {
        // Load training data (50000 samples)
        const int numTrainSamples = 50000;
        std::cout << "Loading training data..." << std::endl;
        
        // Load all training batches
        Eigen::MatrixXf trainImages(32 * 32 * 3, numTrainSamples);
        Eigen::MatrixXf trainLabels = Eigen::MatrixXf::Zero(10, numTrainSamples);
        
        for (int batch = 1; batch <= 5; ++batch) {
            std::string batchPath = "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_" + 
                                  std::to_string(batch) + ".bin";
            
            std::cout << "Loading batch " << batch << "..." << std::endl;
            
            auto [batchImages, batchLabels] = readCIFAR10(batchPath, 10000);
            
            // Copy to the appropriate position in the full dataset
            trainImages.block(0, (batch - 1) * 10000, 32 * 32 * 3, 10000) = batchImages;
            trainLabels.block(0, (batch - 1) * 10000, 10, 10000) = batchLabels;
        }

        // Split into training and test sets (since test_batch.bin is missing)
        const int numTestSamples = 5000;
        const int actualTrainSamples = numTrainSamples - numTestSamples;
        
        std::cout << "Splitting data into " << actualTrainSamples << " training samples and " 
                  << numTestSamples << " test samples" << std::endl;
        
        // Use the last 5000 samples as test data
        Eigen::MatrixXf testImages = trainImages.block(0, actualTrainSamples, 32 * 32 * 3, numTestSamples);
        Eigen::MatrixXf testLabels = trainLabels.block(0, actualTrainSamples, 10, numTestSamples);
        
        // Resize training data to exclude test samples
        trainImages.conservativeResize(Eigen::NoChange, actualTrainSamples);
        trainLabels.conservativeResize(Eigen::NoChange, actualTrainSamples);

        // Create optimizer with lower learning rate
        AdamOptimizer optimizer(model, 0.0001f, 0.9f, 0.999f, 1e-8f);
        
        // Learning rate scheduler
        float initialLearningRate = 0.0001f;
        float minLearningRate = 0.00001f;
        float decayRate = 0.95f;
        float currentLearningRate = initialLearningRate;
        
        // Training loop
        const int numEpochs = 50;
        const int batchSize = 16;  // Reduced batch size to avoid memory issues
        const int numBatches = trainImages.cols() / batchSize;
        
        std::cout << "Starting training with " << numBatches << " batches of size " << batchSize << std::endl;
        for (int epoch = 0; epoch < numEpochs; ++epoch) {
            float epochLoss = 0.0f;
            int correct = 0;
            int total = 0;
            
            std::cout << "Epoch " << epoch + 1 << " preparing data..." << std::endl;
            
            // Shuffle training data
            std::vector<int> indices(trainImages.cols());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
            
            std::cout << "Starting batch training..." << std::endl;
            
            // Training batches
            for (int batch = 0; batch < numBatches; ++batch) {
                if (batch % 100 == 0) {
                    std::cout << "  Processing batch " << batch << "/" << numBatches << std::endl;
                }
                
                // Get batch indices
                std::vector<int> batchIndices(indices.begin() + batch * batchSize,
                                            indices.begin() + (batch + 1) * batchSize);
                
                // Extract batch data
                Eigen::MatrixXf batchInput = trainImages(Eigen::all, batchIndices);
                Eigen::MatrixXf batchTarget = trainLabels(Eigen::all, batchIndices);
                
                // Forward pass
                try {
                    Eigen::MatrixXf output = model.forward(batchInput);
                    
                    // Calculate loss and accuracy
                    float batchLoss = model.computeLoss(output, batchTarget);
                    epochLoss += batchLoss;
                    
                    // Calculate accuracy
                    Eigen::MatrixXf::Index maxIndex;
                    for (int i = 0; i < output.cols(); ++i) {
                        output.col(i).maxCoeff(&maxIndex);
                        if (maxIndex == batchTarget.col(i).maxCoeff()) {
                            correct++;
                        }
                        total++;
                    }
                    
                    // Backward pass with gradient clipping
                    Eigen::MatrixXf gradOutput =
                        (output - batchTarget) / static_cast<float>(batchSize);
                    model.backward(gradOutput, currentLearningRate);
                    
                    // Update weights with gradient clipping
                    optimizer.step();
                } catch (const std::exception& e) {
                    std::cerr << "Error during forward/backward pass: " << e.what() << std::endl;
                    std::cerr << "Batch: " << batch << std::endl;
                    return 1;
                }
            }
            
            // Calculate epoch metrics
            float avgLoss = epochLoss / numBatches;
            float accuracy = (float)correct / total * 100.0f;
            
            // Decay learning rate
            currentLearningRate = std::max(minLearningRate, currentLearningRate * decayRate);
            optimizer.setLearningRate(currentLearningRate);
            
            std::cout << "Epoch " << (epoch + 1) << "/" << numEpochs
                      << " - Loss: " << avgLoss
                      << " - Accuracy: " << accuracy << "%"
                      << " - Learning Rate: " << currentLearningRate << std::endl;

            // Evaluate on test set
            model.setTraining(false);
            
            // Evaluate in batches to save memory
            int correctTest = 0;
            const int evalBatchSize = 100;
            const int evalBatches = numTestSamples / evalBatchSize;
            
            for (int batch = 0; batch < evalBatches; ++batch) {
                Eigen::MatrixXf batchImages = testImages.block(
                    0, batch * evalBatchSize, 32 * 32 * 3, evalBatchSize);
                Eigen::MatrixXf batchLabels = testLabels.block(
                    0, batch * evalBatchSize, 10, evalBatchSize);
                
                Eigen::MatrixXf predictions = model.forward(batchImages);
                
                for (int i = 0; i < evalBatchSize; ++i) {
                    Eigen::MatrixXf::Index maxRow, maxCol;
                    predictions.col(i).maxCoeff(&maxRow, &maxCol);
                    Eigen::MatrixXf::Index trueRow, trueCol;
                    batchLabels.col(i).maxCoeff(&trueRow, &trueCol);
                    if (maxRow == trueRow)
                        correctTest++;
                }
            }

            float testAccuracy = static_cast<float>(correctTest) / numTestSamples * 100.0f;
            std::cout << "Test Accuracy: " << testAccuracy << "%" << std::endl;

            model.setTraining(true);
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 