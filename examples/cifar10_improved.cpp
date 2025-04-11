#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <memory>
#include <string>
#include <filesystem>
#include <Eigen/Dense>

#include "model/adam_optimizer.h"
#include "model/batch_norm_layer.h"
#include "model/dense_layer.h"
#include "model/dropout_layer.h"
#include "model/relu_layer.h"
#include "model/softmax_layer.h"
#include "model/zyraAI_model.h"

using namespace zyraai;
namespace fs = std::filesystem;

// Function to check CIFAR-10 data files
bool checkCIFAR10Files() {
    std::vector<std::string> requiredFiles = {
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_1.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_2.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_3.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_4.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_5.bin",
        "I:/ZyraAI/data/cifar-10-batches-bin/test_batch.bin"
    };

    for (const auto& file : requiredFiles) {
        std::ifstream f(file, std::ios::binary);
        if (!f.good()) {
            std::cerr << "Error: Required file not found: " << file << std::endl;
            std::cerr << "Please download the CIFAR-10 binary version and extract to data/cifar-10-batches-bin/" << std::endl;
            return false;
        }
    }
    return true;
}

// Read CIFAR-10 dataset directly into Eigen format
std::pair<std::pair<std::vector<Eigen::MatrixXf>, std::vector<int>>, 
          std::pair<std::vector<Eigen::MatrixXf>, std::vector<int>>> 
readCIFAR10() {
    const int imageSize = 3 * 32 * 32; // 3 channels, 32x32 pixels
    const int batchSize = 10000;
    
    std::vector<Eigen::MatrixXf> trainImages;
    std::vector<int> trainLabels;
    
    // Variables for normalization
    Eigen::VectorXf sum = Eigen::VectorXf::Zero(imageSize);
    Eigen::VectorXf sumSquared = Eigen::VectorXf::Zero(imageSize);
    int totalTrainSamples = 0;
    
    // Read training data (5 batches)
    for (int batch = 1; batch <= 5; batch++) {
        std::string filename = "I:/ZyraAI/data/cifar-10-batches-bin/data_batch_" + std::to_string(batch) + ".bin";
        std::ifstream file(filename, std::ios::binary);
        
        for (int i = 0; i < batchSize; i++) {
            // Read label (1 byte)
            char label;
            file.read(&label, 1);
            trainLabels.push_back(static_cast<int>(label));
            
            // Read image data (3072 bytes)
            Eigen::MatrixXf image(imageSize, 1);
            for (int j = 0; j < imageSize; j++) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                float pixelValue = static_cast<float>(pixel) / 255.0f;
                image(j, 0) = pixelValue;
                
                // Accumulate for normalization
                sum(j) += pixelValue;
                sumSquared(j) += pixelValue * pixelValue;
            }
            
            trainImages.push_back(image);
            totalTrainSamples++;
        }
    }
    
    // Calculate mean and standard deviation for normalization
    Eigen::VectorXf mean = sum / totalTrainSamples;
    
    // Fix: properly compute variance and std deviation using Eigen array operations
    Eigen::ArrayXf meanArray = mean.array();
    Eigen::ArrayXf varianceArray = (sumSquared.array() / totalTrainSamples) - meanArray.square();
    Eigen::VectorXf stddev = varianceArray.sqrt().matrix();
    
    // Apply normalization to training images
    for (auto& image : trainImages) {
        image = (image.array() - mean.array()) / (stddev.array() + 1e-8f);
    }
    
    // Read test data
    std::vector<Eigen::MatrixXf> testImages;
    std::vector<int> testLabels;
    
    std::string testFilename = "I:/ZyraAI/data/cifar-10-batches-bin/test_batch.bin";
    std::ifstream testFile(testFilename, std::ios::binary);
    
    for (int i = 0; i < batchSize; i++) {
        // Read label (1 byte)
        char label;
        testFile.read(&label, 1);
        testLabels.push_back(static_cast<int>(label));
        
        // Read image data (3072 bytes)
        Eigen::MatrixXf image(imageSize, 1);
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            testFile.read(reinterpret_cast<char*>(&pixel), 1);
            float pixelValue = static_cast<float>(pixel) / 255.0f;
            image(j, 0) = pixelValue;
        }
        
        // Apply same normalization as training data
        image = (image.array() - mean.array()) / (stddev.array() + 1e-8f);
        testImages.push_back(image);
    }
    
    return {{trainImages, trainLabels}, {testImages, testLabels}};
}

// Create efficient mini-batches
Eigen::MatrixXf createMiniBatch(const std::vector<Eigen::MatrixXf>& images, 
                               const std::vector<int>& indices, 
                               int startIdx, int batchSize) {
    int actualBatchSize = std::min(batchSize, static_cast<int>(indices.size() - startIdx));
    int featureSize = images[0].rows();
    
    Eigen::MatrixXf batch(featureSize, actualBatchSize);
    
    for (int i = 0; i < actualBatchSize; i++) {
        batch.col(i) = images[indices[startIdx + i]];
    }
    
    return batch;
}

// Create one-hot encoded batch
Eigen::MatrixXf createOneHotBatch(const std::vector<int>& labels, 
                                 const std::vector<int>& indices,
                                 int startIdx, int batchSize, int numClasses) {
    int actualBatchSize = std::min(batchSize, static_cast<int>(indices.size() - startIdx));
    
    Eigen::MatrixXf batch = Eigen::MatrixXf::Zero(numClasses, actualBatchSize);
    
    for (int i = 0; i < actualBatchSize; i++) {
        batch(labels[indices[startIdx + i]], i) = 1.0f;
    }
    
    return batch;
}

// Create directory if it doesn't exist
void createDirectoryIfNotExists(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

// Save model checkpoint
void saveModelCheckpoint(const ZyraAIModel& model, const std::string& filename) {
    createDirectoryIfNotExists("I:/ZyraAI/checkpoints");
    std::ofstream file(filename, std::ios::binary);
    // We would need to implement serialization for our model
    // For now, just create an empty file as a placeholder
    file.close();
    std::cout << "Checkpoint saved to " << filename << std::endl;
}

int main() {
    std::cout << "Creating improved CIFAR-10 classifier..." << std::endl;
    
    // Check if dataset is available
    if (!checkCIFAR10Files()) {
        return 1;
    }
    
    // Read CIFAR-10 dataset
    auto startTime = std::chrono::high_resolution_clock::now();
    std::cout << "Loading CIFAR-10 dataset..." << std::endl;
    
    auto [trainData, testData] = readCIFAR10();
    auto& [trainImages, trainLabels] = trainData;
    auto& [testImages, testLabels] = testData;
    
    std::cout << "Dataset loaded. " << trainImages.size() << " training examples, " 
              << testImages.size() << " test examples." << std::endl;
    
    // Create a validation set (split off 10% of training data)
    const size_t validationSize = trainImages.size() / 10;
    std::vector<Eigen::MatrixXf> validationImages(trainImages.end() - validationSize, trainImages.end());
    std::vector<int> validationLabels(trainLabels.end() - validationSize, trainLabels.end());
    trainImages.resize(trainImages.size() - validationSize);
    trainLabels.resize(trainLabels.size() - validationSize);
    
    std::cout << "Split into " << trainImages.size() << " training, " 
              << validationImages.size() << " validation, and "
              << testImages.size() << " test examples." << std::endl;
    
    // Define model hyperparameters
    const int numClasses = 10;
    const int inputChannels = 3;
    const int inputHeight = 32;
    const int inputWidth = 32;
    const int batchSize = 64;
    const int epochs = 100;
    const float initialLearningRate = 0.001f;
    
    // Build an efficient model for CIFAR-10
    ZyraAIModel model;
    
    // Input size
    const int inputSize = inputChannels * inputHeight * inputWidth;
    
    // Layer 1: 128 neurons with batch normalization
    model.addLayer(std::make_shared<DenseLayer>("dense1", inputSize, 512));
    model.addLayer(std::make_shared<BatchNormLayer>("bn1", 512));
    model.addLayer(std::make_shared<ReLULayer>("relu1", 512, 512));
    model.addLayer(std::make_shared<DropoutLayer>("dropout1", 512, 0.3f));
    
    // Layer 2: 256 neurons with batch normalization
    model.addLayer(std::make_shared<DenseLayer>("dense2", 512, 256));
    model.addLayer(std::make_shared<BatchNormLayer>("bn2", 256));
    model.addLayer(std::make_shared<ReLULayer>("relu2", 256, 256));
    model.addLayer(std::make_shared<DropoutLayer>("dropout2", 256, 0.3f));
    
    // Output layer
    model.addLayer(std::make_shared<DenseLayer>("dense3", 256, numClasses));
    model.addLayer(std::make_shared<SoftmaxLayer>("softmax", numClasses));
    
    // Initialize Adam optimizer
    AdamOptimizer optimizer(model, initialLearningRate);
    
    // Training statistics
    std::vector<float> trainLossHistory;
    std::vector<float> trainAccuracyHistory;
    std::vector<float> validationLossHistory;
    std::vector<float> validationAccuracyHistory;
    
    // Early stopping variables
    float bestValidationAccuracy = 0.0f;
    int patienceCounter = 0;
    const int patience = 10;
    
    std::cout << "Starting training with efficient mini-batch processing..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Learning rate schedule
        if (epoch == 50 || epoch == 75) {
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.1f);
            std::cout << "Reducing learning rate to " << optimizer.getLearningRate() << std::endl;
        }
        
        // Shuffle training data indices
        std::vector<int> indices(trainImages.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Training metrics
        float trainLoss = 0.0f;
        int trainCorrect = 0;
        
        // Set model to training mode
        model.setTraining(true);
        
        // Process in mini-batches
        for (size_t b = 0; b < trainImages.size(); b += batchSize) {
            // Create mini-batch
            Eigen::MatrixXf batchInput = createMiniBatch(trainImages, indices, b, batchSize);
            Eigen::MatrixXf batchTarget = createOneHotBatch(trainLabels, indices, b, batchSize, numClasses);
            
            int actualBatchSize = batchInput.cols();
            
            // Forward pass
            Eigen::MatrixXf output = model.forward(batchInput);
            
            // Compute loss
            float batchLoss = model.computeLoss(output, batchTarget);
            trainLoss += batchLoss * actualBatchSize;
            
            // Compute accuracy
            for (int i = 0; i < actualBatchSize; i++) {
                Eigen::MatrixXf::Index maxRow, maxCol;
                output.col(i).maxCoeff(&maxRow, &maxCol);
                int predictedClass = static_cast<int>(maxRow);
                if (predictedClass == trainLabels[indices[b + i]]) {
                    trainCorrect++;
                }
            }
            
            // Backward pass
            Eigen::MatrixXf gradOutput = (output - batchTarget) / actualBatchSize;
            model.backward(gradOutput, optimizer.getLearningRate());
            
            // Update parameters
            optimizer.step();
            
            // Print progress for larger batches
            if ((b / batchSize) % 20 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs
                          << ", Batch " << b / batchSize << "/" << trainImages.size() / batchSize
                          << ", Progress: " << std::fixed << std::setprecision(1) 
                          << (100.0f * b / trainImages.size()) << "%" << std::endl;
            }
        }
        
        // Calculate training metrics
        trainLoss /= trainImages.size();
        float trainAccuracy = static_cast<float>(trainCorrect) / trainImages.size();
        
        trainLossHistory.push_back(trainLoss);
        trainAccuracyHistory.push_back(trainAccuracy);
        
        // Validation phase
        float validationLoss = 0.0f;
        int validationCorrect = 0;
        
        // Set model to evaluation mode
        model.setTraining(false);
        
        // Process validation set in mini-batches
        for (size_t b = 0; b < validationImages.size(); b += batchSize) {
            std::vector<int> valIndices(validationSize);
            std::iota(valIndices.begin(), valIndices.end(), 0);
            
            Eigen::MatrixXf batchInput = createMiniBatch(validationImages, valIndices, b, batchSize);
            Eigen::MatrixXf batchTarget = createOneHotBatch(validationLabels, valIndices, b, batchSize, numClasses);
            
            int actualBatchSize = batchInput.cols();
            
            // Forward pass
            Eigen::MatrixXf output = model.forward(batchInput);
            
            // Compute loss
            float batchLoss = model.computeLoss(output, batchTarget);
            validationLoss += batchLoss * actualBatchSize;
            
            // Compute accuracy
            for (int i = 0; i < actualBatchSize; i++) {
                Eigen::MatrixXf::Index maxRow, maxCol;
                output.col(i).maxCoeff(&maxRow, &maxCol);
                int predictedClass = static_cast<int>(maxRow);
                if (predictedClass == validationLabels[b + i]) {
                    validationCorrect++;
                }
            }
        }
        
        validationLoss /= validationImages.size();
        float validationAccuracy = static_cast<float>(validationCorrect) / validationImages.size();
        
        validationLossHistory.push_back(validationLoss);
        validationAccuracyHistory.push_back(validationAccuracy);
        
        // Print epoch results
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " - Training: loss=" << trainLoss << ", accuracy=" << trainAccuracy * 100.0f << "%"
                  << " - Validation: loss=" << validationLoss << ", accuracy=" << validationAccuracy * 100.0f << "%" 
                  << std::endl;
        
        // Save checkpoint if it's the best model
        if (validationAccuracy > bestValidationAccuracy) {
            bestValidationAccuracy = validationAccuracy;
            patienceCounter = 0;
            std::string checkpointPath = "I:/ZyraAI/checkpoints/cifar10_improved_best.bin";
            saveModelCheckpoint(model, checkpointPath);
            std::cout << "New best model saved! Validation accuracy: " << validationAccuracy * 100.0f << "%" << std::endl;
        } else {
            patienceCounter++;
            // Save regular checkpoint every 10 epochs
            if (epoch % 10 == 9) {
                std::string checkpointPath = "I:/ZyraAI/checkpoints/cifar10_improved_epoch_" + std::to_string(epoch + 1) + ".bin";
                saveModelCheckpoint(model, checkpointPath);
            }
        }
        
        // Early stopping
        if (patienceCounter >= patience) {
            std::cout << "Early stopping triggered. No improvement for " << patience << " epochs." << std::endl;
            break;
        }
    }
    
    // Test phase
    model.setTraining(false);
    
    // Evaluate on test set in batches
    float testLoss = 0.0f;
    int testCorrect = 0;
    
    // Set up confusion matrix
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    
    for (size_t b = 0; b < testImages.size(); b += batchSize) {
        std::vector<int> testIndices(testImages.size());
        std::iota(testIndices.begin(), testIndices.end(), 0);
        
        Eigen::MatrixXf batchInput = createMiniBatch(testImages, testIndices, b, batchSize);
        Eigen::MatrixXf batchTarget = createOneHotBatch(testLabels, testIndices, b, batchSize, numClasses);
        
        int actualBatchSize = batchInput.cols();
        
        // Forward pass
        Eigen::MatrixXf output = model.forward(batchInput);
        
        // Compute loss
        float batchLoss = model.computeLoss(output, batchTarget);
        testLoss += batchLoss * actualBatchSize;
        
        // Compute accuracy and fill confusion matrix
        for (int i = 0; i < actualBatchSize; i++) {
            Eigen::MatrixXf::Index maxRow, maxCol;
            output.col(i).maxCoeff(&maxRow, &maxCol);
            int predictedClass = static_cast<int>(maxRow);
            int trueClass = testLabels[b + i];
            
            confusionMatrix[trueClass][predictedClass]++;
            
            if (predictedClass == trueClass) {
                testCorrect++;
            }
        }
    }
    
    testLoss /= testImages.size();
    float testAccuracy = static_cast<float>(testCorrect) / testImages.size();
    
    // Print final results
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "Test accuracy: " << testAccuracy * 100.0f << "%" << std::endl;
    std::cout << "Test loss: " << testLoss << std::endl;
    
    // Print per-class accuracy
    std::cout << "\nPer-class accuracy:" << std::endl;
    for (int i = 0; i < numClasses; i++) {
        int totalSamples = 0;
        for (int j = 0; j < numClasses; j++) {
            totalSamples += confusionMatrix[i][j];
        }
        float classAccuracy = static_cast<float>(confusionMatrix[i][i]) / totalSamples;
        std::cout << "Class " << i << ": " << classAccuracy * 100.0f << "%" << std::endl;
    }
    
    // Calculate and print training time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;
    
    // Print best validation accuracy
    std::cout << "Best validation accuracy: " << bestValidationAccuracy * 100.0f << "%" << std::endl;
    
    return 0;
} 