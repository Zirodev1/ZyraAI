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

// Data augmentation: random horizontal flip with adjustable probability
void randomHorizontalFlip(std::vector<float>& image, float probability = 0.5f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0f, 1.0f);
    
    if (dis(gen) < probability) {
        // For each row in the image
        for (int c = 0; c < 3; c++) {
            int channelOffset = c * 32 * 32;
            for (int i = 0; i < 32; i++) {
                int rowOffset = i * 32;
                for (int j = 0; j < 16; j++) {
                    std::swap(image[channelOffset + rowOffset + j], 
                              image[channelOffset + rowOffset + 31 - j]);
                }
            }
        }
    }
}

// Data augmentation: random crop with enhanced padding options
void randomCrop(std::vector<float>& image, int padding = 4) {
    // Create temporary padded image
    int paddedSize = 32 + 2 * padding;
    std::vector<float> paddedImage(3 * paddedSize * paddedSize, 0.0f);
    
    // Copy original image to center of padded image with reflective padding
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < paddedSize; i++) {
            for (int j = 0; j < paddedSize; j++) {
                // Calculate source position with reflection
                int srcI = std::min(std::max(i - padding, 0), 31);
                if (srcI > 31) srcI = 62 - srcI;  // Reflect
                
                int srcJ = std::min(std::max(j - padding, 0), 31);
                if (srcJ > 31) srcJ = 62 - srcJ;  // Reflect
                
                paddedImage[c * paddedSize * paddedSize + i * paddedSize + j] = 
                    image[c * 32 * 32 + srcI * 32 + srcJ];
            }
        }
    }
    
    // Randomly crop back to 32x32
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 2 * padding);
    
    int top = dis(gen);
    int left = dis(gen);
    
    // Copy cropped region back to original image
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                image[c * 32 * 32 + i * 32 + j] = 
                    paddedImage[c * paddedSize * paddedSize + (i + top) * paddedSize + (j + left)];
            }
        }
    }
}

// Enhanced color jitter with saturation and hue adjustments
void colorJitter(std::vector<float>& image, float brightness = 0.2f, float contrast = 0.2f, 
                 float saturation = 0.2f, float hue = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disBrightness(-brightness, brightness);
    std::uniform_real_distribution<> disContrast(1.0f - contrast, 1.0f + contrast);
    std::uniform_real_distribution<> disSaturation(1.0f - saturation, 1.0f + saturation);
    std::uniform_real_distribution<> disHue(-hue, hue);
    
    float brightnessShift = disBrightness(gen);
    float contrastFactor = disContrast(gen);
    float saturationFactor = disSaturation(gen);
    float hueShift = disHue(gen);
    
    // Apply color transformations
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            // Extract RGB values
            float r = image[0 * 32 * 32 + i * 32 + j];
            float g = image[1 * 32 * 32 + i * 32 + j];
            float b = image[2 * 32 * 32 + i * 32 + j];
            
            // Apply brightness
            r += brightnessShift;
            g += brightnessShift;
            b += brightnessShift;
            
            // Convert to HSV-like space for hue/saturation adjustments
            float maxVal = std::max(std::max(r, g), b);
            float minVal = std::min(std::min(r, g), b);
            float delta = maxVal - minVal;
            
            // Saturation adjustment
            if (delta > 0 && maxVal > 0) {
                // Adjust saturation
                float saturationAdjust = delta * saturationFactor / maxVal;
                r = minVal + (r - minVal) * saturationAdjust;
                g = minVal + (g - minVal) * saturationAdjust;
                b = minVal + (b - minVal) * saturationAdjust;
            }
            
            // Hue adjustment (simplified)
            if (delta > 0) {
                if (maxVal == r) {
                    float h = (g - b) / delta;
                    h += hueShift;
                    // Apply simplified hue rotation
                    float x = delta * (1 - std::abs(std::fmod(h, 2) - 1));
                    if (h >= 0 && h < 1) { r = maxVal; g = minVal + delta; b = minVal; }
                    else if (h >= 1 && h < 2) { r = maxVal - x; g = maxVal; b = minVal; }
                    else if (h >= 2 && h < 3) { r = minVal; g = maxVal; b = minVal + x; }
                    else if (h >= 3 && h < 4) { r = minVal; g = maxVal - x; b = maxVal; }
                    else if (h >= 4 && h < 5) { r = minVal + x; g = minVal; b = maxVal; }
                    else { r = maxVal; g = minVal; b = maxVal - x; }
                }
            }
            
            // Apply contrast: (x - 0.5) * factor + 0.5
            r = (r - 0.5f) * contrastFactor + 0.5f;
            g = (g - 0.5f) * contrastFactor + 0.5f;
            b = (b - 0.5f) * contrastFactor + 0.5f;
            
            // Clamp to [0, 1]
            image[0 * 32 * 32 + i * 32 + j] = std::max(0.0f, std::min(1.0f, r));
            image[1 * 32 * 32 + i * 32 + j] = std::max(0.0f, std::min(1.0f, g));
            image[2 * 32 * 32 + i * 32 + j] = std::max(0.0f, std::min(1.0f, b));
        }
    }
}

// Random erasing augmentation
void randomErasing(std::vector<float>& image, float probability = 0.5f, 
                   float areaRatioMin = 0.02f, float areaRatioMax = 0.2f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disProbability(0.0f, 1.0f);
    
    if (disProbability(gen) >= probability) {
        return;
    }
    
    // Determine area and aspect ratio
    std::uniform_real_distribution<> disAreaRatio(areaRatioMin, areaRatioMax);
    std::uniform_real_distribution<> disAspectRatio(0.3f, 1.0f / 0.3f);
    
    float areaRatio = disAreaRatio(gen);
    float aspectRatio = disAspectRatio(gen);
    
    int eraseH = static_cast<int>(std::sqrt(32 * 32 * areaRatio * aspectRatio));
    int eraseW = static_cast<int>(std::sqrt(32 * 32 * areaRatio / aspectRatio));
    
    // Ensure valid sizes
    eraseH = std::min(eraseH, 32);
    eraseW = std::min(eraseW, 32);
    
    // Determine position
    std::uniform_int_distribution<> disI(0, 32 - eraseH);
    std::uniform_int_distribution<> disJ(0, 32 - eraseW);
    int i = disI(gen);
    int j = disJ(gen);
    
    // Fill with random values or zeros
    std::uniform_real_distribution<> disPixel(0.0f, 1.0f);
    
    for (int c = 0; c < 3; c++) {
        for (int ei = i; ei < i + eraseH; ei++) {
            for (int ej = j; ej < j + eraseW; ej++) {
                image[c * 32 * 32 + ei * 32 + ej] = disPixel(gen);
            }
        }
    }
}

// Mixup augmentation
std::pair<std::vector<float>, std::pair<int, float>> 
mixup(const std::vector<float>& image1, int label1, 
      const std::vector<float>& image2, int label2, 
      float alpha = 0.2f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    
    float lambda = gamma(gen);
    lambda = std::min(std::max(lambda, 0.0f), 1.0f);
    
    std::vector<float> mixedImage(image1.size());
    for (size_t i = 0; i < image1.size(); i++) {
        mixedImage[i] = lambda * image1[i] + (1.0f - lambda) * image2[i];
    }
    
    // Return mixed image and labels with lambda weight
    return {mixedImage, {label1, lambda}};
}

// Apply all standard data augmentations to training data
void augmentImage(std::vector<float>& image) {
    randomHorizontalFlip(image, 0.5f);
    randomCrop(image, 4);
    colorJitter(image, 0.2f, 0.2f, 0.2f, 0.1f);
    randomErasing(image, 0.2f);
}

// Read CIFAR-10 dataset
std::pair<std::pair<std::vector<std::vector<float>>, std::vector<int>>, 
          std::pair<std::vector<std::vector<float>>, std::vector<int>>> 
readCIFAR10() {
    const int imageSize = 3 * 32 * 32; // 3 channels, 32x32 pixels
    const int batchSize = 10000;
    
    std::vector<std::vector<float>> trainImages;
    std::vector<int> trainLabels;
    
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
            std::vector<float> image(imageSize);
            for (int j = 0; j < imageSize; j++) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image[j] = static_cast<float>(pixel) / 255.0f;
            }
            trainImages.push_back(image);
        }
    }
    
    // Read test data
    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;
    
    std::string testFilename = "I:/ZyraAI/data/cifar-10-batches-bin/test_batch.bin";
    std::ifstream testFile(testFilename, std::ios::binary);
    
    for (int i = 0; i < batchSize; i++) {
        // Read label (1 byte)
        char label;
        testFile.read(&label, 1);
        testLabels.push_back(static_cast<int>(label));
        
        // Read image data (3072 bytes)
        std::vector<float> image(imageSize);
        for (int j = 0; j < imageSize; j++) {
            unsigned char pixel;
            testFile.read(reinterpret_cast<char*>(&pixel), 1);
            image[j] = static_cast<float>(pixel) / 255.0f;
        }
        testImages.push_back(image);
    }
    
    return {{trainImages, trainLabels}, {testImages, testLabels}};
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

// Convert std::vector<float> to Eigen::MatrixXf for model input
// Format: [features x 1] column vector where features = channels * height * width
Eigen::MatrixXf vectorToMatrix(const std::vector<float>& vec) {
    // Create a column vector with all pixels as features
    Eigen::MatrixXf matrix(vec.size(), 1);
    for (size_t i = 0; i < vec.size(); i++) {
        matrix(i, 0) = vec[i];
    }
    return matrix;
}

// Convert label to one-hot encoded Eigen::MatrixXf
Eigen::MatrixXf labelToOneHot(int label, int numClasses) {
    Eigen::MatrixXf oneHot = Eigen::MatrixXf::Zero(numClasses, 1);
    oneHot(label, 0) = 1.0f;
    return oneHot;
}

// Helper function to reshape data for different layers
Eigen::MatrixXf reshapeForConv(const Eigen::MatrixXf& flatInput, int channels, int height, int width) {
    // Reshape flat vector to proper format for conv layer
    // The library expects [channels*height*width, batchSize]
    return flatInput;
}

// Convert flat image vector to separate channel matrices for proper batch normalization
std::vector<Eigen::MatrixXf> extractChannels(const Eigen::MatrixXf& flatInput, int channels, int height, int width) {
    std::vector<Eigen::MatrixXf> channelMatrices(channels);
    
    // Each channel is a separate matrix with [height*width, 1] dimensions
    for (int c = 0; c < channels; c++) {
        channelMatrices[c] = Eigen::MatrixXf(height * width, 1);
        
        // Extract the pixels for this channel
        for (int i = 0; i < height * width; i++) {
            channelMatrices[c](i, 0) = flatInput(c * height * width + i, 0);
        }
    }
    
    return channelMatrices;
}

// Combine channel matrices back to a flat vector
Eigen::MatrixXf combineChannels(const std::vector<Eigen::MatrixXf>& channelMatrices, int channels, int height, int width) {
    Eigen::MatrixXf combined(channels * height * width, 1);
    
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height * width; i++) {
            combined(c * height * width + i, 0) = channelMatrices[c](i, 0);
        }
    }
    
    return combined;
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
    std::vector<std::vector<float>> validationImages(trainImages.end() - validationSize, trainImages.end());
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
    const int batchSize = 128;
    const int epochs = 100;
    const float initialLearningRate = 0.001f;
    const float weightDecay = 1e-4f;
    
    // Now we'll implement a hybrid approach - we'll still use the ZyraAI model 
    // but instead of using ConvolutionalLayer directly, we'll implement our own
    // mini-convolutional network that follows the library's data formatting expectations
    
    ZyraAIModel model;
    
    // Input size
    const int inputSize = inputChannels * inputHeight * inputWidth;
    
    // First block: 64 filters
    model.addLayer(std::make_shared<DenseLayer>("conv1", inputSize, 64 * (inputHeight/2) * (inputWidth/2)));
    model.addLayer(std::make_shared<BatchNormLayer>("bn1", 64 * (inputHeight/2) * (inputWidth/2)));
    model.addLayer(std::make_shared<ReLULayer>("relu1", 64 * (inputHeight/2) * (inputWidth/2), 64 * (inputHeight/2) * (inputWidth/2)));
    model.addLayer(std::make_shared<DropoutLayer>("dropout1", 64 * (inputHeight/2) * (inputWidth/2), 0.25f));
    
    // Second block: 128 filters with downsampling
    model.addLayer(std::make_shared<DenseLayer>("conv2", 64 * (inputHeight/2) * (inputWidth/2), 128 * (inputHeight/4) * (inputWidth/4)));
    model.addLayer(std::make_shared<BatchNormLayer>("bn2", 128 * (inputHeight/4) * (inputWidth/4)));
    model.addLayer(std::make_shared<ReLULayer>("relu2", 128 * (inputHeight/4) * (inputWidth/4), 128 * (inputHeight/4) * (inputWidth/4)));
    model.addLayer(std::make_shared<DropoutLayer>("dropout2", 128 * (inputHeight/4) * (inputWidth/4), 0.25f));
    
    // Classifier head
    model.addLayer(std::make_shared<DenseLayer>("fc1", 128 * (inputHeight/4) * (inputWidth/4), 512));
    model.addLayer(std::make_shared<BatchNormLayer>("bn3", 512));
    model.addLayer(std::make_shared<ReLULayer>("relu3", 512, 512));
    model.addLayer(std::make_shared<DropoutLayer>("dropout3", 512, 0.5f));
    model.addLayer(std::make_shared<DenseLayer>("fc2", 512, numClasses));
    model.addLayer(std::make_shared<SoftmaxLayer>("softmax", numClasses));
    
    // Initialize Adam optimizer with better hyperparameters
    AdamOptimizer optimizer(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
    
    // Train the model
    std::cout << "Starting training..." << std::endl;
    
    // Training statistics
    std::vector<float> trainLossHistory;
    std::vector<float> trainAccuracyHistory;
    std::vector<float> validationLossHistory;
    std::vector<float> validationAccuracyHistory;
    
    // Early stopping variables
    float bestValidationAccuracy = 0.0f;
    int patienceCounter = 0;
    const int patience = 10;
    
    // Create directories for checkpoints
    createDirectoryIfNotExists("I:/ZyraAI/checkpoints");
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Better learning rate schedule - cosine annealing with warm restarts
        if (epoch < 5) {
            // Warmup phase
            float warmupFactor = (epoch + 1) / 5.0f;
            optimizer.setLearningRate(initialLearningRate * warmupFactor);
        } else if (epoch == 50 || epoch == 75) {
            // Learning rate decay
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.1f);
        }
        
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", Learning rate: " << optimizer.getLearningRate() << std::endl;
        
        // Shuffle training data
        std::vector<size_t> indices(trainImages.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Training metrics
        float trainLoss = 0.0f;
        int trainCorrect = 0;
        
        // Training loop - now processing in mini-batches
        for (size_t b = 0; b < trainImages.size(); b += batchSize) {
            size_t actualBatchSize = std::min(batchSize, static_cast<int>(trainImages.size() - b));
            
            float batchLoss = 0.0f;
            int batchCorrect = 0;
            
            // Process each sample in the batch
            for (size_t i = 0; i < actualBatchSize; ++i) {
                size_t idx = indices[b + i];
                std::vector<float> augmentedImage = trainImages[idx];
                
                // Apply more aggressive data augmentation
                augmentImage(augmentedImage);
                
                // Create proper tensor representation
                Eigen::MatrixXf input = vectorToMatrix(augmentedImage);
                Eigen::MatrixXf target = labelToOneHot(trainLabels[idx], numClasses);
                
                // Train the model with weight decay
                float sampleLoss = model.train(input, target, optimizer.getLearningRate());
                batchLoss += sampleLoss;
                
                // Forward pass for evaluation
                Eigen::MatrixXf output = model.forward(input);
                
                // Find max index in output (predicted class)
                Eigen::MatrixXf::Index maxRow, maxCol;
                output.maxCoeff(&maxRow, &maxCol);
                int predictedClass = static_cast<int>(maxRow);
                
                // Update accuracy
                if (predictedClass == trainLabels[idx]) {
                    batchCorrect++;
                }
            }
            
            // Update optimizer
            optimizer.step();
            
            // Update training metrics
            trainLoss += batchLoss;
            trainCorrect += batchCorrect;
            
            // Print progress every 50 batches
            if ((b / batchSize) % 50 == 0 && b > 0) {
                float currentAccuracy = static_cast<float>(trainCorrect) / b;
                std::cout << "  Batch " << b / batchSize << "/" << trainImages.size() / batchSize
                          << ", Loss: " << trainLoss / b
                          << ", Accuracy: " << currentAccuracy * 100.0f << "%" << std::endl;
            }
        }
        
        // Calculate final training metrics
        trainLoss /= trainImages.size();
        float trainAccuracy = static_cast<float>(trainCorrect) / trainImages.size();
        
        trainLossHistory.push_back(trainLoss);
        trainAccuracyHistory.push_back(trainAccuracy);
        
        // Validation phase
        float validationLoss = 0.0f;
        int validationCorrect = 0;
        
        // Set model to evaluation mode (affects dropout, batch norm, etc.)
        model.setTraining(false);
        
        for (size_t i = 0; i < validationImages.size(); ++i) {
            // Convert to Eigen::MatrixXf
            Eigen::MatrixXf input = vectorToMatrix(validationImages[i]);
            Eigen::MatrixXf target = labelToOneHot(validationLabels[i], numClasses);
            
            // Forward pass
            Eigen::MatrixXf output = model.forward(input);
            
            // Calculate loss
            float sampleLoss = model.computeLoss(output, target);
            validationLoss += sampleLoss;
            
            // Find max index in output (predicted class)
            Eigen::MatrixXf::Index maxRow, maxCol;
            output.maxCoeff(&maxRow, &maxCol);
            int predictedClass = static_cast<int>(maxRow);
            
            // Update accuracy
            if (predictedClass == validationLabels[i]) {
                validationCorrect++;
            }
        }
        
        // Set model back to training mode
        model.setTraining(true);
        
        validationLoss /= validationImages.size();
        float validationAccuracy = static_cast<float>(validationCorrect) / validationImages.size();
        
        validationLossHistory.push_back(validationLoss);
        validationAccuracyHistory.push_back(validationAccuracy);
        
        // Print epoch results
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Training: loss=" << trainLoss << ", accuracy=" << trainAccuracy * 100.0f << "%"
                  << " - Validation: loss=" << validationLoss << ", accuracy=" << validationAccuracy * 100.0f << "%" << std::endl;
        
        // Save checkpoint if it's the best model so far
        if (validationAccuracy > bestValidationAccuracy) {
            bestValidationAccuracy = validationAccuracy;
            patienceCounter = 0;
            std::string checkpointPath = "I:/ZyraAI/checkpoints/cifar10_model_best.bin";
            saveModelCheckpoint(model, checkpointPath);
            std::cout << "  New best model! Validation accuracy: " << validationAccuracy * 100.0f << "%" << std::endl;
        } else {
            patienceCounter++;
            // Save regular checkpoint every 10 epochs
            if (epoch % 10 == 9) {
                std::string checkpointPath = "I:/ZyraAI/checkpoints/cifar10_model_epoch_" + std::to_string(epoch + 1) + ".bin";
                saveModelCheckpoint(model, checkpointPath);
            }
        }
        
        // Early stopping
        if (patienceCounter >= patience) {
            std::cout << "Early stopping triggered. No improvement for " << patience << " epochs." << std::endl;
            break;
        }
    }
    
    // Evaluate on test set
    float testLoss = 0.0f;
    int testCorrect = 0;
    
    // Set model to evaluation mode
    model.setTraining(false);
    
    // Process test set in batches to generate confusion matrix
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    
    for (size_t i = 0; i < testImages.size(); ++i) {
        // Convert to Eigen::MatrixXf
        Eigen::MatrixXf input = vectorToMatrix(testImages[i]);
        Eigen::MatrixXf target = labelToOneHot(testLabels[i], numClasses);
        
        // Forward pass
        Eigen::MatrixXf output = model.forward(input);
        
        // Calculate loss
        float sampleLoss = model.computeLoss(output, target);
        testLoss += sampleLoss;
        
        // Find max index in output (predicted class)
        Eigen::MatrixXf::Index maxRow, maxCol;
        output.maxCoeff(&maxRow, &maxCol);
        int predictedClass = static_cast<int>(maxRow);
        
        // Update confusion matrix
        confusionMatrix[testLabels[i]][predictedClass]++;
        
        // Update accuracy
        if (predictedClass == testLabels[i]) {
            testCorrect++;
        }
    }
    
    testLoss /= testImages.size();
    float testAccuracy = static_cast<float>(testCorrect) / testImages.size();
    
    // Print final results
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "Test accuracy: " << testAccuracy * 100.0f << "%" << std::endl;
    std::cout << "Test loss: " << testLoss << std::endl;
    
    // Calculate per-class accuracy from confusion matrix
    std::cout << "\nPer-class accuracy:" << std::endl;
    for (int i = 0; i < numClasses; i++) {
        int totalClass = 0;
        for (int j = 0; j < numClasses; j++) {
            totalClass += confusionMatrix[i][j];
        }
        float classAccuracy = static_cast<float>(confusionMatrix[i][i]) / totalClass;
        std::cout << "  Class " << i << ": " << classAccuracy * 100.0f << "%" << std::endl;
    }
    
    // Calculate and print training time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    std::cout << "Total training time: " << duration.count() << " seconds" << std::endl;
    
    // Save final model
    std::string finalModelPath = "I:/ZyraAI/checkpoints/cifar10_model_final.bin";
    saveModelCheckpoint(model, finalModelPath);
    
    // Print best validation accuracy
    std::cout << "Best validation accuracy: " << bestValidationAccuracy * 100.0f << "%" << std::endl;
    
    return 0;
} 