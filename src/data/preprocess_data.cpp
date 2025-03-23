#include "../include/data/preprocess_data.h"
#include "../include/audio/audio_processing.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sndfile.hh>

namespace fs = std::filesystem;

namespace zyraai {

void PreprocessData::preprocessAudio(const std::string &inputDir,
                                     const std::string &outputDir,
                                     int sampleRate) {
  std::string clipsDir = inputDir + "/clips";

  // Ensure the input directory exists
  if (!fs::exists(clipsDir)) {
    throw std::runtime_error("Input directory not found: " + clipsDir);
  }

  // Ensure the output directory exists
  if (!fs::exists(outputDir)) {
    std::cout << "Creating output directory: " << outputDir << std::endl;
    fs::create_directories(outputDir);
  }

  int totalFiles = 0;
  for (const auto &entry : fs::directory_iterator(clipsDir)) {
    if (entry.path().extension() == ".wav") {
      totalFiles++;
    }
  }

  if (totalFiles == 0) {
    std::cerr << "No .wav files found in the input directory: " << clipsDir
              << std::endl;
    return;
  }

  std::cout << "Total audio files to process: " << totalFiles << std::endl;

  int processedFiles = 0;

  // Iterate over all .wav files in the input directory
  for (const auto &entry : fs::directory_iterator(clipsDir)) {
    if (entry.path().extension() == ".wav") {
      std::string audioPath = entry.path().string();
      std::string outputPath =
          outputDir + "/" + entry.path().filename().string() + ".features";

      std::cout << "[" << (processedFiles + 1) << "/" << totalFiles
                << "] Processing audio file: " << audioPath << std::endl;

      try {
        // Extract features (MFCC or other features)
        auto features = PreprocessData::extractFeatures(audioPath, sampleRate);

        // Save the extracted features in the output directory (you can save as
        // text, CSV, etc.)
        std::ofstream outFile(outputPath);
        if (!outFile) {
          std::cerr << "Failed to open output file for writing: " << outputPath
                    << std::endl;
          continue;
        }

        std::cout << "Saving extracted features to: " << outputPath
                  << std::endl;

        for (const auto &frame : features) {
          for (float coef : frame) {
            outFile << coef << " ";
          }
          outFile << std::endl;
        }
        outFile.close();

        std::cout << "Successfully processed and saved file: " << outputPath
                  << std::endl;

        processedFiles++;
      } catch (const std::exception &e) {
        std::cerr << "Error processing file: " << audioPath << " - " << e.what()
                  << std::endl;
      }
    }
  }

  // Final summary message
  std::cout << "Preprocessing completed. " << processedFiles << " out of "
            << totalFiles << " files successfully processed." << std::endl;
}

std::vector<std::vector<float>>
PreprocessData::extractFeatures(const std::string &audioFile, int sampleRate) {
  // Open the audio file using libsndfile
  SndfileHandle file(audioFile);
  if (!file) {
    throw std::runtime_error("Failed to open audio file: " + audioFile);
  }

  // Read audio data into a buffer
  std::vector<float> audioData(file.frames() * file.channels());
  file.readf(audioData.data(), file.frames());

  // Placeholder logic for extracting features (use your actual feature
  // extraction logic)
  std::vector<std::vector<float>>
      features; // Frame-based MFCCs or other features

  // Example: This would be where MFCC extraction happens.
  // You can replace this with actual feature extraction.
  for (size_t i = 0; i < audioData.size();
       i += sampleRate) { // Process in chunks
    std::vector<float> frame(
        13, 0.0f); // Placeholder: 13 MFCC coefficients per frame
    features.push_back(frame);
  }

  return features;
}
} // namespace zyraai

// Add the main function to preprocess data
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: preprocess_data <input_directory> <output_directory>"
              << std::endl;
    return 1;
  }

  std::string inputDir = argv[1];
  std::string outputDir = argv[2];
  int sampleRate = 16000; // Default sample rate

  try {
    std::cout << "Starting preprocessing..." << std::endl;
    zyraai::PreprocessData::preprocessAudio(inputDir, outputDir, sampleRate);
    std::cout << "Preprocessing completed successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during preprocessing: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
