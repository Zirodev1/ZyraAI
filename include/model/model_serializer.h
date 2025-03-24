#pragma once

#include "model/zyraAI_model.h"
#include <fstream>
#include <iostream>
#include <string>

namespace zyraai {

class ModelSerializer {
public:
  static bool saveModel(const ZyraAIModel &model,
                        const ::std::string &filePath) {
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    // Write number of layers
    const auto &layers = model.getLayers();
    size_t numLayers = layers.size();
    file.write(reinterpret_cast<const char *>(&numLayers), sizeof(numLayers));

    // For each layer
    for (const auto &layer : layers) {
      // Write layer name
      ::std::string name = layer->getName();
      size_t nameLength = name.length();
      file.write(reinterpret_cast<const char *>(&nameLength),
                 sizeof(nameLength));
      file.write(name.c_str(), nameLength);

      // Write parameters
      auto params = layer->getParameters();
      size_t numParams = params.size();
      file.write(reinterpret_cast<const char *>(&numParams), sizeof(numParams));

      for (const auto &param : params) {
        // Write matrix dimensions
        int rows = param.rows();
        int cols = param.cols();
        file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

        // Write matrix data
        file.write(reinterpret_cast<const char *>(param.data()),
                   rows * cols * sizeof(float));
      }
    }

    return true;
  }

  static bool loadModel(ZyraAIModel &model, const ::std::string &filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
      return false;
    }

    // Read number of layers
    size_t numLayers;
    file.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

    const auto &layers = model.getLayers();
    if (layers.size() != numLayers) {
      std::cerr << "Model structure mismatch: expected " << numLayers
                << " layers, but got " << layers.size() << std::endl;
      return false; // Model structure mismatch
    }

    // For each layer
    for (size_t i = 0; i < numLayers; ++i) {
      // Read layer name
      size_t nameLength;
      file.read(reinterpret_cast<char *>(&nameLength), sizeof(nameLength));
      ::std::string name(nameLength, ' ');
      file.read(&name[0], nameLength);

      if (name != layers[i]->getName()) {
        std::cerr << "Layer name mismatch at index " << i << ": expected "
                  << name << ", but got " << layers[i]->getName() << std::endl;
        return false; // Layer name mismatch
      }

      // Read parameters
      size_t numParams;
      file.read(reinterpret_cast<char *>(&numParams), sizeof(numParams));

      auto layerParams = layers[i]->getParameters();
      if (layerParams.size() != numParams) {
        std::cerr << "Parameter count mismatch for layer " << name
                  << ": expected " << numParams << ", but got "
                  << layerParams.size() << std::endl;
        return false; // Parameter count mismatch
      }

      for (size_t j = 0; j < numParams; ++j) {
        // Read matrix dimensions
        int rows, cols;
        file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        if (rows != layerParams[j].rows() || cols != layerParams[j].cols()) {
          std::cerr << "Parameter dimension mismatch for layer " << name
                    << " param " << j << ": expected (" << rows << "," << cols
                    << "), but got (" << layerParams[j].rows() << ","
                    << layerParams[j].cols() << ")" << std::endl;
          return false; // Parameter dimension mismatch
        }

        // Create buffer for parameter data
        Eigen::MatrixXf param(rows, cols);
        file.read(reinterpret_cast<char *>(param.data()),
                  rows * cols * sizeof(float));

        // Update parameter in layer
        layers[i]->updateParameter(j, layerParams[j] - param);
      }
    }

    return true;
  }
};

} // namespace zyraai