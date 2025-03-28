/**
 * @file layer.cpp
 * @brief Implementation of the Layer base class
 * @author ZyraAI Team
 */

#include "model/layer.h"
#include <stdexcept>

namespace zyraai {

Layer::Layer(const std::string &name, int inputSize, int outputSize)
    : name_(name), inputSize_(inputSize), outputSize_(outputSize),
      isTraining_(true) {
  
  // Validate input parameters
  if (inputSize <= 0) {
    throw std::invalid_argument("Layer: inputSize must be positive, got " + 
                               std::to_string(inputSize));
  }
  
  if (outputSize <= 0) {
    throw std::invalid_argument("Layer: outputSize must be positive, got " + 
                               std::to_string(outputSize));
  }
  
  if (name.empty()) {
    throw std::invalid_argument("Layer: name cannot be empty");
  }
}

} // namespace zyraai