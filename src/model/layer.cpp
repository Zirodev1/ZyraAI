#include "model/layer.h"

namespace zyraai {

Layer::Layer(const std::string &name, int inputSize, int outputSize)
    : name_(name), inputSize_(inputSize), outputSize_(outputSize),
      isTraining_(true) {}

} // namespace zyraai