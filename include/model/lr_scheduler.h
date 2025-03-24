#pragma once

#include <cmath>
#include <vector>

namespace zyraai {

class LRScheduler {
public:
  virtual float getLearningRate(int epoch) = 0;
  virtual ~LRScheduler() = default;
};

class CosineAnnealingScheduler : public LRScheduler {
public:
  CosineAnnealingScheduler(float initialLR, float minLR, int totalEpochs)
      : initialLR_(initialLR), minLR_(minLR), totalEpochs_(totalEpochs) {}

  float getLearningRate(int epoch) override {
    if (epoch >= totalEpochs_) {
      return minLR_;
    }
    return minLR_ + 0.5f * (initialLR_ - minLR_) *
                        (1.0f + std::cos(M_PI * epoch / totalEpochs_));
  }

private:
  float initialLR_;
  float minLR_;
  int totalEpochs_;
};

class StepScheduler : public LRScheduler {
public:
  StepScheduler(float initialLR, float gamma, int stepSize)
      : initialLR_(initialLR), gamma_(gamma), stepSize_(stepSize) {}

  float getLearningRate(int epoch) override {
    return initialLR_ * std::pow(gamma_, epoch / stepSize_);
  }

private:
  float initialLR_;
  float gamma_;
  int stepSize_;
};

class MultiStepScheduler : public LRScheduler {
public:
  MultiStepScheduler(float initialLR, float gamma,
                     const ::std::vector<int> &milestones)
      : initialLR_(initialLR), gamma_(gamma), milestones_(milestones) {}

  float getLearningRate(int epoch) override {
    int n = 0;
    for (int milestone : milestones_) {
      if (epoch >= milestone) {
        n++;
      }
    }
    return initialLR_ * std::pow(gamma_, n);
  }

private:
  float initialLR_;
  float gamma_;
  ::std::vector<int> milestones_;
};

class WarmupCosineScheduler : public LRScheduler {
public:
  WarmupCosineScheduler(float initialLR, float minLR, int warmupEpochs,
                        int totalEpochs)
      : initialLR_(initialLR), minLR_(minLR), warmupEpochs_(warmupEpochs),
        totalEpochs_(totalEpochs) {}

  float getLearningRate(int epoch) override {
    if (epoch < warmupEpochs_) {
      // Linear warmup
      return minLR_ + (initialLR_ - minLR_) * (float)epoch / warmupEpochs_;
    } else {
      // Cosine annealing
      int adjustedEpoch = epoch - warmupEpochs_;
      int adjustedTotal = totalEpochs_ - warmupEpochs_;
      return minLR_ +
             0.5f * (initialLR_ - minLR_) *
                 (1.0f + std::cos(M_PI * adjustedEpoch / adjustedTotal));
    }
  }

private:
  float initialLR_;
  float minLR_;
  int warmupEpochs_;
  int totalEpochs_;
};

} // namespace zyraai