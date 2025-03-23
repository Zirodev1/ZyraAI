#include "stt/stt_model.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <map>
#include <algorithm>

namespace zyraai {
    STTModel::STTModel() {
        std::cout << "STTModel Initialized" << std::endl;
    }

    STTModel::~STTModel() {
        std::cout << "STTModel Destroyed" << std::endl;
    }

    void STTModel::train(const std::vector<std::vector<std::vector<float>>>& trainingData, const std::vector<std::string>& lables) {
        if(trainingData.empty() || lables.empty() || trainingData.size() != lables.size()){
            throw std::runtime_error("Invalid training data or labels");
        }

        this->trainingData = trainingData;
        this->lables = lables;
        std::cout << "Training STTModel..." << std::endl;
    }

    std::string STTModel::predict(const std::vector<std::vector<float>>& features) {
        if(features.empty()){
            throw std::runtime_error("Empty features for prediction");
        }

        // k-NM algorithm
        int k = 3; // Number of nearest neighbors
        std::vector<std::pair<float, std::string>> distances;

        for( size_t i = 0; i < trainingData.size(); ++i){
            float distance = 0.0;
            for(size_t j = 0; j < features.size() && j < trainingData[i].size(); ++j){
                for(size_t k = 0; k < features[j].size(); ++k){
                    distance += std::pow(features[j][k] - trainingData[i][j][k], 2);
                }
            }
            distance = std::sqrt(distance);
            distances.push_back(std::make_pair(distance, lables[i]));
        }

        std::sort(distances.begin(), distances.end());
        std::map<std::string, int> labelCount;
        for(int i = 0; i < k; ++i) {
            labelCount[distances[i].second]++;
        }

        std::string predictedLabel;
        int maxCount = 0;
        for(const auto& pair : labelCount){
            if(pair.second > maxCount){
                maxCount = pair.second;
                predictedLabel = pair.first;
            }
        }

        return predictedLabel;
    }
}