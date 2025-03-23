#ifndef ZYRAAI_STT_MODEL_H
#define ZYRAAI_STT_MODEL_H

#include <vector>
#include <string>

namespace zyraai {
    class STTModel {
    public:
        STTModel();
        ~STTModel();

        void train(const std::vector<std::vector<std::vector<float>>>& trainingData, const std::vector<std::string>& lables);
        std::string predict(const std::vector<std::vector<float>>& features);

    private:
        std::vector<std::vector<std::vector<float>>> trainingData;
        std::vector<std::string> lables;
    };
}

#endif
