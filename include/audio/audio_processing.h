#ifndef ZYRAA_AUDIO_PROCESSING_H
#define ZYRAA_AUDIO_PROCESSING_H

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace zyraai{
    class AudioProcessing {
    public:
        static std::vector<std::vector<float>> extractMFCC(const std::string& audioFilePath);

    private:
        static std::vector<float> hammingWindow(int N);
        static Eigen::MatrixXf melFilterBank(int numFilters, int fftSize, float sampleRate);
        static std::vector<float> dct(const std::vector<float>& input);
    };
}

#endif