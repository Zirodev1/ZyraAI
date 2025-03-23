#ifndef ZYRAAI_PREPROCESS_DATA_H
#define ZYRAAI_PREPROCESS_DATA_H

#include <string>
#include <vector>

namespace zyraai {
    class PreprocessData {
    public:
        // Preprocess audio data from inputDir and save processed data to outputDir
        static void preprocessAudio(const std::string& inputDir, const std::string& outputDir, int sampleRate = 16000);

        // Helper function to extract features like MFCC (Mel Frequency Cepstral Coefficients)
        static std::vector<std::vector<float>> extractFeatures(const std::string& audioFile, int sampleRate);
    };
}

#endif // ZYRAAI_PREPROCESS_DATA_H
