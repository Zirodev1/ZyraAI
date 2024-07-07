// src/speech/speech_to_text.cpp

#include "speech/speech_to_text.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace zyraai{
    SpeechToText::SpeechToText() {
        std::cout << "SpeechToText Initialized" << std::endl;
    }

    SpeechToText::~SpeechToText() {
        std::cout << "SpeechToText Destroyed" << std::endl;
    }

    std::string SpeechToText::convertSpeechToText(const std::string& audioFilePath) {
        // Placeholder for actual speech to text logic
        std::cout << "Converting speech to text from " << audioFilePath << std::endl;
        auto features = AudioProcessing::extractMFCC(audioFilePath);
        return model.pridict(features);
    }

    void SpeechToText::saveTextToFile(const std::string& text, const std::string& outputFilePath) {
        std::ofstream file(outputFilePath);
        if(!file.is_open()){
            throw std::runtime_error("Failed to open output file: " + outputFilePath);
        }

        file << text;
        file.close();
        
        std::cout << "Saved text to " << outputFilePath << std::endl;
    }
}