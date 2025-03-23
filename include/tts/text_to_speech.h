#ifndef ZYRAAI_TEXT_TO_SPEECH_H
#define ZYRAAI_TEXT_TO_SPEECH_H

#include <string>
#include <vector>

namespace zyraai {
    class TextToSpeech {
    public:
        TextToSpeech();
        ~TextToSpeech();

        void convertTextToSpeech(const std::string& text, const std::string& outputAudioFilePath);

    private:
        std::vector<std::string> textToPhonemes(const std::string& text);
        void synthesizeWaveform(const std::vector<std::string>& phonemes, const std::string& outputAudioFilePath);
    };
}

#endif