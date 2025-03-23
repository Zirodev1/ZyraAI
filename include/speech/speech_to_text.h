// include/speech/speech_to_text.h

#ifndef ZYRAAI_SPEECH_TO_TEXT_H
#define ZYRAAI_SPEECH_TO_TEXT_H

#include <string>
#include "audio/audio_processing.h"
#include "stt/stt_model.h"

namespace zyraai {
    class SpeechToText {
    public:
        SpeechToText();
        ~SpeechToText();

        std::string convertSpeechToText(const std::string& audioFilePath);
        void saveTextToFile(const std::string& text, const std::string& outputFilePath);

    private:
        STTModel model;
    };
}

#endif 