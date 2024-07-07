#include "speech/speech_to_text.h"
#include "text_to_speech.h"
#include <cassert>
#include <iostream>
#include <fstream>

void test_speech_to_text() {
    zyraai::SpeechToText speechToText;
    std::string audioFilePath = "/media/ziro/ZiroHCKR/ZyraAI/tests/84-121123-0001.wav";  // Use absolute path
    auto text = speechToText.convertSpeechToText(audioFilePath);
    assert(!text.empty());

    std::string outputTextFilePath = "output_text.txt";
    speechToText.saveTextToFile(text, outputTextFilePath);

    std::ifstream textFile(outputTextFilePath);
    assert(textFile.is_open() && "Failed to open output text file");
    std::string line;
    bool hasTextContent = false;
    while (std::getline(textFile, line)) {
        if (!line.empty()) {
            hasTextContent = true;
            break;
        }
    }
    textFile.close();
    assert(hasTextContent && "Output text file is empty");

    std::cout << "SpeechToText test passed!" << std::endl;
}

void test_text_to_speech() {
    zyraai::TextToSpeech textToSpeech;
    std::string text = "This is a placeholder transcription.";
    std::string outputAudioFilePath = "output_audio.wav";
    textToSpeech.convertTextToSpeech(text, outputAudioFilePath);

    std::ifstream audioFile(outputAudioFilePath);
    assert(audioFile.is_open() && "Failed to open output audio file");
    bool hasAudioContent = false;
    std::string line;
    while (std::getline(audioFile, line)) {
        if (!line.empty()) {
            hasAudioContent = true;
            break;
        }
    }
    audioFile.close();
    assert(hasAudioContent && "Output audio file is empty");

    std::cout << "TextToSpeech test passed!" << std::endl;
}

int main() {
    test_speech_to_text();
    test_text_to_speech();
    return 0;
}
