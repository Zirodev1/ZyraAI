#include "audio/audio_processing.h"
#include "speech/speech_to_text.h"
#include "stt/stt_model.h"
#include "tts/text_to_speech.h"
#include <cassert>
#include <fstream>
#include <iostream>

using namespace ::std;

void test_speech_to_text() {
  zyraai::SpeechToText speechToText;
  ::std::string audioFilePath = "../tests/84-121123-0001.wav";
  auto features = zyraai::AudioProcessing::extractMFCC(audioFilePath);

  zyraai::STTModel model;

  // Create synthetic training data and labels for demonstration purposes
  ::std::vector<::std::vector<::std::vector<float>>> trainingData;
  ::std::vector<::std::string> labels;

  for (int i = 0; i < 10; ++i) {
    ::std::vector<::std::vector<float>> syntheticFeatures = features;
    trainingData.push_back(syntheticFeatures);
    labels.push_back("This is a synthetic transcription " +
                     ::std::to_string(i));
  }

  model.train(trainingData, labels);

  auto text = model.predict(features);
  assert(!text.empty());

  ::std::string outputTextFilePath = "output_text.txt";
  speechToText.saveTextToFile(text, outputTextFilePath);

  ::std::ifstream textFile(outputTextFilePath);
  assert(textFile.is_open() && "Failed to open output text file");
  ::std::string line;
  bool hasTextContent = false;
  while (::std::getline(textFile, line)) {
    if (!line.empty()) {
      hasTextContent = true;
      break;
    }
  }
  textFile.close();
  assert(hasTextContent && "Output text file is empty");

  cout << "SpeechToText test passed!" << endl;
}

void test_text_to_speech() {
  zyraai::TextToSpeech textToSpeech;
  ::std::string text = "This is a placeholder transcription.";
  ::std::string outputAudioFilePath = "output_audio.wav";
  textToSpeech.convertTextToSpeech(text, outputAudioFilePath);

  ::std::ifstream audioFile(outputAudioFilePath);
  assert(audioFile.is_open() && "Failed to open output audio file");
  bool hasAudioContent = false;
  ::std::string line;
  while (::std::getline(audioFile, line)) {
    if (!line.empty()) {
      hasAudioContent = true;
      break;
    }
  }
  audioFile.close();
  assert(hasAudioContent && "Output audio file is empty");

  cout << "TextToSpeech test passed!" << endl;
}

int main() {
  test_speech_to_text();
  test_text_to_speech();
  return 0;
}
