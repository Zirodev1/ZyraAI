#include "audio/audio_processing.h"
#include <sndfile.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <eigen3/Eigen/Dense>
#include <complex>


namespace zyraai {
    std::vector<std::vector<float>> AudioProcessing::extractMFCC(const std::string& audioFilePath) {
        // Open the audio file using libsndfile
        SF_INFO sfInfo;
        SNDFILE* sndFile = sf_open(audioFilePath.c_str(), SFM_READ, &sfInfo);
        if (!sndFile) {
            throw std::runtime_error("Failed to open audio file: " + audioFilePath);
        }

        // Read the audio data
        std::vector<float> signal(sfInfo.frames * sfInfo.channels);
        sf_read_float(sndFile, signal.data(), sfInfo.frames * sfInfo.channels);
        sf_close(sndFile);

        // Ensure mono channel audio
        if (sfInfo.channels > 1) {
            throw std::runtime_error("Multi-channel audio not supported: " + audioFilePath);
        }

        // Parameters
        int sampleRate = sfInfo.samplerate;
        int numFilters = 26;
        int numCoeffs = 13;
        int frameSize = 512;
        int fftSize = 512;
        int hopSize = 256;

        // Pre-emphasis
        for (size_t i = signal.size() - 1; i > 0; --i) {
            signal[i] = signal[i] - 0.97f * signal[i - 1];
        }

        // Frame blocking and windowing
        std::vector<std::vector<float>> frames;
        for (size_t start = 0; start + frameSize < signal.size(); start += hopSize) {
            std::vector<float> frame(signal.begin() + start, signal.begin() + start + frameSize);
            auto window = hammingWindow(frameSize);
            for (size_t i = 0; i < frame.size(); ++i) {
                frame[i] *= window[i];
            }
            frames.push_back(frame);
        }

        // FFT and Mel filterbank
        Eigen::MatrixXf melFilter = melFilterBank(numFilters, fftSize, sampleRate);
        std::vector<std::vector<float>> mfccs;

        for (const auto& frame : frames) {
            // Perform FFT
            Eigen::VectorXf spectrum = Eigen::VectorXf::Zero(fftSize / 2 + 1);
            for (int k = 0; k <= fftSize / 2; ++k) {
                std::complex<float> sum(0.0, 0.0);
                for (size_t n = 0; n < frame.size(); ++n) {
                    float angle = 2.0f * M_PI * k * n / frame.size();
                    sum += std::complex<float>(frame[n] * std::cos(angle), -frame[n] * std::sin(angle));
                }
                spectrum[k] = std::abs(sum);
            }

            Eigen::VectorXf melEnergies = melFilter * spectrum;
            for (int i = 0; i < melEnergies.size(); ++i) {
                melEnergies[i] = std::log(melEnergies[i] + 1e-10f);
            }

            std::vector<float> mfcc = dct(std::vector<float>(melEnergies.data(), melEnergies.data() + melEnergies.size()));
            mfcc.resize(numCoeffs);
            mfccs.push_back(mfcc);
        }

        return mfccs;
    }

    std::vector<float> AudioProcessing::hammingWindow(int N) {
        std::vector<float> window(N);
        for (int n = 0; n < N; ++n) {
            window[n] = 0.54f - 0.46f * std::cos(2.0f * M_PI * n / (N - 1));
        }
        return window;
    }

    Eigen::MatrixXf AudioProcessing::melFilterBank(int numFilters, int fftSize, float sampleRate) {
        Eigen::MatrixXf filterBank = Eigen::MatrixXf::Zero(numFilters, fftSize / 2 + 1);

        auto hzToMel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
        auto melToHz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

        float melMin = hzToMel(0);
        float melMax = hzToMel(sampleRate / 2);
        float melStep = (melMax - melMin) / (numFilters + 1);

        std::vector<float> melCenters(numFilters + 2);
        for (int i = 0; i < melCenters.size(); ++i) {
            melCenters[i] = melToHz(melMin + i * melStep);
        }

        for (int i = 1; i <= numFilters; ++i) {
            for (int j = 0; j < fftSize / 2 + 1; ++j) {
                float freq = static_cast<float>(j) * sampleRate / fftSize;
                if (freq >= melCenters[i - 1] && freq <= melCenters[i]) {
                    filterBank(i - 1, j) = (freq - melCenters[i - 1]) / (melCenters[i] - melCenters[i - 1]);
                }
                if (freq >= melCenters[i] && freq <= melCenters[i + 1]) {
                    filterBank(i - 1, j) = (melCenters[i + 1] - freq) / (melCenters[i + 1] - melCenters[i]);
                }
            }
        }

        return filterBank;
    }

    std::vector<float> AudioProcessing::dct(const std::vector<float>& input) {
        int N = input.size();
        std::vector<float> output(N, 0.0f);
        for (int k = 0; k < N; ++k) {
            for (int n = 0; n < N; ++n) {
                output[k] += input[n] * std::cos(M_PI / N * (n + 0.5f) * k);
            }
        }
        return output;
    }
}