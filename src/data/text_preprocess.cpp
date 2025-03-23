#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

void preprocessText(const std::string &inputDir, const std::string &outputFile) {
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return;
    }

    // Iterate over all text files in the input directory
    for (const auto &entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream inFile(entry.path());
            if (!inFile.is_open()) {
                std::cerr << "Failed to open input file: " << entry.path() << std::endl;
                continue;
            }

            std::string line;
            while (std::getline(inFile, line)) {
                // Convert to lowercase
                std::transform(line.begin(), line.end(), line.begin(), ::tolower);

                // Remove special characters (excluding spaces)
                line = std::regex_replace(line, std::regex("[^a-zA-Z0-9\\s]"), "");

                // Write the cleaned line back to the output file with a newline
                outFile << line << "\n";
            }
            inFile.close();
        }
    }
    outFile.close();
    std::cout << "Preprocessing completed successfully. Output saved to " << outputFile << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: text_preprocess <input_directory> <output_file>" << std::endl;
        return 1;
    }

    std::string inputDir = argv[1];
    std::string outputFile = argv[2];

    preprocessText(inputDir, outputFile);

    return 0;
}
