// src/data/data_handler.cpp

#include "data/data_handler.h"
#include <iostream>
#include <fstream>

namespace zyraai {
    DataHandler::DataHandler() {
        std::cout << "DataHandler Initialized" << std::endl;
    }

    DataHandler::~DataHandler() {
        std::cout << "DataHandler Destroyed" << std::endl;
    }

    void DataHandler::loadData(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            data.push_back(line);
        }
        file.close();
        std::cout << "Data loaded from " << filePath << std::endl;
    }

    void DataHandler::processData() {
        // Process the data (placeholder for actual processing logic)
        std::cout << "Processing data..." << std::endl;
        for (auto& line : data){
            for(auto& ch : line) {
                ch = tolower(ch);
            }
        }
        std::cout << "Data proccessed" << std::endl;
    }

    void DataHandler::saveData(const std::string& filePath){
        std::ofstream file(filePath);
        for(const auto& line : data){
            file << line << std::endl;
        }
        file.close();
        std::cout << "Data saved to " << filePath << std::endl;
    }

    std::vector<std::string> DataHandler::getData() const {
        return data;
    }
}
