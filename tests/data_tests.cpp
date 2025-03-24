// tests/data_tests.cpp

#include "data/data_handler.h"
#include <cassert>
#include <iostream>

void test_data_handler() {
  zyraai::DataHandler dataHandler;
  const std::string filePath =
      "../tests/sample_data.txt"; // Use a relative path
  std::cout << "Loading data from: " << filePath << std::endl;
  dataHandler.loadData(filePath);
  dataHandler.processData();
  dataHandler.saveData("../tests/sample_data2.txt"); // Use a relative path
  auto data = dataHandler.getData();
  if (data.empty()) {
    std::cerr << "Error: No data loaded from " << filePath << std::endl;
  }
  assert(!data.empty());
  std::cout << "DataHandler test passed!" << std::endl;
}

int main() {
  test_data_handler();
  return 0;
}
