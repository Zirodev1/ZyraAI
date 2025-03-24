// tests/data_tests.cpp

#include "data/data_handler.h"
#include <cassert>
#include <iostream>

using namespace ::std;

void test_data_handler() {
  zyraai::DataHandler dataHandler;
  const ::std::string filePath =
      "../tests/sample_data.txt"; // Use a relative path
  cout << "Loading data from: " << filePath << endl;
  dataHandler.loadData(filePath);
  dataHandler.processData();
  dataHandler.saveData("../tests/sample_data2.txt"); // Use a relative path
  auto data = dataHandler.getData();
  if (data.empty()) {
    cerr << "Error: No data loaded from " << filePath << endl;
  }
  assert(!data.empty());
  cout << "DataHandler test passed!" << endl;
}

int main() {
  test_data_handler();
  return 0;
}
