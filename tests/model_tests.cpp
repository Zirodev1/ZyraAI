// tests/model_tests.cpp

#include "model/zyraAI_model.h"
#include <cassert>
#include <iostream>

void test_ziroai_model() {
    zyraai::ZyraAIModel model;
    model.train();
    model.predict();
    std::cout << "ZyraAI Model test passed!" << std::endl;
}

int main() {
    test_ziroai_model();
    return 0;
}
