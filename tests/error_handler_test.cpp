/**
 * @file error_handler_test.cpp
 * @brief Tests for the ZyraAI error handling system
 * @author ZyraAI Team
 */

#include "model/error_handler.h"
#include <gtest/gtest.h>
#include <string>

using namespace zyraai::error;

// Test the basic ZyraAIError class
TEST(ErrorHandlerTest, BaseErrorClassWorks) {
  try {
    throw ZyraAIError("Test error message", ErrorCategory::RUNTIME_ERROR, "TestComponent");
  } catch (const ZyraAIError& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("RUNTIME_ERROR") != std::string::npos);
    EXPECT_TRUE(message.find("TestComponent") != std::string::npos);
    EXPECT_TRUE(message.find("Test error message") != std::string::npos);
    EXPECT_EQ(e.getCategory(), ErrorCategory::RUNTIME_ERROR);
    EXPECT_EQ(e.getComponent(), "TestComponent");
  }
}

// Test InvalidArgument error class
TEST(ErrorHandlerTest, InvalidArgumentWorks) {
  try {
    throw InvalidArgument("Invalid parameter value", "TestLayer");
  } catch (const ZyraAIError& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("INVALID_ARGUMENT") != std::string::npos);
    EXPECT_TRUE(message.find("TestLayer") != std::string::npos);
    EXPECT_TRUE(message.find("Invalid parameter value") != std::string::npos);
    EXPECT_EQ(e.getCategory(), ErrorCategory::INVALID_ARGUMENT);
  }
}

// Test DimensionMismatch error class
TEST(ErrorHandlerTest, DimensionMismatchWorks) {
  try {
    throw DimensionMismatch(10, 5, "ConvLayer");
  } catch (const ZyraAIError& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("DIMENSION_MISMATCH") != std::string::npos);
    EXPECT_TRUE(message.find("ConvLayer") != std::string::npos);
    EXPECT_TRUE(message.find("Expected dimension: 10") != std::string::npos);
    EXPECT_TRUE(message.find("got: 5") != std::string::npos);
    EXPECT_EQ(e.getCategory(), ErrorCategory::DIMENSION_MISMATCH);
  }
}

// Test OutOfRange error class
TEST(ErrorHandlerTest, OutOfRangeWorks) {
  try {
    throw OutOfRange("Index out of bounds", "DenseLayer");
  } catch (const ZyraAIError& e) {
    std::string message = e.what();
    EXPECT_TRUE(message.find("OUT_OF_RANGE") != std::string::npos);
    EXPECT_TRUE(message.find("DenseLayer") != std::string::npos);
    EXPECT_EQ(e.getCategory(), ErrorCategory::OUT_OF_RANGE);
  }
}

// Test error catching via base class
TEST(ErrorHandlerTest, ErrorHierarchyWorks) {
  try {
    // Choose a random error type to throw
    int errorType = rand() % 3;
    switch (errorType) {
      case 0:
        throw InvalidArgument("Test error", "TestComponent");
      case 1:
        throw DimensionMismatch("Size mismatch", "TestComponent");
      case 2:
        throw OutOfRange("Out of range", "TestComponent");
      default:
        throw ZyraAIError("Default error", ErrorCategory::RUNTIME_ERROR, "TestComponent");
    }
  } catch (const ZyraAIError& e) {
    // Should catch all error types
    SUCCEED();
    return;
  } catch (...) {
    FAIL() << "Error was not caught by the base class";
  }
}

// Test that standard exceptions fail the above test
TEST(ErrorHandlerTest, StandardExceptionsAreNotCaught) {
  bool caught = false;
  try {
    throw std::runtime_error("Standard error");
  } catch (const ZyraAIError& e) {
    caught = true;
  } catch (...) {
    // Expected to reach here
  }
  
  EXPECT_FALSE(caught) << "Standard exception was incorrectly caught as ZyraAIError";
}

// Main function to run all tests
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} 