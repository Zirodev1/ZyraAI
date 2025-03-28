/**
 * @file error_handler.h
 * @brief Standardized error handling system for ZyraAI
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_ERROR_HANDLER_H
#define ZYRAAI_ERROR_HANDLER_H

#include <stdexcept>
#include <string>
#include <sstream>

namespace zyraai {

/**
 * @namespace error
 * @brief Contains error handling utilities for ZyraAI
 */
namespace error {

/**
 * @brief Standardized error categories
 */
enum class ErrorCategory {
  INVALID_ARGUMENT,  ///< Invalid function argument or parameter
  DIMENSION_MISMATCH, ///< Tensor dimension mismatch
  OUT_OF_RANGE,      ///< Index or value out of allowed range
  RUNTIME_ERROR,     ///< Error during program execution
  NUMERIC_ERROR,     ///< Numerical computation error (overflow, underflow, etc.)
  HARDWARE_ERROR,    ///< Error related to hardware (GPU, etc.)
  UNSUPPORTED_OPERATION ///< Operation not supported in current configuration
};

/**
 * @brief Get string representation of error category
 * @param category Error category
 * @return String representation
 */
inline std::string categoryToString(ErrorCategory category) {
  switch (category) {
    case ErrorCategory::INVALID_ARGUMENT: 
      return "INVALID_ARGUMENT";
    case ErrorCategory::DIMENSION_MISMATCH: 
      return "DIMENSION_MISMATCH";
    case ErrorCategory::OUT_OF_RANGE: 
      return "OUT_OF_RANGE";
    case ErrorCategory::RUNTIME_ERROR: 
      return "RUNTIME_ERROR";
    case ErrorCategory::NUMERIC_ERROR: 
      return "NUMERIC_ERROR";
    case ErrorCategory::HARDWARE_ERROR: 
      return "HARDWARE_ERROR";
    case ErrorCategory::UNSUPPORTED_OPERATION: 
      return "UNSUPPORTED_OPERATION";
    default: 
      return "UNKNOWN_ERROR";
  }
}

/**
 * @class ZyraAIError
 * @brief Base class for all ZyraAI exceptions
 */
class ZyraAIError : public std::runtime_error {
public:
  /**
   * @brief Construct a new ZyraAI Error
   * @param message Error message
   * @param category Error category
   * @param component Name of the component where the error occurred
   */
  ZyraAIError(const std::string& message, 
              ErrorCategory category, 
              const std::string& component)
    : std::runtime_error(formatMessage(message, category, component)),
      category_(category),
      component_(component) {}

  /**
   * @brief Get the error category
   * @return Error category
   */
  ErrorCategory getCategory() const { return category_; }
  
  /**
   * @brief Get the component where the error occurred
   * @return Component name
   */
  std::string getComponent() const { return component_; }

private:
  /**
   * @brief Format the error message with consistent structure
   * @param message Raw error message
   * @param category Error category
   * @param component Component name
   * @return Formatted message
   */
  static std::string formatMessage(const std::string& message, 
                                  ErrorCategory category,
                                  const std::string& component) {
    std::stringstream ss;
    ss << "[" << categoryToString(category) << " in " << component << "] " << message;
    return ss.str();
  }

  ErrorCategory category_; ///< Error category
  std::string component_;  ///< Component where the error occurred
};

/**
 * @class InvalidArgument
 * @brief Exception for invalid function arguments or parameters
 */
class InvalidArgument : public ZyraAIError {
public:
  /**
   * @brief Construct a new Invalid Argument exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  InvalidArgument(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::INVALID_ARGUMENT, component) {}
};

/**
 * @class DimensionMismatch
 * @brief Exception for tensor dimension mismatches
 */
class DimensionMismatch : public ZyraAIError {
public:
  /**
   * @brief Construct a new Dimension Mismatch exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  DimensionMismatch(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::DIMENSION_MISMATCH, component) {}
  
  /**
   * @brief Construct a detailed Dimension Mismatch exception
   * @param expectedDim Expected dimension
   * @param actualDim Actual dimension
   * @param component Component where the error occurred
   */
  DimensionMismatch(int expectedDim, int actualDim, const std::string& component)
    : ZyraAIError(
        "Expected dimension: " + std::to_string(expectedDim) + 
        ", got: " + std::to_string(actualDim),
        ErrorCategory::DIMENSION_MISMATCH, 
        component) {}
};

/**
 * @class OutOfRange
 * @brief Exception for index or value out of allowed range
 */
class OutOfRange : public ZyraAIError {
public:
  /**
   * @brief Construct a new Out Of Range exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  OutOfRange(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::OUT_OF_RANGE, component) {}
};

/**
 * @class NumericError
 * @brief Exception for numerical computation errors
 */
class NumericError : public ZyraAIError {
public:
  /**
   * @brief Construct a new Numeric Error exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  NumericError(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::NUMERIC_ERROR, component) {}
};

/**
 * @class HardwareError
 * @brief Exception for hardware-related errors
 */
class HardwareError : public ZyraAIError {
public:
  /**
   * @brief Construct a new Hardware Error exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  HardwareError(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::HARDWARE_ERROR, component) {}
};

/**
 * @class UnsupportedOperation
 * @brief Exception for unsupported operations
 */
class UnsupportedOperation : public ZyraAIError {
public:
  /**
   * @brief Construct a new Unsupported Operation exception
   * @param message Error message
   * @param component Component where the error occurred
   */
  UnsupportedOperation(const std::string& message, const std::string& component)
    : ZyraAIError(message, ErrorCategory::UNSUPPORTED_OPERATION, component) {}
};

} // namespace error
} // namespace zyraai

#endif // ZYRAAI_ERROR_HANDLER_H 