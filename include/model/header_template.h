/**
 * @file header_template.h
 * @brief Template for standardized header files
 * @author ZyraAI Team
 */

#ifndef ZYRAAI_HEADER_TEMPLATE_H
#define ZYRAAI_HEADER_TEMPLATE_H

// Standard library includes first, alphabetically ordered
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// External library includes next, alphabetically ordered
#include <Eigen/Dense>

// ZyraAI includes last, alphabetically ordered
#include "model/layer.h"

namespace zyraai {

/**
 * @class ClassName
 * @brief Brief description of the class
 * 
 * Detailed description of the class, its purpose, functionality,
 * and any relevant implementation details or algorithms used.
 * Include references to papers or other resources if applicable.
 */
class ClassName : public BaseClass {
public:
  /**
   * @brief Constructor with documentation
   * @param param1 Description of first parameter
   * @param param2 Description of second parameter
   * @throws std::invalid_argument If parameters are invalid
   */
  ClassName(const std::string& param1, int param2);
  
  /**
   * @brief Virtual destructor for inheritance support
   */
  virtual ~ClassName() = default;
  
  /**
   * @brief Method description
   * @param input Description of input parameter
   * @return Description of return value
   * @throws std::runtime_error If something goes wrong
   */
  ReturnType publicMethod(const ParamType& input);
  
  /**
   * @brief Getter method for a class property
   * @return Description of returned property
   */
  PropertyType getProperty() const { return property_; }
  
  /**
   * @brief Setter method for a class property
   * @param value New value to set
   * @throws std::invalid_argument If value is invalid
   */
  void setProperty(const PropertyType& value);
  
protected:
  /**
   * @brief Protected method description
   * @param input Description of input parameter
   */
  void protectedMethod(int input);
  
private:
  /**
   * @brief Private method description
   */
  void privateMethod();
  
  // Group related variables together with appropriate comments
  
  // Configuration parameters
  std::string param1_;       ///< Description of param1
  int param2_;               ///< Description of param2
  float configValue_;        ///< Description of configValue
  
  // State variables
  bool isInitialized_;       ///< Whether the object is initialized
  int counter_;              ///< Description of counter
  
  // Data structures
  std::vector<float> data_;  ///< Description of data
  Eigen::MatrixXf matrix_;   ///< Description of matrix
};

// Inline function implementations (for small functions only)
inline void ClassName::setProperty(const PropertyType& value) {
  if (!isValidValue(value)) {
    throw std::invalid_argument("Invalid property value");
  }
  property_ = value;
}

} // namespace zyraai

#endif // ZYRAAI_HEADER_TEMPLATE_H 