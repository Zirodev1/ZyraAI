cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ZyraAI VERSION 1.0)

# Check for Apple Silicon
if(APPLE)
  execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(${ARCH} STREQUAL "arm64")
    set(APPLE_SILICON TRUE)
    message(STATUS "Detected Apple Silicon architecture")
  endif()
endif()

# Enable Metal framework for Apple Silicon
if(APPLE_SILICON)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  message(STATUS "Metal framework found")
  
  # Enable Objective-C++
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

# Enable compiler optimizations with platform-specific flags
if(APPLE_SILICON)
  # Optimizations specifically for Apple Silicon (M1, M2, M3 series)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m1 -mtune=apple-m1 -DNDEBUG -ffast-math -fvectorize -fslp-vectorize -ffp-contract=fast")
  set(CMAKE_OBJCXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
else()
  # General optimizations for other platforms
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -ffast-math")
endif()

# Enable maximum vectorization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvectorize -funroll-loops")
if(APPLE_SILICON)
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fvectorize -funroll-loops")
endif()

# Set build type to Release for optimizations
set(CMAKE_BUILD_TYPE Release)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the module path for custom CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(SndFile REQUIRED)
find_package(OpenMP QUIET)
find_package(GTest REQUIRED)

if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found and enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "OpenMP not found. Multi-threading capabilities will be limited.")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${SNDFILE_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# Add Metal source files
if(APPLE_SILICON)
  file(GLOB METAL_SOURCES "src/metal/*.mm")
  file(GLOB METAL_HEADERS "include/metal/*.h")
  file(GLOB METAL_SHADERS "shaders/metal/*.metal")
  
  # Compile Metal shaders
  foreach(shader ${METAL_SHADERS})
    get_filename_component(shader_name ${shader} NAME_WE)
    set(metallib "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.metallib")
    set(air_file "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.air")
    
    add_custom_command(
      OUTPUT ${metallib}
      COMMAND xcrun -sdk macosx metal -c ${shader} -o ${air_file}
      COMMAND xcrun -sdk macosx metallib ${air_file} -o ${metallib}
    )
    add_custom_target(${shader_name}_metallib ALL DEPENDS ${metallib})
    list(APPEND METAL_LIBS ${shader_name}_metallib)
  endforeach()
endif()

# Add the source files except preprocess_data.cpp and text_preprocess.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/data/preprocess_data.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/data/text_preprocess.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

# Create Metal library target
if(APPLE_SILICON)
  add_library(ZyraAI_metal STATIC
    src/metal/metal_device.mm
    src/metal/metal_buffer.mm
    src/metal/metal_kernel.mm
  )
  target_link_libraries(ZyraAI_metal
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
  )
  target_include_directories(ZyraAI_metal PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  )
  set_target_properties(ZyraAI_metal PROPERTIES
    LINKER_LANGUAGE OBJCXX
  )
  
  # Add Metal shader libraries as dependencies
  add_dependencies(ZyraAI_metal ${METAL_LIBS})
endif()

# Create the main executable ZyraAI
add_executable(ZyraAI src/main.cpp ${SOURCES})

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(ZyraAI
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(ZyraAI
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(ZyraAI ZyraAI_metal)
endif()

# Add an executable for preprocess_data
add_executable(preprocess_data src/data/preprocess_data.cpp src/audio/audio_processing.cpp)
target_link_libraries(preprocess_data
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

# Add an executable for text_preprocess
add_executable(text_preprocess src/data/text_preprocess.cpp)
target_link_libraries(text_preprocess
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

# Add the example executables
add_executable(simple_neural_network examples/simple_neural_network.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_neural_network
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(simple_neural_network
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(simple_neural_network ZyraAI_metal)
endif()

add_executable(mnist_classifier examples/mnist_classifier.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_classifier
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_classifier
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_classifier ZyraAI_metal)
endif()

# Add the CNN example executables
add_executable(mnist_cnn_classifier examples/mnist_cnn_classifier.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_cnn_classifier
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_cnn_classifier
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_cnn_classifier ZyraAI_metal)
endif()

# Add the optimized CNN example executable
add_executable(mnist_optimized_cnn examples/mnist_optimized_cnn.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_optimized_cnn
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_optimized_cnn
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_optimized_cnn ZyraAI_metal)
endif()

# Add the ResNet example executable
add_executable(mnist_resnet examples/mnist_resnet.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_resnet
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_resnet
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_resnet ZyraAI_metal)
endif()

# Add the Enhanced CNN example executable (Part 1 of Phase 1)
add_executable(mnist_enhanced_cnn examples/mnist_enhanced_cnn.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_enhanced_cnn
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_enhanced_cnn
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_enhanced_cnn ZyraAI_metal)
endif()

# Add the Residual Network example executable (Part 2 of Phase 1)
add_executable(mnist_residual_network examples/mnist_residual_network.cpp ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(mnist_residual_network
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(mnist_residual_network
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(mnist_residual_network ZyraAI_metal)
endif()

# Enable testing
enable_testing()

# Add a library target for the main project sources
add_library(ZyraAI_lib ${SOURCES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(ZyraAI_lib
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(ZyraAI_lib
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
endif()

if(APPLE_SILICON)
    target_link_libraries(ZyraAI_lib ZyraAI_metal)
endif()

# Create test executables
foreach(test_name data_tests model_tests speech_test metal_test)
    add_executable(${test_name}_test tests/${test_name}.cpp)
    target_link_libraries(${test_name}_test PRIVATE GTest::gtest_main)
    
    if(${test_name} STREQUAL "metal_test")
        target_link_libraries(${test_name}_test PRIVATE ZyraAI_metal ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
        if(APPLE_SILICON)
            set_target_properties(${test_name}_test PROPERTIES
                LINKER_LANGUAGE OBJCXX
            )
            set_source_files_properties(tests/metal_test.cpp PROPERTIES
                COMPILE_FLAGS "-x objective-c++"
            )
        endif()
    else()
        target_link_libraries(${test_name}_test PRIVATE ZyraAI_lib)
    endif()
    
    add_test(NAME ${test_name} COMMAND ${test_name}_test)
endforeach()
