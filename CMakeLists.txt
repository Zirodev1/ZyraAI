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

# Enable compiler optimizations with platform-specific flags
if(APPLE_SILICON)
  # Optimizations specifically for Apple Silicon (M1, M2, M3 series)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m1 -mtune=apple-m1 -DNDEBUG -ffast-math -fvectorize -fslp-vectorize -ffp-contract=fast")
else()
  # General optimizations for other platforms
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -ffast-math")
endif()

# Enable maximum vectorization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvectorize -funroll-loops")

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
)

# Add the source files except preprocess_data.cpp and text_preprocess.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/data/preprocess_data.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/data/text_preprocess.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

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

# Create separate executables for each test file
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name}_test ${test_source})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${test_name}_test
            ZyraAI_lib
            Eigen3::Eigen
            ${SNDFILE_LIBRARIES}
            OpenMP::OpenMP_CXX
        )
    else()
        target_link_libraries(${test_name}_test
            ZyraAI_lib
            Eigen3::Eigen
            ${SNDFILE_LIBRARIES}
        )
    endif()
    add_test(NAME ${test_name}_test COMMAND ${test_name}_test)
endforeach()
