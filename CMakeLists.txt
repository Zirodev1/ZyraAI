cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ZyraAI VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the module path for custom CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Find required packages
find_package(Eigen3 3.3 REQUIRED)
find_package(SndFile REQUIRED)

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
target_link_libraries(ZyraAI
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

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
target_link_libraries(simple_neural_network
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

add_executable(mnist_classifier examples/mnist_classifier.cpp ${SOURCES})
target_link_libraries(mnist_classifier
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

# Enable testing
enable_testing()

# Add a library target for the main project sources
add_library(ZyraAI_lib ${SOURCES})
target_link_libraries(ZyraAI_lib
    Eigen3::Eigen
    ${SNDFILE_LIBRARIES}
)

# Create separate executables for each test file
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name}
        ZyraAI_lib
        Eigen3::Eigen
        ${SNDFILE_LIBRARIES}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
