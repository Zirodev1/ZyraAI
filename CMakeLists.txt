cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(ZyraAI VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the module path for custom CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Find and link Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add the include directories
include_directories(include ${EIGEN3_INCLUDE_DIR})

# Add the source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Find and link necessary libraries
find_package(SndFile REQUIRED)

# Create the main executable
add_executable(ZyraAI src/main.cpp ${SOURCES})
target_link_libraries(ZyraAI ${SNDFILE_LIBRARIES} Eigen3::Eigen)

# Enable testing
enable_testing()

# Add a library target for the main project sources
add_library(ZyraAI_lib ${SOURCES})
target_link_libraries(ZyraAI_lib ${SNDFILE_LIBRARIES} Eigen3::Eigen)

# Create separate executables for each test file
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} ZyraAI_lib ${SNDFILE_LIBRARIES} Eigen3::Eigen)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
