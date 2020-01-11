cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 20)
# set runtime directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# declare GTEST_ROOT
set(GTEST_ROOT ~/googletest/googletest/)

# Include files directory
include_directories(${GTEST_ROOT}/include/)

# declare AUTODIFF_Path
set(AUTODIFF ../../autodiff/)
include_directories(${AUTODIFF})

# Library directory
link_directories(${GTEST_ROOT}/build/)

# Generate executable file
add_executable(Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/sample.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRungeKutta.cpp)

# Link libraries
target_link_libraries(Test gtest gtest_main pthread)