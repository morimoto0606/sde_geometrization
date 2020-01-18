cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)
# set runtime directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# declare GTEST_ROOT
set(GTEST_ROOT ~/googletest/googletest/)

# Include files directory
include_directories(${GTEST_ROOT}/include/)

# declare AUTODIFF_Path
set(AUTODIFF ../../autodiff/)
include_directories(${AUTODIFF})
set(CODI ../../CoDiPack/include)
include_directories(${CODI})


# Library directory
link_directories(${GTEST_ROOT}/build/)

# Generate executable file
add_executable(Test 
    ${CMAKE_CURRENT_SOURCE_DIR}/sample.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/TestRungeKutta.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAutodiff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestCodi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestTensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestVectorField.cpp
)


# Link libraries
target_link_libraries(Test gtest gtest_main pthread)