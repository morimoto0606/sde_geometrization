cmake_minimum_required(VERSION 2.8)
project (SDE_GEOMETRIZATION)

add_subdirectory(src)
add_subdirectory(test)
enable_testing()
add_test(NAME MyTest COMMAND Test)