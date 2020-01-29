cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)
add_library(sde_geometrization STATIC main.cpp)

# declare AUTODIFF_Path
set(AUTODIFF ../../autodiff/)
set(CODI ../../CoDiPack/include)
set(EIGEN ../../eigen-master)

include_directories(${AUTODIFF})
include_directories(${CODI})
include_directories(${EIGEN})

add_executable(sde_geometraization main.cpp)
target_link_libraries(sde_geometraization)
