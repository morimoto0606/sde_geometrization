cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)
add_library(sde_geometrization STATIC main.cpp)

# declare AUTODIFF_Path
set(AUTODIFF ../../autodiff/)
include_directories(${AUTODIFF})
set(CODI ../../CoDiPack/include)
include_directories(${CODI})

add_executable(sde_geometraization main.cpp)
target_link_libraries(sde_geometraization)
