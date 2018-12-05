cmake_minimum_required(VERSION 3.12)
project(fvs_sample)

set(CMAKE_CXX_STANDARD 14)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(fvs_sample main.cpp)
target_link_libraries (fvs_sample Eigen3::Eigen)