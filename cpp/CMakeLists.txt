cmake_minimum_required(VERSION 3.25)
project(test)
add_compile_options(/utf-8)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        D:/c++lib/xtensor/xtensor-blas-master/include
        D:/c++lib/xtensor/xtensor-master/include
        D:/c++lib/xtensor/xtensor-python-master/include
        D:/c++lib/xtensor/xtl-master/include)
