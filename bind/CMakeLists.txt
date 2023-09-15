cmake_minimum_required(VERSION 3.25)
project(bind)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(tensor)
add_subdirectory(functional)
add_subdirectory(network)
add_subdirectory(optim)
add_subdirectory(loss)
