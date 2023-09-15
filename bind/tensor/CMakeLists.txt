cmake_minimum_required(VERSION 3.25)
project(_tensor)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/utf-8)
    add_compile_options(/bigobj)
endif ()

# 库文件输出地址
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../ChenTensor/_tensor/)

set(CMAKE_CXX_STANDARD 14)

# 两个库目录
set(PYTHONPATH D:/Anaconda/envs/ChenTensor)
set(PYBINDPATH D:/Anaconda/envs/ChenTensor/Lib/site-packages/pybind11)

# 包含目录
include_directories(${PYTHONPATH}/include
        ${PYBINDPATH}/include
        ../../cpp
        D:/c++lib/xtensor/xtensor-blas-master/include
        D:/c++lib/xtensor/xtensor-master/include
        D:/c++lib/xtensor/xtensor-python-master/include
        D:/c++lib/xtensor/xtl-master/include
        D:/Anaconda/envs/ChenTensor/Lib/site-packages/numpy/core/include)

# 链接目录
link_directories(${PYTHONPATH}/libs)

# 搜索源文件
aux_source_directory(./ SRC_LIST)

# 生成动态库
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# 后缀名改成pyd能够被python引用
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")

target_link_libraries(${PROJECT_NAME} python39.lib)