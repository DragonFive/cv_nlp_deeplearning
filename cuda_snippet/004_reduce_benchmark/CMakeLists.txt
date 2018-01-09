cmake_minimum_required(VERSION 3.10)
project(reduce_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加 CUB 头文件路径
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR})

# 编译 utils 库
add_library(utils STATIC
    utils.cc
    utils.h
)
set_target_properties(utils PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 编译可执行文件
add_executable(reduce_v0 reduce_sum.cu)
add_executable(reduce_v1 reduce_sum3.cu)

# 链接 utils 库
target_link_libraries(reduce_v0 utils ${CUDA_LIBRARIES})
target_link_libraries(reduce_v1 utils ${CUDA_LIBRARIES})

# GPU 架构设置
if(NOT DEFINED CUDA_ARCH)
    # 尝试自动检测 GPU 架构
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REPLACE "." "" CUDA_ARCH ${GPU_ARCH})
    else()
        # 默认值
        set(CUDA_ARCH "62")
    endif()
endif()

message(STATUS "Using CUDA architecture: ${CUDA_ARCH}")

# 设置 GPU 架构
set_target_properties(reduce_v0 PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
set_target_properties(reduce_v1 PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

# 添加生成 PTX 文件的命令
add_custom_command(
    TARGET reduce_v0
    POST_BUILD
    COMMAND ${CMAKE_CUDA_COMPILER} 
            -ptx
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_sum.cu
            -o ${CMAKE_BINARY_DIR}/reduce_sum.ptx
    COMMENT "Generating PTX file for reduce_v0"
)

add_custom_command(
    TARGET reduce_v1
    POST_BUILD
    COMMAND ${CMAKE_CUDA_COMPILER} 
            -ptx
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_sum3.cu
            -o ${CMAKE_BINARY_DIR}/reduce_sum3.ptx
    COMMENT "Generating PTX file for reduce_v1"
)