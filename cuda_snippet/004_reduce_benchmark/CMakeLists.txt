cmake_minimum_required(VERSION 3.10)
project(reduce_benchmark CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(reduce_benchmark reduce_sum.cu)
set_target_properties(reduce_benchmark PROPERTIES CUDA_ARCHITECTURES "75")  # 根据你的GPU调整

# 添加生成PTX的自定义命令
add_custom_command(
    TARGET reduce_benchmark
    POST_BUILD
    COMMAND ${CMAKE_CUDA_COMPILER} 
            -ptx
            ${CMAKE_CURRENT_SOURCE_DIR}/reduce_sum.cu
            -o ${CMAKE_BINARY_DIR}/reduce_sum.ptx
    COMMENT "Generating PTX file"
)