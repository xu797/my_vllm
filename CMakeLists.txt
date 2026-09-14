cmake_minimum_required(VERSION 3.20.0)
project(my_vllm)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tensor)
add_subdirectory(graph)
add_subdirectory(layer)
add_subdirectory(operators)