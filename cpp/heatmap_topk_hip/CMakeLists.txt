cmake_minimum_required(VERSION 3.21)
project(heatmap_topk_hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(hip REQUIRED)

add_library(heatmap_topk_hip SHARED
    heatmap_topk_hip.cpp
)

target_include_directories(heatmap_topk_hip PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(heatmap_topk_hip PRIVATE hip::host)

set_target_properties(heatmap_topk_hip PROPERTIES
    OUTPUT_NAME heatmap_topk_hip
)
