cmake_minimum_required(VERSION 3.21)

# Keep the scaffold CXX-only for now.  The current implementation only uses the
# HIP runtime C API from a .cpp translation unit, so enabling CMake's HIP
# language is unnecessary and makes CMake reject the hipcc wrapper when users
# pass -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc.  When real .hip kernels are
# added, this can be switched to LANGUAGES CXX HIP and configured with Clang
# directly or by letting CMake auto-detect the HIP compiler.
project(heatmap_topk_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(hip REQUIRED CONFIG)

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
