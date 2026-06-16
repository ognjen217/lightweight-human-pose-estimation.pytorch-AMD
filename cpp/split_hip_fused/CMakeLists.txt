cmake_minimum_required(VERSION 3.21)

project(split_hip_fused LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS /opt/rocm/bin
    REQUIRED
)

set(SPLIT_HIP_FUSED_OFFLOAD_ARCH "native" CACHE STRING "HIP offload architecture, e.g. native, gfx1100, gfx1150. Empty disables explicit arch flag.")
set(SPLIT_HIP_FUSED_ARCH_FLAGS "")
if(NOT "${SPLIT_HIP_FUSED_OFFLOAD_ARCH}" STREQUAL "")
    list(APPEND SPLIT_HIP_FUSED_ARCH_FLAGS "--offload-arch=${SPLIT_HIP_FUSED_OFFLOAD_ARCH}")
endif()

set(SPLIT_HIP_FUSED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/split_hip_fused_v2.cpp
)

set(SPLIT_HIP_FUSED_OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/libsplit_hip_fused.so
)

add_custom_command(
    OUTPUT ${SPLIT_HIP_FUSED_OUTPUT}
    COMMAND ${HIPCC_EXECUTABLE}
        -O3
        -std=c++17
        -fPIC
        -shared
        ${SPLIT_HIP_FUSED_ARCH_FLAGS}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${CMAKE_CURRENT_SOURCE_DIR}/../heatmap_topk_hip
        ${SPLIT_HIP_FUSED_SOURCES}
        -o ${SPLIT_HIP_FUSED_OUTPUT}
    DEPENDS ${SPLIT_HIP_FUSED_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/split_hip_fused.h
            ${CMAKE_CURRENT_SOURCE_DIR}/../heatmap_topk_hip/heatmap_topk_hip_smart.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../heatmap_topk_hip/heatmap_topk_hip.h
    COMMENT "Building low-level fused HIP split postprocess backend with ${HIPCC_EXECUTABLE}"
    VERBATIM
)

add_custom_target(split_hip_fused ALL
    DEPENDS ${SPLIT_HIP_FUSED_OUTPUT}
)
