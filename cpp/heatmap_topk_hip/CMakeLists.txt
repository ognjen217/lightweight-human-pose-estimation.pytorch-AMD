cmake_minimum_required(VERSION 3.21)

# Use a CXX-only CMake project and invoke hipcc explicitly.  This avoids the
# CMake 3.28 restriction that rejects /opt/rocm/bin/hipcc as CMAKE_HIP_COMPILER,
# while still compiling real HIP kernels into the shared library.
project(heatmap_topk_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS /opt/rocm/bin
    REQUIRED
)

set(HIP_TOPK_OFFLOAD_ARCH "native" CACHE STRING "HIP offload architecture, e.g. native, gfx1100, gfx1150. Empty disables explicit arch flag.")
set(HIP_TOPK_ARCH_FLAGS "")
if(NOT "${HIP_TOPK_OFFLOAD_ARCH}" STREQUAL "")
    list(APPEND HIP_TOPK_ARCH_FLAGS "--offload-arch=${HIP_TOPK_OFFLOAD_ARCH}")
endif()

set(HIP_TOPK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/heatmap_topk_hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/heatmap_topk_hip_profile.cpp
)
set(HIP_TOPK_OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/libheatmap_topk_hip.so
)

add_custom_command(
    OUTPUT ${HIP_TOPK_OUTPUT}
    COMMAND ${HIPCC_EXECUTABLE}
        -O3
        -std=c++17
        -fPIC
        -shared
        ${HIP_TOPK_ARCH_FLAGS}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        ${HIP_TOPK_SOURCES}
        -o ${HIP_TOPK_OUTPUT}
    DEPENDS ${HIP_TOPK_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/heatmap_topk_hip.h
    COMMENT "Building HIP heatmap TopK backend with ${HIPCC_EXECUTABLE}"
    VERBATIM
)

add_custom_target(heatmap_topk_hip ALL
    DEPENDS ${HIP_TOPK_OUTPUT}
)
