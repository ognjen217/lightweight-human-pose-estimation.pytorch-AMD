cmake_minimum_required(VERSION 3.21)

project(paf_prune_hip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS /opt/rocm/bin
    REQUIRED
)

set(HIP_PAF_PRUNE_OFFLOAD_ARCH "native" CACHE STRING "HIP offload architecture, e.g. native, gfx1100, gfx1150. Empty disables explicit arch flag.")
set(HIP_PAF_PRUNE_ARCH_FLAGS "")
if(NOT "${HIP_PAF_PRUNE_OFFLOAD_ARCH}" STREQUAL "")
    list(APPEND HIP_PAF_PRUNE_ARCH_FLAGS "--offload-arch=${HIP_PAF_PRUNE_OFFLOAD_ARCH}")
endif()

set(HIP_PAF_PRUNE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/paf_prune_hip.cpp
)
set(HIP_PAF_PRUNE_OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/libpaf_prune_hip.so
)

add_custom_command(
    OUTPUT ${HIP_PAF_PRUNE_OUTPUT}
    COMMAND ${HIPCC_EXECUTABLE}
        -O3
        -std=c++17
        -fPIC
        -shared
        ${HIP_PAF_PRUNE_ARCH_FLAGS}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        ${HIP_PAF_PRUNE_SOURCES}
        -o ${HIP_PAF_PRUNE_OUTPUT}
    DEPENDS ${HIP_PAF_PRUNE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/paf_prune_hip.h
    COMMENT "Building HIP PAF pruning backend with ${HIPCC_EXECUTABLE}"
    VERBATIM
)

add_custom_target(paf_prune_hip ALL
    DEPENDS ${HIP_PAF_PRUNE_OUTPUT}
)
