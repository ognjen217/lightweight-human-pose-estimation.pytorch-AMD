#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/cpp/split_hip_fused"
BUILD_DIR="$ROOT_DIR/build/split_hip_fused"
ARCH="${HIP_SPLIT_FUSED_OFFLOAD_ARCH:-${SPLIT_HIP_FUSED_OFFLOAD_ARCH:-native}}"

mkdir -p "$BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DSPLIT_HIP_FUSED_OFFLOAD_ARCH="$ARCH"
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Built: $BUILD_DIR/libsplit_hip_fused.so"
