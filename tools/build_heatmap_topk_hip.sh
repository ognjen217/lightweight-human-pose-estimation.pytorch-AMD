#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/heatmap_topk_hip"
ARCH="${HIP_TOPK_OFFLOAD_ARCH:-native}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[build] root=${ROOT_DIR}"
echo "[build] build_dir=${BUILD_DIR}"
echo "[build] HIP_TOPK_OFFLOAD_ARCH=${ARCH}"

cmake "${ROOT_DIR}/cpp/heatmap_topk_hip" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DHIP_TOPK_OFFLOAD_ARCH="${ARCH}"

cmake --build . -j

LIB="${BUILD_DIR}/libheatmap_topk_hip.so"
if [[ ! -f "${LIB}" ]]; then
  echo "[build:error] expected library not found: ${LIB}" >&2
  echo "[build:error] files under build dir:" >&2
  find "${BUILD_DIR}" -maxdepth 4 -type f -print >&2
  exit 1
fi

ls -lh "${LIB}"
echo "[build] OK: ${LIB}"
