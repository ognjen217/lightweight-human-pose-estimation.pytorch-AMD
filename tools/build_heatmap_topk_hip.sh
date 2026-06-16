#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/heatmap_topk_hip"
ARCH="${HIP_TOPK_OFFLOAD_ARCH:-native}"
SMART_THREADS="${HIP_TOPK_SMART_THREADS:-64}"
SMART_SRC="${ROOT_DIR}/cpp/heatmap_topk_hip/heatmap_topk_hip_smart.cpp"

if [[ -f "${SMART_SRC}" ]]; then
  python - "${SMART_SRC}" "${SMART_THREADS}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
threads = int(sys.argv[2])
if threads <= 0:
    raise SystemExit("HIP_TOPK_SMART_THREADS must be positive")
text = path.read_text()
new, count = re.subn(r"const int threads = \d+;", f"const int threads = {threads};", text, count=1)
if count != 1:
    raise SystemExit("Could not patch smart HIP thread count")
if new != text:
    path.write_text(new)
print(f"[build] smart HIP reducer threads={threads}")
PY
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "[build] root=${ROOT_DIR}"
echo "[build] build_dir=${BUILD_DIR}"
echo "[build] HIP_TOPK_OFFLOAD_ARCH=${ARCH}"
echo "[build] HIP_TOPK_SMART_THREADS=${SMART_THREADS}"

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
