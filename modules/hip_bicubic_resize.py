"""ctypes wrapper for a standalone HIP bicubic HWC float32 resize kernel.

This is intentionally independent from PyTorch so it can run in the MIGraphX
process without importing torch.cuda. The first call compiles a small shared
library with hipcc and caches it under build/hip_kernels.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

_LIB = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_path() -> Path:
    return _repo_root() / "hip_kernels" / "hip_bicubic_resize.cpp"


def _library_path() -> Path:
    return _repo_root() / "build" / "hip_kernels" / "libhip_bicubic_resize.so"


def _find_hipcc() -> str:
    candidates = [
        os.environ.get("HIPCC", ""),
        "/opt/rocm/bin/hipcc",
        "hipcc",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            subprocess.check_call([c, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return c
        except Exception:
            pass
    raise RuntimeError("Could not find hipcc. Set HIPCC=/opt/rocm/bin/hipcc or add hipcc to PATH.")


def build(force: bool = False) -> Path:
    src = _source_path()
    out = _library_path()
    if not src.exists():
        raise FileNotFoundError(f"Missing HIP source: {src}")
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    hipcc = _find_hipcc()
    cmd = [hipcc, "-O3", "-fPIC", "-shared", str(src), "-o", str(out)]
    print("[hip-bicubic] compiling:", " ".join(cmd))
    subprocess.check_call(cmd)
    return out


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = build(force=False)
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.hip_bicubic_resize_hwc_f32
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    _LIB = lib
    return _LIB


def resize_hwc(src: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Resize HWC float32 image/tensor with OpenCV-like bicubic interpolation."""
    src = np.ascontiguousarray(src, dtype=np.float32)
    if src.ndim != 3:
        raise ValueError(f"Expected HWC input, got shape={src.shape}")
    in_h, in_w, channels = map(int, src.shape)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    dst = np.empty((out_h, out_w, channels), dtype=np.float32)
    lib = _load()
    rc = lib.hip_bicubic_resize_hwc_f32(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        in_h, in_w, out_h, out_w, channels,
    )
    if rc != 0:
        raise RuntimeError(f"hip_bicubic_resize_hwc_f32 failed with rc={rc}")
    return dst


if __name__ == "__main__":
    build(force=True)
