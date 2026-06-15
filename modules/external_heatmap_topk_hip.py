"""ctypes loader for the native HIP heatmap TopK backend.

This module gives the Python side a stable place to load the shared library,
call the C ABI, and later plug the backend into the split MXR pipeline.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


HIP_TOPK_SUCCESS = 0
HIP_TOPK_INVALID_ARGUMENT = 1
HIP_TOPK_HIP_ERROR = 2
HIP_TOPK_NOT_IMPLEMENTED = 3


@dataclass(frozen=True)
class HipHeatmapTopKShape:
    batch: int = 4
    channels: int = 18
    in_h: int = 68
    in_w: int = 121
    full_h: int = 1080
    full_w: int = 1920
    topk: int = 20
    threshold: float = 0.1
    nms_radius: int = 6


def default_library_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    return root / "build" / "heatmap_topk_hip" / "libheatmap_topk_hip.so"


class HipHeatmapTopKBackend:
    def __init__(self, library_path: str | Path | None = None):
        env_path = os.environ.get("HEATMAP_TOPK_HIP_LIB", "").strip()
        path = Path(library_path or env_path or default_library_path())
        if not path.exists():
            raise FileNotFoundError(
                f"HIP heatmap TopK shared library not found: {path}. "
                "Build it with: cmake --build build/heatmap_topk_hip -j"
            )
        self.path = path
        self.lib = ctypes.CDLL(str(path))

        self.lib.heatmap_topk_hip_status_string.argtypes = [ctypes.c_int]
        self.lib.heatmap_topk_hip_status_string.restype = ctypes.c_char_p

        self.lib.heatmap_topk_hip_run.argtypes = [
            ctypes.c_void_p,       # heatmaps_dev
            ctypes.c_void_p,       # top_scores_dev
            ctypes.c_void_p,       # top_indices_dev
            ctypes.c_int,          # batch
            ctypes.c_int,          # channels
            ctypes.c_int,          # in_h
            ctypes.c_int,          # in_w
            ctypes.c_int,          # full_h
            ctypes.c_int,          # full_w
            ctypes.c_int,          # topk
            ctypes.c_float,        # threshold
            ctypes.c_int,          # nms_radius
            ctypes.c_void_p,       # hip_stream
        ]
        self.lib.heatmap_topk_hip_run.restype = ctypes.c_int

        self.lib.heatmap_topk_hip_run_host.argtypes = [
            ctypes.c_void_p,       # heatmaps_host
            ctypes.c_void_p,       # top_scores_host
            ctypes.c_void_p,       # top_indices_host
            ctypes.c_int,          # batch
            ctypes.c_int,          # channels
            ctypes.c_int,          # in_h
            ctypes.c_int,          # in_w
            ctypes.c_int,          # full_h
            ctypes.c_int,          # full_w
            ctypes.c_int,          # topk
            ctypes.c_float,        # threshold
            ctypes.c_int,          # nms_radius
        ]
        self.lib.heatmap_topk_hip_run_host.restype = ctypes.c_int

    def status_string(self, status: int) -> str:
        raw = self.lib.heatmap_topk_hip_status_string(int(status))
        return raw.decode("utf-8") if raw else f"UNKNOWN_STATUS_{status}"

    def run_raw(
        self,
        *,
        heatmaps_ptr: int,
        top_scores_ptr: int,
        top_indices_ptr: int,
        shape: HipHeatmapTopKShape,
        hip_stream_ptr: int = 0,
        raise_on_error: bool = True,
    ) -> int:
        status = int(self.lib.heatmap_topk_hip_run(
            ctypes.c_void_p(int(heatmaps_ptr)),
            ctypes.c_void_p(int(top_scores_ptr)),
            ctypes.c_void_p(int(top_indices_ptr)),
            int(shape.batch),
            int(shape.channels),
            int(shape.in_h),
            int(shape.in_w),
            int(shape.full_h),
            int(shape.full_w),
            int(shape.topk),
            ctypes.c_float(float(shape.threshold)),
            int(shape.nms_radius),
            ctypes.c_void_p(int(hip_stream_ptr)),
        ))
        if raise_on_error and status != HIP_TOPK_SUCCESS:
            raise RuntimeError(f"heatmap_topk_hip_run failed: {self.status_string(status)} ({status})")
        return status

    def run_host(self, heatmaps: np.ndarray, shape: HipHeatmapTopKShape | None = None) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.ascontiguousarray(np.asarray(heatmaps, dtype=np.float32))
        inferred = HipHeatmapTopKShape(batch=arr.shape[0], channels=arr.shape[1]) if shape is None else shape
        expected = (int(inferred.batch), int(inferred.channels), int(inferred.in_h), int(inferred.in_w))
        if tuple(arr.shape) != expected:
            raise ValueError(f"Expected heatmaps shape {expected}, got {tuple(arr.shape)}")

        top_scores = np.empty((int(inferred.batch), int(inferred.channels), int(inferred.topk)), dtype=np.float32)
        top_indices = np.empty((int(inferred.batch), int(inferred.channels), int(inferred.topk)), dtype=np.int64)
        status = int(self.lib.heatmap_topk_hip_run_host(
            arr.ctypes.data_as(ctypes.c_void_p),
            top_scores.ctypes.data_as(ctypes.c_void_p),
            top_indices.ctypes.data_as(ctypes.c_void_p),
            int(inferred.batch),
            int(inferred.channels),
            int(inferred.in_h),
            int(inferred.in_w),
            int(inferred.full_h),
            int(inferred.full_w),
            int(inferred.topk),
            ctypes.c_float(float(inferred.threshold)),
            int(inferred.nms_radius),
        ))
        if status != HIP_TOPK_SUCCESS:
            raise RuntimeError(f"heatmap_topk_hip_run_host failed: {self.status_string(status)} ({status})")
        return np.ascontiguousarray(top_scores), np.ascontiguousarray(top_indices)
