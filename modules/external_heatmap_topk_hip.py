"""ctypes loader for the native HIP heatmap TopK backend.

This module gives the Python side a stable place to load the shared library,
call the C ABI, and later plug the backend into the split MXR pipeline.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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


@dataclass(frozen=True)
class HipHeatmapTopKProfile:
    h2d_ms: float
    resize_ms: float
    vertical_ms: float
    horizontal_ms: float
    topk_ms: float
    d2h_scores_ms: float
    d2h_indices_ms: float
    device_total_ms: float
    total_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "h2d_ms": float(self.h2d_ms),
            "resize_ms": float(self.resize_ms),
            "vertical_ms": float(self.vertical_ms),
            "horizontal_ms": float(self.horizontal_ms),
            "topk_ms": float(self.topk_ms),
            "d2h_scores_ms": float(self.d2h_scores_ms),
            "d2h_indices_ms": float(self.d2h_indices_ms),
            "device_total_ms": float(self.device_total_ms),
            "total_ms": float(self.total_ms),
        }


class _CHipHeatmapTopKProfile(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("resize_ms", ctypes.c_float),
        ("vertical_ms", ctypes.c_float),
        ("horizontal_ms", ctypes.c_float),
        ("topk_ms", ctypes.c_float),
        ("d2h_scores_ms", ctypes.c_float),
        ("d2h_indices_ms", ctypes.c_float),
        ("device_total_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
    ]

    def to_dataclass(self) -> HipHeatmapTopKProfile:
        return HipHeatmapTopKProfile(
            h2d_ms=float(self.h2d_ms),
            resize_ms=float(self.resize_ms),
            vertical_ms=float(self.vertical_ms),
            horizontal_ms=float(self.horizontal_ms),
            topk_ms=float(self.topk_ms),
            d2h_scores_ms=float(self.d2h_scores_ms),
            d2h_indices_ms=float(self.d2h_indices_ms),
            device_total_ms=float(self.device_total_ms),
            total_ms=float(self.total_ms),
        )


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

        common_host_args = [
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

        self.lib.heatmap_topk_hip_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self.lib.heatmap_topk_hip_run.restype = ctypes.c_int

        self.lib.heatmap_topk_hip_run_host.argtypes = common_host_args
        self.lib.heatmap_topk_hip_run_host.restype = ctypes.c_int
        self.lib.heatmap_topk_hip_run_host_profile.argtypes = common_host_args + [ctypes.POINTER(_CHipHeatmapTopKProfile)]
        self.lib.heatmap_topk_hip_run_host_profile.restype = ctypes.c_int

        self.lib.heatmap_topk_hip_run_host_fused.argtypes = common_host_args
        self.lib.heatmap_topk_hip_run_host_fused.restype = ctypes.c_int
        self.lib.heatmap_topk_hip_run_host_fused_profile.argtypes = common_host_args + [ctypes.POINTER(_CHipHeatmapTopKProfile)]
        self.lib.heatmap_topk_hip_run_host_fused_profile.restype = ctypes.c_int

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
            ctypes.c_void_p(int(heatmaps_ptr)), ctypes.c_void_p(int(top_scores_ptr)), ctypes.c_void_p(int(top_indices_ptr)),
            int(shape.batch), int(shape.channels), int(shape.in_h), int(shape.in_w), int(shape.full_h), int(shape.full_w),
            int(shape.topk), ctypes.c_float(float(shape.threshold)), int(shape.nms_radius), ctypes.c_void_p(int(hip_stream_ptr)),
        ))
        if raise_on_error and status != HIP_TOPK_SUCCESS:
            raise RuntimeError(f"heatmap_topk_hip_run failed: {self.status_string(status)} ({status})")
        return status

    def _prepare_host_io(self, heatmaps: np.ndarray, shape: HipHeatmapTopKShape | None):
        arr = np.ascontiguousarray(np.asarray(heatmaps, dtype=np.float32))
        inferred = HipHeatmapTopKShape(batch=arr.shape[0], channels=arr.shape[1]) if shape is None else shape
        expected = (int(inferred.batch), int(inferred.channels), int(inferred.in_h), int(inferred.in_w))
        if tuple(arr.shape) != expected:
            raise ValueError(f"Expected heatmaps shape {expected}, got {tuple(arr.shape)}")
        top_scores = np.empty((int(inferred.batch), int(inferred.channels), int(inferred.topk)), dtype=np.float32)
        top_indices = np.empty((int(inferred.batch), int(inferred.channels), int(inferred.topk)), dtype=np.int64)
        return arr, inferred, top_scores, top_indices

    def _run_host_symbol(self, symbol_name: str, heatmaps: np.ndarray, shape: HipHeatmapTopKShape | None = None) -> Tuple[np.ndarray, np.ndarray]:
        arr, inferred, top_scores, top_indices = self._prepare_host_io(heatmaps, shape)
        fn = getattr(self.lib, symbol_name)
        status = int(fn(
            arr.ctypes.data_as(ctypes.c_void_p),
            top_scores.ctypes.data_as(ctypes.c_void_p),
            top_indices.ctypes.data_as(ctypes.c_void_p),
            int(inferred.batch), int(inferred.channels), int(inferred.in_h), int(inferred.in_w),
            int(inferred.full_h), int(inferred.full_w), int(inferred.topk), ctypes.c_float(float(inferred.threshold)),
            int(inferred.nms_radius),
        ))
        if status != HIP_TOPK_SUCCESS:
            raise RuntimeError(f"{symbol_name} failed: {self.status_string(status)} ({status})")
        return np.ascontiguousarray(top_scores), np.ascontiguousarray(top_indices)

    def _run_host_profile_symbol(
        self,
        symbol_name: str,
        heatmaps: np.ndarray,
        shape: HipHeatmapTopKShape | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, HipHeatmapTopKProfile]:
        arr, inferred, top_scores, top_indices = self._prepare_host_io(heatmaps, shape)
        c_profile = _CHipHeatmapTopKProfile()
        fn = getattr(self.lib, symbol_name)
        status = int(fn(
            arr.ctypes.data_as(ctypes.c_void_p),
            top_scores.ctypes.data_as(ctypes.c_void_p),
            top_indices.ctypes.data_as(ctypes.c_void_p),
            int(inferred.batch), int(inferred.channels), int(inferred.in_h), int(inferred.in_w),
            int(inferred.full_h), int(inferred.full_w), int(inferred.topk), ctypes.c_float(float(inferred.threshold)),
            int(inferred.nms_radius), ctypes.byref(c_profile),
        ))
        if status != HIP_TOPK_SUCCESS:
            raise RuntimeError(f"{symbol_name} failed: {self.status_string(status)} ({status})")
        return np.ascontiguousarray(top_scores), np.ascontiguousarray(top_indices), c_profile.to_dataclass()

    def run_host(self, heatmaps: np.ndarray, shape: HipHeatmapTopKShape | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self._run_host_symbol("heatmap_topk_hip_run_host", heatmaps, shape)

    def run_host_profile(
        self,
        heatmaps: np.ndarray,
        shape: HipHeatmapTopKShape | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, HipHeatmapTopKProfile]:
        return self._run_host_profile_symbol("heatmap_topk_hip_run_host_profile", heatmaps, shape)

    def run_host_fused(self, heatmaps: np.ndarray, shape: HipHeatmapTopKShape | None = None) -> Tuple[np.ndarray, np.ndarray]:
        return self._run_host_symbol("heatmap_topk_hip_run_host_fused", heatmaps, shape)

    def run_host_fused_profile(
        self,
        heatmaps: np.ndarray,
        shape: HipHeatmapTopKShape | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, HipHeatmapTopKProfile]:
        return self._run_host_profile_symbol("heatmap_topk_hip_run_host_fused_profile", heatmaps, shape)
