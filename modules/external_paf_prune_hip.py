"""ctypes loader for the native HIP PAF pruning backend.

This is the HIP2 correctness-baseline backend for the split pipeline.  It is a
host-mediated wrapper first, matching the existing split HIP heatmap staging
approach.  The raw device-pointer API is exposed for a later zero-copy path.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HIP_PAF_PRUNE_SUCCESS = 0
HIP_PAF_PRUNE_INVALID_ARGUMENT = 1
HIP_PAF_PRUNE_HIP_ERROR = 2
HIP_PAF_PRUNE_NOT_IMPLEMENTED = 3


@dataclass(frozen=True)
class HipPafPruneShape:
    batch: int = 4
    topk: int = 20
    limb_topm: int = 20
    in_h: int = 68
    in_w: int = 121
    full_h: int = 1080
    full_w: int = 1920
    points_per_limb: int = 8
    min_paf_score: float = 0.05
    success_ratio_thr: float = 0.8
    min_pair_score: float = 0.0
    paf_cubic_a: float = -0.75


@dataclass(frozen=True)
class HipPafPruneProfile:
    h2d_ms: float
    score_ms: float
    prune_ms: float
    d2h_ms: float
    device_total_ms: float
    total_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "h2d_ms": float(self.h2d_ms),
            "score_ms": float(self.score_ms),
            "prune_ms": float(self.prune_ms),
            "d2h_ms": float(self.d2h_ms),
            "device_total_ms": float(self.device_total_ms),
            "total_ms": float(self.total_ms),
        }


class _CHipPafPruneProfile(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("score_ms", ctypes.c_float),
        ("prune_ms", ctypes.c_float),
        ("d2h_ms", ctypes.c_float),
        ("device_total_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
    ]

    def to_dataclass(self) -> HipPafPruneProfile:
        return HipPafPruneProfile(
            h2d_ms=float(self.h2d_ms),
            score_ms=float(self.score_ms),
            prune_ms=float(self.prune_ms),
            d2h_ms=float(self.d2h_ms),
            device_total_ms=float(self.device_total_ms),
            total_ms=float(self.total_ms),
        )


def default_library_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    return root / "build" / "paf_prune_hip" / "libpaf_prune_hip.so"


class HipPafPruneBackend:
    def __init__(self, library_path: str | Path | None = None):
        env_path = os.environ.get("PAF_PRUNE_HIP_LIB", "").strip()
        path = Path(library_path or env_path or default_library_path())
        if not path.exists():
            raise FileNotFoundError(f"HIP PAF pruning shared library not found: {path}. Build it with: bash tools/build_paf_prune_hip.sh")
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.paf_prune_hip_status_string.argtypes = [ctypes.c_int]
        self.lib.paf_prune_hip_status_string.restype = ctypes.c_char_p

        common_args = [
            ctypes.c_void_p,  # pafs
            ctypes.c_void_p,  # top_scores
            ctypes.c_void_p,  # top_indices
            ctypes.c_void_p,  # a_idx
            ctypes.c_void_p,  # b_idx
            ctypes.c_void_p,  # pair_score
            ctypes.c_void_p,  # pair_valid
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.paf_prune_hip_run.argtypes = common_args + [ctypes.c_void_p]
        self.lib.paf_prune_hip_run.restype = ctypes.c_int
        self.lib.paf_prune_hip_run_host.argtypes = common_args
        self.lib.paf_prune_hip_run_host.restype = ctypes.c_int
        self.lib.paf_prune_hip_run_host_profile.argtypes = common_args + [ctypes.POINTER(_CHipPafPruneProfile)]
        self.lib.paf_prune_hip_run_host_profile.restype = ctypes.c_int

    def status_string(self, status: int) -> str:
        raw = self.lib.paf_prune_hip_status_string(int(status))
        return raw.decode("utf-8") if raw else f"UNKNOWN_STATUS_{status}"

    def run_raw(
        self,
        *,
        pafs_ptr: int,
        top_scores_ptr: int,
        top_indices_ptr: int,
        a_idx_ptr: int,
        b_idx_ptr: int,
        pair_score_ptr: int,
        pair_valid_ptr: int,
        shape: HipPafPruneShape,
        hip_stream_ptr: int = 0,
        raise_on_error: bool = True,
    ) -> int:
        status = int(self.lib.paf_prune_hip_run(
            ctypes.c_void_p(int(pafs_ptr)),
            ctypes.c_void_p(int(top_scores_ptr)),
            ctypes.c_void_p(int(top_indices_ptr)),
            ctypes.c_void_p(int(a_idx_ptr)),
            ctypes.c_void_p(int(b_idx_ptr)),
            ctypes.c_void_p(int(pair_score_ptr)),
            ctypes.c_void_p(int(pair_valid_ptr)),
            int(shape.batch),
            int(shape.topk),
            int(shape.limb_topm),
            int(shape.in_h),
            int(shape.in_w),
            int(shape.full_h),
            int(shape.full_w),
            int(shape.points_per_limb),
            ctypes.c_float(float(shape.min_paf_score)),
            ctypes.c_float(float(shape.success_ratio_thr)),
            ctypes.c_float(float(shape.min_pair_score)),
            ctypes.c_float(float(shape.paf_cubic_a)),
            ctypes.c_void_p(int(hip_stream_ptr)),
        ))
        if raise_on_error and status != HIP_PAF_PRUNE_SUCCESS:
            raise RuntimeError(f"paf_prune_hip_run failed: {self.status_string(status)} ({status})")
        return status

    def _prepare_host_io(self, pafs: np.ndarray, top_scores: np.ndarray, top_indices: np.ndarray, shape: HipPafPruneShape | None):
        p = np.ascontiguousarray(np.asarray(pafs, dtype=np.float32))
        s = np.ascontiguousarray(np.asarray(top_scores, dtype=np.float32))
        i = np.ascontiguousarray(np.asarray(top_indices, dtype=np.int64))
        inferred = HipPafPruneShape(batch=p.shape[0], topk=s.shape[2]) if shape is None else shape
        if tuple(p.shape) != (int(inferred.batch), 38, int(inferred.in_h), int(inferred.in_w)):
            raise ValueError(f"Expected pafs shape {(int(inferred.batch), 38, int(inferred.in_h), int(inferred.in_w))}, got {tuple(p.shape)}")
        if tuple(s.shape) != (int(inferred.batch), 18, int(inferred.topk)):
            raise ValueError(f"Expected top_scores shape {(int(inferred.batch), 18, int(inferred.topk))}, got {tuple(s.shape)}")
        if tuple(i.shape) != (int(inferred.batch), 18, int(inferred.topk)):
            raise ValueError(f"Expected top_indices shape {(int(inferred.batch), 18, int(inferred.topk))}, got {tuple(i.shape)}")
        a_idx = np.empty((int(inferred.batch), 19, int(inferred.limb_topm)), dtype=np.int64)
        b_idx = np.empty((int(inferred.batch), 19, int(inferred.limb_topm)), dtype=np.int64)
        pair_score = np.empty((int(inferred.batch), 19, int(inferred.limb_topm)), dtype=np.float32)
        pair_valid = np.empty((int(inferred.batch), 19, int(inferred.limb_topm)), dtype=np.float32)
        return p, s, i, inferred, a_idx, b_idx, pair_score, pair_valid

    def run_host(
        self,
        pafs: np.ndarray,
        top_scores: np.ndarray,
        top_indices: np.ndarray,
        shape: HipPafPruneShape | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p, s, i, sh, a, b, score, valid = self._prepare_host_io(pafs, top_scores, top_indices, shape)
        status = int(self.lib.paf_prune_hip_run_host(
            p.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            i.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            score.ctypes.data_as(ctypes.c_void_p),
            valid.ctypes.data_as(ctypes.c_void_p),
            int(sh.batch),
            int(sh.topk),
            int(sh.limb_topm),
            int(sh.in_h),
            int(sh.in_w),
            int(sh.full_h),
            int(sh.full_w),
            int(sh.points_per_limb),
            ctypes.c_float(float(sh.min_paf_score)),
            ctypes.c_float(float(sh.success_ratio_thr)),
            ctypes.c_float(float(sh.min_pair_score)),
            ctypes.c_float(float(sh.paf_cubic_a)),
        ))
        if status != HIP_PAF_PRUNE_SUCCESS:
            raise RuntimeError(f"paf_prune_hip_run_host failed: {self.status_string(status)} ({status})")
        return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(score), np.ascontiguousarray(valid)

    def run_host_profile(
        self,
        pafs: np.ndarray,
        top_scores: np.ndarray,
        top_indices: np.ndarray,
        shape: HipPafPruneShape | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HipPafPruneProfile]:
        p, s, i, sh, a, b, score, valid = self._prepare_host_io(pafs, top_scores, top_indices, shape)
        c_profile = _CHipPafPruneProfile()
        status = int(self.lib.paf_prune_hip_run_host_profile(
            p.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            i.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            score.ctypes.data_as(ctypes.c_void_p),
            valid.ctypes.data_as(ctypes.c_void_p),
            int(sh.batch),
            int(sh.topk),
            int(sh.limb_topm),
            int(sh.in_h),
            int(sh.in_w),
            int(sh.full_h),
            int(sh.full_w),
            int(sh.points_per_limb),
            ctypes.c_float(float(sh.min_paf_score)),
            ctypes.c_float(float(sh.success_ratio_thr)),
            ctypes.c_float(float(sh.min_pair_score)),
            ctypes.c_float(float(sh.paf_cubic_a)),
            ctypes.byref(c_profile),
        ))
        if status != HIP_PAF_PRUNE_SUCCESS:
            raise RuntimeError(f"paf_prune_hip_run_host_profile failed: {self.status_string(status)} ({status})")
        return np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(score), np.ascontiguousarray(valid), c_profile.to_dataclass()
