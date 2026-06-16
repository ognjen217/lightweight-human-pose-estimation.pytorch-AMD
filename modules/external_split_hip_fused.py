"""ctypes wrapper for fused HIP split postprocess backend.

The fused backend combines:
  heatmaps -> smart HIP TopK
  pafs + TopK -> HIP2 PAF pruning

into one shared-library call, keeping top_scores/top_indices on GPU between the
heatmap and PAF stages.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

HIP_SPLIT_FUSED_SUCCESS = 0


@dataclass(frozen=True)
class SplitHipFusedConfig:
    batch_size: int
    in_h: int = 68
    in_w: int = 121
    full_h: int = 1080
    full_w: int = 1920
    heatmap_channels: int = 18
    paf_channels: int = 38
    topk: int = 20
    limb_topm: int = 20
    threshold: float = 0.1
    lowres_nms_radius: int = 1
    smart_proposals: int = 32
    smart_local_radius: int = 4
    points_per_limb: int = 8
    min_paf_score: float = 0.05
    success_ratio_thr: float = 0.8
    min_pair_score: float = 0.0
    paf_cubic_a: float = -0.75


def default_library_path(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    return root / "build" / "split_hip_fused" / "libsplit_hip_fused.so"


class SplitHipFusedBackend:
    def __init__(self, library_path: str | Path | None = None):
        env_path = os.environ.get("SPLIT_HIP_FUSED_LIB", "").strip()
        path = Path(library_path or env_path or default_library_path())
        if not path.exists():
            raise FileNotFoundError(
                f"Fused HIP split backend not found: {path}. Build it with: bash tools/build_split_hip_fused.sh"
            )
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.split_hip_fused_status_string.argtypes = [ctypes.c_int]
        self.lib.split_hip_fused_status_string.restype = ctypes.c_char_p
        self.lib.split_hip_fused_run_host.argtypes = [
            ctypes.c_void_p,  # heatmaps_host
            ctypes.c_void_p,  # pafs_host
            ctypes.c_void_p,  # top_scores_host
            ctypes.c_void_p,  # top_indices_host
            ctypes.c_void_p,  # a_idx_host
            ctypes.c_void_p,  # b_idx_host
            ctypes.c_void_p,  # pair_score_host
            ctypes.c_void_p,  # pair_valid_host
            ctypes.c_int,     # batch
            ctypes.c_int,     # heatmap channels
            ctypes.c_int,     # paf channels
            ctypes.c_int,     # in_h
            ctypes.c_int,     # in_w
            ctypes.c_int,     # full_h
            ctypes.c_int,     # full_w
            ctypes.c_int,     # topk
            ctypes.c_int,     # limb_topm
            ctypes.c_float,   # threshold
            ctypes.c_int,     # lowres_nms_radius
            ctypes.c_int,     # smart_proposals
            ctypes.c_int,     # smart_local_radius
            ctypes.c_int,     # points_per_limb
            ctypes.c_float,   # min_paf_score
            ctypes.c_float,   # success_ratio_thr
            ctypes.c_float,   # min_pair_score
            ctypes.c_float,   # paf_cubic_a
        ]
        self.lib.split_hip_fused_run_host.restype = ctypes.c_int

    def status_string(self, status: int) -> str:
        raw = self.lib.split_hip_fused_status_string(int(status))
        return raw.decode("utf-8") if raw else f"UNKNOWN_STATUS_{status}"

    def run_host(self, heatmaps: np.ndarray, pafs: np.ndarray, cfg: SplitHipFusedConfig):
        heat = np.ascontiguousarray(np.asarray(heatmaps, dtype=np.float32))
        paf = np.ascontiguousarray(np.asarray(pafs, dtype=np.float32))
        expected_heat = (int(cfg.batch_size), int(cfg.heatmap_channels), int(cfg.in_h), int(cfg.in_w))
        expected_paf = (int(cfg.batch_size), int(cfg.paf_channels), int(cfg.in_h), int(cfg.in_w))
        if tuple(heat.shape) != expected_heat:
            raise ValueError(f"Expected heatmaps shape {expected_heat}, got {tuple(heat.shape)}")
        if tuple(paf.shape) != expected_paf:
            raise ValueError(f"Expected pafs shape {expected_paf}, got {tuple(paf.shape)}")

        top_scores = np.empty((int(cfg.batch_size), int(cfg.heatmap_channels), int(cfg.topk)), dtype=np.float32)
        top_indices = np.empty((int(cfg.batch_size), int(cfg.heatmap_channels), int(cfg.topk)), dtype=np.int64)
        out_shape = (int(cfg.batch_size), 19, int(cfg.limb_topm))
        a_idx = np.empty(out_shape, dtype=np.int64)
        b_idx = np.empty(out_shape, dtype=np.int64)
        pair_score = np.empty(out_shape, dtype=np.float32)
        pair_valid = np.empty(out_shape, dtype=np.float32)

        status = int(
            self.lib.split_hip_fused_run_host(
                heat.ctypes.data_as(ctypes.c_void_p),
                paf.ctypes.data_as(ctypes.c_void_p),
                top_scores.ctypes.data_as(ctypes.c_void_p),
                top_indices.ctypes.data_as(ctypes.c_void_p),
                a_idx.ctypes.data_as(ctypes.c_void_p),
                b_idx.ctypes.data_as(ctypes.c_void_p),
                pair_score.ctypes.data_as(ctypes.c_void_p),
                pair_valid.ctypes.data_as(ctypes.c_void_p),
                int(cfg.batch_size),
                int(cfg.heatmap_channels),
                int(cfg.paf_channels),
                int(cfg.in_h),
                int(cfg.in_w),
                int(cfg.full_h),
                int(cfg.full_w),
                int(cfg.topk),
                int(cfg.limb_topm),
                ctypes.c_float(float(cfg.threshold)),
                int(cfg.lowres_nms_radius),
                int(cfg.smart_proposals),
                int(cfg.smart_local_radius),
                int(cfg.points_per_limb),
                ctypes.c_float(float(cfg.min_paf_score)),
                ctypes.c_float(float(cfg.success_ratio_thr)),
                ctypes.c_float(float(cfg.min_pair_score)),
                ctypes.c_float(float(cfg.paf_cubic_a)),
            )
        )
        if status != HIP_SPLIT_FUSED_SUCCESS:
            raise RuntimeError(f"split_hip_fused_run_host failed: {self.status_string(status)} ({status})")

        return (
            np.ascontiguousarray(top_scores),
            np.ascontiguousarray(top_indices),
            np.ascontiguousarray(a_idx),
            np.ascontiguousarray(b_idx),
            np.ascontiguousarray(pair_score),
            np.ascontiguousarray(pair_valid),
        )


def run_external_split_hip_fused(heatmaps: np.ndarray, pafs: np.ndarray, cfg: SplitHipFusedConfig):
    return SplitHipFusedBackend().run_host(heatmaps, pafs, cfg)
