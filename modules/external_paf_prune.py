"""External PAF scoring/pruning backends for split HIP2 experiments.

The output contract matches split MXR2:

    pafs + top_scores + top_indices
      -> limb_top_pair_a_idx, limb_top_pair_b_idx, limb_top_pair_score, limb_top_pair_valid

The first backend, ``hip_host``, is a host-mediated correctness baseline.  It is
intended to be compared directly against MXR2 before replacing the stream path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

PafPruneBackendName = Literal["hip_host"]


@dataclass(frozen=True)
class PafPruneConfig:
    batch_size: int
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


def _validate_inputs(pafs: np.ndarray, top_scores: np.ndarray, top_indices: np.ndarray, cfg: PafPruneConfig):
    p = np.ascontiguousarray(np.asarray(pafs, dtype=np.float32))
    s = np.ascontiguousarray(np.asarray(top_scores, dtype=np.float32))
    i = np.ascontiguousarray(np.asarray(top_indices, dtype=np.int64))
    expected_p = (int(cfg.batch_size), 38, int(cfg.in_h), int(cfg.in_w))
    expected_s = (int(cfg.batch_size), 18, int(cfg.topk))
    if tuple(p.shape) != expected_p:
        raise ValueError(f"Expected pafs shape {expected_p}, got {tuple(p.shape)}")
    if tuple(s.shape) != expected_s:
        raise ValueError(f"Expected top_scores shape {expected_s}, got {tuple(s.shape)}")
    if tuple(i.shape) != expected_s:
        raise ValueError(f"Expected top_indices shape {expected_s}, got {tuple(i.shape)}")
    return p, s, i


def _hip_shape(cfg: PafPruneConfig):
    from modules.external_paf_prune_hip import HipPafPruneShape

    return HipPafPruneShape(
        batch=int(cfg.batch_size),
        topk=int(cfg.topk),
        limb_topm=int(cfg.limb_topm),
        in_h=int(cfg.in_h),
        in_w=int(cfg.in_w),
        full_h=int(cfg.full_h),
        full_w=int(cfg.full_w),
        points_per_limb=int(cfg.points_per_limb),
        min_paf_score=float(cfg.min_paf_score),
        success_ratio_thr=float(cfg.success_ratio_thr),
        min_pair_score=float(cfg.min_pair_score),
        paf_cubic_a=float(cfg.paf_cubic_a),
    )


def hip_host_paf_prune(
    pafs: np.ndarray,
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    cfg: PafPruneConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, s, i = _validate_inputs(pafs, top_scores, top_indices, cfg)
    from modules.external_paf_prune_hip import HipPafPruneBackend

    return HipPafPruneBackend().run_host(p, s, i, _hip_shape(cfg))


def run_external_paf_prune(
    pafs: np.ndarray,
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    cfg: PafPruneConfig,
    *,
    backend: PafPruneBackendName = "hip_host",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if backend == "hip_host":
        return hip_host_paf_prune(pafs, top_scores, top_indices, cfg)
    raise ValueError(f"Unsupported external PAF prune backend: {backend}")
