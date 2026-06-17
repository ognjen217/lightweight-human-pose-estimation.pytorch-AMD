"""Runtime patch for fused HIP1+HIP2 split postprocess backend.

This builds on the existing split HIP patches and adds the backend value:

    --split-paf-backend hip_fused_host

for ``split_hip2_host_smart``.  It replaces the Python-level composition:

    HIP smart heatmap TopK call -> host top_scores/top_indices -> HIP2 PAF call

with one fused shared-library call:

    heatmaps+pafs -> HIP smart TopK -> HIP2 PAF prune -> final small tensors

Only the final TopK and limb tensors are copied back to host for CPU assembly.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from simulation import split_hip_batch_handoff_patch as handoff_patch
from simulation import split_hip_smart_patch as split_base

_ORIGINALS: Dict[str, Any] = {}
_FUSED_BACKENDS = {"hip_fused_host", "hip-fused-host", "fused_hip_host", "fused-hip-host"}


def _backend_key(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _is_fused_backend(value: str) -> bool:
    key = _backend_key(value)
    return key in _FUSED_BACKENDS or key.replace("-", "_") in _FUSED_BACKENDS


def _split_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _split_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _run_split_hip_smart_batch_patched(
    *,
    batch_items: Sequence[Dict[str, Any]],
    map_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    runtime: Dict[str, Any],
    threshold: float,
    use_hip2: bool,
) -> List[Any]:
    backend = os.environ.get("STREAM_SPLIT_HIP_PAF_BACKEND", "hip_host")
    if not (use_hip2 and _is_fused_backend(backend)):
        return _ORIGINALS["_run_split_hip_smart_batch"](
            batch_items=batch_items,
            map_pairs=map_pairs,
            runtime=runtime,
            threshold=threshold,
            use_hip2=use_hip2,
        )

    if not batch_items:
        return []

    compiled_batch_size = _split_env_int("STREAM_SPLIT_HIP_BATCH_SIZE", 2)
    smart_proposals = _split_env_int("STREAM_SPLIT_HIP_SMART_PROPOSALS", 32)
    smart_local_radius = _split_env_int("STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS", 4)
    smart_lowres_nms_radius = _split_env_int("STREAM_SPLIT_HIP_SMART_LOWRES_NMS_RADIUS", 1)
    topk = _split_env_int("STREAM_SPLIT_HIP_TOPK", 20)
    limb_topm = _split_env_int("STREAM_SPLIT_HIP_LIMB_TOPM", 20)

    heat_bchw = np.stack([split_base._heatmap_to_chw(hm) for hm, _pf in map_pairs], axis=0)
    paf_bchw = np.stack([split_base._paf_to_chw(pf) for _hm, pf in map_pairs], axis=0)
    heat_bchw, real_n = split_base._pad_batch(heat_bchw, compiled_batch_size)
    paf_bchw, _ = split_base._pad_batch(paf_bchw, compiled_batch_size)

    from modules.external_split_hip_fused import SplitHipFusedConfig, run_external_split_hip_fused

    fused_cfg = SplitHipFusedConfig(
        batch_size=int(heat_bchw.shape[0]),
        in_h=int(heat_bchw.shape[2]),
        in_w=int(heat_bchw.shape[3]),
        full_h=int(batch_items[0]["original_hw"][0]),
        full_w=int(batch_items[0]["original_hw"][1]),
        heatmap_channels=18,
        paf_channels=38,
        topk=topk,
        limb_topm=limb_topm,
        threshold=float(threshold),
        lowres_nms_radius=smart_lowres_nms_radius,
        smart_proposals=smart_proposals,
        smart_local_radius=smart_local_radius,
        points_per_limb=_split_env_int("STREAM_SPLIT_HIP_POINTS_PER_LIMB", 8),
        min_paf_score=_split_env_float("STREAM_SPLIT_HIP_MIN_PAF_SCORE", 0.05),
        success_ratio_thr=_split_env_float("STREAM_SPLIT_HIP_SUCCESS_RATIO_THR", 0.8),
        min_pair_score=_split_env_float("STREAM_SPLIT_HIP_MIN_PAIR_SCORE", 0.0),
        paf_cubic_a=_split_env_float("STREAM_SPLIT_HIP_PAF_CUBIC_A", -0.75),
    )

    t0 = time.perf_counter()
    top_scores, top_indices, a_idx, b_idx, pair_score, pair_valid = run_external_split_hip_fused(
        heat_bchw,
        paf_bchw,
        fused_cfg,
    )
    t1 = time.perf_counter()
    fused_ms_total = (t1 - t0) * 1000.0
    pair_out = {
        "limb_top_pair_a_idx": a_idx,
        "limb_top_pair_b_idx": b_idx,
        "limb_top_pair_score": pair_score,
        "limb_top_pair_valid": pair_valid,
    }

    assemble_poses_from_pruned_pairs = runtime["assemble_poses_from_pruned_pairs"]
    PostprocessOutput = runtime["PostprocessOutput"]

    outputs = []
    for i, item in enumerate(batch_items[:real_n]):
        t_asm0 = time.perf_counter()
        poses, keypoints, asm_times = assemble_poses_from_pruned_pairs(
            split_base._slice_batched(top_scores, i),
            split_base._slice_batched(top_indices, i),
            split_base._slice_batched(pair_out["limb_top_pair_a_idx"], i),
            split_base._slice_batched(pair_out["limb_top_pair_b_idx"], i),
            split_base._slice_batched(pair_out["limb_top_pair_score"], i),
            split_base._slice_batched(pair_out["limb_top_pair_valid"], i),
            full_width=int(item["original_hw"][1]),
            threshold=float(threshold),
            min_pair_score=0.0,
            return_timing=True,
        )
        asm_ms = (time.perf_counter() - t_asm0) * 1000.0
        valid_topk = float(np.sum(split_base._slice_batched(top_scores, i) > -1.0e8))
        limb_valid = float(np.sum(split_base._slice_batched(pair_out["limb_top_pair_valid"], i) > 0.5))
        per_frame_fused_ms = fused_ms_total / float(max(1, real_n))
        timings: Dict[str, float] = {
            "split_fused_hip": per_frame_fused_ms,
            "split_fused_hip_batch": fused_ms_total,
            "split_smart_heatmap": 0.0,
            "split_smart_heatmap_batch": 0.0,
            "split_hip2": per_frame_fused_ms,
            "split_hip2_batch": fused_ms_total,
            "split_mxr2_replaced_by_hip2": per_frame_fused_ms,
            "split_pair_backend": 3.0,
            "split_cpu_assembly": asm_ms,
            "split_total_batch": fused_ms_total,
            "split_real_batch_size": float(real_n),
            "split_compiled_batch_size": float(heat_bchw.shape[0]),
            "valid_topk_count": valid_topk,
            "limb_valid_count": limb_valid,
            "total_postprocess": per_frame_fused_ms + asm_ms,
        }
        for k, v in dict(asm_times).items():
            try:
                timings[str(k)] = float(v)
            except Exception:
                pass
        outputs.append(PostprocessOutput(np.asarray(poses, dtype=np.float32), np.asarray(keypoints, dtype=np.float32), timings))
    return outputs


def _patch_cli_choices() -> None:
    import simulation.cli as cli

    previous_build_parser = cli.build_parser
    _ORIGINALS.setdefault("build_parser", previous_build_parser)

    def build_parser_patched():
        parser = previous_build_parser()
        for action in parser._actions:
            if getattr(action, "dest", None) == "split_paf_backend" and action.choices is not None:
                choices = list(action.choices)
                if "hip_fused_host" not in choices:
                    choices.append("hip_fused_host")
                action.choices = choices
                break
        return parser

    cli.build_parser = build_parser_patched


def _patch_runtime() -> None:
    _ORIGINALS.setdefault("_run_split_hip_smart_batch", split_base._run_split_hip_smart_batch)
    split_base._run_split_hip_smart_batch = _run_split_hip_smart_batch_patched


def apply_patch() -> None:
    if _ORIGINALS.get("applied"):
        return
    handoff_patch.apply_patch()
    _patch_cli_choices()
    _patch_runtime()
    _ORIGINALS["applied"] = True
