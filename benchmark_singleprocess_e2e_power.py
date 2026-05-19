#!/usr/bin/env python3
"""
benchmark_singleprocess_e2e_power.py

Single-process, end-to-end video speed/power benchmark for the
lightweight-human-pose-estimation.pytorch-AMD MIGraphX pipeline.

This script intentionally does NOT modify video_val.py. It imports the original
PoseEstimator only for:
  - preprocessing
  - MIGraphX pose model inference

All postprocess variants are implemented inside this benchmark so they can be
run under the same single-process conditions and reported with the same timing
schema.

Default variants:
  1. standard_cpu
     Original CPU postprocess:
       full-res heatmap resize + full-res PAF resize
       extract_keypoints per channel
       group_keypoints

  2. optimized_batch_k20_findnonzero_v1_cpu
     Best CPU-only postprocess:
       full-res heatmap resize + full-res PAF resize
       extract_keypoints_batch_cv2(max_keypoints_per_type=20)
       group_keypoints_fast

  3. gpu-fullres-nms-cpu-group
     Hybrid GPU NMS:
       full-res heatmap resize + full-res PAF resize
       torch max_pool2d NMS / top-K extraction on GPU
       group_keypoints_fast on CPU

  4. full-gpu
     Experimental GPU-heavy postprocess:
       full-res heatmap resize + full-res PAF resize
       torch max_pool2d NMS / top-K extraction on GPU
       torch PAF affinity scoring on GPU
       final greedy connection NMS + pose assembly on CPU
     Note: the final OpenPose pose assembly remains CPU because it is dynamic,
     variable-length control flow.

  5. migraphx-nms
     MIGraphX heatmap NMS:
       full-res heatmap resize + full-res PAF resize
       compiled MIGraphX NMS head produces dense peak mask
       CPU extraction from mask
       group_keypoints_fast on CPU

Recommended:
  python benchmark_singleprocess_e2e_power.py \
    --video cctv_1280x720_24fps_original.mp4 \
    --model pose_model1_fp16_ref1.mxr \
    --migraphx-nms-mxr models/heatmap_nms_head.mxr \
    --frames 100 \
    --warmup 5 \
    --csv benchmark_singleprocess_e2e_power_summary.csv \
    --detailed-csv benchmark_singleprocess_e2e_power_detailed.csv \
    --md benchmark_singleprocess_e2e_power_report.md

Power note:
  Power is sampled through rocm-smi. Reported energy and FPS/W are estimates
  based on GPU package power, not whole-system wall power.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from video_val import PoseEstimator
except Exception as exc:
    raise SystemExit(
        "Could not import PoseEstimator from video_val.py. "
        "Run this script from the repository root where video_val.py exists."
    ) from exc

from modules.keypoints import (
    BODY_PARTS_KPT_IDS,
    BODY_PARTS_PAF_IDS,
    connections_nms,
    extract_keypoints,
    extract_keypoints_batch_cv2,
    group_keypoints,
    group_keypoints_fast,
)


Timing = Dict[str, float]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def nanmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def fmt_float(x: Any, digits: int = 2, na: str = "N/A") -> str:
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return na
        return f"{x:.{digits}f}"
    except Exception:
        return na


def torch_device_from_arg(device_arg: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for GPU postprocess variants but could not be imported.")

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg in {"cuda", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_arg == "cuda":
            raise RuntimeError("Requested --torch-device cuda but torch.cuda.is_available() is false.")
        return torch.device("cpu")

    raise ValueError(f"Unsupported torch device: {device_arg}")


def sync_torch_device(device) -> None:
    if torch is None:
        return
    try:
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass


def get_gpu_power_w() -> float:
    """
    Read AMD GPU package power from rocm-smi.

    Returns NaN if rocm-smi is not available or the output cannot be parsed.
    """
    commands = [
        ["rocm-smi", "--showpower"],
        ["/opt/rocm/bin/rocm-smi", "--showpower"],
    ]

    patterns = [
        r"Current\s+Socket\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Average\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Socket\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Average\s+Graphics\s+Package\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
        r"Graphics\s+Package\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
        r"Current\s+Socket\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
        r"Socket\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
        r"Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
    ]

    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
            text = completed.stdout + "\n" + completed.stderr

            if os.environ.get("BENCH_POWER_DEBUG") == "1":
                print("\n--- POWER DEBUG ---")
                print("CMD:", " ".join(cmd))
                print(text)
                print("--- END POWER DEBUG ---\n")

            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception:
            continue

    return float("nan")


# ---------------------------------------------------------------------------
# Shared decode / extraction helpers
# ---------------------------------------------------------------------------
def decode_outputs(results: Any, out_h: int, out_w: int) -> Tuple[np.ndarray, np.ndarray]:
    heatmaps = np.asarray(results[0], dtype=np.float32).reshape(19, out_h, out_w)
    pafs = np.asarray(results[1], dtype=np.float32).reshape(38, out_h, out_w)

    heatmaps = np.moveaxis(heatmaps, 0, -1)  # H x W x 19
    pafs = np.moveaxis(pafs, 0, -1)          # H x W x 38

    return heatmaps, pafs


def resize_fullres(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    timings: Timing,
) -> Tuple[np.ndarray, np.ndarray]:
    orig_h, orig_w = original_hw

    with Timer() as t:
        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmaps = np.ascontiguousarray(heatmaps, dtype=np.float32)
    timings["hm_resize_ms"] = t.ms

    with Timer() as t:
        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        pafs = np.ascontiguousarray(pafs, dtype=np.float32)
    timings["paf_resize_ms"] = t.ms

    return heatmaps, pafs


def extract_keypoints_from_peak_mask(
    heatmaps_hwc: np.ndarray,
    peak_mask_hwc: np.ndarray,
    max_candidates_per_part: Optional[int] = None,
) -> Tuple[List[List[Tuple[int, int, float, int]]], int]:
    """
    Convert dense peak mask into the all_keypoints_by_type format used by
    group_keypoints/group_keypoints_fast.

    heatmaps_hwc and peak_mask_hwc must have matching H x W x C shape.
    Only the first 18 channels are used for body keypoints.
    """
    all_keypoints_by_type: List[List[Tuple[int, int, float, int]]] = []
    total = 0

    channels = min(18, heatmaps_hwc.shape[2], peak_mask_hwc.shape[2])

    for kpt_idx in range(channels):
        ys, xs = np.nonzero(peak_mask_hwc[:, :, kpt_idx] > 0)

        if len(xs) == 0:
            all_keypoints_by_type.append([])
            continue

        scores = heatmaps_hwc[ys, xs, kpt_idx].astype(np.float32)

        order = np.argsort(scores)[::-1]
        if max_candidates_per_part is not None:
            order = order[:max_candidates_per_part]

        xs = xs[order]
        ys = ys[order]
        scores = scores[order]

        keypoints = []
        for i in range(len(xs)):
            keypoints.append(
                (int(xs[i]), int(ys[i]), float(scores[i]), total + i)
            )

        all_keypoints_by_type.append(keypoints)
        total += len(keypoints)

    while len(all_keypoints_by_type) < 18:
        all_keypoints_by_type.append([])

    return all_keypoints_by_type, total


def call_group_keypoints_fast(all_kpts, pafs, timings: Timing) -> Tuple[np.ndarray, np.ndarray]:
    with Timer() as t:
        out = group_keypoints_fast(
            all_kpts,
            pafs,
            points_per_limb=8,
            return_timing=True,
        )

    timings["group_ms"] = t.ms

    if isinstance(out, tuple) and len(out) == 3:
        poses, kpts, group_times = out
        # Normalize group timing field names.
        for k, v in group_times.items():
            try:
                timings[k] = float(v)
            except Exception:
                pass
    else:
        poses, kpts = out  # type: ignore

    return np.asarray(poses), np.asarray(kpts)


# ---------------------------------------------------------------------------
# Variant 1: original standard CPU
# ---------------------------------------------------------------------------
def postprocess_standard_cpu(results: Any, original_hw: Tuple[int, int], engine: Any, ctx: Dict[str, Any]):
    timings: Timing = {}
    t_total = time.perf_counter()

    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    timings["mx_nms_ms"] = 0.0
    timings["extract_from_mask_ms"] = 0.0

    heatmaps, pafs = resize_fullres(heatmaps, pafs, original_hw, timings)

    with Timer() as t:
        all_kpts = []
        total = 0
        for kpt_idx in range(18):
            total += extract_keypoints(heatmaps[:, :, kpt_idx], all_kpts, total)
    timings["extract_ms"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs, points_per_limb=8)
    timings["group_ms"] = t.ms

    timings["post_ms"] = (time.perf_counter() - t_total) * 1000.0
    return np.asarray(poses), np.asarray(kpts), timings


# ---------------------------------------------------------------------------
# Variant 2: best CPU-only K20/findNonZero + fast group
# ---------------------------------------------------------------------------
def postprocess_k20_findnonzero_cpu(results: Any, original_hw: Tuple[int, int], engine: Any, ctx: Dict[str, Any]):
    timings: Timing = {}
    t_total = time.perf_counter()

    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    timings["mx_nms_ms"] = 0.0
    timings["extract_from_mask_ms"] = 0.0

    heatmaps, pafs = resize_fullres(heatmaps, pafs, original_hw, timings)

    with Timer() as t:
        all_kpts, _ = extract_keypoints_batch_cv2(
            heatmaps[:, :, :18],
            max_keypoints_per_type=20,
        )
    timings["extract_ms"] = t.ms

    poses, kpts = call_group_keypoints_fast(all_kpts, pafs, timings)

    timings["post_ms"] = (time.perf_counter() - t_total) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------------------------
# Variant 3: full-res torch GPU NMS + CPU fast group
# ---------------------------------------------------------------------------
def torch_extract_keypoints_gpu_nms_fullres(
    heatmaps: np.ndarray,
    device,
    max_keypoints_per_type: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
) -> Tuple[List[List[Tuple[int, int, float, int]]], int]:
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for GPU NMS variant.")

    heatmaps_np = np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float32)

    heatmaps_t = (
        torch.from_numpy(heatmaps_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    kernel_size = 2 * nms_radius + 1
    pooled = F.max_pool2d(
        heatmaps_t,
        kernel_size=kernel_size,
        stride=1,
        padding=nms_radius,
    )

    peaks = (heatmaps_t == pooled) & (heatmaps_t > threshold)

    all_keypoints_by_type: List[List[Tuple[int, int, float, int]]] = []
    total = 0

    for kpt_idx in range(18):
        coords = torch.nonzero(peaks[0, kpt_idx], as_tuple=False)

        if coords.numel() == 0:
            all_keypoints_by_type.append([])
            continue

        ys = coords[:, 0]
        xs = coords[:, 1]
        scores = heatmaps_t[0, kpt_idx, ys, xs]

        keep = min(max_keypoints_per_type, int(scores.numel()))
        top_scores, order = torch.topk(scores, k=keep, largest=True, sorted=True)

        xs_np = xs[order].detach().cpu().numpy()
        ys_np = ys[order].detach().cpu().numpy()
        scores_np = top_scores.detach().cpu().numpy()

        keypoints = []
        for i in range(keep):
            keypoints.append((int(xs_np[i]), int(ys_np[i]), float(scores_np[i]), total + i))

        all_keypoints_by_type.append(keypoints)
        total += len(keypoints)

    return all_keypoints_by_type, total


def postprocess_gpu_fullres_nms_cpu_group(results: Any, original_hw: Tuple[int, int], engine: Any, ctx: Dict[str, Any]):
    timings: Timing = {}
    t_total = time.perf_counter()

    device = ctx["torch_device"]
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    timings["mx_nms_ms"] = 0.0
    timings["extract_from_mask_ms"] = 0.0

    heatmaps, pafs = resize_fullres(heatmaps, pafs, original_hw, timings)

    with Timer() as t:
        all_kpts, _ = torch_extract_keypoints_gpu_nms_fullres(
            heatmaps,
            device=device,
            max_keypoints_per_type=20,
            threshold=ctx["threshold"],
            nms_radius=ctx["nms_radius"],
        )
        sync_torch_device(device)
    timings["extract_ms"] = t.ms

    poses, kpts = call_group_keypoints_fast(all_kpts, pafs, timings)

    timings["post_ms"] = (time.perf_counter() - t_total) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------------------------
# Variant 4: experimental full-gpu-ish NMS + GPU PAF affinity + CPU assembly
# ---------------------------------------------------------------------------
def assemble_pose_entries_from_connections(
    all_keypoints_by_type: List[List[Tuple[int, int, float, int]]],
    connections_by_part: List[List[Tuple[int, int, float]]],
    pose_entry_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    non_empty = [
        np.asarray(kpts, dtype=np.float32)
        for kpts in all_keypoints_by_type
        if len(kpts) > 0
    ]
    if non_empty:
        all_keypoints = np.concatenate(non_empty, axis=0)
    else:
        all_keypoints = np.empty((0, 4), dtype=np.float32)

    pose_entries: List[np.ndarray] = []

    for part_id, connections in enumerate(connections_by_part):
        if not connections:
            continue

        if part_id == 0:
            pose_entries = [
                np.ones(pose_entry_size, dtype=np.float32) * -1
                for _ in range(len(connections))
            ]

            for i, (a_id, b_id, score) in enumerate(connections):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = a_id
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = b_id
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = (
                    np.sum(all_keypoints[[a_id, b_id], 2]) + score
                    if all_keypoints.shape[0] > max(a_id, b_id)
                    else score
                )

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            for a_id, b_id, _score in connections:
                for pose in pose_entries:
                    if pose[kpt_a_id] == a_id and pose[kpt_b_id] == -1:
                        pose[kpt_b_id] = b_id
                    elif pose[kpt_b_id] == b_id and pose[kpt_a_id] == -1:
                        pose[kpt_a_id] = a_id

        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            for a_id, b_id, score in connections:
                attached = 0

                for pose in pose_entries:
                    if pose[kpt_a_id] == a_id:
                        pose[kpt_b_id] = b_id
                        attached += 1
                        pose[-1] += 1

                        if all_keypoints.shape[0] > b_id:
                            pose[-2] += all_keypoints[b_id, 2] + score
                        else:
                            pose[-2] += score

                if attached == 0:
                    pose_entry = np.ones(pose_entry_size, dtype=np.float32) * -1
                    pose_entry[kpt_a_id] = a_id
                    pose_entry[kpt_b_id] = b_id
                    pose_entry[-1] = 2
                    pose_entry[-2] = (
                        np.sum(all_keypoints[[a_id, b_id], 2]) + score
                        if all_keypoints.shape[0] > max(a_id, b_id)
                        else score
                    )
                    pose_entries.append(pose_entry)

    filtered = []
    for pose in pose_entries:
        if pose[-1] < 3:
            continue
        if pose[-2] / pose[-1] < 0.2:
            continue
        filtered.append(pose)

    pose_entries_arr = np.asarray(filtered, dtype=np.float32)
    return pose_entries_arr, all_keypoints


def group_keypoints_gpu_paf(
    all_keypoints_by_type: List[List[Tuple[int, int, float, int]]],
    pafs: np.ndarray,
    device,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Timing]:
    if torch is None:
        raise RuntimeError("PyTorch is required for full-gpu variant.")

    timings: Timing = {}
    t_group = time.perf_counter()
    tm_prepare = tm_pairs = tm_sample = tm_affinity = tm_nms = tm_pose = tm_filter = 0.0

    t0 = time.perf_counter()
    all_np = [np.asarray(kpts, dtype=np.float32) for kpts in all_keypoints_by_type]
    pafs_t = torch.from_numpy(np.ascontiguousarray(pafs, dtype=np.float32)).to(device)
    grid = torch.arange(points_per_limb, device=device, dtype=torch.float32).view(1, -1, 1)
    paf_h, paf_w = pafs.shape[:2]
    connections_by_part: List[List[Tuple[int, int, float]]] = []
    tm_prepare += time.perf_counter() - t0

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        t0 = time.perf_counter()

        paf_x_id, paf_y_id = BODY_PARTS_PAF_IDS[part_id]
        kpt_a_type = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_type = BODY_PARTS_KPT_IDS[part_id][1]

        kpts_a_np = all_np[kpt_a_type]
        kpts_b_np = all_np[kpt_b_type]

        n = len(kpts_a_np)
        m = len(kpts_b_np)

        if n == 0 or m == 0:
            connections_by_part.append([])
            tm_prepare += time.perf_counter() - t0
            continue

        a = torch.from_numpy(np.ascontiguousarray(kpts_a_np[:, :2], dtype=np.float32)).to(device)
        b = torch.from_numpy(np.ascontiguousarray(kpts_b_np[:, :2], dtype=np.float32)).to(device)

        tm_prepare += time.perf_counter() - t0

        # Candidate limb vectors.
        t0 = time.perf_counter()

        vec_raw = (b[:, None, :] - a[None, :, :]).reshape(-1, 2)
        vec_norm = torch.linalg.norm(vec_raw, dim=-1, keepdim=True)
        valid_vec = vec_norm.squeeze(-1) > 1e-6

        if not bool(torch.any(valid_vec).detach().cpu().item()):
            connections_by_part.append([])
            tm_pairs += time.perf_counter() - t0
            continue

        pair_ids = torch.nonzero(valid_vec, as_tuple=False).flatten()
        vec_raw_valid = vec_raw[valid_vec]
        vec_norm_valid = vec_norm[valid_vec]

        b_pair_idx = torch.div(pair_ids, n, rounding_mode="floor")
        a_pair_idx = pair_ids % n

        tm_pairs += time.perf_counter() - t0

        # Sample points along candidate limbs.
        t0 = time.perf_counter()

        steps = vec_raw_valid.view(-1, 1, 2) / float(points_per_limb - 1)
        a_points = a[a_pair_idx].view(-1, 1, 2)

        points = torch.round(steps * grid + a_points).to(torch.long)

        x = torch.clamp(points[..., 0].reshape(-1), 0, paf_w - 1)
        y = torch.clamp(points[..., 1].reshape(-1), 0, paf_h - 1)

        tm_sample += time.perf_counter() - t0

        # GPU PAF affinity.
        t0 = time.perf_counter()

        field_x = pafs_t[y, x, paf_x_id]
        field_y = pafs_t[y, x, paf_y_id]
        field = torch.stack((field_x, field_y), dim=-1).view(-1, points_per_limb, 2)

        vec_unit = (vec_raw_valid / (vec_norm_valid + 1e-6)).view(-1, 1, 2)

        affinity_per_point = (field * vec_unit).sum(dim=-1)
        valid_affinity = affinity_per_point > min_paf_score
        valid_num = valid_affinity.sum(dim=1).to(torch.float32)

        affinity_scores = (
            affinity_per_point * valid_affinity.to(torch.float32)
        ).sum(dim=1) / (valid_num + 1e-6)

        success_ratio = valid_num / float(points_per_limb)

        valid_limb_local = torch.nonzero(
            (affinity_scores > 0) & (success_ratio > success_ratio_thr),
            as_tuple=False,
        ).flatten()

        sync_torch_device(device)

        if valid_limb_local.numel() == 0:
            connections_by_part.append([])
            tm_affinity += time.perf_counter() - t0
            continue

        a_idx_np = a_pair_idx[valid_limb_local].detach().cpu().numpy().astype(np.int32)
        b_idx_np = b_pair_idx[valid_limb_local].detach().cpu().numpy().astype(np.int32)
        scores_np = affinity_scores[valid_limb_local].detach().cpu().numpy().astype(np.float32)

        tm_affinity += time.perf_counter() - t0

        # CPU greedy NMS for candidate connections.
        t0 = time.perf_counter()

        a_idx_np, b_idx_np, scores_np = connections_nms(a_idx_np, b_idx_np, scores_np)

        connections = list(
            zip(
                kpts_a_np[a_idx_np, 3].astype(np.int32).tolist(),
                kpts_b_np[b_idx_np, 3].astype(np.int32).tolist(),
                scores_np.astype(np.float32).tolist(),
            )
        )

        connections_by_part.append(connections)
        tm_nms += time.perf_counter() - t0

    t0 = time.perf_counter()
    poses, all_keypoints = assemble_pose_entries_from_connections(
        all_keypoints_by_type,
        connections_by_part,
        pose_entry_size=20,
    )
    tm_pose += time.perf_counter() - t0

    timings["group_prepare"] = tm_prepare * 1000.0
    timings["group_pairs"] = tm_pairs * 1000.0
    timings["group_sample"] = tm_sample * 1000.0
    timings["group_affinity"] = tm_affinity * 1000.0
    timings["group_nms"] = tm_nms * 1000.0
    timings["group_pose"] = tm_pose * 1000.0
    timings["group_filter"] = tm_filter * 1000.0
    timings["group_total"] = (time.perf_counter() - t_group) * 1000.0

    return poses, all_keypoints, timings


def postprocess_full_gpu(results: Any, original_hw: Tuple[int, int], engine: Any, ctx: Dict[str, Any]):
    timings: Timing = {}
    t_total = time.perf_counter()

    device = ctx["torch_device"]
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    timings["mx_nms_ms"] = 0.0
    timings["extract_from_mask_ms"] = 0.0

    heatmaps, pafs = resize_fullres(heatmaps, pafs, original_hw, timings)

    with Timer() as t:
        all_kpts, _ = torch_extract_keypoints_gpu_nms_fullres(
            heatmaps,
            device=device,
            max_keypoints_per_type=20,
            threshold=ctx["threshold"],
            nms_radius=ctx["nms_radius"],
        )
        sync_torch_device(device)
    timings["extract_ms"] = t.ms

    with Timer() as t:
        poses, kpts, group_times = group_keypoints_gpu_paf(
            all_kpts,
            pafs,
            device=device,
            points_per_limb=8,
        )
        sync_torch_device(device)
    timings["group_ms"] = t.ms
    timings.update(group_times)

    timings["post_ms"] = (time.perf_counter() - t_total) * 1000.0
    return np.asarray(poses), np.asarray(kpts), timings


# ---------------------------------------------------------------------------
# Variant 5: MIGraphX NMS + CPU extraction/grouping
# ---------------------------------------------------------------------------
def load_migraphx_nms_head(mxr_path: str):
    try:
        from modules.migraphx_nms import MIGraphXNMSHead
    except Exception as exc:
        raise RuntimeError(
            "Could not import modules.migraphx_nms.MIGraphXNMSHead. "
            "Make sure modules/migraphx_nms.py exists in the repo."
        ) from exc

    return MIGraphXNMSHead(mxr_path, input_name="heatmaps")


def postprocess_migraphx_nms(results: Any, original_hw: Tuple[int, int], engine: Any, ctx: Dict[str, Any]):
    timings: Timing = {}
    t_total = time.perf_counter()

    mx_nms_head = ctx.get("migraphx_nms_head")
    if mx_nms_head is None:
        raise RuntimeError("migraphx-nms variant requested but no NMS head is loaded.")

    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride
    orig_h, orig_w = original_hw

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    heatmaps, pafs = resize_fullres(heatmaps, pafs, original_hw, timings)

    # Full-res HWC -> NCHW for MIGraphX NMS head.
    heatmaps_nchw = np.moveaxis(heatmaps, -1, 0)[np.newaxis, :, :, :]
    heatmaps_nchw = np.ascontiguousarray(heatmaps_nchw, dtype=np.float32)

    with Timer() as t:
        peak_mask_nchw = mx_nms_head.run(heatmaps_nchw)
    timings["mx_nms_ms"] = t.ms

    with Timer() as t:
        peak_mask_nchw = np.asarray(peak_mask_nchw, dtype=np.float32).reshape(1, 19, orig_h, orig_w)
        peak_mask_hwc = np.moveaxis(peak_mask_nchw.squeeze(0), 0, -1)
        peak_mask_hwc = np.ascontiguousarray(peak_mask_hwc, dtype=np.float32)

        all_kpts, _ = extract_keypoints_from_peak_mask(
            heatmaps[:, :, :18],
            peak_mask_hwc[:, :, :18],
            max_candidates_per_part=ctx.get("migraphx_max_candidates"),
        )
    timings["extract_from_mask_ms"] = t.ms
    timings["extract_ms"] = t.ms

    poses, kpts = call_group_keypoints_fast(all_kpts, pafs, timings)

    timings["post_ms"] = (time.perf_counter() - t_total) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
VARIANT_SPECS = {
    "standard_cpu": {
        "fn": postprocess_standard_cpu,
        "description": "Original full-res CPU: extract_keypoints per channel + group_keypoints.",
        "requires_torch": False,
        "requires_migraphx_nms": False,
    },
    "optimized_batch_k20_findnonzero_v1_cpu": {
        "fn": postprocess_k20_findnonzero_cpu,
        "description": "Best CPU-only: full-res batch cv2/findNonZero K20 extraction + group_keypoints_fast.",
        "requires_torch": False,
        "requires_migraphx_nms": False,
    },
    "gpu-fullres-nms-cpu-group": {
        "fn": postprocess_gpu_fullres_nms_cpu_group,
        "description": "Full-res torch GPU NMS/top-K extraction + CPU group_keypoints_fast.",
        "requires_torch": True,
        "requires_migraphx_nms": False,
    },
    "full-gpu": {
        "fn": postprocess_full_gpu,
        "description": "Experimental: torch GPU NMS/top-K + torch GPU PAF affinity scoring; final dynamic pose assembly on CPU.",
        "requires_torch": True,
        "requires_migraphx_nms": False,
    },
    "migraphx-nms": {
        "fn": postprocess_migraphx_nms,
        "description": "Full-res MIGraphX NMS dense peak mask + CPU extract_from_mask + CPU group_keypoints_fast.",
        "requires_torch": False,
        "requires_migraphx_nms": True,
        "migraphx_max_candidates": None,
    },
    "migraphx-nms-k20": {
        "fn": postprocess_migraphx_nms,
        "description": "Full-res MIGraphX NMS dense peak mask + top-K=20 CPU extract_from_mask + CPU group_keypoints_fast.",
        "requires_torch": False,
        "requires_migraphx_nms": True,
        "migraphx_max_candidates": 20,
    },
}


DEFAULT_VARIANTS = [
    "standard_cpu",
    "optimized_batch_k20_findnonzero_v1_cpu",
    "gpu-fullres-nms-cpu-group",
    "full-gpu",
    "migraphx-nms",
]


SUMMARY_FIELDS = [
    "variant",
    "description",
    "frames",
    "video_width",
    "video_height",
    "video_fps_metadata",
    "preprocess_mean_ms",
    "preprocess_p95_ms",
    "inference_mean_ms",
    "inference_p95_ms",
    "decode_mean_ms",
    "hm_resize_mean_ms",
    "paf_resize_mean_ms",
    "mx_nms_mean_ms",
    "extract_mean_ms",
    "extract_from_mask_mean_ms",
    "group_mean_ms",
    "post_mean_ms",
    "post_p95_ms",
    "e2e_compute_mean_ms",
    "e2e_compute_p95_ms",
    "loop_total_mean_ms",
    "loop_total_p95_ms",
    "e2e_fps",
    "loop_fps",
    "avg_gpu_power_w",
    "min_gpu_power_w",
    "max_gpu_power_w",
    "energy_j_per_frame",
    "fps_per_watt",
    "e2e_speedup_vs_standard",
    "post_speedup_vs_standard",
]


DETAILED_FIELDS = [
    "variant",
    "frame_idx",
    "measured_frame_idx",
    "read_ms",
    "preprocess_ms",
    "inference_ms",
    "decode_ms",
    "hm_resize_ms",
    "paf_resize_ms",
    "mx_nms_ms",
    "extract_ms",
    "extract_from_mask_ms",
    "group_ms",
    "post_ms",
    "e2e_compute_ms",
    "loop_total_ms",
    "power_w",
]


def summarize_rows(
    variant: str,
    description: str,
    rows: List[Dict[str, Any]],
    power_samples: List[float],
    video_meta: Dict[str, Any],
    baseline_e2e_ms: Optional[float],
    baseline_post_ms: Optional[float],
) -> Dict[str, Any]:
    def col(name: str) -> List[float]:
        return [float(r.get(name, 0.0) or 0.0) for r in rows]

    e2e_mean = mean(col("e2e_compute_ms"))
    post_mean = mean(col("post_ms"))

    avg_power = nanmean(power_samples)
    min_power = float(np.nanmin(power_samples)) if power_samples and not all(math.isnan(p) for p in power_samples) else float("nan")
    max_power = float(np.nanmax(power_samples)) if power_samples and not all(math.isnan(p) for p in power_samples) else float("nan")

    e2e_fps = 1000.0 / e2e_mean if e2e_mean > 0 else 0.0
    loop_mean = mean(col("loop_total_ms"))
    loop_fps = 1000.0 / loop_mean if loop_mean > 0 else 0.0

    energy_j = avg_power * (e2e_mean / 1000.0) if avg_power > 0 and not math.isnan(avg_power) else float("nan")
    fps_per_watt = e2e_fps / avg_power if avg_power > 0 and not math.isnan(avg_power) else float("nan")

    e2e_speedup = (baseline_e2e_ms / e2e_mean) if baseline_e2e_ms and e2e_mean > 0 else 1.0
    post_speedup = (baseline_post_ms / post_mean) if baseline_post_ms and post_mean > 0 else 1.0

    return {
        "variant": variant,
        "description": description,
        "frames": len(rows),
        "video_width": video_meta.get("width", 0),
        "video_height": video_meta.get("height", 0),
        "video_fps_metadata": video_meta.get("fps", 0.0),
        "preprocess_mean_ms": mean(col("preprocess_ms")),
        "preprocess_p95_ms": percentile(col("preprocess_ms"), 95),
        "inference_mean_ms": mean(col("inference_ms")),
        "inference_p95_ms": percentile(col("inference_ms"), 95),
        "decode_mean_ms": mean(col("decode_ms")),
        "hm_resize_mean_ms": mean(col("hm_resize_ms")),
        "paf_resize_mean_ms": mean(col("paf_resize_ms")),
        "mx_nms_mean_ms": mean(col("mx_nms_ms")),
        "extract_mean_ms": mean(col("extract_ms")),
        "extract_from_mask_mean_ms": mean(col("extract_from_mask_ms")),
        "group_mean_ms": mean(col("group_ms")),
        "post_mean_ms": post_mean,
        "post_p95_ms": percentile(col("post_ms"), 95),
        "e2e_compute_mean_ms": e2e_mean,
        "e2e_compute_p95_ms": percentile(col("e2e_compute_ms"), 95),
        "loop_total_mean_ms": loop_mean,
        "loop_total_p95_ms": percentile(col("loop_total_ms"), 95),
        "e2e_fps": e2e_fps,
        "loop_fps": loop_fps,
        "avg_gpu_power_w": avg_power,
        "min_gpu_power_w": min_power,
        "max_gpu_power_w": max_power,
        "energy_j_per_frame": energy_j,
        "fps_per_watt": fps_per_watt,
        "e2e_speedup_vs_standard": e2e_speedup,
        "post_speedup_vs_standard": post_speedup,
    }


def print_variant_descriptions(variants: Sequence[str]) -> None:
    print("\nVariants:")
    for name in variants:
        spec = VARIANT_SPECS[name]
        print(f"  {name:<40} {spec['description']}")


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 190)
    print("SINGLE-PROCESS E2E VIDEO BENCHMARK SUMMARY")
    print("=" * 190)
    print(
        f"{'variant':<40} "
        f"{'frames':>6} "
        f"{'pre':>8} "
        f"{'infer':>8} "
        f"{'post':>9} "
        f"{'e2e':>9} "
        f"{'p95':>9} "
        f"{'FPS':>8} "
        f"{'Power':>8} "
        f"{'J/frame':>9} "
        f"{'FPS/W':>8} "
        f"{'post_spd':>9} "
        f"{'e2e_spd':>9} "
        f"{'mx_nms':>8} "
        f"{'extract':>9} "
        f"{'group':>8}"
    )
    print("-" * 190)

    for r in rows:
        print(
            f"{r['variant']:<40.40} "
            f"{int(r['frames']):>6} "
            f"{r['preprocess_mean_ms']:>8.2f} "
            f"{r['inference_mean_ms']:>8.2f} "
            f"{r['post_mean_ms']:>9.2f} "
            f"{r['e2e_compute_mean_ms']:>9.2f} "
            f"{r['e2e_compute_p95_ms']:>9.2f} "
            f"{r['e2e_fps']:>8.2f} "
            f"{fmt_float(r['avg_gpu_power_w']):>8} "
            f"{fmt_float(r['energy_j_per_frame'], 4):>9} "
            f"{fmt_float(r['fps_per_watt']):>8} "
            f"{r['post_speedup_vs_standard']:>9.2f} "
            f"{r['e2e_speedup_vs_standard']:>9.2f} "
            f"{r['mx_nms_mean_ms']:>8.2f} "
            f"{r['extract_mean_ms']:>9.2f} "
            f"{r['group_mean_ms']:>8.2f}"
        )

    print("=" * 190)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved: {path}")


def write_markdown_report(path: str, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if not path:
        return

    lines = []
    lines.append("# Single-process E2E postprocess benchmark\n")
    lines.append("This benchmark measures speed only, not AP/AR accuracy.\n")
    lines.append("All variants are executed as separate single-process video passes: frame read, preprocess, MIGraphX inference, and one selected postprocess path. Drawing and video writing are intentionally excluded.\n")
    lines.append("GPU power is sampled with `rocm-smi`; energy and FPS/W are estimates based on GPU package power, not whole-system power.\n")

    lines.append("## Command context\n")
    lines.append(f"- Video: `{args.video}`")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Frames: `{args.frames}` measured, `{args.warmup}` warmup")
    lines.append(f"- MIGraphX NMS MXR: `{args.migraphx_nms_mxr}`")
    lines.append("")

    lines.append("## Variant definitions\n")
    for name in args.variants:
        lines.append(f"- `{name}`: {VARIANT_SPECS[name]['description']}")
    lines.append("")

    lines.append("## Summary\n")
    headers = [
        "variant", "pre ms", "infer ms", "post ms", "e2e ms", "p95 ms",
        "FPS", "Power W", "J/frame", "FPS/W", "post speedup", "e2e speedup",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                str(r["variant"]),
                f"{r['preprocess_mean_ms']:.2f}",
                f"{r['inference_mean_ms']:.2f}",
                f"{r['post_mean_ms']:.2f}",
                f"{r['e2e_compute_mean_ms']:.2f}",
                f"{r['e2e_compute_p95_ms']:.2f}",
                f"{r['e2e_fps']:.2f}",
                fmt_float(r["avg_gpu_power_w"]),
                fmt_float(r["energy_j_per_frame"], 4),
                fmt_float(r["fps_per_watt"]),
                f"{r['post_speedup_vs_standard']:.2f}x",
                f"{r['e2e_speedup_vs_standard']:.2f}x",
            ])
            + " |"
        )

    lines.append("")
    lines.append("## Postprocess breakdown\n")
    headers = [
        "variant", "decode", "hm resize", "paf resize", "mx nms", "extract",
        "mask extract", "group", "post total",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                str(r["variant"]),
                f"{r['decode_mean_ms']:.2f}",
                f"{r['hm_resize_mean_ms']:.2f}",
                f"{r['paf_resize_mean_ms']:.2f}",
                f"{r['mx_nms_mean_ms']:.2f}",
                f"{r['extract_mean_ms']:.2f}",
                f"{r['extract_from_mask_mean_ms']:.2f}",
                f"{r['group_mean_ms']:.2f}",
                f"{r['post_mean_ms']:.2f}",
            ])
            + " |"
        )

    lines.append("")
    lines.append("## Notes\n")
    lines.append("- `standard_cpu` uses the original CPU keypoint extraction and original CPU grouping.")
    lines.append("- `optimized_batch_k20_findnonzero_v1_cpu` is the best CPU-only path: batched K20 extraction plus `group_keypoints_fast`.")
    lines.append("- `gpu-fullres-nms-cpu-group` only moves heatmap NMS/top-K extraction to torch GPU; grouping stays CPU fast group.")
    lines.append("- `full-gpu` is GPU-heavy but not literally 100% GPU: NMS and PAF affinity scoring run on torch GPU, while final dynamic greedy pose assembly remains CPU.")
    lines.append("- `migraphx-nms` uses one fixed full-resolution NMS MXR, appropriate for fixed-size 1280x720 video.")
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved: {path}")


def benchmark_one_variant(
    variant: str,
    args: argparse.Namespace,
    torch_device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    spec = VARIANT_SPECS[variant]

    print("\n" + "=" * 120)
    print(f"Running variant: {variant}")
    print("=" * 120)
    print(spec["description"])

    if spec["requires_torch"] and torch is None:
        raise RuntimeError(f"Variant {variant} requires PyTorch, but PyTorch could not be imported.")

    ctx: Dict[str, Any] = {
        "torch_device": torch_device,
        "threshold": args.threshold,
        "nms_radius": args.nms_radius,
    }

    if spec.get("requires_migraphx_nms"):
        mxr_path = Path(args.migraphx_nms_mxr)
        if not mxr_path.exists():
            raise FileNotFoundError(f"MIGraphX NMS MXR not found: {mxr_path}")
        ctx["migraphx_nms_head"] = load_migraphx_nms_head(str(mxr_path))
        ctx["migraphx_max_candidates"] = spec.get("migraphx_max_candidates")
        print(f"Loaded MIGraphX NMS head: {mxr_path}")

    engine = PoseEstimator(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    video_meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }

    print(f"Video: {args.video}")
    print(f"Resolution: {video_meta['width']}x{video_meta['height']}, metadata FPS={video_meta['fps']:.2f}")
    print(f"Measured frames: {args.frames}, warmup frames: {args.warmup}")
    print(f"Torch device: {torch_device}")

    detailed_rows: List[Dict[str, Any]] = []
    power_samples: List[float] = []

    frame_idx = 0
    measured_idx = 0

    post_fn = spec["fn"]

    while True:
        loop_t0 = time.perf_counter()

        with Timer() as read_t:
            ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1
        is_warmup = frame_idx <= args.warmup

        if not is_warmup and measured_idx >= args.frames:
            break

        e2e_t0 = time.perf_counter()

        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)

        with Timer() as infer_t:
            raw_results = engine.model.run({"input": input_tensor})

        poses, keypoints, post_times = post_fn(
            raw_results,
            frame.shape[:2],
            engine,
            ctx,
        )

        e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
        loop_ms = (time.perf_counter() - loop_t0) * 1000.0

        if is_warmup:
            continue

        measured_idx += 1

        power_w = float("nan")
        if not args.no_power and args.power_every > 0:
            if measured_idx == 1 or measured_idx % args.power_every == 0:
                power_w = get_gpu_power_w()
                if not math.isnan(power_w):
                    power_samples.append(power_w)

        row = {
            "variant": variant,
            "frame_idx": frame_idx,
            "measured_frame_idx": measured_idx,
            "read_ms": read_t.ms,
            "preprocess_ms": pre_t.ms,
            "inference_ms": infer_t.ms,
            "decode_ms": post_times.get("decode_ms", 0.0),
            "hm_resize_ms": post_times.get("hm_resize_ms", 0.0),
            "paf_resize_ms": post_times.get("paf_resize_ms", 0.0),
            "mx_nms_ms": post_times.get("mx_nms_ms", 0.0),
            "extract_ms": post_times.get("extract_ms", 0.0),
            "extract_from_mask_ms": post_times.get("extract_from_mask_ms", 0.0),
            "group_ms": post_times.get("group_ms", 0.0),
            "post_ms": post_times.get("post_ms", 0.0),
            "e2e_compute_ms": e2e_ms,
            "loop_total_ms": loop_ms,
            "power_w": power_w,
        }
        detailed_rows.append(row)

        if measured_idx == 1 or (args.print_every > 0 and measured_idx % args.print_every == 0):
            print(
                f"[{variant}] {measured_idx:4d}/{args.frames} "
                f"pre={pre_t.ms:6.2f} ms "
                f"infer={infer_t.ms:6.2f} ms "
                f"post={row['post_ms']:7.2f} ms "
                f"e2e={e2e_ms:7.2f} ms "
                f"power={fmt_float(power_w)} W"
            )

    cap.release()

    if measured_idx == 0:
        raise RuntimeError(f"No frames were measured for variant {variant}.")

    # Save video_meta into every summary via closure.
    summary = summarize_rows(
        variant=variant,
        description=spec["description"],
        rows=detailed_rows,
        power_samples=power_samples,
        video_meta=video_meta,
        baseline_e2e_ms=None,
        baseline_post_ms=None,
    )

    return summary, detailed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-process E2E video benchmark with GPU power estimates."
    )

    parser.add_argument("--video", default="cctv_1280x720_24fps_original.mp4")
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument("--migraphx-nms-mxr", default="models/heatmap_nms_head.mxr")

    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)

    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=sorted(VARIANT_SPECS.keys()),
    )

    parser.add_argument(
        "--torch-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for torch-based postprocess variants.",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius", type=int, default=6)

    parser.add_argument("--power-every", type=int, default=10)
    parser.add_argument("--no-power", action="store_true")

    parser.add_argument("--csv", default="benchmark_singleprocess_e2e_power_summary.csv")
    parser.add_argument("--detailed-csv", default="benchmark_singleprocess_e2e_power_detailed.csv")
    parser.add_argument("--json", default="benchmark_singleprocess_e2e_power_summary.json")
    parser.add_argument("--md", default="benchmark_singleprocess_e2e_power_report.md")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    torch_device = torch_device_from_arg(args.torch_device) if torch is not None else None

    print("\nSingle-process E2E benchmark")
    print(f"Video:              {args.video}")
    print(f"Model:              {args.model}")
    print(f"MIGraphX NMS MXR:    {args.migraphx_nms_mxr}")
    print(f"Frames/warmup:      {args.frames}/{args.warmup}")
    print(f"Power sampling:     {'disabled' if args.no_power else f'every {args.power_every} measured frame(s)'}")
    print_variant_descriptions(args.variants)

    all_summaries: List[Dict[str, Any]] = []
    all_detailed: List[Dict[str, Any]] = []

    for variant in args.variants:
        summary, detailed = benchmark_one_variant(
            variant=variant,
            args=args,
            torch_device=torch_device,
        )
        all_summaries.append(summary)
        all_detailed.extend(detailed)

    # Recompute speedups after all variants are known.
    baseline = next((r for r in all_summaries if r["variant"] == "standard_cpu"), all_summaries[0])
    baseline_e2e = float(baseline["e2e_compute_mean_ms"])
    baseline_post = float(baseline["post_mean_ms"])

    for r in all_summaries:
        r["e2e_speedup_vs_standard"] = baseline_e2e / float(r["e2e_compute_mean_ms"]) if float(r["e2e_compute_mean_ms"]) > 0 else 1.0
        r["post_speedup_vs_standard"] = baseline_post / float(r["post_mean_ms"]) if float(r["post_mean_ms"]) > 0 else 1.0

    print_summary_table(all_summaries)

    write_csv(args.csv, all_summaries, SUMMARY_FIELDS)
    write_csv(args.detailed_csv, all_detailed, DETAILED_FIELDS)
    write_json(args.json, all_summaries)
    write_markdown_report(args.md, all_summaries, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
