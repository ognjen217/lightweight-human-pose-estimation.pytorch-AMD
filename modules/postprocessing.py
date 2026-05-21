#!/usr/bin/env python3
"""
Centralized post-processing registry for lightweight-human-pose-estimation.pytorch-AMD.

Goal
----
This module is the single source of truth for all post-processing variants used by
video speed validation and COCO AP/AR validation. Validation scripts should call
`postprocess_from_results(...)` or `postprocess_from_maps(...)` instead of
re-implementing post-processing locally.

Input contract
--------------
- `postprocess_from_results(...)` accepts raw MIGraphX outputs and decodes them.
- `postprocess_from_maps(...)` accepts low-resolution HWC heatmaps and PAFs.
- Both return `PostprocessOutput(pose_entries, all_keypoints, timings)`.

Timing schema
-------------
All variants return the same timing keys where possible:
    decode, resize_heatmaps, resize_pafs, extract_keypoints,
    group_keypoints, group_total, group_prepare, group_pairs, group_sample,
    group_affinity, group_nms, group_pose, group_filter,
    group_pairs_total, group_valid_limbs, group_connections,
    scale_keypoints, total_postprocess
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path

import cv2
import numpy as np

# Keep PyTorch/ROCm lazy. Importing Torch at module import time can create
# conflicts with MIGraphX on some ROCm setups. CPU-only code paths and the
# MIGraphX inference process must be able to import this module without
# touching torch.cuda. GPU helpers call _ensure_torch_imported() only inside
# the process that actually performs Torch post-processing.
torch = None
F = None

from modules.keypoints import (
    BODY_PARTS_KPT_IDS,
    BODY_PARTS_PAF_IDS,
    connections_nms,
    extract_keypoints,
    extract_keypoints_batch,
    extract_keypoints_from_peak_mask,
    group_keypoints,
    group_keypoints_fast,
)

try:
    from modules.keypoints import extract_keypoints_batch_cv2
except Exception:  # pragma: no cover - keep compatibility with older keypoints.py
    extract_keypoints_batch_cv2 = None


TimingDict = Dict[str, float]
PoseEntries = np.ndarray
AllKeypoints = np.ndarray


@dataclass
class PostprocessConfig:
    """Runtime knobs shared by all post-processing variants."""

    max_keypoints_per_type: int = 20
    threshold: float = 0.1
    points_per_limb: int = 8
    nms_radius_fullres: int = 6
    nms_radius_lowres: int = 1
    min_paf_score: float = 0.05
    success_ratio_thr: float = 0.8
    torch_device: str = "auto"  # auto | cuda | cpu
    require_gpu: bool = False
    force_cv2_batch_extractor: bool = True
    migraphx_nms_mxr: Optional[str] = None
    migraphx_nms_cache_dir: Optional[str] = None
    migraphx_nms_input_name: str = "heatmaps"
    debug: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PostprocessOutput:
    pose_entries: PoseEntries
    all_keypoints: AllKeypoints
    timings: TimingDict


@dataclass(frozen=True)
class VariantInfo:
    canonical: str
    description: str
    accuracy_note: str
    aliases: Tuple[str, ...]


VARIANT_INFOS: Tuple[VariantInfo, ...] = (
    VariantInfo(
        canonical="standard",
        description="Full-res resize + original per-channel extract_keypoints + original group_keypoints.",
        accuracy_note="Reference implementation and AP/AR baseline.",
        aliases=("reference", "baseline"),
    ),
    VariantInfo(
        canonical="fast_no_resize",
        description="Low-res original extraction/grouping, then scale keypoints to original frame.",
        accuracy_note="Very fast but expected to lose AP/AR because grouping is done at output resolution.",
        aliases=("fast", "lowres", "no-resize"),
    ),
    VariantInfo(
        canonical="optimized_batch_k10",
        description="Full-res resize + batched keypoint extraction with K=10 + original grouping.",
        accuracy_note="Can lose AP/AR when more than 10 candidates per keypoint type are needed.",
        aliases=("k10", "batch-k10", "optimized-k10"),
    ),
    VariantInfo(
        canonical="optimized_batch_k20",
        description="Full-res resize + batched keypoint extraction with K=20 + original grouping.",
        accuracy_note="Conservative accuracy-preserving batch extraction path.",
        aliases=("k20", "batch-k20", "optimized-k20"),
    ),
    VariantInfo(
        canonical="optimized_batch_k20_fast",
        description="Full-res resize + batched K=20 extraction + group_keypoints_fast.",
        accuracy_note="Best CPU-only accuracy-preserving path in the current experiments.",
        aliases=("k20-fast", "k20_fast", "cpu", "optimized", "cpu-k20-fast"),
    ),
    VariantInfo(
        canonical="lowres_cpu_group",
        description="Low-res batched K=20 extraction + group_keypoints_fast, then scale keypoints.",
        accuracy_note="Fast approximation; validate AP/AR before using for production accuracy.",
        aliases=("lowres-cpu-group", "cpu-lowres", "fast_cpu_group"),
    ),
    VariantInfo(
        canonical="migraphx_nms",
        description="Full-res resize + compiled MIGraphX dense heatmap NMS mask + CPU mask extraction + group_keypoints_fast.",
        accuracy_note="Experimental path; AP should match full-res NMS semantics, but CPU mask-to-keypoint extraction is currently the bottleneck.",
        aliases=("migraphx-nms", "mx-nms", "mx_nms", "migraphx_nms_fullres"),
    ),
    VariantInfo(
        canonical="migraphx_nms_k20",
        description="Same as migraphx_nms, but keeps only top K candidates per keypoint type after mask extraction.",
        accuracy_note="K=20-compatible MIGraphX NMS path for comparison against optimized K20 CPU/GPU variants.",
        aliases=("migraphx-nms-k20", "mx-nms-k20", "mx_nms_k20"),
    ),
    VariantInfo(
        canonical="gpu_nms_fullres_cpu_group",
        description="Full-res resize + Torch GPU NMS/keypoint extraction + CPU group_keypoints_fast.",
        accuracy_note="Best tested GPU/hybrid accuracy-preserving path.",
        aliases=("gpu-nms", "gpu_nms", "hybrid", "hybrid-gpu-nms", "gpu-nms-fullres"),
    ),
    VariantInfo(
        canonical="gpu_nms_lowres_cpu_group",
        description="Low-res Torch GPU NMS/keypoint extraction + CPU group_keypoints_fast, then scale keypoints.",
        accuracy_note="GPU low-res approximation; expected AP/AR drop versus full-res grouping.",
        aliases=("gpu-nms-lowres", "gpu_lowres_cpu_group"),
    ),
    VariantInfo(
        canonical="gpu_nms_fullres_two_process",
        description="Two-process pipeline: MIGraphX inference process + Torch GPU-NMS full-res postprocess process.",
        accuracy_note="Same GPU-NMS full-res algorithm, but safe for ROCm setups where MIGraphX and PyTorch cannot share one process.",
        aliases=("gpu-nms-two-process", "gpu_nms_two_process", "two-process-gpu-nms", "gpu-nms-fullres-two-process"),
    ),
    VariantInfo(
        canonical="gpu_nms_lowres_two_process",
        description="Two-process pipeline: MIGraphX inference process + Torch GPU-NMS low-res postprocess process.",
        accuracy_note="Two-process low-res approximation; validate AP/AR before using as accuracy-preserving path.",
        aliases=("gpu-nms-lowres-two-process", "two-process-gpu-nms-lowres"),
    ),
    VariantInfo(
        canonical="cpu_k20_fast_two_process",
        description="Two-process pipeline with CPU K20 fast postprocess in the second process.",
        accuracy_note="Useful as a two-process CPU control path for queue/shared-memory overhead measurement.",
        aliases=("cpu-k20-fast-two-process", "two-process-cpu-k20-fast"),
    ),
    VariantInfo(
        canonical="gpu_fullres_paf",
        description="Full-res CPU K=20 extraction + Torch GPU PAF affinity scoring + CPU pose assembly.",
        accuracy_note="Research path; preserves more geometry but may be slower than GPU NMS hybrid.",
        aliases=("gpu-fullres-paf", "gpu_fullres_paf"),
    ),
    VariantInfo(
        canonical="gpu_lowres_paf",
        description="Low-res Torch GPU NMS + Torch GPU PAF affinity scoring + CPU pose assembly, then scale.",
        accuracy_note="Fastest GPU-heavy approximation; validate AP/AR carefully.",
        aliases=("gpu-lowres-paf", "gpu_lowres_paf", "gpu", "full-gpu", "gpu_accelerated", "gpu_acceletated"),
    ),
)

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _info in VARIANT_INFOS:
    _ALIAS_TO_CANONICAL[_info.canonical] = _info.canonical
    for _alias in _info.aliases:
        _ALIAS_TO_CANONICAL[_alias] = _info.canonical

DEFAULT_SPEED_VARIANTS: Tuple[str, ...] = (
    "standard",
    "optimized_batch_k20_fast",
    "gpu_nms_fullres_two_process",
)

DEFAULT_ACCURACY_VARIANTS: Tuple[str, ...] = (
    "standard",
    "optimized_batch_k20_fast",
    "gpu_nms_fullres_two_process",
)

TWO_PROCESS_VARIANTS: Tuple[str, ...] = (
    "gpu_nms_fullres_two_process",
    "gpu_nms_lowres_two_process",
    "cpu_k20_fast_two_process",
)


def is_two_process_mode(mode: str) -> bool:
    """Return True when a variant must be run by the two-process runner."""
    return normalize_mode(mode) in TWO_PROCESS_VARIANTS


def two_process_worker_mode(mode: str) -> str:
    """Map registry names/aliases to the worker mode used by the two-process support module."""
    canonical = normalize_mode(mode)
    if canonical == "gpu_nms_fullres_two_process":
        return "gpu-nms-fullres"
    if canonical == "gpu_nms_lowres_two_process":
        return "gpu-nms-lowres"
    if canonical == "cpu_k20_fast_two_process":
        return "cpu-k20-fast"
    raise ValueError(f"Mode {mode!r} is not a two-process postprocess mode.")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def normalize_mode(mode: str) -> str:
    """Normalize a variant name or alias to the canonical registry key."""
    key = str(mode).strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    # Keep canonical names with underscores compatible too.
    key_underscore = key.replace("-", "_")
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    if key_underscore in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key_underscore]
    available = ", ".join(sorted(_ALIAS_TO_CANONICAL))
    raise ValueError(f"Unknown postprocess mode '{mode}'. Available names/aliases: {available}")


def available_modes(include_aliases: bool = False) -> List[str]:
    if include_aliases:
        return sorted(_ALIAS_TO_CANONICAL)
    return [info.canonical for info in VARIANT_INFOS]


def variant_table() -> List[Dict[str, str]]:
    return [
        {
            "variant": info.canonical,
            "aliases": ", ".join(info.aliases),
            "description": info.description,
            "accuracy_note": info.accuracy_note,
        }
        for info in VARIANT_INFOS
    ]


def empty_timings() -> TimingDict:
    return {
        "decode": 0.0,
        "resize_heatmaps": 0.0,
        "resize_pafs": 0.0,
        "extract_keypoints": 0.0,
        "mx_nms": 0.0,
        "extract_from_mask": 0.0,
        "group_keypoints": 0.0,
        "group_total": 0.0,
        "group_prepare": 0.0,
        "group_pairs": 0.0,
        "group_sample": 0.0,
        "group_affinity": 0.0,
        "group_nms": 0.0,
        "group_pose": 0.0,
        "group_filter": 0.0,
        "group_pairs_total": 0.0,
        "group_valid_limbs": 0.0,
        "group_connections": 0.0,
        "scale_keypoints": 0.0,
        "total_postprocess": 0.0,
    }


def _as_output(poses: Any, kpts: Any, timings: TimingDict) -> PostprocessOutput:
    return PostprocessOutput(
        pose_entries=np.asarray(poses, dtype=np.float32),
        all_keypoints=np.asarray(kpts, dtype=np.float32),
        timings=timings,
    )


def _merge_group_timings(timings: TimingDict, group_times: Optional[Mapping[str, Any]]) -> None:
    if not group_times:
        return
    for key, value in group_times.items():
        try:
            timings[str(key)] = float(value)
        except Exception:
            pass


def _batch_extract(heatmaps_18: np.ndarray, max_keypoints_per_type: int, force_cv2: bool = True):
    extractor = extract_keypoints_batch_cv2 if (force_cv2 and extract_keypoints_batch_cv2 is not None) else extract_keypoints_batch
    return extractor(heatmaps_18, max_keypoints_per_type=max_keypoints_per_type)


def decode_migraphx_outputs(
    results: Any,
    target_dim: Tuple[int, int] = (968, 544),
    stride: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode raw MIGraphX outputs into low-resolution HWC heatmaps and PAFs.

    Uses the final two outputs when a model returns more than two tensors, matching
    the COCO validation path. For the common two-output .mxr models this is
    equivalent to results[0], results[1].
    """
    target_w, target_h = target_dim
    out_h = target_h // stride
    out_w = target_w // stride

    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) < 2:
        raise ValueError("MIGraphX results must contain at least heatmaps and PAFs")

    heat_raw = results[-2]
    paf_raw = results[-1]

    heatmaps = np.asarray(heat_raw, dtype=np.float32).reshape(19, out_h, out_w)
    pafs = np.asarray(paf_raw, dtype=np.float32).reshape(38, out_h, out_w)

    heatmaps = np.moveaxis(heatmaps, 0, -1)
    pafs = np.moveaxis(pafs, 0, -1)
    return np.ascontiguousarray(heatmaps), np.ascontiguousarray(pafs)


def resize_fullres(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    timings: TimingDict,
    heatmap_dtype=np.float32,
    paf_dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    orig_h, orig_w = original_hw

    with Timer() as t:
        heatmaps_full = cv2.resize(
            np.ascontiguousarray(heatmaps, dtype=np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        heatmaps_full = np.ascontiguousarray(heatmaps_full, dtype=heatmap_dtype)
    timings["resize_heatmaps"] = t.ms

    with Timer() as t:
        pafs_full = cv2.resize(
            np.ascontiguousarray(pafs, dtype=np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        pafs_full = np.ascontiguousarray(pafs_full, dtype=paf_dtype)
    timings["resize_pafs"] = t.ms

    return heatmaps_full, pafs_full


def _scale_keypoints_to_original(kpts: np.ndarray, original_hw: Tuple[int, int], lowres_hw: Tuple[int, int], timings: TimingDict) -> None:
    if kpts is None or len(kpts) == 0:
        return
    orig_h, orig_w = original_hw
    out_h, out_w = lowres_hw
    with Timer() as t:
        kpts[:, 0] *= float(orig_w) / float(out_w)
        kpts[:, 1] *= float(orig_h) / float(out_h)
    timings["scale_keypoints"] = t.ms


def _group_standard_timed(all_kpts, pafs: np.ndarray, timings: TimingDict, points_per_limb: int):
    with Timer() as t:
        try:
            poses, kpts = group_keypoints(all_kpts, pafs, points_per_limb=points_per_limb)
        except TypeError:
            poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_keypoints"] = t.ms
    timings["group_total"] = t.ms
    return np.asarray(poses, dtype=np.float32), np.asarray(kpts, dtype=np.float32)


def _group_fast_timed(all_kpts, pafs: np.ndarray, timings: TimingDict, points_per_limb: int):
    with Timer() as t:
        try:
            out = group_keypoints_fast(
                all_kpts,
                pafs,
                points_per_limb=points_per_limb,
                return_timing=True,
            )
        except TypeError:
            out = group_keypoints_fast(all_kpts, pafs, points_per_limb=points_per_limb)
    timings["group_keypoints"] = t.ms

    if isinstance(out, tuple) and len(out) == 3:
        poses, kpts, group_times = out
        _merge_group_timings(timings, group_times)
    else:
        poses, kpts = out

    timings["group_total"] = timings.get("group_total", 0.0) or timings["group_keypoints"]
    return np.asarray(poses, dtype=np.float32), np.asarray(kpts, dtype=np.float32)


def _ensure_torch_imported():
    global torch, F
    if torch is not None and F is not None:
        return torch, F
    try:
        import torch as _torch
        import torch.nn.functional as _F
    except Exception as exc:  # pragma: no cover - depends on local ROCm install
        raise RuntimeError(
            "PyTorch is required for GPU postprocess variants but could not be imported."
        ) from exc
    torch = _torch
    F = _F
    return torch, F


def _resolve_torch_device(device_arg: str, require_gpu: bool = False):
    _ensure_torch_imported()

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested torch_device='cuda', but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unsupported torch_device: {device_arg}")

    if require_gpu and device.type != "cuda":
        raise RuntimeError("GPU postprocess variant requested, but resolved torch device is CPU.")
    return device


def _sync_if_gpu(device) -> None:
    if torch is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def extract_keypoints_gpu_nms(
    heatmaps: np.ndarray,
    *,
    max_keypoints_per_type: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    torch_device: str = "auto",
    require_gpu: bool = False,
    nms_impl: str = "2d",
    compute_dtype: str = "float32",
) -> Tuple[List[List[Tuple[float, float, float, int]]], int]:
    """Torch max_pool2d NMS over HWC heatmaps; returns OpenPose keypoint lists."""
    _ensure_torch_imported()
    device = _resolve_torch_device(torch_device, require_gpu=require_gpu)

    if compute_dtype == "float16":
        heatmaps_np = np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float16)
        torch_dtype = torch.float16
    elif compute_dtype == "float32":
        heatmaps_np = np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float32)
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

    hm = torch.from_numpy(heatmaps_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch_dtype)

    if require_gpu and hm.device.type != "cuda":
        raise RuntimeError("gpu-nms expected a CUDA/ROCm tensor, but the heatmap tensor is on CPU.")

    radius = int(nms_radius)
    k = 2 * radius + 1
    if nms_impl == "2d":
        pooled = F.max_pool2d(
            hm,
            kernel_size=k,
            stride=1,
            padding=radius,
        )
    elif nms_impl == "separable":
        pooled = F.max_pool2d(
            hm,
            kernel_size=(k, 1),
            stride=1,
            padding=(radius, 0),
        )
        pooled = F.max_pool2d(
            pooled,
            kernel_size=(1, k),
            stride=1,
            padding=(0, radius),
        )
    else:
        raise ValueError(f"Unsupported nms_impl: {nms_impl}. Use '2d' or 'separable'.")

    peaks = (hm == pooled) & (hm > float(threshold))

    all_kpts: List[List[Tuple[float, float, float, int]]] = []
    total = 0

    for kpt_idx in range(18):
        coords = torch.nonzero(peaks[0, kpt_idx], as_tuple=False)
        if coords.numel() == 0:
            all_kpts.append([])
            continue

        ys = coords[:, 0]
        xs = coords[:, 1]
        scores = hm[0, kpt_idx, ys, xs]
        keep = min(int(max_keypoints_per_type), int(scores.numel()))
        top_scores, order = torch.topk(scores, k=keep, largest=True, sorted=True)

        xs_np = xs[order].detach().cpu().numpy()
        ys_np = ys[order].detach().cpu().numpy()
        scores_np = top_scores.detach().cpu().numpy().astype(np.float32, copy=False)

        pts = [
            (float(xs_np[i]), float(ys_np[i]), float(scores_np[i]), int(total + i))
            for i in range(keep)
        ]
        all_kpts.append(pts)
        total += len(pts)

    _sync_if_gpu(device)
    return all_kpts, total


def assemble_pose_entries_from_connections(
    all_keypoints_by_type,
    connections_by_part,
    pose_entry_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    non_empty = [np.asarray(k, dtype=np.float32) for k in all_keypoints_by_type if len(k) > 0]
    if non_empty:
        all_keypoints = np.concatenate(non_empty, axis=0)
    else:
        return np.empty((0, pose_entry_size), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    pose_entries: List[np.ndarray] = []

    for part_id, connections in enumerate(connections_by_part):
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size, dtype=np.float32) * -1 for _ in range(len(connections))]
            for i, conn in enumerate(connections):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = conn[0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = conn[1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[[int(conn[0]), int(conn[1])], 2]) + conn[2]

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for conn in connections:
                for pose in pose_entries:
                    if pose[kpt_a_id] == conn[0] and pose[kpt_b_id] == -1:
                        pose[kpt_b_id] = conn[1]
                    elif pose[kpt_b_id] == conn[1] and pose[kpt_a_id] == -1:
                        pose[kpt_a_id] = conn[0]

        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for conn in connections:
                attached = 0
                for pose in pose_entries:
                    if pose[kpt_a_id] == conn[0]:
                        pose[kpt_b_id] = conn[1]
                        attached += 1
                        pose[-1] += 1
                        pose[-2] += all_keypoints[int(conn[1]), 2] + conn[2]

                if attached == 0:
                    pose_entry = np.ones(pose_entry_size, dtype=np.float32) * -1
                    pose_entry[kpt_a_id] = conn[0]
                    pose_entry[kpt_b_id] = conn[1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[[int(conn[0]), int(conn[1])], 2]) + conn[2]
                    pose_entries.append(pose_entry)

    filtered = []
    for pose_entry in pose_entries:
        if pose_entry[-1] < 3:
            continue
        if pose_entry[-2] / pose_entry[-1] < 0.2:
            continue
        filtered.append(pose_entry)

    return np.asarray(filtered, dtype=np.float32), all_keypoints


def score_paf_connections_gpu(
    all_keypoints_by_type,
    pafs: np.ndarray,
    *,
    torch_device: str = "auto",
    require_gpu: bool = False,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
) -> Tuple[List[List[Tuple[int, int, float]]], TimingDict]:
    _ensure_torch_imported()
    device = _resolve_torch_device(torch_device, require_gpu=require_gpu)

    pafs_t = torch.as_tensor(pafs, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    paf_h, paf_w = pafs.shape[:2]
    grid = torch.arange(points_per_limb, device=device, dtype=torch.float32).view(1, points_per_limb, 1)

    connections_by_part = []
    stats: TimingDict = {
        "group_pairs_total": 0.0,
        "group_valid_limbs": 0.0,
        "group_connections": 0.0,
    }

    for part_id, paf_ids in enumerate(BODY_PARTS_PAF_IDS):
        kpts_a_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]], dtype=np.float32)
        kpts_b_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]], dtype=np.float32)

        n, m = len(kpts_a_np), len(kpts_b_np)
        if n == 0 or m == 0:
            connections_by_part.append([])
            continue

        stats["group_pairs_total"] += float(n * m)

        kpts_a = torch.as_tensor(kpts_a_np[:, :2], device=device, dtype=torch.float32)
        kpts_b = torch.as_tensor(kpts_b_np[:, :2], device=device, dtype=torch.float32)

        vec_raw = (kpts_b[:, None, :] - kpts_a[None, :, :]).reshape(-1, 1, 2)
        vec_norm = torch.linalg.norm(vec_raw, dim=-1, keepdim=True)
        valid_vec = vec_norm.reshape(-1) > 1e-6

        if not bool(valid_vec.any().item()):
            connections_by_part.append([])
            continue

        pair_ids = torch.nonzero(valid_vec, as_tuple=False).reshape(-1)
        vec_raw_valid = vec_raw[valid_vec]
        vec_norm_valid = vec_norm[valid_vec]

        b_pair_idx = torch.div(pair_ids, n, rounding_mode="floor")
        a_pair_idx = pair_ids - b_pair_idx * n

        steps = vec_raw_valid / float(points_per_limb - 1)
        a_points = kpts_a[a_pair_idx].reshape(-1, 1, 2)
        points = torch.round(steps * grid + a_points).long()

        x = points[..., 0].reshape(-1).clamp(0, paf_w - 1)
        y = points[..., 1].reshape(-1).clamp(0, paf_h - 1)

        paf_x_id, paf_y_id = int(paf_ids[0]), int(paf_ids[1])
        field = torch.stack(
            (pafs_t[paf_x_id, y, x], pafs_t[paf_y_id, y, x]),
            dim=-1,
        ).reshape(-1, points_per_limb, 2)

        vec = vec_raw_valid / (vec_norm_valid + 1e-6)
        scores_per_point = (field * vec).sum(dim=-1)
        valid_scores = scores_per_point > min_paf_score
        valid_num = valid_scores.sum(dim=1)

        affinity = (scores_per_point * valid_scores.float()).sum(dim=1) / (valid_num.float() + 1e-6)
        success_ratio = valid_num.float() / float(points_per_limb)

        valid_limb_local = torch.nonzero(
            (affinity > 0) & (success_ratio > success_ratio_thr),
            as_tuple=False,
        ).reshape(-1)

        stats["group_valid_limbs"] += float(valid_limb_local.numel())

        if valid_limb_local.numel() == 0:
            connections_by_part.append([])
            continue

        valid_limbs = pair_ids[valid_limb_local]
        b_idx_t = torch.div(valid_limbs, n, rounding_mode="floor")
        a_idx_t = valid_limbs - b_idx_t * n

        a_idx = a_idx_t.detach().cpu().numpy().astype(np.int32)
        b_idx = b_idx_t.detach().cpu().numpy().astype(np.int32)
        scores = affinity[valid_limb_local].detach().cpu().numpy().astype(np.float32)

        a_idx, b_idx, scores = connections_nms(a_idx, b_idx, scores)
        connections = list(
            zip(
                kpts_a_np[a_idx, 3].astype(np.int32),
                kpts_b_np[b_idx, 3].astype(np.int32),
                scores,
            )
        )

        stats["group_connections"] += float(len(connections))
        connections_by_part.append(connections)

    _sync_if_gpu(device)
    return connections_by_part, stats



_MIGRAPHX_NMS_CACHE: Dict[Tuple[str, str], Any] = {}


def _resolve_migraphx_nms_path(original_hw: Tuple[int, int], config: PostprocessConfig) -> str:
    """Return the MXR path for the full-resolution NMS head for this image/frame size."""
    h, w = int(original_hw[0]), int(original_hw[1])

    if config.migraphx_nms_cache_dir:
        path = Path(config.migraphx_nms_cache_dir) / f"heatmap_nms_head_{h}x{w}.mxr"
        if not path.exists():
            raise FileNotFoundError(
                "Missing MIGraphX NMS cache file for current full-resolution shape.\n"
                f"  shape:    {h}x{w}\n"
                f"  expected: {path}\n"
                "Generate the cache first with modules/migraphx_compiler.py."
            )
        return str(path)

    if not config.migraphx_nms_mxr:
        raise ValueError(
            "MIGraphX NMS mode requires PostprocessConfig.migraphx_nms_mxr "
            "or PostprocessConfig.migraphx_nms_cache_dir."
        )

    path = Path(config.migraphx_nms_mxr)
    if not path.exists():
        raise FileNotFoundError(f"MIGraphX NMS .mxr not found: {path}")
    return str(path)


def _get_migraphx_nms_head(mxr_path: str, input_name: str):
    """Lazy-load and cache MIGraphX NMS programs by MXR path/input name."""
    key = (str(mxr_path), str(input_name))
    if key not in _MIGRAPHX_NMS_CACHE:
        from modules.migraphx_nms import MIGraphXNMSHead
        _MIGRAPHX_NMS_CACHE[key] = MIGraphXNMSHead(str(mxr_path), input_name=input_name)
    return _MIGRAPHX_NMS_CACHE[key]


def _postprocess_migraphx_nms(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    config: PostprocessConfig,
    timings: TimingDict,
    *,
    limit_k20: bool,
) -> PostprocessOutput:
    """Full-res MIGraphX NMS followed by CPU mask extraction and fast grouping."""
    heatmaps_full, pafs_full = resize_fullres(heatmaps, pafs, original_hw, timings)

    mxr_path = _resolve_migraphx_nms_path(original_hw, config)
    mx_head = _get_migraphx_nms_head(mxr_path, config.migraphx_nms_input_name)

    heatmaps_nchw = np.moveaxis(np.ascontiguousarray(heatmaps_full, dtype=np.float32), -1, 0)[np.newaxis, ...]
    with Timer() as t:
        peak_mask = mx_head.run(heatmaps_nchw)
    timings["mx_nms"] = t.ms

    with Timer() as t:
        all_kpts, _ = extract_keypoints_from_peak_mask(
            heatmaps_full,
            peak_mask,
            max_candidates_per_part=(config.max_keypoints_per_type if limit_k20 else None),
            num_keypoint_types=18,
        )
    timings["extract_from_mask"] = t.ms
    timings["extract_keypoints"] = t.ms

    poses, kpts = _group_fast_timed(all_kpts, pafs_full, timings, config.points_per_limb)
    return _as_output(poses, kpts, timings)


def _postprocess_maps_impl(
    mode: str,
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    config: PostprocessConfig,
    timings: TimingDict,
) -> PostprocessOutput:
    canonical = normalize_mode(mode)

    if canonical in TWO_PROCESS_VARIANTS:
        raise RuntimeError(
            f"Mode '{canonical}' cannot be executed with postprocess_from_maps/results. "
            "Use run_two_process_postprocessing(...) so MIGraphX and PyTorch run in separate processes."
        )

    if canonical == "standard":
        heatmaps_full, pafs_full = resize_fullres(heatmaps, pafs, original_hw, timings)
        with Timer() as t:
            all_kpts = []
            total = 0
            for kpt_idx in range(18):
                total += extract_keypoints(heatmaps_full[:, :, kpt_idx], all_kpts, total)
        timings["extract_keypoints"] = t.ms
        poses, kpts = _group_standard_timed(all_kpts, pafs_full, timings, config.points_per_limb)
        return _as_output(poses, kpts, timings)

    if canonical == "fast_no_resize":
        out_h, out_w = heatmaps.shape[:2]
        with Timer() as t:
            all_kpts = []
            total = 0
            for kpt_idx in range(18):
                total += extract_keypoints(heatmaps[:, :, kpt_idx], all_kpts, total)
        timings["extract_keypoints"] = t.ms
        poses, kpts = _group_standard_timed(all_kpts, pafs, timings, config.points_per_limb)
        _scale_keypoints_to_original(kpts, original_hw, (out_h, out_w), timings)
        return _as_output(poses, kpts, timings)

    if canonical == "lowres_cpu_group":
        out_h, out_w = heatmaps.shape[:2]
        with Timer() as t:
            all_kpts, _ = _batch_extract(
                heatmaps[:, :, :18],
                max_keypoints_per_type=config.max_keypoints_per_type,
                force_cv2=config.force_cv2_batch_extractor,
            )
        timings["extract_keypoints"] = t.ms
        poses, kpts = _group_fast_timed(all_kpts, pafs, timings, config.points_per_limb)
        _scale_keypoints_to_original(kpts, original_hw, (out_h, out_w), timings)
        return _as_output(poses, kpts, timings)

    if canonical in {"optimized_batch_k10", "optimized_batch_k20", "optimized_batch_k20_fast"}:
        k = 10 if canonical == "optimized_batch_k10" else config.max_keypoints_per_type
        heatmaps_full, pafs_full = resize_fullres(heatmaps, pafs, original_hw, timings)
        with Timer() as t:
            all_kpts, _ = _batch_extract(
                heatmaps_full[:, :, :18],
                max_keypoints_per_type=k,
                force_cv2=config.force_cv2_batch_extractor,
            )
        timings["extract_keypoints"] = t.ms
        if canonical == "optimized_batch_k20_fast":
            poses, kpts = _group_fast_timed(all_kpts, pafs_full, timings, config.points_per_limb)
        else:
            poses, kpts = _group_standard_timed(all_kpts, pafs_full, timings, config.points_per_limb)
        return _as_output(poses, kpts, timings)

    if canonical == "migraphx_nms":
        return _postprocess_migraphx_nms(heatmaps, pafs, original_hw, config, timings, limit_k20=False)

    if canonical == "migraphx_nms_k20":
        return _postprocess_migraphx_nms(heatmaps, pafs, original_hw, config, timings, limit_k20=True)

    if canonical == "gpu_nms_fullres_cpu_group":
        heatmaps_full, pafs_full = resize_fullres(heatmaps, pafs, original_hw, timings)
        with Timer() as t:
            all_kpts, _ = extract_keypoints_gpu_nms(
                heatmaps_full,
                max_keypoints_per_type=config.max_keypoints_per_type,
                threshold=config.threshold,
                nms_radius=config.nms_radius_fullres,
                torch_device=config.torch_device,
                require_gpu=config.require_gpu,
                nms_impl=config.extra.get("nms_impl", "2d"),
                compute_dtype=config.extra.get("gpu_compute_dtype", "float32"),
            )
        timings["extract_keypoints"] = t.ms
        poses, kpts = _group_fast_timed(all_kpts, pafs_full, timings, config.points_per_limb)
        return _as_output(poses, kpts, timings)

    if canonical == "gpu_nms_lowres_cpu_group":
        out_h, out_w = heatmaps.shape[:2]
        with Timer() as t:
            all_kpts, _ = extract_keypoints_gpu_nms(
                heatmaps,
                max_keypoints_per_type=config.max_keypoints_per_type,
                threshold=config.threshold,
                nms_radius=config.nms_radius_lowres,
                torch_device=config.torch_device,
                require_gpu=config.require_gpu,
                nms_impl=config.extra.get("nms_impl", "2d"),
                compute_dtype=config.extra.get("gpu_compute_dtype", "float32"),
            )
        timings["extract_keypoints"] = t.ms
        poses, kpts = _group_fast_timed(all_kpts, pafs, timings, config.points_per_limb)
        _scale_keypoints_to_original(kpts, original_hw, (out_h, out_w), timings)
        return _as_output(poses, kpts, timings)

    if canonical == "gpu_fullres_paf":
        heatmaps_full, pafs_full = resize_fullres(heatmaps, pafs, original_hw, timings)
        with Timer() as t:
            all_kpts, _ = _batch_extract(
                heatmaps_full[:, :, :18],
                max_keypoints_per_type=config.max_keypoints_per_type,
                force_cv2=config.force_cv2_batch_extractor,
            )
        timings["extract_keypoints"] = t.ms
        with Timer() as t:
            connections, stats = score_paf_connections_gpu(
                all_kpts,
                pafs_full,
                torch_device=config.torch_device,
                require_gpu=config.require_gpu,
                points_per_limb=config.points_per_limb,
                min_paf_score=config.min_paf_score,
                success_ratio_thr=config.success_ratio_thr,
            )
        timings["group_affinity"] = t.ms
        _merge_group_timings(timings, stats)
        with Timer() as t:
            poses, kpts = assemble_pose_entries_from_connections(all_kpts, connections)
        timings["group_pose"] = t.ms
        timings["group_keypoints"] = timings["group_affinity"] + timings["group_pose"]
        timings["group_total"] = timings["group_keypoints"]
        return _as_output(poses, kpts, timings)

    if canonical == "gpu_lowres_paf":
        out_h, out_w = heatmaps.shape[:2]
        with Timer() as t:
            all_kpts, _ = extract_keypoints_gpu_nms(
                heatmaps,
                max_keypoints_per_type=config.max_keypoints_per_type,
                threshold=config.threshold,
                nms_radius=config.nms_radius_lowres,
                torch_device=config.torch_device,
                require_gpu=config.require_gpu,
                nms_impl=config.extra.get("nms_impl", "2d"),
                compute_dtype=config.extra.get("gpu_compute_dtype", "float32"),
            )
        timings["extract_keypoints"] = t.ms
        with Timer() as t:
            connections, stats = score_paf_connections_gpu(
                all_kpts,
                pafs,
                torch_device=config.torch_device,
                require_gpu=config.require_gpu,
                points_per_limb=config.points_per_limb,
                min_paf_score=config.min_paf_score,
                success_ratio_thr=config.success_ratio_thr,
            )
        timings["group_affinity"] = t.ms
        _merge_group_timings(timings, stats)
        with Timer() as t:
            poses, kpts = assemble_pose_entries_from_connections(all_kpts, connections)
        timings["group_pose"] = t.ms
        timings["group_keypoints"] = timings["group_affinity"] + timings["group_pose"]
        timings["group_total"] = timings["group_keypoints"]
        _scale_keypoints_to_original(kpts, original_hw, (out_h, out_w), timings)
        return _as_output(poses, kpts, timings)

    raise AssertionError(f"Unhandled canonical mode: {canonical}")


def postprocess_from_maps(
    mode: str,
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    config: Optional[PostprocessConfig] = None,
) -> PostprocessOutput:
    """Run a postprocess variant on already-decoded low-res heatmaps/PAFs."""
    config = config or PostprocessConfig()
    timings = empty_timings()
    total_start = time.perf_counter()
    out = _postprocess_maps_impl(mode, heatmaps, pafs, original_hw, config, timings)
    out.timings["total_postprocess"] = (time.perf_counter() - total_start) * 1000.0
    return out


def postprocess_from_results(
    mode: str,
    results: Any,
    original_hw: Tuple[int, int],
    target_dim: Tuple[int, int] = (968, 544),
    stride: int = 8,
    config: Optional[PostprocessConfig] = None,
) -> PostprocessOutput:
    """Decode raw MIGraphX results and run the selected postprocess variant."""
    config = config or PostprocessConfig()
    timings = empty_timings()
    total_start = time.perf_counter()
    with Timer() as t:
        heatmaps, pafs = decode_migraphx_outputs(results, target_dim=target_dim, stride=stride)
    timings["decode"] = t.ms
    out = _postprocess_maps_impl(mode, heatmaps, pafs, original_hw, config, timings)
    out.timings["total_postprocess"] = (time.perf_counter() - total_start) * 1000.0
    return out




def run_two_process_postprocessing(
    *,
    video_path: str,
    model_path: str,
    mode: str = "gpu_nms_fullres_two_process",
    target_width: int = 968,
    target_height: int = 544,
    stride: int = 8,
    max_frames: int = 100,
    warmup_frames: int = 5,
    slots: int = 3,
    print_every: int = 10,
    torch_device: str = "cuda",
    shared_dtype: str = "float32",
    gpu_compute_dtype: str = "float32",
    max_keypoints: int = 20,
    threshold: float = 0.1,
    nms_radius_fullres: int = 6,
    nms_radius_lowres: int = 1,
    nms_impl: str = "2d",
    collect_rows: bool = True,
) -> Dict[str, Any]:
    """Run the two-process MIGraphX + postprocess pipeline.

    This is the safe ROCm path for GPU-NMS when MIGraphX and PyTorch cannot
    coexist in one Python process. The inference process imports/uses MIGraphX;
    the postprocess process imports/uses Torch only when GPU mode is selected.

    Returns a dictionary with:
        - variant: canonical registry name
        - worker_mode: two-process worker mode
        - rows: per-frame timing rows, if collect_rows=True
        - summary: aggregate timing/FPS summary
    """
    canonical = normalize_mode(mode)
    if canonical not in TWO_PROCESS_VARIANTS:
        raise ValueError(
            f"run_two_process_postprocessing expected one of {TWO_PROCESS_VARIANTS}, got {mode!r}."
        )

    from modules.multiprocess_postprocessing_support import run_two_process_pipeline

    return run_two_process_pipeline(
        video_path=video_path,
        model_path=model_path,
        variant_name=canonical,
        worker_mode=two_process_worker_mode(canonical),
        target_width=target_width,
        target_height=target_height,
        stride=stride,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        slots=slots,
        print_every=print_every,
        torch_device=torch_device,
        shared_dtype=shared_dtype,
        gpu_compute_dtype=gpu_compute_dtype,
        max_keypoints=max_keypoints,
        threshold=threshold,
        nms_radius_fullres=nms_radius_fullres,
        nms_radius_lowres=nms_radius_lowres,
        nms_impl=nms_impl,
        collect_rows=collect_rows,
    )


def get_postprocess_fn(mode: str) -> Callable[..., PostprocessOutput]:
    """Return a callable using the common map-based signature.

    Signature of returned function:
        fn(heatmaps, pafs, original_hw, config=None) -> PostprocessOutput
    """
    canonical = normalize_mode(mode)

    def _fn(heatmaps: np.ndarray, pafs: np.ndarray, original_hw: Tuple[int, int], config: Optional[PostprocessConfig] = None):
        return postprocess_from_maps(canonical, heatmaps, pafs, original_hw, config=config)

    return _fn
