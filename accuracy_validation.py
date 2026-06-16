#!/usr/bin/env python3
"""Unified COCO AP/AR + latency validation for MIGraphX postprocess variants.

Additions for the fused/pruned postprocess experiments:

* accepts ``merged_fused_pruned`` as an alias for the registered pruned variant
* can auto-compile missing fused/pruned/manual postprocess heads per COCO shape
* can select a shape-controlled subset of COCO images using dominant dimensions
* evaluates COCO metrics only on the selected image IDs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import migraphx
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from modules.postprocessing import (
    DEFAULT_ACCURACY_VARIANTS,
    PostprocessConfig,
    available_modes,
    is_two_process_mode,
    normalize_mode,
    postprocess_from_maps,
    variant_table,
)
from modules.postprocess_head_autocompile import (
    ensure_shape_postprocess_heads,
    normalize_validation_variant_name,
    postprocess_extra_from_args,
)

try:
    from val import normalize, pad_width
except Exception:
    def normalize(img, img_mean=(128, 128, 128), img_scale=1 / 256):
        return (np.asarray(img, dtype=np.float32) - img_mean) * img_scale

    def pad_width(img, stride, pad_value, min_dims):
        h, w, _ = img.shape
        min_dims = list(min_dims)
        min_dims[0] = int(math.ceil(min_dims[0] / float(stride)) * stride)
        min_dims[1] = int(max(min_dims[1], w))
        min_dims[1] = int(math.ceil(min_dims[1] / float(stride)) * stride)
        pad = [
            int(math.floor((min_dims[0] - h) / 2.0)),
            int(math.floor((min_dims[1] - w) / 2.0)),
            int(min_dims[0] - h - math.floor((min_dims[0] - h) / 2.0)),
            int(min_dims[1] - w - math.floor((min_dims[1] - w) / 2.0)),
        ]
        padded = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value)
        return padded, pad

BASE_HEIGHT = 544
BASE_WIDTH = 968
STRIDE = 8
COCO_KPT_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

SPLIT_HIP_SMART_MODE = "split_hip_host_smart"
SPLIT_HIP_SMART_ALIASES = {
    SPLIT_HIP_SMART_MODE,
    "split-hip-host-smart",
    "split_hip_smart",
    "split-hip-smart",
    "mxr1_hip_smart_mxr2",
    "mxr1-hip-smart-mxr2",
    "mxr1_hip_host_smart_mxr2",
    "mxr1-hip-host-smart-mxr2",
}


def is_split_hip_smart_variant(name: str) -> bool:
    key = str(name or "").strip().lower().replace(" ", "-")
    return key in SPLIT_HIP_SMART_ALIASES or key.replace("-", "_") in SPLIT_HIP_SMART_ALIASES


def normalize_accuracy_variant(name: str) -> str:
    if is_split_hip_smart_variant(name):
        return SPLIT_HIP_SMART_MODE
    return normalize_mode(normalize_validation_variant_name(name))


def uses_split_hip_smart(variants: Sequence[str]) -> bool:
    return any(str(v) == SPLIT_HIP_SMART_MODE or is_split_hip_smart_variant(v) for v in variants)


@dataclass
class VariantMeter:
    model: str
    variant: str
    detections: List[dict] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    mx_nms_ms: List[float] = field(default_factory=list)
    fused_post_mx_ms: List[float] = field(default_factory=list)
    fused_pruned_mx_ms: List[float] = field(default_factory=list)
    topk_adapter_ms: List[float] = field(default_factory=list)
    mx_assembly_total_ms: List[float] = field(default_factory=list)
    pruned_cpu_tail_ms: List[float] = field(default_factory=list)
    split_smart_heatmap_ms: List[float] = field(default_factory=list)
    split_mxr2_ms: List[float] = field(default_factory=list)
    split_cpu_assembly_ms: List[float] = field(default_factory=list)
    split_real_batch_size: List[float] = field(default_factory=list)
    post_ms: List[float] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)
    power_w_samples: List[float] = field(default_factory=list)

    def add_timing(self, pre: float, infer: float, decode: float, post: float) -> None:
        self.preprocess_ms.append(float(pre))
        self.inference_ms.append(float(infer))
        self.decode_ms.append(float(decode))
        self.post_ms.append(float(post))
        self.e2e_ms.append(float(pre) + float(infer) + float(decode) + float(post))

    def add_post_details(self, timings: Dict[str, Any]) -> None:
        self.mx_nms_ms.append(float(timings.get("mx_nms", 0.0) or 0.0))
        self.fused_post_mx_ms.append(float(timings.get("fused_post_mx", 0.0) or 0.0))
        self.fused_pruned_mx_ms.append(float(timings.get("fused_pruned_mx", 0.0) or 0.0))
        self.topk_adapter_ms.append(float(timings.get("topk_adapter", 0.0) or 0.0))
        self.mx_assembly_total_ms.append(float(timings.get("mx_assembly_total", 0.0) or 0.0))
        self.pruned_cpu_tail_ms.append(float(timings.get("pruned_cpu_tail", 0.0) or 0.0))
        self.split_smart_heatmap_ms.append(float(timings.get("split_smart_heatmap", 0.0) or 0.0))
        self.split_mxr2_ms.append(float(timings.get("split_mxr2", 0.0) or 0.0))
        self.split_cpu_assembly_ms.append(float(timings.get("split_cpu_assembly", 0.0) or 0.0))
        self.split_real_batch_size.append(float(timings.get("split_real_batch_size", 0.0) or 0.0))

    def add_power(self, power_w: float) -> None:
        try:
            value = float(power_w)
        except Exception:
            return
        if value > 0 and not math.isnan(value) and not math.isinf(value):
            self.power_w_samples.append(value)

    def latency_summary(self) -> Dict[str, float]:
        post_avg = mean(self.post_ms)
        e2e_avg = mean(self.e2e_ms)
        avg_power = mean(self.power_w_samples)
        e2e_fps = 1000.0 / e2e_avg if e2e_avg > 0 else 0.0
        post_fps = 1000.0 / post_avg if post_avg > 0 else 0.0
        fps_per_watt = e2e_fps / avg_power if avg_power > 0 else float("nan")
        energy_j_per_frame = avg_power * (e2e_avg / 1000.0) if avg_power > 0 else float("nan")
        return {
            "images": len(self.post_ms),
            "preprocess_ms": mean(self.preprocess_ms),
            "inference_ms": mean(self.inference_ms),
            "decode_ms": mean(self.decode_ms),
            "mx_nms_ms": mean(self.mx_nms_ms),
            "fused_post_mx_ms": mean(self.fused_post_mx_ms),
            "fused_pruned_mx_ms": mean(self.fused_pruned_mx_ms),
            "topk_adapter_ms": mean(self.topk_adapter_ms),
            "mx_assembly_total_ms": mean(self.mx_assembly_total_ms),
            "pruned_cpu_tail_ms": mean(self.pruned_cpu_tail_ms),
            "split_smart_heatmap_ms": mean(self.split_smart_heatmap_ms),
            "split_mxr2_ms": mean(self.split_mxr2_ms),
            "split_cpu_assembly_ms": mean(self.split_cpu_assembly_ms),
            "split_real_batch_size": mean(self.split_real_batch_size),
            "post_avg_ms": post_avg,
            "post_p50_ms": percentile(self.post_ms, 50),
            "post_p95_ms": percentile(self.post_ms, 95),
            "e2e_avg_ms": e2e_avg,
            "e2e_p95_ms": percentile(self.e2e_ms, 95),
            "post_fps": post_fps,
            "e2e_fps": e2e_fps,
            "avg_gpu_power_w": avg_power if avg_power > 0 else float("nan"),
            "fps_per_watt": fps_per_watt,
            "energy_j_per_frame": energy_j_per_frame,
            "power_samples": len(self.power_w_samples),
        }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def fmt_float(x: Any, digits: int = 3, na: str = "N/A") -> str:
    try:
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return na
        return f"{value:.{digits}f}"
    except Exception:
        return na


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_gpu_power_w() -> float:
    for cmd in (["rocm-smi", "--showpower", "--json"], ["/opt/rocm/bin/rocm-smi", "--showpower", "--json"]):
        try:
            raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2.0).decode("utf-8")
            data = json.loads(raw)
            for card_data in data.values():
                if not isinstance(card_data, dict):
                    continue
                for key, value in card_data.items():
                    if "power" in key.lower():
                        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
                        if match:
                            return float(match.group(1))
        except Exception:
            pass
    return float("nan")


def load_migraphx_model(model_path: str, onnx_path: str = "", quantization: str = "fp16", exhaustive_tune: bool = False):
    if os.path.exists(model_path):
        print(f"--- Loading compiled model: {model_path} ---")
        return migraphx.load(model_path)
    if not onnx_path:
        raise FileNotFoundError(f"Compiled model not found: {model_path}. Pass --onnx to compile it.")
    print(f"--- Compiling ONNX model: {onnx_path} ---")
    model = migraphx.parse_onnx(onnx_path)
    target = migraphx.get_target("gpu")
    if quantization == "fp16":
        migraphx.quantize_fp16(model)
    elif quantization == "bf16":
        migraphx.quantize_bf16(model)
    elif quantization == "int8":
        migraphx.quantize_int8(model, target, [])
    elif quantization != "fp32":
        raise ValueError(f"Unsupported quantization: {quantization}")
    model.compile(target, exhaustive_tune=exhaustive_tune)
    migraphx.save(model, model_path)
    print(f"--- Saved compiled model: {model_path} ---")
    return model


def cast_input_for_model(model, tensor_nchw: np.ndarray) -> np.ndarray:
    expected_type = str(model.get_parameter_shapes()["input"].type())
    if "half" in expected_type:
        tensor_nchw = tensor_nchw.astype(np.float16)
    elif "bfloat" in expected_type:
        tensor_nchw = tensor_nchw.astype(np.float32)
    else:
        tensor_nchw = tensor_nchw.astype(np.float32)
    return np.ascontiguousarray(tensor_nchw)


def prepare_coco_input(img: np.ndarray, *, base_height: int, base_width: int, stride: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    normed_img = normalize(img, (128, 128, 128), 1 / 256)
    orig_h, orig_w, _ = normed_img.shape
    ratio = min(base_height / orig_h, base_width / orig_w)
    scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    padded_img, pad = pad_width(scaled_img, stride, (0, 0, 0), [base_height, base_width])
    tensor = np.ascontiguousarray(padded_img.transpose(2, 0, 1)[np.newaxis, ...])
    return tensor, {"orig_h": int(orig_h), "orig_w": int(orig_w), "pad": [int(x) for x in pad], "stride": int(stride)}


def run_model_on_image(model, img: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, float]]:
    timings: Dict[str, float] = {}
    with Timer() as t:
        tensor, meta = prepare_coco_input(img, base_height=args.base_height, base_width=args.base_width, stride=args.stride)
        tensor = cast_input_for_model(model, tensor)
    timings["preprocess_ms"] = t.ms

    with Timer() as t:
        raw_results = model.run({"input": tensor})
    timings["inference_ms"] = t.ms

    with Timer() as t:
        if not isinstance(raw_results, (list, tuple)):
            raw_results = list(raw_results)
        if len(raw_results) < 2:
            raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")
        heatmaps = np.transpose(np.asarray(raw_results[-2]).squeeze().astype(np.float32), (1, 2, 0))
        pafs = np.transpose(np.asarray(raw_results[-1]).squeeze().astype(np.float32), (1, 2, 0))
        top, left, bottom, right = [p // args.stride for p in meta["pad"]]
        h_end = heatmaps.shape[0] - bottom if bottom > 0 else heatmaps.shape[0]
        w_end = heatmaps.shape[1] - right if right > 0 else heatmaps.shape[1]
        heatmaps = np.ascontiguousarray(heatmaps[top:h_end, left:w_end, :], dtype=np.float32)
        pafs = np.ascontiguousarray(pafs[top:h_end, left:w_end, :], dtype=np.float32)
    timings["decode_ms"] = t.ms
    return heatmaps, pafs, (meta["orig_h"], meta["orig_w"]), timings


def model_input_batch_size(model) -> int:
    """Best-effort static batch-size detection for MIGraphX models."""
    try:
        shape = model.get_parameter_shapes()["input"]
    except Exception:
        return 1
    dims = None
    for attr in ("lens", "lengths"):
        fn = getattr(shape, attr, None)
        if callable(fn):
            try:
                dims = list(fn())
                break
            except Exception:
                pass
    if dims is None:
        match = re.search(r"\{([^}]+)\}", str(shape))
        if match:
            try:
                dims = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
            except Exception:
                dims = None
    if dims:
        try:
            return max(1, int(dims[0]))
        except Exception:
            pass
    return 1


def validation_batch_size_for_model(model, args, variants: Sequence[str]) -> int:
    explicit = int(getattr(args, "validation_batch_size", 0) or 0)
    if explicit > 0:
        return explicit
    detected = model_input_batch_size(model)
    if detected > 1:
        return detected
    if uses_split_hip_smart(variants):
        return max(1, int(getattr(args, "split_mxr2_batch_size", 4)))
    return 1


def _raw_output_to_bchw(raw: Any, *, batch_size: int, out_h: int, out_w: int, allowed_channels: Sequence[int]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    channels = tuple(int(c) for c in allowed_channels)
    batch_size = int(batch_size)
    if arr.ndim == 4 and arr.shape[0] >= batch_size and arr.shape[1] in channels:
        return np.ascontiguousarray(arr[:batch_size])
    if arr.ndim == 4 and arr.shape[0] >= batch_size and arr.shape[-1] in channels:
        return np.ascontiguousarray(np.transpose(arr[:batch_size], (0, 3, 1, 2)))
    per_spatial = int(batch_size) * int(out_h) * int(out_w)
    if arr.size % per_spatial != 0:
        raise RuntimeError(
            f"Cannot decode batched MIGraphX output: raw_shape={arr.shape}, size={arr.size}, "
            f"batch={batch_size}, out_h={out_h}, out_w={out_w}"
        )
    inferred_c = int(arr.size // per_spatial)
    if inferred_c not in channels:
        raise RuntimeError(f"Unexpected channel count {inferred_c}; expected one of {channels}, raw_shape={arr.shape}")
    return np.ascontiguousarray(arr.reshape(batch_size, inferred_c, int(out_h), int(out_w)))


def run_model_on_image_batch(model, imgs: Sequence[np.ndarray], args, *, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, float]]]:
    """Run a static-batch MIGraphX model on a group of COCO images.

    The final incomplete batch is padded by repeating the last real image. Only the
    real images are returned and evaluated.
    """
    if not imgs:
        return []
    batch_size = max(1, int(batch_size))
    real_n = len(imgs)
    if real_n > batch_size:
        raise ValueError(f"run_model_on_image_batch got {real_n} images for batch_size={batch_size}")

    tensors: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    pre_ms: List[float] = []
    for img in imgs:
        with Timer() as t:
            tensor, meta = prepare_coco_input(img, base_height=args.base_height, base_width=args.base_width, stride=args.stride)
        tensors.append(tensor)
        metas.append(meta)
        pre_ms.append(t.ms)

    if real_n < batch_size:
        for _ in range(batch_size - real_n):
            tensors.append(tensors[-1])

    tensor_nchw = np.concatenate(tensors, axis=0)
    tensor_nchw = cast_input_for_model(model, tensor_nchw)

    with Timer() as t:
        raw_results = model.run({"input": tensor_nchw})
    inference_ms_per_image = t.ms / float(max(1, real_n))

    with Timer() as t:
        if not isinstance(raw_results, (list, tuple)):
            raw_results = list(raw_results)
        if len(raw_results) < 2:
            raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")
        out_h = int(args.base_height) // int(args.stride)
        out_w = int(args.base_width) // int(args.stride)
        heat_bchw = _raw_output_to_bchw(raw_results[-2], batch_size=batch_size, out_h=out_h, out_w=out_w, allowed_channels=(18, 19))
        paf_bchw = _raw_output_to_bchw(raw_results[-1], batch_size=batch_size, out_h=out_h, out_w=out_w, allowed_channels=(38,))
    decode_ms_per_image = t.ms / float(max(1, real_n))

    decoded: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, float]]] = []
    for i in range(real_n):
        meta = metas[i]
        heatmaps = np.transpose(heat_bchw[i].astype(np.float32, copy=False), (1, 2, 0))
        pafs = np.transpose(paf_bchw[i].astype(np.float32, copy=False), (1, 2, 0))
        top, left, bottom, right = [int(p) // int(args.stride) for p in meta["pad"]]
        h_end = heatmaps.shape[0] - bottom if bottom > 0 else heatmaps.shape[0]
        w_end = heatmaps.shape[1] - right if right > 0 else heatmaps.shape[1]
        heatmaps = np.ascontiguousarray(heatmaps[top:h_end, left:w_end, :], dtype=np.float32)
        pafs = np.ascontiguousarray(pafs[top:h_end, left:w_end, :], dtype=np.float32)
        timings = {
            "preprocess_ms": float(pre_ms[i]),
            "inference_ms": float(inference_ms_per_image),
            "decode_ms": float(decode_ms_per_image),
        }
        decoded.append((heatmaps, pafs, (int(meta["orig_h"]), int(meta["orig_w"])), timings))
    return decoded


_SPLIT_MXR2_PROGRAM_CACHE: Dict[str, Any] = {}
_SPLIT_HIP_BACKEND_CACHE: Any = None


def _split_chw_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmaps)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW heatmaps rank-3, got {arr.shape}")
    if arr.shape[-1] in (18, 19):
        arr = np.moveaxis(arr[..., :18], -1, 0)
    elif arr.shape[0] in (18, 19):
        arr = arr[:18]
    else:
        raise ValueError(f"Cannot identify heatmap channel axis in shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _split_chw_paf(pafs: np.ndarray) -> np.ndarray:
    arr = np.asarray(pafs)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW PAF rank-3, got {arr.shape}")
    if arr.shape[-1] == 38:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] != 38:
        raise ValueError(f"Cannot identify PAF channel axis in shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _pad_to_batch(arr: np.ndarray, batch_size: int) -> Tuple[np.ndarray, int]:
    real_n = int(arr.shape[0])
    compiled = max(1, int(batch_size))
    if compiled < real_n:
        raise ValueError(f"Compiled split batch size {compiled} is smaller than real batch {real_n}")
    if compiled > real_n:
        pad = np.repeat(arr[-1:, ...], compiled - real_n, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return np.ascontiguousarray(arr), real_n


def _slice_batched(arr: np.ndarray, i: int) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim >= 3 and x.shape[0] > i:
        return np.ascontiguousarray(x[i : i + 1])
    return np.ascontiguousarray(x)


def _migraphx_argument(arr: np.ndarray):
    return migraphx.argument(np.ascontiguousarray(arr))


def _split_mxr2_name(args, heatmaps_hw: Tuple[int, int], original_hw: Tuple[int, int]) -> str:
    from tools.export_split_paf_pruning_from_topk import default_name

    return default_name(
        batch_size=int(args.split_mxr2_batch_size),
        in_h=int(heatmaps_hw[0]),
        in_w=int(heatmaps_hw[1]),
        full_h=int(original_hw[0]),
        full_w=int(original_hw[1]),
        topk=int(args.max_keypoints),
        limb_topm=int(args.limb_topm),
        points_per_limb=int(args.points_per_limb),
        min_paf_score=float(args.min_paf_score),
        success_ratio_thr=float(args.success_ratio_thr),
        paf_cubic_a=float(args.paf_cubic_a),
        min_pair_score=float(args.min_pair_score),
    )


def resolve_or_compile_split_mxr2(args, heatmaps_hw: Tuple[int, int], original_hw: Tuple[int, int]) -> str:
    explicit = str(getattr(args, "split_mxr2", "") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"--split-mxr2 not found: {path}")
        return str(path)

    name = _split_mxr2_name(args, heatmaps_hw, original_hw)
    cache_dir = Path(str(getattr(args, "split_mxr2_cache_dir", "models/split_paf_pruning_from_topk")))
    mxr_path = cache_dir / f"{name}.mxr"
    onnx_path = cache_dir / f"{name}.onnx"
    force = bool(getattr(args, "force_compile_postprocess_heads", False))
    if mxr_path.exists() and not force:
        return str(mxr_path)
    if not bool(getattr(args, "split_mxr2_auto_compile", False)):
        raise FileNotFoundError(
            f"Missing split MXR2 for shape {heatmaps_hw[0]}x{heatmaps_hw[1]} -> {original_hw[0]}x{original_hw[1]}: {mxr_path}\n"
            "Pass --split-mxr2 PATH or enable --split-mxr2-auto-compile."
        )

    from tools.export_split_paf_pruning_from_topk import compile_mxr, export_paf_pruning

    print(f"[split_mxr2] compiling B{int(args.split_mxr2_batch_size)} for {heatmaps_hw[0]}x{heatmaps_hw[1]} -> {original_hw[0]}x{original_hw[1]}")
    exported = export_paf_pruning(
        output_onnx=onnx_path,
        batch_size=int(args.split_mxr2_batch_size),
        in_h=int(heatmaps_hw[0]),
        in_w=int(heatmaps_hw[1]),
        full_h=int(original_hw[0]),
        full_w=int(original_hw[1]),
        topk=int(args.max_keypoints),
        limb_topm=int(args.limb_topm),
        points_per_limb=int(args.points_per_limb),
        min_paf_score=float(args.min_paf_score),
        success_ratio_thr=float(args.success_ratio_thr),
        paf_cubic_a=float(args.paf_cubic_a),
        min_pair_score=float(args.min_pair_score),
    )
    compile_mxr(exported, mxr_path, exhaustive_tune=bool(getattr(args, "exhaustive_tune", False)))
    return str(mxr_path)


def _get_split_mxr2_program(path: str):
    key = str(path)
    if key not in _SPLIT_MXR2_PROGRAM_CACHE:
        print(f"[split_mxr2] loading: {key}")
        _SPLIT_MXR2_PROGRAM_CACHE[key] = migraphx.load(key)
    return _SPLIT_MXR2_PROGRAM_CACHE[key]


def _run_split_mxr2(program: Any, paf_bchw: np.ndarray, top_scores: np.ndarray, top_indices: np.ndarray) -> Dict[str, np.ndarray]:
    outputs = program.run(
        {
            "pafs": _migraphx_argument(np.ascontiguousarray(paf_bchw.astype(np.float32, copy=False))),
            "top_scores": _migraphx_argument(np.ascontiguousarray(top_scores.astype(np.float32, copy=False))),
            "top_indices": _migraphx_argument(np.ascontiguousarray(top_indices.astype(np.int64, copy=False))),
        }
    )
    outs = [np.ascontiguousarray(np.asarray(o)) for o in outputs]
    names = ["limb_top_pair_a_idx", "limb_top_pair_b_idx", "limb_top_pair_score", "limb_top_pair_valid"]
    if len(outs) != len(names):
        raise RuntimeError(f"MXR2 returned {len(outs)} outputs, expected {len(names)}")
    return dict(zip(names, outs))


def _get_split_hip_backend():
    global _SPLIT_HIP_BACKEND_CACHE
    if _SPLIT_HIP_BACKEND_CACHE is None:
        from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend
        _SPLIT_HIP_BACKEND_CACHE = HipHeatmapTopKBackend()
    return _SPLIT_HIP_BACKEND_CACHE


def postprocess_split_hip_host_smart_batch(
    samples: Sequence[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]],
    *,
    args,
    config: PostprocessConfig,
) -> List[Any]:
    if not samples:
        return []
    original_hws = {tuple(s[2]) for s in samples}
    if len(original_hws) != 1:
        raise RuntimeError(f"split_hip_host_smart requires one full-resolution shape per batch, got {sorted(original_hws)}")
    original_hw = tuple(samples[0][2])
    heatmaps_hw = tuple(samples[0][0].shape[:2])
    if any(tuple(s[0].shape[:2]) != heatmaps_hw for s in samples):
        raise RuntimeError("split_hip_host_smart requires one low-resolution heatmap shape per batch")

    compiled_batch = max(1, int(args.split_mxr2_batch_size))
    heat_bchw = np.stack([_split_chw_heatmap(hm) for hm, _pf, _hw in samples], axis=0)
    paf_bchw = np.stack([_split_chw_paf(pf) for _hm, pf, _hw in samples], axis=0)
    heat_bchw, real_n = _pad_to_batch(heat_bchw, compiled_batch)
    paf_bchw, _ = _pad_to_batch(paf_bchw, compiled_batch)

    from modules.external_heatmap_topk_hip import HipHeatmapTopKSmartShape
    from modules.mx_pair_assembly_pruned import assemble_poses_from_pruned_pairs
    from modules.postprocessing import PostprocessOutput

    hip_shape = HipHeatmapTopKSmartShape(
        batch=int(heat_bchw.shape[0]),
        channels=18,
        in_h=int(heat_bchw.shape[2]),
        in_w=int(heat_bchw.shape[3]),
        full_h=int(original_hw[0]),
        full_w=int(original_hw[1]),
        topk=int(config.max_keypoints_per_type),
        threshold=float(config.threshold),
        lowres_nms_radius=int(args.smart_lowres_nms_radius),
        smart_proposals=int(args.smart_proposals),
        smart_local_radius=int(args.smart_local_radius),
    )

    mxr2_path = resolve_or_compile_split_mxr2(args, heatmaps_hw, original_hw)
    mxr2 = _get_split_mxr2_program(mxr2_path)

    t0 = time.perf_counter()
    top_scores, top_indices = _get_split_hip_backend().run_host_smart(heat_bchw, hip_shape)
    t1 = time.perf_counter()
    mxr2_out = _run_split_mxr2(mxr2, paf_bchw, top_scores, top_indices)
    t2 = time.perf_counter()

    heatmap_ms_total = (t1 - t0) * 1000.0
    mxr2_ms_total = (t2 - t1) * 1000.0
    outputs: List[Any] = []
    for i in range(real_n):
        t_asm0 = time.perf_counter()
        poses, keypoints, asm_times = assemble_poses_from_pruned_pairs(
            _slice_batched(top_scores, i),
            _slice_batched(top_indices, i),
            _slice_batched(mxr2_out["limb_top_pair_a_idx"], i),
            _slice_batched(mxr2_out["limb_top_pair_b_idx"], i),
            _slice_batched(mxr2_out["limb_top_pair_score"], i),
            _slice_batched(mxr2_out["limb_top_pair_valid"], i),
            full_width=int(original_hw[1]),
            threshold=float(config.threshold),
            min_pair_score=float(config.extra.get("min_pair_score", 0.0)),
            return_timing=True,
        )
        asm_ms = (time.perf_counter() - t_asm0) * 1000.0
        per_frame_gpu_ms = (heatmap_ms_total + mxr2_ms_total) / float(max(1, real_n))
        timings: Dict[str, float] = {
            "split_smart_heatmap": heatmap_ms_total / float(max(1, real_n)),
            "split_smart_heatmap_batch": heatmap_ms_total,
            "split_mxr2": mxr2_ms_total / float(max(1, real_n)),
            "split_mxr2_batch": mxr2_ms_total,
            "split_cpu_assembly": asm_ms,
            "split_total_batch": heatmap_ms_total + mxr2_ms_total,
            "split_real_batch_size": float(real_n),
            "split_compiled_batch_size": float(heat_bchw.shape[0]),
            "valid_topk_count": float(np.sum(_slice_batched(top_scores, i) > -1.0e8)),
            "limb_valid_count": float(np.sum(_slice_batched(mxr2_out["limb_top_pair_valid"], i) > 0.5)),
            "fused_pruned_mx": mxr2_ms_total / float(max(1, real_n)),
            "fused_post_mx": (heatmap_ms_total + mxr2_ms_total) / float(max(1, real_n)),
            "pruned_cpu_tail": asm_ms,
            "total_postprocess": per_frame_gpu_ms + asm_ms,
        }
        for key, value in dict(asm_times).items():
            try:
                timings[str(key)] = float(value)
            except Exception:
                pass
        outputs.append(PostprocessOutput(np.asarray(poses, dtype=np.float32), np.asarray(keypoints, dtype=np.float32), timings))
    return outputs


def coco_eval_stats(gt_file_path: str, dt_file_path: str, img_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, "keypoints")
    if img_ids is not None:
        result.params.imgIds = [int(x) for x in img_ids]
    result.evaluate()
    result.accumulate()
    result.summarize()
    keys = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
    return {key: float(value) for key, value in zip(keys, result.stats.copy())}


def build_coco_detections(image_id: int, pose_entries: np.ndarray, all_keypoints: np.ndarray) -> List[dict]:
    coco_result = []
    if pose_entries is None or len(pose_entries) == 0:
        return coco_result
    all_keypoints = np.asarray(all_keypoints, dtype=np.float32)
    for pose_entry in pose_entries:
        if len(pose_entry) == 0:
            continue
        keypoints = [0] * 17 * 3
        person_score = float(pose_entry[-2])
        position_id = -1
        for keypoint_id in pose_entry[:-2]:
            position_id += 1
            if position_id == 1:
                continue
            cx, cy, visibility = 0.0, 0.0, 0
            if keypoint_id != -1:
                kp = all_keypoints[int(keypoint_id)]
                cx = float(kp[0] + 0.5)
                cy = float(kp[1] + 0.5)
                visibility = 1
            coco_idx = COCO_KPT_MAP[position_id]
            if coco_idx >= 0:
                keypoints[coco_idx * 3 + 0] = cx
                keypoints[coco_idx * 3 + 1] = cy
                keypoints[coco_idx * 3 + 2] = visibility
        score = person_score * max(0.0, float(pose_entry[-1] - 1))
        coco_result.append({"image_id": image_id, "category_id": 1, "keypoints": keypoints, "score": float(score)})
    return coco_result


def parse_model_metadata(model_path: str) -> Dict[str, str]:
    name = Path(model_path).name.lower()
    precision = "unknown"
    for token in ["fp32", "fp16", "bf16", "int8"]:
        if token in name:
            precision = token
            break
    match = re.search(r"ref(\d+)", name)
    return {"precision": precision, "refinement_stages": match.group(1) if match else "unknown"}


def select_coco_images(args) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(args.labels, "r") as f:
        labels = json.load(f)
    images = list(labels.get("images", []))
    if args.skip_images:
        images = images[int(args.skip_images):]

    requested = args.num_of_test_img if args.num_of_test_img is not None else args.max_images
    if requested is not None:
        requested = int(requested)

    if args.image_selection == "sequential":
        selected = images[:requested] if requested is not None else images
        return selected, {"selection": "sequential", "requested": requested, "selected": len(selected)}

    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for info in images:
        groups[(int(info.get("height", 0)), int(info.get("width", 0)))].append(info)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    selected: List[Dict[str, Any]] = []
    used_groups = []
    for (h, w), group in ordered:
        need = None if requested is None else requested - len(selected)
        if need is not None and need <= 0:
            break
        take = group if need is None else group[:need]
        selected.extend(take)
        used_groups.append({"height": h, "width": w, "available": len(group), "selected": len(take)})
    manifest = {"selection": "dominant-dimensions", "requested": requested, "selected": len(selected), "groups_used": used_groups}
    return selected, manifest


def write_flat_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    preferred_order = [
        "model", "variant", "images", "AP", "AP50", "AP75", "AR", "preprocess_ms", "inference_ms", "decode_ms",
        "mx_nms_ms", "fused_post_mx_ms", "fused_pruned_mx_ms", "topk_adapter_ms", "mx_assembly_total_ms",
        "pruned_cpu_tail_ms", "split_smart_heatmap_ms", "split_mxr2_ms", "split_cpu_assembly_ms",
        "split_real_batch_size", "post_avg_ms", "post_p50_ms", "post_p95_ms", "e2e_avg_ms", "e2e_p95_ms", "e2e_fps",
        "avg_gpu_power_w", "fps_per_watt", "energy_j_per_frame", "detections_path",
    ]
    all_keys = set().union(*(row.keys() for row in rows))
    fieldnames = [k for k in preferred_order if k in all_keys]
    fieldnames.extend(sorted(k for k in all_keys if k not in fieldnames))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_variant_descriptions(variants: Sequence[str]) -> None:
    info_by_name = {row["variant"]: row for row in variant_table()}
    print("\nVariants:")
    for variant in variants:
        if variant == SPLIT_HIP_SMART_MODE:
            print(f"  {variant:<40} Split MXR1 -> native HIP smart heatmap TopK -> MXR2 pruned PAF scoring -> CPU assembly.")
            continue
        row = info_by_name[variant]
        print(f"  {variant:<40} {row['description']}")


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 190)
    print("ACCURACY / SPEED VALIDATION SUMMARY")
    print("=" * 190)
    print(
        f"{'model':<28} {'variant':<40} {'img':>5} {'AP':>6} {'AP50':>6} {'AP75':>6} {'AR':>6} "
        f"{'pre':>7} {'infer':>7} {'dec':>7} {'fused':>8} {'pruned':>8} {'post':>8} {'e2e':>8} {'FPS':>8}"
    )
    print("-" * 190)
    for r in rows:
        print(
            f"{r['model']:<28} {r['variant']:<40} {int(r['images']):>5} "
            f"{fmt_float(r.get('AP'), 3):>6} {fmt_float(r.get('AP50'), 3):>6} {fmt_float(r.get('AP75'), 3):>6} {fmt_float(r.get('AR'), 3):>6} "
            f"{fmt_float(r.get('preprocess_ms'), 2):>7} {fmt_float(r.get('inference_ms'), 2):>7} {fmt_float(r.get('decode_ms'), 2):>7} "
            f"{fmt_float(r.get('fused_post_mx_ms'), 2):>8} {fmt_float(r.get('fused_pruned_mx_ms'), 2):>8} "
            f"{fmt_float(r.get('post_avg_ms'), 2):>8} {fmt_float(r.get('e2e_avg_ms'), 2):>8} {fmt_float(r.get('e2e_fps'), 2):>8}"
        )
    print("=" * 190)


def build_postprocess_config(args) -> PostprocessConfig:
    return PostprocessConfig(
        max_keypoints_per_type=args.max_keypoints,
        threshold=args.threshold,
        points_per_limb=args.points_per_limb,
        nms_radius_fullres=args.nms_radius_fullres,
        nms_radius_lowres=args.nms_radius_lowres,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        torch_device=args.torch_device,
        require_gpu=args.require_gpu,
        migraphx_nms_mxr=args.migraphx_nms_mxr,
        migraphx_nms_cache_dir=args.migraphx_nms_cache_dir,
        migraphx_manual_cubic_topk_mxr=args.migraphx_manual_cubic_topk_mxr,
        migraphx_manual_cubic_topk_cache_dir=args.migraphx_manual_cubic_topk_cache_dir,
        extra=postprocess_extra_from_args(args),
    )


def _assert_accuracy_variants_are_migraphx_safe(variants: Sequence[str]) -> None:
    unsafe = [v for v in variants if v.startswith("gpu_") and not is_two_process_mode(v)]
    if unsafe:
        raise RuntimeError(f"Unsafe single-process GPU postprocess variants in accuracy validation: {unsafe}")
    if any(is_two_process_mode(v) for v in variants):
        raise RuntimeError("This accuracy_validation.py version focuses on single-process CPU/MIGraphX variants. Use older two-process scripts for Torch GPU variants.")


def _assert_single_shape_for_split(selected_images: Sequence[Dict[str, Any]], variants: Sequence[str]) -> None:
    if not uses_split_hip_smart(variants):
        return
    dims = sorted({(int(x.get("height", 0)), int(x.get("width", 0))) for x in selected_images})
    if len(dims) > 1:
        raise RuntimeError(
            "split_hip_host_smart uses a statically compiled MXR2 and therefore requires one selected full-resolution shape. "
            f"Selected shapes: {dims}. Reduce --num-of-test-img or use a stricter dominant-dimensions subset."
        )


def validate_accuracy(args) -> List[Dict[str, Any]]:
    ensure_dir(args.output_dir)
    variants = [normalize_accuracy_variant(v) for v in args.variants]
    variants = list(dict.fromkeys(variants))
    _assert_accuracy_variants_are_migraphx_safe(variants)

    selected_images, selection_manifest = select_coco_images(args)
    _assert_single_shape_for_split(selected_images, variants)
    selected_image_ids = [int(x["id"]) for x in selected_images]
    target_count = len(selected_images)
    manifest_path = Path(args.output_dir) / "selected_images_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({**selection_manifest, "image_ids": selected_image_ids, "images": selected_images}, f, indent=2)

    config = build_postprocess_config(args)
    compiled_shapes = set()
    all_rows: List[Dict[str, Any]] = []

    print("\nRunning COCO accuracy validation")
    print(f"Labels:        {args.labels}")
    print(f"Images folder: {args.images_folder}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Models:        {', '.join(args.models)}")
    print(f"Selected imgs: {target_count}")
    print(f"Selection:     {selection_manifest['selection']}")
    print(f"Auto compile:  {args.compile_missing_postprocess_heads}")
    if uses_split_hip_smart(variants):
        print(f"Split MXR2:    {args.split_mxr2 or '[auto]' }")
        print(f"Split auto:    {args.split_mxr2_auto_compile}")
        print(f"Split B:       {args.split_mxr2_batch_size}")
    print_variant_descriptions(variants)

    for model_path in args.models:
        model = load_migraphx_model(model_path, args.onnx, args.quantization, args.exhaustive_tune)
        model_name = Path(model_path).name
        model_dir = Path(args.output_dir) / Path(model_path).stem
        model_dir.mkdir(parents=True, exist_ok=True)
        meters = {variant: VariantMeter(model=model_name, variant=variant) for variant in variants}
        validation_batch_size = validation_batch_size_for_model(model, args, variants)

        print(f"\n--- Evaluating model: {model_name} ---")
        print(f"  validation batch size: {validation_batch_size}")
        processed = 0
        while processed < target_count:
            batch_infos = selected_images[processed : processed + validation_batch_size]
            batch_imgs = []
            for j, info in enumerate(batch_infos):
                file_name = info["file_name"]
                if processed == 0 or (args.progress_every > 0 and processed % args.progress_every == 0 and j == 0):
                    print(f"  image {processed + 1}/{target_count}: {file_name} ({info.get('height')}x{info.get('width')})")
                img = cv2.imread(os.path.join(args.images_folder, file_name), cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Could not read COCO image: {file_name}")
                batch_imgs.append(img)

            batch_maps = run_model_on_image_batch(model, batch_imgs, args, batch_size=validation_batch_size)
            if len(batch_maps) != len(batch_infos):
                raise RuntimeError(f"Internal batch decode mismatch: got {len(batch_maps)}, expected {len(batch_infos)}")

            for heatmaps, _pafs, original_hw, _infer_timings in batch_maps:
                ensure_shape_postprocess_heads(args, variants, heatmaps.shape[:2], original_hw, compiled=compiled_shapes)

            split_outputs_cache: Optional[List[Any]] = None
            if SPLIT_HIP_SMART_MODE in variants:
                split_samples = [(hm, pf, hw) for hm, pf, hw, _timings in batch_maps]
                split_outputs_cache = postprocess_split_hip_host_smart_batch(split_samples, args=args, config=config)

            for local_idx, (info, maps_tuple) in enumerate(zip(batch_infos, batch_maps)):
                image_id = int(info["id"])
                heatmaps, pafs, original_hw, infer_timings = maps_tuple
                for variant in variants:
                    if variant == SPLIT_HIP_SMART_MODE:
                        if split_outputs_cache is None:
                            raise RuntimeError("Missing split output cache")
                        out = split_outputs_cache[local_idx]
                    else:
                        out = postprocess_from_maps(variant, heatmaps, pafs, original_hw, config=config)
                    meter = meters[variant]
                    meter.add_timing(
                        pre=infer_timings["preprocess_ms"],
                        infer=infer_timings["inference_ms"],
                        decode=infer_timings["decode_ms"],
                        post=out.timings.get("total_postprocess", 0.0),
                    )
                    meter.add_post_details(out.timings)
                    meter.detections.extend(build_coco_detections(image_id, out.pose_entries, out.all_keypoints))
                    if args.power_every > 0 and processed % args.power_every == 0:
                        meter.add_power(get_gpu_power_w())
                processed += 1

        for variant, meter in meters.items():
            detections_path = model_dir / f"detections_{variant}.json"
            with open(detections_path, "w") as f:
                json.dump(meter.detections, f)
            print(f"\n--- COCO eval: model={model_name}, variant={variant} ---")
            metrics = coco_eval_stats(args.labels, str(detections_path), img_ids=selected_image_ids)
            latency = meter.latency_summary()
            meta = parse_model_metadata(model_path)
            all_rows.append({
                "model": model_name,
                "model_path": model_path,
                "variant": variant,
                **meta,
                **latency,
                **metrics,
                "detections_path": str(detections_path),
                "selected_images_manifest": str(manifest_path),
                "image_selection": selection_manifest["selection"],
            })
        del model

    summary_json = Path(args.output_dir) / "accuracy_validation_summary.json"
    summary_csv = Path(args.output_dir) / "accuracy_validation_summary.csv"
    with open(summary_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    write_flat_csv(str(summary_csv), all_rows)
    print_summary_table(all_rows)
    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV:  {summary_csv}")
    print(f"Saved image manifest: {manifest_path}")
    return all_rows

def parse_args():
    parser = argparse.ArgumentParser(description="Unified COCO AP/AR and speed validation using modules.postprocessing.")
    parser.add_argument("--models", nargs="+", default=["pose_model1_fp16_ref1.mxr"])
    parser.add_argument("--model", dest="single_model", default=None)
    parser.add_argument("--onnx", default="")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--exhaustive-tune", action="store_true")
    parser.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    parser.add_argument("--images-folder", default="coco/val2017/")
    parser.add_argument("--output-dir", default="outputs/accuracy_validation")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--num-of-test-img", type=int, default=None, help="Number of selected COCO images. Overrides --max_images when set.")
    parser.add_argument("--image-selection", choices=["sequential", "dominant-dimensions"], default="sequential")
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--power-every", type=int, default=10)
    parser.add_argument("--validation-batch-size", type=int, default=0, help="Static evaluation batch size. 0 means infer from model input shape.")
    parser.add_argument("--base-height", type=int, default=BASE_HEIGHT)
    parser.add_argument("--base-width", type=int, default=BASE_WIDTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_ACCURACY_VARIANTS), help=f"Postprocess variants or aliases. Canonical modes: {', '.join(available_modes())}")
    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--points-per-limb", type=int, default=8)
    parser.add_argument("--min-paf-score", type=float, default=0.05)
    parser.add_argument("--success-ratio-thr", type=float, default=0.8)
    parser.add_argument("--migraphx-nms-mxr", default="")
    parser.add_argument("--migraphx-nms-cache-dir", default="")
    parser.add_argument("--compile-missing-postprocess-heads", action="store_true")
    parser.add_argument("--force-compile-postprocess-heads", action="store_true")
    parser.add_argument("--keep-postprocess-onnx", action="store_true")
    parser.add_argument("--migraphx-manual-cubic-topk-mxr", default="")
    parser.add_argument("--migraphx-manual-cubic-topk-cache-dir", default="models/manual_cubic_nms_topk_cache")
    parser.add_argument("--manual-cubic-topk", type=int, default=20)
    parser.add_argument("--manual-cubic-threshold", type=float, default=0.1)
    parser.add_argument("--manual-cubic-nms-radius", type=int, default=6)
    parser.add_argument("--manual-cubic-nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--manual-cubic-a", type=float, default=-0.75)
    parser.add_argument("--fused-postprocess-mxr", default="")
    parser.add_argument("--fused-postprocess-cache-dir", default="models/fused_postprocess_cache")
    parser.add_argument("--fused-pruned-postprocess-mxr", default="")
    parser.add_argument("--fused-pruned-postprocess-cache-dir", default="models/fused_postprocess_pruned_cache")
    parser.add_argument("--split-mxr2", default="", help="Explicit MXR2 model for split_hip_host_smart. If omitted, use/compile from --split-mxr2-cache-dir.")
    parser.add_argument("--split-mxr2-cache-dir", default="models/split_paf_pruning_from_topk")
    parser.add_argument("--split-mxr2-auto-compile", action="store_true", help="Auto-export/compile B4 MXR2 for the selected dominant COCO full-resolution shape.")
    parser.add_argument("--split-mxr2-batch-size", type=int, default=4)
    parser.add_argument("--smart-proposals", type=int, default=32)
    parser.add_argument("--smart-local-radius", type=int, default=4)
    parser.add_argument("--smart-lowres-nms-radius", type=int, default=1)
    parser.add_argument("--limb-topm", type=int, default=20)
    parser.add_argument("--min-pair-score", type=float, default=0.0)
    parser.add_argument("--paf-cubic-a", type=float, default=-0.75)
    args = parser.parse_args()
    if args.single_model:
        args.models = [args.single_model]
    return args


if __name__ == "__main__":
    validate_accuracy(parse_args())
