#!/usr/bin/env python3
"""
accuracy_validation.py

Unified COCO AP/AR + latency + power/energy validation for MIGraphX models and
post-processing variants.

This replaces the overlapping benchmark_migraphx_video.py,
benchmark_postprocess_accuracy_with_gpu_acc.py, and related benchmark scripts.
All post-processing variants are imported from modules/postprocessing.py.

Example
-------
python accuracy_validation.py \
  --models pose_model1_fp16_ref1.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017 \
  --variants standard optimized_batch_k20_fast gpu_nms_fullres_cpu_group \
  --max-images 5000 \
  --output-dir outputs/accuracy_validation
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import migraphx
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Do not import or query torch.cuda in the main accuracy process.
# On this ROCm setup, MIGraphX inference and PyTorch GPU postprocess must live
# in different Python processes. GPU postprocess workers import Torch inside
# modules.multiprocess_accuracy_support.
torch = None

from datasets.coco import CocoValDataset
from modules.postprocessing import (
    DEFAULT_ACCURACY_VARIANTS,
    PostprocessConfig,
    available_modes,
    is_two_process_mode,
    normalize_mode,
    postprocess_from_maps,
    two_process_worker_mode,
    variant_table,
)
from modules.multiprocess_accuracy_support import (
    run_accuracy_postprocess_item,
    start_accuracy_postprocess_worker,
    stop_accuracy_postprocess_worker,
)

try:
    from val import normalize, pad_width
except Exception:  # pragma: no cover - fallback keeps this script usable in trimmed repos.
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


@dataclass
class VariantMeter:
    model: str
    variant: str
    detections: List[dict] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    mx_nms_ms: List[float] = field(default_factory=list)
    extract_from_mask_ms: List[float] = field(default_factory=list)
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
        try:
            self.mx_nms_ms.append(float(timings.get("mx_nms", 0.0) or 0.0))
            self.extract_from_mask_ms.append(float(timings.get("extract_from_mask", 0.0) or 0.0))
        except Exception:
            self.mx_nms_ms.append(0.0)
            self.extract_from_mask_ms.append(0.0)

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
            "extract_from_mask_ms": mean(self.extract_from_mask_ms),
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


def sync_if_gpu() -> None:
    # Intentionally no-op. Do not touch torch.cuda in the MIGraphX accuracy process.
    return


def get_gpu_power_w() -> float:
    """Read AMD GPU package power through rocm-smi. Returns NaN on failure."""
    json_commands = [
        ["rocm-smi", "--showpower", "--json"],
        ["/opt/rocm/bin/rocm-smi", "--showpower", "--json"],
    ]
    for cmd in json_commands:
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

    text_commands = [
        ["rocm-smi", "--showpower"],
        ["/opt/rocm/bin/rocm-smi", "--showpower"],
    ]
    patterns = [
        r"Current\s+Socket\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Average\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
    ]
    for cmd in text_commands:
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2.0)
            text = completed.stdout + "\n" + completed.stderr
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
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
        # MIGraphX Python input is safest as float32 for bf16 models.
        tensor_nchw = tensor_nchw.astype(np.float32)
    else:
        tensor_nchw = tensor_nchw.astype(np.float32)
    return np.ascontiguousarray(tensor_nchw)


def prepare_coco_input(
    img: np.ndarray,
    *,
    base_height: int,
    base_width: int,
    stride: int,
    pad_value=(0, 0, 0),
    img_mean=(128, 128, 128),
    img_scale=1 / 256,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    normed_img = normalize(img, img_mean, img_scale)
    orig_h, orig_w, _ = normed_img.shape
    ratio = min(base_height / orig_h, base_width / orig_w)

    scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    scaled_h, scaled_w, _ = scaled_img.shape
    padded_img, pad = pad_width(scaled_img, stride, pad_value, [base_height, base_width])

    tensor = padded_img.transpose(2, 0, 1)[np.newaxis, ...]
    tensor = np.ascontiguousarray(tensor)
    meta = {
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "scaled_h": int(scaled_h),
        "scaled_w": int(scaled_w),
        "pad": [int(x) for x in pad],
        "stride": int(stride),
        "base_height": int(base_height),
        "base_width": int(base_width),
    }
    return tensor, meta


def run_model_on_image(model, img: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Dict[str, float]]:
    timings: Dict[str, float] = {}

    with Timer() as t:
        tensor, meta = prepare_coco_input(
            img,
            base_height=args.base_height,
            base_width=args.base_width,
            stride=args.stride,
        )
        tensor = cast_input_for_model(model, tensor)
    timings["preprocess_ms"] = t.ms

    with Timer() as t:
        raw_results = model.run({"input": tensor})
        sync_if_gpu()
    timings["inference_ms"] = t.ms

    with Timer() as t:
        if not isinstance(raw_results, (list, tuple)):
            raw_results = list(raw_results)
        if len(raw_results) < 2:
            raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")
        heatmaps = np.transpose(np.asarray(raw_results[-2]).squeeze().astype(np.float32), (1, 2, 0))
        pafs = np.transpose(np.asarray(raw_results[-1]).squeeze().astype(np.float32), (1, 2, 0))

        pad = meta["pad"]
        scaled_pad = [p // args.stride for p in pad]
        top, left, bottom, right = scaled_pad
        h_end = heatmaps.shape[0] - bottom if bottom > 0 else heatmaps.shape[0]
        w_end = heatmaps.shape[1] - right if right > 0 else heatmaps.shape[1]
        heatmaps = heatmaps[top:h_end, left:w_end, :]
        pafs = pafs[top:h_end, left:w_end, :]
        heatmaps = np.ascontiguousarray(heatmaps, dtype=np.float32)
        pafs = np.ascontiguousarray(pafs, dtype=np.float32)
    timings["decode_ms"] = t.ms

    return heatmaps, pafs, (meta["orig_h"], meta["orig_w"]), timings


def coco_eval_stats(gt_file_path: str, dt_file_path: str) -> Dict[str, float]:
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, "keypoints")
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
                continue  # neck is not a COCO keypoint

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
        coco_result.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "score": float(score),
            }
        )

    return coco_result


def parse_model_metadata(model_path: str) -> Dict[str, str]:
    name = Path(model_path).name.lower()
    precision = "unknown"
    for token in ["fp32", "fp16", "bf16", "int8"]:
        if token in name:
            precision = token
            break
    ref = "unknown"
    match = re.search(r"ref(\d+)", name)
    if match:
        ref = match.group(1)
    return {"precision": precision, "refinement_stages": ref}


def write_flat_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # Rows can come from different execution paths (single-process and
    # two-process).  Two-process summaries contain extra fields such as
    # pipeline_wall_s / pipeline_fps, so the CSV header must be the union
    # of all row keys, not only keys from the first row.
    preferred_order = [
        "variant", "model", "model_path", "precision", "refinement_stages",
        "frames", "images",
        "preprocess_ms", "inference_ms", "decode_ms",
        "hm_resize_ms", "paf_resize_ms", "mx_nms_ms", "extract_ms", "extract_from_mask_ms", "group_ms",
        "post_avg_ms", "post_p50_ms", "post_p95_ms", "post_fps",
        "e2e_avg_ms", "e2e_p95_ms", "e2e_fps",
        "e2e_speedup_vs_standard", "e2e_delta_pct_vs_standard",
        "avg_power_w", "fps_per_watt", "energy_j_per_frame",
        "pipeline_wall_s", "pipeline_fps",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
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
        row = info_by_name[variant]
        print(f"  {variant:<32} {row['description']}")


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 170)
    print("ACCURACY / ENERGY VALIDATION SUMMARY")
    print("=" * 170)
    print(
        f"{'model':<30} {'variant':<32} {'img':>5} "
        f"{'AP':>6} {'AP50':>6} {'AP75':>6} {'AR':>6} "
        f"{'pre':>8} {'infer':>8} {'dec':>7} {'mx_nms':>8} {'mask':>8} {'post':>8} {'e2e':>8} "
        f"{'FPS':>8} {'W':>8} {'FPS/W':>8} {'J/frame':>9}"
    )
    print("-" * 170)
    for r in rows:
        print(
            f"{r['model']:<30} {r['variant']:<32} {int(r['images']):>5} "
            f"{fmt_float(r.get('AP'), 3):>6} {fmt_float(r.get('AP50'), 3):>6} "
            f"{fmt_float(r.get('AP75'), 3):>6} {fmt_float(r.get('AR'), 3):>6} "
            f"{fmt_float(r.get('preprocess_ms'), 2):>8} {fmt_float(r.get('inference_ms'), 2):>8} "
            f"{fmt_float(r.get('decode_ms'), 2):>7} {fmt_float(r.get('mx_nms_ms'), 2):>8} "
            f"{fmt_float(r.get('extract_from_mask_ms'), 2):>8} {fmt_float(r.get('post_avg_ms'), 2):>8} "
            f"{fmt_float(r.get('e2e_avg_ms'), 2):>8} {fmt_float(r.get('e2e_fps'), 2):>8} "
            f"{fmt_float(r.get('avg_gpu_power_w'), 2):>8} {fmt_float(r.get('fps_per_watt'), 3):>8} "
            f"{fmt_float(r.get('energy_j_per_frame'), 4):>9}"
        )
    print("=" * 170)



def _assert_accuracy_variants_are_migraphx_safe(variants: Sequence[str]) -> None:
    """Guard COCO validation from mixing MIGraphX and Torch GPU in one process."""
    unsafe = [v for v in variants if v.startswith("gpu_") and not is_two_process_mode(v)]
    if unsafe:
        raise RuntimeError(
            "Unsafe single-process GPU postprocess variant requested in COCO accuracy validation. "
            "MIGraphX and PyTorch ROCm must not run in the same process on this setup. "
            f"Requested: {unsafe}. Use the corresponding *_two_process variant, e.g. "
            "gpu_nms_fullres_two_process or gpu_nms_lowres_two_process."
        )


def validate_accuracy(args) -> List[Dict[str, Any]]:
    ensure_dir(args.output_dir)
    variants = [normalize_mode(v) for v in args.variants]
    variants = list(dict.fromkeys(variants))

    _assert_accuracy_variants_are_migraphx_safe(variants)

    if any(v in {"migraphx_nms", "migraphx_nms_k20"} for v in variants):
        if args.compile_migraphx_nms_cache:
            from modules.migraphx_compiler import compile_nms_cache_for_coco

            cache_dir = args.migraphx_nms_cache_dir or "models/nms_fullres_cache"
            compile_nms_cache_for_coco(
                annotations=args.labels,
                output_dir=cache_dir,
                max_images=args.max_images,
                skip_images=args.skip_images,
                threshold=args.threshold,
                radius=args.nms_radius_fullres,
                force=args.force_compile_migraphx_nms,
                keep_onnx=args.keep_migraphx_nms_onnx,
                exhaustive_tune=args.exhaustive_tune_migraphx_nms,
            )
            args.migraphx_nms_cache_dir = cache_dir
            print(f"MIGraphX NMS cache ready: {cache_dir}")
        elif not args.migraphx_nms_cache_dir and not args.migraphx_nms_mxr:
            raise RuntimeError(
                "migraphx_nms variants require --migraphx-nms-cache-dir or --migraphx-nms-mxr. "
                "For COCO accuracy validation, use --compile-migraphx-nms-cache so all required per-resolution heads are built before AP/AR evaluation."
            )

    config = PostprocessConfig(
        max_keypoints_per_type=args.max_keypoints,
        threshold=args.threshold,
        nms_radius_fullres=args.nms_radius_fullres,
        nms_radius_lowres=args.nms_radius_lowres,
        torch_device=args.torch_device,
        require_gpu=args.require_gpu,
        migraphx_nms_mxr=args.migraphx_nms_mxr,
        migraphx_nms_cache_dir=args.migraphx_nms_cache_dir,
        extra={"gpu_compute_dtype": args.gpu_compute_dtype, "nms_impl": args.nms_impl},
    )

    single_process_variants = [v for v in variants if not is_two_process_mode(v)]
    two_process_variants = [v for v in variants if is_two_process_mode(v)]

    print("\nRunning COCO accuracy validation")
    print(f"Labels:        {args.labels}")
    print(f"Images folder: {args.images_folder}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Models:        {', '.join(args.models)}")
    print(f"Max images:    {args.max_images}")
    print(f"Skip images:   {args.skip_images}")
    print(f"Power every:   {args.power_every}")
    print(f"Torch device:  {args.torch_device}")
    print(f"NMS impl:      {args.nms_impl}")
    print(f"GPU dtype:     {args.gpu_compute_dtype}")
    print(f"Single-process variants: {', '.join(single_process_variants) if single_process_variants else 'none'}")
    print(f"Two-process variants:    {', '.join(two_process_variants) if two_process_variants else 'none'}")
    print_variant_descriptions(variants)

    dataset = CocoValDataset(args.labels, args.images_folder)
    total_images = len(dataset)
    target_count = args.max_images if args.max_images is not None else max(0, total_images - args.skip_images)

    all_rows: List[Dict[str, Any]] = []

    for model_path in args.models:
        model = load_migraphx_model(model_path, args.onnx, args.quantization, args.exhaustive_tune)
        model_name = Path(model_path).name
        model_dir = Path(args.output_dir) / Path(model_path).stem
        model_dir.mkdir(parents=True, exist_ok=True)

        meters = {variant: VariantMeter(model=model_name, variant=variant) for variant in variants}
        processed = 0

        print(f"\n--- Evaluating model: {model_name} ---")

        workers = {}
        for variant in two_process_variants:
            workers[variant] = start_accuracy_postprocess_worker(
                variant_name=variant,
                worker_mode=two_process_worker_mode(variant),
                torch_device="cuda" if args.torch_device == "auto" else args.torch_device,
                max_keypoints=args.max_keypoints,
                threshold=args.threshold,
                nms_radius_fullres=args.nms_radius_fullres,
                nms_radius_lowres=args.nms_radius_lowres,
                nms_impl=args.nms_impl,
                gpu_compute_dtype=args.gpu_compute_dtype,
                queue_size=args.two_process_queue_size,
            )

        try:
            for i, sample in enumerate(dataset):  # type: ignore
                if i < args.skip_images:
                    continue
                if args.max_images is not None and processed >= args.max_images:
                    break

                file_name = sample["file_name"]
                img = sample["img"]
                image_id = int(file_name[0 : file_name.rfind(".")])

                if processed == 0 or (args.progress_every > 0 and processed % args.progress_every == 0):
                    print(f"  image {processed + 1}/{target_count}: {file_name}")

                heatmaps, pafs, original_hw, infer_timings = run_model_on_image(model, img, args)

                for variant in single_process_variants:
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

                for variant in two_process_variants:
                    result = run_accuracy_postprocess_item(
                        workers[variant],
                        image_id=image_id,
                        heatmaps=heatmaps,
                        pafs=pafs,
                        original_hw=original_hw,
                        timeout_s=args.two_process_timeout,
                    )

                    meter = meters[variant]
                    post_timing = result.get("timings", {}).get("total_postprocess", 0.0)
                    meter.add_timing(
                        pre=infer_timings["preprocess_ms"],
                        infer=infer_timings["inference_ms"],
                        decode=infer_timings["decode_ms"],
                        post=post_timing,
                    )
                    meter.add_post_details(result.get("timings", {}))
                    meter.detections.extend(
                        build_coco_detections(
                            image_id,
                            result["pose_entries"],
                            result["all_keypoints"],
                        )
                    )

                    if args.power_every > 0 and processed % args.power_every == 0:
                        meter.add_power(get_gpu_power_w())

                processed += 1
        finally:
            for worker in workers.values():
                stop_accuracy_postprocess_worker(worker)

        for variant, meter in meters.items():
            detections_path = model_dir / f"detections_{variant}.json"
            with open(detections_path, "w") as f:
                json.dump(meter.detections, f)

            print(f"\n--- COCO eval: model={model_name}, variant={variant} ---")
            metrics = coco_eval_stats(args.labels, str(detections_path))
            latency = meter.latency_summary()
            meta = parse_model_metadata(model_path)

            row = {
                "model": model_name,
                "model_path": model_path,
                "variant": variant,
                **meta,
                **latency,
                **metrics,
                "detections_path": str(detections_path),
            }
            all_rows.append(row)

        # pycocotools/dataset can be reused, but explicitly drop model reference between models.
        del model

    summary_json = Path(args.output_dir) / "accuracy_validation_summary.json"
    summary_csv = Path(args.output_dir) / "accuracy_validation_summary.csv"
    with open(summary_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    write_flat_csv(str(summary_csv), all_rows)

    print_summary_table(all_rows)
    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV:  {summary_csv}")
    return all_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified COCO AP/AR, speed, power, and energy validation using modules.postprocessing."
    )
    parser.add_argument("--models", nargs="+", default=["pose_model1_fp16_ref1.mxr"], help="One or more compiled MIGraphX .mxr models.")
    parser.add_argument("--model", dest="single_model", default=None, help="Backward-compatible single model argument.")
    parser.add_argument("--onnx", default="", help="Optional ONNX path if a requested .mxr model does not exist.")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--exhaustive-tune", action="store_true")
    parser.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    parser.add_argument("--images-folder", default="coco/val2017/")
    parser.add_argument("--output-dir", default="outputs/accuracy_validation")
    parser.add_argument("--max-images", type=int, default=5000)
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--power-every", type=int, default=10, help="Sample rocm-smi every N processed images per variant. Use 0 to disable.")
    parser.add_argument("--base-height", type=int, default=BASE_HEIGHT)
    parser.add_argument("--base-width", type=int, default=BASE_WIDTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_ACCURACY_VARIANTS),
        help=f"Postprocess variants or aliases. Canonical modes: {', '.join(available_modes())}",
    )
    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of CPU fallback for GPU variants.")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--migraphx-nms-mxr", default="", help="Static compiled MIGraphX NMS .mxr. Mainly useful for fixed-size subsets.")
    parser.add_argument("--migraphx-nms-cache-dir", default="", help="Directory with per-resolution heatmap_nms_head_<H>x<W>.mxr files.")
    parser.add_argument("--compile-migraphx-nms-cache", action="store_true", help="Compile all required COCO per-resolution MIGraphX NMS heads before evaluation.")
    parser.add_argument("--force-compile-migraphx-nms", action="store_true", help="Recompile MIGraphX NMS heads even if MXR files already exist.")
    parser.add_argument("--keep-migraphx-nms-onnx", action="store_true", help="Keep ONNX files generated while compiling MIGraphX NMS heads.")
    parser.add_argument("--exhaustive-tune-migraphx-nms", action="store_true", help="Pass exhaustive_tune=True when compiling MIGraphX NMS heads.")
    parser.add_argument("--two-process-queue-size", type=int, default=2)
    parser.add_argument("--two-process-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.single_model:
        args.models = [args.single_model]
    return args


if __name__ == "__main__":
    validate_accuracy(parse_args())
