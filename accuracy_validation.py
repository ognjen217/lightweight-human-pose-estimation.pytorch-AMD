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
        "pruned_cpu_tail_ms", "post_avg_ms", "post_p50_ms", "post_p95_ms", "e2e_avg_ms", "e2e_p95_ms", "e2e_fps",
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


def validate_accuracy(args) -> List[Dict[str, Any]]:
    ensure_dir(args.output_dir)
    variants = [normalize_mode(normalize_validation_variant_name(v)) for v in args.variants]
    variants = list(dict.fromkeys(variants))
    _assert_accuracy_variants_are_migraphx_safe(variants)

    selected_images, selection_manifest = select_coco_images(args)
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
    print_variant_descriptions(variants)

    for model_path in args.models:
        model = load_migraphx_model(model_path, args.onnx, args.quantization, args.exhaustive_tune)
        model_name = Path(model_path).name
        model_dir = Path(args.output_dir) / Path(model_path).stem
        model_dir.mkdir(parents=True, exist_ok=True)
        meters = {variant: VariantMeter(model=model_name, variant=variant) for variant in variants}

        print(f"\n--- Evaluating model: {model_name} ---")
        for processed, info in enumerate(selected_images):
            file_name = info["file_name"]
            image_id = int(info["id"])
            if processed == 0 or (args.progress_every > 0 and processed % args.progress_every == 0):
                print(f"  image {processed + 1}/{target_count}: {file_name} ({info.get('height')}x{info.get('width')})")
            img = cv2.imread(os.path.join(args.images_folder, file_name), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read COCO image: {file_name}")

            heatmaps, pafs, original_hw, infer_timings = run_model_on_image(model, img, args)
            ensure_shape_postprocess_heads(args, variants, heatmaps.shape[:2], original_hw, compiled=compiled_shapes)

            for variant in variants:
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
    parser.add_argument("--num-of-test-img", type=int, default=None, help="Number of selected COCO images. Overrides --max-images when set.")
    parser.add_argument("--image-selection", choices=["sequential", "dominant-dimensions"], default="sequential")
    parser.add_argument("--skip-images", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--power-every", type=int, default=10)
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
    parser.add_argument("--limb-topm", type=int, default=20)
    parser.add_argument("--min-pair-score", type=float, default=0.0)
    parser.add_argument("--paf-cubic-a", type=float, default=-0.75)
    args = parser.parse_args()
    if args.single_model:
        args.models = [args.single_model]
    return args


if __name__ == "__main__":
    validate_accuracy(parse_args())
