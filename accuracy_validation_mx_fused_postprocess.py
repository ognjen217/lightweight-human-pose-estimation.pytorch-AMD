#!/usr/bin/env python3
"""
COCO accuracy validation for fused postprocess MXR with shape-aware sampling.

Compares:
  1) optimized_batch_k20_fast
  2) mx_fused_cubic_topk_fullres_paf_k20

Before inference, this script analyzes COCO image dimensions and inferred
postprocess shape keys:

    (cropped_lowres_h, cropped_lowres_w, original_h, original_w)

It selects the most frequent shape groups until they cover --max-images, then
randomly samples images from those groups.  This limits the number of compiled
fixed-shape fused .mxr heads while still giving a useful accuracy/recall signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import migraphx
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.coco import CocoValDataset
from modules.migraphx_fused_postprocess import MIGraphXFusedPostprocess
from modules.migraphx_fused_postprocess_compiler import (
    compile_fused_postprocess_head,
    fused_head_name,
)
from modules.mx_pair_assembly import topk_to_keypoint_lists, group_keypoints_from_mx_pair_scores
from modules.postprocessing import PostprocessConfig, postprocess_from_maps

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
        padded = cv2.copyMakeBorder(
            img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value
        )
        return padded, pad


COCO_KPT_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


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


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def shape_key_to_str(key: Tuple[int, int, int, int]) -> str:
    in_h, in_w, full_h, full_w = key
    return f"in{in_h}x{in_w}_full{full_h}x{full_w}"


def parse_shape_key(s: str) -> Tuple[int, int, int, int]:
    a, b = s.split("_full")
    in_h, in_w = [int(x) for x in a.replace("in", "").split("x")]
    full_h, full_w = [int(x) for x in b.split("x")]
    return in_h, in_w, full_h, full_w


def estimate_cropped_lowres_shape_from_hw(*, orig_h: int, orig_w: int, base_height: int, base_width: int, stride: int) -> Tuple[int, int]:
    ratio = min(float(base_height) / float(orig_h), float(base_width) / float(orig_w))
    scaled_h = max(1, int(round(orig_h * ratio)))
    scaled_w = max(1, int(round(orig_w * ratio)))

    padded_h = int(math.ceil(base_height / float(stride)) * stride)
    padded_w = int(max(base_width, scaled_w))
    padded_w = int(math.ceil(padded_w / float(stride)) * stride)

    top = int(math.floor((padded_h - scaled_h) / 2.0))
    bottom = int(padded_h - scaled_h - top)
    left = int(math.floor((padded_w - scaled_w) / 2.0))
    right = int(padded_w - scaled_w - left)

    raw_h = int(padded_h // stride)
    raw_w = int(padded_w // stride)
    crop_h = raw_h - int(top // stride) - int(bottom // stride)
    crop_w = raw_w - int(left // stride) - int(right // stride)
    return int(crop_h), int(crop_w)


def analyze_coco_shapes(args) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int, int, int], List[Dict[str, Any]]]]:
    coco = COCO(args.labels)
    img_ids = sorted(coco.getImgIds())
    imgs = coco.loadImgs(img_ids)

    by_shape: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for img in imgs:
        full_w = int(img["width"])
        full_h = int(img["height"])
        in_h, in_w = estimate_cropped_lowres_shape_from_hw(
            orig_h=full_h,
            orig_w=full_w,
            base_height=args.base_height,
            base_width=args.base_width,
            stride=args.stride,
        )
        key = (in_h, in_w, full_h, full_w)
        item = {
            "image_id": int(img["id"]),
            "file_name": img["file_name"],
            "full_h": full_h,
            "full_w": full_w,
            "in_h": in_h,
            "in_w": in_w,
            "shape_key": shape_key_to_str(key),
        }
        by_shape[key].append(item)

    rows = []
    for key, items in by_shape.items():
        in_h, in_w, full_h, full_w = key
        rows.append({
            "shape_key": shape_key_to_str(key),
            "count": len(items),
            "in_h": in_h,
            "in_w": in_w,
            "full_h": full_h,
            "full_w": full_w,
            "example_file": items[0]["file_name"],
        })
    rows.sort(key=lambda r: (-int(r["count"]), r["shape_key"]))
    return rows, by_shape


def select_shape_aware_subset(*, shape_rows: List[Dict[str, Any]], by_shape: Dict[Tuple[int, int, int, int], List[Dict[str, Any]]], max_images: int, seed: int, selection_mode: str, max_shapes: Optional[int] = None):
    rng = random.Random(int(seed))
    selected_shape_rows = []
    covered = 0
    for row in shape_rows:
        if max_shapes is not None and len(selected_shape_rows) >= int(max_shapes):
            break
        selected_shape_rows.append(row)
        covered += int(row["count"])
        if covered >= int(max_images):
            break

    selected_keys = [parse_shape_key(r["shape_key"]) for r in selected_shape_rows]
    pool: List[Dict[str, Any]] = []
    for key in selected_keys:
        pool.extend(by_shape[key])

    if selection_mode == "all-selected-dims":
        selected = list(pool)
    elif selection_mode == "top-dims-random":
        selected = list(pool) if len(pool) <= int(max_images) else rng.sample(pool, int(max_images))
    elif selection_mode == "top-dims-first":
        selected = []
        for key in selected_keys:
            selected.extend(by_shape[key])
            if len(selected) >= int(max_images):
                selected = selected[: int(max_images)]
                break
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    selected.sort(key=lambda x: int(x["image_id"]))

    actual_keys = sorted(
        {(int(x["in_h"]), int(x["in_w"]), int(x["full_h"]), int(x["full_w"])) for x in selected},
        key=lambda k: (-len(by_shape[k]), shape_key_to_str(k)),
    )
    actual_shape_rows = []
    for k in actual_keys:
        actual_shape_rows.append({
            "shape_key": shape_key_to_str(k),
            "count_in_full_coco": len(by_shape[k]),
            "selected_count": sum(1 for x in selected if (x["in_h"], x["in_w"], x["full_h"], x["full_w"]) == k),
            "in_h": k[0],
            "in_w": k[1],
            "full_h": k[2],
            "full_w": k[3],
            "example_file": by_shape[k][0]["file_name"],
        })
    return selected, actual_keys, actual_shape_rows


def write_dicts_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    extra = sorted({k for row in rows for k in row.keys()} - set(keys))
    keys += extra
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_selection_reports(args, shape_rows, selected_images, selected_shape_rows) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_dicts_csv(out / "shape_counts.csv", shape_rows)
    (out / "shape_counts.json").write_text(json.dumps(shape_rows, indent=2))
    write_dicts_csv(out / "selected_images.csv", selected_images)
    (out / "selected_images.json").write_text(json.dumps(selected_images, indent=2))
    write_dicts_csv(out / "selected_shapes.csv", selected_shape_rows)
    (out / "selected_shapes.json").write_text(json.dumps(selected_shape_rows, indent=2))


def print_shape_plan(shape_rows, selected_images, selected_shape_rows, args) -> None:
    print("\n" + "=" * 150)
    print("COCO SHAPE ANALYSIS / SELECTION PLAN")
    print("=" * 150)
    print(f"Total unique postprocess shapes: {len(shape_rows)}")
    print(f"Requested max images:           {args.max_images}")
    print(f"Selection mode:                 {args.selection_mode}")
    print(f"Selected images:                {len(selected_images)}")
    print(f"Selected unique shapes:         {len(selected_shape_rows)}")
    print("\nTop 15 shapes in full COCO:")
    print(f"{'rank':>4} {'count':>6} {'shape':<32} {'example'}")
    for i, row in enumerate(shape_rows[:15], start=1):
        print(f"{i:>4} {int(row['count']):>6} {row['shape_key']:<32} {row['example_file']}")
    print("\nShapes to compile/use:")
    print(f"{'rank':>4} {'full_count':>10} {'selected':>8} {'shape':<32} {'example'}")
    for i, row in enumerate(selected_shape_rows, start=1):
        print(f"{i:>4} {int(row['count_in_full_coco']):>10} {int(row['selected_count']):>8} {row['shape_key']:<32} {row['example_file']}")
    print("=" * 150)


def load_migraphx_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Compiled model not found: {model_path}")
    print(f"--- Loading compiled model: {model_path} ---")
    return migraphx.load(model_path)


def cast_input_for_model(model, tensor_nchw: np.ndarray) -> np.ndarray:
    expected_type = str(model.get_parameter_shapes()["input"].type())
    if "half" in expected_type:
        tensor_nchw = tensor_nchw.astype(np.float16)
    else:
        tensor_nchw = tensor_nchw.astype(np.float32)
    return np.ascontiguousarray(tensor_nchw)


def prepare_coco_input(img: np.ndarray, *, base_height: int, base_width: int, stride: int, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
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


def run_model_on_image(model, img: np.ndarray, args):
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
        pad = meta["pad"]
        top, left, bottom, right = [p // args.stride for p in pad]
        h_end = heatmaps.shape[0] - bottom if bottom > 0 else heatmaps.shape[0]
        w_end = heatmaps.shape[1] - right if right > 0 else heatmaps.shape[1]
        heatmaps = np.ascontiguousarray(heatmaps[top:h_end, left:w_end, :], dtype=np.float32)
        pafs = np.ascontiguousarray(pafs[top:h_end, left:w_end, :], dtype=np.float32)
    timings["decode_ms"] = t.ms
    return heatmaps, pafs, (meta["orig_h"], meta["orig_w"]), timings


def coco_eval_stats(gt_file_path: str, dt_file_path: str, image_ids: Sequence[int]) -> Dict[str, float]:
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, "keypoints")
    result.params.imgIds = sorted([int(x) for x in image_ids])
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


@dataclass
class VariantMeter:
    model: str
    variant: str
    detections: List[dict] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    fused_post_mx_ms: List[float] = field(default_factory=list)
    topk_adapter_ms: List[float] = field(default_factory=list)
    mx_pair_assembly_ms: List[float] = field(default_factory=list)
    post_ms: List[float] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)

    def add_timing(self, pre: float, infer: float, decode: float, post: float, timings: Dict[str, Any]) -> None:
        self.preprocess_ms.append(float(pre))
        self.inference_ms.append(float(infer))
        self.decode_ms.append(float(decode))
        self.post_ms.append(float(post))
        self.e2e_ms.append(float(pre) + float(infer) + float(decode) + float(post))
        self.fused_post_mx_ms.append(float(timings.get("fused_post_mx", 0.0) or 0.0))
        self.topk_adapter_ms.append(float(timings.get("topk_adapter", 0.0) or 0.0))
        self.mx_pair_assembly_ms.append(float(timings.get("mx_assembly_total", 0.0) or 0.0))

    def latency_summary(self) -> Dict[str, float]:
        post_avg = mean(self.post_ms)
        e2e_avg = mean(self.e2e_ms)
        return {
            "images": len(self.post_ms),
            "preprocess_ms": mean(self.preprocess_ms),
            "inference_ms": mean(self.inference_ms),
            "decode_ms": mean(self.decode_ms),
            "fused_post_mx_ms": mean(self.fused_post_mx_ms),
            "topk_adapter_ms": mean(self.topk_adapter_ms),
            "mx_pair_assembly_ms": mean(self.mx_pair_assembly_ms),
            "post_avg_ms": post_avg,
            "post_p50_ms": percentile(self.post_ms, 50),
            "post_p95_ms": percentile(self.post_ms, 95),
            "post_fps": 1000.0 / post_avg if post_avg > 0 else 0.0,
            "e2e_avg_ms": e2e_avg,
            "e2e_p95_ms": percentile(self.e2e_ms, 95),
            "e2e_fps": 1000.0 / e2e_avg if e2e_avg > 0 else 0.0,
        }


_FUSED_CACHE: Dict[str, MIGraphXFusedPostprocess] = {}


def fused_mxr_path_for_shape(key: Tuple[int, int, int, int], args) -> Path:
    in_h, in_w, full_h, full_w = key
    name = fused_head_name(
        in_h, in_w, full_h, full_w,
        topk=args.topk,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        heatmap_cubic_a=args.cubic_a,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
    )
    return Path(args.fused_cache_dir) / f"{name}.mxr"


def compile_shape_heads(shape_keys: Sequence[Tuple[int, int, int, int]], args) -> None:
    if not args.compile_heads:
        missing = [str(fused_mxr_path_for_shape(k, args)) for k in shape_keys if not fused_mxr_path_for_shape(k, args).exists()]
        if missing:
            raise FileNotFoundError("Missing fused .mxr files and --compile-heads was not specified.\n" + "\n".join(missing[:20]))
        return
    print("\nPrecompiling fused heads for selected shapes")
    print("-" * 120)
    for i, key in enumerate(shape_keys, start=1):
        in_h, in_w, full_h, full_w = key
        print(f"[{i}/{len(shape_keys)}] {shape_key_to_str(key)}")
        compile_fused_postprocess_head(
            in_h=in_h,
            in_w=in_w,
            full_h=full_h,
            full_w=full_w,
            output_dir=args.fused_cache_dir,
            parts_dir=args.fused_parts_dir,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            heatmap_cubic_a=args.cubic_a,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            paf_cubic_a=args.paf_cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force_compile_heads,
            keep_onnx=True,
        )
    print("-" * 120)


def get_fused_head_for_runtime(heatmaps: np.ndarray, original_hw: Tuple[int, int], args) -> MIGraphXFusedPostprocess:
    key = (int(heatmaps.shape[0]), int(heatmaps.shape[1]), int(original_hw[0]), int(original_hw[1]))
    path = fused_mxr_path_for_shape(key, args)
    if not path.exists():
        if not args.compile_missing_at_runtime:
            raise FileNotFoundError(f"Missing fused .mxr for runtime shape {shape_key_to_str(key)}: {path}")
        compile_fused_postprocess_head(
            in_h=key[0], in_w=key[1], full_h=key[2], full_w=key[3],
            output_dir=args.fused_cache_dir,
            parts_dir=args.fused_parts_dir,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            heatmap_cubic_a=args.cubic_a,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            paf_cubic_a=args.paf_cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=False,
            keep_onnx=True,
        )
    k = str(path)
    if k not in _FUSED_CACHE:
        _FUSED_CACHE[k] = MIGraphXFusedPostprocess(k)
    return _FUSED_CACHE[k]


def postprocess_fused(heatmaps: np.ndarray, pafs: np.ndarray, original_hw: Tuple[int, int], args):
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    full_h, full_w = int(original_hw[0]), int(original_hw[1])
    fused = get_fused_head_for_runtime(heatmaps, original_hw, args)
    heatmaps_nchw = np.moveaxis(np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float32), -1, 0)[np.newaxis, :]
    pafs_nchw = np.moveaxis(np.ascontiguousarray(pafs, dtype=np.float32), -1, 0)[np.newaxis, :]
    with Timer() as t:
        pair_scores, pair_valid, top_scores, top_indices = fused.run(heatmaps_nchw, pafs_nchw)
    timings["fused_post_mx"] = t.ms
    with Timer() as t:
        all_kpts_by_type, _ = topk_to_keypoint_lists(top_scores, top_indices, full_width=full_w, threshold=args.threshold, num_keypoint_types=18)
    timings["topk_adapter"] = t.ms
    poses, all_kpts, asm_times = group_keypoints_from_mx_pair_scores(all_kpts_by_type, pair_scores, pair_valid, min_pair_score=args.min_pair_score, return_timing=True)
    timings.update(asm_times)
    timings["total_postprocess"] = (time.perf_counter() - t_total) * 1000.0
    return np.asarray(poses, dtype=np.float32), np.asarray(all_kpts, dtype=np.float32), timings


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 180)
    print("FUSED POSTPROCESS ACCURACY SUMMARY")
    print("=" * 180)
    print(f"{'variant':<42} {'img':>5} {'AP':>6} {'AP50':>6} {'AP75':>6} {'AR':>6} {'fused':>8} {'adapt':>8} {'asm':>8} {'post':>8} {'e2e':>8} {'fps':>8} {'speedup':>8}")
    print("-" * 180)
    for r in rows:
        print(f"{r['variant']:<42} {int(r['images']):>5} {float(r.get('AP', 0.0)):>6.3f} {float(r.get('AP50', 0.0)):>6.3f} {float(r.get('AP75', 0.0)):>6.3f} {float(r.get('AR', 0.0)):>6.3f} {float(r.get('fused_post_mx_ms', 0.0)):>8.2f} {float(r.get('topk_adapter_ms', 0.0)):>8.2f} {float(r.get('mx_pair_assembly_ms', 0.0)):>8.2f} {float(r.get('post_avg_ms', 0.0)):>8.2f} {float(r.get('e2e_avg_ms', 0.0)):>8.2f} {float(r.get('e2e_fps', 0.0)):>8.2f} {float(r.get('e2e_speedup_vs_baseline', 1.0)):>8.2f}")
    print("=" * 180)


def run(args) -> List[Dict[str, Any]]:
    ensure_dir(args.output_dir)
    ensure_dir(args.fused_cache_dir)
    shape_rows, by_shape = analyze_coco_shapes(args)
    selected_images, selected_shape_keys, selected_shape_rows = select_shape_aware_subset(
        shape_rows=shape_rows,
        by_shape=by_shape,
        max_images=args.max_images,
        seed=args.seed,
        selection_mode=args.selection_mode,
        max_shapes=args.max_shapes,
    )
    save_selection_reports(args, shape_rows, selected_images, selected_shape_rows)
    print_shape_plan(shape_rows, selected_images, selected_shape_rows, args)
    if args.analyze_only:
        print("\nAnalyze-only mode enabled. No model inference and no COCO eval executed.")
        return []
    compile_shape_heads(selected_shape_keys, args)
    model = load_migraphx_model(args.model)
    model_name = Path(args.model).name
    model_dir = Path(args.output_dir) / Path(args.model).stem
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset = CocoValDataset(args.labels, args.images_folder)
    selected_by_file = {x["file_name"]: x for x in selected_images}
    selected_image_ids = [int(x["image_id"]) for x in selected_images]
    baseline_name = "optimized_batch_k20_fast"
    fused_name = f"mx_fused_cubic_topk_fullres_paf_k{args.topk}"
    meters = {
        baseline_name: VariantMeter(model=model_name, variant=baseline_name),
        fused_name: VariantMeter(model=model_name, variant=fused_name),
    }
    config = PostprocessConfig(max_keypoints_per_type=args.topk, threshold=args.threshold, nms_radius_fullres=args.nms_radius)
    print("\nRunning COCO accuracy validation for fused postprocess")
    print(f"Model:             {args.model}")
    print(f"Selected images:   {len(selected_images)}")
    print(f"Selected shapes:   {len(selected_shape_keys)}")
    print(f"Fused cache:       {args.fused_cache_dir}")
    processed = 0
    selected_files = set(selected_by_file.keys())
    for sample in dataset:
        file_name = sample["file_name"]
        if file_name not in selected_files:
            continue
        img = sample["img"]
        image_id = int(file_name[0:file_name.rfind(".")])
        if processed == 0 or (args.progress_every > 0 and processed % args.progress_every == 0):
            meta = selected_by_file[file_name]
            print(f"  image {processed + 1}/{len(selected_images)}: {file_name} shape={meta['shape_key']}")
        heatmaps, pafs, original_hw, infer_timings = run_model_on_image(model, img, args)
        out_base = postprocess_from_maps(baseline_name, heatmaps, pafs, original_hw, config=config)
        meters[baseline_name].add_timing(pre=infer_timings["preprocess_ms"], infer=infer_timings["inference_ms"], decode=infer_timings["decode_ms"], post=out_base.timings.get("total_postprocess", 0.0), timings=out_base.timings)
        meters[baseline_name].detections.extend(build_coco_detections(image_id, out_base.pose_entries, out_base.all_keypoints))
        poses, kpts, timings = postprocess_fused(heatmaps, pafs, original_hw, args)
        meters[fused_name].add_timing(pre=infer_timings["preprocess_ms"], infer=infer_timings["inference_ms"], decode=infer_timings["decode_ms"], post=timings.get("total_postprocess", 0.0), timings=timings)
        meters[fused_name].detections.extend(build_coco_detections(image_id, poses, kpts))
        processed += 1
    if processed == 0:
        raise RuntimeError("No selected images were processed. Check dataset file names and selected_images report.")
    all_rows: List[Dict[str, Any]] = []
    baseline_e2e = None
    for variant, meter in meters.items():
        detections_path = model_dir / f"detections_{variant}.json"
        with detections_path.open("w") as f:
            json.dump(meter.detections, f)
        print(f"\n--- COCO eval: variant={variant} on selected image subset ---")
        metrics = coco_eval_stats(args.labels, str(detections_path), selected_image_ids)
        latency = meter.latency_summary()
        row = {"model": model_name, "model_path": args.model, "variant": variant, "images": processed, "selected_unique_shapes": len(selected_shape_keys), **latency, **metrics, "detections_path": str(detections_path)}
        all_rows.append(row)
        if variant == baseline_name:
            baseline_e2e = float(row.get("e2e_avg_ms", 0.0) or 0.0)
    if baseline_e2e and baseline_e2e > 0:
        for row in all_rows:
            e2e = float(row.get("e2e_avg_ms", 0.0) or 0.0)
            if e2e > 0:
                row["e2e_speedup_vs_baseline"] = baseline_e2e / e2e
                row["e2e_delta_pct_vs_baseline"] = ((e2e - baseline_e2e) / baseline_e2e) * 100.0
            else:
                row["e2e_speedup_vs_baseline"] = 1.0
                row["e2e_delta_pct_vs_baseline"] = 0.0
    summary_json = Path(args.output_dir) / "accuracy_mx_fused_postprocess_summary.json"
    summary_csv = Path(args.output_dir) / "accuracy_mx_fused_postprocess_summary.csv"
    with summary_json.open("w") as f:
        json.dump(all_rows, f, indent=2)
    write_dicts_csv(summary_csv, all_rows)
    print_summary_table(all_rows)
    print(f"\nSaved JSON: {summary_json}")
    print(f"Saved CSV:  {summary_csv}")
    return all_rows


def parse_args():
    p = argparse.ArgumentParser(description="COCO AP/AR validation for fused postprocess with shape-aware sampling.")
    p.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    p.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    p.add_argument("--images-folder", default="coco/val2017/")
    p.add_argument("--output-dir", default="outputs/accuracy_mx_fused_postprocess")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--base-height", type=int, default=544)
    p.add_argument("--base-width", type=int, default=968)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--min-pair-score", type=float, default=0.0)
    p.add_argument("--fused-cache-dir", default="models/fused_postprocess_coco_cache")
    p.add_argument("--fused-parts-dir", default="")
    p.add_argument("--compile-heads", action="store_true", help="Compile selected fused .mxr heads before running accuracy.")
    p.add_argument("--compile-missing-at-runtime", action="store_true", help="Compile a head if runtime shape was not planned.")
    p.add_argument("--force-compile-heads", action="store_true")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--selection-mode", choices=["top-dims-random", "top-dims-first", "all-selected-dims"], default="top-dims-random")
    p.add_argument("--max-shapes", type=int, default=None, help="Optional cap on how many shape groups to select.")
    p.add_argument("--analyze-only", action="store_true", help="Only write shape/selection reports; do not compile or run inference.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
