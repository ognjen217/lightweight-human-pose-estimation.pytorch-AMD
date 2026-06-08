#!/usr/bin/env python3
"""
COCO AP/AR validation for:

  optimized_batch_k20_fast
    Existing optimized CPU postprocess baseline.

  mx_cubic_topk_paf_scorer_k20
    Experimental path:
      - manual cubic heatmap resize + NMS + TopK in MIGraphX
      - PAF sampling + limb pair scoring in MIGraphX
      - CPU connection NMS + pose assembly only

This script depends on these experimental modules:
  modules/migraphx_manual_cubic_topk_compiler.py
  modules/migraphx_paf_pair_scorer.py
  modules/migraphx_paf_pair_scorer_compiler.py
  modules/mx_pair_assembly.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import migraphx
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.coco import CocoValDataset
from modules.migraphx_manual_cubic_topk_compiler import (
    compile_manual_cubic_nms_topk_head,
    head_name as manual_cubic_head_name,
)
from modules.migraphx_paf_pair_scorer import MIGraphXPAFPairScorer
from modules.migraphx_paf_pair_scorer_compiler import (
    compile_paf_pair_scorer_head,
    head_name as paf_scorer_head_name,
)
from modules.mx_pair_assembly import (
    topk_to_keypoint_lists,
    group_keypoints_from_mx_pair_scores,
)
from modules.postprocessing import (
    PostprocessConfig,
    postprocess_from_maps,
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
        padded = cv2.copyMakeBorder(
            img,
            pad[0],
            pad[2],
            pad[1],
            pad[3],
            cv2.BORDER_CONSTANT,
            value=pad_value,
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


class MXRRunner:
    """Small wrapper for the manual-cubic TopK MXR head."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.program = migraphx.load(self.path)

    def run(self, inputs: Dict[str, np.ndarray]):
        clean = {k: np.ascontiguousarray(v) for k, v in inputs.items()}
        out = self.program.run(clean)
        if not isinstance(out, (list, tuple)):
            out = list(out)
        return out


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


@dataclass
class VariantMeter:
    model: str
    variant: str
    detections: List[dict] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    manual_cubic_mx_topk_ms: List[float] = field(default_factory=list)
    paf_pair_scorer_mx_ms: List[float] = field(default_factory=list)
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

        self.manual_cubic_mx_topk_ms.append(float(timings.get("manual_cubic_mx_topk", 0.0) or 0.0))
        self.paf_pair_scorer_mx_ms.append(float(timings.get("paf_pair_scorer_mx", 0.0) or 0.0))
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
            "manual_cubic_mx_topk_ms": mean(self.manual_cubic_mx_topk_ms),
            "paf_pair_scorer_mx_ms": mean(self.paf_pair_scorer_mx_ms),
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


_MANUAL_HEAD_CACHE: Dict[str, MXRRunner] = {}
_PAF_SCORER_CACHE: Dict[str, MIGraphXPAFPairScorer] = {}


def resolve_manual_head_path(
    *,
    heatmaps_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
    args,
) -> Path:
    in_h, in_w = int(heatmaps_hw[0]), int(heatmaps_hw[1])
    out_h, out_w = int(original_hw[0]), int(original_hw[1])
    name = manual_cubic_head_name(
        in_h,
        in_w,
        out_h,
        out_w,
        topk=args.topk,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        cubic_a=args.cubic_a,
    )
    path = Path(args.manual_cubic_cache_dir) / f"{name}.mxr"
    if path.exists():
        return path

    if not args.compile_heads:
        raise FileNotFoundError(
            f"Missing manual cubic TopK head: {path}\n"
            "Run with --compile-heads or precompile required heads."
        )

    return compile_manual_cubic_nms_topk_head(
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        output_dir=args.manual_cubic_cache_dir,
        channels=18,
        topk=args.topk,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        cubic_a=args.cubic_a,
        opset=args.opset,
        exhaustive_tune=args.exhaustive_tune,
        force=args.force_compile_heads,
        keep_onnx=args.keep_onnx,
    )


def resolve_paf_scorer_path(
    *,
    paf_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
    args,
) -> Path:
    paf_h, paf_w = int(paf_hw[0]), int(paf_hw[1])
    full_h, full_w = int(original_hw[0]), int(original_hw[1])

    name = paf_scorer_head_name(
        paf_h,
        paf_w,
        full_h,
        full_w,
        topk=args.topk,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
    )
    path = Path(args.paf_scorer_cache_dir) / f"{name}.mxr"
    if path.exists():
        return path

    if not args.compile_heads:
        raise FileNotFoundError(
            f"Missing PAF pair scorer head: {path}\n"
            "Run with --compile-heads or precompile required heads."
        )

    return compile_paf_pair_scorer_head(
        paf_h=paf_h,
        paf_w=paf_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=args.paf_scorer_cache_dir,
        topk=args.topk,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        opset=args.opset,
        exhaustive_tune=args.exhaustive_tune,
        force=args.force_compile_heads,
        keep_onnx=args.keep_onnx,
    )


def get_manual_head(path: Path) -> MXRRunner:
    key = str(path)
    if key not in _MANUAL_HEAD_CACHE:
        _MANUAL_HEAD_CACHE[key] = MXRRunner(key)
    return _MANUAL_HEAD_CACHE[key]


def get_paf_scorer(path: Path) -> MIGraphXPAFPairScorer:
    key = str(path)
    if key not in _PAF_SCORER_CACHE:
        _PAF_SCORER_CACHE[key] = MIGraphXPAFPairScorer(key)
    return _PAF_SCORER_CACHE[key]


def postprocess_mx_paf_pair_scorer(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()
    full_h, full_w = int(original_hw[0]), int(original_hw[1])

    manual_path = resolve_manual_head_path(
        heatmaps_hw=heatmaps.shape[:2],
        original_hw=original_hw,
        args=args,
    )
    paf_path = resolve_paf_scorer_path(
        paf_hw=pafs.shape[:2],
        original_hw=original_hw,
        args=args,
    )

    manual_head = get_manual_head(manual_path)
    paf_scorer = get_paf_scorer(paf_path)

    heatmaps_nchw = np.moveaxis(np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float32), -1, 0)[np.newaxis, :]
    with Timer() as t:
        top_scores, top_indices = manual_head.run({"heatmaps": heatmaps_nchw})
    timings["manual_cubic_mx_topk"] = t.ms
    top_scores = np.asarray(top_scores, dtype=np.float32)
    top_indices = np.asarray(top_indices, dtype=np.float32)

    pafs_nchw = np.moveaxis(np.ascontiguousarray(pafs, dtype=np.float32), -1, 0)[np.newaxis, :]
    with Timer() as t:
        pair_scores, pair_valid = paf_scorer.run(top_scores, top_indices, pafs_nchw)
    timings["paf_pair_scorer_mx"] = t.ms

    with Timer() as t:
        all_kpts_by_type, _ = topk_to_keypoint_lists(
            top_scores,
            top_indices,
            full_width=full_w,
            threshold=args.threshold,
            num_keypoint_types=18,
        )
    timings["topk_adapter"] = t.ms

    poses, all_kpts, asm_times = group_keypoints_from_mx_pair_scores(
        all_kpts_by_type,
        pair_scores,
        pair_valid,
        min_pair_score=args.min_pair_score,
        return_timing=True,
    )
    timings.update(asm_times)
    timings["total_postprocess"] = (time.perf_counter() - t_total) * 1000.0
    return np.asarray(poses, dtype=np.float32), np.asarray(all_kpts, dtype=np.float32), timings


def write_flat_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "model", "variant", "precision", "refinement_stages", "images",
        "AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl",
        "preprocess_ms", "inference_ms", "decode_ms",
        "manual_cubic_mx_topk_ms", "paf_pair_scorer_mx_ms",
        "topk_adapter_ms", "mx_pair_assembly_ms",
        "post_avg_ms", "post_p50_ms", "post_p95_ms", "post_fps",
        "e2e_avg_ms", "e2e_p95_ms", "e2e_fps",
        "e2e_speedup_vs_baseline", "e2e_delta_pct_vs_baseline",
        "detections_path",
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = [k for k in preferred if k in all_keys]
    fieldnames.extend(sorted(k for k in all_keys if k not in fieldnames))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 190)
    print("MX CUBIC TOPK + MX PAF PAIR SCORER ACCURACY SUMMARY")
    print("=" * 190)
    print(
        f"{'variant':<42} {'img':>5} "
        f"{'AP':>6} {'AP50':>6} {'AP75':>6} {'AR':>6} "
        f"{'topk':>8} {'paf_mx':>8} {'adapt':>8} {'asm':>8} "
        f"{'post':>8} {'e2e':>8} {'fps':>8} {'speedup':>8}"
    )
    print("-" * 190)
    for r in rows:
        print(
            f"{r['variant']:<42} {int(r['images']):>5} "
            f"{float(r.get('AP', 0.0)):>6.3f} {float(r.get('AP50', 0.0)):>6.3f} "
            f"{float(r.get('AP75', 0.0)):>6.3f} {float(r.get('AR', 0.0)):>6.3f} "
            f"{float(r.get('manual_cubic_mx_topk_ms', 0.0)):>8.2f} "
            f"{float(r.get('paf_pair_scorer_mx_ms', 0.0)):>8.2f} "
            f"{float(r.get('topk_adapter_ms', 0.0)):>8.2f} "
            f"{float(r.get('mx_pair_assembly_ms', 0.0)):>8.2f} "
            f"{float(r.get('post_avg_ms', 0.0)):>8.2f} "
            f"{float(r.get('e2e_avg_ms', 0.0)):>8.2f} "
            f"{float(r.get('e2e_fps', 0.0)):>8.2f} "
            f"{float(r.get('e2e_speedup_vs_baseline', 1.0)):>8.2f}"
        )
    print("=" * 190)


def run(args) -> List[Dict[str, Any]]:
    ensure_dir(args.output_dir)
    ensure_dir(args.manual_cubic_cache_dir)
    ensure_dir(args.paf_scorer_cache_dir)

    model = load_migraphx_model(args.model)
    model_name = Path(args.model).name
    model_dir = Path(args.output_dir) / Path(args.model).stem
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset = CocoValDataset(args.labels, args.images_folder)
    target_count = args.max_images if args.max_images is not None else len(dataset)

    baseline_name = "optimized_batch_k20_fast"
    mx_name = f"mx_cubic_topk_paf_scorer_k{args.topk}"
    meters = {
        baseline_name: VariantMeter(model=model_name, variant=baseline_name),
        mx_name: VariantMeter(model=model_name, variant=mx_name),
    }

    config = PostprocessConfig(
        max_keypoints_per_type=args.topk,
        threshold=args.threshold,
        nms_radius_fullres=args.nms_radius,
    )

    print("\nRunning COCO accuracy validation for MX cubic TopK + MX PAF pair scorer")
    print(f"Model:             {args.model}")
    print(f"Labels:            {args.labels}")
    print(f"Images:            {args.images_folder}")
    print(f"Max images:        {args.max_images}")
    print(f"TopK:              {args.topk}")
    print(f"Threshold:         {args.threshold}")
    print(f"NMS radius:        {args.nms_radius}")
    print(f"NMS impl:          {args.nms_impl}")
    print(f"Cubic a:           {args.cubic_a}")
    print(f"Points per limb:   {args.points_per_limb}")
    print(f"Min PAF score:     {args.min_paf_score}")
    print(f"Success ratio thr: {args.success_ratio_thr}")
    print(f"Compile heads:     {args.compile_heads}")
    print(f"Manual cache:      {args.manual_cubic_cache_dir}")
    print(f"PAF scorer cache:  {args.paf_scorer_cache_dir}")

    processed = 0
    for i, sample in enumerate(dataset):
        if i < args.skip_images:
            continue
        if args.max_images is not None and processed >= args.max_images:
            break

        file_name = sample["file_name"]
        img = sample["img"]
        image_id = int(file_name[0:file_name.rfind(".")])

        if processed == 0 or (args.progress_every > 0 and processed % args.progress_every == 0):
            print(f"  image {processed + 1}/{target_count}: {file_name}")

        heatmaps, pafs, original_hw, infer_timings = run_model_on_image(model, img, args)

        # Baseline
        out_base = postprocess_from_maps(
            baseline_name,
            heatmaps,
            pafs,
            original_hw,
            config=config,
        )
        meters[baseline_name].add_timing(
            pre=infer_timings["preprocess_ms"],
            infer=infer_timings["inference_ms"],
            decode=infer_timings["decode_ms"],
            post=out_base.timings.get("total_postprocess", 0.0),
            timings=out_base.timings,
        )
        meters[baseline_name].detections.extend(
            build_coco_detections(image_id, out_base.pose_entries, out_base.all_keypoints)
        )

        # Experimental MXR PAF scorer path
        poses, kpts, timings = postprocess_mx_paf_pair_scorer(heatmaps, pafs, original_hw, args)
        meters[mx_name].add_timing(
            pre=infer_timings["preprocess_ms"],
            infer=infer_timings["inference_ms"],
            decode=infer_timings["decode_ms"],
            post=timings.get("total_postprocess", 0.0),
            timings=timings,
        )
        meters[mx_name].detections.extend(build_coco_detections(image_id, poses, kpts))

        processed += 1

    all_rows: List[Dict[str, Any]] = []
    baseline_e2e = None

    for variant, meter in meters.items():
        detections_path = model_dir / f"detections_{variant}.json"
        with detections_path.open("w") as f:
            json.dump(meter.detections, f)

        print(f"\n--- COCO eval: variant={variant} ---")
        metrics = coco_eval_stats(args.labels, str(detections_path))
        latency = meter.latency_summary()
        meta = parse_model_metadata(args.model)

        row = {
            "model": model_name,
            "model_path": args.model,
            "variant": variant,
            **meta,
            **latency,
            **metrics,
            "detections_path": str(detections_path),
        }
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

    summary_json = Path(args.output_dir) / "accuracy_mx_paf_pair_scorer_summary.json"
    summary_csv = Path(args.output_dir) / "accuracy_mx_paf_pair_scorer_summary.csv"
    with summary_json.open("w") as f:
        json.dump(all_rows, f, indent=2)
    write_flat_csv(summary_csv, all_rows)

    print_summary_table(all_rows)
    print(f"\nSaved JSON: {summary_json}")
    print(f"Saved CSV:  {summary_csv}")
    return all_rows


def parse_args():
    p = argparse.ArgumentParser(description="COCO AP/AR validation for MX cubic TopK + MX PAF pair scorer.")
    p.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    p.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    p.add_argument("--images-folder", default="coco/val2017/")
    p.add_argument("--output-dir", default="outputs/accuracy_mx_paf_pair_scorer")
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--skip-images", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=20)

    p.add_argument("--base-height", type=int, default=544)
    p.add_argument("--base-width", type=int, default=968)
    p.add_argument("--stride", type=int, default=8)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)

    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--min-pair-score", type=float, default=0.0)

    p.add_argument("--manual-cubic-cache-dir", default="models/manual_cubic_nms_topk_coco_cache")
    p.add_argument("--paf-scorer-cache-dir", default="models/paf_pair_scorer_coco_cache")
    p.add_argument("--compile-heads", action="store_true", help="Compile missing per-shape MXR heads on demand.")
    p.add_argument("--force-compile-heads", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
