#!/usr/bin/env python3
"""
Standalone COCO accuracy validation for experimental MIGraphX resize+NMS+TopK head.

This intentionally avoids modifying modules/postprocessing.py while we validate AP/AR.
It compares:
  - optimized_batch_k20_fast from the existing registry
  - migraphx_resize_nms_topk_k20: cropped low-res heatmaps -> MIGraphX full-res resize+NMS+TopK -> CPU PAF resize + group_keypoints_fast

Note: the TopK head has fixed input/output shape, so COCO evaluation compiles/loads one
head per unique (low_h, low_w, orig_h, orig_w) shape encountered.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from datasets.coco import CocoValDataset
from modules.keypoints import group_keypoints_fast
from modules.migraphx_resize_topk import MIGraphXResizeNMSTopKHead, topk_to_keypoint_lists
from modules.migraphx_resize_topk_compiler import compile_resize_nms_topk_head, head_name
from modules.postprocessing import PostprocessConfig, postprocess_from_maps

from accuracy_validation import (
    Timer,
    build_coco_detections,
    coco_eval_stats,
    load_migraphx_model,
    parse_model_metadata,
    run_model_on_image,
    write_flat_csv,
)


@dataclass
class Meter:
    variant: str
    detections: List[dict] = field(default_factory=list)
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    decode_ms: List[float] = field(default_factory=list)
    hm_resize_mx_topk_ms: List[float] = field(default_factory=list)
    paf_resize_ms: List[float] = field(default_factory=list)
    topk_adapter_ms: List[float] = field(default_factory=list)
    group_ms: List[float] = field(default_factory=list)
    post_ms: List[float] = field(default_factory=list)
    e2e_ms: List[float] = field(default_factory=list)

    def add(self, infer_timings: Dict[str, float], post_timings: Dict[str, float]) -> None:
        pre = float(infer_timings.get("preprocess_ms", 0.0) or 0.0)
        infer = float(infer_timings.get("inference_ms", 0.0) or 0.0)
        dec = float(infer_timings.get("decode_ms", 0.0) or 0.0)
        post = float(post_timings.get("total_postprocess", 0.0) or 0.0)
        self.preprocess_ms.append(pre)
        self.inference_ms.append(infer)
        self.decode_ms.append(dec)
        self.hm_resize_mx_topk_ms.append(float(post_timings.get("hm_resize_mx_topk", 0.0) or 0.0))
        self.paf_resize_ms.append(float(post_timings.get("resize_pafs", post_timings.get("paf_resize", 0.0)) or 0.0))
        self.topk_adapter_ms.append(float(post_timings.get("topk_adapter", 0.0) or 0.0))
        self.group_ms.append(float(post_timings.get("group_keypoints", post_timings.get("group_total", 0.0)) or 0.0))
        self.post_ms.append(post)
        self.e2e_ms.append(pre + infer + dec + post)

    def latency(self) -> Dict[str, float]:
        post_avg = mean(self.post_ms)
        e2e_avg = mean(self.e2e_ms)
        return {
            "images": len(self.post_ms),
            "preprocess_ms": mean(self.preprocess_ms),
            "inference_ms": mean(self.inference_ms),
            "decode_ms": mean(self.decode_ms),
            "hm_resize_mx_topk_ms": mean(self.hm_resize_mx_topk_ms),
            "paf_resize_ms": mean(self.paf_resize_ms),
            "topk_adapter_ms": mean(self.topk_adapter_ms),
            "group_ms": mean(self.group_ms),
            "post_avg_ms": post_avg,
            "post_p50_ms": percentile(self.post_ms, 50),
            "post_p95_ms": percentile(self.post_ms, 95),
            "post_fps": 1000.0 / post_avg if post_avg > 0 else 0.0,
            "e2e_avg_ms": e2e_avg,
            "e2e_p95_ms": percentile(self.e2e_ms, 95),
            "e2e_fps": 1000.0 / e2e_avg if e2e_avg > 0 else 0.0,
        }


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def postprocess_maps_resize_nms_topk(
    heatmaps: np.ndarray,
    pafs: np.ndarray,
    original_hw: Tuple[int, int],
    *,
    head: MIGraphXResizeNMSTopKHead,
    topk: int,
    threshold: float,
    points_per_limb: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    timings: Dict[str, float] = {}
    total_t0 = time.perf_counter()
    orig_h, orig_w = int(original_hw[0]), int(original_hw[1])

    heatmaps_nchw = np.moveaxis(np.ascontiguousarray(heatmaps[:, :, :18], dtype=np.float32), -1, 0)[np.newaxis, ...]

    with Timer() as t:
        scores, indices = head.run(heatmaps_nchw)
    timings["hm_resize_mx_topk"] = t.ms

    with Timer() as t:
        all_kpts, _ = topk_to_keypoint_lists(
            scores,
            indices,
            full_width=orig_w,
            threshold=threshold,
            num_keypoint_types=18,
        )
    timings["topk_adapter"] = t.ms

    with Timer() as t:
        pafs_full = cv2.resize(
            np.ascontiguousarray(pafs, dtype=np.float32),
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
        pafs_full = np.ascontiguousarray(pafs_full, dtype=np.float32)
    timings["resize_pafs"] = t.ms

    with Timer() as t:
        try:
            out = group_keypoints_fast(all_kpts, pafs_full, points_per_limb=points_per_limb, return_timing=True)
        except TypeError:
            out = group_keypoints_fast(all_kpts, pafs_full, points_per_limb=points_per_limb)
    timings["group_keypoints"] = t.ms

    if isinstance(out, tuple) and len(out) == 3:
        poses, kpts, group_times = out
        for key, value in group_times.items():
            try:
                timings[str(key)] = float(value)
            except Exception:
                pass
    else:
        poses, kpts = out

    timings["total_postprocess"] = (time.perf_counter() - total_t0) * 1000.0
    return np.asarray(poses, dtype=np.float32), np.asarray(kpts, dtype=np.float32), timings


class HeadCache:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cache: Dict[Tuple[int, int, int, int], MIGraphXResizeNMSTopKHead] = {}
        self.compiled = 0

    def get(self, heatmaps: np.ndarray, original_hw: Tuple[int, int]) -> MIGraphXResizeNMSTopKHead:
        in_h, in_w = int(heatmaps.shape[0]), int(heatmaps.shape[1])
        out_h, out_w = int(original_hw[0]), int(original_hw[1])
        key = (in_h, in_w, out_h, out_w)
        if key in self.cache:
            return self.cache[key]

        output_dir = Path(self.args.head_cache_dir)
        name = head_name(
            in_h,
            in_w,
            out_h,
            out_w,
            topk=self.args.topk,
            resize_mode=self.args.resize_mode,
            nms_impl=self.args.nms_impl,
        )
        mxr = output_dir / f"{name}.mxr"

        if self.args.compile_heads or not mxr.exists():
            mxr = compile_resize_nms_topk_head(
                in_h=in_h,
                in_w=in_w,
                out_h=out_h,
                out_w=out_w,
                output_dir=output_dir,
                channels=18,
                topk=self.args.topk,
                threshold=self.args.threshold,
                nms_radius=self.args.nms_radius,
                nms_impl=self.args.nms_impl,
                resize_mode=self.args.resize_mode,
                force=self.args.force_compile_heads,
                keep_onnx=self.args.keep_onnx,
                exhaustive_tune=self.args.exhaustive_tune,
            )
            self.compiled += 1

        head = MIGraphXResizeNMSTopKHead(str(mxr))
        self.cache[key] = head
        return head


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = CocoValDataset(args.labels, args.images_folder)
    model = load_migraphx_model(args.model, args.onnx, args.quantization, args.exhaustive_tune)
    head_cache = HeadCache(args)

    config = PostprocessConfig(max_keypoints_per_type=args.topk, threshold=args.threshold, nms_radius_fullres=args.nms_radius)
    meters = {
        "optimized_batch_k20_fast": Meter("optimized_batch_k20_fast"),
        "migraphx_resize_nms_topk_k20": Meter("migraphx_resize_nms_topk_k20"),
    }

    target_count = args.max_images if args.max_images is not None else max(0, len(dataset) - args.skip_images)
    print("\nRunning COCO accuracy validation for resize+NMS+TopK")
    print(f"Model:        {args.model}")
    print(f"Labels:       {args.labels}")
    print(f"Images:       {args.images_folder}")
    print(f"Max images:   {args.max_images}")
    print(f"Resize mode:  {args.resize_mode}")
    print(f"NMS impl:     {args.nms_impl}")
    print(f"TopK:         {args.topk}")
    print(f"Head cache:   {args.head_cache_dir}")

    processed = 0
    for i, sample in enumerate(dataset):  # type: ignore
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

        # Baseline: existing registry path.
        base_out = postprocess_from_maps(
            "optimized_batch_k20_fast",
            heatmaps,
            pafs,
            original_hw,
            config=config,
        )
        meters["optimized_batch_k20_fast"].add(infer_timings, base_out.timings)
        meters["optimized_batch_k20_fast"].detections.extend(
            build_coco_detections(image_id, base_out.pose_entries, base_out.all_keypoints)
        )

        # Experimental MIGraphX resize+NMS+TopK path.
        head = head_cache.get(heatmaps, original_hw)
        poses, kpts, topk_timings = postprocess_maps_resize_nms_topk(
            heatmaps,
            pafs,
            original_hw,
            head=head,
            topk=args.topk,
            threshold=args.threshold,
            points_per_limb=args.points_per_limb,
        )
        meters["migraphx_resize_nms_topk_k20"].add(infer_timings, topk_timings)
        meters["migraphx_resize_nms_topk_k20"].detections.extend(
            build_coco_detections(image_id, poses, kpts)
        )

        processed += 1

    rows: List[Dict[str, Any]] = []
    model_name = Path(args.model).name
    model_dir = Path(args.output_dir) / Path(args.model).stem
    model_dir.mkdir(parents=True, exist_ok=True)

    for variant, meter in meters.items():
        detections_path = model_dir / f"detections_{variant}.json"
        with open(detections_path, "w") as f:
            json.dump(meter.detections, f)

        print(f"\n--- COCO eval: variant={variant} ---")
        metrics = coco_eval_stats(args.labels, str(detections_path))
        row = {
            "model": model_name,
            "model_path": args.model,
            "variant": variant,
            **parse_model_metadata(args.model),
            **meter.latency(),
            **metrics,
            "detections_path": str(detections_path),
            "compiled_heads_this_run": head_cache.compiled,
        }
        rows.append(row)

    # Speedup columns relative to baseline.
    baseline = rows[0]
    base_e2e = float(baseline.get("e2e_avg_ms", 0.0) or 0.0)
    for row in rows:
        e2e = float(row.get("e2e_avg_ms", 0.0) or 0.0)
        if row is baseline or base_e2e <= 0 or e2e <= 0:
            row["e2e_speedup_vs_baseline"] = 1.0
            row["e2e_delta_pct_vs_baseline"] = 0.0
        else:
            row["e2e_speedup_vs_baseline"] = base_e2e / e2e
            row["e2e_delta_pct_vs_baseline"] = ((e2e - base_e2e) / base_e2e) * 100.0

    summary_json = Path(args.output_dir) / "accuracy_resize_nms_topk_summary.json"
    summary_csv = Path(args.output_dir) / "accuracy_resize_nms_topk_summary.csv"
    with open(summary_json, "w") as f:
        json.dump(rows, f, indent=2)
    write_flat_csv(str(summary_csv), rows)

    print("\n" + "=" * 176)
    print("RESIZE + NMS + TOPK ACCURACY SUMMARY")
    print("=" * 176)
    print(
        f"{'variant':<34} {'img':>5} {'AP':>6} {'AP50':>6} {'AP75':>6} {'AR':>6} "
        f"{'mx_topk':>8} {'paf':>8} {'adapt':>8} {'group':>8} {'post':>8} {'e2e':>8} {'fps':>8} {'speedup':>8}"
    )
    print("-" * 176)
    for r in rows:
        print(
            f"{r['variant']:<34} {int(r['images']):>5} "
            f"{float(r.get('AP', 0.0)):>6.3f} {float(r.get('AP50', 0.0)):>6.3f} "
            f"{float(r.get('AP75', 0.0)):>6.3f} {float(r.get('AR', 0.0)):>6.3f} "
            f"{float(r.get('hm_resize_mx_topk_ms', 0.0)):>8.2f} {float(r.get('paf_resize_ms', 0.0)):>8.2f} "
            f"{float(r.get('topk_adapter_ms', 0.0)):>8.2f} {float(r.get('group_ms', 0.0)):>8.2f} "
            f"{float(r.get('post_avg_ms', 0.0)):>8.2f} {float(r.get('e2e_avg_ms', 0.0)):>8.2f} "
            f"{float(r.get('e2e_fps', 0.0)):>8.2f} {float(r.get('e2e_speedup_vs_baseline', 1.0)):>8.2f}"
        )
    print("=" * 176)
    print(f"Saved JSON: {summary_json}")
    print(f"Saved CSV:  {summary_csv}")
    print(f"Compiled/loaded unique heads: {len(head_cache.cache)} loaded, {head_cache.compiled} compiled this run")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COCO accuracy validation for experimental MIGraphX resize+NMS+TopK postprocess.")
    p.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    p.add_argument("--models", nargs="+", default=None, help="Compatibility: first model is used.")
    p.add_argument("--onnx", default="")
    p.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    p.add_argument("--images-folder", default="coco/val2017")
    p.add_argument("--output-dir", default="outputs/accuracy_resize_nms_topk")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--skip-images", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--base-height", type=int, default=544)
    p.add_argument("--base-width", type=int, default=968)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--resize-mode", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--head-cache-dir", default="models/resize_nms_topk_coco_cache")
    p.add_argument("--compile-heads", action="store_true", help="Compile missing per-shape heads on demand.")
    p.add_argument("--force-compile-heads", action="store_true", help="Recompile heads even when present.")
    p.add_argument("--keep-onnx", action="store_true")
    args = p.parse_args()
    if args.models:
        args.model = args.models[0]
    return args


if __name__ == "__main__":
    run(parse_args())
