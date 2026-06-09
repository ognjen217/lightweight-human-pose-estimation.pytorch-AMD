#!/usr/bin/env python3
"""Unified video speed validation for MIGraphX + postprocessing variants.

The script keeps postprocessing implementations centralized in
``modules.postprocessing``.  It adds validation-time conveniences for the newer
MIGraphX postprocess heads:

* auto-compile missing manual/fused/fused-pruned heads for the video shape
* accept the report alias ``merged_fused_pruned``
* pass all fused/pruned tuning parameters through ``PostprocessConfig``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import migraphx
import numpy as np

from modules.postprocessing import (
    DEFAULT_SPEED_VARIANTS,
    PostprocessConfig,
    available_modes,
    is_two_process_mode,
    normalize_mode,
    postprocess_from_results,
    run_two_process_postprocessing,
    variant_table,
)
from modules.postprocess_head_autocompile import (
    ensure_video_postprocess_heads,
    normalize_validation_variant_name,
    postprocess_extra_from_args,
)

TimingDict = Dict[str, float]

POSE_POSTPROCESS_MERGED_VARIANT = "pose_postprocessing_merged"
POSE_POSTPROCESS_MERGED_ALIASES = {
    "pose_postprocessing_merged",
    "pose-postprocessing-merged",
    "merged_pose_fused_pruned",
    "merged-pose-fused-pruned",
    "pose_fused_pruned",
    "pose-fused-pruned",
}


def normalize_speed_variant_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "-").replace("_", "-")
    aliases = {x.replace("_", "-") for x in POSE_POSTPROCESS_MERGED_ALIASES}
    if key in aliases:
        return POSE_POSTPROCESS_MERGED_VARIANT
    return normalize_mode(normalize_validation_variant_name(name))


SUMMARY_NUMERIC_KEYS = [
    "frames",
    "preprocess_ms",
    "inference_ms",
    "decode_ms",
    "hm_resize_ms",
    "paf_resize_ms",
    "mx_nms_ms",
    "manual_cubic_topk_ms",
    "fused_post_mx_ms",
    "fused_pruned_mx_ms",
    "extract_ms",
    "extract_from_mask_ms",
    "topk_adapter_ms",
    "group_ms",
    "mx_assembly_total_ms",
    "pruned_cpu_tail_ms",
    "post_avg_ms",
    "post_p50_ms",
    "post_p95_ms",
    "post_fps",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "e2e_fps",
    "e2e_speedup_vs_standard",
    "e2e_delta_pct_vs_standard",
]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


class MIGraphXVideoEngine:
    def __init__(self, model_path: str, target_dim: Tuple[int, int] = (968, 544), stride: int = 8):
        self.model_path = model_path
        self.w, self.h = target_dim
        self.stride = stride
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")
        self.model = migraphx.load(model_path)
        self.expected_dtype = str(self.model.get_parameter_shapes()["input"].type())

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.w, self.h))
        img = (img.astype(np.float32) - 128.0) / 256.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        img = np.ascontiguousarray(img)
        if "half" in self.expected_dtype:
            return img.astype(np.float16)
        return img.astype(np.float32)

    def infer(self, input_tensor: np.ndarray):
        return self.model.run({"input": input_tensor})


def mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def safe_get(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def ensure_parent(path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Fill optional metrics that are absent in some execution paths.

    Single-process summaries are produced by ``summarize_variant`` and contain
    all report columns. Two-process summaries are returned by
    ``run_two_process_postprocessing`` and may omit metrics that are not
    meaningful for that path, such as mx_nms/manual/fused/pruned stage timings.
    The validator should still be able to print and serialize those summaries.
    """
    for key in SUMMARY_NUMERIC_KEYS:
        if key == "frames":
            summary[key] = int(safe_get(summary, key))
        else:
            summary[key] = safe_get(summary, key)
    summary.setdefault("variant", "unknown")
    return summary


def summarize_variant(name: str, rows: List[TimingDict], baseline_e2e_ms: Optional[float]) -> Dict[str, Any]:
    def col(k: str):
        return [safe_get(r, k) for r in rows]

    post = col("total_postprocess")
    e2e = col("e2e")
    post_avg = mean(post)
    e2e_avg = mean(e2e)
    if baseline_e2e_ms and baseline_e2e_ms > 0 and e2e_avg > 0:
        speedup = baseline_e2e_ms / e2e_avg
        delta_pct = ((e2e_avg - baseline_e2e_ms) / baseline_e2e_ms) * 100.0
    else:
        speedup = 1.0
        delta_pct = 0.0

    return normalize_summary_row({
        "variant": name,
        "frames": len(rows),
        "preprocess_ms": mean(col("preprocess")),
        "inference_ms": mean(col("inference")),
        "decode_ms": mean(col("decode")),
        "hm_resize_ms": mean(col("resize_heatmaps")),
        "paf_resize_ms": mean(col("resize_pafs")),
        "mx_nms_ms": mean(col("mx_nms")),
        "manual_cubic_topk_ms": mean(col("manual_cubic_topk")),
        "fused_post_mx_ms": mean(col("fused_post_mx")),
        "fused_pruned_mx_ms": mean(col("fused_pruned_mx")),
        "pose_postprocess_merged_mx_ms": mean(col("pose_postprocess_merged_mx")),
        "extract_ms": mean(col("extract_keypoints")),
        "extract_from_mask_ms": mean(col("extract_from_mask")),
        "topk_adapter_ms": mean(col("topk_adapter")),
        "group_ms": mean(col("group_keypoints")),
        "mx_assembly_total_ms": mean(col("mx_assembly_total")),
        "pruned_cpu_tail_ms": mean(col("pruned_cpu_tail")),
        "post_avg_ms": post_avg,
        "post_p50_ms": percentile(post, 50),
        "post_p95_ms": percentile(post, 95),
        "e2e_avg_ms": e2e_avg,
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_fps": 1000.0 / e2e_avg if e2e_avg > 0 else 0.0,
        "post_fps": 1000.0 / post_avg if post_avg > 0 else 0.0,
        "e2e_speedup_vs_standard": speedup,
        "e2e_delta_pct_vs_standard": delta_pct,
    })


def print_variant_descriptions(variants: Sequence[str]) -> None:
    info_by_name = {row["variant"]: row for row in variant_table()}
    print("\nVariants:")
    for variant in variants:
        canonical = normalize_speed_variant_name(variant)
        if canonical == POSE_POSTPROCESS_MERGED_VARIANT:
            print(
                f"  {canonical:<40} "
                "Single MXR graph containing pose model + fused-pruned postprocess; CPU final pose assembly tail."
            )
            continue
        row = info_by_name[canonical]
        print(f"  {canonical:<40} {row['description']}")


def print_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 220)
    print("SPEED VALIDATION SUMMARY")
    print("=" * 220)
    print(
        f"{'variant':<40} {'frames':>6} {'pre':>8} {'infer':>8} {'decode':>8} "
        f"{'mx_nms':>8} {'manual':>8} {'fused':>8} {'pruned':>8} {'adapt':>8} {'asm':>8} {'tail':>8} "
        f"{'post':>9} {'post_p95':>9} {'e2e':>9} {'e2e_p95':>9} {'FPS':>8} {'speedup':>9} {'Δe2e%':>9}"
    )
    print("-" * 220)
    for s in summaries:
        s = normalize_summary_row(s)
        print(
            f"{str(s.get('variant', 'unknown')):<40} {int(s.get('frames', 0)):>6} "
            f"{safe_get(s, 'preprocess_ms'):>8.2f} {safe_get(s, 'inference_ms'):>8.2f} {safe_get(s, 'decode_ms'):>8.2f} "
            f"{safe_get(s, 'mx_nms_ms'):>8.2f} {safe_get(s, 'manual_cubic_topk_ms'):>8.2f} "
            f"{safe_get(s, 'fused_post_mx_ms'):>8.2f} {safe_get(s, 'fused_pruned_mx_ms'):>8.2f} "
            f"{safe_get(s, 'topk_adapter_ms'):>8.2f} {safe_get(s, 'mx_assembly_total_ms'):>8.2f} {safe_get(s, 'pruned_cpu_tail_ms'):>8.2f} "
            f"{safe_get(s, 'post_avg_ms'):>9.2f} {safe_get(s, 'post_p95_ms'):>9.2f} "
            f"{safe_get(s, 'e2e_avg_ms'):>9.2f} {safe_get(s, 'e2e_p95_ms'):>9.2f} "
            f"{safe_get(s, 'e2e_fps'):>8.2f} {safe_get(s, 'e2e_speedup_vs_standard'):>9.2f} {safe_get(s, 'e2e_delta_pct_vs_standard'):>9.2f}"
        )
    print("=" * 220)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path or not rows:
        return
    ensure_parent(path)
    rows = [normalize_summary_row(dict(row)) for row in rows]
    preferred_order = [
        "variant", "model", "frames", "preprocess_ms", "inference_ms", "decode_ms",
        "hm_resize_ms", "paf_resize_ms", "mx_nms_ms", "manual_cubic_topk_ms",
        "fused_post_mx_ms", "fused_pruned_mx_ms", "pose_postprocess_merged_mx_ms", "extract_ms", "extract_from_mask_ms",
        "topk_adapter_ms", "group_ms", "mx_assembly_total_ms", "pruned_cpu_tail_ms",
        "post_avg_ms", "post_p50_ms", "post_p95_ms", "post_fps",
        "e2e_avg_ms", "e2e_p95_ms", "e2e_fps", "e2e_speedup_vs_standard", "e2e_delta_pct_vs_standard",
    ]
    all_keys = set().union(*(row.keys() for row in rows))
    fieldnames = [k for k in preferred_order if k in all_keys]
    fieldnames.extend(sorted(k for k in all_keys if k not in fieldnames))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved:  {path}")


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    ensure_parent(path)
    rows = [normalize_summary_row(dict(row)) for row in rows]
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved: {path}")


def _assert_single_process_variants_are_migraphx_safe(variants: Sequence[str]) -> None:
    unsafe = [v for v in variants if v.startswith("gpu_")]
    if unsafe:
        raise RuntimeError(
            "Unsafe single-process GPU postprocess variant requested. Use the corresponding *_two_process variant. "
            f"Requested: {unsafe}."
        )


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


def _run_single_process_speed(args, variants: Sequence[str]) -> List[Dict[str, Any]]:
    if not variants:
        return []
    _assert_single_process_variants_are_migraphx_safe(variants)
    config = build_postprocess_config(args)
    engine = MIGraphXVideoEngine(args.model, target_dim=(args.target_width, args.target_height), stride=args.stride)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    per_variant_rows: Dict[str, List[TimingDict]] = {name: [] for name in variants}
    frame_idx = 0
    measured = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)
        with Timer() as infer_t:
            raw_results = engine.infer(input_tensor)
        original_hw = frame.shape[:2]
        is_warmup = frame_idx <= args.warmup
        if not is_warmup and measured >= args.frames:
            break

        for variant in variants:
            try:
                out = postprocess_from_results(
                    variant,
                    raw_results,
                    original_hw,
                    target_dim=(args.target_width, args.target_height),
                    stride=args.stride,
                    config=config,
                )
            except Exception as exc:
                cap.release()
                raise RuntimeError(f"Variant '{variant}' failed on frame {frame_idx}: {exc}") from exc
            if not is_warmup:
                row = dict(out.timings)
                row["preprocess"] = pre_t.ms
                row["inference"] = infer_t.ms
                row["e2e"] = pre_t.ms + infer_t.ms + row.get("total_postprocess", 0.0)
                per_variant_rows[variant].append(row)

        if not is_warmup:
            measured += 1
            if measured == 1 or (args.print_every > 0 and measured % args.print_every == 0):
                print(f"Processed measured frames: {measured}/{args.frames}")
    cap.release()
    if measured == 0:
        raise RuntimeError("No measured frames. Check --video, --warmup, and --frames.")

    baseline_name = "standard" if "standard" in per_variant_rows else variants[0]
    baseline_e2e = mean([safe_get(r, "e2e") for r in per_variant_rows[baseline_name]])
    return [
        summarize_variant(
            name=variant,
            rows=per_variant_rows[variant],
            baseline_e2e_ms=None if variant == baseline_name else baseline_e2e,
        )
        for variant in variants
    ]


def _run_pose_postprocessing_merged_speed(args, variants: Sequence[str]) -> List[Dict[str, Any]]:
    """Run a single merged MXR graph: pose model + fused-pruned postprocess.

    Expected graph outputs must match modules.migraphx_fused_postprocess_pruned:
      top_scores, top_indices, a_idx, b_idx, pair_score, pair_valid

    Final dynamic pose assembly is still done on CPU.
    """
    if not variants:
        return []

    merged_model = args.pose_postprocessing_merged_model or args.model
    engine = MIGraphXVideoEngine(
        merged_model,
        target_dim=(args.target_width, args.target_height),
        stride=args.stride,
    )

    from modules.mx_pair_assembly_pruned import assemble_poses_from_pruned_pairs

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    rows: List[TimingDict] = []
    frame_idx = 0
    measured = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        original_h, original_w = frame.shape[:2]

        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)

        with Timer() as graph_t:
            raw_results = engine.infer(input_tensor)

        if not isinstance(raw_results, (list, tuple)):
            raw_results = list(raw_results)

        if len(raw_results) < 6:
            cap.release()
            raise RuntimeError(
                "pose_postprocessing_merged expects a merged MXR with at least 6 outputs: "
                "top_scores, top_indices, a_idx, b_idx, pair_score, pair_valid. "
                f"Got {len(raw_results)} outputs."
            )

        with Timer() as tail_t:
            poses, kpts, asm_times = assemble_poses_from_pruned_pairs(
                np.asarray(raw_results[0], dtype=np.float32),
                np.asarray(raw_results[1], dtype=np.int64),
                np.asarray(raw_results[2], dtype=np.int64),
                np.asarray(raw_results[3], dtype=np.int64),
                np.asarray(raw_results[4], dtype=np.float32),
                np.asarray(raw_results[5], dtype=np.float32),
                full_width=int(original_w),
                threshold=args.threshold,
                min_pair_score=args.min_pair_score,
                return_timing=True,
            )

        is_warmup = frame_idx <= args.warmup
        if not is_warmup and measured >= args.frames:
            break

        if not is_warmup:
            row: TimingDict = {
                "decode": 0.0,
                "pose_postprocess_merged_mx": graph_t.ms,
                # Reuse existing summary columns; for this special mode this is the whole merged graph time.
                "fused_pruned_mx": graph_t.ms,
                "fused_post_mx": graph_t.ms,
                "mx_nms": graph_t.ms,
                "topk_adapter": float(asm_times.get("topk_adapter", 0.0)),
                "mx_assembly_total": float(asm_times.get("mx_assembly_total", tail_t.ms)),
                "pruned_cpu_tail": tail_t.ms,
                "group_keypoints": tail_t.ms,
                "group_total": tail_t.ms,
                "total_postprocess": tail_t.ms,
                "preprocess": pre_t.ms,
                "inference": graph_t.ms,
                "e2e": pre_t.ms + graph_t.ms + tail_t.ms,
            }
            rows.append(row)
            measured += 1

            if measured == 1 or (args.print_every > 0 and measured % args.print_every == 0):
                print(f"Processed measured frames: {measured}/{args.frames}")

    cap.release()

    if measured == 0:
        raise RuntimeError("No measured frames. Check --video, --warmup, and --frames.")

    return [
        summarize_variant(
            name=POSE_POSTPROCESS_MERGED_VARIANT,
            rows=rows,
            baseline_e2e_ms=None,
        )
    ]


def _run_two_process_speed(args, variants: Sequence[str]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for variant in variants:
        result = run_two_process_postprocessing(
            video_path=args.video,
            model_path=args.model,
            mode=variant,
            target_width=args.target_width,
            target_height=args.target_height,
            stride=args.stride,
            max_frames=args.frames,
            warmup_frames=args.warmup,
            slots=args.two_process_slots,
            print_every=args.print_every,
            torch_device="cuda" if args.torch_device == "auto" else args.torch_device,
            shared_dtype=args.shared_dtype,
            gpu_compute_dtype=args.gpu_compute_dtype,
            max_keypoints=args.max_keypoints,
            threshold=args.threshold,
            nms_radius_fullres=args.nms_radius_fullres,
            nms_radius_lowres=args.nms_radius_lowres,
            nms_impl=args.nms_impl,
            collect_rows=False,
        )
        summary = dict(result["summary"])
        summary.setdefault("variant", variant)
        summaries.append(normalize_summary_row(summary))
    return summaries


def _recompute_speedups(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    baseline = next((s for s in summaries if s.get("variant") == "standard"), summaries[0])
    baseline_e2e = float(baseline.get("e2e_avg_ms", 0.0) or 0.0)
    for s in summaries:
        e2e = float(s.get("e2e_avg_ms", 0.0) or 0.0)
        if s is baseline or baseline_e2e <= 0 or e2e <= 0:
            s["e2e_speedup_vs_standard"] = 1.0
            s["e2e_delta_pct_vs_standard"] = 0.0
        else:
            s["e2e_speedup_vs_standard"] = baseline_e2e / e2e
            s["e2e_delta_pct_vs_standard"] = ((e2e - baseline_e2e) / baseline_e2e) * 100.0


def validate_speed(args) -> List[Dict[str, Any]]:
    variants = [normalize_speed_variant_name(v) for v in args.variants]
    variants = list(dict.fromkeys(variants))

    if any(v in {"migraphx_nms", "migraphx_nms_k20"} for v in variants):
        if args.compile_migraphx_nms:
            from modules.migraphx_compiler import compile_nms_cache_for_video

            cache_dir = args.migraphx_nms_cache_dir or "models/nms_fullres_cache"
            mxr_path = compile_nms_cache_for_video(
                video_path=args.video,
                output_dir=cache_dir,
                threshold=args.threshold,
                radius=args.nms_radius_fullres,
                force=args.force_compile_migraphx_nms,
                keep_onnx=args.keep_migraphx_nms_onnx,
                exhaustive_tune=args.exhaustive_tune_migraphx_nms,
            )
            args.migraphx_nms_cache_dir = cache_dir
            args.migraphx_nms_mxr = str(mxr_path)
            print(f"MIGraphX NMS head ready: {mxr_path}")
        elif not args.migraphx_nms_cache_dir and not args.migraphx_nms_mxr:
            raise RuntimeError("migraphx_nms variants require --migraphx-nms-mxr, --migraphx-nms-cache-dir, or --compile-migraphx-nms.")

    ensure_video_postprocess_heads(args, variants, args.video)

    merged_pose_variants = [v for v in variants if v == POSE_POSTPROCESS_MERGED_VARIANT]
    regular_variants = [v for v in variants if v != POSE_POSTPROCESS_MERGED_VARIANT]
    single_process_variants = [v for v in regular_variants if not is_two_process_mode(v)]
    two_process_variants = [v for v in regular_variants if is_two_process_mode(v)]

    print("\nRunning speed validation")
    print(f"Video:          {args.video}")
    print(f"Model:          {args.model}")
    print(f"Target dim:     {args.target_width}x{args.target_height}")
    print(f"Stride:         {args.stride}")
    print(f"Warmup frames:  {args.warmup}")
    print(f"Measured frames:{args.frames}")
    print(f"Torch device:   {args.torch_device}")
    print(f"Auto compile:   {args.compile_missing_postprocess_heads}")
    print_variant_descriptions(variants)

    summaries: List[Dict[str, Any]] = []
    if merged_pose_variants:
        print("\nMerged pose+postprocess variants")
        print("--------------------------------")
        summaries.extend(_run_pose_postprocessing_merged_speed(args, merged_pose_variants))
    if single_process_variants:
        print("\nSingle-process variants")
        print("-----------------------")
        summaries.extend(_run_single_process_speed(args, single_process_variants))
    if two_process_variants:
        print("\nTwo-process variants")
        print("--------------------")
        summaries.extend(_run_two_process_speed(args, two_process_variants))

    summaries = [normalize_summary_row(s) for s in summaries]
    _recompute_speedups(summaries)
    print_table(summaries)
    write_csv(args.csv, summaries)
    write_json(args.json, summaries)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Unified video speed validation using modules.postprocessing as the single source of truth.")
    parser.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument(
        "--pose-postprocessing-merged-model",
        default="",
        help="MXR containing pose model + fused-pruned postprocess. Defaults to --model when pose_postprocessing_merged is used.",
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--csv", default="outputs/speed_validation_summary.csv")
    parser.add_argument("--json", default="outputs/speed_validation_summary.json")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_SPEED_VARIANTS), help=f"Postprocess variants or aliases. Canonical modes: {', '.join(available_modes())}")
    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--points-per-limb", type=int, default=8)
    parser.add_argument("--min-paf-score", type=float, default=0.05)
    parser.add_argument("--success-ratio-thr", type=float, default=0.8)
    parser.add_argument("--two-process-slots", type=int, default=3)
    parser.add_argument("--shared-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--prealloc-resize-buffers", action="store_true")

    parser.add_argument("--migraphx-nms-mxr", default="")
    parser.add_argument("--migraphx-nms-cache-dir", default="")
    parser.add_argument("--compile-migraphx-nms", action="store_true")
    parser.add_argument("--force-compile-migraphx-nms", action="store_true")
    parser.add_argument("--keep-migraphx-nms-onnx", action="store_true")
    parser.add_argument("--exhaustive-tune-migraphx-nms", action="store_true")

    parser.add_argument("--compile-missing-postprocess-heads", action="store_true", help="Compile missing manual/fused/fused-pruned postprocess heads for the requested shape.")
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
    return parser.parse_args()


if __name__ == "__main__":
    validate_speed(parse_args())
