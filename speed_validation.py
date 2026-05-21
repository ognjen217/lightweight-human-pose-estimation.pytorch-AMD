#!/usr/bin/env python3
"""
speed_validation.py

Unified video speed validation for the MIGraphX + post-processing pipeline.
This replaces the overlapping video_val*.py and benchmark_postprocess_variants.py
style scripts for single-process timing.

What it measures
----------------
For each measured video frame:
  1. preprocess frame for MIGraphX
  2. run MIGraphX inference once
  3. run one or more post-processing variants on the same raw outputs
  4. collect a consistent timing schema and print a summary table

All post-processing implementations are imported from modules/postprocessing.py.
No post-processing logic is implemented in this file.

Example
-------
python speed_validation.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --warmup 5 \
  --variants standard optimized_batch_k20_fast gpu_nms_fullres_cpu_group \
  --csv outputs/speed_summary.csv \
  --json outputs/speed_summary.json
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
    normalize_mode,
    is_two_process_mode,
    postprocess_from_results,
    run_two_process_postprocessing,
    variant_table,
)

TimingDict = Dict[str, float]


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


def safe_get(row: TimingDict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def ensure_parent(path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def summarize_variant(name: str, rows: List[TimingDict], baseline_e2e_ms: Optional[float]) -> Dict[str, Any]:
    pre = [safe_get(r, "preprocess") for r in rows]
    infer = [safe_get(r, "inference") for r in rows]
    decode = [safe_get(r, "decode") for r in rows]
    hm_resize = [safe_get(r, "resize_heatmaps") for r in rows]
    paf_resize = [safe_get(r, "resize_pafs") for r in rows]
    extract = [safe_get(r, "extract_keypoints") for r in rows]
    mx_nms = [safe_get(r, "mx_nms") for r in rows]
    mask_extract = [safe_get(r, "extract_from_mask") for r in rows]
    group = [safe_get(r, "group_keypoints") for r in rows]
    post = [safe_get(r, "total_postprocess") for r in rows]
    e2e = [safe_get(r, "e2e") for r in rows]

    post_avg = mean(post)
    e2e_avg = mean(e2e)

    if baseline_e2e_ms and baseline_e2e_ms > 0 and e2e_avg > 0:
        speedup = baseline_e2e_ms / e2e_avg
        delta_pct = ((e2e_avg - baseline_e2e_ms) / baseline_e2e_ms) * 100.0
    else:
        speedup = 1.0
        delta_pct = 0.0

    return {
        "variant": name,
        "frames": len(rows),
        "preprocess_ms": mean(pre),
        "inference_ms": mean(infer),
        "decode_ms": mean(decode),
        "hm_resize_ms": mean(hm_resize),
        "paf_resize_ms": mean(paf_resize),
        "extract_ms": mean(extract),
        "mx_nms_ms": mean(mx_nms),
        "extract_from_mask_ms": mean(mask_extract),
        "group_ms": mean(group),
        "post_avg_ms": post_avg,
        "post_p50_ms": percentile(post, 50),
        "post_p95_ms": percentile(post, 95),
        "e2e_avg_ms": e2e_avg,
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_fps": 1000.0 / e2e_avg if e2e_avg > 0 else 0.0,
        "post_fps": 1000.0 / post_avg if post_avg > 0 else 0.0,
        "e2e_speedup_vs_standard": speedup,
        "e2e_delta_pct_vs_standard": delta_pct,
    }


def print_variant_descriptions(variants: Sequence[str]) -> None:
    info_by_name = {row["variant"]: row for row in variant_table()}
    print("\nVariants:")
    for variant in variants:
        canonical = normalize_mode(variant)
        row = info_by_name[canonical]
        print(f"  {canonical:<32} {row['description']}")


def print_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 190)
    print("SPEED VALIDATION SUMMARY")
    print("=" * 190)
    print(
        f"{'variant':<34} {'frames':>6} {'pre':>8} {'infer':>8} {'decode':>8} "
        f"{'hm_res':>8} {'paf_res':>8} {'mx_nms':>8} {'extract':>9} {'mask_ext':>9} {'group':>9} "
        f"{'post':>9} {'post_p95':>9} {'e2e':>9} {'e2e_p95':>9} "
        f"{'FPS':>8} {'speedup':>9} {'Δe2e%':>9}"
    )
    print("-" * 190)
    for s in summaries:
        print(
            f"{s['variant']:<34} {int(s['frames']):>6} "
            f"{s['preprocess_ms']:>8.2f} {s['inference_ms']:>8.2f} {s['decode_ms']:>8.2f} "
            f"{s['hm_resize_ms']:>8.2f} {s['paf_resize_ms']:>8.2f} "
            f"{s.get('mx_nms_ms', 0.0):>8.2f} {s['extract_ms']:>9.2f} "
            f"{s.get('extract_from_mask_ms', 0.0):>9.2f} {s['group_ms']:>9.2f} "
            f"{s['post_avg_ms']:>9.2f} {s['post_p95_ms']:>9.2f} "
            f"{s['e2e_avg_ms']:>9.2f} {s['e2e_p95_ms']:>9.2f} "
            f"{s['e2e_fps']:>8.2f} {s['e2e_speedup_vs_standard']:>9.2f} "
            f"{s['e2e_delta_pct_vs_standard']:>9.2f}"
        )
    print("=" * 190)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path or not rows:
        return
    ensure_parent(path)
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
    print(f"CSV saved:  {path}")


def write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved: {path}")



def _assert_single_process_variants_are_migraphx_safe(variants: Sequence[str]) -> None:
    """Guard against accidentally mixing MIGraphX and Torch GPU in one process.

    On the target ROCm setup, MIGraphX inference and PyTorch GPU postprocessing
    must not be initialized in the same Python process. CPU-only postprocess
    variants are safe because they do not touch torch.cuda. GPU-NMS/PAF variants
    must be requested through their *_two_process aliases instead.
    """
    unsafe = [v for v in variants if v.startswith("gpu_")]
    if unsafe:
        mapping = {
            "gpu_nms_fullres_cpu_group": "gpu_nms_fullres_two_process",
            "gpu_nms_lowres_cpu_group": "gpu_nms_lowres_two_process",
        }
        suggestions = [mapping.get(v, f"{v} is single-process GPU and should be moved to a two-process mode") for v in unsafe]
        raise RuntimeError(
            "Unsafe single-process GPU postprocess variant requested. "
            "This setup must not initialize MIGraphX and PyTorch ROCm in the same process. "
            f"Requested: {unsafe}. Use instead: {suggestions}."
        )


def _run_single_process_speed(args, variants: Sequence[str]) -> List[Dict[str, Any]]:
    if not variants:
        return []

    _assert_single_process_variants_are_migraphx_safe(variants)

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

    engine = MIGraphXVideoEngine(
        args.model,
        target_dim=(args.target_width, args.target_height),
        stride=args.stride,
    )

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
        summaries.append(result["summary"])
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
    variants = [normalize_mode(v) for v in args.variants]
    # Keep user order but remove duplicates after alias normalization.
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
            raise RuntimeError(
                "migraphx_nms variants require --migraphx-nms-mxr or --migraphx-nms-cache-dir. "
                "For video speed validation, you can pass --compile-migraphx-nms to build the one needed head from video resolution."
            )

    single_process_variants = [v for v in variants if not is_two_process_mode(v)]
    two_process_variants = [v for v in variants if is_two_process_mode(v)]

    print("\nRunning speed validation")
    print(f"Video:          {args.video}")
    print(f"Model:          {args.model}")
    print(f"Target dim:     {args.target_width}x{args.target_height}")
    print(f"Stride:         {args.stride}")
    print(f"Warmup frames:  {args.warmup}")
    print(f"Measured frames:{args.frames}")
    print(f"Torch device:   {args.torch_device}")
    print(f"Draw/write:     disabled")
    print_variant_descriptions(variants)

    summaries: List[Dict[str, Any]] = []

    if single_process_variants:
        print("\nSingle-process variants")
        print("-----------------------")
        summaries.extend(_run_single_process_speed(args, single_process_variants))

    if two_process_variants:
        print("\nTwo-process variants")
        print("--------------------")
        summaries.extend(_run_two_process_speed(args, two_process_variants))

    _recompute_speedups(summaries)
    print_table(summaries)
    write_csv(args.csv, summaries)
    write_json(args.json, summaries)
    return summaries

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified video speed validation using modules.postprocessing as the single source of truth."
    )
    parser.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--csv", default="outputs/speed_validation_summary.csv")
    parser.add_argument("--json", default="outputs/speed_validation_summary.json")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_SPEED_VARIANTS),
        help=f"Postprocess variants or aliases. Canonical modes: {', '.join(available_modes())}",
    )
    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of falling back to CPU for GPU variants.")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--two-process-slots", type=int, default=3)
    parser.add_argument("--shared-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="2d")
    parser.add_argument("--migraphx-nms-mxr", default="", help="Static compiled MIGraphX NMS head .mxr for fixed-resolution video.")
    parser.add_argument("--migraphx-nms-cache-dir", default="", help="Directory with heatmap_nms_head_<H>x<W>.mxr files.")
    parser.add_argument("--compile-migraphx-nms", action="store_true", help="Compile the one video-resolution MIGraphX NMS head before running speed validation.")
    parser.add_argument("--force-compile-migraphx-nms", action="store_true", help="Recompile MIGraphX NMS even if the MXR already exists.")
    parser.add_argument("--keep-migraphx-nms-onnx", action="store_true", help="Keep temporary ONNX files generated for MIGraphX NMS compilation.")
    parser.add_argument("--exhaustive-tune-migraphx-nms", action="store_true", help="Pass exhaustive_tune=True when compiling MIGraphX NMS.")
    return parser.parse_args()


if __name__ == "__main__":
    validate_speed(parse_args())
