#!/usr/bin/env python3
"""
Benchmark all post-processing variants exposed by video_val_cli_postprocess.py.

This script reuses PoseEstimator.get_postprocess_fn(mode), so the benchmarked
variants stay synchronized with the CLI implementation.

Recommended run:
    python benchmark_all_postprocess_variants.py \
        --video cctv_1280x720_24fps_3.mp4 \
        --model pose_model1_fp16_ref1.mxr \
        --frames 100 \
        --warmup 5 \
        --no-gpu-paf

Full run, including experimental GPU PAF variants:
    python benchmark_all_postprocess_variants.py \
        --video cctv_1280x720_24fps_3.mp4 \
        --model pose_model1_fp16_ref1.mxr \
        --frames 100 \
        --warmup 5

Important ROCm note:
    On some systems MIGraphX and PyTorch ROCm may conflict when both are used in
    one process. If gpu-nms / gpu-paf fall back to CPU or crash, use the cached
    two-process benchmark path for final accuracy/energy evaluation.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from video_val_cli_postprocess import PoseEstimator, normalize_postprocess_mode


TimingDict = Dict[str, float]


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


# -------------------------------------------------------------------------
# Variants
# -------------------------------------------------------------------------
# Each item is:
#   output_name, cli_mode, description
#
# cli_mode must be accepted by PoseEstimator.get_postprocess_fn().
# output_name is what appears in the benchmark table/CSV.
#
# Aliases such as "cpu", "optimized", "k20-fast", and "k20_fast" point to the
# same implementation, so this default list keeps only unique implementations.
DEFAULT_VARIANTS: List[Tuple[str, str, str]] = [
    (
        "standard",
        "standard",
        "Original full-res CPU: per-channel extract_keypoints + group_keypoints.",
    ),
    (
        "fast_no_resize",
        "fast",
        "Low-res CPU postprocess without heatmap/PAF resize, then scale keypoints.",
    ),
    (
        "k20_standard_group",
        "k20",
        "Full-res batch K20 extraction + standard group_keypoints.",
    ),
    (
        "k20_fast_cpu",
        "k20-fast",
        "Full-res batch K20 extraction + group_keypoints_fast. Best pure CPU baseline.",
    ),
    (
        "lowres_cpu_group",
        "lowres-cpu-group",
        "Low-res batched K20 extraction + CPU group_keypoints_fast.",
    ),
    (
        "gpu_nms_fullres_cpu_group",
        "gpu-nms",
        "Best hybrid: full-res resize + GPU NMS/keypoint extraction + CPU fast grouping.",
    ),
    (
        "gpu_fullres_paf",
        "gpu-fullres-paf",
        "Full-res CPU K20 extraction + GPU PAF scoring + CPU pose assembly.",
    ),
    (
        "gpu_lowres_paf",
        "gpu-lowres-paf",
        "Low-res GPU NMS + GPU PAF scoring, then scale keypoints.",
    ),
]


# Optional alias variants, useful only if you want to prove aliases are equivalent.
ALIAS_VARIANTS: List[Tuple[str, str, str]] = [
    ("cpu_alias", "cpu", "Alias for k20_fast_cpu."),
    ("optimized_alias", "optimized", "Alias for k20_fast_cpu."),
    ("gpu_alias", "gpu", "Alias for gpu_lowres_paf."),
    ("hybrid_alias", "hybrid", "Alias for gpu_nms_fullres_cpu_group."),
]


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def safe_get(row: TimingDict, key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def summarize_variant(
    name: str,
    cli_mode: str,
    description: str,
    rows: List[TimingDict],
    baseline_total_ms: Optional[float],
) -> Dict[str, float]:
    totals = [safe_get(r, "total_postprocess") for r in rows]

    avg_total = mean(totals)
    med_total = float(np.median(totals)) if totals else 0.0
    p95_total = percentile(totals, 95)

    if baseline_total_ms is None or avg_total <= 0:
        speedup = 1.0
        delta_ms = 0.0
        delta_pct = 0.0
    else:
        speedup = baseline_total_ms / avg_total
        delta_ms = avg_total - baseline_total_ms
        delta_pct = (delta_ms / baseline_total_ms) * 100.0 if baseline_total_ms > 0 else 0.0

    return {
        "variant": name,
        "cli_mode": cli_mode,
        "frames": len(rows),
        "preprocess_ms": mean([safe_get(r, "preprocess") for r in rows]),
        "inference_ms": mean([safe_get(r, "inference") for r in rows]),
        "decode_ms": mean([safe_get(r, "decode") for r in rows]),
        "hm_resize_ms": mean([safe_get(r, "resize_heatmaps") for r in rows]),
        "paf_resize_ms": mean([safe_get(r, "resize_pafs") for r in rows]),
        "extract_ms": mean([safe_get(r, "extract_keypoints") for r in rows]),
        "group_ms": mean([safe_get(r, "group_keypoints") for r in rows]),
        "group_prepare_ms": mean([safe_get(r, "group_prepare") for r in rows]),
        "group_pairs_ms": mean([safe_get(r, "group_pairs") for r in rows]),
        "group_sample_ms": mean([safe_get(r, "group_sample") for r in rows]),
        "group_affinity_ms": mean([safe_get(r, "group_affinity") for r in rows]),
        "group_nms_ms": mean([safe_get(r, "group_nms") for r in rows]),
        "group_pose_ms": mean([safe_get(r, "group_pose") for r in rows]),
        "group_filter_ms": mean([safe_get(r, "group_filter") for r in rows]),
        "scale_ms": mean([safe_get(r, "scale_keypoints") for r in rows]),
        "post_avg_ms": avg_total,
        "post_median_ms": med_total,
        "post_p95_ms": p95_total,
        "post_min_ms": float(np.min(totals)) if totals else 0.0,
        "post_max_ms": float(np.max(totals)) if totals else 0.0,
        "post_fps": 1000.0 / avg_total if avg_total > 0 else 0.0,
        "speedup_vs_standard": speedup,
        "delta_ms_vs_standard": delta_ms,
        "delta_pct_vs_standard": delta_pct,
        "description": description,
    } # type: ignore


def print_table(summaries: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 180)
    print("POSTPROCESS BENCHMARK SUMMARY")
    print("=" * 180)
    print(
        f"{'variant':<32} "
        f"{'mode':<18} "
        f"{'frames':>6} "
        f"{'pre':>8} "
        f"{'infer':>8} "
        f"{'decode':>8} "
        f"{'hm_res':>8} "
        f"{'paf_res':>8} "
        f"{'extract':>9} "
        f"{'group':>9} "
        f"{'post_avg':>10} "
        f"{'p95':>9} "
        f"{'FPS':>8} "
        f"{'speedup':>9} "
        f"{'Δms':>9} "
        f"{'Δ%':>8}"
    )
    print("-" * 180)

    for s in summaries:
        print(
            f"{str(s['variant']):<32} "
            f"{str(s['cli_mode']):<18} "
            f"{int(s['frames']):>6} "
            f"{s['preprocess_ms']:>8.2f} "
            f"{s['inference_ms']:>8.2f} "
            f"{s['decode_ms']:>8.2f} "
            f"{s['hm_resize_ms']:>8.2f} "
            f"{s['paf_resize_ms']:>8.2f} "
            f"{s['extract_ms']:>9.2f} "
            f"{s['group_ms']:>9.2f} "
            f"{s['post_avg_ms']:>10.2f} "
            f"{s['post_p95_ms']:>9.2f} "
            f"{s['post_fps']:>8.2f} "
            f"{s['speedup_vs_standard']:>9.2f} "
            f"{s['delta_ms_vs_standard']:>9.2f} "
            f"{s['delta_pct_vs_standard']:>8.2f}"
        )

    print("=" * 180)


def print_descriptions(variants: List[Tuple[str, str, str]]) -> None:
    print("\nVariants to benchmark:")
    for name, mode, description in variants:
        print(f"  {name:<32} mode={mode:<18} {description}")


def save_csv(path: str, summaries: List[Dict[str, float]]) -> None:
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nCSV saved to: {path}")


def save_json(path: str, summaries: List[Dict[str, float]]) -> None:
    if not summaries:
        return

    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"JSON saved to: {path}")


def build_variants(
    selected_modes: Optional[List[str]],
    include_aliases: bool,
    no_gpu_paf: bool,
    only_gpu_nms: bool,
) -> List[Tuple[str, str, str]]:
    variants = list(DEFAULT_VARIANTS)

    if include_aliases:
        variants.extend(ALIAS_VARIANTS)

    if no_gpu_paf:
        blocked_modes = {"gpu-fullres-paf", "gpu-lowres-paf", "gpu"}
        variants = [
            v for v in variants
            if normalize_postprocess_mode(v[1]) not in blocked_modes
        ]

    if only_gpu_nms:
        keep = {"standard", "k20-fast", "cpu", "optimized", "gpu-nms", "hybrid", "hybrid-gpu-nms"}
        variants = [
            v for v in variants
            if normalize_postprocess_mode(v[1]) in keep
        ]

    if selected_modes:
        requested = {normalize_postprocess_mode(m) for m in selected_modes}
        variants = [
            v for v in variants
            if normalize_postprocess_mode(v[1]) in requested
            or normalize_postprocess_mode(v[0]) in requested
        ]

    if not variants:
        raise ValueError("No variants selected. Check --variants / --no-gpu-paf / --only-gpu-nms options.")

    return variants


def benchmark_postprocess_variants(
    video_path: str,
    model_path: str,
    max_frames: int = 100,
    warmup_frames: int = 5,
    output_csv: str = "postprocess_benchmark_summary.csv",
    output_json: str = "postprocess_benchmark_summary.json",
    target_dim: Tuple[int, int] = (968, 544),
    stride: int = 8,
    torch_device: str = "auto",
    selected_modes: Optional[List[str]] = None,
    include_aliases: bool = False,
    no_gpu_paf: bool = False,
    only_gpu_nms: bool = False,
    print_every: int = 10,
) -> List[Dict[str, float]]:
    engine = PoseEstimator(
        model_path,
        target_dim=target_dim,
        stride=stride,
        torch_device=torch_device,
    )

    variants = build_variants(
        selected_modes=selected_modes,
        include_aliases=include_aliases,
        no_gpu_paf=no_gpu_paf,
        only_gpu_nms=only_gpu_nms,
    )

    # Resolve functions once, before reading the video, so invalid modes fail early.
    variant_fns = []
    for name, mode, description in variants:
        fn = engine.get_postprocess_fn(mode)
        variant_fns.append((name, mode, description, fn))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    per_variant_rows: Dict[str, List[TimingDict]] = {name: [] for name, _, _, _ in variant_fns}

    frame_idx = 0
    measured_frames = 0

    print("\nRunning all postprocess variant benchmark")
    print(f"Video:          {video_path}")
    print(f"Model:          {model_path}")
    print(f"Target dim:     {target_dim[0]}x{target_dim[1]}")
    print(f"Stride:         {stride}")
    print(f"Torch device:   {torch_device}")
    print(f"Warmup frames:  {warmup_frames}")
    print(f"Measured frames:{max_frames}")
    print("Drawing and video writing are disabled.")
    print_descriptions(variants)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)

        with Timer() as infer_t:
            raw_results = engine.model.run({"input": input_tensor})

        original_hw = frame.shape[:2]

        # Run postprocess variants even during warmup, but do not record them.
        # This warms CPU caches and GPU kernels more fairly.
        is_warmup = frame_idx <= warmup_frames

        if not is_warmup and measured_frames >= max_frames:
            break

        for name, mode, description, fn in variant_fns:
            try:
                _, _, timings = fn(raw_results, original_hw)
            except Exception as exc:
                cap.release()
                raise RuntimeError(
                    f"Variant '{name}' / mode '{mode}' failed on frame {frame_idx}: {exc}"
                ) from exc

            if not is_warmup:
                timings = dict(timings)
                timings["preprocess"] = pre_t.ms
                timings["inference"] = infer_t.ms
                per_variant_rows[name].append(timings)

        if not is_warmup:
            measured_frames += 1
            if measured_frames == 1 or (print_every > 0 and measured_frames % print_every == 0):
                print(f"Processed measured frames: {measured_frames}/{max_frames}")

    cap.release()

    if measured_frames == 0:
        raise RuntimeError("No frames were measured. Check video path, warmup, or max_frames.")

    baseline_name = "standard"
    if baseline_name not in per_variant_rows:
        # Fallback to first selected variant if user intentionally excluded standard.
        baseline_name = variant_fns[0][0]

    baseline_total_ms = mean([
        safe_get(r, "total_postprocess")
        for r in per_variant_rows[baseline_name]
    ])

    summaries: List[Dict[str, float]] = []
    for name, mode, description, _ in variant_fns:
        baseline = None if name == baseline_name else baseline_total_ms
        summaries.append(
            summarize_variant(
                name=name,
                cli_mode=mode,
                description=description,
                rows=per_variant_rows[name],
                baseline_total_ms=baseline,
            )
        )

    print_table(summaries)

    if output_csv:
        save_csv(output_csv, summaries)

    if output_json:
        save_json(output_json, summaries)

    return summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all post-processing variants from video_val_cli_postprocess.py."
    )

    parser.add_argument(
        "--video",
        default="cctv_1280x720_24fps_3.mp4",
        help="Path to input video.",
    )
    parser.add_argument(
        "--model",
        default="pose_model1_fp16_ref1.mxr",
        help="Path to MIGraphX .mxr model.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of measured frames.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup frames ignored in stats.",
    )
    parser.add_argument(
        "--csv",
        default="postprocess_benchmark_summary.csv",
        help="Output CSV path. Use empty string to disable.",
    )
    parser.add_argument(
        "--json",
        default="postprocess_benchmark_summary.json",
        help="Output JSON path. Use empty string to disable.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=968,
        help="Network input width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=544,
        help="Network input height.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Model stride.",
    )
    parser.add_argument(
        "--torch-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Torch device for GPU postprocess variants.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Optional subset of variants/modes to run. Examples: "
            "--variants standard k20-fast gpu-nms"
        ),
    )
    parser.add_argument(
        "--include-aliases",
        action="store_true",
        help="Also benchmark CLI aliases such as cpu, optimized, gpu, hybrid.",
    )
    parser.add_argument(
        "--no-gpu-paf",
        action="store_true",
        help="Skip gpu-fullres-paf and gpu-lowres-paf. Keeps gpu-nms.",
    )
    parser.add_argument(
        "--only-gpu-nms",
        action="store_true",
        help="Run only standard/k20_fast/gpu-nms family variants.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Progress print interval in measured frames. Use 0 to disable after first frame.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    benchmark_postprocess_variants(
        video_path=args.video,
        model_path=args.model,
        max_frames=args.frames,
        warmup_frames=args.warmup,
        output_csv=args.csv,
        output_json=args.json,
        target_dim=(args.target_width, args.target_height),
        stride=args.stride,
        torch_device=args.torch_device,
        selected_modes=args.variants,
        include_aliases=args.include_aliases,
        no_gpu_paf=args.no_gpu_paf,
        only_gpu_nms=args.only_gpu_nms,
        print_every=args.print_every,
    )