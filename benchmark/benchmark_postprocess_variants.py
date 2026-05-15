import argparse
import csv
import time
import os

import cv2
import numpy as np

from video_val import PoseEstimator
from modules.keypoints import (
    extract_keypoints,
    extract_keypoints_batch,
    extract_keypoints_batch_cv2,
    group_keypoints,
)


# ---------------------------------------------------------
# Small timing helper
# ---------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


# ---------------------------------------------------------
# Shared decode
# ---------------------------------------------------------
def decode_outputs(results, out_h, out_w):
    heatmaps = np.asarray(results[0], dtype=np.float32).reshape(19, out_h, out_w)
    pafs = np.asarray(results[1], dtype=np.float32).reshape(38, out_h, out_w)

    heatmaps = np.moveaxis(heatmaps, 0, -1)  # H x W x 19
    pafs = np.moveaxis(pafs, 0, -1)          # H x W x 38

    return heatmaps, pafs


# ---------------------------------------------------------
# Variant 1: original / standard postprocess
# resize heatmaps + resize pafs + original extract_keypoints
# ---------------------------------------------------------
def postprocess_standard_timed(results, original_hw, engine):
    timings = {}
    total_start = time.perf_counter()

    orig_h, orig_w = original_hw
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode"] = t.ms

    with Timer() as t:
        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )
    timings["resize_heatmaps"] = t.ms

    with Timer() as t:
        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )
    timings["resize_pafs"] = t.ms

    with Timer() as t:
        all_kpts = []
        total = 0
        for kpt_idx in range(18):
            total += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_kpts,
                total
            )
    timings["extract_keypoints"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_keypoints"] = t.ms

    timings["total_postprocess"] = (time.perf_counter() - total_start) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------
# Variant 2: fast postprocess
# no heatmap/paf resize before grouping
# group at output resolution, then scale keypoints
# ---------------------------------------------------------
def postprocess_fast_timed(results, original_hw, engine):
    timings = {}
    total_start = time.perf_counter()

    orig_h, orig_w = original_hw
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode"] = t.ms

    timings["resize_heatmaps"] = 0.0
    timings["resize_pafs"] = 0.0

    with Timer() as t:
        all_kpts = []
        total = 0
        for kpt_idx in range(18):
            total += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_kpts,
                total
            )
    timings["extract_keypoints"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_keypoints"] = t.ms

    with Timer() as t:
        scale_x = orig_w / out_w
        scale_y = orig_h / out_h

        for kpt in kpts:
            kpt[0] *= scale_x
            kpt[1] *= scale_y
    timings["scale_keypoints"] = t.ms

    timings["total_postprocess"] = (time.perf_counter() - total_start) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------
# Variant 3/4: optimized batch postprocess
# resize heatmaps + resize pafs + extract_keypoints_batch
# ---------------------------------------------------------
def postprocess_optimized_batch_timed(
    results,
    original_hw,
    engine,
    max_keypoints_per_type=20
):
    timings = {}
    total_start = time.perf_counter()

    orig_h, orig_w = original_hw
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode"] = t.ms

    with Timer() as t:
        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )
    timings["resize_heatmaps"] = t.ms

    with Timer() as t:
        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )
    timings["resize_pafs"] = t.ms

    with Timer() as t:
        all_kpts, total = extract_keypoints_batch_cv2(
            heatmaps[:, :, :18],
            max_keypoints_per_type=max_keypoints_per_type
        )
    timings["extract_keypoints"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_keypoints"] = t.ms

    timings["scale_keypoints"] = 0.0
    timings["total_postprocess"] = (time.perf_counter() - total_start) * 1000.0

    return poses, kpts, timings


def postprocess_optimized_batch_k10_timed(results, original_hw, engine):
    return postprocess_optimized_batch_timed(
        results,
        original_hw,
        engine,
        max_keypoints_per_type=10
    )


def postprocess_optimized_batch_k20_timed(results, original_hw, engine):
    return postprocess_optimized_batch_timed(
        results,
        original_hw,
        engine,
        max_keypoints_per_type=20
    )


# ---------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------
def mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def median(values):
    if not values:
        return 0.0
    return float(np.median(values))


def summarize_variant(name, rows, baseline_total=None):
    totals = [r["total_postprocess"] for r in rows]

    avg_total = mean(totals)
    med_total = median(totals)

    if baseline_total is None or avg_total == 0:
        speedup = 1.0
        delta_ms = 0.0
        delta_pct = 0.0
    else:
        speedup = baseline_total / avg_total
        delta_ms = avg_total - baseline_total
        delta_pct = ((avg_total - baseline_total) / baseline_total) * 100.0

    return {
        "variant": name,
        "frames": len(rows),
        "decode_ms": mean([r.get("decode", 0.0) for r in rows]),
        "hm_resize_ms": mean([r.get("resize_heatmaps", 0.0) for r in rows]),
        "paf_resize_ms": mean([r.get("resize_pafs", 0.0) for r in rows]),
        "extract_ms": mean([r.get("extract_keypoints", 0.0) for r in rows]),
        "group_ms": mean([r.get("group_keypoints", 0.0) for r in rows]),
        "scale_ms": mean([r.get("scale_keypoints", 0.0) for r in rows]),
        "post_avg_ms": avg_total,
        "post_median_ms": med_total,
        "post_min_ms": float(np.min(totals)) if totals else 0.0,
        "post_max_ms": float(np.max(totals)) if totals else 0.0,
        "post_fps": 1000.0 / avg_total if avg_total > 0 else 0.0,
        "speedup_vs_standard": speedup,
        "delta_ms_vs_standard": delta_ms,
        "delta_pct_vs_standard": delta_pct,
    }


def print_table(summaries):
    headers = [
        "variant",
        "frames",
        "decode",
        "hm_resize",
        "paf_resize",
        "extract",
        "group",
        "scale",
        "post_avg",
        "post_med",
        "post_fps",
        "speedup",
        "delta_ms",
        "delta_%",
    ]

    print("\n" + "=" * 150)
    print("POSTPROCESS BENCHMARK SUMMARY")
    print("=" * 150)

    print(
        f"{'variant':<28} "
        f"{'frames':>6} "
        f"{'decode':>9} "
        f"{'hm_res':>9} "
        f"{'paf_res':>9} "
        f"{'extract':>9} "
        f"{'group':>9} "
        f"{'scale':>8} "
        f"{'post_avg':>10} "
        f"{'post_med':>10} "
        f"{'FPS':>8} "
        f"{'speedup':>9} "
        f"{'Δms':>9} "
        f"{'Δ%':>8}"
    )

    print("-" * 150)

    for s in summaries:
        print(
            f"{s['variant']:<28} "
            f"{s['frames']:>6} "
            f"{s['decode_ms']:>9.2f} "
            f"{s['hm_resize_ms']:>9.2f} "
            f"{s['paf_resize_ms']:>9.2f} "
            f"{s['extract_ms']:>9.2f} "
            f"{s['group_ms']:>9.2f} "
            f"{s['scale_ms']:>8.2f} "
            f"{s['post_avg_ms']:>10.2f} "
            f"{s['post_median_ms']:>10.2f} "
            f"{s['post_fps']:>8.2f} "
            f"{s['speedup_vs_standard']:>9.2f} "
            f"{s['delta_ms_vs_standard']:>9.2f} "
            f"{s['delta_pct_vs_standard']:>8.2f}"
        )

    print("=" * 150)


def save_csv(path, summaries):
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print(f"\nCSV saved to: {path}")


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------
def benchmark_postprocess_variants(
    video_path,
    model_path,
    max_frames=100,
    warmup_frames=5,
    output_csv="postprocess_benchmark_summary.csv",
):
    engine = PoseEstimator(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    variants = [
        ("standard", postprocess_standard_timed),
        ("fast_no_resize", postprocess_fast_timed),
        #("optimized_batch_k10", postprocess_optimized_batch_k10_timed),
        ("optimized_batch_k20", postprocess_optimized_batch_k20_timed),
    ]

    per_variant_rows = {name: [] for name, _ in variants}

    frame_idx = 0
    measured_frames = 0

    print("\nRunning postprocess variant benchmark")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Warmup frames: {warmup_frames}")
    print(f"Measured frames: {max_frames}")
    print("Drawing and video writing are disabled.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)

        with Timer() as infer_t:
            raw_results = engine.model.run({
                "input": input_tensor
            })

        # Skip warmup frames from statistics
        if frame_idx <= warmup_frames:
            continue

        if measured_frames >= max_frames:
            break

        original_hw = frame.shape[:2]

        for name, fn in variants:
            _, _, timings = fn(raw_results, original_hw, engine)
            timings["preprocess"] = pre_t.ms
            timings["inference"] = infer_t.ms
            per_variant_rows[name].append(timings)

        measured_frames += 1

        if measured_frames % 10 == 0:
            print(f"Processed measured frames: {measured_frames}/{max_frames}")

    cap.release()

    if measured_frames == 0:
        raise RuntimeError("No frames were measured. Check video path or max_frames.")

    # Standard is baseline
    baseline_rows = per_variant_rows["standard"]
    baseline_total = mean([r["total_postprocess"] for r in baseline_rows])

    summaries = []
    for name, _ in variants:
        baseline = None if name == "standard" else baseline_total
        summaries.append(
            summarize_variant(
                name,
                per_variant_rows[name],
                baseline_total=baseline
            )
        )

    print_table(summaries)

    if output_csv:
        save_csv(output_csv, summaries)

    return summaries


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default="cctv_1280x720_24fps_3.mp4",
        help="Path to input video."
    )

    parser.add_argument(
        "--model",
        default="pose_model1_fp16_ref1.mxr",
        help="Path to MIGraphX .mxr model."
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of measured frames."
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup frames ignored in stats."
    )

    parser.add_argument(
        "--csv",
        default="postprocess_benchmark_summary.csv",
        help="Output CSV path."
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
    )
