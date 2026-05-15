import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from video_val import PoseEstimator
from modules.keypoints import (
    extract_keypoints,
    extract_keypoints_batch_cv2,
    group_keypoints,
)


# ---------------------------------------------------------
# Timing helper
# ---------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def safe_sync():
    """
    Best-effort GPU synchronization.

    If PyTorch with ROCm/CUDA is installed and available, this synchronizes
    before stopping the timer. If not, the function silently does nothing.
    MIGraphX runs are usually effectively blocking from Python's perspective,
    but this helps if your wrapper internally uses torch tensors.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------
# GPU power helper
# ---------------------------------------------------------
def get_gpu_power_w():
    """
    Reads AMD GPU power with rocm-smi.

    Supports ROCm outputs such as:
        Current Socket Graphics Package Power (W): 43.02
        Average Graphics Package Power: 43.02 W
        Power: 43.02 W

    Returns:
        float power in W, or np.nan if reading fails.
    """
    commands = [
        ["rocm-smi", "--showpower"],
        ["/opt/rocm/bin/rocm-smi", "--showpower"],
    ]

    for cmd in commands:
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

            text = completed.stdout + "\n" + completed.stderr

            # Enable this with:
            # BENCH_POWER_DEBUG=1 python benchmark_postprocess_models_report.py ...
            if os.environ.get("BENCH_POWER_DEBUG") == "1":
                print("\n--- POWER DEBUG ---")
                print("CMD:", " ".join(cmd))
                print(text)
                print("--- END POWER DEBUG ---\n")

            patterns = [
                # Your ROCm format:
                # GPU[0] : Current Socket Graphics Package Power (W): 43.02
                r"Current\s+Socket\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",

                # Other possible ROCm formats:
                r"Average\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r"Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
                r"Socket\s+Graphics\s+Package\s+Power\s*\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",

                # Formats where unit comes after the value:
                r"Average\s+Graphics\s+Package\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
                r"Graphics\s+Package\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
                r"Current\s+Socket\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
                r"Socket\s+Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
                r"Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*W",
            ]

            for pattern in patterns:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    return float(m.group(1))

        except Exception:
            pass

    return float("nan")


def fmt_float(x, digits=2, na="N/A"):
    try:
        if x is None or np.isnan(float(x)) or np.isinf(float(x)):
            return na
        return f"{float(x):.{digits}f}"
    except Exception:
        return na


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
    timings["decode_ms"] = t.ms

    with Timer() as t:
        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
    timings["hm_resize_ms"] = t.ms

    with Timer() as t:
        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
    timings["paf_resize_ms"] = t.ms

    with Timer() as t:
        all_kpts = []
        total = 0
        for kpt_idx in range(18):
            total += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_kpts,
                total,
            )
    timings["extract_ms"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_ms"] = t.ms

    timings["scale_ms"] = 0.0
    timings["post_ms"] = (time.perf_counter() - total_start) * 1000.0
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
    timings["decode_ms"] = t.ms

    timings["hm_resize_ms"] = 0.0
    timings["paf_resize_ms"] = 0.0

    with Timer() as t:
        all_kpts = []
        total = 0
        for kpt_idx in range(18):
            total += extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_kpts,
                total,
            )
    timings["extract_ms"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_ms"] = t.ms

    with Timer() as t:
        scale_x = orig_w / out_w
        scale_y = orig_h / out_h

        for kpt in kpts:
            kpt[0] *= scale_x
            kpt[1] *= scale_y
    timings["scale_ms"] = t.ms

    timings["post_ms"] = (time.perf_counter() - total_start) * 1000.0
    return poses, kpts, timings


# ---------------------------------------------------------
# Variant 3/4: optimized batch postprocess
# resize heatmaps + resize pafs + extract_keypoints_batch_cv2
# ---------------------------------------------------------
def postprocess_optimized_batch_timed(
    results,
    original_hw,
    engine,
    max_keypoints_per_type=20,
):
    timings = {}
    total_start = time.perf_counter()

    orig_h, orig_w = original_hw
    out_h = engine.h // engine.stride
    out_w = engine.w // engine.stride

    with Timer() as t:
        heatmaps, pafs = decode_outputs(results, out_h, out_w)
    timings["decode_ms"] = t.ms

    with Timer() as t:
        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
    timings["hm_resize_ms"] = t.ms

    with Timer() as t:
        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC,
        )
    timings["paf_resize_ms"] = t.ms

    with Timer() as t:
        all_kpts, total = extract_keypoints_batch_cv2(
            heatmaps[:, :, :18],
            max_keypoints_per_type=max_keypoints_per_type,
        )
    timings["extract_ms"] = t.ms

    with Timer() as t:
        poses, kpts = group_keypoints(all_kpts, pafs)
    timings["group_ms"] = t.ms

    timings["scale_ms"] = 0.0
    timings["post_ms"] = (time.perf_counter() - total_start) * 1000.0

    return poses, kpts, timings


def postprocess_optimized_batch_k10_timed(results, original_hw, engine):
    return postprocess_optimized_batch_timed(
        results,
        original_hw,
        engine,
        max_keypoints_per_type=10,
    )


def postprocess_optimized_batch_k20_timed(results, original_hw, engine):
    return postprocess_optimized_batch_timed(
        results,
        original_hw,
        engine,
        max_keypoints_per_type=20,
    )


# ---------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------
def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return float(np.mean(values))


def percentile(values, q):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def parse_model_metadata(model_path):
    """
    Heuristic parser for names like:
        pose_model1_fp16_ref1.mxr
        pose_model3_int8_ref2.mxr

    You can override this by renaming the file or by using the exact model
    path as the model identifier in the report.
    """
    name = Path(model_path).name

    mode = "unknown"
    for token in ["fp32", "fp16", "bf16", "int8", "quant", "float"]:
        if token in name.lower():
            mode = token
            break

    stages = "unknown"
    m = re.search(r"ref(\d+)", name.lower())
    if m:
        stages = m.group(1)

    return mode, stages


def summarize_variant(model_path, variant_name, rows, avg_power_w, baseline_post_ms=None, baseline_e2e_ms=None):
    post = [r["post_ms"] for r in rows]
    pre = [r["preprocess_ms"] for r in rows]
    infer = [r["inference_ms"] for r in rows]
    e2e = [r["e2e_ms"] for r in rows]

    post_avg = mean(post)
    e2e_avg = mean(e2e)

    if baseline_post_ms is None or post_avg == 0:
        post_speedup = 1.0
        post_delta_ms = 0.0
        post_delta_pct = 0.0
    else:
        post_speedup = baseline_post_ms / post_avg
        post_delta_ms = post_avg - baseline_post_ms
        post_delta_pct = ((post_avg - baseline_post_ms) / baseline_post_ms) * 100.0

    if baseline_e2e_ms is None or e2e_avg == 0:
        e2e_speedup = 1.0
        e2e_delta_ms = 0.0
        e2e_delta_pct = 0.0
    else:
        e2e_speedup = baseline_e2e_ms / e2e_avg
        e2e_delta_ms = e2e_avg - baseline_e2e_ms
        e2e_delta_pct = ((e2e_avg - baseline_e2e_ms) / baseline_e2e_ms) * 100.0

    e2e_fps = 1000.0 / e2e_avg if e2e_avg > 0 else 0.0
    post_fps = 1000.0 / post_avg if post_avg > 0 else 0.0

    fps_per_watt = e2e_fps / avg_power_w if avg_power_w and not np.isnan(avg_power_w) and avg_power_w > 0 else float("nan")
    energy_j_per_frame = avg_power_w * (e2e_avg / 1000.0) if avg_power_w and not np.isnan(avg_power_w) and avg_power_w > 0 else float("nan")

    mode, stages = parse_model_metadata(model_path)

    return {
        "model": Path(model_path).name,
        "model_path": str(model_path),
        "mode": mode,
        "stages": stages,
        "variant": variant_name,
        "frames": len(rows),

        "preprocess_avg_ms": mean(pre),
        "inference_avg_ms": mean(infer),

        "decode_avg_ms": mean([r.get("decode_ms", 0.0) for r in rows]),
        "hm_resize_avg_ms": mean([r.get("hm_resize_ms", 0.0) for r in rows]),
        "paf_resize_avg_ms": mean([r.get("paf_resize_ms", 0.0) for r in rows]),
        "extract_avg_ms": mean([r.get("extract_ms", 0.0) for r in rows]),
        "group_avg_ms": mean([r.get("group_ms", 0.0) for r in rows]),
        "scale_avg_ms": mean([r.get("scale_ms", 0.0) for r in rows]),

        "post_avg_ms": post_avg,
        "post_p50_ms": percentile(post, 50),
        "post_p95_ms": percentile(post, 95),
        "post_min_ms": float(np.min(post)) if post else 0.0,
        "post_max_ms": float(np.max(post)) if post else 0.0,
        "post_fps": post_fps,

        "e2e_avg_ms": e2e_avg,
        "e2e_p50_ms": percentile(e2e, 50),
        "e2e_p95_ms": percentile(e2e, 95),
        "e2e_min_ms": float(np.min(e2e)) if e2e else 0.0,
        "e2e_max_ms": float(np.max(e2e)) if e2e else 0.0,
        "e2e_fps": e2e_fps,

        "avg_power_w": avg_power_w,
        "fps_per_watt": fps_per_watt,
        "energy_j_per_frame": energy_j_per_frame,

        "post_speedup_vs_standard": post_speedup,
        "post_delta_ms_vs_standard": post_delta_ms,
        "post_delta_pct_vs_standard": post_delta_pct,

        "e2e_speedup_vs_standard": e2e_speedup,
        "e2e_delta_ms_vs_standard": e2e_delta_ms,
        "e2e_delta_pct_vs_standard": e2e_delta_pct,
    }


def print_summary_table(summaries):
    print("\n" + "=" * 190)
    print("MODEL + POSTPROCESS BENCHMARK SUMMARY")
    print("=" * 190)
    print(
        f"{'model':<32} "
        f"{'variant':<24} "
        f"{'mode':>7} "
        f"{'ref':>4} "
        f"{'frames':>6} "
        f"{'pre':>8} "
        f"{'infer':>8} "
        f"{'post':>9} "
        f"{'e2e':>9} "
        f"{'FPS':>8} "
        f"{'Power':>8} "
        f"{'FPS/W':>8} "
        f"{'J/frame':>9} "
        f"{'post_spd':>9} "
        f"{'e2e_spd':>9} "
        f"{'Δe2e%':>9}"
    )
    print("-" * 190)

    for s in summaries:
        print(
            f"{s['model']:<32.32} "
            f"{s['variant']:<24.24} "
            f"{s['mode']:>7} "
            f"{str(s['stages']):>4} "
            f"{s['frames']:>6} "
            f"{s['preprocess_avg_ms']:>8.2f} "
            f"{s['inference_avg_ms']:>8.2f} "
            f"{s['post_avg_ms']:>9.2f} "
            f"{s['e2e_avg_ms']:>9.2f} "
            f"{s['e2e_fps']:>8.2f} "
            f"{fmt_float(s['avg_power_w']):>8} "
            f"{fmt_float(s['fps_per_watt']):>8} "
            f"{fmt_float(s['energy_j_per_frame'], 4):>9} "
            f"{s['post_speedup_vs_standard']:>9.2f} "
            f"{s['e2e_speedup_vs_standard']:>9.2f} "
            f"{s['e2e_delta_pct_vs_standard']:>9.2f}"
        )

    print("=" * 190)


def save_csv(path, rows):
    if not rows or not path:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nCSV saved to: {path}")


def save_markdown_report(path, summaries):
    if not path:
        return

    headers = [
        "model", "variant", "mode", "stages", "frames",
        "preprocess_avg_ms", "inference_avg_ms", "post_avg_ms", "e2e_avg_ms",
        "e2e_fps", "avg_power_w", "fps_per_watt", "energy_j_per_frame",
        "post_speedup_vs_standard", "e2e_speedup_vs_standard",
    ]

    def row_to_md(row):
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                if np.isnan(v):
                    cells.append("N/A")
                elif h == "energy_j_per_frame":
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        return "| " + " | ".join(cells) + " |"

    lines = []
    lines.append("# Model + Postprocess Benchmark Report")
    lines.append("")
    lines.append("Lower `e2e_avg_ms` is better. Higher `e2e_fps`, `FPS/W`, and speedup are better.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for s in summaries:
        lines.append(row_to_md(s))
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved to: {path}")


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------
def benchmark_one_model(
    video_path,
    model_path,
    max_frames=100,
    warmup_frames=5,
    sample_power_every=10,
    disable_power=False,
):
    engine = PoseEstimator(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    variants = [
        ("standard", postprocess_standard_timed),
        ("fast_no_resize", postprocess_fast_timed),
        ("optimized_batch_k10", postprocess_optimized_batch_k10_timed),
        ("optimized_batch_k20", postprocess_optimized_batch_k20_timed),
    ]

    per_variant_rows = {name: [] for name, _ in variants}
    power_samples = []

    frame_idx = 0
    measured_frames = 0

    print("\n" + "-" * 90)
    print("Running benchmark")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Warmup frames: {warmup_frames}")
    print(f"Measured frames: {max_frames}")
    print(f"Power sampling: {'disabled' if disable_power else f'every {sample_power_every} measured frames'}")
    print("Drawing and video writing are disabled.")
    print("-" * 90)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Preprocess
        with Timer() as pre_t:
            input_tensor = engine.preprocess(frame)

        # Inference
        with Timer() as infer_t:
            raw_results = engine.model.run({"input": input_tensor})
            safe_sync()

        # Skip warmup frames from statistics
        if frame_idx <= warmup_frames:
            continue

        if measured_frames >= max_frames:
            break

        original_hw = frame.shape[:2]

        # Coarse power sampling once per measured frame interval.
        # Same power value is later used for all variants of this model.
        if not disable_power and sample_power_every > 0 and measured_frames % sample_power_every == 0:
            p = get_gpu_power_w()
            if not np.isnan(p):
                power_samples.append(p)

        for variant_name, fn in variants:
            _, _, timings = fn(raw_results, original_hw, engine)

            timings["frame_idx"] = frame_idx
            timings["model"] = Path(model_path).name
            timings["model_path"] = str(model_path)
            timings["variant"] = variant_name
            timings["preprocess_ms"] = pre_t.ms
            timings["inference_ms"] = infer_t.ms
            timings["e2e_ms"] = pre_t.ms + infer_t.ms + timings["post_ms"]

            per_variant_rows[variant_name].append(timings)

        measured_frames += 1

        if measured_frames % 10 == 0:
            print(f"Processed measured frames: {measured_frames}/{max_frames}")

    cap.release()

    if measured_frames == 0:
        raise RuntimeError("No frames were measured. Check video path or max_frames.")

    avg_power_w = nanmean(power_samples)

    baseline_rows = per_variant_rows["standard"]
    baseline_post_ms = mean([r["post_ms"] for r in baseline_rows])
    baseline_e2e_ms = mean([r["e2e_ms"] for r in baseline_rows])

    summaries = []
    detailed_rows = []

    for variant_name, _ in variants:
        rows = per_variant_rows[variant_name]

        baseline_post = None if variant_name == "standard" else baseline_post_ms
        baseline_e2e = None if variant_name == "standard" else baseline_e2e_ms

        summaries.append(
            summarize_variant(
                model_path=model_path,
                variant_name=variant_name,
                rows=rows,
                avg_power_w=avg_power_w,
                baseline_post_ms=baseline_post,
                baseline_e2e_ms=baseline_e2e,
            )
        )

        detailed_rows.extend(rows)

    return summaries, detailed_rows


def benchmark_models(
    video_path,
    model_paths,
    max_frames=100,
    warmup_frames=5,
    output_csv="model_postprocess_benchmark_summary.csv",
    detailed_csv=None,
    markdown_report="model_postprocess_benchmark_report.md",
    sample_power_every=10,
    disable_power=False,
):
    all_summaries = []
    all_detailed_rows = []

    for model_path in model_paths:
        summaries, detailed_rows = benchmark_one_model(
            video_path=video_path,
            model_path=model_path,
            max_frames=max_frames,
            warmup_frames=warmup_frames,
            sample_power_every=sample_power_every,
            disable_power=disable_power,
        )
        all_summaries.extend(summaries)
        all_detailed_rows.extend(detailed_rows)

    print_summary_table(all_summaries)

    save_csv(output_csv, all_summaries)

    if detailed_csv:
        save_csv(detailed_csv, all_detailed_rows)

    if markdown_report:
        save_markdown_report(markdown_report, all_summaries)

    return all_summaries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark multiple MIGraphX pose models and postprocess variants."
    )

    parser.add_argument(
        "--video",
        default="cctv_1280x720_24fps_3.mp4",
        help="Path to input video.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Single MIGraphX .mxr model path. Kept for backward compatibility.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="One or more MIGraphX .mxr model paths for comparison.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of measured frames per model.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup frames ignored in stats.",
    )

    parser.add_argument(
        "--csv",
        default="model_postprocess_benchmark_summary.csv",
        help="Output summary CSV path.",
    )

    parser.add_argument(
        "--detailed-csv",
        default=None,
        help="Optional per-frame CSV path.",
    )

    parser.add_argument(
        "--md",
        default="model_postprocess_benchmark_report.md",
        help="Output Markdown report path.",
    )

    parser.add_argument(
        "--power-every",
        type=int,
        default=10,
        help="Sample GPU power every N measured frames.",
    )

    parser.add_argument(
        "--no-power",
        action="store_true",
        help="Disable rocm-smi power reading.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        model_paths = ["pose_model1_fp16_ref1.mxr"]

    benchmark_models(
        video_path=args.video,
        model_paths=model_paths,
        max_frames=args.frames,
        warmup_frames=args.warmup,
        output_csv=args.csv,
        detailed_csv=args.detailed_csv,
        markdown_report=args.md,
        sample_power_every=args.power_every,
        disable_power=args.no_power,
    )
