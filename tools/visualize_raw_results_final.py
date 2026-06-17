#!/usr/bin/env python3
"""
Visualize raw multi-camera scaling results from simulate_camera_stream.py.

Expected input layout:
  outputs/raw_results_final/
    cam01_grid1x1_..._summary.json
    cam01_grid1x1_..._detailed.csv
    cam02_grid1x2_..._summary.json
    ...

Outputs:
  plots/*.png
  scaling_summary.csv
  scaling_summary.md

Usage:
  python tools/visualize_raw_results_final.py \
    --input-dir outputs/raw_results_final \
    --output-dir outputs/raw_results_final/plots

Optional:
  --dpi 180
  --window-s 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_RE = re.compile(r"cam(?P<cams>\d+)_(?P<grid>grid\d+x\d+).*_summary\.json$")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _stage(summary: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [s for s in summary.get("stage_stats", []) if s.get("stage") == name]


def _first_stage(summary: dict[str, Any], name: str) -> dict[str, Any]:
    stages = _stage(summary, name)
    return stages[0] if stages else {}


def _sum_stage(summary: dict[str, Any], name: str, key: str) -> float:
    return sum(_num(s.get(key)) for s in _stage(summary, name))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_summaries(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for path in sorted(input_dir.glob("cam*_summary.json")):
        match = SUMMARY_RE.search(path.name)
        if not match:
            continue

        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        inference = _first_stage(summary, "inference")
        cameras = _stage(summary, "camera_preprocess")
        post_stages = _stage(summary, "postprocess")
        per_camera = summary.get("per_camera", []) or []

        attempted = sum(_num(c.get("attempted")) for c in cameras)
        dropped = sum(_num(c.get("dropped")) for c in cameras)
        enqueued = sum(_num(c.get("enqueued")) for c in cameras)
        published = sum(_num(c.get("published")) for c in cameras)
        drop_ratio = dropped / attempted if attempted > 0 else 0.0
        processed_ratio = _num(summary.get("total_processed_frames")) / attempted if attempted > 0 else 0.0

        pc_fps = [_num(c.get("fps")) for c in per_camera]
        pc_e2e = [_num(c.get("avg_e2e_ms")) for c in per_camera]
        pc_post = [_num(c.get("avg_post_ms")) for c in per_camera]

        row = {
            "summary_file": str(path),
            "detailed_file": str(path).replace("_summary.json", "_detailed.csv"),
            "num_cameras": int(summary.get("num_cameras") or match.group("cams")),
            "grid": match.group("grid"),
            "total_processed_frames": int(summary.get("total_processed_frames", 0)),
            "wall_s": _num(summary.get("wall_s")),
            "aggregate_output_fps": _num(summary.get("aggregate_output_fps")),
            "avg_output_fps_per_camera": _num(summary.get("avg_output_fps_per_camera")),
            "avg_preprocess_ms": _num(summary.get("avg_preprocess_ms")),
            "avg_queue_pre_to_infer_ms": _num(summary.get("avg_queue_pre_to_infer_ms")),
            "avg_inference_ms": _num(summary.get("avg_inference_ms")),
            "avg_decode_ms": _num(summary.get("avg_decode_ms")),
            "avg_queue_infer_to_post_ms": _num(summary.get("avg_queue_infer_to_post_ms")),
            "avg_post_ms": _num(summary.get("avg_post_ms")),
            "p95_post_ms": _num(summary.get("p95_post_ms")),
            "avg_e2e_ms": _num(summary.get("avg_e2e_ms")),
            "p95_e2e_ms": _num(summary.get("p95_e2e_ms")),
            "avg_real_batch_size": _num(inference.get("avg_real_batch_size")),
            "p95_real_batch_size": _num(inference.get("p95_real_batch_size")),
            "configured_migraphx_batch_size": _num(inference.get("configured_migraphx_batch_size")),
            "batch_runs": _num(inference.get("batch_runs")),
            "shared_map_misses": _num(inference.get("shared_map_misses")),
            "skipped_due_backpressure": _num(inference.get("skipped_due_backpressure")),
            "stale_records_discarded_pre_batch": _num(summary.get("stale_records_discarded_pre_batch")),
            "camera_attempted": attempted,
            "camera_published": published,
            "camera_enqueued": enqueued,
            "camera_dropped": dropped,
            "camera_drop_ratio": drop_ratio,
            "processed_ratio_of_attempted": processed_ratio,
            "theoretical_input_fps": 24.0 * int(summary.get("num_cameras") or match.group("cams")),
            "output_efficiency_vs_24fps": _num(summary.get("aggregate_output_fps")) / (24.0 * int(summary.get("num_cameras") or match.group("cams"))),
            "per_camera_fps_min": min(pc_fps) if pc_fps else 0.0,
            "per_camera_fps_max": max(pc_fps) if pc_fps else 0.0,
            "per_camera_fps_std": pd.Series(pc_fps).std(ddof=0) if len(pc_fps) > 1 else 0.0,
            "per_camera_e2e_min": min(pc_e2e) if pc_e2e else 0.0,
            "per_camera_e2e_max": max(pc_e2e) if pc_e2e else 0.0,
            "per_camera_post_min": min(pc_post) if pc_post else 0.0,
            "per_camera_post_max": max(pc_post) if pc_post else 0.0,
            "post_workers": int(summary.get("post_workers", 0)),
            "infer_workers": int(summary.get("infer_workers", 0)),
            "backpressure_mode": summary.get("backpressure_mode", ""),
            "model": summary.get("model", ""),
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No cam*_summary.json files found in {input_dir}")

    df = pd.DataFrame(rows).sort_values("num_cameras").reset_index(drop=True)
    return df


def save_table(df: pd.DataFrame, output_dir: Path) -> None:
    summary_cols = [
        "num_cameras",
        "grid",
        "aggregate_output_fps",
        "avg_output_fps_per_camera",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "avg_post_ms",
        "p95_post_ms",
        "avg_inference_ms",
        "avg_real_batch_size",
        "camera_drop_ratio",
        "output_efficiency_vs_24fps",
    ]
    table = df[summary_cols].copy()
    table.to_csv(output_dir / "scaling_summary.csv", index=False)

    rounded = table.copy()
    for col in rounded.columns:
        if col not in {"num_cameras", "grid"}:
            rounded[col] = rounded[col].map(lambda x: f"{x:.3f}")
    (output_dir / "scaling_summary.md").write_text(rounded.to_markdown(index=False), encoding="utf-8")


def _finish_plot(path: Path, title: str, xlabel: str, ylabel: str, dpi: int) -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_scaling(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    x = df["num_cameras"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["aggregate_output_fps"], marker="o", label="Aggregate output FPS")
    plt.axhline(24, linestyle="--", linewidth=1, label="1 camera x 24 FPS")
    plt.axhline(48, linestyle="--", linewidth=1, label="2 cameras x 24 FPS")
    plt.axhline(96, linestyle="--", linewidth=1, label="4 cameras x 24 FPS")
    plt.legend()
    _finish_plot(output_dir / "01_aggregate_fps_scaling.png", "Aggregate throughput scaling", "Number of cameras", "FPS", dpi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["avg_output_fps_per_camera"], marker="o", label="Output FPS / camera")
    plt.axhline(24, linestyle="--", linewidth=1, label="Source FPS target")
    plt.axhline(12, linestyle="--", linewidth=1, label="12 FPS / camera")
    plt.axhline(10, linestyle="--", linewidth=1, label="10 FPS / camera")
    plt.legend()
    _finish_plot(output_dir / "02_fps_per_camera_scaling.png", "Per-camera throughput scaling", "Number of cameras", "FPS / camera", dpi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, 100.0 * df["output_efficiency_vs_24fps"], marker="o")
    _finish_plot(output_dir / "03_output_efficiency_vs_source.png", "Delivered output vs 24 FPS input demand", "Number of cameras", "Output / requested frames [%]", dpi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["camera_drop_ratio"] * 100.0, marker="o")
    _finish_plot(output_dir / "04_camera_drop_ratio.png", "Camera-side frame replacement/drop ratio", "Number of cameras", "Dropped / attempted [%]", dpi)


def plot_latency(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    x = df["num_cameras"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["avg_e2e_ms"], marker="o", label="Average E2E")
    plt.plot(x, df["p95_e2e_ms"], marker="o", label="P95 E2E")
    plt.legend()
    _finish_plot(output_dir / "05_e2e_latency_scaling.png", "End-to-end latency scaling", "Number of cameras", "Latency [ms]", dpi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["avg_post_ms"], marker="o", label="Average postprocess")
    plt.plot(x, df["p95_post_ms"], marker="o", label="P95 postprocess")
    plt.legend()
    _finish_plot(output_dir / "06_postprocess_latency_scaling.png", "Postprocess latency scaling", "Number of cameras", "Latency [ms]", dpi)

    plt.figure(figsize=(8, 5))
    plt.plot(x, df["avg_queue_pre_to_infer_ms"], marker="o", label="Queue pre→infer")
    plt.plot(x, df["avg_queue_infer_to_post_ms"], marker="o", label="Queue infer→post")
    plt.legend()
    _finish_plot(output_dir / "07_queue_latency_scaling.png", "Queue latency scaling", "Number of cameras", "Latency [ms]", dpi)


def plot_stage_breakdown(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    cols = [
        "avg_preprocess_ms",
        "avg_queue_pre_to_infer_ms",
        "avg_inference_ms",
        "avg_decode_ms",
        "avg_queue_infer_to_post_ms",
        "avg_post_ms",
    ]
    labels = {
        "avg_preprocess_ms": "preprocess",
        "avg_queue_pre_to_infer_ms": "queue pre→infer",
        "avg_inference_ms": "inference",
        "avg_decode_ms": "decode",
        "avg_queue_infer_to_post_ms": "queue infer→post",
        "avg_post_ms": "postprocess",
    }
    plot_df = df.set_index("num_cameras")[cols].rename(columns=labels)
    ax = plot_df.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_title("Average stage contribution to E2E latency")
    ax.set_xlabel("Number of cameras")
    ax.set_ylabel("Latency [ms]")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "08_stage_breakdown_stacked.png", dpi=dpi)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["num_cameras"], df["avg_real_batch_size"], marker="o", label="Average real batch")
    plt.plot(df["num_cameras"], df["p95_real_batch_size"], marker="o", label="P95 real batch")
    plt.axhline(df["configured_migraphx_batch_size"].max(), linestyle="--", linewidth=1, label="Configured batch size")
    plt.legend()
    _finish_plot(output_dir / "09_batch_fill_scaling.png", "MIGraphX real batch fill", "Number of cameras", "Batch size", dpi)


def load_detailed_csvs(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    detailed: dict[int, pd.DataFrame] = {}
    for _, row in df.iterrows():
        path = Path(row["detailed_file"])
        if not path.exists():
            continue
        try:
            d = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            continue
        if not d.empty:
            detailed[int(row["num_cameras"])] = d
    return detailed


def plot_detailed(detailed: dict[int, pd.DataFrame], output_dir: Path, dpi: int, window_s: float) -> None:
    if not detailed:
        print("[WARN] No detailed CSVs found; skipping detailed plots")
        return

    # Rolling/windowed output FPS over time.
    plt.figure(figsize=(10, 5))
    for cams, d in sorted(detailed.items()):
        if "post_done_ts" not in d.columns:
            continue
        ts = pd.to_numeric(d["post_done_ts"], errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            continue
        rel = ts - ts.iloc[0]
        bins = (rel // window_s).astype(int)
        fps = bins.value_counts().sort_index() / window_s
        plt.plot(fps.index * window_s, fps.values, marker="o", linewidth=1, label=f"{cams} cam")
    plt.legend()
    _finish_plot(output_dir / "10_windowed_output_fps.png", f"Windowed output FPS ({window_s:g}s bins)", "Time [s]", "FPS", dpi)

    # Cumulative FPS convergence.
    plt.figure(figsize=(10, 5))
    for cams, d in sorted(detailed.items()):
        if "post_done_ts" not in d.columns:
            continue
        ts = pd.to_numeric(d["post_done_ts"], errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            continue
        rel = ts - ts.iloc[0]
        rel = rel[rel > 0]
        cumulative = pd.Series(range(1, len(rel) + 1), index=rel.index) / rel.values
        plt.plot(rel.values, cumulative.values, linewidth=1, label=f"{cams} cam")
    plt.legend()
    _finish_plot(output_dir / "11_cumulative_output_fps.png", "Cumulative output FPS convergence", "Time [s]", "FPS", dpi)

    # E2E latency distribution.
    e2e_values = []
    labels = []
    for cams, d in sorted(detailed.items()):
        if "e2e_ms" in d.columns:
            vals = pd.to_numeric(d["e2e_ms"], errors="coerce").dropna()
            if not vals.empty:
                e2e_values.append(vals.values)
                labels.append(str(cams))
    if e2e_values:
        plt.figure(figsize=(9, 5))
        plt.boxplot(e2e_values, labels=labels, showfliers=False)
        _finish_plot(output_dir / "12_e2e_latency_distribution.png", "E2E latency distribution", "Number of cameras", "Latency [ms]", dpi)

    # Postprocess latency distribution.
    post_values = []
    labels = []
    for cams, d in sorted(detailed.items()):
        if "post_ms" in d.columns:
            vals = pd.to_numeric(d["post_ms"], errors="coerce").dropna()
            if not vals.empty:
                post_values.append(vals.values)
                labels.append(str(cams))
    if post_values:
        plt.figure(figsize=(9, 5))
        plt.boxplot(post_values, labels=labels, showfliers=False)
        _finish_plot(output_dir / "13_postprocess_latency_distribution.png", "Postprocess latency distribution", "Number of cameras", "Latency [ms]", dpi)

    # Per-camera FPS fairness from detailed CSV, independent of summary per_camera block.
    per_cam_rows = []
    for cams, d in sorted(detailed.items()):
        if "camera_id" not in d.columns or "post_done_ts" not in d.columns:
            continue
        duration = pd.to_numeric(d["post_done_ts"], errors="coerce").max() - pd.to_numeric(d["post_done_ts"], errors="coerce").min()
        if not duration or duration <= 0:
            continue
        counts = d.groupby("camera_id").size()
        for camera_id, count in counts.items():
            per_cam_rows.append({"num_cameras": cams, "camera_id": int(camera_id), "fps": count / duration})
    if per_cam_rows:
        pc = pd.DataFrame(per_cam_rows)
        pc.to_csv(output_dir / "per_camera_fps_from_detailed.csv", index=False)
        plt.figure(figsize=(11, 5))
        for cams, group in pc.groupby("num_cameras"):
            plt.plot(group["camera_id"], group["fps"], marker="o", linewidth=1, label=f"{cams} cam")
        plt.legend()
        _finish_plot(output_dir / "14_per_camera_fps_fairness.png", "Per-camera FPS fairness", "Camera ID", "FPS", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize multi-camera raw scaling results.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/raw_results_final"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--window-s", type=float, default=5.0)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summaries(input_dir)
    save_table(df, output_dir)
    plot_scaling(df, output_dir, args.dpi)
    plot_latency(df, output_dir, args.dpi)
    plot_stage_breakdown(df, output_dir, args.dpi)
    detailed = load_detailed_csvs(df)
    plot_detailed(detailed, output_dir, args.dpi, args.window_s)

    print("\nScaling summary:")
    print(df[[
        "num_cameras",
        "grid",
        "aggregate_output_fps",
        "avg_output_fps_per_camera",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "avg_post_ms",
        "avg_real_batch_size",
        "camera_drop_ratio",
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved plots and tables to: {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
