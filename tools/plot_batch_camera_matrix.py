#!/usr/bin/env python3
"""
Run, cache, normalize, and plot performance results for merged pose+fused-pruned
MIGraphX multi-camera simulations.

Default workflow:
    python tools/plot_batch_camera_matrix.py --repo-root .

This reads cached summaries from outputs/plot_cache/ and generates report-ready plots.
To execute missing benchmark runs first:
    python tools/plot_batch_camera_matrix.py --repo-root . --run-missing

Cache/output layout:
    outputs/plot_cache/
      merged_b4_t0ms_10cam_summary.json
      merged_b4_t0ms_10cam_detailed.csv
      summary_matrix_records.csv
      01_best_matrix_steady_fps.png
      ...

The old nested 5x4 matrix with 2x2 timeout cells is intentionally not generated
by default because it becomes too dense for reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


EPS = 1e-9
DEFAULT_MODEL_TEMPLATE = (
    "models/merged_pose_fused_pruned_batchaware/"
    "pose_fused_pruned_batchaware_b{batch}_1080x1920_k20_m20_thr0p1_r6_separable.mxr"
)
SUMMARY_RE = re.compile(
    r"merged_b(?P<batch>\d+)(?:_t(?P<timeout>\d+)ms)?_(?P<cameras>\d+)cam_summary\.json$"
)

BATCH_COLORS = {
    1: "tab:blue",
    2: "tab:orange",
    4: "tab:green",
    8: "tab:red",
    16: "tab:purple",
}
CAMERA_MARKERS = {
    1: "o",
    2: "s",
    4: "^",
    8: "D",
    10: "P",
    16: "X",
}

REPORT_PLOTS = [
    "01_best_matrix_steady_fps.png",
    "02_best_matrix_steady_fps_per_camera.png",
    "03_best_matrix_avg_e2e_ms.png",
    "04_best_matrix_p95_e2e_ms.png",
    "05_best_matrix_inference_per_frame_ms.png",
    "06_best_matrix_drop_ratio.png",
    "07_timeout_sensitivity_steady_fps.png",
    "08_timeout_sensitivity_p95_e2e_ms.png",
    "09_timeout_sensitivity_avg_e2e_ms.png",
    "10_scaling_best_steady_fps_vs_num_cameras.png",
    "11_scaling_best_fps_per_camera_vs_num_cameras.png",
    "12_scaling_best_p95_e2e_vs_num_cameras.png",
    "13_scaling_best_avg_e2e_vs_num_cameras.png",
    "14_scaling_best_inference_per_frame_vs_num_cameras.png",
    "15_pareto_steady_fps_vs_p95_e2e.png",
    "16_pareto_fps_per_camera_vs_p95_e2e.png",
    "17_pareto_steady_fps_vs_avg_e2e.png",
    "18_drop_ratio_vs_num_cameras.png",
    "19_fairness_fps_std_heatmap.png",
    "20_fairness_min_max_fps_range.png",
    "21_latency_breakdown_best_configs.png",
]


@dataclass(frozen=True)
class RunKey:
    num_cameras: int
    batch_size: int
    timeout_ms: int


def parse_csv_ints(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def nanmean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def nanstd(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(vals)) if vals else float("nan")


def parse_summary_filename(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    match = SUMMARY_RE.match(path.name)
    if not match:
        return None, None, None
    batch = int(match.group("batch"))
    timeout = int(match.group("timeout")) if match.group("timeout") is not None else None
    cameras = int(match.group("cameras"))
    return batch, timeout, cameras


def run_name(batch_size: int, timeout_ms: int, num_cameras: int) -> str:
    return f"merged_b{batch_size}_t{timeout_ms}ms_{num_cameras}cam"


def summary_path(output_dir: Path, key: RunKey) -> Path:
    return output_dir / f"{run_name(key.batch_size, key.timeout_ms, key.num_cameras)}_summary.json"


def detail_path(output_dir: Path, key: RunKey) -> Path:
    return output_dir / f"{run_name(key.batch_size, key.timeout_ms, key.num_cameras)}_detailed.csv"


def iter_plan(num_cameras: Sequence[int], batch_sizes: Sequence[int], timeouts: Sequence[int]) -> Iterable[RunKey]:
    for ncam in num_cameras:
        for batch in batch_sizes:
            timeout_values = [0] if int(batch) == 1 else list(timeouts)
            for timeout in timeout_values:
                yield RunKey(int(ncam), int(batch), int(timeout))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            print(f"[WARN] {path} is not a JSON object; skipping")
            return None
        return obj
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def get_inference_stage(data: Dict[str, Any]) -> Dict[str, Any]:
    for row in data.get("stage_stats", []) or []:
        if isinstance(row, dict) and row.get("stage") == "inference":
            return row
    return {}


def get_camera_stage_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        row for row in (data.get("stage_stats", []) or [])
        if isinstance(row, dict) and row.get("stage") == "camera_preprocess"
    ]


def infer_identity_from_data(path: Path, data: Dict[str, Any]) -> RunKey:
    file_batch, file_timeout, file_cameras = parse_summary_filename(path)
    inf = get_inference_stage(data)

    batch = file_batch
    if batch is None:
        batch = safe_int(
            data.get("migraphx_batch_size", inf.get("configured_migraphx_batch_size")),
            default=1,
        )

    timeout = file_timeout
    if timeout is None:
        timeout = int(round(safe_float(
            data.get("migraphx_batch_timeout_ms", inf.get("migraphx_batch_timeout_ms", 0.0)),
            default=0.0,
        )))

    cameras = file_cameras
    if cameras is None:
        cameras = safe_int(data.get("num_cameras", data.get("active_cameras")), default=0)
    if cameras <= 0:
        cameras = max(1, len(data.get("per_camera", []) or []))

    # Timeout is irrelevant for B=1; normalize it to 0.
    if int(batch) == 1:
        timeout = 0

    return RunKey(num_cameras=int(cameras), batch_size=int(batch), timeout_ms=int(timeout))


def normalize_summary(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    key = infer_identity_from_data(path, data)
    inf = get_inference_stage(data)
    cam_rows = get_camera_stage_rows(data)

    wall_s = safe_float(data.get("wall_s"), 0.0)
    warmup_s = safe_float(data.get("warmup_s"), 0.0)
    measured_s = max(EPS, wall_s - warmup_s)
    total_processed = safe_int(data.get("total_processed_frames"), 0)
    active_cameras = safe_int(data.get("active_cameras", key.num_cameras), key.num_cameras)
    active_cameras = max(1, active_cameras)

    attempted_total = sum(safe_int(row.get("attempted"), 0) for row in cam_rows)
    dropped_total = sum(safe_int(row.get("dropped"), 0) for row in cam_rows)
    replaced_before_infer_total = sum(safe_int(row.get("replaced_before_infer"), 0) for row in cam_rows)
    drop_ratio = dropped_total / attempted_total if attempted_total > 0 else float("nan")

    per_camera = data.get("per_camera", []) or []
    per_cam_fps = [safe_float(row.get("fps")) for row in per_camera if isinstance(row, dict)]
    per_cam_frames = [safe_float(row.get("frames")) for row in per_camera if isinstance(row, dict)]

    steady_fps = total_processed / measured_s
    steady_fps_per_camera = steady_fps / active_cameras

    avg_inf_batch = safe_float(inf.get("avg_inference_ms"))
    avg_real_batch = safe_float(inf.get("avg_real_batch_size"))
    if math.isfinite(avg_inf_batch) and math.isfinite(avg_real_batch) and avg_real_batch > 0:
        derived_inf_per_real_frame = avg_inf_batch / avg_real_batch
    else:
        derived_inf_per_real_frame = float("nan")

    return {
        "path": str(path),
        "model": data.get("model", ""),
        "variant": data.get("variant", ""),
        "num_cameras": key.num_cameras,
        "active_cameras": active_cameras,
        "batch_size": key.batch_size,
        "timeout_ms": key.timeout_ms,
        "total_processed_frames": total_processed,
        "raw_total_processed_frames": safe_int(data.get("raw_total_processed_frames"), 0),
        "warmup_discarded_frames": safe_int(data.get("warmup_discarded_frames"), 0),
        "wall_s": wall_s,
        "warmup_s": warmup_s,
        "measured_s": measured_s,
        "aggregate_output_fps_reported": safe_float(data.get("aggregate_output_fps")),
        "steady_fps": steady_fps,
        "steady_fps_per_camera": steady_fps_per_camera,
        "avg_output_fps_per_camera_reported": safe_float(data.get("avg_output_fps_per_camera")),
        "avg_e2e_ms": safe_float(data.get("avg_e2e_ms")),
        "p95_e2e_ms": safe_float(data.get("p95_e2e_ms")),
        "avg_preprocess_ms": safe_float(data.get("avg_preprocess_ms")),
        "avg_queue_pre_to_infer_ms": safe_float(data.get("avg_queue_pre_to_infer_ms")),
        "avg_inference_ms_per_frame": safe_float(data.get("avg_inference_ms")),
        "avg_decode_ms": safe_float(data.get("avg_decode_ms")),
        "avg_queue_infer_to_post_ms": safe_float(data.get("avg_queue_infer_to_post_ms")),
        "avg_post_ms": safe_float(data.get("avg_post_ms")),
        "p95_post_ms": safe_float(data.get("p95_post_ms")),
        "inference_processed": safe_int(inf.get("processed"), 0),
        "batch_runs": safe_int(inf.get("batch_runs"), 0),
        "avg_real_batch_size": safe_float(inf.get("avg_real_batch_size")),
        "p95_real_batch_size": safe_float(inf.get("p95_real_batch_size")),
        "configured_migraphx_batch_size": safe_int(inf.get("configured_migraphx_batch_size"), key.batch_size),
        "avg_inference_ms_per_batch_run": avg_inf_batch,
        "p95_inference_ms_per_batch_run": safe_float(inf.get("p95_inference_ms")),
        "derived_inference_ms_per_real_frame": derived_inf_per_real_frame,
        "skipped_due_backpressure": safe_int(inf.get("skipped_due_backpressure"), 0),
        "replaced_before_post": safe_int(inf.get("replaced_before_post"), 0),
        "attempted_total": attempted_total,
        "dropped_total": dropped_total,
        "replaced_before_infer_total": replaced_before_infer_total,
        "drop_ratio": drop_ratio,
        "per_camera_fps_mean": nanmean(per_cam_fps),
        "per_camera_fps_min": min(per_cam_fps) if per_cam_fps else float("nan"),
        "per_camera_fps_max": max(per_cam_fps) if per_cam_fps else float("nan"),
        "per_camera_fps_std": nanstd(per_cam_fps),
        "per_camera_frames_min": min(per_cam_frames) if per_cam_frames else float("nan"),
        "per_camera_frames_max": max(per_cam_frames) if per_cam_frames else float("nan"),
        "per_camera_frames_std": nanstd(per_cam_frames),
    }


def scan_cached_records(output_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_keys: Dict[RunKey, Dict[str, Any]] = {}
    for path in sorted(output_dir.glob("*summary.json")):
        data = load_json(path)
        if data is None:
            continue
        rec = normalize_summary(path, data)
        key = RunKey(rec["num_cameras"], rec["batch_size"], rec["timeout_ms"])
        # If duplicate summaries exist, prefer the one with more processed frames.
        old = seen_keys.get(key)
        if old is None or rec.get("total_processed_frames", 0) > old.get("total_processed_frames", 0):
            seen_keys[key] = rec
    records = list(seen_keys.values())
    records.sort(key=lambda r: (r["num_cameras"], r["batch_size"], r["timeout_ms"], r["path"]))
    return records


def write_records(records: List[Dict[str, Any]], output_dir: Path) -> Tuple[Path, Path]:
    csv_path = output_dir / "summary_matrix_records.csv"
    json_path = output_dir / "summary_matrix_records.json"

    if records:
        fields: List[str] = []
        for rec in records:
            for key in rec.keys():
                if key not in fields:
                    fields.append(key)
    else:
        fields = [
            "path", "model", "variant", "num_cameras", "batch_size", "timeout_ms",
            "total_processed_frames", "wall_s", "warmup_s", "steady_fps",
            "steady_fps_per_camera", "avg_e2e_ms", "p95_e2e_ms",
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return csv_path, json_path


def metric_value(rec: Dict[str, Any], metric: str) -> float:
    return safe_float(rec.get(metric))


def records_for_group(records: List[Dict[str, Any]], ncam: int, batch: int) -> List[Dict[str, Any]]:
    return [r for r in records if int(r["num_cameras"]) == int(ncam) and int(r["batch_size"]) == int(batch)]


def best_record(
    records: List[Dict[str, Any]],
    metric: str,
    maximize: bool,
    ncam: Optional[int] = None,
    batch: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    subset = records
    if ncam is not None:
        subset = [r for r in subset if int(r["num_cameras"]) == int(ncam)]
    if batch is not None:
        subset = [r for r in subset if int(r["batch_size"]) == int(batch)]
    subset = [r for r in subset if math.isfinite(metric_value(r, metric))]
    if not subset:
        return None
    return max(subset, key=lambda r: metric_value(r, metric)) if maximize else min(subset, key=lambda r: metric_value(r, metric))


def best_records_grid(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    metric: str,
    maximize: bool,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ncam in num_cameras:
        for batch in batch_sizes:
            rec = best_record(records, metric, maximize, ncam=ncam, batch=batch)
            if rec is not None:
                out[(int(ncam), int(batch))] = rec
    return out


def finite_values(records: List[Dict[str, Any]], metric: str) -> List[float]:
    return [metric_value(r, metric) for r in records if math.isfinite(metric_value(r, metric))]


def setup_output_path(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def savefig(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {output_path}")


def plot_best_heatmap(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    metric: str,
    maximize: bool,
    title: str,
    output_path: Path,
    fmt: str = "{:.1f}",
    cmap_name: Optional[str] = None,
    percent: bool = False,
) -> None:
    grid = best_records_grid(records, num_cameras, batch_sizes, metric, maximize)
    values = np.full((len(num_cameras), len(batch_sizes)), np.nan)
    labels: List[List[str]] = [["missing" for _ in batch_sizes] for _ in num_cameras]

    for i, ncam in enumerate(num_cameras):
        for j, batch in enumerate(batch_sizes):
            rec = grid.get((int(ncam), int(batch)))
            if rec is None:
                continue
            val = metric_value(rec, metric)
            if percent:
                val *= 100.0
            values[i, j] = val
            t = "-" if int(batch) == 1 else str(int(rec["timeout_ms"]))
            labels[i][j] = f"{fmt.format(val)}\nt={t}"

    valid = values[np.isfinite(values)]
    if valid.size == 0:
        print(f"[WARN] No data for {output_path.name}")
        return

    cmap = plt.get_cmap(cmap_name or ("viridis" if maximize else "viridis_r")).copy()
    cmap.set_bad("#e6e6e6")

    fig_w = max(7.5, 1.45 * len(batch_sizes) + 3.2)
    fig_h = max(5.2, 0.85 * len(num_cameras) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([f"B{b}" for b in batch_sizes])
    ax.set_yticks(np.arange(len(num_cameras)))
    ax.set_yticklabels([str(c) for c in num_cameras])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Number of cameras")
    ax.set_title(title)

    for i in range(len(num_cameras)):
        for j in range(len(batch_sizes)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
    cbar.set_label(metric.replace("_", " "))
    ax.text(
        0.5, -0.13,
        "Each cell shows the best timeout for the plotted metric. B=1 ignores timeout.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    savefig(fig, output_path)


def plot_timeout_sensitivity(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    timeouts: Sequence[int],
    metric: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    batches = [b for b in batch_sizes if int(b) != 1]
    if not batches:
        return
    fig, axes = plt.subplots(1, len(batches), figsize=(5.0 * len(batches), 4.6), sharey=False, constrained_layout=True)
    if len(batches) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        for ncam in num_cameras:
            xs: List[int] = []
            ys: List[float] = []
            for timeout in timeouts:
                subset = [
                    r for r in records
                    if int(r["num_cameras"]) == int(ncam)
                    and int(r["batch_size"]) == int(batch)
                    and int(r["timeout_ms"]) == int(timeout)
                ]
                if not subset:
                    continue
                y = metric_value(subset[0], metric)
                if math.isfinite(y):
                    xs.append(int(timeout))
                    ys.append(y)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.7, label=f"C{ncam}")
        ax.set_title(f"B{batch}")
        ax.set_xlabel("Timeout [ms]")
        ax.set_xticks(list(timeouts))
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.02))
    savefig(fig, output_path)


def plot_scaling_best(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    metric: str,
    maximize: bool,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    for batch in batch_sizes:
        xs: List[int] = []
        ys: List[float] = []
        labels: List[str] = []
        for ncam in num_cameras:
            rec = best_record(records, metric, maximize, ncam=ncam, batch=batch)
            if rec is None:
                continue
            val = metric_value(rec, metric)
            if math.isfinite(val):
                xs.append(int(ncam))
                ys.append(val)
                labels.append("-" if int(batch) == 1 else str(int(rec["timeout_ms"])))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"B{batch}", color=BATCH_COLORS.get(int(batch), None))
            for x, y, t in zip(xs, ys, labels):
                ax.annotate(f"t{t}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Number of cameras")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(list(num_cameras))
    ax.grid(True, alpha=0.35)
    ax.legend(title="Batch")
    savefig(fig, output_path)


def is_pareto_efficient(rows: List[Dict[str, Any]], latency_col: str, throughput_col: str) -> List[bool]:
    vals = []
    for r in rows:
        x = metric_value(r, latency_col)
        y = metric_value(r, throughput_col)
        vals.append((x, y))
    flags: List[bool] = []
    for i, (xi, yi) in enumerate(vals):
        if not (math.isfinite(xi) and math.isfinite(yi)):
            flags.append(False)
            continue
        dominated = False
        for j, (xj, yj) in enumerate(vals):
            if i == j or not (math.isfinite(xj) and math.isfinite(yj)):
                continue
            # x lower is better, y higher is better.
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated = True
                break
        flags.append(not dominated)
    return flags


def plot_pareto(
    records: List[Dict[str, Any]],
    latency_col: str,
    throughput_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    rows = [r for r in records if math.isfinite(metric_value(r, latency_col)) and math.isfinite(metric_value(r, throughput_col))]
    if not rows:
        print(f"[WARN] No data for {output_path.name}")
        return
    pareto = is_pareto_efficient(rows, latency_col, throughput_col)

    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    for r, is_eff in zip(rows, pareto):
        batch = int(r["batch_size"])
        ncam = int(r["num_cameras"])
        x = metric_value(r, latency_col)
        y = metric_value(r, throughput_col)
        ax.scatter(
            x,
            y,
            s=95 if is_eff else 55,
            marker=CAMERA_MARKERS.get(ncam, "o"),
            color=BATCH_COLORS.get(batch, None),
            edgecolor="black" if is_eff else "none",
            linewidth=1.2 if is_eff else 0.0,
            alpha=0.9 if is_eff else 0.55,
        )
        if is_eff:
            ax.annotate(
                f"C{ncam} B{batch} t{int(r['timeout_ms'])}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)

    batch_handles = [Patch(color=BATCH_COLORS.get(int(b), "gray"), label=f"B{b}") for b in sorted({int(r["batch_size"]) for r in rows})]
    cam_handles = [
        Line2D([0], [0], marker=CAMERA_MARKERS.get(int(c), "o"), color="black", linestyle="None", label=f"C{c}")
        for c in sorted({int(r["num_cameras"]) for r in rows})
    ]
    leg1 = ax.legend(handles=batch_handles, title="Batch", loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=cam_handles, title="Cameras", loc="lower right")
    savefig(fig, output_path)


def plot_drop_ratio(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    for batch in batch_sizes:
        xs: List[int] = []
        ys: List[float] = []
        for ncam in num_cameras:
            rec = best_record(records, "steady_fps", True, ncam=ncam, batch=batch)
            if rec is None:
                continue
            y = metric_value(rec, "drop_ratio")
            if math.isfinite(y):
                xs.append(int(ncam))
                ys.append(100.0 * y)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"B{batch}", color=BATCH_COLORS.get(int(batch), None))
    ax.set_xlabel("Number of cameras")
    ax.set_ylabel("Dropped/replaced before infer [%]")
    ax.set_title("Drop ratio vs number of cameras — best timeout selected by steady FPS")
    ax.set_xticks(list(num_cameras))
    ax.grid(True, alpha=0.35)
    ax.legend(title="Batch")
    savefig(fig, output_path)


def plot_fairness_range(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    for batch in batch_sizes:
        xs: List[int] = []
        means: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        for ncam in num_cameras:
            rec = best_record(records, "steady_fps", True, ncam=ncam, batch=batch)
            if rec is None:
                continue
            mean = metric_value(rec, "per_camera_fps_mean")
            mn = metric_value(rec, "per_camera_fps_min")
            mx = metric_value(rec, "per_camera_fps_max")
            if math.isfinite(mean) and math.isfinite(mn) and math.isfinite(mx):
                xs.append(int(ncam))
                means.append(mean)
                lower.append(max(0.0, mean - mn))
                upper.append(max(0.0, mx - mean))
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=np.array([lower, upper]),
                marker="o",
                capsize=4,
                linewidth=2.0,
                label=f"B{batch}",
                color=BATCH_COLORS.get(int(batch), None),
            )
    ax.set_xlabel("Number of cameras")
    ax.set_ylabel("Per-camera output FPS")
    ax.set_title("Per-camera fairness range — mean with min/max error bars")
    ax.set_xticks(list(num_cameras))
    ax.grid(True, alpha=0.35)
    ax.legend(title="Batch")
    savefig(fig, output_path)


def plot_latency_breakdown(records: List[Dict[str, Any]], batch_sizes: Sequence[int], output_path: Path) -> None:
    selected: List[Dict[str, Any]] = []
    seen_labels = set()
    for batch in batch_sizes:
        rec = best_record(records, "steady_fps", True, batch=batch)
        if rec is not None:
            label = f"C{int(rec['num_cameras'])} B{int(batch)} t{int(rec['timeout_ms'])}"
            if label not in seen_labels:
                selected.append(rec)
                seen_labels.add(label)
    low_lat = best_record(records, "p95_e2e_ms", False)
    if low_lat is not None:
        label = f"C{int(low_lat['num_cameras'])} B{int(low_lat['batch_size'])} t{int(low_lat['timeout_ms'])}"
        if label not in seen_labels:
            selected.append(low_lat)
            seen_labels.add(label)

    if not selected:
        print(f"[WARN] No data for {output_path.name}")
        return

    components = [
        ("preprocess", "avg_preprocess_ms"),
        ("queue pre→infer", "avg_queue_pre_to_infer_ms"),
        ("inference", "avg_inference_ms_per_frame"),
        ("decode", "avg_decode_ms"),
        ("queue infer→post", "avg_queue_infer_to_post_ms"),
        ("postprocess", "avg_post_ms"),
    ]
    labels = [f"C{int(r['num_cameras'])} B{int(r['batch_size'])} t{int(r['timeout_ms'])}" for r in selected]
    y = np.arange(len(selected))
    left = np.zeros(len(selected))

    fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.55 * len(selected) + 2.2)), constrained_layout=True)
    for comp_name, col in components:
        vals = np.array([max(0.0, metric_value(r, col)) if math.isfinite(metric_value(r, col)) else 0.0 for r in selected])
        ax.barh(y, vals, left=left, label=comp_name)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average latency component [ms]")
    ax.set_title("Latency breakdown for representative best configurations")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    savefig(fig, output_path)


def plot_nested_matrix_optional(
    records: List[Dict[str, Any]],
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    timeouts: Sequence[int],
    output_dir: Path,
) -> None:
    # Kept only as a diagnostic. It is intentionally simpler than the original
    # and reserves a dedicated colorbar axis so it cannot overlap data.
    metric = "steady_fps"
    vals = finite_values(records, metric)
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(13, 2.15 * len(num_cameras) + 1.8), constrained_layout=True)
    gs = fig.add_gridspec(len(num_cameras), len(batch_sizes) + 1, width_ratios=[1] * len(batch_sizes) + [0.08])
    for i, ncam in enumerate(num_cameras):
        for j, batch in enumerate(batch_sizes):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"B={batch}")
            if j == 0:
                ax.set_ylabel(f"{ncam} cam", rotation=0, ha="right", va="center", labelpad=28)
            layout = [[0, 1], [2, 4]]
            for rr, row in enumerate(layout):
                for cc, timeout in enumerate(row):
                    actual_timeout = 0 if int(batch) == 1 else int(timeout)
                    subset = [
                        r for r in records
                        if int(r["num_cameras"]) == int(ncam)
                        and int(r["batch_size"]) == int(batch)
                        and int(r["timeout_ms"]) == actual_timeout
                    ]
                    val = metric_value(subset[0], metric) if subset else float("nan")
                    x, y = cc, 1 - rr
                    face = "#e6e6e6" if not math.isfinite(val) else cmap(norm(val))
                    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
                    text = "missing" if not math.isfinite(val) else f"t{timeout}\n{val:.1f}"
                    if int(batch) == 1 and timeout != 0:
                        text = ""
                    ax.text(x + 0.5, y + 0.5, text, ha="center", va="center", fontsize=7)
            ax.add_patch(Rectangle((0, 0), 2, 2, facecolor="none", edgecolor="black", linewidth=0.8))
    cax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Post-warmup aggregate FPS")
    fig.suptitle("Diagnostic nested matrix: inner 2×2 = timeout 0/1/2/4 ms")
    savefig(fig, output_dir / "diagnostic_nested_matrix_steady_fps.png")


def model_path_from_template(repo_root: Path, template: str, batch_size: int) -> Path:
    p = repo_root / template.format(batch=batch_size)
    return p


def build_command(args: argparse.Namespace, key: RunKey, summary: Path, detailed: Path) -> List[str]:
    repo_root = Path(args.repo_root).resolve()
    model_path = model_path_from_template(repo_root, args.model_template, key.batch_size)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for B{key.batch_size}: {model_path}")

    cmd = [
        sys.executable,
        "simulate_camera_stream.py",
        "--model", str(model_path),
        "--variant", args.variant,
        "--migraphx-batch-size", str(key.batch_size),
        "--migraphx-batch-timeout-ms", str(key.timeout_ms),
        "--num-cameras", str(key.num_cameras),
        "--frames-per-camera", "0",
        "--duration-s", str(args.duration_s),
        "--realtime",
        "--camera-fps", str(args.camera_fps),
        "--buffer-mode", args.buffer_mode,
        "--backpressure-mode", args.backpressure_mode,
        "--infer-workers", str(args.infer_workers),
        "--post-workers", str(args.post_workers),
        "--shared-input-slots", str(key.num_cameras),
        "--shared-input-dtype", args.shared_input_dtype,
        "--require-gpu",
        "--warmup-s", str(args.warmup_s),
        "--summary-json", str(summary),
        "--detailed-csv", str(detailed),
    ]

    if args.pin_cpus:
        cmd.extend([
            "--pin-cpus",
            "--pin-camera-base", str(args.pin_camera_base),
            "--pin-inference-base", str(args.pin_inference_base),
            "--pin-post-base", str(args.pin_post_base),
        ])
        if args.report_affinity:
            cmd.append("--report-affinity")

    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def looks_valid_summary(path: Path) -> bool:
    data = load_json(path)
    if data is None:
        return False
    return safe_int(data.get("total_processed_frames"), -1) >= 0 and safe_float(data.get("wall_s"), 0.0) > 0


def run_missing(args: argparse.Namespace, output_dir: Path, plan: Sequence[RunKey]) -> None:
    repo_root = Path(args.repo_root).resolve()
    for key in plan:
        s_path = summary_path(output_dir, key)
        d_path = detail_path(output_dir, key)
        if s_path.exists() and not args.force_rerun and looks_valid_summary(s_path):
            print(f"[SKIP] existing B{key.batch_size} C{key.num_cameras} t{key.timeout_ms}: {s_path.name}")
            continue
        cmd = build_command(args, key, s_path, d_path)
        print(f"[RUN] B{key.batch_size} C{key.num_cameras} t{key.timeout_ms} -> {s_path.name}")
        print("      " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        s_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, cwd=str(repo_root), check=True)


def generate_plots(
    records: List[Dict[str, Any]],
    output_dir: Path,
    num_cameras: Sequence[int],
    batch_sizes: Sequence[int],
    timeouts: Sequence[int],
    include_nested_matrix: bool = False,
) -> None:
    if not records:
        print("[WARN] No records loaded. CSV/JSON were written, but no plots can be generated.")
        return

    plot_best_heatmap(records, num_cameras, batch_sizes, "steady_fps", True,
                      "Best post-warmup aggregate FPS by cameras and batch size",
                      setup_output_path(output_dir, "01_best_matrix_steady_fps.png"), "{:.1f}")
    plot_best_heatmap(records, num_cameras, batch_sizes, "steady_fps_per_camera", True,
                      "Best post-warmup FPS per camera by cameras and batch size",
                      setup_output_path(output_dir, "02_best_matrix_steady_fps_per_camera.png"), "{:.2f}")
    plot_best_heatmap(records, num_cameras, batch_sizes, "avg_e2e_ms", False,
                      "Best average E2E latency by cameras and batch size — lower is better",
                      setup_output_path(output_dir, "03_best_matrix_avg_e2e_ms.png"), "{:.0f}")
    plot_best_heatmap(records, num_cameras, batch_sizes, "p95_e2e_ms", False,
                      "Best P95 E2E latency by cameras and batch size — lower is better",
                      setup_output_path(output_dir, "04_best_matrix_p95_e2e_ms.png"), "{:.0f}")
    plot_best_heatmap(records, num_cameras, batch_sizes, "avg_inference_ms_per_frame", False,
                      "Best average inference per frame by cameras and batch size — lower is better",
                      setup_output_path(output_dir, "05_best_matrix_inference_per_frame_ms.png"), "{:.1f}")
    plot_best_heatmap(records, num_cameras, batch_sizes, "drop_ratio", False,
                      "Best drop ratio by cameras and batch size — lower is better",
                      setup_output_path(output_dir, "06_best_matrix_drop_ratio.png"), "{:.1f}", percent=True)

    plot_timeout_sensitivity(records, num_cameras, batch_sizes, timeouts, "steady_fps",
                             "Timeout sensitivity: post-warmup aggregate FPS",
                             "Post-warmup aggregate FPS",
                             setup_output_path(output_dir, "07_timeout_sensitivity_steady_fps.png"))
    plot_timeout_sensitivity(records, num_cameras, batch_sizes, timeouts, "p95_e2e_ms",
                             "Timeout sensitivity: P95 E2E latency",
                             "P95 E2E latency [ms]",
                             setup_output_path(output_dir, "08_timeout_sensitivity_p95_e2e_ms.png"))
    plot_timeout_sensitivity(records, num_cameras, batch_sizes, timeouts, "avg_e2e_ms",
                             "Timeout sensitivity: average E2E latency",
                             "Average E2E latency [ms]",
                             setup_output_path(output_dir, "09_timeout_sensitivity_avg_e2e_ms.png"))

    plot_scaling_best(records, num_cameras, batch_sizes, "steady_fps", True,
                      "Scaling: best post-warmup aggregate FPS vs number of cameras",
                      "Post-warmup aggregate FPS",
                      setup_output_path(output_dir, "10_scaling_best_steady_fps_vs_num_cameras.png"))
    plot_scaling_best(records, num_cameras, batch_sizes, "steady_fps_per_camera", True,
                      "Scaling: best post-warmup FPS per camera vs number of cameras",
                      "Post-warmup FPS / camera",
                      setup_output_path(output_dir, "11_scaling_best_fps_per_camera_vs_num_cameras.png"))
    plot_scaling_best(records, num_cameras, batch_sizes, "p95_e2e_ms", False,
                      "Scaling: best P95 E2E latency vs number of cameras — lower is better",
                      "P95 E2E latency [ms]",
                      setup_output_path(output_dir, "12_scaling_best_p95_e2e_vs_num_cameras.png"))
    plot_scaling_best(records, num_cameras, batch_sizes, "avg_e2e_ms", False,
                      "Scaling: best average E2E latency vs number of cameras — lower is better",
                      "Average E2E latency [ms]",
                      setup_output_path(output_dir, "13_scaling_best_avg_e2e_vs_num_cameras.png"))
    plot_scaling_best(records, num_cameras, batch_sizes, "avg_inference_ms_per_frame", False,
                      "Scaling: best average inference per frame vs number of cameras — lower is better",
                      "Average inference / frame [ms]",
                      setup_output_path(output_dir, "14_scaling_best_inference_per_frame_vs_num_cameras.png"))

    plot_pareto(records, "p95_e2e_ms", "steady_fps",
                "Pareto: aggregate FPS vs P95 E2E latency",
                "P95 E2E latency [ms] — lower is better",
                "Post-warmup aggregate FPS — higher is better",
                setup_output_path(output_dir, "15_pareto_steady_fps_vs_p95_e2e.png"))
    plot_pareto(records, "p95_e2e_ms", "steady_fps_per_camera",
                "Pareto: FPS per camera vs P95 E2E latency",
                "P95 E2E latency [ms] — lower is better",
                "Post-warmup FPS / camera — higher is better",
                setup_output_path(output_dir, "16_pareto_fps_per_camera_vs_p95_e2e.png"))
    plot_pareto(records, "avg_e2e_ms", "steady_fps",
                "Pareto: aggregate FPS vs average E2E latency",
                "Average E2E latency [ms] — lower is better",
                "Post-warmup aggregate FPS — higher is better",
                setup_output_path(output_dir, "17_pareto_steady_fps_vs_avg_e2e.png"))

    plot_drop_ratio(records, num_cameras, batch_sizes, setup_output_path(output_dir, "18_drop_ratio_vs_num_cameras.png"))
    plot_best_heatmap(records, num_cameras, batch_sizes, "per_camera_fps_std", False,
                      "Per-camera FPS standard deviation — lower is fairer",
                      setup_output_path(output_dir, "19_fairness_fps_std_heatmap.png"), "{:.3f}")
    plot_fairness_range(records, num_cameras, batch_sizes, setup_output_path(output_dir, "20_fairness_min_max_fps_range.png"))
    plot_latency_breakdown(records, batch_sizes, setup_output_path(output_dir, "21_latency_breakdown_best_configs.png"))

    if include_nested_matrix:
        plot_nested_matrix_optional(records, num_cameras, batch_sizes, timeouts, output_dir)


def print_summary(records: List[Dict[str, Any]]) -> None:
    print(f"[INFO] loaded normalized records: {len(records)}")
    best_fps = best_record(records, "steady_fps", True)
    best_p95 = best_record(records, "p95_e2e_ms", False)

    def describe(rec: Optional[Dict[str, Any]], metric: str) -> str:
        if rec is None:
            return "n/a"
        return (
            f"C{int(rec['num_cameras'])} B{int(rec['batch_size'])} t{int(rec['timeout_ms'])} "
            f"= {metric_value(rec, metric):.3f}"
        )

    print(f"[BEST] steady_fps: {describe(best_fps, 'steady_fps')}")
    print(f"[BEST] p95_e2e_ms: {describe(best_p95, 'p95_e2e_ms')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/cache/plot report-ready batch-size x camera-count MIGraphX stream simulation results."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root containing simulate_camera_stream.py")
    parser.add_argument("--output-dir", "--cache-dir", dest="output_dir", default="outputs/plot_cache",
                        help="Directory for cached summaries/details and generated plots")
    parser.add_argument("--run-missing", action="store_true", help="Run missing simulations before plotting")
    parser.add_argument("--force-rerun", "--overwrite", dest="force_rerun", action="store_true",
                        help="Re-run simulations even when valid summaries already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--num-cameras", default="1,2,4,8,10", help="Comma-separated camera counts")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated batch/model sizes")
    parser.add_argument("--timeouts", default="0,1,2,4", help="Comma-separated timeout values in ms")
    parser.add_argument("--duration-s", type=float, default=130.0)
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--camera-fps", type=float, default=24.0)
    parser.add_argument("--variant", default="mx_merged_pose_fused_pruned")
    parser.add_argument("--model-template", default=DEFAULT_MODEL_TEMPLATE)
    parser.add_argument("--buffer-mode", default="latest")
    parser.add_argument("--backpressure-mode", default="soft", choices=["off", "strict", "soft"])
    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=4)
    parser.add_argument("--shared-input-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--pin-cpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-camera-base", type=int, default=2)
    parser.add_argument("--pin-inference-base", type=int, default=20)
    parser.add_argument("--pin-post-base", type=int, default=24)
    parser.add_argument("--report-affinity", action="store_true")
    parser.add_argument("--include-nested-matrix", action="store_true", help="Also generate old diagnostic nested matrix")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args appended to simulate_camera_stream.py after --extra-args")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_cameras = parse_csv_ints(args.num_cameras)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    timeouts = parse_csv_ints(args.timeouts)
    plan = list(iter_plan(num_cameras, batch_sizes, timeouts))

    print(f"[CONFIG] repo_root={repo_root}")
    print(f"[CONFIG] output_dir={output_dir}")
    print(f"[CONFIG] planned combinations={len(plan)} (B=1 uses t0 only)")

    if args.run_missing:
        run_missing(args, output_dir, plan)

    records = scan_cached_records(output_dir)
    csv_path, json_path = write_records(records, output_dir)
    print(f"[WRITE] {csv_path}")
    print(f"[WRITE] {json_path}")

    print_summary(records)
    generate_plots(records, output_dir, num_cameras, batch_sizes, timeouts, args.include_nested_matrix)
    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
