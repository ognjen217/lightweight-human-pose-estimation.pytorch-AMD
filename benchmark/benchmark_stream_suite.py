#!/usr/bin/env python3
"""
benchmark_stream_suite.py

Run, collect, analyze and plot multi-camera stream benchmark results.

Designed for simulate_10_camera_stream.py outputs:
  - --summary-json <run_summary.json>
  - --detailed-csv <run_detailed.csv>

Main workflows
--------------
1) Analyze already existing results:

    python benchmark/benchmark_stream_suite.py \
        --manifest benchmark/benchmark_suite_example.json \
        --report-dir benchmark_report \
        --analyze-only

2) Run a full suite and analyze afterwards:

    python benchmark/benchmark_stream_suite.py \
        --manifest benchmark/benchmark_suite_run_matrix.json \
        --report-dir benchmark_report_matrix

Manifest format
---------------
{
  "baseline": "strict_baseline",
  "ap_ar_guard": {
    "baseline_ap": 0.415,
    "baseline_ar": 0.473,
    "max_ap_drop": 0.005,
    "max_ar_drop": 0.005
  },
  "runs": [
    {
      "name": "strict_baseline",
      "summary_json": "outputs/strict_summary.json",
      "detailed_csv": "outputs/strict.csv",
      "command": "python simulate_10_camera_stream.py --model models/pose_model1_fp16_ref1.mxr ...",
      "ap": 0.415,
      "ar": 0.473
    },
    {
      "name": "manual_previous_run",
      "summary_inline": {
        "aggregate_output_fps": 15.27,
        "avg_inference_ms": 36.5,
        "avg_queue_infer_to_post_ms": 184,
        "avg_post_ms": 274.7,
        "avg_e2e_ms": 585,
        "p95_e2e_ms": 791
      }
    }
  ]
}

Optional power support
----------------------
Each run can include either:
  - "power_csv": "path/to/power_samples.csv"
  - "power_shell": "rocm-smi --showpower --csv ..."  (sampled while the run executes)

Power CSV should contain a power column named one of:
  watts, power_w, avg_power_w, gpu_power_w, socket_power_w

Outputs
-------
report-dir/
  absolute_metrics.csv
  relative_metrics.csv
  per_camera_metrics.csv
  absolute_metrics.json
  relative_metrics.json
  benchmark_report.md
  plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Matplotlib is imported lazily inside plotting so --no-plots works even on headless setups.


# -----------------------------
# Generic helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def pct_change(new: float, base: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return (new - base) / base * 100.0


def improvement_pct(metric: str, new: float, base: float) -> float:
    """Positive means better.

    For latency/energy/power counters lower is usually better.
    For FPS/fairness/frames higher is better.
    """
    lower_is_better_tokens = (
        "latency", "e2e", "p95", "post", "infer", "queue", "preprocess", "decode",
        "ms", "energy", "joule", "power", "waste", "skips", "replaced", "dropped",
    )
    lower_is_better = any(tok in metric.lower() for tok in lower_is_better_tokens)
    if lower_is_better:
        return pct_change(base, new)
    return pct_change(new, base)


def jain_fairness(values: Iterable[float]) -> float:
    vals = [safe_float(v) for v in values if safe_float(v) > 0]
    if not vals:
        return 0.0
    s = sum(vals)
    ss = sum(v * v for v in vals)
    return (s * s) / (len(vals) * ss) if ss > 0 else 0.0


def flatten_stage_stats(stage_stats: List[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    infer_processed = 0
    replaced_before_post = 0
    backpressure_skips = 0
    soft_overrides = 0
    throttle_skips = 0
    dropped_pre = 0
    replaced_before_infer = 0

    for st in stage_stats or []:
        stage = st.get("stage", "")
        if stage == "inference":
            infer_processed += safe_int(st.get("processed"))
            replaced_before_post += safe_int(st.get("replaced_before_post"))
            backpressure_skips += safe_int(st.get("skipped_due_backpressure"))
            soft_overrides += safe_int(st.get("soft_backpressure_overrides"))
            throttle_skips += safe_int(st.get("throttle_skips"))
        elif stage == "camera_preprocess":
            dropped_pre += safe_int(st.get("dropped"))
            replaced_before_infer += safe_int(st.get("replaced_before_infer"))

    out["infer_processed"] = float(infer_processed)
    out["replaced_before_post"] = float(replaced_before_post)
    out["backpressure_skips"] = float(backpressure_skips)
    out["soft_backpressure_overrides"] = float(soft_overrides)
    out["throttle_skips"] = float(throttle_skips)
    out["camera_dropped_or_replaced_before_infer"] = float(max(dropped_pre, replaced_before_infer))
    out["post_overwrite_rate_pct"] = (replaced_before_post / infer_processed * 100.0) if infer_processed else 0.0
    return out


def summarize_power_csv(path: Path, wall_s: float, frames: int) -> Dict[str, float]:
    if not path or not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

    power_col = None
    candidates = ["watts", "power_w", "avg_power_w", "gpu_power_w", "socket_power_w", "power"]
    normalized = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in normalized:
            power_col = normalized[cand]
            break
    if power_col is None:
        # fallback: first numeric column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        power_col = numeric_cols[0] if numeric_cols else None
    if power_col is None:
        return {}

    p = pd.to_numeric(df[power_col], errors="coerce").dropna()
    if p.empty:
        return {}

    avg_w = float(p.mean())
    peak_w = float(p.max())
    energy_j = avg_w * wall_s if wall_s > 0 else 0.0
    return {
        "avg_power_w": avg_w,
        "peak_power_w": peak_w,
        "energy_j": energy_j,
        "energy_per_frame_j": energy_j / frames if frames else 0.0,
    }


def power_sampler(power_shell: str, out_csv: Path, stop_event: threading.Event, interval_s: float = 1.0) -> None:
    """Sample a shell command that prints watts or CSV-like output.

    This is intentionally simple and robust: it extracts the first float from stdout.
    """
    ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_s", "watts", "raw"])
        writer.writeheader()
        t0 = time.perf_counter()
        while not stop_event.is_set():
            raw = ""
            watts = ""
            try:
                proc = subprocess.run(power_shell, shell=True, capture_output=True, text=True, timeout=max(1.0, interval_s))
                raw = (proc.stdout or proc.stderr or "").strip().replace("\n", " | ")
                tokens = raw.replace(",", " ").replace("W", " ").split()
                for tok in tokens:
                    try:
                        watts = str(float(tok))
                        break
                    except Exception:
                        continue
            except Exception as exc:
                raw = f"ERROR: {exc}"
            writer.writerow({"timestamp_s": time.perf_counter() - t0, "watts": watts, "raw": raw})
            f.flush()
            stop_event.wait(interval_s)


# -----------------------------
# Manifest model
# -----------------------------
@dataclass
class RunSpec:
    name: str
    command: str = ""
    summary_json: str = ""
    detailed_csv: str = ""
    summary_inline: Dict[str, Any] = field(default_factory=dict)
    ap: Optional[float] = None
    ar: Optional[float] = None
    power_csv: str = ""
    power_shell: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSpec":
        return cls(
            name=str(d["name"]),
            command=str(d.get("command", "")),
            summary_json=str(d.get("summary_json", "")),
            detailed_csv=str(d.get("detailed_csv", "")),
            summary_inline=dict(d.get("summary_inline", {}) or {}),
            ap=d.get("ap", None),
            ar=d.get("ar", None),
            power_csv=str(d.get("power_csv", "")),
            power_shell=str(d.get("power_shell", "")),
            tags=dict(d.get("tags", {}) or {}),
        )


@dataclass
class SuiteSpec:
    baseline: str
    runs: List[RunSpec]
    ap_ar_guard: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "SuiteSpec":
        data = read_json(path)
        runs = [RunSpec.from_dict(r) for r in data.get("runs", [])]
        if not runs:
            raise ValueError("Manifest must contain at least one run in 'runs'.")
        baseline = str(data.get("baseline", runs[0].name))
        return cls(baseline=baseline, runs=runs, ap_ar_guard=dict(data.get("ap_ar_guard", {}) or {}))


# -----------------------------
# Running benchmarks
# -----------------------------
def run_one(spec: RunSpec, report_dir: Path, force: bool = False, power_interval_s: float = 1.0) -> None:
    if not spec.command:
        return

    log_dir = report_dir / "logs"
    ensure_dir(log_dir)
    log_path = log_dir / f"{spec.name}.log"

    summary_path = Path(spec.summary_json) if spec.summary_json else None
    csv_path = Path(spec.detailed_csv) if spec.detailed_csv else None

    if not force and summary_path and summary_path.exists():
        print(f"[skip] {spec.name}: summary already exists: {summary_path}")
        return

    print(f"[run] {spec.name}")
    print(f"      {spec.command}")

    stop_power = threading.Event()
    sampler_thread = None
    if spec.power_shell:
        power_out = Path(spec.power_csv) if spec.power_csv else (report_dir / "power" / f"{spec.name}_power.csv")
        spec.power_csv = str(power_out)
        sampler_thread = threading.Thread(
            target=power_sampler,
            args=(spec.power_shell, power_out, stop_power, power_interval_s),
            daemon=True,
        )
        sampler_thread.start()

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            spec.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()

    stop_power.set()
    if sampler_thread:
        sampler_thread.join(timeout=5.0)

    dt = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"Run failed: {spec.name}, rc={rc}. See {log_path}")
    print(f"[done] {spec.name}: {dt:.1f}s, log={log_path}")


def run_suite(suite: SuiteSpec, report_dir: Path, force: bool = False, power_interval_s: float = 1.0) -> None:
    for spec in suite.runs:
        run_one(spec, report_dir=report_dir, force=force, power_interval_s=power_interval_s)


# -----------------------------
# Loading and analysis
# -----------------------------
def load_detailed_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def build_absolute_row(spec: RunSpec, guard: Dict[str, Any]) -> Dict[str, Any]:
    if spec.summary_inline:
        summary = dict(spec.summary_inline)
    elif spec.summary_json and Path(spec.summary_json).exists():
        summary = read_json(Path(spec.summary_json))
    else:
        raise FileNotFoundError(f"Missing summary for run '{spec.name}'. Provide summary_json or summary_inline.")

    detailed = load_detailed_csv(spec.detailed_csv)
    stage_counters = flatten_stage_stats(summary.get("stage_stats", []))
    per_camera = summary.get("per_camera", []) or []
    per_cam_fps = [safe_float(c.get("fps")) for c in per_camera]
    per_cam_e2e = [safe_float(c.get("avg_e2e_ms")) for c in per_camera]
    per_cam_frames = [safe_float(c.get("frames")) for c in per_camera]

    frames = safe_int(summary.get("total_processed_frames"), len(detailed) if not detailed.empty else 0)
    wall_s = safe_float(summary.get("wall_s"))

    row: Dict[str, Any] = {
        "run": spec.name,
        "variant": summary.get("variant", spec.tags.get("variant", "")),
        "registry_mode": summary.get("registry_mode", ""),
        "buffer_mode": summary.get("buffer_mode", ""),
        "backpressure_mode": summary.get("backpressure_mode", ""),
        "num_cameras": safe_int(summary.get("num_cameras")),
        "infer_workers": safe_int(summary.get("infer_workers")),
        "post_workers": safe_int(summary.get("post_workers")),
        "target_output_fps_per_camera": safe_float(summary.get("target_output_fps_per_camera")),
        "max_pending_age_ms": safe_float(summary.get("max_pending_age_ms")),
        "total_processed_frames": frames,
        "wall_s": wall_s,
        "aggregate_output_fps": safe_float(summary.get("aggregate_output_fps")),
        "avg_output_fps_per_camera": safe_float(summary.get("avg_output_fps_per_camera")),
        "avg_preprocess_ms": safe_float(summary.get("avg_preprocess_ms")),
        "avg_queue_pre_to_infer_ms": safe_float(summary.get("avg_queue_pre_to_infer_ms")),
        "avg_inference_ms": safe_float(summary.get("avg_inference_ms")),
        "avg_decode_ms": safe_float(summary.get("avg_decode_ms")),
        "avg_queue_infer_to_post_ms": safe_float(summary.get("avg_queue_infer_to_post_ms")),
        "avg_post_ms": safe_float(summary.get("avg_post_ms")),
        "avg_e2e_ms": safe_float(summary.get("avg_e2e_ms")),
        "p95_e2e_ms": safe_float(summary.get("p95_e2e_ms")),
        "p95_post_ms": safe_float(summary.get("p95_post_ms")),
        "ap": safe_float(spec.ap) if spec.ap is not None else safe_float(summary.get("ap")),
        "ar": safe_float(spec.ar) if spec.ar is not None else safe_float(summary.get("ar")),
        "per_camera_fps_min": min(per_cam_fps) if per_cam_fps else 0.0,
        "per_camera_fps_max": max(per_cam_fps) if per_cam_fps else 0.0,
        "per_camera_fps_spread": (max(per_cam_fps) - min(per_cam_fps)) if per_cam_fps else 0.0,
        "per_camera_fps_jain": jain_fairness(per_cam_fps),
        "per_camera_frames_jain": jain_fairness(per_cam_frames),
        "per_camera_avg_e2e_max_ms": max(per_cam_e2e) if per_cam_e2e else 0.0,
    }
    row.update(stage_counters)

    # Detailed CSV-derived percentiles, if available.
    if not detailed.empty:
        for col in ["e2e_ms", "post_ms", "inference_ms", "queue_infer_to_post_ms", "queue_pre_to_infer_ms"]:
            if col in detailed.columns:
                s = pd.to_numeric(detailed[col], errors="coerce").dropna()
                if not s.empty:
                    row[f"csv_p50_{col}"] = float(s.quantile(0.50))
                    row[f"csv_p90_{col}"] = float(s.quantile(0.90))
                    row[f"csv_p95_{col}"] = float(s.quantile(0.95))
                    row[f"csv_p99_{col}"] = float(s.quantile(0.99))

    # Power / energy.
    power = summarize_power_csv(Path(spec.power_csv), wall_s=wall_s, frames=frames) if spec.power_csv else {}
    row.update(power)

    # AP/AR guard.
    baseline_ap = guard.get("baseline_ap")
    baseline_ar = guard.get("baseline_ar")
    max_ap_drop = safe_float(guard.get("max_ap_drop"), 0.0)
    max_ar_drop = safe_float(guard.get("max_ar_drop"), 0.0)
    if baseline_ap is not None and row.get("ap", 0.0) > 0:
        row["ap_drop_vs_guard"] = safe_float(baseline_ap) - row["ap"]
        row["ap_guard_pass"] = bool(row["ap_drop_vs_guard"] <= max_ap_drop)
    if baseline_ar is not None and row.get("ar", 0.0) > 0:
        row["ar_drop_vs_guard"] = safe_float(baseline_ar) - row["ar"]
        row["ar_guard_pass"] = bool(row["ar_drop_vs_guard"] <= max_ar_drop)

    # Copy tags into columns.
    for k, v in spec.tags.items():
        row[f"tag_{k}"] = v
    return row


def build_per_camera_rows(spec: RunSpec) -> List[Dict[str, Any]]:
    if spec.summary_inline:
        return []
    if not spec.summary_json or not Path(spec.summary_json).exists():
        return []
    summary = read_json(Path(spec.summary_json))
    rows = []
    for c in summary.get("per_camera", []) or []:
        rows.append({
            "run": spec.name,
            "camera_id": safe_int(c.get("camera_id")),
            "source": c.get("source", ""),
            "frames": safe_int(c.get("frames")),
            "fps": safe_float(c.get("fps")),
            "avg_e2e_ms": safe_float(c.get("avg_e2e_ms")),
            "p95_e2e_ms": safe_float(c.get("p95_e2e_ms")),
            "avg_post_ms": safe_float(c.get("avg_post_ms")),
        })
    return rows


def build_relative_df(abs_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    if abs_df.empty:
        return pd.DataFrame()
    if baseline_name not in set(abs_df["run"]):
        baseline_name = str(abs_df.iloc[0]["run"])
    base = abs_df[abs_df["run"] == baseline_name].iloc[0].to_dict()

    metric_cols = [
        "aggregate_output_fps", "avg_output_fps_per_camera", "total_processed_frames",
        "avg_preprocess_ms", "avg_queue_pre_to_infer_ms", "avg_inference_ms", "avg_decode_ms",
        "avg_queue_infer_to_post_ms", "avg_post_ms", "avg_e2e_ms", "p95_e2e_ms", "p95_post_ms",
        "per_camera_fps_jain", "per_camera_fps_spread",
        "throttle_skips", "backpressure_skips", "soft_backpressure_overrides", "replaced_before_post",
        "post_overwrite_rate_pct", "avg_power_w", "peak_power_w", "energy_j", "energy_per_frame_j",
        "ap", "ar",
    ]
    existing = [c for c in metric_cols if c in abs_df.columns]
    rows = []
    for _, r in abs_df.iterrows():
        out: Dict[str, Any] = {"run": r["run"], "baseline": baseline_name}
        for c in existing:
            new = safe_float(r.get(c))
            b = safe_float(base.get(c))
            out[f"{c}_abs_delta"] = new - b
            out[f"{c}_pct_change"] = pct_change(new, b)
            out[f"{c}_improvement_pct"] = improvement_pct(c, new, b)
        rows.append(out)
    return pd.DataFrame(rows)


def analyze_suite(suite: SuiteSpec, report_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    abs_rows = [build_absolute_row(spec, suite.ap_ar_guard) for spec in suite.runs]
    cam_rows: List[Dict[str, Any]] = []
    for spec in suite.runs:
        cam_rows.extend(build_per_camera_rows(spec))

    abs_df = pd.DataFrame(abs_rows)
    rel_df = build_relative_df(abs_df, suite.baseline)
    cam_df = pd.DataFrame(cam_rows)

    ensure_dir(report_dir)
    abs_df.to_csv(report_dir / "absolute_metrics.csv", index=False)
    rel_df.to_csv(report_dir / "relative_metrics.csv", index=False)
    if not cam_df.empty:
        cam_df.to_csv(report_dir / "per_camera_metrics.csv", index=False)
    write_json(report_dir / "absolute_metrics.json", abs_df.to_dict(orient="records"))
    write_json(report_dir / "relative_metrics.json", rel_df.to_dict(orient="records"))
    return abs_df, rel_df, cam_df


# -----------------------------
# Plotting
# -----------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, rotate: int = 25) -> None:
    if df.empty or y not in df.columns:
        return
    plt = _plt()
    fig = plt.figure(figsize=(max(8, 1.2 * len(df)), 5))
    ax = fig.add_subplot(111)
    ax.bar(df[x].astype(str), pd.to_numeric(df[y], errors="coerce"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_absolute_metrics(abs_df: pd.DataFrame, plots_dir: Path) -> None:
    metrics = [
        ("aggregate_output_fps", "Aggregate output FPS", "FPS"),
        ("avg_output_fps_per_camera", "Average output FPS per camera", "FPS/camera"),
        ("avg_inference_ms", "Average inference latency", "ms"),
        ("avg_queue_infer_to_post_ms", "Average infer → post queue", "ms"),
        ("avg_post_ms", "Average postprocess latency", "ms"),
        ("avg_e2e_ms", "Average E2E latency", "ms"),
        ("p95_e2e_ms", "P95 E2E latency", "ms"),
        ("per_camera_fps_jain", "Per-camera FPS fairness / Jain index", "0-1"),
        ("energy_per_frame_j", "Energy per output frame", "J/frame"),
    ]
    for col, title, ylabel in metrics:
        save_bar(abs_df, "run", col, title, ylabel, plots_dir / f"{col}.png")


def plot_latency_breakdown(abs_df: pd.DataFrame, plots_dir: Path) -> None:
    cols = [
        "avg_preprocess_ms", "avg_queue_pre_to_infer_ms", "avg_inference_ms",
        "avg_decode_ms", "avg_queue_infer_to_post_ms", "avg_post_ms",
    ]
    cols = [c for c in cols if c in abs_df.columns]
    if not cols:
        return
    plt = _plt()
    fig = plt.figure(figsize=(max(9, 1.4 * len(abs_df)), 6))
    ax = fig.add_subplot(111)
    bottom = [0.0] * len(abs_df)
    xlabels = abs_df["run"].astype(str).tolist()
    for col in cols:
        vals = pd.to_numeric(abs_df[col], errors="coerce").fillna(0.0).tolist()
        ax.bar(xlabels, vals, bottom=bottom, label=col.replace("avg_", "").replace("_ms", ""))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Average latency breakdown")
    ax.set_ylabel("ms")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "avg_latency_breakdown.png", dpi=150)
    plt.close(fig)


def plot_relative_improvements(rel_df: pd.DataFrame, plots_dir: Path) -> None:
    wanted = [
        "aggregate_output_fps_improvement_pct",
        "avg_e2e_ms_improvement_pct",
        "p95_e2e_ms_improvement_pct",
        "avg_queue_infer_to_post_ms_improvement_pct",
        "avg_post_ms_improvement_pct",
        "avg_inference_ms_improvement_pct",
        "energy_per_frame_j_improvement_pct",
    ]
    cols = [c for c in wanted if c in rel_df.columns]
    if not cols or rel_df.empty:
        return
    plt = _plt()
    plot_df = rel_df.set_index("run")[cols].copy()
    fig = plt.figure(figsize=(max(10, 1.6 * len(plot_df)), 6))
    ax = fig.add_subplot(111)
    plot_df.plot(kind="bar", ax=ax)
    ax.axhline(0, linewidth=1)
    ax.set_title("Relative improvement vs baseline (positive is better)")
    ax.set_ylabel("improvement [%]")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "relative_improvements_vs_baseline.png", dpi=150)
    plt.close(fig)


def plot_scheduler_counters(abs_df: pd.DataFrame, plots_dir: Path) -> None:
    cols = ["throttle_skips", "backpressure_skips", "soft_backpressure_overrides", "replaced_before_post"]
    cols = [c for c in cols if c in abs_df.columns]
    if not cols:
        return
    plt = _plt()
    plot_df = abs_df.set_index("run")[cols].copy()
    fig = plt.figure(figsize=(max(10, 1.5 * len(plot_df)), 6))
    ax = fig.add_subplot(111)
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Scheduler counters / wasted work indicators")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "scheduler_counters.png", dpi=150)
    plt.close(fig)


def plot_per_camera(cam_df: pd.DataFrame, plots_dir: Path) -> None:
    if cam_df.empty:
        return
    plt = _plt()
    for metric, title, ylabel in [
        ("fps", "Per-camera FPS", "FPS"),
        ("avg_e2e_ms", "Per-camera average E2E latency", "ms"),
        ("p95_e2e_ms", "Per-camera P95 E2E latency", "ms"),
        ("frames", "Per-camera processed output frames", "frames"),
    ]:
        if metric not in cam_df.columns:
            continue
        pivot = cam_df.pivot_table(index="camera_id", columns="run", values=metric, aggfunc="mean")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("camera_id")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"per_camera_{metric}.png", dpi=150)
        plt.close(fig)


def plot_cdfs(suite: SuiteSpec, plots_dir: Path) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    any_data = False
    for spec in suite.runs:
        df = load_detailed_csv(spec.detailed_csv)
        if df.empty or "e2e_ms" not in df.columns:
            continue
        s = pd.to_numeric(df["e2e_ms"], errors="coerce").dropna().sort_values()
        if s.empty:
            continue
        y = [(i + 1) / len(s) for i in range(len(s))]
        ax.plot(s.tolist(), y, label=spec.name)
        any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_title("E2E latency CDF")
    ax.set_xlabel("E2E latency [ms]")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "e2e_latency_cdf.png", dpi=150)
    plt.close(fig)


def make_plots(suite: SuiteSpec, abs_df: pd.DataFrame, rel_df: pd.DataFrame, cam_df: pd.DataFrame, report_dir: Path) -> None:
    plots_dir = report_dir / "plots"
    ensure_dir(plots_dir)
    plot_absolute_metrics(abs_df, plots_dir)
    plot_latency_breakdown(abs_df, plots_dir)
    plot_relative_improvements(rel_df, plots_dir)
    plot_scheduler_counters(abs_df, plots_dir)
    plot_per_camera(cam_df, plots_dir)
    plot_cdfs(suite, plots_dir)


# -----------------------------
# Report generation
# -----------------------------
def fmt(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def top_relative_findings(rel_df: pd.DataFrame, baseline: str) -> List[str]:
    if rel_df.empty:
        return []
    rows = []
    for _, r in rel_df.iterrows():
        run = r["run"]
        if run == baseline:
            continue
        fps_imp = safe_float(r.get("aggregate_output_fps_improvement_pct"))
        e2e_imp = safe_float(r.get("avg_e2e_ms_improvement_pct"))
        p95_imp = safe_float(r.get("p95_e2e_ms_improvement_pct"))
        q_imp = safe_float(r.get("avg_queue_infer_to_post_ms_improvement_pct"))
        rows.append(
            f"- `{run}` vs `{baseline}`: FPS improvement {fps_imp:+.1f}%, "
            f"Avg E2E improvement {e2e_imp:+.1f}%, P95 E2E improvement {p95_imp:+.1f}%, "
            f"infer→post queue improvement {q_imp:+.1f}%."
        )
    return rows


def write_report(suite: SuiteSpec, abs_df: pd.DataFrame, rel_df: pd.DataFrame, cam_df: pd.DataFrame, report_dir: Path) -> None:
    path = report_dir / "benchmark_report.md"
    metrics = [
        "aggregate_output_fps", "avg_output_fps_per_camera", "avg_inference_ms",
        "avg_queue_infer_to_post_ms", "avg_post_ms", "avg_e2e_ms", "p95_e2e_ms",
        "per_camera_fps_jain", "throttle_skips", "backpressure_skips", "soft_backpressure_overrides",
        "replaced_before_post", "post_overwrite_rate_pct", "ap", "ar", "energy_per_frame_j",
    ]
    cols = ["run"] + [c for c in metrics if c in abs_df.columns]

    lines: List[str] = []
    lines.append("# Benchmark stream suite report")
    lines.append("")
    lines.append(f"Baseline: `{suite.baseline}`")
    lines.append("")
    lines.append("## Key findings")
    lines.extend(top_relative_findings(rel_df, suite.baseline) or ["- Not enough relative data to compute findings."])
    lines.append("")

    # AP/AR guard summary.
    if "ap_guard_pass" in abs_df.columns or "ar_guard_pass" in abs_df.columns:
        lines.append("## AP/AR guard")
        for _, r in abs_df.iterrows():
            ap_msg = ""
            ar_msg = ""
            if "ap_guard_pass" in abs_df.columns and not pd.isna(r.get("ap_guard_pass")):
                ap_msg = f"AP pass={bool(r.get('ap_guard_pass'))}, AP drop={fmt(r.get('ap_drop_vs_guard'))}"
            if "ar_guard_pass" in abs_df.columns and not pd.isna(r.get("ar_guard_pass")):
                ar_msg = f"AR pass={bool(r.get('ar_guard_pass'))}, AR drop={fmt(r.get('ar_drop_vs_guard'))}"
            if ap_msg or ar_msg:
                lines.append(f"- `{r['run']}`: {ap_msg} {ar_msg}".strip())
        lines.append("")

    lines.append("## Absolute metrics")
    lines.append(abs_df[cols].to_markdown(index=False))
    lines.append("")

    if not rel_df.empty:
        interesting_rel = [
            "run",
            "aggregate_output_fps_improvement_pct",
            "avg_e2e_ms_improvement_pct",
            "p95_e2e_ms_improvement_pct",
            "avg_queue_infer_to_post_ms_improvement_pct",
            "avg_post_ms_improvement_pct",
            "avg_inference_ms_improvement_pct",
            "energy_per_frame_j_improvement_pct",
        ]
        rel_cols = [c for c in interesting_rel if c in rel_df.columns]
        lines.append("## Relative improvements vs baseline")
        lines.append("Positive values mean better. For latency, queue, postprocess and energy metrics, lower raw value is treated as improvement.")
        lines.append("")
        lines.append(rel_df[rel_cols].to_markdown(index=False))
        lines.append("")

    if not cam_df.empty:
        lines.append("## Per-camera metrics")
        lines.append(cam_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Generated plot files")
    plots_dir = report_dir / "plots"
    if plots_dir.exists():
        for p in sorted(plots_dir.glob("*.png")):
            lines.append(f"- `plots/{p.name}`")
    else:
        lines.append("- No plots generated.")
    lines.append("")

    lines.append("## FPS improvements that should not affect AP/AR")
    lines.append("These are runtime/scheduling changes, not algorithmic postprocessing changes, so they should preserve AP/AR as long as the selected postprocess variant, threshold, NMS radius, max keypoints and resize path remain unchanged:")
    lines.append("- tune `--target-output-fps-per-camera` close to the real delivered per-camera FPS instead of setting it above capacity;")
    lines.append("- sweep `--max-pending-age-ms` around roughly 1.2–1.5× measured average postprocess latency;")
    lines.append("- reduce overwrite-before-post by avoiding too-aggressive soft override when post slots are already saturated;")
    lines.append("- keep AP/AR guard values in the manifest and fail/report any run that changes model/postprocess accuracy.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Example manifest creation
# -----------------------------
def write_example_manifests(out_dir: Path) -> None:
    ensure_dir(out_dir)
    example = {
        "baseline": "previous_strict_manual",
        "ap_ar_guard": {"baseline_ap": 0.415, "baseline_ar": 0.473, "max_ap_drop": 0.005, "max_ar_drop": 0.005},
        "runs": [
            {
                "name": "previous_strict_manual",
                "summary_inline": {
                    "aggregate_output_fps": 15.27,
                    "avg_inference_ms": 36.5,
                    "avg_queue_infer_to_post_ms": 184.0,
                    "avg_post_ms": 274.7,
                    "avg_e2e_ms": 585.0,
                    "p95_e2e_ms": 791.0,
                    "num_cameras": 10,
                    "infer_workers": 1,
                    "post_workers": 5,
                    "buffer_mode": "latest",
                    "backpressure_mode": "strict",
                },
                "ap": 0.415,
                "ar": 0.473,
            },
            {
                "name": "softbp_300_throttle_3p0",
                "summary_json": "outputs/stream_10cam_softbp_opt_summary.json",
                "detailed_csv": "outputs/stream_10cam_softbp_opt.csv",
                "ap": 0.415,
                "ar": 0.473,
            },
        ],
    }
    matrix = {
        "baseline": "strict_baseline",
        "ap_ar_guard": {"baseline_ap": 0.415, "baseline_ar": 0.473, "max_ap_drop": 0.005, "max_ar_drop": 0.005},
        "runs": [
            {
                "name": "strict_baseline",
                "summary_json": "outputs/stream_10cam_strict_summary.json",
                "detailed_csv": "outputs/stream_10cam_strict.csv",
                "command": "python simulate_10_camera_stream.py --model models/pose_model1_fp16_ref1.mxr --variant gpu_nms_fullres_two_process --num-cameras 10 --frames-per-camera 700 --buffer-mode latest --backpressure-mode strict --infer-workers 1 --post-workers 5 --detailed-csv outputs/stream_10cam_strict.csv --summary-json outputs/stream_10cam_strict_summary.json",
                "ap": 0.415,
                "ar": 0.473,
            },
            {
                "name": "softbp_300_throttle_3p0",
                "summary_json": "outputs/stream_10cam_softbp_300_thr3_summary.json",
                "detailed_csv": "outputs/stream_10cam_softbp_300_thr3.csv",
                "command": "python simulate_10_camera_stream.py --model models/pose_model1_fp16_ref1.mxr --variant gpu_nms_fullres_two_process --num-cameras 10 --frames-per-camera 700 --buffer-mode latest --backpressure-mode soft --max-pending-age-ms 300 --infer-workers 2 --post-workers 5 --target-output-fps-per-camera 3.0 --detailed-csv outputs/stream_10cam_softbp_300_thr3.csv --summary-json outputs/stream_10cam_softbp_300_thr3_summary.json",
                "ap": 0.415,
                "ar": 0.473,
            },
            {
                "name": "softbp_350_throttle_1p5",
                "summary_json": "outputs/stream_10cam_softbp_350_thr1p5_summary.json",
                "detailed_csv": "outputs/stream_10cam_softbp_350_thr1p5.csv",
                "command": "python simulate_10_camera_stream.py --model models/pose_model1_fp16_ref1.mxr --variant gpu_nms_fullres_two_process --num-cameras 10 --frames-per-camera 700 --buffer-mode latest --backpressure-mode soft --max-pending-age-ms 350 --infer-workers 2 --post-workers 5 --target-output-fps-per-camera 1.5 --detailed-csv outputs/stream_10cam_softbp_350_thr1p5.csv --summary-json outputs/stream_10cam_softbp_350_thr1p5_summary.json",
                "ap": 0.415,
                "ar": 0.473,
            },
        ],
    }
    write_json(out_dir / "benchmark_suite_example.json", example)
    write_json(out_dir / "benchmark_suite_run_matrix.json", matrix)
    print(f"Wrote examples to {out_dir}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run/analyze/plot stream benchmark suites.")
    p.add_argument("--manifest", default="benchmark_suite_example.json", help="Suite manifest JSON.")
    p.add_argument("--report-dir", default="benchmark_report", help="Output directory for report artifacts.")
    p.add_argument("--analyze-only", action="store_true", help="Do not execute commands; only analyze available outputs.")
    p.add_argument("--force", action="store_true", help="Re-run even if summary JSON already exists.")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation.")
    p.add_argument("--write-examples", action="store_true", help="Write example manifests and exit.")
    p.add_argument("--examples-dir", default=".", help="Directory for --write-examples outputs.")
    p.add_argument("--power-interval-s", type=float, default=1.0, help="Power sampling interval when power_shell is used.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_examples:
        write_example_manifests(Path(args.examples_dir))
        return 0

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Create one or run: "
            f"python {Path(__file__).name} --write-examples --examples-dir ."
        )

    suite = SuiteSpec.from_path(manifest_path)
    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)

    if not args.analyze_only:
        run_suite(suite, report_dir=report_dir, force=args.force, power_interval_s=args.power_interval_s)

    abs_df, rel_df, cam_df = analyze_suite(suite, report_dir=report_dir)
    if not args.no_plots:
        make_plots(suite, abs_df, rel_df, cam_df, report_dir=report_dir)
    write_report(suite, abs_df, rel_df, cam_df, report_dir=report_dir)

    print(f"\nSaved report artifacts to: {report_dir}")
    print(f"- {report_dir / 'absolute_metrics.csv'}")
    print(f"- {report_dir / 'relative_metrics.csv'}")
    print(f"- {report_dir / 'benchmark_report.md'}")
    if not args.no_plots:
        print(f"- {report_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
