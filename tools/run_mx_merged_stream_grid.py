#!/usr/bin/env python3
"""
Small grid search runner for the merged fused-pruned MIGraphX multi-camera stream
pipeline.

The script runs simulate_10_camera_stream.py across the key parameters that were
introduced/changed during the stream optimization work:

  * MIGraphX static batch size: B2 / B4 / B8
  * camera -> inference transport: old Queue payload vs shared-memory input slots
  * CPU process placement: unpinned vs pinned workers
  * batch timeout: either auto per batch size or explicit values

For every configuration it creates a dedicated run directory containing:

  * run.log
  * summary.json
  * detailed.csv
  * command.json

After each run it refreshes:

  * grid_summary.csv
  * grid_summary.json

These summary files are intentionally flat and report-friendly, so a later
Markdown/plots report can be generated from them without parsing logs again.

Example:

  python tools/run_mx_merged_stream_grid.py \
    --duration-s 250 \
    --warmup-s 30 \
    --camera-fps-values 24 \
    --batch-sizes 2 4 8 \
    --shared-input-options 0 1 \
    --pin-options 0 1

Recommended faster smoke run:

  python tools/run_mx_merged_stream_grid.py --duration-s 60 --warmup-s 10 --smoke
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODEL_TEMPLATE = (
    "models/merged_pose_fused_pruned_batchaware/"
    "pose_fused_pruned_batchaware_b{batch}_1080x1920_k20_m20_thr0p1_r6_separable.mxr"
)

DEFAULT_VIDEOS = [
    "cctv_1280x720_24fps_1.mp4",
    "cctv_1280x720_24fps_original.mp4",
    "cctv_1280x720_24fps_3.mp4",
    "cctv_1280x720_24fps_2.mp4",
]


@dataclass(frozen=True)
class GridConfig:
    batch_size: int
    batch_timeout_ms: float
    shared_input: int
    shared_input_dtype: str
    pinned: int
    pin_all_threads: int
    worker_threads: int
    post_workers: int
    infer_workers: int
    camera_fps: float
    max_pending_age_ms: float
    target_output_fps_per_camera: float

    @property
    def label(self) -> str:
        shared = "shmin" if self.shared_input else "queuein"
        pin = "pin" if self.pinned else "nopin"
        fps = f"fps{self.camera_fps:g}".replace(".", "p")
        timeout = f"to{self.batch_timeout_ms:g}".replace(".", "p")
        target = (
            f"_target{self.target_output_fps_per_camera:g}".replace(".", "p")
            if self.target_output_fps_per_camera > 0
            else ""
        )
        return (
            f"b{self.batch_size}_{shared}_{pin}_{fps}_{timeout}"
            f"_post{self.post_workers}_thr{self.worker_threads}{target}"
        )


def parse_int_options(values: Iterable[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            out.append(int(part))
    return out


def parse_float_options(values: Iterable[str]) -> List[float]:
    out: List[float] = []
    for v in values:
        for part in str(v).replace(",", " ").split():
            out.append(float(part))
    return out


def auto_timeout_ms(batch_size: int) -> float:
    # Values used in manual experiments: B2=4 ms, B4=8 ms, B8=12 ms.
    if batch_size <= 2:
        return 4.0
    if batch_size <= 4:
        return 8.0
    return 12.0


def unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_configs(args: argparse.Namespace) -> List[GridConfig]:
    batch_sizes = unique_preserve_order(args.batch_sizes)
    shared_opts = unique_preserve_order(args.shared_input_options)
    pin_opts = unique_preserve_order(args.pin_options)
    post_workers_opts = unique_preserve_order(args.post_workers_values)
    camera_fps_opts = unique_preserve_order(args.camera_fps_values)
    max_pending_opts = unique_preserve_order(args.max_pending_age_values)
    target_fps_opts = unique_preserve_order(args.target_output_fps_values)

    configs: List[GridConfig] = []
    for batch_size, shared_input, pinned, post_workers, camera_fps, max_pending, target_fps in itertools.product(
        batch_sizes,
        shared_opts,
        pin_opts,
        post_workers_opts,
        camera_fps_opts,
        max_pending_opts,
        target_fps_opts,
    ):
        timeout_values = args.batch_timeout_values
        if args.auto_batch_timeout:
            timeout_values = [auto_timeout_ms(batch_size)]

        for timeout_ms in timeout_values:
            cfg = GridConfig(
                batch_size=int(batch_size),
                batch_timeout_ms=float(timeout_ms),
                shared_input=int(shared_input),
                shared_input_dtype=args.shared_input_dtype,
                pinned=int(pinned),
                pin_all_threads=int(args.pin_all_threads),
                worker_threads=int(args.worker_threads),
                post_workers=int(post_workers),
                infer_workers=int(args.infer_workers),
                camera_fps=float(camera_fps),
                max_pending_age_ms=float(max_pending),
                target_output_fps_per_camera=float(target_fps),
            )
            configs.append(cfg)

    if args.smoke:
        # Keep one representative run per batch size: shared-input + pinned.
        configs = [
            c
            for c in configs
            if c.shared_input == 1 and c.pinned == 1 and c.post_workers == args.post_workers_values[0]
        ]

    if args.focused:
        # A compact matrix: all batches with the best-known optimized transport/placement,
        # plus B4 ablations for shared-input and pinning.
        keep = []
        for c in configs:
            best_path = c.shared_input == 1 and c.pinned == 1
            b4_ablation = c.batch_size == 4
            if best_path or b4_ablation:
                keep.append(c)
        configs = keep

    return configs


def run_command(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        log.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = d.get(key, default)
    return default if val is None else val


def find_stage(summary: Dict[str, Any], stage: str, worker_id: Optional[int] = None) -> Dict[str, Any]:
    for st in summary.get("stage_stats", []) or []:
        if st.get("stage") != stage:
            continue
        if worker_id is not None and st.get("worker_id") != worker_id:
            continue
        return st
    return {}


def sum_camera_stage(summary: Dict[str, Any], field: str) -> float:
    total = 0.0
    for st in summary.get("stage_stats", []) or []:
        if st.get("stage") == "camera_preprocess":
            try:
                total += float(st.get(field, 0.0) or 0.0)
            except Exception:
                pass
    return total


def summarize_run(
    cfg: GridConfig,
    run_dir: Path,
    returncode: int,
    started_ts: float,
    ended_ts: float,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    row: Dict[str, Any] = {
        "run_id": cfg.label,
        "status": "ok" if returncode == 0 and summary_path.exists() else "failed",
        "returncode": returncode,
        "run_dir": str(run_dir),
        "started_ts": started_ts,
        "ended_ts": ended_ts,
        "elapsed_wall_s_external": ended_ts - started_ts,
        **asdict(cfg),
    }

    if not summary_path.exists():
        return row

    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        row["status"] = "summary_parse_failed"
        row["error"] = repr(exc)
        return row

    inf = find_stage(summary, "inference", 0)
    gpu = summary.get("system_profile", {}) or {}

    attempted = sum_camera_stage(summary, "attempted")
    dropped = sum_camera_stage(summary, "dropped")
    replaced_before_infer = sum_camera_stage(summary, "replaced_before_infer")
    drop_rate = dropped / attempted if attempted > 0 else 0.0

    row.update(
        {
            "total_processed_frames": safe_get(summary, "total_processed_frames", 0),
            "raw_total_processed_frames": safe_get(summary, "raw_total_processed_frames", 0),
            "warmup_discarded_frames": safe_get(summary, "warmup_discarded_frames", 0),
            "wall_s": safe_get(summary, "wall_s", 0.0),
            "aggregate_output_fps": safe_get(summary, "aggregate_output_fps", 0.0),
            "avg_output_fps_per_camera": safe_get(summary, "avg_output_fps_per_camera", 0.0),
            "avg_preprocess_ms": safe_get(summary, "avg_preprocess_ms", 0.0),
            "avg_queue_pre_to_infer_ms": safe_get(summary, "avg_queue_pre_to_infer_ms", 0.0),
            "avg_inference_ms_per_frame": safe_get(summary, "avg_inference_ms", 0.0),
            "avg_decode_ms": safe_get(summary, "avg_decode_ms", 0.0),
            "avg_queue_infer_to_post_ms": safe_get(summary, "avg_queue_infer_to_post_ms", 0.0),
            "avg_post_ms": safe_get(summary, "avg_post_ms", 0.0),
            "p95_post_ms": safe_get(summary, "p95_post_ms", 0.0),
            "avg_e2e_ms": safe_get(summary, "avg_e2e_ms", 0.0),
            "p95_e2e_ms": safe_get(summary, "p95_e2e_ms", 0.0),
            "inference_processed": safe_get(inf, "processed", 0),
            "inference_batch_runs": safe_get(inf, "batch_runs", 0),
            "avg_real_batch_size": safe_get(inf, "avg_real_batch_size", 0.0),
            "p95_real_batch_size": safe_get(inf, "p95_real_batch_size", 0.0),
            "avg_inference_ms_per_batch": safe_get(inf, "avg_inference_ms", 0.0),
            "p95_inference_ms_per_batch": safe_get(inf, "p95_inference_ms", 0.0),
            "inference_replaced_before_post": safe_get(inf, "replaced_before_post", 0),
            "backpressure_skips": safe_get(inf, "skipped_due_backpressure", 0),
            "soft_backpressure_overrides": safe_get(inf, "soft_backpressure_overrides", 0),
            "backpressure_idle_loops": safe_get(inf, "backpressure_idle_loops", 0),
            "camera_attempted": attempted,
            "camera_dropped": dropped,
            "camera_replaced_before_infer": replaced_before_infer,
            "camera_drop_rate": drop_rate,
            "gpu_avg_pct": safe_get(gpu, "gpu_avg_pct", 0.0),
            "gpu_peak_pct": safe_get(gpu, "gpu_peak_pct", 0.0),
            "gpu_idle_pct": safe_get(gpu, "gpu_idle_pct", 0.0),
            "vram_avg_mb": safe_get(gpu, "vram_avg_mb", 0.0),
            "vram_peak_mb": safe_get(gpu, "vram_peak_mb", 0.0),
        }
    )
    return row


def write_table(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2))

    if not rows:
        csv_path.write_text("")
        return

    preferred = [
        "run_id",
        "status",
        "batch_size",
        "batch_timeout_ms",
        "shared_input",
        "shared_input_dtype",
        "pinned",
        "pin_all_threads",
        "worker_threads",
        "post_workers",
        "infer_workers",
        "camera_fps",
        "max_pending_age_ms",
        "target_output_fps_per_camera",
        "aggregate_output_fps",
        "avg_output_fps_per_camera",
        "raw_total_processed_frames",
        "total_processed_frames",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "avg_inference_ms_per_frame",
        "avg_inference_ms_per_batch",
        "p95_inference_ms_per_batch",
        "avg_real_batch_size",
        "camera_drop_rate",
        "camera_attempted",
        "camera_dropped",
        "avg_preprocess_ms",
        "avg_queue_pre_to_infer_ms",
        "avg_queue_infer_to_post_ms",
        "avg_post_ms",
        "p95_post_ms",
        "inference_replaced_before_post",
        "backpressure_skips",
        "gpu_avg_pct",
        "gpu_peak_pct",
        "gpu_idle_pct",
        "vram_avg_mb",
        "vram_peak_mb",
        "run_dir",
        "returncode",
    ]
    keys = []
    for k in preferred:
        if any(k in row for row in rows):
            keys.append(k)
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_command(args: argparse.Namespace, cfg: GridConfig, run_dir: Path) -> List[str]:
    model_path = args.model_template.format(batch=cfg.batch_size)
    cmd = [
        sys.executable,
        "-u",
        args.sim_script,
        "--model",
        model_path,
        "--variant",
        "mx_merged_pose_fused_pruned",
        "--num-cameras",
        str(args.num_cameras),
        "--duration-s",
        str(args.duration_s),
        "--frames-per-camera",
        "0",
        "--realtime",
        "--camera-fps",
        str(cfg.camera_fps),
        "--buffer-mode",
        "latest",
        "--backpressure-mode",
        args.backpressure_mode,
        "--max-pending-age-ms",
        str(cfg.max_pending_age_ms),
        "--infer-workers",
        str(cfg.infer_workers),
        "--post-workers",
        str(cfg.post_workers),
        "--migraphx-batch-size",
        str(cfg.batch_size),
        "--migraphx-batch-timeout-ms",
        str(cfg.batch_timeout_ms),
        "--target-width",
        str(args.target_width),
        "--target-height",
        str(args.target_height),
        "--stride",
        str(args.stride),
        "--torch-device",
        args.torch_device,
        "--warmup-s",
        str(args.warmup_s),
        "--worker-threads",
        str(cfg.worker_threads),
        "--profile-system",
        "--detailed-csv",
        str(run_dir / "detailed.csv"),
        "--summary-json",
        str(run_dir / "summary.json"),
    ]

    if args.preprocess_queue_size is not None:
        cmd += ["--preprocess-queue-size", str(args.preprocess_queue_size)]
    if args.postprocess_queue_size is not None:
        cmd += ["--postprocess-queue-size", str(args.postprocess_queue_size)]

    if cfg.shared_input:
        cmd += [
            "--shared-input-slots",
            str(max(args.num_cameras, args.shared_input_slots if args.shared_input_slots > 0 else args.num_cameras)),
            "--shared-input-dtype",
            cfg.shared_input_dtype,
        ]

    if cfg.pinned:
        cmd += [
            "--pin-cpus",
            "--pin-camera-base",
            str(args.pin_camera_base),
            "--pin-inference-base",
            str(args.pin_inference_base),
            "--pin-post-base",
            str(args.pin_post_base),
        ]
        if cfg.pin_all_threads:
            cmd += ["--pin-all-threads"]

    if cfg.target_output_fps_per_camera > 0:
        cmd += ["--target-output-fps-per-camera", str(cfg.target_output_fps_per_camera)]

    if args.extra_args:
        cmd += args.extra_args

    return cmd


def load_existing_rows(summary_csv: Path) -> List[Dict[str, Any]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sim-script", default="simulate_10_camera_stream.py")
    parser.add_argument("--model-template", default=DEFAULT_MODEL_TEMPLATE)
    parser.add_argument("--out-root", default="outputs/rocm721_stream_tests/mx_merged_stream_grid")

    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument(
        "--shared-input-options",
        nargs="+",
        type=int,
        choices=[0, 1],
        default=[0, 1],
        help="0 = old Queue tensor payload, 1 = shared-memory input slots",
    )
    parser.add_argument(
        "--pin-options",
        nargs="+",
        type=int,
        choices=[0, 1],
        default=[0, 1],
        help="0 = unpinned, 1 = --pin-cpus placement",
    )
    parser.add_argument("--post-workers-values", nargs="+", type=int, default=[3])
    parser.add_argument("--camera-fps-values", nargs="+", type=float, default=[24.0])
    parser.add_argument("--max-pending-age-values", nargs="+", type=float, default=[300.0])
    parser.add_argument("--target-output-fps-values", nargs="+", type=float, default=[0.0])

    parser.add_argument("--auto-batch-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-timeout-values", nargs="+", type=float, default=[4.0, 8.0, 12.0])

    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--num-cameras", type=int, default=10)
    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--torch-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--backpressure-mode", default="soft", choices=["off", "strict", "soft"])

    parser.add_argument("--shared-input-slots", type=int, default=0, help="0 means use --num-cameras when shared input is enabled")
    parser.add_argument("--shared-input-dtype", choices=["float32", "float16"], default="float32")

    parser.add_argument("--pin-camera-base", type=int, default=0)
    parser.add_argument("--pin-inference-base", type=int, default=10)
    parser.add_argument("--pin-post-base", type=int, default=12)
    parser.add_argument("--pin-all-threads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--worker-threads", type=int, default=1)

    parser.add_argument("--preprocess-queue-size", type=int, default=None)
    parser.add_argument("--postprocess-queue-size", type=int, default=None)

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="rerun even if summary.json already exists")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="only run optimized pinned+shared configs")
    parser.add_argument("--focused", action="store_true", help="optimized path for all batches plus B4 ablations")
    parser.add_argument("--limit", type=int, default=0, help="run only first N configs after expansion")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    if args.limit > 0:
        configs = configs[: args.limit]

    plan_path = out_root / "grid_plan.json"
    plan_path.write_text(json.dumps([asdict(c) | {"run_id": c.label} for c in configs], indent=2))

    print(f"[GRID] total configs: {len(configs)}")
    print(f"[GRID] out_root: {out_root}")
    print(f"[GRID] plan: {plan_path}")

    summary_csv = out_root / "grid_summary.csv"
    summary_json = out_root / "grid_summary.json"
    rows: List[Dict[str, Any]] = []

    for idx, cfg in enumerate(configs, start=1):
        run_dir = out_root / cfg.label
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        log_path = run_dir / "run.log"
        command_path = run_dir / "command.json"

        cmd = build_command(args, cfg, run_dir)
        model_path = Path(args.model_template.format(batch=cfg.batch_size))
        if not model_path.exists():
            print(f"[SKIP] missing model for {cfg.label}: {model_path}")
            row = {"run_id": cfg.label, "status": "missing_model", "run_dir": str(run_dir), **asdict(cfg)}
            rows.append(row)
            write_table(rows, summary_csv, summary_json)
            continue

        command_path.write_text(json.dumps({"cmd": cmd, "config": asdict(cfg)}, indent=2))
        print("=" * 100)
        print(f"[GRID] {idx}/{len(configs)} {cfg.label}")
        print("[CMD] " + " ".join(shlex.quote(x) for x in cmd))

        if args.dry_run:
            rows.append({"run_id": cfg.label, "status": "dry_run", "run_dir": str(run_dir), **asdict(cfg)})
            continue

        if args.resume and not args.force and summary_path.exists():
            print(f"[RESUME] summary exists, skipping: {summary_path}")
            row = summarize_run(cfg, run_dir, 0, time.time(), time.time())
            row["status"] = "ok_resumed"
            rows.append(row)
            write_table(rows, summary_csv, summary_json)
            continue

        started = time.time()
        rc = run_command(cmd, log_path)
        ended = time.time()
        row = summarize_run(cfg, run_dir, rc, started, ended)
        rows.append(row)
        write_table(rows, summary_csv, summary_json)

        print(f"[DONE] {cfg.label} status={row.get('status')} fps={row.get('aggregate_output_fps')} e2e={row.get('avg_e2e_ms')}")
        print(f"[GRID] partial summary: {summary_csv}")

    write_table(rows, summary_csv, summary_json)
    print("=" * 100)
    print(f"[GRID] complete: {summary_csv}")
    print(f"[GRID] complete: {summary_json}")

    ok_rows = [r for r in rows if str(r.get("status", "")).startswith("ok")]
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r.get("aggregate_output_fps") or 0.0))
        print(
            "[BEST FPS] "
            f"{best.get('run_id')} fps={best.get('aggregate_output_fps')} "
            f"p95_e2e={best.get('p95_e2e_ms')} gpu={best.get('gpu_avg_pct')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
