"""Warmup filtering, summaries, and output writers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .utils import ensure_parent, json_safe, mean, percentile, safe_float


def apply_warmup_filter(rows: List[Dict[str, Any]], args) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    warmup_s = max(0.0, float(getattr(args, "warmup_s", 0.0) or 0.0))
    warmup_frames = max(0, int(getattr(args, "warmup_output_frames", 0) or 0))
    if not rows or (warmup_s <= 0.0 and warmup_frames <= 0):
        return rows, {
            "warmup_s": warmup_s,
            "warmup_output_frames": warmup_frames,
            "raw_total_processed_frames": len(rows),
            "warmup_discarded_frames": 0,
        }

    ordered = sorted(rows, key=lambda r: (safe_float(r.get("post_done_ts", 0.0)), int(r.get("camera_id", 0)), int(r.get("frame_id", 0))))
    filtered = ordered
    if warmup_s > 0.0:
        ts_values = [safe_float(r.get("post_done_ts", 0.0)) for r in ordered if safe_float(r.get("post_done_ts", 0.0)) > 0.0]
        if ts_values:
            cutoff = min(ts_values) + warmup_s
            filtered = [r for r in filtered if safe_float(r.get("post_done_ts", 0.0)) >= cutoff]
    if warmup_frames > 0:
        filtered = filtered[warmup_frames:]

    return filtered, {
        "warmup_s": warmup_s,
        "warmup_output_frames": warmup_frames,
        "raw_total_processed_frames": len(rows),
        "warmup_discarded_frames": len(rows) - len(filtered),
    }


def summarize(rows: List[Dict[str, Any]], stage_stats: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    total = len(rows)
    cameras = sorted({int(r["camera_id"]) for r in rows})

    summary: Dict[str, Any] = {
        "total_processed_frames": total,
        "wall_s": wall_s,
        "aggregate_output_fps": total / wall_s if wall_s > 0 else 0.0,
        "active_cameras": len(cameras),
        "avg_output_fps_per_camera": (total / wall_s / len(cameras)) if wall_s > 0 and cameras else 0.0,
        "avg_preprocess_ms": mean([r["preprocess_ms"] for r in rows]),
        "avg_queue_pre_to_infer_ms": mean([r["queue_pre_to_infer_ms"] for r in rows]),
        "avg_inference_ms": mean([r["inference_ms"] for r in rows]),
        "avg_decode_ms": mean([r["decode_ms"] for r in rows]),
        "avg_queue_infer_to_post_ms": mean([r["queue_infer_to_post_ms"] for r in rows]),
        "avg_post_ms": mean([r["post_ms"] for r in rows]),
        "avg_e2e_ms": mean([r["e2e_ms"] for r in rows]),
        "p95_e2e_ms": percentile([r["e2e_ms"] for r in rows], 95),
        "p95_post_ms": percentile([r["post_ms"] for r in rows], 95),
        "stage_stats": stage_stats,
    }

    by_cam: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cam[int(row["camera_id"])].append(row)

    summary["per_camera"] = []
    for cam_id in sorted(by_cam):
        cam_rows = by_cam[cam_id]
        summary["per_camera"].append(
            {
                "camera_id": cam_id,
                "frames": len(cam_rows),
                "fps": len(cam_rows) / wall_s if wall_s > 0 else 0.0,
                "avg_e2e_ms": mean([r["e2e_ms"] for r in cam_rows]),
                "p95_e2e_ms": percentile([r["e2e_ms"] for r in cam_rows], 95),
                "avg_post_ms": mean([r["post_ms"] for r in cam_rows]),
                "source": cam_rows[0].get("source", ""),
            }
        )

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 150)
    print("10-CAMERA STREAM SIMULATION SUMMARY")
    print("=" * 150)
    print(f"Processed frames:          {summary['total_processed_frames']}")
    if summary.get("warmup_discarded_frames", 0):
        print(
            f"Warmup discarded:         {summary['warmup_discarded_frames']} / "
            f"{summary.get('raw_total_processed_frames', summary['total_processed_frames'])} frames"
        )
    print(f"Wall time:                 {summary['wall_s']:.2f} s")
    print(f"Aggregate output FPS:      {summary['aggregate_output_fps']:.2f}")
    print(f"Avg output FPS / camera:   {summary['avg_output_fps_per_camera']:.2f}")
    print(f"Avg preprocess:            {summary['avg_preprocess_ms']:.2f} ms")
    print(f"Avg queue pre->infer:      {summary['avg_queue_pre_to_infer_ms']:.2f} ms")
    print(f"Avg inference:             {summary['avg_inference_ms']:.2f} ms")
    print(f"Avg decode:                {summary['avg_decode_ms']:.2f} ms")
    print(f"Avg queue infer->post:     {summary['avg_queue_infer_to_post_ms']:.2f} ms")
    print(f"Avg postprocess:           {summary['avg_post_ms']:.2f} ms")
    print(f"Avg E2E latency:           {summary['avg_e2e_ms']:.2f} ms")
    print(f"P95 E2E latency:           {summary['p95_e2e_ms']:.2f} ms")

    print("\nPer-camera output:")
    print(f"{'cam':>3} {'frames':>8} {'fps':>8} {'avg_e2e':>10} {'p95_e2e':>10} {'avg_post':>10} source")
    print("-" * 150)
    for cam in summary["per_camera"]:
        print(
            f"{cam['camera_id']:>3} {cam['frames']:>8} {cam['fps']:>8.2f} "
            f"{cam['avg_e2e_ms']:>10.2f} {cam['p95_e2e_ms']:>10.2f} "
            f"{cam['avg_post_ms']:>10.2f} {cam['source']}"
        )
    print("=" * 150)


def write_detailed_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path or not rows:
        return
    ensure_parent(path)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    preferred = [
        "camera_id",
        "frame_id",
        "source",
        "variant",
        "registry_mode",
        "post_worker_id",
        "preprocess_ms",
        "queue_pre_to_infer_ms",
        "inference_ms",
        "decode_ms",
        "queue_infer_to_post_ms",
        "post_ms",
        "e2e_ms",
        "post_done_ts",
        "num_poses",
        "num_keypoints",
    ]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detailed CSV saved: {path}")


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Summary JSON saved: {path}")
