#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import migraphx
import numpy as np

from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend, HipHeatmapTopKShape


PROFILE_KEYS = [
    "h2d_ms", "resize_ms", "vertical_ms", "horizontal_ms", "topk_ms",
    "d2h_scores_ms", "d2h_indices_ms", "device_total_ms", "total_ms",
]


def load_mxr(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return migraphx.load(path)


def run_mxr1(model, x: np.ndarray):
    out = model.run({"input": x})
    arrays = [np.array(o) for o in out]
    if len(arrays) != 2:
        raise RuntimeError(f"MXR1 expected 2 outputs, got {len(arrays)}")
    # Exported adapter returns heatmaps and pafs.  Detect by channel count.
    a, b = arrays
    return (a, b) if a.shape[1] == 18 else (b, a)


def preprocess_frame(frame: np.ndarray, w: int, h: int, dtype: str) -> np.ndarray:
    img = cv2.resize(frame, (w, h))
    img = (img.astype(np.float32) - 128.0) / 256.0
    img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img.astype(np.float16 if dtype == "float16" else np.float32))


def read_batch(video: str, batch_size: int, frame_index: int, stride: int, w: int, h: int, dtype: str):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    meta = []
    for b in range(batch_size):
        idx = frame_index + b * stride
        if frame_count > 0:
            idx %= frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {idx} from {video}")
        frames.append(preprocess_frame(frame, w, h, dtype))
        meta.append({"batch_index": b, "frame_index": idx})
    cap.release()
    return np.ascontiguousarray(np.stack(frames, axis=0)), meta


def avg(rows: List[Dict[str, float]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def p95(rows: List[Dict[str, float]], key: str) -> float:
    return float(np.percentile(np.asarray([float(r[key]) for r in rows], dtype=np.float64), 95)) if rows else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Profile HIP heatmap TopK internal stages on real frames")
    p.add_argument("--mxr1", required=True)
    p.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--run-frame-stride", type=int, default=24)
    p.add_argument("--batch-frame-stride", type=int, default=1)
    p.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--target-w", type=int, default=968)
    p.add_argument("--target-h", type=int, default=544)
    p.add_argument("--json", default="outputs/split_pipeline_compare/b4_hip_stage_profile_r3.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/b4_hip_stage_profile_r3.md")
    return p.parse_args()


def write_md(path: Path, payload: Dict[str, object]):
    lines = [
        "# HIP heatmap TopK stage profile", "",
        f"- backend: `{payload['backend']}`",
        f"- video: `{payload['video']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- runs: `{payload['runs']}`", "",
        "| stage | avg ms | p95 ms |", "|---|---:|---:|",
    ]
    for k in PROFILE_KEYS:
        lines.append(f"| {k} | {payload['profile_ms_avg'][k]:.4f} | {payload['profile_ms_p95'][k]:.4f} |")
    lines += ["", "## Context", "", "| metric | avg |", "|---|---:|"]
    for k, v in payload["context_avg"].items():
        lines.append(f"| {k} | {float(v):.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    mxr1 = load_mxr(args.mxr1)
    backend = HipHeatmapTopKBackend()
    shape = HipHeatmapTopKShape(batch=args.batch_size, channels=18)

    details = []
    profiles = []
    contexts = []
    for i in range(args.warmup + args.runs):
        measured = i >= args.warmup
        frame_index = args.frame_index + i * args.run_frame_stride
        t0 = time.perf_counter()
        x, meta = read_batch(args.video, args.batch_size, frame_index, args.batch_frame_stride, args.target_w, args.target_h, args.input_dtype)
        t1 = time.perf_counter()
        heatmaps, _ = run_mxr1(mxr1, x)
        t2 = time.perf_counter()
        scores, _indices, prof = backend.run_host_profile(heatmaps, shape)
        t3 = time.perf_counter()
        pd = prof.as_dict()
        ctx = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "mxr1_ms": (t2 - t1) * 1000.0,
            "python_profile_call_ms": (t3 - t2) * 1000.0,
            "valid_topk_count": float(np.sum(scores > -1.0e8)),
        }
        label = "warmup" if not measured else f"run {i - args.warmup}"
        print(f"[{label}] total={pd['total_ms']:.3f} device={pd['device_total_ms']:.3f} resize={pd['resize_ms']:.3f} vertical={pd['vertical_ms']:.3f} horizontal={pd['horizontal_ms']:.3f} topk={pd['topk_ms']:.3f} valid={int(ctx['valid_topk_count'])}")
        details.append({"measured": measured, "frame_index": frame_index, "batch_frames": meta, "profile_ms": pd, "context": ctx})
        if measured:
            profiles.append(pd)
            contexts.append(ctx)

    payload = {
        "backend": "hip_host_profile",
        "mxr1": args.mxr1,
        "video": args.video,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "warmup": args.warmup,
        "profile_ms_avg": {k: avg(profiles, k) for k in PROFILE_KEYS},
        "profile_ms_p95": {k: p95(profiles, k) for k in PROFILE_KEYS},
        "context_avg": {k: avg(contexts, k) for k in contexts[0].keys()} if contexts else {},
        "runs_detail": details,
    }
    jp = Path(args.json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2))
    print(f"[write] {jp}")
    mp = Path(args.markdown)
    write_md(mp, payload)
    print(f"[write] {mp}")


if __name__ == "__main__":
    main()
