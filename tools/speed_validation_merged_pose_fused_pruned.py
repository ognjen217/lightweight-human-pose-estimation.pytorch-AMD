#!/usr/bin/env python3
"""
Smoke/speed validation for merged pose+fused-pruned MXR.

This script only checks that the merged MXR runs, reports output shapes and
measures runtime. It is intentionally simple because the first goal is to
validate the monolithic MIGraphX path before stream integration.

Example:
  python tools/speed_validation_merged_pose_fused_pruned.py \
    --model models/merged_pose_fused_pruned/pose_fused_pruned_b2_....mxr \
    --video cctv_1280x720_24fps_2.mp4 \
    --frames 100 \
    --warmup 10 \
    --json outputs/speed_merged_pose_fused_pruned_b2.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def _shape_lens(shape_obj) -> List[int]:
    if hasattr(shape_obj, "lens"):
        return [int(x) for x in shape_obj.lens()]
    if hasattr(shape_obj, "lengths"):
        return [int(x) for x in shape_obj.lengths()]
    raise RuntimeError(f"Cannot read shape lens from {shape_obj}")


def _shape_type(shape_obj) -> str:
    if hasattr(shape_obj, "type"):
        return str(shape_obj.type())
    return str(shape_obj)


def _dtype_from_type(type_str: str):
    s = type_str.lower()
    if "half" in s or "float16" in s or "fp16" in s:
        return np.float16
    return np.float32


def _summ(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _load_model(path: str | Path):
    import migraphx  # type: ignore

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"[load] {path}")
    program = migraphx.load(str(path))
    shapes = program.get_parameter_shapes()
    if "input" in shapes:
        input_name = "input"
    else:
        input_name = list(shapes.keys())[0]
    lens = _shape_lens(shapes[input_name])
    typ = _shape_type(shapes[input_name])
    print(f"[input] {input_name} shape={lens} type={typ}")
    return program, input_name, lens, _dtype_from_type(typ)


def _read_frames(video: str | Path, n: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    frames: List[np.ndarray] = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read")
    return frames


def _preprocess(frame: np.ndarray, w: int, h: int, dtype) -> np.ndarray:
    resized = cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return np.ascontiguousarray(x.astype(dtype, copy=False))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    program, input_name, lens, dtype = _load_model(args.model)
    b, c, h, w = lens

    total_batches = int(args.warmup) + int(args.frames)
    frames = _read_frames(args.video, total_batches * b)

    pre_ms: List[float] = []
    run_ms: List[float] = []
    output_shapes = None
    output_dtypes = None
    cursor = 0

    for batch_idx in range(total_batches):
        batch_frames = frames[cursor : cursor + b]
        cursor += b
        if len(batch_frames) < b:
            break

        t0 = time.perf_counter()
        x = np.stack([_preprocess(f, w, h, dtype) for f in batch_frames], axis=0)
        x = np.ascontiguousarray(x)
        t1 = time.perf_counter()

        results = program.run({input_name: x})
        if not isinstance(results, (list, tuple)):
            results = list(results)
        outs = [np.asarray(r) for r in results]
        t2 = time.perf_counter()

        if batch_idx >= int(args.warmup):
            pre_ms.append((t1 - t0) * 1000.0)
            run_ms.append((t2 - t1) * 1000.0)
            if output_shapes is None:
                output_shapes = [list(o.shape) for o in outs]
                output_dtypes = [str(o.dtype) for o in outs]

        if args.print_every and (batch_idx + 1) % args.print_every == 0:
            print(f"batch {batch_idx + 1}/{total_batches}")

    measured_batches = len(run_ms)
    measured_frames = measured_batches * b
    pre_s = _summ(pre_ms)
    run_s = _summ(run_ms)

    summary = {
        "model": str(args.model),
        "video": str(args.video),
        "batch_size": int(b),
        "input_name": input_name,
        "input_shape": lens,
        "measured_batches": measured_batches,
        "measured_frames": measured_frames,
        "preprocess_ms": pre_s,
        "merged_mxr_ms": run_s,
        "merged_mxr_per_frame_ms_avg": run_s["avg"] / max(b, 1),
        "merged_mxr_fps": 1000.0 * b / max(run_s["avg"], 1e-9),
        "preprocess_plus_mxr_fps": 1000.0 * b / max(pre_s["avg"] + run_s["avg"], 1e-9),
        "output_shapes": output_shapes,
        "output_dtypes": output_dtypes,
    }

    print("\n" + "=" * 120)
    print("MERGED POSE + FUSED-PRUNED MXR SPEED SUMMARY")
    print("=" * 120)
    print(f"model:                 {summary['model']}")
    print(f"batch size:            {b}")
    print(f"input shape:           {lens}")
    print(f"measured frames:       {measured_frames}")
    print(f"preprocess avg:        {pre_s['avg']:.3f} ms/batch")
    print(f"merged mxr avg:        {run_s['avg']:.3f} ms/batch")
    print(f"merged mxr p95:        {run_s['p95']:.3f} ms/batch")
    print(f"mxr per-frame avg:     {summary['merged_mxr_per_frame_ms_avg']:.3f} ms/frame")
    print(f"mxr-only fps:          {summary['merged_mxr_fps']:.2f}")
    print(f"preprocess+mxr fps:    {summary['preprocess_plus_mxr_fps']:.2f}")
    print(f"output shapes:         {output_shapes}")
    print(f"output dtypes:         {output_dtypes}")
    print("=" * 120)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"JSON saved: {out}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", default="cctv_1280x720_24fps_2.mp4")
    ap.add_argument("--frames", type=int, default=100, help="Measured batches, not raw frames.")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup batches.")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
