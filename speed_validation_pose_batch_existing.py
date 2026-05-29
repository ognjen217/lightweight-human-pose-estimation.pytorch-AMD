#!/usr/bin/env python3
"""
Benchmark existing MIGraphX pose batch models.

Purpose:
  Compare existing pose_model_b1/b2/b4/b8_fp16.mxr models without changing
  stream code.

It measures:
  - preprocessing time
  - MIGraphX batch inference time
  - effective per-frame inference time
  - output tensor shapes
  - throughput FPS

Expected model input:
  [B, 3, H, W], usually [B, 3, 544, 968]

This script is intentionally inference-only. It does NOT run pose postprocessing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


def _shape_lens(shape_obj) -> List[int]:
    # MIGraphX shape lens API differs a bit across versions.
    if hasattr(shape_obj, "lens"):
        lens = shape_obj.lens()
    elif hasattr(shape_obj, "lengths"):
        lens = shape_obj.lengths()
    else:
        s = str(shape_obj)
        raise RuntimeError(f"Cannot read MIGraphX shape lens from object: {s}")
    return [int(x) for x in lens]


def _shape_type_string(shape_obj) -> str:
    if hasattr(shape_obj, "type"):
        return str(shape_obj.type())
    return str(shape_obj)


def load_migraphx_model(path: str | Path):
    import migraphx  # type: ignore

    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    print(f"[load] {path}")
    program = migraphx.load(path)
    param_shapes = program.get_parameter_shapes()
    if not param_shapes:
        raise RuntimeError("MIGraphX model has no parameters")

    # Prefer "input" if present, otherwise first param.
    input_name = "input" if "input" in param_shapes else list(param_shapes.keys())[0]
    in_shape = param_shapes[input_name]
    in_lens = _shape_lens(in_shape)
    in_type = _shape_type_string(in_shape)

    if len(in_lens) != 4:
        raise RuntimeError(f"Expected NCHW 4D input, got {input_name} shape={in_lens}")

    print(f"[input] name={input_name} shape={in_lens} type={in_type}")
    return program, input_name, in_lens, in_type


def dtype_from_migraphx_type(type_str: str):
    s = str(type_str).lower()
    if "half" in s or "float16" in s or "fp16" in s:
        return np.float16
    return np.float32


def preprocess_frame(frame_bgr: np.ndarray, width: int, height: int, dtype) -> np.ndarray:
    # Same basic shape contract as current validation/stream path:
    # resize to model input, BGR->RGB, normalize to [0,1], CHW.
    resized = cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    if dtype == np.float16:
        x = x.astype(np.float16, copy=False)
    else:
        x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def read_video_frames(video: str | Path, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    frames: List[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok or frame is None:
            # loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video}")
    return frames


def summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    program, input_name, in_lens, in_type = load_migraphx_model(args.model)
    batch, channels, height, width = in_lens
    dtype = dtype_from_migraphx_type(in_type)

    if args.batch_size is not None and int(args.batch_size) != int(batch):
        raise RuntimeError(
            f"--batch-size={args.batch_size} does not match model input batch={batch}. "
            "Use the matching pose_model_bN_fp16.mxr."
        )

    total_needed = (int(args.warmup_batches) + int(args.batches)) * int(batch)
    frames = read_video_frames(args.video, total_needed)

    pre_ms: List[float] = []
    infer_ms: List[float] = []
    output_shapes = None

    cursor = 0
    measured_batches = 0

    for batch_idx in range(int(args.warmup_batches) + int(args.batches)):
        batch_frames = frames[cursor : cursor + int(batch)]
        cursor += int(batch)
        if len(batch_frames) < int(batch):
            break

        t0 = time.perf_counter()
        xs = [preprocess_frame(f, width, height, dtype) for f in batch_frames]
        x = np.stack(xs, axis=0)
        x = np.ascontiguousarray(x)
        t1 = time.perf_counter()

        result = program.run({input_name: x})
        # Force materialization to numpy arrays so timing includes GPU wait/copy behavior
        out_np = [np.asarray(r) for r in result]
        t2 = time.perf_counter()

        if batch_idx >= int(args.warmup_batches):
            pre_ms.append((t1 - t0) * 1000.0)
            infer_ms.append((t2 - t1) * 1000.0)
            measured_batches += 1
            if output_shapes is None:
                output_shapes = [list(o.shape) for o in out_np]

        if args.print_every and (batch_idx + 1) % int(args.print_every) == 0:
            phase = "warmup" if batch_idx < int(args.warmup_batches) else "measured"
            print(f"[{phase}] batch {batch_idx + 1}/{int(args.warmup_batches) + int(args.batches)}")

    frames_measured = measured_batches * int(batch)
    pre_s = summarize_ms(pre_ms)
    infer_s = summarize_ms(infer_ms)
    avg_batch_ms = infer_s["avg"]
    avg_frame_ms = avg_batch_ms / max(float(batch), 1.0)
    infer_fps = 1000.0 * float(batch) / max(avg_batch_ms, 1e-9)
    e2e_batch_ms = pre_s["avg"] + infer_s["avg"]
    e2e_fps = 1000.0 * float(batch) / max(e2e_batch_ms, 1e-9)

    summary = {
        "model": str(args.model),
        "video": str(args.video),
        "input_name": input_name,
        "input_shape": in_lens,
        "input_dtype": str(dtype),
        "batch_size": int(batch),
        "measured_batches": int(measured_batches),
        "measured_frames": int(frames_measured),
        "preprocess_ms": pre_s,
        "inference_batch_ms": infer_s,
        "inference_per_frame_ms_avg": float(avg_frame_ms),
        "inference_only_fps": float(infer_fps),
        "preprocess_plus_infer_batch_ms_avg": float(e2e_batch_ms),
        "preprocess_plus_infer_fps": float(e2e_fps),
        "output_shapes": output_shapes,
    }

    print("\n" + "=" * 120)
    print("POSE BATCH MIGRAPHX SPEED SUMMARY")
    print("=" * 120)
    print(f"model:                  {summary['model']}")
    print(f"input:                  {input_name} {in_lens} {dtype}")
    print(f"measured batches:       {measured_batches}")
    print(f"measured frames:        {frames_measured}")
    print(f"preprocess avg:         {pre_s['avg']:.3f} ms/batch")
    print(f"infer avg:              {infer_s['avg']:.3f} ms/batch")
    print(f"infer p95:              {infer_s['p95']:.3f} ms/batch")
    print(f"infer per frame avg:    {avg_frame_ms:.3f} ms/frame")
    print(f"infer-only FPS:         {infer_fps:.2f}")
    print(f"pre+infer FPS:          {e2e_fps:.2f}")
    print(f"output shapes:          {output_shapes}")
    print("=" * 120)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"JSON saved: {out}")

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", default="cctv_1280x720_24fps_2.mp4")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--warmup-batches", type=int, default=10)
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--json", default="")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
