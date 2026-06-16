#!/usr/bin/env python3
"""Smoke-test the native HIP heatmap TopK shared-library ABI.

This uses the host-mediated test entrypoint so it does not require PyTorch ROCm.
The default shape is intentionally small to validate build/link/kernel launch
without allocating the full 1080p B4 dense buffers.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend, HipHeatmapTopKShape
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend, HipHeatmapTopKShape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test HIP heatmap TopK backend ABI and kernels.")
    p.add_argument("--lib", default="", help="Path to libheatmap_topk_hip.so. Empty = default build path.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--in-h", type=int, default=8)
    p.add_argument("--in-w", type=int, default=8)
    p.add_argument("--full-h", type=int, default=64)
    p.add_argument("--full-w", type=int, default=64)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backend = HipHeatmapTopKBackend(args.lib or None)
    shape = HipHeatmapTopKShape(
        batch=int(args.batch),
        channels=int(args.channels),
        in_h=int(args.in_h),
        in_w=int(args.in_w),
        full_h=int(args.full_h),
        full_w=int(args.full_w),
        topk=int(args.topk),
        threshold=float(args.threshold),
        nms_radius=int(args.nms_radius),
    )

    rng = np.random.default_rng(int(args.seed))
    heatmaps = rng.normal(loc=0.0, scale=0.05, size=(shape.batch, shape.channels, shape.in_h, shape.in_w)).astype(np.float32)
    # Force one obvious peak so the smoke test validates valid-output path too.
    heatmaps[:, :, shape.in_h // 2, shape.in_w // 2] = 1.0

    t0 = time.perf_counter()
    scores, indices = backend.run_host(heatmaps, shape)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("library:", backend.path)
    print("shape:", shape)
    print(f"elapsed_ms: {dt_ms:.3f}")
    print("scores shape:", scores.shape, "dtype:", scores.dtype)
    print("indices shape:", indices.shape, "dtype:", indices.dtype)
    print("top scores sample:", scores.reshape(-1, shape.topk)[0].tolist())
    print("top indices sample:", indices.reshape(-1, shape.topk)[0].tolist())

    if scores.shape != (shape.batch, shape.channels, shape.topk):
        raise SystemExit("Unexpected scores shape")
    if indices.shape != (shape.batch, shape.channels, shape.topk):
        raise SystemExit("Unexpected indices shape")
    if not np.isfinite(scores).all():
        raise SystemExit("Non-finite scores returned")
    if float(scores.max()) <= float(args.threshold):
        raise SystemExit("Expected at least one valid score above threshold")
    print("HIP heatmap TopK smoke test passed")


if __name__ == "__main__":
    main()
