#!/usr/bin/env python3
"""Numerical sanity test for PyTorch vs MIGraphX HeatmapNMSHead.

Usage after exporting/compiling:
    python tests/test_migraphx_nms_sanity.py --mxr models/heatmap_nms_head.mxr --channels 19 --height 720 --width 1280
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_heatmap_nms_head import HeatmapNMSHead
from modules.migraphx_nms import MIGraphXNMSHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and MIGraphX NMS outputs")
    parser.add_argument("--mxr", required=True, help="Compiled MIGraphX NMS .mxr path")
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--height", type=int, default=46)
    parser.add_argument("--width", type=int, default=82)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    heatmaps = rng.normal(size=(1, args.channels, args.height, args.width)).astype(np.float32)

    torch_head = HeatmapNMSHead(threshold=args.threshold).eval()
    with torch.no_grad():
        torch_out = torch_head(torch.from_numpy(heatmaps)).cpu().numpy()

    mx_head = MIGraphXNMSHead(args.mxr, input_name="heatmaps")
    mx_out = mx_head.run(heatmaps)

    if torch_out.shape != mx_out.shape:
        raise AssertionError(f"Shape mismatch: torch={torch_out.shape}, migraphx={mx_out.shape}")

    max_abs_diff = float(np.max(np.abs(torch_out - mx_out)))
    torch_active = int(np.count_nonzero(torch_out))
    mx_active = int(np.count_nonzero(mx_out))

    print(f"torch_out shape: {torch_out.shape}")
    print(f"mx_out shape:    {mx_out.shape}")
    print(f"max_abs_diff:    {max_abs_diff:.8f}")
    print(f"active torch:    {torch_active}")
    print(f"active migraphx: {mx_active}")

    if max_abs_diff > 1e-5:
        raise AssertionError(f"max_abs_diff too high: {max_abs_diff}")
    if torch_active != mx_active:
        raise AssertionError(f"Active peak count mismatch: torch={torch_active}, migraphx={mx_active}")

    print("Sanity test passed.")


if __name__ == "__main__":
    main()
