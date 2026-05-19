#!/usr/bin/env python3
"""Export a small heatmap NMS head to ONNX.

The exported graph is intentionally limited to dense tensor operations:
max-pooling based local-maximum detection, thresholding, and float mask output.
It does not attempt to implement OpenPose PAF grouping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapNMSHead(nn.Module):
    """Dense heatmap local-maximum detector.

    Input:  heatmaps [N, C, H, W]
    Output: peak_mask [N, C, H, W], float32 values in {0, 1}
    """

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        pooled = F.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)
        is_local_max = heatmaps == pooled
        above_threshold = heatmaps > self.threshold
        peak_mask = (is_local_max & above_threshold).to(dtype=heatmaps.dtype)
        return peak_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HeatmapNMSHead to ONNX")
    parser.add_argument("--output", default="heatmap_nms_head.onnx", help="Output ONNX path")
    parser.add_argument("--channels", type=int, default=19, help="Number of heatmap channels")
    parser.add_argument("--height", type=int, default=46, help="Heatmap height")
    parser.add_argument("--width", type=int, default=82, help="Heatmap width")
    parser.add_argument("--threshold", type=float, default=0.1, help="Peak threshold")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = HeatmapNMSHead(threshold=args.threshold).eval()
    dummy = torch.randn(1, args.channels, args.height, args.width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["heatmaps"],
        output_names=["peak_mask"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported ONNX NMS head: {output}")
    print(f"Input shape: [1, {args.channels}, {args.height}, {args.width}]")
    print(f"Threshold: {args.threshold}")


if __name__ == "__main__":
    main()
