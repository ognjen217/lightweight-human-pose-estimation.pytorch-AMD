#!/usr/bin/env python3
"""
modules/migraphx_compiler.py

Compile fixed-shape MIGraphX heatmap NMS heads for video and COCO validation.

Important ROCm/MIGraphX note
----------------------------
PyTorch ONNX export and MIGraphX GPU compilation are intentionally separated.
The ONNX export runs in a short child process. The parent process then imports
MIGraphX and compiles the exported ONNX. This avoids initializing PyTorch ROCm
and MIGraphX GPU target in the same Python process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def nms_head_name(height: int, width: int) -> str:
    return f"heatmap_nms_head_{int(height)}x{int(width)}"


def nms_mxr_path(output_dir: Path, height: int, width: int) -> Path:
    return output_dir / f"{nms_head_name(height, width)}.mxr"


def nms_onnx_path(output_dir: Path, height: int, width: int) -> Path:
    return output_dir / f"{nms_head_name(height, width)}.onnx"


def _repo_root() -> Path:
    # modules/migraphx_compiler.py -> repo root
    return Path(__file__).resolve().parents[1]


def _run_onnx_export_subprocess(
    *,
    onnx_path: Path,
    height: int,
    width: int,
    channels: int,
    threshold: float,
    nms_radius: int,
    opset: int,
) -> None:
    """Export the torch NMS head in a separate Python process.

    This child process may initialize PyTorch ROCm/MIOpen. It exits before the
    parent process imports MIGraphX and compiles for the GPU target.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    worker_code = r'''
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapNMSHead(nn.Module):
    def __init__(self, threshold: float = 0.1, nms_radius: int = 6):
        super().__init__()
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)

    def forward(self, heatmaps):
        r = self.nms_radius
        k = 2 * r + 1
        pooled = F.max_pool2d(heatmaps, kernel_size=k, stride=1, padding=r)
        peaks = (heatmaps == pooled) & (heatmaps > self.threshold)
        return peaks.to(dtype=heatmaps.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model = HeatmapNMSHead(
        threshold=args.threshold,
        nms_radius=args.nms_radius,
    ).eval()

    dummy = torch.randn(1, args.channels, args.height, args.width, dtype=torch.float32)
    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["heatmaps"],
            output_names=["peak_mask"],
            opset_version=args.opset,
            do_constant_folding=True,
        )


if __name__ == "__main__":
    main()
'''

    with tempfile.NamedTemporaryFile("w", suffix="_export_heatmap_nms.py", delete=False) as f:
        worker_path = Path(f.name)
        f.write(worker_code)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.check_call(
            [
                sys.executable,
                str(worker_path),
                "--onnx",
                str(onnx_path),
                "--height",
                str(int(height)),
                "--width",
                str(int(width)),
                "--channels",
                str(int(channels)),
                "--threshold",
                str(float(threshold)),
                "--nms-radius",
                str(int(nms_radius)),
                "--opset",
                str(int(opset)),
            ],
            env=env,
        )
    finally:
        try:
            worker_path.unlink()
        except FileNotFoundError:
            pass


def compile_nms_head_migraphx(
    *,
    height: int,
    width: int,
    output_dir: str | Path,
    channels: int = 19,
    threshold: float = 0.1,
    nms_radius: int = 6,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> Path:
    """Compile one fixed-shape full-resolution heatmap NMS head to .mxr."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height = int(height)
    width = int(width)
    mxr_path = nms_mxr_path(output_dir, height, width)
    onnx_path = nms_onnx_path(output_dir, height, width)

    if mxr_path.exists() and not force:
        print(f"[mx-nms-cache] exists, skipping: {mxr_path}")
        return mxr_path

    print(f"[mx-nms-cache] exporting ONNX in isolated process: {height}x{width}")
    _run_onnx_export_subprocess(
        onnx_path=onnx_path,
        height=height,
        width=width,
        channels=channels,
        threshold=threshold,
        nms_radius=nms_radius,
        opset=opset,
    )

    print(f"[mx-nms-cache] compiling MIGraphX GPU target: {onnx_path.name} -> {mxr_path.name}")
    import migraphx  # imported only after the torch export subprocess exits

    program = migraphx.parse_onnx(str(onnx_path))
    target = migraphx.get_target("gpu")
    program.compile(target, exhaustive_tune=bool(exhaustive_tune))
    migraphx.save(program, str(mxr_path))

    if not keep_onnx:
        try:
            onnx_path.unlink()
        except FileNotFoundError:
            pass

    print(f"[mx-nms-cache] saved: {mxr_path}")
    return mxr_path


def _iter_coco_shapes_from_annotations(
    annotations: str | Path,
    *,
    limit: int = 0,
) -> List[Tuple[int, int]]:
    annotations = Path(annotations)
    with annotations.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []
    for img in data.get("images", []):
        h = int(img["height"])
        w = int(img["width"])
        shapes.append((h, w))
        if limit and len(shapes) >= int(limit):
            break

    return sorted(set(shapes))


def compile_nms_cache_for_shapes(
    shapes: Iterable[Tuple[int, int]],
    *,
    output_dir: str | Path,
    channels: int = 19,
    threshold: float = 0.1,
    nms_radius: int = 6,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> List[Path]:
    output_dir = Path(output_dir)
    unique_shapes = sorted({(int(h), int(w)) for h, w in shapes})
    paths: List[Path] = []

    total = len(unique_shapes)
    for idx, (h, w) in enumerate(unique_shapes, start=1):
        print(f"[mx-nms-cache] compiling {idx}/{total}: {h}x{w}")
        paths.append(
            compile_nms_head_migraphx(
                height=h,
                width=w,
                output_dir=output_dir,
                channels=channels,
                threshold=threshold,
                nms_radius=nms_radius,
                opset=opset,
                exhaustive_tune=exhaustive_tune,
                force=force,
                keep_onnx=keep_onnx,
            )
        )
    return paths


def compile_nms_cache_for_coco(
    *,
    annotations: str | Path,
    output_dir: str | Path,
    limit: int = 0,
    channels: int = 19,
    threshold: float = 0.1,
    nms_radius: int = 6,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> List[Path]:
    shapes = _iter_coco_shapes_from_annotations(annotations, limit=limit)
    print(f"[mx-nms-cache] unique COCO full-res shapes: {len(shapes)}")
    return compile_nms_cache_for_shapes(
        shapes,
        output_dir=output_dir,
        channels=channels,
        threshold=threshold,
        nms_radius=nms_radius,
        opset=opset,
        exhaustive_tune=exhaustive_tune,
        force=force,
        keep_onnx=keep_onnx,
    )


def compile_nms_cache_for_video(
    *,
    video: str | Path,
    output_dir: str | Path,
    channels: int = 19,
    threshold: float = 0.1,
    nms_radius: int = 6,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {video}")

    h, w = frame.shape[:2]
    print(f"[mx-nms-cache] video full-res shape: {h}x{w}")
    return compile_nms_head_migraphx(
        height=h,
        width=w,
        output_dir=output_dir,
        channels=channels,
        threshold=threshold,
        nms_radius=nms_radius,
        opset=opset,
        exhaustive_tune=exhaustive_tune,
        force=force,
        keep_onnx=keep_onnx,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile MIGraphX heatmap NMS heads.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--annotations", help="COCO annotations JSON used to collect unique image HxW shapes.")
    src.add_argument("--video", help="Video path; compiles one NMS head for the first-frame resolution.")
    src.add_argument("--shape", nargs=2, type=int, metavar=("H", "W"), help="Compile one explicit H W shape.")

    p.add_argument("--output-dir", default="models/nms_fullres_cache")
    p.add_argument("--limit", type=int, default=0, help="Limit number of COCO images read from annotations before unique-shape collection.")
    p.add_argument("--channels", type=int, default=19)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.annotations:
        compile_nms_cache_for_coco(
            annotations=args.annotations,
            output_dir=args.output_dir,
            limit=args.limit,
            channels=args.channels,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )
    elif args.video:
        compile_nms_cache_for_video(
            video=args.video,
            output_dir=args.output_dir,
            channels=args.channels,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )
    else:
        h, w = args.shape
        compile_nms_head_migraphx(
            height=h,
            width=w,
            output_dir=args.output_dir,
            channels=args.channels,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )


if __name__ == "__main__":
    main()