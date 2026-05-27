#!/usr/bin/env python3
"""
Compile a fixed-shape MIGraphX post-processing head that performs manual
OpenCV-like bicubic heatmap resize without using ONNX Resize(mode="cubic"):

    low-res heatmaps [1, C, in_h, in_w]
      -> manual separable bicubic resize to [1, C, out_h, out_w]
         implemented with Gather/index_select + precomputed cubic weights
      -> local-max NMS via MaxPool
      -> per-channel TopK over H*W

The compiled head returns:
    top_scores:  [1, C, K]
    top_indices: [1, C, K] flattened full-res indices (y * out_w + x)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def head_name(
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    *,
    topk: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    cubic_a: float,
) -> str:
    safe_impl = str(nms_impl).replace("-", "_")
    thr = _safe_float_token(threshold)
    ca = _safe_float_token(cubic_a)
    return (
        f"heatmap_manual_cubic_nms_topk_"
        f"{int(in_h)}x{int(in_w)}_to_{int(out_h)}x{int(out_w)}_"
        f"k{int(topk)}_thr{thr}_r{int(nms_radius)}_{safe_impl}_a{ca}"
    )


def mxr_path(output_dir: Path, in_h: int, in_w: int, out_h: int, out_w: int, *, topk: int, threshold: float, nms_radius: int, nms_impl: str, cubic_a: float) -> Path:
    return output_dir / f"{head_name(in_h, in_w, out_h, out_w, topk=topk, threshold=threshold, nms_radius=nms_radius, nms_impl=nms_impl, cubic_a=cubic_a)}.mxr"


def onnx_path(output_dir: Path, in_h: int, in_w: int, out_h: int, out_w: int, *, topk: int, threshold: float, nms_radius: int, nms_impl: str, cubic_a: float) -> Path:
    return output_dir / f"{head_name(in_h, in_w, out_h, out_w, topk=topk, threshold=threshold, nms_radius=nms_radius, nms_impl=nms_impl, cubic_a=cubic_a)}.onnx"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


WORKER_CODE = r"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def cubic_weights_and_indices(in_size: int, out_size: int, a: float = -0.75):
    scale = float(in_size) / float(out_size)
    dst = torch.arange(out_size, dtype=torch.float32)
    src = (dst + 0.5) * scale - 0.5
    base = torch.floor(src).to(torch.int64)

    idxs = []
    weights = []
    aa = float(a)

    for off in (-1, 0, 1, 2):
        raw_idx = base + int(off)
        dist = torch.abs(src - raw_idx.to(torch.float32))

        dist2 = dist * dist
        dist3 = dist2 * dist

        w1 = (aa + 2.0) * dist3 - (aa + 3.0) * dist2 + 1.0
        w2 = aa * dist3 - 5.0 * aa * dist2 + 8.0 * aa * dist - 4.0 * aa
        w = torch.where(dist <= 1.0, w1, torch.where(dist < 2.0, w2, torch.zeros_like(dist)))

        idx = torch.clamp(raw_idx, 0, int(in_size) - 1)
        idxs.append(idx.to(torch.int64))
        weights.append(w.to(torch.float32))

    return torch.stack(idxs, dim=0), torch.stack(weights, dim=0)


class ManualCubicResizeNMSTopKHead(nn.Module):
    def __init__(
        self,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
        topk: int = 20,
        threshold: float = 0.1,
        nms_radius: int = 6,
        nms_impl: str = "separable",
        cubic_a: float = -0.75,
    ):
        super().__init__()
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.nms_impl = str(nms_impl)

        x_idx, x_w = cubic_weights_and_indices(self.in_w, self.out_w, a=float(cubic_a))
        y_idx, y_w = cubic_weights_and_indices(self.in_h, self.out_h, a=float(cubic_a))

        self.register_buffer("x_idx", x_idx, persistent=True)
        self.register_buffer("x_w", x_w, persistent=True)
        self.register_buffer("y_idx", y_idx, persistent=True)
        self.register_buffer("y_w", y_w, persistent=True)

    def manual_cubic_resize(self, heatmaps):
        x0 = torch.index_select(heatmaps, 3, self.x_idx[0]) * self.x_w[0].view(1, 1, 1, self.out_w)
        x1 = torch.index_select(heatmaps, 3, self.x_idx[1]) * self.x_w[1].view(1, 1, 1, self.out_w)
        x2 = torch.index_select(heatmaps, 3, self.x_idx[2]) * self.x_w[2].view(1, 1, 1, self.out_w)
        x3 = torch.index_select(heatmaps, 3, self.x_idx[3]) * self.x_w[3].view(1, 1, 1, self.out_w)
        tmp = x0 + x1 + x2 + x3

        y0 = torch.index_select(tmp, 2, self.y_idx[0]) * self.y_w[0].view(1, 1, self.out_h, 1)
        y1 = torch.index_select(tmp, 2, self.y_idx[1]) * self.y_w[1].view(1, 1, self.out_h, 1)
        y2 = torch.index_select(tmp, 2, self.y_idx[2]) * self.y_w[2].view(1, 1, self.out_h, 1)
        y3 = torch.index_select(tmp, 2, self.y_idx[3]) * self.y_w[3].view(1, 1, self.out_h, 1)
        return y0 + y1 + y2 + y3

    def forward(self, heatmaps):
        hm = self.manual_cubic_resize(heatmaps)

        r = self.nms_radius
        k = 2 * r + 1
        if self.nms_impl == "2d":
            pooled = F.max_pool2d(hm, kernel_size=k, stride=1, padding=r)
        elif self.nms_impl == "separable":
            pooled = F.max_pool2d(hm, kernel_size=(k, 1), stride=1, padding=(r, 0))
            pooled = F.max_pool2d(pooled, kernel_size=(1, k), stride=1, padding=(0, r))
        else:
            raise RuntimeError(f"Unsupported nms_impl={self.nms_impl}")

        peaks = (hm == pooled) & (hm > self.threshold)
        masked = torch.where(peaks, hm, torch.full_like(hm, -1.0e9))
        flat = masked.flatten(start_dim=2)
        top_scores, top_indices = torch.topk(flat, k=self.topk, dim=2, largest=True, sorted=True)
        return top_scores, top_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--in-h", type=int, required=True)
    p.add_argument("--in-w", type=int, required=True)
    p.add_argument("--out-h", type=int, required=True)
    p.add_argument("--out-w", type=int, required=True)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model = ManualCubicResizeNMSTopKHead(
        in_h=args.in_h,
        in_w=args.in_w,
        out_h=args.out_h,
        out_w=args.out_w,
        topk=args.topk,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        cubic_a=args.cubic_a,
    ).eval()

    dummy = torch.randn(1, args.channels, args.in_h, args.in_w, dtype=torch.float32)
    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["heatmaps"],
            output_names=["top_scores", "top_indices"],
            opset_version=args.opset,
            do_constant_folding=True,
        )


if __name__ == "__main__":
    main()
"""


def _run_export_subprocess(
    *,
    output_onnx: Path,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    channels: int,
    topk: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    cubic_a: float,
    opset: int,
) -> None:
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix="_export_manual_cubic_nms_topk.py", delete=False) as f:
        worker_path = Path(f.name)
        f.write(WORKER_CODE)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root()) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.check_call(
            [
                sys.executable,
                str(worker_path),
                "--onnx", str(output_onnx),
                "--in-h", str(int(in_h)),
                "--in-w", str(int(in_w)),
                "--out-h", str(int(out_h)),
                "--out-w", str(int(out_w)),
                "--channels", str(int(channels)),
                "--topk", str(int(topk)),
                "--threshold", str(float(threshold)),
                "--nms-radius", str(int(nms_radius)),
                "--nms-impl", str(nms_impl),
                "--cubic-a", str(float(cubic_a)),
                "--opset", str(int(opset)),
            ],
            env=env,
        )
    finally:
        try:
            worker_path.unlink()
        except FileNotFoundError:
            pass


def compile_manual_cubic_nms_topk_head(
    *,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    output_dir: str | Path,
    channels: int = 18,
    topk: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    nms_impl: str = "separable",
    cubic_a: float = -0.75,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mxr = mxr_path(output_dir, in_h, in_w, out_h, out_w, topk=topk, threshold=threshold, nms_radius=nms_radius, nms_impl=nms_impl, cubic_a=cubic_a)
    onnx = onnx_path(output_dir, in_h, in_w, out_h, out_w, topk=topk, threshold=threshold, nms_radius=nms_radius, nms_impl=nms_impl, cubic_a=cubic_a)

    if mxr.exists() and not force:
        print(f"[manual-cubic-topk] exists, skipping: {mxr}")
        return mxr

    print(
        "[manual-cubic-topk] exporting ONNX: "
        f"{int(in_h)}x{int(in_w)} -> {int(out_h)}x{int(out_w)}, "
        f"C={int(channels)}, K={int(topk)}, thr={float(threshold)}, "
        f"impl={nms_impl}, radius={int(nms_radius)}, cubic_a={float(cubic_a)}"
    )
    _run_export_subprocess(
        output_onnx=onnx,
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        channels=channels,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=cubic_a,
        opset=opset,
    )

    print(f"[manual-cubic-topk] compiling MIGraphX GPU target: {onnx.name} -> {mxr.name}")
    import migraphx  # type: ignore

    program = migraphx.parse_onnx(str(onnx))
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    migraphx.save(program, str(mxr))

    if not keep_onnx:
        try:
            onnx.unlink()
        except FileNotFoundError:
            pass

    print(f"[manual-cubic-topk] saved: {mxr}")
    return mxr


def compile_for_video(
    *,
    video: str | Path,
    target_width: int = 968,
    target_height: int = 544,
    stride: int = 8,
    output_dir: str | Path = "models/manual_cubic_nms_topk_cache",
    channels: int = 18,
    topk: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    nms_impl: str = "separable",
    cubic_a: float = -0.75,
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

    out_h, out_w = frame.shape[:2]
    in_h = int(target_height) // int(stride)
    in_w = int(target_width) // int(stride)

    print(
        f"[manual-cubic-topk] video full-res shape: {out_h}x{out_w}; "
        f"low-res model shape: {in_h}x{in_w}"
    )
    return compile_manual_cubic_nms_topk_head(
        in_h=in_h,
        in_w=in_w,
        out_h=out_h,
        out_w=out_w,
        output_dir=output_dir,
        channels=channels,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=cubic_a,
        opset=opset,
        exhaustive_tune=exhaustive_tune,
        force=force,
        keep_onnx=keep_onnx,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile manual cubic resize + NMS + TopK MIGraphX head.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--shape", nargs=4, type=int, metavar=("IN_H", "IN_W", "OUT_H", "OUT_W"))

    p.add_argument("--output-dir", default="models/manual_cubic_nms_topk_cache")
    p.add_argument("--target-width", type=int, default=968)
    p.add_argument("--target-height", type=int, default=544)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.video:
        compile_for_video(
            video=args.video,
            target_width=args.target_width,
            target_height=args.target_height,
            stride=args.stride,
            output_dir=args.output_dir,
            channels=args.channels,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            cubic_a=args.cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )
    else:
        in_h, in_w, out_h, out_w = args.shape
        compile_manual_cubic_nms_topk_head(
            in_h=in_h,
            in_w=in_w,
            out_h=out_h,
            out_w=out_w,
            output_dir=args.output_dir,
            channels=args.channels,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            cubic_a=args.cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )


if __name__ == "__main__":
    main()
