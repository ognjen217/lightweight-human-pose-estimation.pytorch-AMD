#!/usr/bin/env python3
"""
Compile a fixed-shape MIGraphX PAF pair scorer.

This head receives:
  top_scores  : [1, 18, K] float32
  top_indices : [1, 18, K] float32 flattened full-resolution index y*full_w + x
  pafs        : [1, 38, paf_h, paf_w] float32 low-resolution PAF tensor

It returns:
  pair_scores : [1, 19, K, K] float32
  pair_valid  : [1, 19, K, K] float32, 1 for valid limb pair, 0 otherwise

The graph replaces the expensive CPU part:
  full-res PAF resize + PAF sampling + pair score calculation

It intentionally does NOT implement greedy connection NMS or final pose assembly.
Those remain CPU-side because they are dynamic/list-like control flow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BODY_PARTS_KPT_IDS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17],
]
BODY_PARTS_PAF_IDS = [
    [12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1],
    [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35],
    [32, 33], [36, 37], [18, 19], [26, 27],
]


def _safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def head_name(
    paf_h: int,
    paf_w: int,
    full_h: int,
    full_w: int,
    *,
    topk: int,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
) -> str:
    return (
        "paf_pair_scorer_"
        f"paf{int(paf_h)}x{int(paf_w)}_full{int(full_h)}x{int(full_w)}_"
        f"k{int(topk)}_p{int(points_per_limb)}_"
        f"min{_safe_float_token(min_paf_score)}_sr{_safe_float_token(success_ratio_thr)}"
    )


def mxr_path(output_dir: str | Path, paf_h: int, paf_w: int, full_h: int, full_w: int, *, topk: int, points_per_limb: int, min_paf_score: float, success_ratio_thr: float) -> Path:
    return Path(output_dir) / f"{head_name(paf_h, paf_w, full_h, full_w, topk=topk, points_per_limb=points_per_limb, min_paf_score=min_paf_score, success_ratio_thr=success_ratio_thr)}.mxr"


def onnx_path(output_dir: str | Path, paf_h: int, paf_w: int, full_h: int, full_w: int, *, topk: int, points_per_limb: int, min_paf_score: float, success_ratio_thr: float) -> Path:
    return Path(output_dir) / f"{head_name(paf_h, paf_w, full_h, full_w, topk=topk, points_per_limb=points_per_limb, min_paf_score=min_paf_score, success_ratio_thr=success_ratio_thr)}.onnx"


WORKER_CODE = r"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn


BODY_PARTS_KPT_IDS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
    [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
    [2, 16], [5, 17],
]
BODY_PARTS_PAF_IDS = [
    [12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1],
    [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35],
    [32, 33], [36, 37], [18, 19], [26, 27],
]


class PAFPairScorerHead(nn.Module):
    def __init__(
        self,
        paf_h: int,
        paf_w: int,
        full_h: int,
        full_w: int,
        topk: int = 20,
        points_per_limb: int = 8,
        min_paf_score: float = 0.05,
        success_ratio_thr: float = 0.8,
        score_threshold: float = -1.0e8,
    ):
        super().__init__()
        self.paf_h = int(paf_h)
        self.paf_w = int(paf_w)
        self.full_h = int(full_h)
        self.full_w = int(full_w)
        self.topk = int(topk)
        self.points_per_limb = int(points_per_limb)
        self.min_paf_score = float(min_paf_score)
        self.success_ratio_thr = float(success_ratio_thr)
        self.score_threshold = float(score_threshold)

        # Fixed [P] alpha grid.
        alpha = torch.arange(self.points_per_limb, dtype=torch.float32) / float(max(1, self.points_per_limb - 1))
        self.register_buffer("alpha", alpha.view(1, 1, self.points_per_limb), persistent=True)

    def _xy_from_flat(self, indices_f):
        # indices_f: [1, 18, K] float32 flattened full-res index
        y = torch.floor(indices_f / float(self.full_w))
        x = indices_f - y * float(self.full_w)
        return x, y

    def _sample_channel(self, channel_flat, x_full, y_full):
        # Map full-res coordinates to low-res PAF nearest coordinates.
        # x_full/y_full: [K, K, P]
        sx_f = x_full * (float(self.paf_w) / float(self.full_w))
        sy_f = y_full * (float(self.paf_h) / float(self.full_h))

        sx = torch.floor(sx_f + 0.5).to(torch.int64)
        sy = torch.floor(sy_f + 0.5).to(torch.int64)

        sx = torch.clamp(sx, 0, self.paf_w - 1)
        sy = torch.clamp(sy, 0, self.paf_h - 1)

        flat_idx = sy * int(self.paf_w) + sx
        values = torch.gather(channel_flat, 1, flat_idx.reshape(1, -1))
        return values.reshape(self.topk, self.topk, self.points_per_limb)

    def forward(self, top_scores, top_indices, pafs):
        # top_scores/top_indices: [1,18,K], pafs: [1,38,H,W]
        x_all, y_all = self._xy_from_flat(top_indices)

        pair_scores_all = []
        pair_valid_all = []

        for part_id in range(len(BODY_PARTS_KPT_IDS)):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            paf_x_id = BODY_PARTS_PAF_IDS[part_id][0]
            paf_y_id = BODY_PARTS_PAF_IDS[part_id][1]

            ax = x_all[:, kpt_a_id, :].reshape(self.topk, 1)
            ay = y_all[:, kpt_a_id, :].reshape(self.topk, 1)
            bx = x_all[:, kpt_b_id, :].reshape(1, self.topk)
            by = y_all[:, kpt_b_id, :].reshape(1, self.topk)

            score_a = top_scores[:, kpt_a_id, :].reshape(self.topk, 1)
            score_b = top_scores[:, kpt_b_id, :].reshape(1, self.topk)
            valid_kpts = (score_a > self.score_threshold) & (score_b > self.score_threshold)

            dx = bx - ax
            dy = by - ay
            norm = torch.sqrt(dx * dx + dy * dy)
            valid_vec = norm > 1.0e-6

            vx = dx / (norm + 1.0e-6)
            vy = dy / (norm + 1.0e-6)

            # Sample full-res line points [K,K,P].
            px = ax.reshape(self.topk, 1, 1) + dx.reshape(self.topk, self.topk, 1) * self.alpha
            py = ay.reshape(self.topk, 1, 1) + dy.reshape(self.topk, self.topk, 1) * self.alpha

            paf_x_flat = pafs[:, paf_x_id, :, :].reshape(1, -1)
            paf_y_flat = pafs[:, paf_y_id, :, :].reshape(1, -1)

            field_x = self._sample_channel(paf_x_flat, px, py)
            field_y = self._sample_channel(paf_y_flat, px, py)

            dot = field_x * vx.reshape(self.topk, self.topk, 1) + field_y * vy.reshape(self.topk, self.topk, 1)
            valid_points = dot > self.min_paf_score
            valid_num = valid_points.to(torch.float32).sum(dim=2)
            score_sum = (dot * valid_points.to(torch.float32)).sum(dim=2)

            affinity = score_sum / (valid_num + 1.0e-6)
            success_ratio = valid_num / float(self.points_per_limb)

            valid = valid_kpts & valid_vec & (affinity > 0.0) & (success_ratio > self.success_ratio_thr)
            scores = torch.where(valid, affinity, torch.full_like(affinity, -1.0e9))

            pair_scores_all.append(scores)
            pair_valid_all.append(valid.to(torch.float32))

        pair_scores = torch.stack(pair_scores_all, dim=0).unsqueeze(0)
        pair_valid = torch.stack(pair_valid_all, dim=0).unsqueeze(0)
        return pair_scores, pair_valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--paf-h", type=int, required=True)
    p.add_argument("--paf-w", type=int, required=True)
    p.add_argument("--full-h", type=int, required=True)
    p.add_argument("--full-w", type=int, required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    model = PAFPairScorerHead(
        paf_h=args.paf_h,
        paf_w=args.paf_w,
        full_h=args.full_h,
        full_w=args.full_w,
        topk=args.topk,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
    ).eval()

    top_scores = torch.randn(1, 18, args.topk, dtype=torch.float32)
    top_indices = torch.randint(0, args.full_h * args.full_w, (1, 18, args.topk), dtype=torch.int64).to(torch.float32)
    pafs = torch.randn(1, 38, args.paf_h, args.paf_w, dtype=torch.float32)

    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (top_scores, top_indices, pafs),
            str(out_path),
            input_names=["top_scores", "top_indices", "pafs"],
            output_names=["pair_scores", "pair_valid"],
            opset_version=args.opset,
            do_constant_folding=True,
        )


if __name__ == "__main__":
    main()
"""


def _run_export_subprocess(
    *,
    output_onnx: Path,
    paf_h: int,
    paf_w: int,
    full_h: int,
    full_w: int,
    topk: int,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
    opset: int,
) -> None:
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix="_export_paf_pair_scorer.py", delete=False) as f:
        worker_path = Path(f.name)
        f.write(WORKER_CODE)

    try:
        subprocess.check_call(
            [
                sys.executable,
                str(worker_path),
                "--onnx", str(output_onnx),
                "--paf-h", str(int(paf_h)),
                "--paf-w", str(int(paf_w)),
                "--full-h", str(int(full_h)),
                "--full-w", str(int(full_w)),
                "--topk", str(int(topk)),
                "--points-per-limb", str(int(points_per_limb)),
                "--min-paf-score", str(float(min_paf_score)),
                "--success-ratio-thr", str(float(success_ratio_thr)),
                "--opset", str(int(opset)),
            ]
        )
    finally:
        try:
            worker_path.unlink()
        except FileNotFoundError:
            pass


def compile_paf_pair_scorer_head(
    *,
    paf_h: int,
    paf_w: int,
    full_h: int,
    full_w: int,
    output_dir: str | Path,
    topk: int = 20,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mxr = mxr_path(
        output_dir,
        paf_h,
        paf_w,
        full_h,
        full_w,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
    )
    onnx = onnx_path(
        output_dir,
        paf_h,
        paf_w,
        full_h,
        full_w,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
    )

    if mxr.exists() and not force:
        print(f"[paf-pair-scorer] exists, skipping: {mxr}")
        return mxr

    print(
        "[paf-pair-scorer] exporting ONNX: "
        f"paf={int(paf_h)}x{int(paf_w)}, full={int(full_h)}x{int(full_w)}, "
        f"K={int(topk)}, P={int(points_per_limb)}, min_paf={float(min_paf_score)}, "
        f"success_thr={float(success_ratio_thr)}"
    )
    _run_export_subprocess(
        output_onnx=onnx,
        paf_h=paf_h,
        paf_w=paf_w,
        full_h=full_h,
        full_w=full_w,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        opset=opset,
    )

    print(f"[paf-pair-scorer] compiling MIGraphX GPU target: {onnx.name} -> {mxr.name}")
    import migraphx  # type: ignore

    program = migraphx.parse_onnx(str(onnx))
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    migraphx.save(program, str(mxr))

    if not keep_onnx:
        try:
            onnx.unlink()
        except FileNotFoundError:
            pass

    print(f"[paf-pair-scorer] saved: {mxr}")
    return mxr


def compile_for_video(
    *,
    video: str | Path,
    target_width: int = 968,
    target_height: int = 544,
    stride: int = 8,
    output_dir: str | Path = "models/paf_pair_scorer_cache",
    topk: int = 20,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
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

    full_h, full_w = frame.shape[:2]
    paf_h = int(target_height) // int(stride)
    paf_w = int(target_width) // int(stride)

    print(
        f"[paf-pair-scorer] video full-res shape: {full_h}x{full_w}; "
        f"low-res PAF shape: {paf_h}x{paf_w}"
    )
    return compile_paf_pair_scorer_head(
        paf_h=paf_h,
        paf_w=paf_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=output_dir,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        opset=opset,
        exhaustive_tune=exhaustive_tune,
        force=force,
        keep_onnx=keep_onnx,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile MIGraphX PAF pair scoring head.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--shape", nargs=4, type=int, metavar=("PAF_H", "PAF_W", "FULL_H", "FULL_W"))

    p.add_argument("--output-dir", default="models/paf_pair_scorer_cache")
    p.add_argument("--target-width", type=int, default=968)
    p.add_argument("--target-height", type=int, default=544)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
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
            topk=args.topk,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )
    else:
        paf_h, paf_w, full_h, full_w = args.shape
        compile_paf_pair_scorer_head(
            paf_h=paf_h,
            paf_w=paf_w,
            full_h=full_h,
            full_w=full_w,
            output_dir=args.output_dir,
            topk=args.topk,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )


if __name__ == "__main__":
    main()
