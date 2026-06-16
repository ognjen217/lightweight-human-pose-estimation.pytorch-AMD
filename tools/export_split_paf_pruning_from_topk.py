#!/usr/bin/env python3
"""Export/compile MXR2 for the split external-heatmap pipeline.

MXR2 contains only the PAF scoring and pair pruning tail.  The heatmap TopK
stage is externalized, so top_scores/top_indices are graph inputs.

Inputs:
    pafs        [B,38,68,121] fp32
    top_scores  [B,18,K] fp32
    top_indices [B,18,K] int64

Outputs:
    limb_top_pair_a_idx [B,19,M] int64
    limb_top_pair_b_idx [B,19,M] int64
    limb_top_pair_score [B,19,M] fp32
    limb_top_pair_valid [B,19,M] fp32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    from tools.export_batchaware_fused_pruned_postprocess import BatchAwareFusedPrunedPostprocess, safe_float_token
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from tools.export_batchaware_fused_pruned_postprocess import BatchAwareFusedPrunedPostprocess, safe_float_token


class PafPruningFromTopK(nn.Module):
    """PAF scoring/pruning tail with externally supplied heatmap TopK outputs."""

    def __init__(
        self,
        *,
        batch_size: int,
        in_h: int = 68,
        in_w: int = 121,
        full_h: int = 1080,
        full_w: int = 1920,
        topk: int = 20,
        limb_topm: int = 20,
        points_per_limb: int = 8,
        min_paf_score: float = 0.05,
        success_ratio_thr: float = 0.8,
        paf_cubic_a: float = -0.75,
        min_pair_score: float = 0.0,
    ):
        super().__init__()
        self.core = BatchAwareFusedPrunedPostprocess(
            batch_size=int(batch_size),
            in_h=int(in_h),
            in_w=int(in_w),
            full_h=int(full_h),
            full_w=int(full_w),
            topk=int(topk),
            limb_topm=int(limb_topm),
            threshold=0.1,
            nms_radius=6,
            nms_impl="separable",
            heatmap_cubic_a=float(paf_cubic_a),
            points_per_limb=int(points_per_limb),
            min_paf_score=float(min_paf_score),
            success_ratio_thr=float(success_ratio_thr),
            paf_cubic_a=float(paf_cubic_a),
            min_pair_score=float(min_pair_score),
        )

    def forward(self, pafs, top_scores, top_indices):
        pair_scores, pair_valid = self.core.score_pairs(top_scores, top_indices, pafs)
        a_idx, b_idx, limb_score, limb_valid = self.core.prune_pairs(pair_scores, pair_valid)
        return a_idx, b_idx, limb_score, limb_valid


def default_name(
    *,
    batch_size: int,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    topk: int,
    limb_topm: int,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
    paf_cubic_a: float,
    min_pair_score: float,
) -> str:
    return (
        "split_paf_pruning_from_topk_"
        f"b{batch_size}_{in_h}x{in_w}_to_{full_h}x{full_w}_"
        f"k{topk}_m{limb_topm}_p{points_per_limb}_"
        f"min{safe_float_token(min_paf_score)}_sr{safe_float_token(success_ratio_thr)}_"
        f"pa{safe_float_token(paf_cubic_a)}_mp{safe_float_token(min_pair_score)}"
    )


def export_paf_pruning(
    *,
    output_onnx: str | Path,
    batch_size: int,
    in_h: int = 68,
    in_w: int = 121,
    full_h: int = 1080,
    full_w: int = 1920,
    topk: int = 20,
    limb_topm: int = 20,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
    paf_cubic_a: float = -0.75,
    min_pair_score: float = 0.0,
    opset: int = 18,
) -> Path:
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    model = PafPruningFromTopK(
        batch_size=int(batch_size),
        in_h=int(in_h),
        in_w=int(in_w),
        full_h=int(full_h),
        full_w=int(full_w),
        topk=int(topk),
        limb_topm=int(limb_topm),
        points_per_limb=int(points_per_limb),
        min_paf_score=float(min_paf_score),
        success_ratio_thr=float(success_ratio_thr),
        paf_cubic_a=float(paf_cubic_a),
        min_pair_score=float(min_pair_score),
    ).eval()

    pafs = torch.randn(int(batch_size), 38, int(in_h), int(in_w), dtype=torch.float32)
    top_scores = torch.randn(int(batch_size), 18, int(topk), dtype=torch.float32)
    top_indices = torch.randint(0, int(full_h) * int(full_w), (int(batch_size), 18, int(topk)), dtype=torch.int64)

    torch.onnx.export(
        model,
        (pafs, top_scores, top_indices),
        str(output_onnx),
        input_names=["pafs", "top_scores", "top_indices"],
        output_names=[
            "limb_top_pair_a_idx",
            "limb_top_pair_b_idx",
            "limb_top_pair_score",
            "limb_top_pair_valid",
        ],
        dynamic_axes=None,
        opset_version=int(opset),
        do_constant_folding=True,
    )

    output_onnx.with_suffix(".debug.json").write_text(json.dumps({
        "kind": "split_paf_pruning_mxr2",
        "output_onnx": str(output_onnx),
        "batch_size": int(batch_size),
        "in_h": int(in_h),
        "in_w": int(in_w),
        "full_h": int(full_h),
        "full_w": int(full_w),
        "topk": int(topk),
        "limb_topm": int(limb_topm),
        "points_per_limb": int(points_per_limb),
        "min_paf_score": float(min_paf_score),
        "success_ratio_thr": float(success_ratio_thr),
        "paf_cubic_a": float(paf_cubic_a),
        "min_pair_score": float(min_pair_score),
        "inputs": {
            "pafs": [int(batch_size), 38, int(in_h), int(in_w)],
            "top_scores": [int(batch_size), 18, int(topk)],
            "top_indices": [int(batch_size), 18, int(topk)],
        },
        "outputs": {
            "limb_top_pair_a_idx": [int(batch_size), 19, int(limb_topm)],
            "limb_top_pair_b_idx": [int(batch_size), 19, int(limb_topm)],
            "limb_top_pair_score": [int(batch_size), 19, int(limb_topm)],
            "limb_top_pair_valid": [int(batch_size), 19, int(limb_topm)],
        },
    }, indent=2))
    return output_onnx


def compile_mxr(onnx_path: str | Path, mxr_path: str | Path, exhaustive_tune: bool = False) -> Path:
    import migraphx  # type: ignore

    onnx_path = Path(onnx_path)
    mxr_path = Path(mxr_path)
    print(f"[compile] parse_onnx: {onnx_path}")
    program = migraphx.parse_onnx(str(onnx_path))
    print("[compile] input shapes:", program.get_parameter_shapes())
    print("[compile] compile target=gpu")
    t0 = time.time()
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    mxr_path.parent.mkdir(parents=True, exist_ok=True)
    migraphx.save(program, str(mxr_path))
    print(f"[compile] saved: {mxr_path}")
    print(f"[compile] elapsed_s: {time.time() - t0:.2f}")
    return mxr_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export/compile split MXR2: pafs + topk -> pruned limb pairs.")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--in-h", type=int, default=68)
    p.add_argument("--in-w", type=int, default=121)
    p.add_argument("--full-h", type=int, default=1080)
    p.add_argument("--full-w", type=int, default=1920)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limb-topm", type=int, default=20)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--min-pair-score", type=float, default=0.0)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--output-onnx", default="")
    p.add_argument("--output-dir", default="models/split_paf_pruning_from_topk")
    p.add_argument("--output-mxr", default="")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--exhaustive-tune", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = default_name(
        batch_size=args.batch_size,
        in_h=args.in_h,
        in_w=args.in_w,
        full_h=args.full_h,
        full_w=args.full_w,
        topk=args.topk,
        limb_topm=args.limb_topm,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
        min_pair_score=args.min_pair_score,
    )
    output_onnx = Path(args.output_onnx) if args.output_onnx else Path(args.output_dir) / f"{name}.onnx"
    output_mxr = Path(args.output_mxr) if args.output_mxr else Path(args.output_dir) / f"{name}.mxr"

    exported = export_paf_pruning(
        output_onnx=output_onnx,
        batch_size=args.batch_size,
        in_h=args.in_h,
        in_w=args.in_w,
        full_h=args.full_h,
        full_w=args.full_w,
        topk=args.topk,
        limb_topm=args.limb_topm,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
        min_pair_score=args.min_pair_score,
        opset=args.opset,
    )
    print(f"[export] saved: {exported}")
    if args.compile or args.output_mxr:
        compile_mxr(exported, output_mxr, exhaustive_tune=args.exhaustive_tune)


if __name__ == "__main__":
    main()
