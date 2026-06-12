#!/usr/bin/env python3
"""
Export an experimental batch-aware fused-pruned postprocess ONNX that keeps the
pose model outputs in fp16 and returns int32 index tensors.

This is a diagnostic/experimental variant.  It is intended to test the cost of
removing the adapter fp16->fp32 casts and carrying fp16 through the fused
postprocess path.

Inputs:
  heatmaps [B,18,68,121] fp16
  pafs     [B,38,68,121] fp16

Outputs:
  top_scores              [B,18,K] fp16
  top_indices             [B,18,K] int32
  limb_top_pair_a_idx     [B,19,M] int32
  limb_top_pair_b_idx     [B,19,M] int32
  limb_top_pair_score     [B,19,M] fp16
  limb_top_pair_valid     [B,19,M] fp16

Notes:
  * ONNX TopK naturally produces int64 indices.  This exporter casts the public
    TopK/pruned index outputs to int32.
  * PyTorch gather/index_select still requires int64 index tensors internally,
    so the exported graph may still contain int64 helper indices around gather
    operations.  The test target is: no adapter fp16->fp32 cast, fp16 public
    floating outputs, int32 public index outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BatchAwareFusedPrunedPostprocessFp16I32(nn.Module):
    def __init__(
        self,
        *,
        batch_size: int,
        in_h: int,
        in_w: int,
        full_h: int,
        full_w: int,
        topk: int = 20,
        limb_topm: int = 20,
        threshold: float = 0.1,
        nms_radius: int = 6,
        nms_impl: str = "separable",
        heatmap_cubic_a: float = -0.75,
        points_per_limb: int = 8,
        min_paf_score: float = 0.05,
        success_ratio_thr: float = 0.8,
        paf_cubic_a: float = -0.75,
        min_pair_score: float = 0.0,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.full_h = int(full_h)
        self.full_w = int(full_w)
        self.topk = int(topk)
        self.limb_topm = int(limb_topm)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.nms_impl = str(nms_impl)
        self.points_per_limb = int(points_per_limb)
        self.min_paf_score = float(min_paf_score)
        self.success_ratio_thr = float(success_ratio_thr)
        self.paf_cubic_a = float(paf_cubic_a)
        self.min_pair_score = float(min_pair_score)
        # fp16 cannot represent -1e9.  This sentinel is still safely below all
        # score thresholds used by the CPU tail.
        self.invalid_value = -65504.0

        x_idx, x_w = cubic_weights_and_indices(self.in_w, self.full_w, a=float(heatmap_cubic_a))
        y_idx, y_w = cubic_weights_and_indices(self.in_h, self.full_h, a=float(heatmap_cubic_a))
        self.register_buffer("x_idx", x_idx, persistent=True)
        self.register_buffer("x_w", x_w.to(torch.float16), persistent=True)
        self.register_buffer("y_idx", y_idx, persistent=True)
        self.register_buffer("y_w", y_w.to(torch.float16), persistent=True)

        alpha = torch.arange(self.points_per_limb, dtype=torch.float16) / float(max(1, self.points_per_limb - 1))
        self.register_buffer("alpha", alpha.view(1, 1, 1, self.points_per_limb), persistent=True)

    def _invalid_like(self, x):
        return torch.full_like(x, self.invalid_value)

    def manual_cubic_resize_heatmaps(self, heatmaps):
        x0 = torch.index_select(heatmaps, 3, self.x_idx[0]) * self.x_w[0].view(1, 1, 1, self.full_w)
        x1 = torch.index_select(heatmaps, 3, self.x_idx[1]) * self.x_w[1].view(1, 1, 1, self.full_w)
        x2 = torch.index_select(heatmaps, 3, self.x_idx[2]) * self.x_w[2].view(1, 1, 1, self.full_w)
        x3 = torch.index_select(heatmaps, 3, self.x_idx[3]) * self.x_w[3].view(1, 1, 1, self.full_w)
        tmp = x0 + x1 + x2 + x3
        y0 = torch.index_select(tmp, 2, self.y_idx[0]) * self.y_w[0].view(1, 1, self.full_h, 1)
        y1 = torch.index_select(tmp, 2, self.y_idx[1]) * self.y_w[1].view(1, 1, self.full_h, 1)
        y2 = torch.index_select(tmp, 2, self.y_idx[2]) * self.y_w[2].view(1, 1, self.full_h, 1)
        y3 = torch.index_select(tmp, 2, self.y_idx[3]) * self.y_w[3].view(1, 1, self.full_h, 1)
        return y0 + y1 + y2 + y3

    def topk_heatmaps(self, heatmaps):
        hm = self.manual_cubic_resize_heatmaps(heatmaps)
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
        masked = torch.where(peaks, hm, self._invalid_like(hm))
        flat = masked.flatten(start_dim=2)
        top_scores, top_indices_i64 = torch.topk(flat, k=self.topk, dim=2, largest=True, sorted=True)
        return top_scores, top_indices_i64.to(torch.int32)

    def _xy_from_flat(self, indices_i32):
        # Keep flat indices as int32.  Decode with integer arithmetic before
        # casting the small x/y coordinates to fp16.  Casting the flat index
        # directly to fp16 would overflow for values up to 2,073,599.
        y_i = torch.div(indices_i32, int(self.full_w), rounding_mode="floor")
        x_i = indices_i32 - y_i * int(self.full_w)
        return x_i.to(torch.float16), y_i.to(torch.float16)

    def _cubic_kernel(self, dist):
        a = self.paf_cubic_a
        dist2 = dist * dist
        dist3 = dist2 * dist
        w1 = (a + 2.0) * dist3 - (a + 3.0) * dist2 + 1.0
        w2 = a * dist3 - 5.0 * a * dist2 + 8.0 * a * dist - 4.0 * a
        return torch.where(dist <= 1.0, w1, torch.where(dist < 2.0, w2, torch.zeros_like(dist)))

    def _sample_paf_channel_cubic(self, channel_flat, x_full, y_full):
        b = channel_flat.shape[0]
        src_x = (x_full + 0.5) * (float(self.in_w) / float(self.full_w)) - 0.5
        src_y = (y_full + 0.5) * (float(self.in_h) / float(self.full_h)) - 0.5
        base_x = torch.floor(src_x)
        base_y = torch.floor(src_y)
        out = torch.zeros_like(src_x)

        for oy in (-1, 0, 1, 2):
            iy_f = base_y + float(oy)
            wy = self._cubic_kernel(torch.abs(src_y - iy_f))
            iy = torch.clamp(iy_f.to(torch.int32), 0, self.in_h - 1)
            for ox in (-1, 0, 1, 2):
                ix_f = base_x + float(ox)
                wx = self._cubic_kernel(torch.abs(src_x - ix_f))
                ix = torch.clamp(ix_f.to(torch.int32), 0, self.in_w - 1)
                flat_idx_i32 = iy * int(self.in_w) + ix
                # torch.gather requires int64 indices during export.  Public
                # outputs are still int32, but this internal helper may remain
                # int64 in the exported ONNX graph.
                values = torch.gather(channel_flat, 1, flat_idx_i32.to(torch.int64).reshape(b, -1)).reshape(
                    b, self.topk, self.topk, self.points_per_limb
                )
                out = out + values * (wx * wy)
        return out

    def score_pairs(self, top_scores, top_indices_i32, pafs):
        x_all, y_all = self._xy_from_flat(top_indices_i32)
        b = top_scores.shape[0]
        pair_scores_all = []
        pair_valid_all = []

        for part_id in range(len(BODY_PARTS_KPT_IDS)):
            kpt_a_id, kpt_b_id = BODY_PARTS_KPT_IDS[part_id]
            paf_x_id, paf_y_id = BODY_PARTS_PAF_IDS[part_id]

            ax = x_all[:, kpt_a_id, :].reshape(b, self.topk, 1)
            ay = y_all[:, kpt_a_id, :].reshape(b, self.topk, 1)
            bx = x_all[:, kpt_b_id, :].reshape(b, 1, self.topk)
            by = y_all[:, kpt_b_id, :].reshape(b, 1, self.topk)

            score_a = top_scores[:, kpt_a_id, :].reshape(b, self.topk, 1)
            score_b = top_scores[:, kpt_b_id, :].reshape(b, 1, self.topk)
            valid_kpts = (score_a > self.invalid_value * 0.5) & (score_b > self.invalid_value * 0.5)

            dx = bx - ax
            dy = by - ay
            norm = torch.sqrt(dx * dx + dy * dy)
            valid_vec = norm > 1.0e-6
            vx = dx / (norm + 1.0e-6)
            vy = dy / (norm + 1.0e-6)

            px = ax.reshape(b, self.topk, 1, 1) + dx.reshape(b, self.topk, self.topk, 1) * self.alpha
            py = ay.reshape(b, self.topk, 1, 1) + dy.reshape(b, self.topk, self.topk, 1) * self.alpha

            paf_x_flat = pafs[:, paf_x_id, :, :].reshape(b, -1)
            paf_y_flat = pafs[:, paf_y_id, :, :].reshape(b, -1)

            field_x = self._sample_paf_channel_cubic(paf_x_flat, px, py)
            field_y = self._sample_paf_channel_cubic(paf_y_flat, px, py)

            dot = field_x * vx.reshape(b, self.topk, self.topk, 1) + field_y * vy.reshape(b, self.topk, self.topk, 1)
            valid_points = dot > self.min_paf_score
            valid_points_f = valid_points.to(dot.dtype)
            valid_num = valid_points_f.sum(dim=3)
            score_sum = (dot * valid_points_f).sum(dim=3)
            affinity = score_sum / (valid_num + 1.0e-6)
            success_ratio = valid_num / float(self.points_per_limb)
            valid = valid_kpts & valid_vec & (affinity > 0.0) & (success_ratio > self.success_ratio_thr)
            scores = torch.where(valid, affinity, self._invalid_like(affinity))
            pair_scores_all.append(scores)
            pair_valid_all.append(valid.to(dot.dtype))

        return torch.stack(pair_scores_all, dim=1), torch.stack(pair_valid_all, dim=1)

    def prune_pairs(self, pair_scores, pair_valid):
        b = pair_scores.shape[0]
        flat_dim = self.topk * self.topk
        scores_flat = pair_scores.reshape(b, 19, flat_dim)
        valid_flat = pair_valid.reshape(b, 19, flat_dim) > 0.0
        masked = torch.where(valid_flat, scores_flat, self._invalid_like(scores_flat))
        limb_score, flat_idx_i64 = torch.topk(masked, k=self.limb_topm, dim=2, largest=True, sorted=True)
        limb_valid = (limb_score > self.min_pair_score).to(limb_score.dtype)
        flat_idx_i32 = flat_idx_i64.to(torch.int32)
        a_i32 = torch.div(flat_idx_i32, int(self.topk), rounding_mode="floor")
        b_i32 = flat_idx_i32 - a_i32 * int(self.topk)
        return a_i32, b_i32, limb_score, limb_valid

    def forward(self, heatmaps, pafs):
        top_scores, top_indices_i32 = self.topk_heatmaps(heatmaps)
        pair_scores, pair_valid = self.score_pairs(top_scores, top_indices_i32, pafs)
        a_idx, b_idx, limb_score, limb_valid = self.prune_pairs(pair_scores, pair_valid)
        return top_scores, top_indices_i32, a_idx, b_idx, limb_score, limb_valid


def parse_args():
    p = argparse.ArgumentParser(description="Export fp16/int32 batch-aware fused-pruned postprocess ONNX.")
    p.add_argument("--onnx", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--in-h", type=int, default=68)
    p.add_argument("--in-w", type=int, default=121)
    p.add_argument("--full-h", type=int, default=1080)
    p.add_argument("--full-w", type=int, default=1920)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limb-topm", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--min-pair-score", type=float, default=0.0)
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    model = BatchAwareFusedPrunedPostprocessFp16I32(
        batch_size=args.batch_size,
        in_h=args.in_h,
        in_w=args.in_w,
        full_h=args.full_h,
        full_w=args.full_w,
        topk=args.topk,
        limb_topm=args.limb_topm,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        heatmap_cubic_a=args.heatmap_cubic_a,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
        min_pair_score=args.min_pair_score,
    ).eval()

    b = int(args.batch_size)
    heatmaps = torch.randn(b, 18, args.in_h, args.in_w, dtype=torch.float16)
    pafs = torch.randn(b, 38, args.in_h, args.in_w, dtype=torch.float16)

    out_path = Path(args.onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (heatmaps, pafs),
            str(out_path),
            input_names=["heatmaps", "pafs"],
            output_names=[
                "top_scores",
                "top_indices",
                "limb_top_pair_a_idx",
                "limb_top_pair_b_idx",
                "limb_top_pair_score",
                "limb_top_pair_valid",
            ],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"[export-fp16-i32] saved: {out_path}")


if __name__ == "__main__":
    main()
