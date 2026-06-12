#!/usr/bin/env python3
"""
Export a true batch-aware fused-pruned postprocess ONNX.

Inputs:
  heatmaps [B,18,68,121] float32
  pafs     [B,38,68,121] float32

Outputs:
  top_scores              [B,18,K]
  top_indices             [B,18,K]
  limb_top_pair_a_idx     [B,19,M]
  limb_top_pair_b_idx     [B,19,M]
  limb_top_pair_score     [B,19,M]
  limb_top_pair_valid     [B,19,M]

This avoids the previous replicated-batch strategy. The pair scorer and pruning
tail are batch-aware:
  pair_scores [B,19,K,K] -> reshape [B,19,K*K] -> TopK(axis=2).

Heatmap candidate modes:
  full-res
      The validated baseline path. Resize the whole heatmap tensor to full
      resolution, run full-res NMS/mask, then TopK over H*W.

  smart-full-res
      Experimental candidate path. Run proposal TopK on the low-resolution
      heatmap, evaluate the manual cubic full-resolution heatmap only inside
      a local window around each proposal, and return full-resolution indices.
      This keeps the CPU-facing output contract unchanged, but it is not
      expected to be numerically identical to the validated full-res path until
      it is validated with compare_onnx_outputs.py and COCO accuracy.
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


def safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def default_name(
    *,
    batch_size: int,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    topk: int,
    limb_topm: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    heatmap_cubic_a: float,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
    paf_cubic_a: float,
    min_pair_score: float,
    heatmap_mode: str = "full-res",
    smart_proposals: int = 64,
    smart_local_radius: int = 8,
    smart_lowres_nms_radius: int = 1,
) -> str:
    mode_token = ""
    if heatmap_mode != "full-res":
        mode_token = (
            f"_hm{heatmap_mode.replace('-', '')}"
            f"_sp{int(smart_proposals)}"
            f"_lr{int(smart_local_radius)}"
            f"_lnms{int(smart_lowres_nms_radius)}"
        )
    return (
        "fused_pruned_batchaware_"
        f"b{batch_size}_{in_h}x{in_w}_to_{full_h}x{full_w}_"
        f"k{topk}_m{limb_topm}_thr{safe_float_token(threshold)}_"
        f"r{nms_radius}_{nms_impl}_ha{safe_float_token(heatmap_cubic_a)}_"
        f"p{points_per_limb}_min{safe_float_token(min_paf_score)}_"
        f"sr{safe_float_token(success_ratio_thr)}_pa{safe_float_token(paf_cubic_a)}_"
        f"mp{safe_float_token(min_pair_score)}"
        f"{mode_token}"
    )


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


class BatchAwareFusedPrunedPostprocess(nn.Module):
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
        heatmap_mode: str = "full-res",
        smart_proposals: int = 64,
        smart_local_radius: int = 8,
        smart_lowres_nms_radius: int = 1,
    ):
        super().__init__()
        if heatmap_mode not in {"full-res", "smart-full-res"}:
            raise ValueError("heatmap_mode must be 'full-res' or 'smart-full-res'")
        if nms_impl not in {"2d", "separable"}:
            raise ValueError("nms_impl must be '2d' or 'separable'")

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
        self.heatmap_cubic_a = float(heatmap_cubic_a)
        self.paf_cubic_a = float(paf_cubic_a)
        self.min_pair_score = float(min_pair_score)
        self.heatmap_mode = str(heatmap_mode)

        # The smart path keeps K/M/threshold/full-resolution output contract, but
        # changes candidate generation. It intentionally over-generates low-res
        # proposals and refines only a local full-res window around them.
        max_lowres = self.in_h * self.in_w
        self.smart_proposals = max(self.topk, min(int(smart_proposals), int(max_lowres)))
        self.smart_local_radius = max(0, int(smart_local_radius))
        self.smart_lowres_nms_radius = max(0, int(smart_lowres_nms_radius))

        x_idx, x_w = cubic_weights_and_indices(self.in_w, self.full_w, a=float(heatmap_cubic_a))
        y_idx, y_w = cubic_weights_and_indices(self.in_h, self.full_h, a=float(heatmap_cubic_a))
        self.register_buffer("x_idx", x_idx, persistent=True)
        self.register_buffer("x_w", x_w, persistent=True)
        self.register_buffer("y_idx", y_idx, persistent=True)
        self.register_buffer("y_w", y_w, persistent=True)

        rr = torch.arange(-self.smart_local_radius, self.smart_local_radius + 1, dtype=torch.float32)
        off_y, off_x = torch.meshgrid(rr, rr, indexing="ij")
        self.smart_window_area = int(off_x.numel())
        self.register_buffer("smart_off_x", off_x.reshape(1, 1, 1, self.smart_window_area), persistent=True)
        self.register_buffer("smart_off_y", off_y.reshape(1, 1, 1, self.smart_window_area), persistent=True)

        alpha = torch.arange(self.points_per_limb, dtype=torch.float32) / float(max(1, self.points_per_limb - 1))
        self.register_buffer("alpha", alpha.view(1, 1, 1, self.points_per_limb), persistent=True)

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

    def _nms_pool(self, hm, radius: int):
        r = int(radius)
        if r <= 0:
            return hm
        k = 2 * r + 1
        if self.nms_impl == "2d":
            return F.max_pool2d(hm, kernel_size=k, stride=1, padding=r)
        pooled = F.max_pool2d(hm, kernel_size=(k, 1), stride=1, padding=(r, 0))
        pooled = F.max_pool2d(pooled, kernel_size=(1, k), stride=1, padding=(0, r))
        return pooled

    def topk_heatmaps_full_res(self, heatmaps):
        hm = self.manual_cubic_resize_heatmaps(heatmaps)
        pooled = self._nms_pool(hm, self.nms_radius)
        peaks = (hm == pooled) & (hm > self.threshold)
        masked = torch.where(peaks, hm, torch.full_like(hm, -1.0e9))
        flat = masked.flatten(start_dim=2)
        return torch.topk(flat, k=self.topk, dim=2, largest=True, sorted=True)

    def _xy_from_lowres_flat(self, indices):
        indices_f = indices.to(torch.float32)
        y = torch.floor(indices_f / float(self.in_w))
        x = indices_f - y * float(self.in_w)
        return x, y

    def _sample_heatmaps_cubic_points(self, heatmaps, x_full, y_full):
        # heatmaps: [B,18,in_h,in_w], x/y: [B,18,P,S] in full-res pixel coordinates.
        b = self.batch_size
        c = 18
        src_x = (x_full + 0.5) * (float(self.in_w) / float(self.full_w)) - 0.5
        src_y = (y_full + 0.5) * (float(self.in_h) / float(self.full_h)) - 0.5
        base_x = torch.floor(src_x)
        base_y = torch.floor(src_y)

        flat = heatmaps.reshape(b * c, -1)
        out = torch.zeros_like(src_x)

        for oy in (-1, 0, 1, 2):
            iy_f = base_y + float(oy)
            wy = self._heatmap_cubic_kernel(torch.abs(src_y - iy_f))
            iy = torch.clamp(iy_f.to(torch.int64), 0, self.in_h - 1)
            for ox in (-1, 0, 1, 2):
                ix_f = base_x + float(ox)
                wx = self._heatmap_cubic_kernel(torch.abs(src_x - ix_f))
                ix = torch.clamp(ix_f.to(torch.int64), 0, self.in_w - 1)
                flat_idx = iy * int(self.in_w) + ix
                values = torch.gather(flat, 1, flat_idx.reshape(b * c, -1)).reshape(
                    b, c, self.smart_proposals, self.smart_window_area
                )
                out = out + values * (wx * wy)
        return out

    def topk_heatmaps_smart_full_res(self, heatmaps):
        # 1) Low-res proposal generation.
        pooled_lr = self._nms_pool(heatmaps, self.smart_lowres_nms_radius)
        peaks_lr = (heatmaps == pooled_lr) & (heatmaps > self.threshold)
        masked_lr = torch.where(peaks_lr, heatmaps, torch.full_like(heatmaps, -1.0e9))
        prop_scores_lr, prop_idx_lr = torch.topk(
            masked_lr.flatten(start_dim=2),
            k=self.smart_proposals,
            dim=2,
            largest=True,
            sorted=True,
        )

        # 2) Convert low-res proposal centers into full-res local windows.
        x_lr, y_lr = self._xy_from_lowres_flat(prop_idx_lr)
        cx = torch.floor((x_lr + 0.5) * (float(self.full_w) / float(self.in_w)))
        cy = torch.floor((y_lr + 0.5) * (float(self.full_h) / float(self.in_h)))
        x_full = torch.clamp(cx.unsqueeze(3) + self.smart_off_x, 0.0, float(self.full_w - 1))
        y_full = torch.clamp(cy.unsqueeze(3) + self.smart_off_y, 0.0, float(self.full_h - 1))

        # 3) Evaluate manual cubic full-res heatmap only around proposals.
        local_values = self._sample_heatmaps_cubic_points(heatmaps, x_full, y_full)
        local_scores, local_pos = torch.topk(local_values, k=1, dim=3, largest=True, sorted=True)
        local_scores = local_scores.squeeze(3)
        local_pos = local_pos.to(torch.int64)

        full_idx_grid = y_full.to(torch.int64) * int(self.full_w) + x_full.to(torch.int64)
        local_indices = torch.gather(full_idx_grid, 3, local_pos).squeeze(3)

        # 4) Keep final output contract: top K scores and full-res flat indices.
        local_scores = torch.where(local_scores > self.threshold, local_scores, torch.full_like(local_scores, -1.0e9))
        top_scores, order = torch.topk(local_scores, k=self.topk, dim=2, largest=True, sorted=True)
        top_indices = torch.gather(local_indices, 2, order)
        return top_scores, top_indices

    def topk_heatmaps(self, heatmaps):
        if self.heatmap_mode == "smart-full-res":
            return self.topk_heatmaps_smart_full_res(heatmaps)
        return self.topk_heatmaps_full_res(heatmaps)

    def _xy_from_flat(self, indices):
        indices_f = indices.to(torch.float32)
        y = torch.floor(indices_f / float(self.full_w))
        x = indices_f - y * float(self.full_w)
        return x, y

    def _heatmap_cubic_kernel(self, dist):
        a = self.heatmap_cubic_a
        dist2 = dist * dist
        dist3 = dist2 * dist
        w1 = (a + 2.0) * dist3 - (a + 3.0) * dist2 + 1.0
        w2 = a * dist3 - 5.0 * a * dist2 + 8.0 * a * dist - 4.0 * a
        return torch.where(dist <= 1.0, w1, torch.where(dist < 2.0, w2, torch.zeros_like(dist)))

    def _cubic_kernel(self, dist):
        a = self.paf_cubic_a
        dist2 = dist * dist
        dist3 = dist2 * dist
        w1 = (a + 2.0) * dist3 - (a + 3.0) * dist2 + 1.0
        w2 = a * dist3 - 5.0 * a * dist2 + 8.0 * a * dist - 4.0 * a
        return torch.where(dist <= 1.0, w1, torch.where(dist < 2.0, w2, torch.zeros_like(dist)))

    def _sample_paf_channel_cubic(self, channel_flat, x_full, y_full):
        # channel_flat: [B, in_h*in_w], x/y: [B,K,K,P]
        b = channel_flat.shape[0]
        src_x = (x_full + 0.5) * (float(self.in_w) / float(self.full_w)) - 0.5
        src_y = (y_full + 0.5) * (float(self.in_h) / float(self.full_h)) - 0.5
        base_x = torch.floor(src_x)
        base_y = torch.floor(src_y)
        out = torch.zeros_like(src_x)

        for oy in (-1, 0, 1, 2):
            iy_f = base_y + float(oy)
            wy = self._cubic_kernel(torch.abs(src_y - iy_f))
            iy = torch.clamp(iy_f.to(torch.int64), 0, self.in_h - 1)
            for ox in (-1, 0, 1, 2):
                ix_f = base_x + float(ox)
                wx = self._cubic_kernel(torch.abs(src_x - ix_f))
                ix = torch.clamp(ix_f.to(torch.int64), 0, self.in_w - 1)
                flat_idx = iy * int(self.in_w) + ix
                values = torch.gather(channel_flat, 1, flat_idx.reshape(b, -1)).reshape(
                    b, self.topk, self.topk, self.points_per_limb
                )
                out = out + values * (wx * wy)
        return out

    def score_pairs(self, top_scores, top_indices, pafs):
        x_all, y_all = self._xy_from_flat(top_indices)
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
            valid_kpts = (score_a > -1.0e8) & (score_b > -1.0e8)

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
            valid_num = valid_points.to(torch.float32).sum(dim=3)
            score_sum = (dot * valid_points.to(torch.float32)).sum(dim=3)
            affinity = score_sum / (valid_num + 1.0e-6)
            success_ratio = valid_num / float(self.points_per_limb)
            valid = valid_kpts & valid_vec & (affinity > 0.0) & (success_ratio > self.success_ratio_thr)
            scores = torch.where(valid, affinity, torch.full_like(affinity, -1.0e9))
            pair_scores_all.append(scores)
            pair_valid_all.append(valid.to(torch.float32))

        return torch.stack(pair_scores_all, dim=1), torch.stack(pair_valid_all, dim=1)

    def prune_pairs(self, pair_scores, pair_valid):
        b = pair_scores.shape[0]
        flat_dim = self.topk * self.topk
        scores_flat = pair_scores.reshape(b, 19, flat_dim)
        valid_flat = pair_valid.reshape(b, 19, flat_dim) > 0.0
        masked = torch.where(valid_flat, scores_flat, torch.full_like(scores_flat, -1.0e9))
        limb_score, flat_idx = torch.topk(masked, k=self.limb_topm, dim=2, largest=True, sorted=True)
        limb_valid = (limb_score > self.min_pair_score).to(torch.float32)
        flat_f = flat_idx.to(torch.float32)
        a_f = torch.floor(flat_f / float(self.topk))
        b_f = flat_f - a_f * float(self.topk)
        return a_f.to(torch.int64), b_f.to(torch.int64), limb_score, limb_valid

    def forward(self, heatmaps, pafs):
        top_scores, top_indices = self.topk_heatmaps(heatmaps)
        pair_scores, pair_valid = self.score_pairs(top_scores, top_indices, pafs)
        a_idx, b_idx, limb_score, limb_valid = self.prune_pairs(pair_scores, pair_valid)
        return top_scores, top_indices, a_idx, b_idx, limb_score, limb_valid


def parse_args():
    p = argparse.ArgumentParser(description="Export batch-aware fused-pruned postprocess ONNX.")
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
    p.add_argument("--heatmap-mode", choices=["full-res", "smart-full-res"], default="full-res")
    p.add_argument("--smart-proposals", type=int, default=64)
    p.add_argument("--smart-local-radius", type=int, default=8)
    p.add_argument("--smart-lowres-nms-radius", type=int, default=1)
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    model = BatchAwareFusedPrunedPostprocess(
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
        heatmap_mode=args.heatmap_mode,
        smart_proposals=args.smart_proposals,
        smart_local_radius=args.smart_local_radius,
        smart_lowres_nms_radius=args.smart_lowres_nms_radius,
    ).eval()

    b = int(args.batch_size)
    heatmaps = torch.randn(b, 18, args.in_h, args.in_w, dtype=torch.float32)
    pafs = torch.randn(b, 38, args.in_h, args.in_w, dtype=torch.float32)

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

    print(f"[export-batchaware] saved: {out_path}")
    print(f"[export-batchaware] heatmap_mode={args.heatmap_mode}")
    if args.heatmap_mode == "smart-full-res":
        print(
            "[export-batchaware] smart params: "
            f"proposals={args.smart_proposals} "
            f"local_radius={args.smart_local_radius} "
            f"lowres_nms_radius={args.smart_lowres_nms_radius}"
        )


if __name__ == "__main__":
    main()
