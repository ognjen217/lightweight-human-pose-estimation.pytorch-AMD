#!/usr/bin/env python3
"""Profile the experimental E4 smart-full-res HIP heatmap TopK backend.

Profile field mapping for the generic HeatmapTopKHipProfile:
  resize_ms   -> low-resolution proposal selection
  vertical_ms -> local full-resolution refinement
  topk_ms     -> final TopK over refined proposals
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend, HipHeatmapTopKSmartShape
    from tools.profile_heatmap_topk_hip_real_frame import avg, load_mxr, p95, read_batch, run_mxr1
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend, HipHeatmapTopKSmartShape
    from tools.profile_heatmap_topk_hip_real_frame import avg, load_mxr, p95, read_batch, run_mxr1


PROFILE_KEYS = [
    "h2d_ms", "resize_ms", "vertical_ms", "horizontal_ms", "topk_ms",
    "d2h_scores_ms", "d2h_indices_ms", "device_total_ms", "total_ms",
]


def parse_args():
    p = argparse.ArgumentParser(description="Profile experimental smart-full-res HIP heatmap TopK stages on real frames")
    p.add_argument("--mxr1", required=True)
    p.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--run-frame-stride", type=int, default=24)
    p.add_argument("--batch-frame-stride", type=int, default=1)
    p.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--target-w", type=int, default=968)
    p.add_argument("--target-h", type=int, default=544)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--smart-proposals", type=int, default=32)
    p.add_argument("--smart-local-radius", type=int, default=4)
    p.add_argument("--smart-lowres-nms-radius", type=int, default=1)
    p.add_argument("--json", default="outputs/split_pipeline_compare/b4_hip_smart_stage_profile_sp32_lr4_r3.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/b4_hip_smart_stage_profile_sp32_lr4_r3.md")
    return p.parse_args()


def write_md(path: Path, payload: Dict[str, object]):
    avg_payload = payload["profile_ms_avg"]
    p95_payload = payload["profile_ms_p95"]
    labels = {
        "h2d_ms": "h2d_ms",
        "resize_ms": "proposal_lowres_ms",
        "vertical_ms": "local_refine_fullres_ms",
        "horizontal_ms": "unused_horizontal_ms",
        "topk_ms": "final_topk_ms",
        "d2h_scores_ms": "d2h_scores_ms",
        "d2h_indices_ms": "d2h_indices_ms",
        "device_total_ms": "device_total_ms",
        "total_ms": "total_ms",
    }
    lines = [
        "# Smart HIP heatmap TopK stage profile", "",
        f"- backend: `{payload['backend']}`",
        f"- video: `{payload['video']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- runs: `{payload['runs']}`",
        f"- smart_proposals: `{payload['smart_proposals']}`",
        f"- smart_local_radius: `{payload['smart_local_radius']}`",
        f"- smart_lowres_nms_radius: `{payload['smart_lowres_nms_radius']}`",
        "", "| stage | avg ms | p95 ms |", "|---|---:|---:|",
    ]
    for key in PROFILE_KEYS:
        lines.append(f"| {labels[key]} | {avg_payload[key]:.4f} | {p95_payload[key]:.4f} |")
    lines += ["", "## Context", "", "| metric | avg |", "|---|---:|"]
    for k, v in payload["context_avg"].items():
        lines.append(f"| {k} | {float(v):.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    mxr1 = load_mxr(args.mxr1)
    backend = HipHeatmapTopKBackend()
    shape = HipHeatmapTopKSmartShape(
        batch=args.batch_size,
        channels=18,
        topk=args.topk,
        threshold=args.threshold,
        smart_proposals=args.smart_proposals,
        smart_local_radius=args.smart_local_radius,
        lowres_nms_radius=args.smart_lowres_nms_radius,
    )

    details = []
    profiles: List[Dict[str, float]] = []
    contexts: List[Dict[str, float]] = []
    for i in range(args.warmup + args.runs):
        measured = i >= args.warmup
        frame_index = args.frame_index + i * args.run_frame_stride
        t0 = time.perf_counter()
        x, meta = read_batch(args.video, args.batch_size, frame_index, args.batch_frame_stride, args.target_w, args.target_h, args.input_dtype)
        t1 = time.perf_counter()
        heatmaps, _ = run_mxr1(mxr1, x)
        t2 = time.perf_counter()
        scores, _indices, prof = backend.run_host_smart_profile(heatmaps, shape)
        t3 = time.perf_counter()
        pd = prof.as_dict()
        ctx = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "mxr1_ms": (t2 - t1) * 1000.0,
            "python_profile_call_ms": (t3 - t2) * 1000.0,
            "valid_topk_count": float(np.sum(scores > -1.0e8)),
            "top_scores_max": float(np.max(scores)),
            "top_scores_min_valid": float(np.min(scores[scores > -1.0e8])) if np.any(scores > -1.0e8) else -1.0e9,
        }
        label = "warmup" if not measured else f"run {i - args.warmup}"
        print(
            f"[{label}] total={pd['total_ms']:.3f} device={pd['device_total_ms']:.3f} "
            f"proposal={pd['resize_ms']:.3f} refine={pd['vertical_ms']:.3f} "
            f"final_topk={pd['topk_ms']:.3f} valid={int(ctx['valid_topk_count'])}"
        )
        details.append({"measured": measured, "frame_index": frame_index, "batch_frames": meta, "profile_ms": pd, "context": ctx})
        if measured:
            profiles.append(pd)
            contexts.append(ctx)

    payload = {
        "backend": "hip_host_smart_profile",
        "mxr1": args.mxr1,
        "video": args.video,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "warmup": args.warmup,
        "smart_proposals": args.smart_proposals,
        "smart_local_radius": args.smart_local_radius,
        "smart_lowres_nms_radius": args.smart_lowres_nms_radius,
        "profile_ms_avg": {k: avg(profiles, k) for k in PROFILE_KEYS},
        "profile_ms_p95": {k: p95(profiles, k) for k in PROFILE_KEYS},
        "context_avg": {k: avg(contexts, k) for k in contexts[0].keys()} if contexts else {},
        "runs_detail": details,
    }
    jp = Path(args.json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2))
    print(f"[write] {jp}")
    mp = Path(args.markdown)
    write_md(mp, payload)
    print(f"[write] {mp}")


if __name__ == "__main__":
    main()
