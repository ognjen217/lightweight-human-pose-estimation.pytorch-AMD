#!/usr/bin/env python3
"""Run MXR1 -> smart HIP heatmap -> MXR2 on real frames without strict compare.

The smart-full-res backend is approximate by design, so full-res merged-output
strict comparison is not the primary validation signal.  This runner checks that
it produces non-empty TopK outputs and that MXR2 accepts the output contract.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, MutableMapping

import numpy as np

try:
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from tools.compare_split_pipeline_hip_host_real_frame import read_video_batch
    from tools.compare_split_pipeline_vs_merged import _load_mxr, run_mxr1, run_mxr2
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from tools.compare_split_pipeline_hip_host_real_frame import read_video_batch
    from tools.compare_split_pipeline_vs_merged import _load_mxr, run_mxr1, run_mxr2


def parse_args():
    p = argparse.ArgumentParser(description="Run real-frame split pipeline using smart HIP heatmap backend")
    p.add_argument("--mxr1", required=True)
    p.add_argument("--mxr2", required=True)
    p.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--runs", type=int, default=3)
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
    p.add_argument("--json", default="outputs/split_pipeline_compare/b4_hip_smart_split_sanity_sp32_lr4_r3.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/b4_hip_smart_split_sanity_sp32_lr4_r3.md")
    return p.parse_args()


def write_md(path: Path, payload: Dict[str, object]):
    lines = [
        "# Smart HIP split pipeline sanity", "",
        f"- backend: `{payload['backend']}`",
        f"- video: `{payload['video']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- runs: `{payload['runs']}`", "",
        "## Timings", "", "| stage | avg ms |", "|---|---:|",
    ]
    for k, v in payload["timing_ms_avg"].items():
        lines.append(f"| {k} | {float(v):.4f} |")
    lines += ["", "## Output health", "", "| metric | value |", "|---|---:|"]
    for k, v in payload["health_avg"].items():
        lines.append(f"| {k} | {float(v):.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    mxr1 = _load_mxr(args.mxr1)
    mxr2 = _load_mxr(args.mxr2)
    cfg = HeatmapTopKConfig(
        batch_size=args.batch_size,
        topk=args.topk,
        threshold=args.threshold,
        smart_proposals=args.smart_proposals,
        smart_local_radius=args.smart_local_radius,
        smart_lowres_nms_radius=args.smart_lowres_nms_radius,
    )
    timing_acc: MutableMapping[str, List[float]] = {"preprocess": [], "mxr1": [], "smart_heatmap": [], "mxr2": [], "split_total": []}
    health_rows: List[Dict[str, float]] = []
    details = []
    for i in range(args.runs):
        frame_index = args.frame_index + i * args.run_frame_stride
        t0 = time.perf_counter()
        x, meta = read_video_batch(args.video, batch_size=args.batch_size, frame_index=frame_index, batch_stride=args.batch_frame_stride, target_w=args.target_w, target_h=args.target_h, dtype=args.input_dtype)
        t1 = time.perf_counter()
        split_t0 = time.perf_counter()
        heatmaps, pafs = run_mxr1(mxr1, x)
        t2 = time.perf_counter()
        top_scores, top_indices = run_external_heatmap_topk(heatmaps, cfg, backend="hip_host_smart")
        t3 = time.perf_counter()
        mxr2_out = run_mxr2(mxr2, pafs, top_scores, top_indices)
        t4 = time.perf_counter()
        health = {
            "valid_topk_count": float(np.sum(top_scores > -1.0e8)),
            "top_scores_max": float(np.max(top_scores)),
            "top_scores_min_valid": float(np.min(top_scores[top_scores > -1.0e8])) if np.any(top_scores > -1.0e8) else -1.0e9,
            "limb_valid_count": float(np.sum(mxr2_out.get("limb_top_pair_valid", 0) > 0.5)) if "limb_top_pair_valid" in mxr2_out else 0.0,
        }
        timing = {
            "preprocess": (t1 - t0) * 1000.0,
            "mxr1": (t2 - split_t0) * 1000.0,
            "smart_heatmap": (t3 - t2) * 1000.0,
            "mxr2": (t4 - t3) * 1000.0,
            "split_total": (t4 - split_t0) * 1000.0,
        }
        for k, v in timing.items():
            timing_acc[k].append(v)
        health_rows.append(health)
        details.append({"run_index": i, "start_frame_index": frame_index, "batch_frames": meta, "timing_ms": timing, "health": health})
        print(f"[run {i}] split_ms={timing['split_total']:.3f} smart_ms={timing['smart_heatmap']:.3f} valid_topk={int(health['valid_topk_count'])} limb_valid={int(health['limb_valid_count'])}")

    payload = {
        "backend": "hip_host_smart_real_frame",
        "mxr1": args.mxr1,
        "mxr2": args.mxr2,
        "video": args.video,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "smart_proposals": args.smart_proposals,
        "smart_local_radius": args.smart_local_radius,
        "smart_lowres_nms_radius": args.smart_lowres_nms_radius,
        "timing_ms_avg": {k: float(np.mean(v)) if v else 0.0 for k, v in timing_acc.items()},
        "health_avg": {k: float(np.mean([row[k] for row in health_rows])) if health_rows else 0.0 for k in health_rows[0].keys()} if health_rows else {},
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
