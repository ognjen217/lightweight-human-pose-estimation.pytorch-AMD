#!/usr/bin/env python3
"""Benchmark several implemented postprocess modes on a video input.

This script reports video timing only. COCO AP/AR evaluation should be run with
an existing validation harness and then joined with this timing table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from video_val import run_benchmarked_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare postprocess variants")
    parser.add_argument("--video", default="cctv_1280x720_24fps_original.mp4")
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument("--migraphx-nms-mxr", default="models/heatmap_nms_head.mxr")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output-csv", default="benchmark_migraphx_postprocess.csv")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard", "optimized_batch_k20_findnonzero_v1", "gpu-nms", "migraphx-nms", "migraphx-nms-k20"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for variant in args.variants:
        print("\n" + "=" * 120)
        print(f"Running variant: {variant}")
        print("=" * 120)
        summary = run_benchmarked_session(
            video_path=args.video,
            model_path=args.model,
            postprocess=variant,
            migraphx_nms_mxr=args.migraphx_nms_mxr,
            no_draw=True,
            no_write=True,
            max_frames=args.max_frames,
        )
        if not summary:
            continue

        rows.append({
            "variant": variant,
            "preprocess_ms": summary.get("preprocess_mean", 0.0),
            "infer_ms": summary.get("infer_mean", 0.0),
            "decode_ms": summary.get("decode_mean", 0.0),
            "hm_resize_ms": summary.get("resize_heatmaps_mean", 0.0),
            "paf_resize_ms": summary.get("resize_pafs_mean", 0.0),
            "mx_nms_ms": summary.get("mx_nms_mean", 0.0),
            "extract_ms": summary.get("extract_keypoints_mean", 0.0),
            "extract_from_mask_ms": summary.get("extract_from_mask_mean", 0.0),
            "group_ms": summary.get("group_keypoints_mean", 0.0),
            "post_total_ms": summary.get("total_postprocess_mean", 0.0),
            "total_frame_ms": summary.get("total_frame_mean", 0.0),
            "p95_total_frame_ms": summary.get("total_frame_p95", 0.0),
            "AP": "",
            "AP50": "",
            "AP75": "",
            "AR": "",
        })

    if not rows:
        print("No benchmark rows generated.")
        return

    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nComparison summary:")
    header = (
        f"{'variant':34s} {'pre':>8s} {'infer':>8s} {'decode':>8s} {'hm_rs':>8s} "
        f"{'paf_rs':>8s} {'mx_nms':>8s} {'extract':>8s} {'mask_ext':>9s} "
        f"{'group':>8s} {'post':>8s} {'frame':>8s} {'p95':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['variant']:34s} "
            f"{row['preprocess_ms']:8.2f} {row['infer_ms']:8.2f} {row['decode_ms']:8.2f} "
            f"{row['hm_resize_ms']:8.2f} {row['paf_resize_ms']:8.2f} {row['mx_nms_ms']:8.2f} "
            f"{row['extract_ms']:8.2f} {row['extract_from_mask_ms']:9.2f} {row['group_ms']:8.2f} "
            f"{row['post_total_ms']:8.2f} {row['total_frame_ms']:8.2f} {row['p95_total_frame_ms']:8.2f}"
        )

    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()
