#!/usr/bin/env python3
"""Export batch-aware fused-pruned postprocess ONNX and run conservative cleanup.

This wrapper keeps the original exporter unchanged. It first exports a raw ONNX
using tools/export_batchaware_fused_pruned_postprocess.py, then runs
modules.onnx_graph_optimizer.optimize_onnx_for_migraphx into the requested final
--onnx path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export and optimize batch-aware fused-pruned postprocess ONNX.")
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
    p.add_argument("--onnxsim", action="store_true", help="Also run onnxsim if installed. Disabled by default because it can be slow.")
    p.add_argument("--keep-raw-onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    final_path = Path(args.onnx)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = final_path.with_suffix(".raw.onnx")

    exporter = Path(__file__).with_name("export_batchaware_fused_pruned_postprocess.py")
    cmd = [
        sys.executable,
        str(exporter),
        "--onnx", str(raw_path),
        "--batch-size", str(args.batch_size),
        "--in-h", str(args.in_h),
        "--in-w", str(args.in_w),
        "--full-h", str(args.full_h),
        "--full-w", str(args.full_w),
        "--topk", str(args.topk),
        "--limb-topm", str(args.limb_topm),
        "--threshold", str(args.threshold),
        "--nms-radius", str(args.nms_radius),
        "--nms-impl", str(args.nms_impl),
        "--heatmap-cubic-a", str(args.heatmap_cubic_a),
        "--points-per-limb", str(args.points_per_limb),
        "--min-paf-score", str(args.min_paf_score),
        "--success-ratio-thr", str(args.success_ratio_thr),
        "--paf-cubic-a", str(args.paf_cubic_a),
        "--min-pair-score", str(args.min_pair_score),
        "--heatmap-mode", str(args.heatmap_mode),
        "--smart-proposals", str(args.smart_proposals),
        "--smart-local-radius", str(args.smart_local_radius),
        "--smart-lowres-nms-radius", str(args.smart_lowres_nms_radius),
        "--opset", str(args.opset),
    ]

    print("[export-opt] exporting raw ONNX")
    print("[export-opt] " + " ".join(cmd))
    subprocess.check_call(cmd)

    from modules.onnx_graph_optimizer import optimize_onnx_for_migraphx

    report_path = final_path.with_suffix(".opt.json")
    print(f"[export-opt] optimizing: {raw_path} -> {final_path}")
    report = optimize_onnx_for_migraphx(
        raw_path,
        final_path,
        use_onnxoptimizer=True,
        use_shape_inference=True,
        use_onnxsim=bool(args.onnxsim),
        report_json=report_path,
        input_shapes={
            "heatmaps": [int(args.batch_size), 18, int(args.in_h), int(args.in_w)],
            "pafs": [int(args.batch_size), 38, int(args.in_h), int(args.in_w)],
        },
    )
    print(
        "[export-opt] nodes "
        f"{report['before']['num_nodes']} -> {report['after']['num_nodes']} "
        f"(delta {report['node_delta']})"
    )
    print(
        "[export-opt] initializers "
        f"{report['before']['num_initializers']} -> {report['after']['num_initializers']} "
        f"(delta {report['initializer_delta']})"
    )
    print(f"[export-opt] report: {report_path}")
    print(f"[export-opt] saved: {final_path}")

    if not args.keep_raw_onnx:
        try:
            raw_path.unlink()
            print(f"[export-opt] removed raw ONNX: {raw_path}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
