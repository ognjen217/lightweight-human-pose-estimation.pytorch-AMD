#!/usr/bin/env python3
"""Compare HIP2 PAF pruning against split MXR2.

This is the Step-1 correctness tool for replacing MXR2 with a custom HIP PAF
scoring/pruning backend.  The path is intentionally host-mediated first:

    input -> MXR1 -> heatmaps/pafs
    heatmaps -> external heatmap TopK backend -> top_scores/top_indices
    pafs + top_scores/top_indices -> MXR2 reference
    pafs + top_scores/top_indices -> HIP2 candidate

The comparison target is only the four MXR2 output tensors.  The heatmap TopK
stage is shared, so mismatches isolate the PAF scoring/pruning replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping

import numpy as np

try:
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from modules.external_paf_prune import PafPruneConfig, run_external_paf_prune
    from tools.compare_split_pipeline_vs_merged import _load_mxr, make_input, run_mxr1, run_mxr2, compare_output_dicts
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from modules.external_paf_prune import PafPruneConfig, run_external_paf_prune
    from tools.compare_split_pipeline_vs_merged import _load_mxr, make_input, run_mxr1, run_mxr2, compare_output_dicts


MXR2_OUTPUT_NAMES = [
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]


def write_markdown_report(path: Path, payload: Mapping[str, object]) -> None:
    lines: List[str] = []
    lines.append("# HIP2 PAF pruning vs MXR2 comparison")
    lines.append("")
    lines.append(f"- heatmap_backend: `{payload['heatmap_backend']}`")
    lines.append(f"- paf_backend: `{payload['paf_backend']}`")
    lines.append(f"- batch_size: `{payload['batch_size']}`")
    lines.append(f"- runs: `{payload['runs']}`")
    lines.append(f"- passed_all_runs: `{payload['passed_all_runs']}`")
    lines.append(f"- strict_passed_all_runs: `{payload['strict_passed_all_runs']}`")
    lines.append(f"- semantic_passed_all_runs: `{payload['semantic_passed_all_runs']}`")
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    lines.append("| stage | avg ms |")
    lines.append("|---|---:|")
    timings = payload.get("timing_ms_avg", {})
    if isinstance(timings, Mapping):
        for k, v in timings.items():
            lines.append(f"| {k} | {float(v):.4f} |")
    lines.append("")
    lines.append("## Per-run output checks")
    lines.append("")
    lines.append("| run | passed | strict | semantic | output | allclose | exact | max_abs | mean_abs | mismatches |")
    lines.append("|---:|:---:|:---:|:---:|---|:---:|:---:|---:|---:|---:|")
    for run in payload.get("runs_detail", []):
        if not isinstance(run, Mapping):
            continue
        comp = run.get("comparison", {})
        if not isinstance(comp, Mapping):
            continue
        outs = comp.get("outputs", {})
        if not isinstance(outs, Mapping):
            continue
        for name, metrics in outs.items():
            if not isinstance(metrics, Mapping):
                continue
            lines.append(
                f"| {run.get('run_index')} | {comp.get('passed')} | {comp.get('strict_passed')} | "
                f"{comp.get('semantic_passed')} | {name} | {metrics.get('allclose')} | {metrics.get('exact')} | "
                f"{metrics.get('max_abs')} | {metrics.get('mean_abs')} | {metrics.get('mismatch_count')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare HIP2 PAF pruning backend against split MXR2.")
    p.add_argument("--mxr1", required=True, help="Split MXR1: input -> heatmaps/pafs")
    p.add_argument("--mxr2", required=True, help="Split MXR2 reference: pafs + topk -> limb pair tensors")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--heatmap-backend", choices=["torch_manual", "torch_bicubic", "hip_host", "hip_host_fused", "hip_host_smart"], default="hip_host_smart")
    p.add_argument("--paf-backend", choices=["hip_host"], default="hip_host")
    p.add_argument("--device", default="", help="PyTorch device for torch heatmap backends. Empty = auto.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--full-h", type=int, default=1080)
    p.add_argument("--full-w", type=int, default=1920)
    p.add_argument("--in-h", type=int, default=68)
    p.add_argument("--in-w", type=int, default=121)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limb-topm", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", default="separable")
    p.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--min-pair-score", type=float, default=0.0)
    p.add_argument("--ignore-invalid-topk-indices", action="store_true", help="Unused here, kept for compare_output_dicts compatibility.")
    p.add_argument("--invalid-score-threshold", type=float, default=-1.0e8)
    p.add_argument("--json", default="outputs/split_pipeline_compare/hip2_vs_mxr2.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/hip2_vs_mxr2.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mxr1 = _load_mxr(args.mxr1)
    mxr2 = _load_mxr(args.mxr2)

    heat_cfg = HeatmapTopKConfig(
        batch_size=int(args.batch_size),
        in_h=int(args.in_h),
        in_w=int(args.in_w),
        full_h=int(args.full_h),
        full_w=int(args.full_w),
        channels=18,
        topk=int(args.topk),
        threshold=float(args.threshold),
        nms_radius=int(args.nms_radius),
        nms_impl=str(args.nms_impl),
        cubic_a=float(args.heatmap_cubic_a),
    )
    paf_cfg = PafPruneConfig(
        batch_size=int(args.batch_size),
        topk=int(args.topk),
        limb_topm=int(args.limb_topm),
        in_h=int(args.in_h),
        in_w=int(args.in_w),
        full_h=int(args.full_h),
        full_w=int(args.full_w),
        points_per_limb=int(args.points_per_limb),
        min_paf_score=float(args.min_paf_score),
        success_ratio_thr=float(args.success_ratio_thr),
        min_pair_score=float(args.min_pair_score),
        paf_cubic_a=float(args.paf_cubic_a),
    )

    runs_detail: List[Dict[str, object]] = []
    timing_acc: MutableMapping[str, List[float]] = {"mxr1": [], "heatmap_topk": [], "mxr2": [], "hip2": []}

    for run_idx in range(int(args.runs)):
        input_batch = make_input(int(args.batch_size), int(args.seed) + run_idx, dtype=str(args.input_dtype))

        t0 = time.perf_counter()
        heatmaps, pafs = run_mxr1(mxr1, input_batch)
        t1 = time.perf_counter()

        top_scores, top_indices = run_external_heatmap_topk(
            heatmaps,
            heat_cfg,
            backend=args.heatmap_backend,
            device=args.device or None,
        )
        t2 = time.perf_counter()

        mxr2_out = run_mxr2(mxr2, pafs, top_scores, top_indices)
        t3 = time.perf_counter()

        a_idx, b_idx, pair_score, pair_valid = run_external_paf_prune(
            pafs,
            top_scores,
            top_indices,
            paf_cfg,
            backend=args.paf_backend,
        )
        t4 = time.perf_counter()

        hip2_out = {
            "limb_top_pair_a_idx": a_idx,
            "limb_top_pair_b_idx": b_idx,
            "limb_top_pair_score": pair_score,
            "limb_top_pair_valid": pair_valid,
        }
        comparison = compare_output_dicts(
            mxr2_out,
            hip2_out,
            MXR2_OUTPUT_NAMES,
            atol=float(args.atol),
            rtol=float(args.rtol),
            ignore_invalid_topk_indices=False,
            invalid_score_threshold=float(args.invalid_score_threshold),
        )

        timing_acc["mxr1"].append((t1 - t0) * 1000.0)
        timing_acc["heatmap_topk"].append((t2 - t1) * 1000.0)
        timing_acc["mxr2"].append((t3 - t2) * 1000.0)
        timing_acc["hip2"].append((t4 - t3) * 1000.0)

        runs_detail.append({
            "run_index": run_idx,
            "seed": int(args.seed) + run_idx,
            "comparison": comparison,
            "timing_ms": {k: v[-1] for k, v in timing_acc.items()},
        })
        print(
            f"[run {run_idx}] passed={comparison['passed']} strict={comparison['strict_passed']} "
            f"semantic={comparison['semantic_passed']} mxr2_ms={timing_acc['mxr2'][-1]:.3f} hip2_ms={timing_acc['hip2'][-1]:.3f}"
        )

    payload: Dict[str, object] = {
        "mxr1": str(args.mxr1),
        "mxr2": str(args.mxr2),
        "heatmap_backend": str(args.heatmap_backend),
        "paf_backend": str(args.paf_backend),
        "batch_size": int(args.batch_size),
        "runs": int(args.runs),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "passed_all_runs": all(bool(r["comparison"]["passed"]) for r in runs_detail),
        "strict_passed_all_runs": all(bool(r["comparison"]["strict_passed"]) for r in runs_detail),
        "semantic_passed_all_runs": all(bool(r["comparison"]["semantic_passed"]) for r in runs_detail),
        "timing_ms_avg": {k: float(np.mean(v)) if v else math.nan for k, v in timing_acc.items()},
        "runs_detail": runs_detail,
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[write] {json_path}")

    md_path = Path(args.markdown)
    write_markdown_report(md_path, payload)
    print(f"[write] {md_path}")

    if not payload["passed_all_runs"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
