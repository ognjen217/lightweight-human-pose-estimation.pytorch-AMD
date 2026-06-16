#!/usr/bin/env python3
"""Compare split MXR pipeline using native HIP host-mediated heatmap backend.

This is a focused wrapper around compare_split_pipeline_vs_merged.py while the
main comparison CLI still exposes only the original backend choices.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, MutableMapping

import numpy as np

try:
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from tools.compare_split_pipeline_vs_merged import (
        MERGED_OUTPUT_NAMES,
        compare_output_dicts,
        make_input,
        run_merged,
        run_mxr1,
        run_mxr2,
        _load_mxr,
        write_markdown_report,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from tools.compare_split_pipeline_vs_merged import (
        MERGED_OUTPUT_NAMES,
        compare_output_dicts,
        make_input,
        run_merged,
        run_mxr1,
        run_mxr2,
        _load_mxr,
        write_markdown_report,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare split MXR pipeline with HIP host heatmap backend.")
    p.add_argument("--merged-mxr", required=True)
    p.add_argument("--mxr1", required=True)
    p.add_argument("--mxr2", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--full-h", type=int, default=1080)
    p.add_argument("--full-w", type=int, default=1920)
    p.add_argument("--in-h", type=int, default=68)
    p.add_argument("--in-w", type=int, default=121)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)
    p.add_argument("--ignore-invalid-topk-indices", action="store_true")
    p.add_argument("--invalid-score-threshold", type=float, default=-1.0e8)
    p.add_argument("--json", default="outputs/split_pipeline_compare/b4_hip_host.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/b4_hip_host.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged = _load_mxr(args.merged_mxr)
    mxr1 = _load_mxr(args.mxr1)
    mxr2 = _load_mxr(args.mxr2)

    cfg = HeatmapTopKConfig(
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
        cubic_a=float(args.cubic_a),
    )

    runs_detail: List[Dict[str, object]] = []
    timing_acc: MutableMapping[str, List[float]] = {
        "merged": [],
        "mxr1": [],
        "external_heatmap": [],
        "mxr2": [],
        "split_total": [],
    }

    for i in range(int(args.runs)):
        input_batch = make_input(int(args.batch_size), int(args.seed) + i, dtype=str(args.input_dtype))

        t0 = time.perf_counter()
        merged_out = run_merged(merged, input_batch)
        t1 = time.perf_counter()

        split_t0 = time.perf_counter()
        heatmaps, pafs = run_mxr1(mxr1, input_batch)
        t2 = time.perf_counter()

        top_scores, top_indices = run_external_heatmap_topk(heatmaps, cfg, backend="hip_host")
        t3 = time.perf_counter()

        mxr2_out = run_mxr2(mxr2, pafs, top_scores, top_indices)
        t4 = time.perf_counter()

        split_out: Dict[str, np.ndarray] = {
            "top_scores": top_scores,
            "top_indices": top_indices,
            **mxr2_out,
        }
        comparison = compare_output_dicts(
            merged_out,
            split_out,
            MERGED_OUTPUT_NAMES,
            atol=float(args.atol),
            rtol=float(args.rtol),
            ignore_invalid_topk_indices=bool(args.ignore_invalid_topk_indices),
            invalid_score_threshold=float(args.invalid_score_threshold),
        )

        timing_acc["merged"].append((t1 - t0) * 1000.0)
        timing_acc["mxr1"].append((t2 - split_t0) * 1000.0)
        timing_acc["external_heatmap"].append((t3 - t2) * 1000.0)
        timing_acc["mxr2"].append((t4 - t3) * 1000.0)
        timing_acc["split_total"].append((t4 - split_t0) * 1000.0)

        runs_detail.append({
            "run_index": i,
            "seed": int(args.seed) + i,
            "comparison": comparison,
            "timing_ms": {k: v[-1] for k, v in timing_acc.items()},
        })
        print(
            f"[run {i}] passed={comparison['passed']} strict={comparison['strict_passed']} "
            f"semantic={comparison['semantic_passed']} exact={comparison['exact']} "
            f"merged_ms={timing_acc['merged'][-1]:.3f} split_ms={timing_acc['split_total'][-1]:.3f} "
            f"hip_heatmap_ms={timing_acc['external_heatmap'][-1]:.3f}"
        )
        sem = comparison.get("topk_index_semantics")
        if isinstance(sem, dict):
            print(
                f"        topk_valid_mismatch={sem.get('valid_index_mismatch_count')} "
                f"topk_invalid_mismatch={sem.get('invalid_index_mismatch_count')} "
                f"topk_total_mismatch={sem.get('total_index_mismatch_count')}"
            )

    payload: Dict[str, object] = {
        "merged_mxr": str(args.merged_mxr),
        "mxr1": str(args.mxr1),
        "mxr2": str(args.mxr2),
        "backend": "hip_host",
        "batch_size": int(args.batch_size),
        "runs": int(args.runs),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "ignore_invalid_topk_indices": bool(args.ignore_invalid_topk_indices),
        "invalid_score_threshold": float(args.invalid_score_threshold),
        "passed_all_runs": all(bool(r["comparison"]["passed"]) for r in runs_detail),
        "strict_passed_all_runs": all(bool(r["comparison"]["strict_passed"]) for r in runs_detail),
        "semantic_passed_all_runs": all(bool(r["comparison"]["semantic_passed"]) for r in runs_detail),
        "exact_all_runs": all(bool(r["comparison"]["exact"]) for r in runs_detail),
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
