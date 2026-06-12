#!/usr/bin/env python3
"""Compare split MXR1 + external heatmap + MXR2 against merged baseline.

This is the first correctness tool for the proposed split architecture:

    merged baseline:
        input -> merged MXR -> six final tensors

    split path:
        input -> MXR1 -> heatmaps/pafs
        heatmaps -> external heatmap TopK backend -> top_scores/top_indices
        pafs + top_scores/top_indices -> MXR2 -> pruned limb tensors

The initial implementation is intentionally host-mediated.  It proves graph
semantics and output contracts before a true GPU-resident handoff is added.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
except ModuleNotFoundError:  # pragma: no cover
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk


MERGED_OUTPUT_NAMES = [
    "top_scores",
    "top_indices",
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]

SPLIT_MXR2_OUTPUT_NAMES = [
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]


def _as_numpy(x) -> np.ndarray:
    """Best-effort conversion from MIGraphX/Python outputs to numpy."""

    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    try:
        return np.ascontiguousarray(np.asarray(x))
    except Exception:
        pass
    # Some MIGraphX argument wrappers expose host data through helper methods in
    # different versions.  Keep this fallback defensive so the script remains
    # usable across ROCm/MIGraphX installs.
    for attr in ("to_host", "get_data", "data"):
        fn = getattr(x, attr, None)
        if callable(fn):
            try:
                return np.ascontiguousarray(np.asarray(fn()))
            except Exception:
                continue
    raise TypeError(f"Could not convert output of type {type(x)!r} to numpy")


def _load_mxr(path: str | Path):
    import migraphx  # type: ignore

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return migraphx.load(str(path))


def _migraphx_argument(arr: np.ndarray):
    import migraphx  # type: ignore

    return migraphx.argument(np.ascontiguousarray(arr))


def run_program(program, inputs: Mapping[str, np.ndarray]) -> List[np.ndarray]:
    args = {name: _migraphx_argument(arr) for name, arr in inputs.items()}
    outputs = program.run(args)
    return [_as_numpy(o) for o in outputs]


def run_merged(program, input_batch: np.ndarray) -> Dict[str, np.ndarray]:
    outputs = run_program(program, {"input": input_batch})
    if len(outputs) != len(MERGED_OUTPUT_NAMES):
        raise RuntimeError(f"Merged model returned {len(outputs)} outputs, expected {len(MERGED_OUTPUT_NAMES)}")
    return dict(zip(MERGED_OUTPUT_NAMES, outputs))


def run_mxr1(program, input_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    outputs = run_program(program, {"input": input_batch})
    if len(outputs) != 2:
        raise RuntimeError(f"MXR1 returned {len(outputs)} outputs, expected 2: heatmaps, pafs")
    heatmaps, pafs = outputs
    return np.ascontiguousarray(heatmaps.astype(np.float32, copy=False)), np.ascontiguousarray(pafs.astype(np.float32, copy=False))


def run_mxr2(program, pafs: np.ndarray, top_scores: np.ndarray, top_indices: np.ndarray) -> Dict[str, np.ndarray]:
    outputs = run_program(program, {
        "pafs": np.ascontiguousarray(pafs.astype(np.float32, copy=False)),
        "top_scores": np.ascontiguousarray(top_scores.astype(np.float32, copy=False)),
        "top_indices": np.ascontiguousarray(top_indices.astype(np.int64, copy=False)),
    })
    if len(outputs) != len(SPLIT_MXR2_OUTPUT_NAMES):
        raise RuntimeError(f"MXR2 returned {len(outputs)} outputs, expected {len(SPLIT_MXR2_OUTPUT_NAMES)}")
    return dict(zip(SPLIT_MXR2_OUTPUT_NAMES, outputs))


def make_input(batch_size: int, seed: int, dtype: str = "float16") -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((int(batch_size), 3, 544, 968), dtype=np.float32)
    if dtype == "float16":
        return np.ascontiguousarray(x.astype(np.float16))
    if dtype == "float32":
        return np.ascontiguousarray(x.astype(np.float32))
    raise ValueError(f"Unsupported input dtype: {dtype}")


def compare_arrays(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float) -> Dict[str, object]:
    a = np.asarray(a)
    b = np.asarray(b)
    out: Dict[str, object] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "same_shape": tuple(a.shape) == tuple(b.shape),
    }
    if tuple(a.shape) != tuple(b.shape):
        out.update({
            "exact": False,
            "allclose": False,
            "max_abs": None,
            "mean_abs": None,
            "mismatch_count": None,
        })
        return out

    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(b.dtype, np.integer):
        exact = bool(np.array_equal(a, b))
        mismatch_count = int(np.count_nonzero(a != b))
        max_abs = int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0
        out.update({
            "exact": exact,
            "allclose": exact,
            "max_abs": max_abs,
            "mean_abs": float(np.mean(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0.0,
            "mismatch_count": mismatch_count,
        })
        return out

    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    exact = bool(np.array_equal(a, b))
    allclose = bool(np.allclose(a, b, atol=float(atol), rtol=float(rtol), equal_nan=True))
    out.update({
        "exact": exact,
        "allclose": allclose,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "mismatch_count": int(np.count_nonzero(~np.isclose(a, b, atol=float(atol), rtol=float(rtol), equal_nan=True))),
    })
    return out


def compare_topk_index_semantics(
    ref_scores: np.ndarray,
    ref_indices: np.ndarray,
    cand_scores: np.ndarray,
    cand_indices: np.ndarray,
    *,
    invalid_score_threshold: float = -1.0e8,
) -> Dict[str, object]:
    """Classify TopK index mismatches into valid and invalid/tie slots.

    MIGraphX TopK and PyTorch topk can legally pick different indices when the
    score values are tied.  In this graph the most common tie is the invalid
    sentinel value -1e9.  Those invalid index mismatches are not semantically
    meaningful because downstream PAF scoring rejects them via the score mask.
    """

    rs = np.asarray(ref_scores)
    cs = np.asarray(cand_scores)
    ri = np.asarray(ref_indices)
    ci = np.asarray(cand_indices)
    if tuple(rs.shape) != tuple(cs.shape) or tuple(ri.shape) != tuple(ci.shape) or tuple(rs.shape) != tuple(ri.shape):
        return {
            "same_shape": False,
            "semantic_match": False,
            "reason": "shape mismatch between scores/indices",
            "valid_topk_count": None,
            "valid_index_mismatch_count": None,
            "invalid_index_mismatch_count": None,
        }

    valid_ref = rs > float(invalid_score_threshold)
    valid_cand = cs > float(invalid_score_threshold)
    valid_any = valid_ref | valid_cand
    valid_mask_match = bool(np.array_equal(valid_ref, valid_cand))
    index_mismatch = ri != ci
    valid_mismatch = index_mismatch & valid_any
    invalid_mismatch = index_mismatch & (~valid_any)

    valid_count = int(np.count_nonzero(valid_any))
    valid_index_mismatch_count = int(np.count_nonzero(valid_mismatch))
    invalid_index_mismatch_count = int(np.count_nonzero(invalid_mismatch))

    return {
        "same_shape": True,
        "semantic_match": bool(valid_mask_match and valid_index_mismatch_count == 0),
        "valid_mask_match": valid_mask_match,
        "valid_topk_count": valid_count,
        "valid_index_mismatch_count": valid_index_mismatch_count,
        "invalid_index_mismatch_count": invalid_index_mismatch_count,
        "total_index_mismatch_count": int(np.count_nonzero(index_mismatch)),
        "invalid_score_threshold": float(invalid_score_threshold),
    }


def compare_output_dicts(
    ref: Mapping[str, np.ndarray],
    cand: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    atol: float,
    rtol: float,
    ignore_invalid_topk_indices: bool,
    invalid_score_threshold: float,
) -> Dict[str, object]:
    per_output = {name: compare_arrays(ref[name], cand[name], atol=atol, rtol=rtol) for name in names}
    strict_passed = all(bool(v.get("allclose", False)) for v in per_output.values())
    exact = all(bool(v.get("exact", False)) for v in per_output.values())

    semantic_outputs_passed = dict((name, bool(metrics.get("allclose", False))) for name, metrics in per_output.items())
    topk_semantics = None
    if "top_scores" in ref and "top_scores" in cand and "top_indices" in ref and "top_indices" in cand:
        topk_semantics = compare_topk_index_semantics(
            ref["top_scores"],
            ref["top_indices"],
            cand["top_scores"],
            cand["top_indices"],
            invalid_score_threshold=float(invalid_score_threshold),
        )
        if ignore_invalid_topk_indices:
            semantic_outputs_passed["top_indices"] = bool(topk_semantics.get("semantic_match", False))

    semantic_passed = all(semantic_outputs_passed.values())
    return {
        "passed": bool(semantic_passed if ignore_invalid_topk_indices else strict_passed),
        "strict_passed": bool(strict_passed),
        "semantic_passed": bool(semantic_passed),
        "exact": bool(exact),
        "ignore_invalid_topk_indices": bool(ignore_invalid_topk_indices),
        "outputs": per_output,
        "semantic_outputs_passed": semantic_outputs_passed,
        "topk_index_semantics": topk_semantics,
    }


def write_markdown_report(path: Path, payload: Mapping[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Split pipeline vs merged baseline comparison")
    lines.append("")
    lines.append(f"- backend: `{payload['backend']}`")
    lines.append(f"- batch_size: `{payload['batch_size']}`")
    lines.append(f"- runs: `{payload['runs']}`")
    lines.append(f"- passed_all_runs: `{payload['passed_all_runs']}`")
    lines.append(f"- strict_passed_all_runs: `{payload['strict_passed_all_runs']}`")
    lines.append(f"- semantic_passed_all_runs: `{payload['semantic_passed_all_runs']}`")
    lines.append(f"- exact_all_runs: `{payload['exact_all_runs']}`")
    lines.append(f"- ignore_invalid_topk_indices: `{payload['ignore_invalid_topk_indices']}`")
    lines.append("")
    lines.append("## Timings")
    lines.append("")
    timings = payload.get("timing_ms_avg", {})
    lines.append("| stage | avg ms |")
    lines.append("|---|---:|")
    if isinstance(timings, Mapping):
        for k, v in timings.items():
            lines.append(f"| {k} | {float(v):.4f} |")
    lines.append("")
    lines.append("## TopK index semantics")
    lines.append("")
    lines.append("| run | semantic_match | valid_topk | valid_idx_mismatch | invalid_idx_mismatch | total_idx_mismatch |")
    lines.append("|---:|:---:|---:|---:|---:|---:|")
    for run in payload.get("runs_detail", []):
        if not isinstance(run, Mapping):
            continue
        comp = run.get("comparison", {})
        if not isinstance(comp, Mapping):
            continue
        sem = comp.get("topk_index_semantics")
        if not isinstance(sem, Mapping):
            continue
        lines.append(
            f"| {run.get('run_index')} | {sem.get('semantic_match')} | {sem.get('valid_topk_count')} | "
            f"{sem.get('valid_index_mismatch_count')} | {sem.get('invalid_index_mismatch_count')} | "
            f"{sem.get('total_index_mismatch_count')} |"
        )
    lines.append("")
    lines.append("## Per-run output checks")
    lines.append("")
    lines.append("| run | passed | strict | semantic | exact | output | allclose | exact output | semantic output | max_abs | mean_abs | mismatches |")
    lines.append("|---:|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|---:|---:|---:|")
    for run in payload.get("runs_detail", []):
        if not isinstance(run, Mapping):
            continue
        comp = run.get("comparison", {})
        if not isinstance(comp, Mapping):
            continue
        outs = comp.get("outputs", {})
        sem_outs = comp.get("semantic_outputs_passed", {})
        if not isinstance(outs, Mapping):
            continue
        for name, metrics in outs.items():
            if not isinstance(metrics, Mapping):
                continue
            semantic_output = sem_outs.get(name) if isinstance(sem_outs, Mapping) else None
            lines.append(
                f"| {run.get('run_index')} | {comp.get('passed')} | {comp.get('strict_passed')} | "
                f"{comp.get('semantic_passed')} | {comp.get('exact')} | {name} | "
                f"{metrics.get('allclose')} | {metrics.get('exact')} | {semantic_output} | "
                f"{metrics.get('max_abs')} | {metrics.get('mean_abs')} | {metrics.get('mismatch_count')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare split MXR pipeline against merged baseline.")
    p.add_argument("--merged-mxr", required=True)
    p.add_argument("--mxr1", required=True, help="Split MXR1: input -> heatmaps/pafs")
    p.add_argument("--mxr2", required=True, help="Split MXR2: pafs + topk -> limb pair tensors")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--backend", choices=["torch_manual", "torch_bicubic", "merged_topk_oracle"], default="torch_manual")
    p.add_argument("--device", default="", help="PyTorch device for external heatmap backend, e.g. cuda or cpu. Empty = auto.")
    p.add_argument("--runs", type=int, default=3)
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
    p.add_argument("--ignore-invalid-topk-indices", action="store_true", help="Treat TopK index mismatches in invalid -1e9 slots as semantic ties instead of failures.")
    p.add_argument("--invalid-score-threshold", type=float, default=-1.0e8)
    p.add_argument("--json", default="outputs/split_pipeline_compare/compare.json")
    p.add_argument("--markdown", default="outputs/split_pipeline_compare/compare.md")
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

        if args.backend == "merged_topk_oracle":
            top_scores = np.ascontiguousarray(merged_out["top_scores"].astype(np.float32, copy=False))
            top_indices = np.ascontiguousarray(merged_out["top_indices"].astype(np.int64, copy=False))
        else:
            top_scores, top_indices = run_external_heatmap_topk(
                heatmaps,
                cfg,
                backend=args.backend,
                device=args.device or None,
            )
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
            f"merged_ms={timing_acc['merged'][-1]:.3f} split_ms={timing_acc['split_total'][-1]:.3f}"
        )
        sem = comparison.get("topk_index_semantics")
        if isinstance(sem, Mapping):
            print(
                f"        topk_valid_mismatch={sem.get('valid_index_mismatch_count')} "
                f"topk_invalid_mismatch={sem.get('invalid_index_mismatch_count')} "
                f"topk_total_mismatch={sem.get('total_index_mismatch_count')}"
            )

    payload: Dict[str, object] = {
        "merged_mxr": str(args.merged_mxr),
        "mxr1": str(args.mxr1),
        "mxr2": str(args.mxr2),
        "backend": str(args.backend),
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
