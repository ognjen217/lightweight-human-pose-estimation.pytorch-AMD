#!/usr/bin/env python3
"""
Compare outputs of two ONNX or MIGraphX MXR models on identical inputs.

This tool is meant for accuracy-preserving graph rewrites. It checks whether a
candidate optimized graph keeps the six merged/fused-pruned outputs equivalent
to a validated baseline:

    top_scores, top_indices, limb_top_pair_a_idx, limb_top_pair_b_idx,
    limb_top_pair_score, limb_top_pair_valid

It can also compare generic output lists by index.

Examples:
    # Compare two ONNX models with ONNX Runtime.
    python tools/compare_onnx_outputs.py baseline.onnx candidate.onnx \
      --backend onnxruntime --random-inputs 5 --json outputs/compare.json

    # Compare two MXR models with MIGraphX.
    python tools/compare_onnx_outputs.py baseline.mxr candidate.mxr \
      --backend migraphx --random-inputs 10 --input-shape 4,3,544,968

    # Use saved input tensors.
    python tools/compare_onnx_outputs.py baseline.mxr candidate.mxr \
      --backend migraphx --input-npz inputs.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_OUTPUT_NAMES = [
    "top_scores",
    "top_indices",
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]

INTEGER_LIKE_NAME_TOKENS = ("index", "indices", "idx", "valid", "mask")
FLOAT_LIKE_NAME_TOKENS = ("score", "scores", "heat", "paf")


@dataclass
class ArrayComparison:
    output_index: int
    name: str
    baseline_shape: List[int]
    candidate_shape: List[int]
    baseline_dtype: str
    candidate_dtype: str
    comparable: bool
    exact_equal: Optional[bool] = None
    exact_match_rate: Optional[float] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    p95_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    mean_rel_diff: Optional[float] = None
    allclose: Optional[bool] = None
    baseline_nan_count: int = 0
    candidate_nan_count: int = 0
    reason: str = ""


@dataclass
class RunComparison:
    run_index: int
    seed: Optional[int]
    input_names: List[str]
    output_comparisons: List[ArrayComparison]
    baseline_ms: float
    candidate_ms: float
    passed: bool


@dataclass
class ComparisonReport:
    baseline_model: str
    candidate_model: str
    backend: str
    runs: List[RunComparison]
    passed: bool
    atol: float
    rtol: float
    exact_integer_like: bool
    notes: List[str]


def _parse_shape(text: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in text.replace("x", ",").split(",") if x.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape {text!r}. Use e.g. 4,3,544,968") from exc


def _shape_list(arr: np.ndarray) -> List[int]:
    return [int(x) for x in arr.shape]


def _as_numpy_list(outputs: Any) -> List[np.ndarray]:
    if isinstance(outputs, dict):
        return [np.asarray(v) for _, v in outputs.items()]
    if not isinstance(outputs, (list, tuple)):
        try:
            outputs = list(outputs)
        except TypeError:
            outputs = [outputs]
    return [np.asarray(x) for x in outputs]


def _looks_integer_like(name: str, arr_a: np.ndarray, arr_b: np.ndarray) -> bool:
    key = str(name).lower()
    if any(tok in key for tok in INTEGER_LIKE_NAME_TOKENS):
        return True
    if np.issubdtype(arr_a.dtype, np.integer) or np.issubdtype(arr_b.dtype, np.integer):
        return True
    if np.issubdtype(arr_a.dtype, np.bool_) or np.issubdtype(arr_b.dtype, np.bool_):
        return True
    return False


def compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    *,
    name: str,
    output_index: int,
    atol: float,
    rtol: float,
    exact_integer_like: bool,
) -> ArrayComparison:
    a = np.asarray(a)
    b = np.asarray(b)
    rec = ArrayComparison(
        output_index=output_index,
        name=name,
        baseline_shape=_shape_list(a),
        candidate_shape=_shape_list(b),
        baseline_dtype=str(a.dtype),
        candidate_dtype=str(b.dtype),
        comparable=a.shape == b.shape,
        baseline_nan_count=int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
        candidate_nan_count=int(np.isnan(b).sum()) if np.issubdtype(b.dtype, np.floating) else 0,
    )

    if a.shape != b.shape:
        rec.reason = "shape mismatch"
        return rec

    if a.size == 0:
        rec.exact_equal = bool(np.array_equal(a, b))
        rec.exact_match_rate = 1.0 if rec.exact_equal else 0.0
        rec.allclose = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
        return rec

    integer_like = _looks_integer_like(name, a, b)
    if integer_like and exact_integer_like:
        eq = a == b
        rec.exact_equal = bool(np.array_equal(a, b))
        rec.exact_match_rate = float(np.mean(eq))
        rec.allclose = rec.exact_equal
        if not rec.exact_equal:
            mismatch = np.flatnonzero(~eq.reshape(-1))[:10]
            rec.reason = f"integer-like output mismatch, first mismatch flat indices={mismatch.tolist()}"
        return rec

    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    diff = np.abs(af - bf)
    denom = np.maximum(np.abs(af), np.float64(1e-12))
    rel = diff / denom

    rec.exact_equal = bool(np.array_equal(a, b))
    rec.exact_match_rate = float(np.mean(a == b))
    rec.max_abs_diff = float(np.nanmax(diff))
    rec.mean_abs_diff = float(np.nanmean(diff))
    rec.p95_abs_diff = float(np.nanpercentile(diff, 95))
    rec.max_rel_diff = float(np.nanmax(rel))
    rec.mean_rel_diff = float(np.nanmean(rel))
    rec.allclose = bool(np.allclose(af, bf, atol=atol, rtol=rtol, equal_nan=True))
    if not rec.allclose:
        rec.reason = "not allclose"
    return rec


class BaseRunner:
    def input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        raise NotImplementedError

    def run(self, inputs: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class OnnxRuntimeRunner(BaseRunner):
    def __init__(self, model_path: Path, providers: Optional[Sequence[str]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Missing dependency: onnxruntime. Install it or use --backend migraphx.") from exc
        self.ort = ort
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(model_path), providers=list(providers or ["CPUExecutionProvider"]))

    def input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        for inp in self.session.get_inputs():
            shape = tuple(int(x) if isinstance(x, int) else -1 for x in inp.shape)
            dtype = np.float32
            if "float16" in inp.type:
                dtype = np.float16
            elif "double" in inp.type:
                dtype = np.float64
            elif "int64" in inp.type:
                dtype = np.int64
            elif "int32" in inp.type:
                dtype = np.int32
            specs[inp.name] = (shape, dtype)
        return specs

    def run(self, inputs: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        names = [o.name for o in self.session.get_outputs()]
        result = self.session.run(names, dict(inputs))
        return [np.asarray(x) for x in result]


class MIGraphXRunner(BaseRunner):
    def __init__(self, model_path: Path) -> None:
        try:
            import migraphx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Missing dependency: migraphx. Activate the ROCm/MIGraphX environment.") from exc
        self.migraphx = migraphx
        self.model_path = Path(model_path)
        self.program = migraphx.load(str(model_path))

    def input_specs(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        shapes = self.program.get_parameter_shapes()
        for name, shape in shapes.items():
            lens = tuple(int(x) for x in shape.lens())
            st = str(shape.type()).lower()
            if "half" in st or "float16" in st:
                dtype = np.float16
            elif "double" in st:
                dtype = np.float64
            elif "int64" in st:
                dtype = np.int64
            elif "int32" in st:
                dtype = np.int32
            else:
                dtype = np.float32
            specs[str(name)] = (lens, dtype)
        return specs

    def run(self, inputs: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        args = {}
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr)
            try:
                args[name] = self.migraphx.argument(arr)
            except Exception:
                args[name] = arr
        result = self.program.run(args)
        return [np.asarray(x) for x in result]


def _make_runner(path: Path, backend: str, providers: Optional[Sequence[str]]) -> BaseRunner:
    backend = backend.lower()
    if backend == "auto":
        suffix = path.suffix.lower()
        backend = "migraphx" if suffix == ".mxr" else "onnxruntime"
    if backend in {"ort", "onnxruntime", "onnx"}:
        return OnnxRuntimeRunner(path, providers=providers)
    if backend in {"migraphx", "mxr", "mx"}:
        return MIGraphXRunner(path)
    raise ValueError(f"Unsupported backend={backend!r}")


def _resolve_input_specs(
    baseline: BaseRunner,
    candidate: BaseRunner,
    input_shape: Optional[Tuple[int, ...]],
    input_dtype: str,
) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    specs = baseline.input_specs()
    cand_specs = candidate.input_specs()
    if set(specs) != set(cand_specs):
        print(f"[WARN] input names differ: baseline={list(specs)}, candidate={list(cand_specs)}")

    out: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    for i, (name, (shape, dtype)) in enumerate(specs.items()):
        if input_shape is not None and i == 0:
            shape = tuple(input_shape)
        if any(d <= 0 for d in shape):
            raise SystemExit(
                f"Input {name!r} has dynamic/unknown shape {shape}. Provide --input-shape for the first input or use --input-npz."
            )
        if input_dtype != "auto":
            dtype = np.dtype(input_dtype)
        out[name] = (tuple(shape), np.dtype(dtype))
    return out


def _load_inputs_from_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _generate_inputs(
    specs: Mapping[str, Tuple[Tuple[int, ...], np.dtype]],
    *,
    seed: int,
    scale: float,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: Dict[str, np.ndarray] = {}
    for name, (shape, dtype) in specs.items():
        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(shape).astype(dtype) * np.asarray(scale, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            arr = rng.integers(low=0, high=10, size=shape, dtype=dtype)
        elif np.issubdtype(dtype, np.bool_):
            arr = rng.random(shape) > 0.5
        else:
            arr = rng.standard_normal(shape).astype(np.float32)
        inputs[name] = np.ascontiguousarray(arr)
    return inputs


def _output_names_from_arg(names: Optional[str], n: int) -> List[str]:
    if names:
        parsed = [x.strip() for x in names.split(",") if x.strip()]
    else:
        parsed = DEFAULT_OUTPUT_NAMES[:]
    while len(parsed) < n:
        parsed.append(f"output_{len(parsed)}")
    return parsed[:n]


def compare_models(
    baseline_model: Path,
    candidate_model: Path,
    *,
    backend: str,
    providers: Optional[Sequence[str]],
    input_npz: Optional[Path],
    input_shape: Optional[Tuple[int, ...]],
    input_dtype: str,
    random_inputs: int,
    seed: int,
    input_scale: float,
    output_names: Optional[str],
    atol: float,
    rtol: float,
    exact_integer_like: bool,
    warmup: int,
) -> ComparisonReport:
    baseline = _make_runner(baseline_model, backend, providers)
    candidate = _make_runner(candidate_model, backend, providers)

    runs: List[RunComparison] = []
    notes: List[str] = []

    if input_npz:
        inputs_list = [(_load_inputs_from_npz(input_npz), None)]
    else:
        specs = _resolve_input_specs(baseline, candidate, input_shape, input_dtype)
        n_runs = max(1, int(random_inputs))
        inputs_list = [(_generate_inputs(specs, seed=seed + i, scale=input_scale), seed + i) for i in range(n_runs)]

    # Warmup on first input only. Keep it before measured comparison to avoid one-time compile/cache effects.
    if warmup > 0 and inputs_list:
        warm_inputs = inputs_list[0][0]
        for _ in range(int(warmup)):
            baseline.run(warm_inputs)
            candidate.run(warm_inputs)

    for run_idx, (inputs, run_seed) in enumerate(inputs_list):
        t0 = time.perf_counter()
        out_a = baseline.run(inputs)
        baseline_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        out_b = candidate.run(inputs)
        candidate_ms = (time.perf_counter() - t0) * 1000.0

        n = min(len(out_a), len(out_b))
        if len(out_a) != len(out_b):
            notes.append(f"Run {run_idx}: output count mismatch baseline={len(out_a)} candidate={len(out_b)}; comparing first {n}.")
        names = _output_names_from_arg(output_names, n)
        comps = [
            compare_arrays(
                out_a[i],
                out_b[i],
                name=names[i],
                output_index=i,
                atol=atol,
                rtol=rtol,
                exact_integer_like=exact_integer_like,
            )
            for i in range(n)
        ]
        passed = all(c.comparable and (c.allclose is True) for c in comps) and len(out_a) == len(out_b)
        runs.append(
            RunComparison(
                run_index=run_idx,
                seed=run_seed,
                input_names=list(inputs.keys()),
                output_comparisons=comps,
                baseline_ms=float(baseline_ms),
                candidate_ms=float(candidate_ms),
                passed=bool(passed),
            )
        )

    return ComparisonReport(
        baseline_model=str(baseline_model),
        candidate_model=str(candidate_model),
        backend=backend,
        runs=runs,
        passed=all(r.passed for r in runs),
        atol=float(atol),
        rtol=float(rtol),
        exact_integer_like=bool(exact_integer_like),
        notes=notes,
    )


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def write_markdown(report: ComparisonReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# ONNX/MXR output comparison")
    lines.append("")
    lines.append(f"- Baseline: `{report.baseline_model}`")
    lines.append(f"- Candidate: `{report.candidate_model}`")
    lines.append(f"- Backend: `{report.backend}`")
    lines.append(f"- Passed: **{report.passed}**")
    lines.append(f"- Tolerances: `atol={report.atol}`, `rtol={report.rtol}`")
    lines.append(f"- Exact integer-like outputs: `{report.exact_integer_like}`")
    lines.append("")
    if report.notes:
        lines.append("## Notes")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    for run in report.runs:
        lines.append(f"## Run {run.run_index}")
        lines.append("")
        lines.append(f"- Seed: `{run.seed}`")
        lines.append(f"- Passed: **{run.passed}**")
        lines.append(f"- Baseline time: `{run.baseline_ms:.3f} ms`")
        lines.append(f"- Candidate time: `{run.candidate_ms:.3f} ms`")
        lines.append("")
        lines.append("| Output | Shape A | Shape B | Dtype A | Dtype B | Exact match | Max abs | Mean abs | P95 abs | Allclose | Reason |")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|---|")
        for c in run.output_comparisons:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{c.name}`",
                        f"`{c.baseline_shape}`",
                        f"`{c.candidate_shape}`",
                        f"`{c.baseline_dtype}`",
                        f"`{c.candidate_dtype}`",
                        "" if c.exact_match_rate is None else f"{100.0 * c.exact_match_rate:.4f}%",
                        "" if c.max_abs_diff is None else f"{c.max_abs_diff:.6g}",
                        "" if c.mean_abs_diff is None else f"{c.mean_abs_diff:.6g}",
                        "" if c.p95_abs_diff is None else f"{c.p95_abs_diff:.6g}",
                        str(c.allclose),
                        c.reason,
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare outputs of two ONNX or MIGraphX MXR models.")
    p.add_argument("baseline_model", type=Path)
    p.add_argument("candidate_model", type=Path)
    p.add_argument("--backend", choices=["auto", "onnxruntime", "ort", "onnx", "migraphx", "mxr", "mx"], default="auto")
    p.add_argument("--providers", default="", help="Comma-separated ONNX Runtime providers. Default: CPUExecutionProvider.")
    p.add_argument("--input-npz", type=Path, default=None, help="NPZ file containing named input arrays.")
    p.add_argument("--input-shape", type=_parse_shape, default=None, help="Override first input shape, e.g. 4,3,544,968.")
    p.add_argument("--input-dtype", default="auto", help="Override generated input dtype: auto,float16,float32,int64,...")
    p.add_argument("--random-inputs", type=int, default=1, help="Number of random generated inputs when --input-npz is not used.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--output-names", default="", help="Comma-separated output labels for reporting.")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--no-exact-integer-like", action="store_true", help="Use allclose instead of exact comparison for index/valid outputs.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs before measured comparison.")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--markdown", "--md", type=Path, default=None)
    p.add_argument("--fail-on-diff", action="store_true", help="Exit 2 when comparison fails.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    providers = [x.strip() for x in args.providers.split(",") if x.strip()] or None
    report = compare_models(
        args.baseline_model,
        args.candidate_model,
        backend=args.backend,
        providers=providers,
        input_npz=args.input_npz,
        input_shape=args.input_shape,
        input_dtype=args.input_dtype,
        random_inputs=args.random_inputs,
        seed=args.seed,
        input_scale=args.input_scale,
        output_names=args.output_names or None,
        atol=args.atol,
        rtol=args.rtol,
        exact_integer_like=not args.no_exact_integer_like,
        warmup=args.warmup,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        print(f"[OK] wrote JSON: {args.json}")
    if args.markdown:
        write_markdown(report, args.markdown)
        print(f"[OK] wrote Markdown: {args.markdown}")

    print(json.dumps({
        "passed": report.passed,
        "runs": len(report.runs),
        "baseline_model": report.baseline_model,
        "candidate_model": report.candidate_model,
        "backend": report.backend,
        "avg_baseline_ms": float(np.mean([r.baseline_ms for r in report.runs])) if report.runs else None,
        "avg_candidate_ms": float(np.mean([r.candidate_ms for r in report.runs])) if report.runs else None,
    }, indent=2))

    if args.fail_on_diff and not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
