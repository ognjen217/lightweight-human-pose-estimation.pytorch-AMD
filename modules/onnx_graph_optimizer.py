#!/usr/bin/env python3
"""Conservative ONNX graph optimization helpers for large MIGraphX graphs.

The helpers are intentionally safe-by-default. They do not rewrite model
mathematics; they only run standard ONNX cleanup/simplification passes when the
corresponding optional packages are installed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import onnx

SAFE_ONNXOPT_PASSES = [
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_transpose",
    "eliminate_unused_initializer",
    "extract_constant_to_initializer",
    "eliminate_deadend",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
]

MIGRAPHX_INTERESTING_OPS = [
    "Cast",
    "Concat",
    "Constant",
    "Div",
    "Expand",
    "Floor",
    "Gather",
    "Greater",
    "Identity",
    "Mul",
    "ReduceSum",
    "Reshape",
    "Shape",
    "Slice",
    "Sub",
    "TopK",
    "Transpose",
    "Unsqueeze",
    "Where",
]


def op_counts(model: onnx.ModelProto) -> dict[str, int]:
    return dict(Counter(node.op_type for node in model.graph.node).most_common())


def interesting_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts = Counter(node.op_type for node in model.graph.node)
    return {op: int(counts.get(op, 0)) for op in MIGRAPHX_INTERESTING_OPS}


def graph_io(model: onnx.ModelProto) -> dict[str, list[str]]:
    initializer_names = {init.name for init in model.graph.initializer}
    return {
        "inputs": [inp.name for inp in model.graph.input if inp.name not in initializer_names],
        "outputs": [out.name for out in model.graph.output],
    }


def summarize(model: onnx.ModelProto) -> dict[str, Any]:
    return {
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": opset.domain or "ai.onnx", "version": int(opset.version)}
            for opset in model.opset_import
        ],
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "io": graph_io(model),
        "op_counts": op_counts(model),
        "interesting_counts": interesting_counts(model),
    }


def _available_onnxoptimizer_passes() -> set[str]:
    try:
        import onnxoptimizer  # type: ignore
    except ImportError:
        return set()
    return set(onnxoptimizer.get_available_passes())


def run_onnxoptimizer(model: onnx.ModelProto, passes: Iterable[str] | None = None) -> tuple[onnx.ModelProto, list[str], str | None]:
    try:
        import onnxoptimizer  # type: ignore
    except ImportError:
        return model, [], "onnxoptimizer is not installed"

    requested = list(passes or SAFE_ONNXOPT_PASSES)
    available = _available_onnxoptimizer_passes()
    selected = [p for p in requested if p in available]
    skipped = [p for p in requested if p not in available]
    if skipped:
        print(f"[onnx-opt] skipped unavailable passes: {skipped}")
    if not selected:
        return model, [], "no requested onnxoptimizer passes are available"

    optimized = onnxoptimizer.optimize(model, selected)
    onnx.checker.check_model(optimized)
    return optimized, selected, None


def run_shape_inference(model: onnx.ModelProto) -> tuple[onnx.ModelProto, str | None]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(inferred)
        return inferred, None
    except Exception as exc:
        return model, f"shape inference skipped/failed: {exc}"


def run_onnxsim(model: onnx.ModelProto, *, input_shapes: dict[str, list[int]] | None = None) -> tuple[onnx.ModelProto, str | None]:
    try:
        import onnxsim  # type: ignore
    except ImportError:
        return model, "onnxsim is not installed"

    kwargs: dict[str, Any] = {}
    if input_shapes:
        kwargs["input_shapes"] = input_shapes
    simplified, ok = onnxsim.simplify(model, **kwargs)
    if not ok:
        return model, "onnxsim returned check=False"
    onnx.checker.check_model(simplified)
    return simplified, None


def optimize_onnx_for_migraphx(
    input_path: str | Path,
    output_path: str | Path,
    *,
    use_onnxoptimizer: bool = True,
    use_shape_inference: bool = True,
    use_onnxsim: bool = False,
    onnxoptimizer_passes: Iterable[str] | None = None,
    report_json: str | Path | None = None,
    input_shapes: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)

    before = summarize(model)
    steps: list[dict[str, Any]] = []

    if use_onnxoptimizer:
        model, selected, warning = run_onnxoptimizer(model, onnxoptimizer_passes)
        steps.append({"step": "onnxoptimizer", "passes": selected, "warning": warning})

    if use_shape_inference:
        model, warning = run_shape_inference(model)
        steps.append({"step": "shape_inference", "warning": warning})

    if use_onnxsim:
        model, warning = run_onnxsim(model, input_shapes=input_shapes)
        steps.append({"step": "onnxsim", "warning": warning})

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    after = summarize(model)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "steps": steps,
        "before": before,
        "after": after,
        "node_delta": int(after["num_nodes"] - before["num_nodes"]),
        "initializer_delta": int(after["num_initializers"] - before["num_initializers"]),
    }

    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
