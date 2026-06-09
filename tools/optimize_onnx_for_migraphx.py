#!/usr/bin/env python3
"""Optimize an existing ONNX graph before MIGraphX compilation.

This is useful for large generated postprocess graphs such as fused/pruned heads.
The tool is conservative by default and writes a before/after JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.onnx_graph_optimizer import optimize_onnx_for_migraphx


def _parse_shape(items: list[str] | None) -> dict[str, list[int]] | None:
    if not items:
        return None
    result: dict[str, list[int]] = {}
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"Expected NAME:D0,D1,... got {item!r}")
        name, raw_dims = item.split(":", 1)
        dims = [int(x) for x in raw_dims.split(",") if x.strip()]
        if not name or not dims:
            raise argparse.ArgumentTypeError(f"Invalid input shape spec: {item!r}")
        result[name] = dims
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--report-json", type=Path, default=None)
    p.add_argument("--no-onnxoptimizer", action="store_true")
    p.add_argument("--no-shape-inference", action="store_true")
    p.add_argument("--onnxsim", action="store_true")
    p.add_argument(
        "--input-shape",
        action="append",
        default=None,
        help="Static input shape for onnxsim, e.g. heatmaps:1,18,68,121. Can be repeated.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = optimize_onnx_for_migraphx(
        args.input,
        args.output,
        use_onnxoptimizer=not args.no_onnxoptimizer,
        use_shape_inference=not args.no_shape_inference,
        use_onnxsim=args.onnxsim,
        input_shapes=_parse_shape(args.input_shape),
        report_json=args.report_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
