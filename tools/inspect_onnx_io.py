#!/usr/bin/env python3
"""
Inspect ONNX input/output names, dtypes and shapes.

Usage:
  python tools/inspect_onnx_io.py models/fp16_refinment1.onnx
  python tools/inspect_onnx_io.py models/fused_postprocess_pruned_cache/*.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import onnx


def _dtype_name(elem_type: int) -> str:
    try:
        return onnx.TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def _dims(value_info: Any) -> List[Any]:
    shape = value_info.type.tensor_type.shape
    out = []
    for d in shape.dim:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def _print_value(prefix: str, vi: Any) -> None:
    tt = vi.type.tensor_type
    print(f"{prefix:10s} {vi.name:50s} dtype={_dtype_name(tt.elem_type):12s} shape={_dims(vi)}")


def inspect(path: str | Path) -> None:
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    print("=" * 120)
    print(path)
    print("=" * 120)
    print("IR version:", model.ir_version)
    print("Opsets:    ", [(o.domain or "ai.onnx", o.version) for o in model.opset_import])
    print("Nodes:     ", len(model.graph.node))
    print("Initializers:", len(model.graph.initializer))
    print()
    print("INPUTS")
    for x in model.graph.input:
        _print_value("input", x)
    print()
    print("OUTPUTS")
    for x in model.graph.output:
        _print_value("output", x)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_files", nargs="+")
    args = ap.parse_args()
    for p in args.onnx_files:
        inspect(p)


if __name__ == "__main__":
    main()
