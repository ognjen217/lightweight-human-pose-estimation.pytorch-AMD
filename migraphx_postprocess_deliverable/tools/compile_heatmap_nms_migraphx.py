#!/usr/bin/env python3
"""Compile the exported ONNX heatmap NMS head to a MIGraphX .mxr file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile HeatmapNMSHead ONNX with MIGraphX")
    parser.add_argument("--onnx", default="heatmap_nms_head.onnx", help="Input ONNX path")
    parser.add_argument("--mxr", default="heatmap_nms_head.mxr", help="Output MIGraphX .mxr path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    mxr_path = Path(args.mxr)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        import migraphx  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        print("Failed to import migraphx. Make sure ROCm/MIGraphX Python bindings are available.", file=sys.stderr)
        raise exc

    try:
        print(f"Parsing ONNX: {onnx_path}")
        program = migraphx.parse_onnx(str(onnx_path))
        print("Compiling for MIGraphX GPU target...")
        program.compile(migraphx.get_target("gpu"))
        mxr_path.parent.mkdir(parents=True, exist_ok=True)
        migraphx.save(program, str(mxr_path))
        print(f"Saved MIGraphX program: {mxr_path}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"MIGraphX compilation failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
