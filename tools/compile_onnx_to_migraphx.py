#!/usr/bin/env python3
"""Compile an arbitrary ONNX graph to MIGraphX MXR.

Unlike compile_migraphx_static_batches.py, this tool does not assume a pose-model
input named `input`. It is meant for generated postprocess graphs whose ONNX
already contains fixed/static shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true", help="Run migraphx.quantize_fp16 before compile.")
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--print-program-on-error", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    import migraphx

    args = parse_args()
    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling ONNX: {onnx_path}")
    program = migraphx.parse_onnx(str(onnx_path), print_program_on_error=args.print_program_on_error)
    print("Parsed OK")
    print("Input shapes:", program.get_parameter_shapes())

    if args.fp16:
        migraphx.quantize_fp16(program)
        print("FP16 quantization OK")

    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(args.exhaustive_tune))
    migraphx.save(program, str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
