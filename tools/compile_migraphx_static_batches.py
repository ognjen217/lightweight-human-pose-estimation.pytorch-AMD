#!/usr/bin/env python3
"""Compile static-batch MIGraphX models from an ONNX file.

The original script was intentionally simple and hard-coded. This version keeps
those defaults but adds CLI flags so graph-clean ONNX variants can be compiled
without editing the file.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import migraphx


def parse_batches(raw: list[str]) -> list[int]:
    batches: list[int] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if value <= 0:
                raise argparse.ArgumentTypeError("batch sizes must be positive")
            batches.append(value)

    if not batches:
        raise argparse.ArgumentTypeError("at least one batch size is required")

    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="pose_model_dynamic.onnx")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=968)
    parser.add_argument("--input-name", default="input")
    parser.add_argument(
        "--batches",
        nargs="+",
        default=["1", "2", "4", "8"],
        help="Batch sizes, either space-separated or comma-separated. Default: 1 2 4 8",
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to the ONNX stem.",
    )
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Compile without migraphx.quantize_fp16.",
    )
    parser.add_argument(
        "--exhaustive-tune",
        action="store_true",
        help="Enable MIGraphX exhaustive tuning during compile.",
    )
    parser.add_argument(
        "--print-program-on-error",
        action="store_true",
        default=True,
        help="Pass print_program_on_error=True to migraphx.parse_onnx.",
    )
    return parser.parse_args()


def compile_one(args: argparse.Namespace, batch_size: int) -> Path:
    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or onnx_path.stem
    dtype_suffix = "fp16" if args.fp16 else "fp32"
    out_path = out_dir / f"{prefix}_b{batch_size}_{dtype_suffix}.mxr"

    print(f"\n{'=' * 70}")
    print(f"Compiling static batch={batch_size}")
    print(f"ONNX: {onnx_path}")
    print(f"Output: {out_path}")
    print(f"{'=' * 70}")

    model = migraphx.parse_onnx(
        str(onnx_path),
        map_input_dims={
            args.input_name: [batch_size, 3, args.height, args.width]
        },
        print_program_on_error=args.print_program_on_error,
    )

    print("Parsed OK")
    print("Input shapes:", model.get_parameter_shapes())

    if args.fp16:
        migraphx.quantize_fp16(model)
        print("FP16 quantization OK")
    else:
        print("FP16 quantization skipped")

    model.compile(
        migraphx.get_target("gpu"),
        exhaustive_tune=args.exhaustive_tune,
    )

    migraphx.save(model, str(out_path))
    print(f"Saved: {out_path}")

    return out_path


def main() -> None:
    args = parse_args()
    args.batches = parse_batches(args.batches)

    failed: list[int] = []
    outputs: list[Path] = []

    for batch_size in args.batches:
        try:
            outputs.append(compile_one(args, batch_size))
        except Exception:
            print(f"FAILED batch={batch_size}")
            traceback.print_exc()
            failed.append(batch_size)

    print("\nDone.")
    if outputs:
        print("Compiled outputs:")
        for output in outputs:
            print(f"  - {output}")

    if failed:
        print("Failed batches:", failed)
        raise SystemExit(1)

    print("All static batch models compiled successfully.")


if __name__ == "__main__":
    main()
