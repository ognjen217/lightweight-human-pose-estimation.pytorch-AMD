#!/usr/bin/env python3
"""Compile an arbitrary ONNX graph to MIGraphX MXR.

Unlike tools/compile_migraphx_static_batches.py, this tool does not assume a pose-model
input named `input`. It is meant for generated postprocess graphs whose ONNX
already contains fixed/static shapes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true", help="Run migraphx.quantize_fp16 before compile. Python MIGraphX API is required for this option.")
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--print-program-on-error", action="store_true", default=True)
    return p.parse_args()


def compile_with_python_api(args: argparse.Namespace, onnx_path: Path, out_path: Path) -> bool:
    try:
        import migraphx  # type: ignore
    except ModuleNotFoundError:
        return False

    print("Using Python MIGraphX API")
    program = migraphx.parse_onnx(str(onnx_path), print_program_on_error=args.print_program_on_error)
    print("Parsed OK")
    print("Input shapes:", program.get_parameter_shapes())

    if args.fp16:
        migraphx.quantize_fp16(program)
        print("FP16 quantization OK")

    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(args.exhaustive_tune))
    migraphx.save(program, str(out_path))
    return True


def compile_with_driver(args: argparse.Namespace, onnx_path: Path, out_path: Path) -> None:
    if args.fp16:
        raise RuntimeError(
            "--fp16 requires the Python MIGraphX API because migraphx-driver fallback "
            "does not apply migraphx.quantize_fp16 in this script. Activate the venv that "
            "contains migraphx or rerun without --fp16."
        )

    driver = shutil.which("migraphx-driver") or "/opt/rocm/bin/migraphx-driver"
    if not Path(driver).exists() and shutil.which(driver) is None:
        raise RuntimeError(
            "Python migraphx is unavailable and migraphx-driver was not found. "
            "Run `source rocm721/activate_rocm721.sh` or check ROCm/MIGraphX install."
        )

    cmd = [driver, "compile", str(onnx_path), "--onnx", "--gpu", "--binary", "-o", str(out_path)]
    if args.exhaustive_tune:
        print("[warning] --exhaustive-tune is ignored by migraphx-driver fallback in this script")
    print("Using migraphx-driver fallback")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling ONNX: {onnx_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if not compile_with_python_api(args, onnx_path, out_path):
        compile_with_driver(args, onnx_path, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
