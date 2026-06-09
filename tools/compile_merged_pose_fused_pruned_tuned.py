#!/usr/bin/env python3
import argparse
from pathlib import Path
import migraphx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--exhaustive-tune", action="store_true")
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[parse] {onnx_path}")
    model = migraphx.parse_onnx(str(onnx_path))

    if args.fp16:
        print("[quantize] fp16")
        migraphx.quantize_fp16(model)

    print(f"[compile] gpu exhaustive_tune={args.exhaustive_tune}")
    model.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(args.exhaustive_tune))

    print(f"[save] {out_path}")
    migraphx.save(model, str(out_path))

if __name__ == "__main__":
    main()
