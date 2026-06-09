#!/usr/bin/env python3
import argparse
from pathlib import Path
import onnx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {inp}")
    model = onnx.load(str(inp))
    onnx.checker.check_model(model)

    try:
        print("[shape_inference]")
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[shape_inference] skipped: {e}")

    try:
        import onnxoptimizer
        passes = [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_consecutive_transposes",
            "fuse_add_bias_into_conv",
            "fuse_transpose_into_gemm",
        ]
        print("[onnxoptimizer]", passes)
        model = onnxoptimizer.optimize(model, passes)
    except Exception as e:
        print(f"[onnxoptimizer] skipped: {e}")

    try:
        from onnxsim import simplify
        print("[onnxsim] simplify")
        model_simplified, ok = simplify(model)
        if ok:
            model = model_simplified
        else:
            print("[onnxsim] simplify returned check=False; keeping previous model")
    except Exception as e:
        print(f"[onnxsim] skipped: {e}")

    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    print(f"[save] {out}")

if __name__ == "__main__":
    main()
