import migraphx
import traceback

ONNX_PATH = "pose_model_dynamic.onnx"
H = 544
W = 968

BATCHES = [1, 2, 4, 8]

def compile_one(batch_size: int):
    print(f"\n{'=' * 70}")
    print(f"Compiling static batch={batch_size}")
    print(f"{'=' * 70}")

    model = migraphx.parse_onnx(
        ONNX_PATH,
        map_input_dims={
            "input": [batch_size, 3, H, W]
        },
        print_program_on_error=True
    )

    print("Parsed OK")
    print("Input shapes:", model.get_parameter_shapes())

    migraphx.quantize_fp16(model)
    print("FP16 quantization OK")

    model.compile(
        migraphx.get_target("gpu"),
        exhaustive_tune=False
    )

    out = f"pose_model_b{batch_size}_fp16.mxr"
    migraphx.save(model, out)

    print(f"Saved: {out}")


def main():
    failed = []

    for b in BATCHES:
        try:
            compile_one(b)
        except Exception:
            print(f"FAILED batch={b}")
            traceback.print_exc()
            failed.append(b)

    print("\nDone.")
    if failed:
        print("Failed batches:", failed)
    else:
        print("All static batch models compiled successfully.")


if __name__ == "__main__":
    main()
