import migraphx

ONNX_PATH = "pose_model_dynamic.onnx"
OUT = "pose_model_dynamic_b1_b8.mxr"

def main():
    batch = migraphx.shape.dynamic_dimension(1, 8, {1, 2, 4, 8})

    model = migraphx.parse_onnx(
        ONNX_PATH,
        dim_params={
            "batch_size": batch
        },
        print_program_on_error=True
    )

    print("Parsed OK")
    print(model.get_parameter_shapes())

    migraphx.quantize_fp16(model)
    print("FP16 OK")

    model.compile(
        migraphx.get_target("gpu"),
        exhaustive_tune=False
    )

    print("Compiled OK")

    migraphx.save(model, OUT)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
