#!/usr/bin/env python3
"""Export/compile MXR1 for the split external-heatmap pipeline.

MXR1 contains only:

    image input -> pose model -> adapter -> heatmaps/pafs

Inputs:
    input    [B,3,544,968] fp16

Outputs:
    heatmaps [B,18,68,121] fp32
    pafs     [B,38,68,121] fp32

The adapter mirrors the bridge used by the merged fused-pruned model:
    stage_heatmaps -> Cast(float) -> Slice channels 0:18
    stage_pafs     -> Cast(float)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def set_input_batch(model, input_name: str, batch_size: int) -> None:
    for inp in model.graph.input:
        if inp.name == input_name:
            tt = inp.type.tensor_type
            if tt.HasField("shape"):
                tt.shape.dim[0].dim_value = int(batch_size)


def export_pose_adapter(
    *,
    pose_onnx: str | Path,
    output_onnx: str | Path,
    batch_size: int,
    input_name: str = "input",
    pose_heatmaps_output: str = "stage_heatmaps",
    pose_pafs_output: str = "stage_pafs",
) -> Path:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    pose = onnx.load(str(pose_onnx))
    onnx.checker.check_model(pose)
    set_input_batch(pose, input_name, batch_size)

    graph = pose.graph

    def add_init(name: str, arr) -> None:
        # Replace an existing initializer with the same name if rerun through an
        # already adapted graph.
        kept = [i for i in graph.initializer if i.name != name]
        del graph.initializer[:]
        graph.initializer.extend(kept)
        graph.initializer.append(numpy_helper.from_array(np.asarray(arr), name=name))

    add_init("split_hm_starts", [0, 0, 0, 0])
    add_init("split_hm_ends", [int(batch_size), 18, 68, 121])
    add_init("split_hm_axes", [0, 1, 2, 3])
    add_init("split_hm_steps", [1, 1, 1, 1])

    graph.node.extend([
        helper.make_node(
            "Cast",
            [pose_heatmaps_output],
            ["split_heatmaps_f32_19"],
            name="split/cast_heatmaps_f32",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Slice",
            ["split_heatmaps_f32_19", "split_hm_starts", "split_hm_ends", "split_hm_axes", "split_hm_steps"],
            ["heatmaps"],
            name="split/slice_heatmaps_18",
        ),
        helper.make_node(
            "Cast",
            [pose_pafs_output],
            ["pafs"],
            name="split/cast_pafs_f32",
            to=TensorProto.FLOAT,
        ),
    ])

    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("heatmaps", TensorProto.FLOAT, [int(batch_size), 18, 68, 121]),
        helper.make_tensor_value_info("pafs", TensorProto.FLOAT, [int(batch_size), 38, 68, 121]),
    ])

    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(pose)
    onnx.save(pose, str(output_onnx))
    output_onnx.with_suffix(".debug.json").write_text(json.dumps({
        "kind": "split_pose_adapter_mxr1",
        "pose_onnx": str(pose_onnx),
        "output_onnx": str(output_onnx),
        "batch_size": int(batch_size),
        "input": input_name,
        "outputs": ["heatmaps", "pafs"],
        "output_shapes": {
            "heatmaps": [int(batch_size), 18, 68, 121],
            "pafs": [int(batch_size), 38, 68, 121],
        },
    }, indent=2))
    return output_onnx


def compile_mxr(onnx_path: str | Path, mxr_path: str | Path, batch_size: int, input_name: str = "input", exhaustive_tune: bool = False) -> Path:
    import migraphx  # type: ignore

    onnx_path = Path(onnx_path)
    mxr_path = Path(mxr_path)
    print(f"[compile] parse_onnx: {onnx_path}")
    parse_kwargs = {"map_input_dims": {input_name: [int(batch_size), 3, 544, 968]}}
    print(f"[compile] parse kwargs: {parse_kwargs}")
    program = migraphx.parse_onnx(str(onnx_path), **parse_kwargs)
    print("[compile] compile target=gpu")
    t0 = time.time()
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    mxr_path.parent.mkdir(parents=True, exist_ok=True)
    migraphx.save(program, str(mxr_path))
    print(f"[compile] saved: {mxr_path}")
    print(f"[compile] elapsed_s: {time.time() - t0:.2f}")
    return mxr_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export/compile split MXR1: pose + adapter -> heatmaps/pafs.")
    p.add_argument("--pose-onnx", default="models/fp16_refinment1.onnx")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--output-onnx", required=True)
    p.add_argument("--output-mxr", default="")
    p.add_argument("--input-name", default="input")
    p.add_argument("--pose-heatmaps-output", default="stage_heatmaps")
    p.add_argument("--pose-pafs-output", default="stage_pafs")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--exhaustive-tune", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_onnx = export_pose_adapter(
        pose_onnx=args.pose_onnx,
        output_onnx=args.output_onnx,
        batch_size=int(args.batch_size),
        input_name=args.input_name,
        pose_heatmaps_output=args.pose_heatmaps_output,
        pose_pafs_output=args.pose_pafs_output,
    )
    print(f"[export] saved: {output_onnx}")
    if args.compile or args.output_mxr:
        if not args.output_mxr:
            raise ValueError("--output-mxr is required when --compile is set")
        compile_mxr(output_onnx, args.output_mxr, int(args.batch_size), input_name=args.input_name, exhaustive_tune=args.exhaustive_tune)


if __name__ == "__main__":
    main()
