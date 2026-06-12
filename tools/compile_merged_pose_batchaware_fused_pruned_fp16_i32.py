#!/usr/bin/env python3
"""
Merge pose ONNX with the experimental fp16/int32 fused-pruned postprocess ONNX.

Differences from the fp32 baseline merge:
  * stage_heatmaps stay fp16; only channel 0:18 is sliced.
  * stage_pafs stay fp16; no adapter Cast to fp32.
  * postprocess ONNX is expected to consume fp16 heatmaps/pafs and return fp16
    score tensors plus int32 index tensors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def set_input_batch(model, input_name: str, batch_size: int):
    for inp in model.graph.input:
        if inp.name == input_name:
            tt = inp.type.tensor_type
            if tt.HasField("shape"):
                tt.shape.dim[0].dim_value = int(batch_size)


def graph_inputs(model):
    initializer_names = {i.name for i in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def merge_pose_post(
    *,
    pose_onnx: str | Path,
    post_onnx: str | Path,
    output_onnx: str | Path,
    batch_size: int,
    input_name: str = "input",
    pose_heatmaps_output: str = "stage_heatmaps",
    pose_pafs_output: str = "stage_pafs",
):
    import onnx
    import numpy as np
    from onnx import compose, helper, numpy_helper

    pose = onnx.load(str(pose_onnx))
    post = onnx.load(str(post_onnx))
    onnx.checker.check_model(pose)
    onnx.checker.check_model(post)

    set_input_batch(pose, input_name, batch_size)

    pose_p = compose.add_prefix(pose, "pose/")
    post_p = compose.add_prefix(post, "post/")
    merged = compose.merge_models(pose_p, post_p, io_map=[])

    pose_input_p = "pose/" + input_name
    heatmaps_src = "pose/" + pose_heatmaps_output
    pafs_src = "pose/" + pose_pafs_output
    post_heatmaps_p = "post/heatmaps"
    post_pafs_p = "post/pafs"

    for gi in merged.graph.input:
        if gi.name == pose_input_p:
            gi.name = input_name
    for node in merged.graph.node:
        for i, n in enumerate(node.input):
            if n == pose_input_p:
                node.input[i] = input_name

    kept = []
    for gi in merged.graph.input:
        if gi.name not in {post_heatmaps_p, post_pafs_p}:
            kept.append(gi)
    del merged.graph.input[:]
    merged.graph.input.extend(kept)

    def add_init(name, arr):
        merged.graph.initializer.append(numpy_helper.from_array(np.asarray(arr), name=name))

    add_init("merge_hm_starts", [0, 0, 0, 0])
    add_init("merge_hm_ends", [int(batch_size), 18, 68, 121])
    add_init("merge_hm_axes", [0, 1, 2, 3])
    add_init("merge_hm_steps", [1, 1, 1, 1])

    # fp16 adapter: no Cast.  Heatmaps only need channel slicing; PAFs are wired
    # directly from pose output to postprocess input.
    merged.graph.node.extend([
        helper.make_node(
            "Slice",
            [heatmaps_src, "merge_hm_starts", "merge_hm_ends", "merge_hm_axes", "merge_hm_steps"],
            [post_heatmaps_p],
            name="merge/slice_heatmaps_18_fp16",
        ),
        helper.make_node("Identity", [pafs_src], [post_pafs_p], name="merge/identity_pafs_fp16"),
    ])

    del merged.graph.output[:]
    for out in post.graph.output:
        elem_type = out.type.tensor_type.elem_type
        shape = []
        if out.type.tensor_type.HasField("shape"):
            for j, d in enumerate(out.type.tensor_type.shape.dim):
                if j == 0:
                    shape.append(int(batch_size))
                elif d.HasField("dim_value"):
                    shape.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    shape.append(str(d.dim_param))
                else:
                    shape.append(None)
        else:
            shape = None
        merged.graph.output.append(helper.make_tensor_value_info("post/" + out.name, elem_type, shape))

    old_nodes = list(merged.graph.node)
    pose_nodes = []
    post_nodes = []
    adapter_nodes_reordered = []
    other_nodes = []

    for n in old_nodes:
        node_name = str(n.name)
        outputs = [str(o) for o in n.output]
        inputs = [str(i) for i in n.input]
        is_post = node_name.startswith("post/") or any(o.startswith("post/") for o in outputs)
        is_pose = node_name.startswith("pose/") or any(o.startswith("pose/") for o in outputs)
        is_adapter = (
            node_name.startswith("merge/")
            or any(o in ("post/heatmaps", "post/pafs") for o in outputs)
            or any(o.startswith("merge_") for o in outputs)
            or any(i.startswith("pose/") for i in inputs) and any(o.startswith("post/") for o in outputs)
        )

        if is_adapter:
            adapter_nodes_reordered.append(n)
        elif is_post:
            post_nodes.append(n)
        elif is_pose:
            pose_nodes.append(n)
        else:
            other_nodes.append(n)

    del merged.graph.node[:]
    merged.graph.node.extend(pose_nodes)
    merged.graph.node.extend(other_nodes)
    merged.graph.node.extend(adapter_nodes_reordered)
    merged.graph.node.extend(post_nodes)

    onnx.checker.check_model(merged)
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(output_onnx))
    output_onnx.with_suffix(".debug.json").write_text(json.dumps({
        "pose_onnx": str(pose_onnx),
        "post_onnx": str(post_onnx),
        "output_onnx": str(output_onnx),
        "batch_size": int(batch_size),
        "graph_inputs": graph_inputs(merged),
        "graph_outputs": [o.name for o in merged.graph.output],
        "adapter": "fp16_no_cast_heatmaps_slice_pafs_identity",
    }, indent=2))
    return output_onnx


def compile_mxr(onnx_path, mxr_path, batch_size, input_name="input", exhaustive_tune=False):
    import migraphx  # type: ignore

    print(f"[compile] parse_onnx: {onnx_path}")
    parse_kwargs = {"map_input_dims": {input_name: [int(batch_size), 3, 544, 968]}}
    print(f"[compile] parse kwargs: {parse_kwargs}")
    program = migraphx.parse_onnx(str(onnx_path), **parse_kwargs)
    print("[compile] compile target=gpu")
    t0 = time.time()
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    mxr_path = Path(mxr_path)
    mxr_path.parent.mkdir(parents=True, exist_ok=True)
    migraphx.save(program, str(mxr_path))
    print(f"[compile] saved: {mxr_path}")
    print(f"[compile] elapsed_s: {time.time() - t0:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge/compile pose + fp16/int32 batch-aware fused-pruned ONNX.")
    p.add_argument("--pose-onnx", default="models/fp16_refinment1.onnx")
    p.add_argument("--post-onnx", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--output-onnx", required=True)
    p.add_argument("--output-mxr", default="")
    p.add_argument("--input-name", default="input")
    p.add_argument("--pose-heatmaps-output", default="stage_heatmaps")
    p.add_argument("--pose-pafs-output", default="stage_pafs")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--exhaustive-tune", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_onnx = Path(args.output_onnx)

    if not args.compile_only:
        merge_pose_post(
            pose_onnx=args.pose_onnx,
            post_onnx=args.post_onnx,
            output_onnx=out_onnx,
            batch_size=args.batch_size,
            input_name=args.input_name,
            pose_heatmaps_output=args.pose_heatmaps_output,
            pose_pafs_output=args.pose_pafs_output,
        )
        print(f"[merge] saved: {out_onnx}")

    if args.merge_only:
        return

    mxr = args.output_mxr or str(out_onnx.with_suffix(".mxr"))
    compile_mxr(out_onnx, mxr, args.batch_size, args.input_name, args.exhaustive_tune)


if __name__ == "__main__":
    main()
