#!/usr/bin/env python3
import onnx
import numpy as np
from pathlib import Path
from onnx import numpy_helper

ONNX = Path("models/merged_pose_fused_pruned_batchaware/pose_fused_pruned_batchaware_b1_1080x1920_k20_m20_thr0p1_r6_separable.onnx")

m = onnx.load(str(ONNX))

init = {x.name: numpy_helper.to_array(x) for x in m.graph.initializer}

print("=== INPUTS ===")
for x in m.graph.input:
    dims = []
    for d in x.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else d.dim_param)
    print(x.name, dims)

print("\n=== OUTPUTS ===")
for x in m.graph.output:
    dims = []
    for d in x.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else d.dim_param)
    print(x.name, dims)

print("\n=== RESHAPE NODES ===")
for i, n in enumerate(m.graph.node):
    if n.op_type != "Reshape":
        continue

    shape_info = ""
    if len(n.input) >= 2 and n.input[1] in init:
        arr = init[n.input[1]]
        shape_info = f" shape_const={arr.tolist()} product={int(np.prod(arr)) if np.all(arr > 0) else 'dynamic'}"

    print(f"[{i}] name={n.name}")
    print(f"    inputs={list(n.input)}")
    print(f"    outputs={list(n.output)}")
    print(f"   {shape_info}")

print("\n=== CHECK MODEL ===")
try:
    onnx.checker.check_model(m)
    print("onnx.checker: OK")
except Exception as e:
    print("onnx.checker ERROR:", repr(e))
