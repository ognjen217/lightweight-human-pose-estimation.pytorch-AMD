#!/usr/bin/env python3
"""Inspect ONNX graph structure for MIGraphX graph-cleanup experiments.

The script is intentionally dependency-light: it only requires onnx and writes a
human-readable summary plus an optional JSON report. It helps compare baseline
and graph-cleaned exports before running MIGraphX/rocprof.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


INTERESTING_OPS = {
    "Cast",
    "Concat",
    "Constant",
    "DequantizeLinear",
    "Div",
    "Gather",
    "Identity",
    "Mul",
    "QuantizeLinear",
    "ReduceSum",
    "Reshape",
    "Shape",
    "Slice",
    "Sqrt",
    "TopK",
    "Transpose",
    "Unsqueeze",
}


def dtype_name(elem_type: int | None) -> str:
    if elem_type is None:
        return "unknown"
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return f"unknown({elem_type})"


def value_info_dtype_map(model: onnx.ModelProto) -> dict[str, str]:
    dtype_by_name: dict[str, str] = {}

    def read_value_info(value_info: Any) -> None:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type:
            dtype_by_name[value_info.name] = dtype_name(tensor_type.elem_type)

    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        read_value_info(item)

    for init in model.graph.initializer:
        dtype_by_name[init.name] = dtype_name(init.data_type)

    return dtype_by_name


def tensor_shape(value_info: Any) -> list[str]:
    tensor_type = value_info.type.tensor_type
    dims: list[str] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return dims


def summarize_graph(model: onnx.ModelProto) -> dict[str, Any]:
    op_counts = Counter(node.op_type for node in model.graph.node)
    dtype_by_name = value_info_dtype_map(model)

    inputs = [
        {
            "name": inp.name,
            "dtype": dtype_by_name.get(inp.name, "unknown"),
            "shape": tensor_shape(inp),
        }
        for inp in model.graph.input
    ]
    outputs = [
        {
            "name": out.name,
            "dtype": dtype_by_name.get(out.name, "unknown"),
            "shape": tensor_shape(out),
        }
        for out in model.graph.output
    ]

    interesting_nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, node in enumerate(model.graph.node):
        if node.op_type in INTERESTING_OPS:
            interesting_nodes[node.op_type].append(
                {
                    "index": idx,
                    "name": node.name or f"{node.op_type}_{idx}",
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "input_dtypes": [dtype_by_name.get(name, "unknown") for name in node.input],
                    "output_dtypes": [dtype_by_name.get(name, "unknown") for name in node.output],
                }
            )

    producer_by_output = {}
    for idx, node in enumerate(model.graph.node):
        for output in node.output:
            producer_by_output[output] = {
                "index": idx,
                "op_type": node.op_type,
                "name": node.name or f"{node.op_type}_{idx}",
                "inputs": list(node.input),
                "outputs": list(node.output),
            }

    output_producers = {
        output["name"]: producer_by_output.get(output["name"], None)
        for output in outputs
    }

    suspicious_tail_nodes = []
    output_names = {output["name"] for output in outputs}
    for idx, node in enumerate(model.graph.node):
        if any(out in output_names for out in node.output) or node.op_type in {"Cast", "DequantizeLinear"}:
            suspicious_tail_nodes.append(
                {
                    "index": idx,
                    "op_type": node.op_type,
                    "name": node.name or f"{node.op_type}_{idx}",
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "input_dtypes": [dtype_by_name.get(name, "unknown") for name in node.input],
                    "output_dtypes": [dtype_by_name.get(name, "unknown") for name in node.output],
                }
            )

    return {
        "ir_version": model.ir_version,
        "opset_imports": [
            {"domain": opset.domain or "ai.onnx", "version": opset.version}
            for opset in model.opset_import
        ],
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
        "inputs": inputs,
        "outputs": outputs,
        "op_counts": dict(op_counts.most_common()),
        "interesting_op_counts": {op: op_counts.get(op, 0) for op in sorted(INTERESTING_OPS)},
        "interesting_nodes": dict(interesting_nodes),
        "output_producers": output_producers,
        "suspicious_tail_nodes": suspicious_tail_nodes,
    }


def format_summary(summary: dict[str, Any], max_nodes_per_op: int) -> str:
    lines = []
    lines.append("ONNX graph inspection")
    lines.append("=" * 80)
    lines.append(f"Nodes:        {summary['num_nodes']}")
    lines.append(f"Initializers: {summary['num_initializers']}")
    lines.append(f"Opsets:       {summary['opset_imports']}")
    lines.append("")

    lines.append("Inputs")
    for inp in summary["inputs"]:
        lines.append(f"  - {inp['name']}: dtype={inp['dtype']} shape={inp['shape']}")
    lines.append("")

    lines.append("Outputs")
    for out in summary["outputs"]:
        lines.append(f"  - {out['name']}: dtype={out['dtype']} shape={out['shape']}")
    lines.append("")

    lines.append("Top operator counts")
    for op, count in list(summary["op_counts"].items())[:30]:
        lines.append(f"  {op:24s} {count}")
    lines.append("")

    lines.append("Interesting operator counts")
    for op, count in summary["interesting_op_counts"].items():
        if count:
            lines.append(f"  {op:24s} {count}")
    lines.append("")

    lines.append("Output producers")
    for name, producer in summary["output_producers"].items():
        lines.append(f"  - {name}: {producer}")
    lines.append("")

    lines.append("Suspicious tail / conversion nodes")
    for node in summary["suspicious_tail_nodes"][: max_nodes_per_op * 4]:
        lines.append(
            f"  [{node['index']}] {node['op_type']} {node['name']} "
            f"in={node['input_dtypes']} out={node['output_dtypes']}"
        )
    lines.append("")

    lines.append(f"Interesting node samples, max {max_nodes_per_op} per op")
    for op, nodes in sorted(summary["interesting_nodes"].items()):
        if not nodes:
            continue
        lines.append(f"\n{op}:")
        for node in nodes[:max_nodes_per_op]:
            lines.append(
                f"  [{node['index']}] {node['name']} "
                f"in={node['input_dtypes']} out={node['output_dtypes']}"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--txt-out", type=Path, default=None)
    parser.add_argument("--max-nodes-per-op", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.onnx_path)
    summary = summarize_graph(model)
    text = format_summary(summary, args.max_nodes_per_op)

    print(text)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON report: {args.json_out}")

    if args.txt_out:
        args.txt_out.parent.mkdir(parents=True, exist_ok=True)
        args.txt_out.write_text(text, encoding="utf-8")
        print(f"Wrote text report: {args.txt_out}")


if __name__ == "__main__":
    main()
