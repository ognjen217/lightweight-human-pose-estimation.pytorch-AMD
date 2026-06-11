#!/usr/bin/env python3
"""
Analyze ONNX graphs used for MIGraphX post-processing optimization.

The tool is intentionally read-only. It summarizes graph structure so that
accuracy-preserving rewrites can be targeted at the expensive parts of the
compiled MXR graph: TopK, Gather/GatherElements, Where/logical chains, Casts,
Reshapes, and large intermediate tensors.

Example:
    python tools/analyze_mx_onnx_graph.py \
      models/merged_pose_fused_pruned_batchaware/model.onnx \
      --json outputs/graph_analysis.json \
      --markdown outputs/graph_analysis.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: onnx. Install it in the active environment, e.g. `pip install onnx`."
    ) from exc


POST_OPS_OF_INTEREST = {
    "TopK",
    "Gather",
    "GatherElements",
    "Where",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Equal",
    "And",
    "Or",
    "Not",
    "Cast",
    "Reshape",
    "Transpose",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "ReduceSum",
    "MaxPool",
    "Mul",
    "Add",
    "Sub",
    "Div",
    "Clip",
}

SHAPE_UNKNOWN = "?"


@dataclass
class TensorInfo:
    name: str
    shape: List[Any]
    dtype: str = ""
    numel: Optional[int] = None


@dataclass
class NodeRecord:
    index: int
    name: str
    op_type: str
    namespace: str
    inputs: List[str]
    outputs: List[str]
    input_shapes: Dict[str, List[Any]]
    output_shapes: Dict[str, List[Any]]
    attrs: Dict[str, Any]


@dataclass
class GraphAnalysis:
    model_path: str
    ir_version: int
    opset_imports: Dict[str, int]
    num_nodes: int
    num_initializers: int
    num_inputs: int
    num_outputs: int
    op_counts: Dict[str, int]
    namespace_counts: Dict[str, int]
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    topk_nodes: List[NodeRecord]
    gather_nodes: List[NodeRecord]
    mask_logic_nodes: List[NodeRecord]
    cast_nodes: List[NodeRecord]
    reshape_like_nodes: List[NodeRecord]
    largest_intermediate_tensors: List[TensorInfo]
    largest_initializers: List[TensorInfo]
    suspicious_chains: Dict[str, List[List[str]]]
    duplicate_initializer_groups: List[Dict[str, Any]]
    recommendations: List[str]


def _safe_dim(dim: Any) -> Any:
    if dim is None:
        return SHAPE_UNKNOWN
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        return dim if dim else SHAPE_UNKNOWN
    return str(dim)


def _tensor_shape_from_value_info(value_info: Any) -> Tuple[List[Any], str]:
    try:
        t = value_info.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(t.elem_type) if t.elem_type else ""
        dims: List[Any] = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(str(d.dim_param))
            else:
                dims.append(SHAPE_UNKNOWN)
        return dims, dtype
    except Exception:
        return [], ""


def _numel(shape: Sequence[Any]) -> Optional[int]:
    n = 1
    for d in shape:
        if not isinstance(d, int) or d < 0:
            return None
        n *= d
    return int(n)


def _namespace(node_name: str) -> str:
    if not node_name:
        return "<unnamed>"
    if "/" in node_name:
        return node_name.split("/", 1)[0]
    return "<root>"


def _attr_to_python(attr: Any) -> Any:
    from onnx import helper

    value = helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="replace"))
            elif hasattr(v, "name") and hasattr(v, "dims"):
                out.append(f"Tensor<{getattr(v, 'name', '')}, dims={list(getattr(v, 'dims', []))}>")
            else:
                out.append(v)
        return out
    if hasattr(value, "name") and hasattr(value, "dims"):
        return f"Tensor<{getattr(value, 'name', '')}, dims={list(getattr(value, 'dims', []))}>"
    return value


def _shape_map(model: Any) -> Dict[str, TensorInfo]:
    info: Dict[str, TensorInfo] = {}

    def add_vi(vi: Any) -> None:
        shape, dtype = _tensor_shape_from_value_info(vi)
        info[vi.name] = TensorInfo(vi.name, shape, dtype, _numel(shape))

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        add_vi(vi)

    for init in model.graph.initializer:
        shape = [int(x) for x in init.dims]
        dtype = onnx.TensorProto.DataType.Name(init.data_type) if init.data_type else ""
        info[init.name] = TensorInfo(init.name, shape, dtype, _numel(shape))

    return info


def _infer_shapes(model: Any) -> Any:
    try:
        return shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception as exc:
        print(f"[WARN] ONNX shape inference failed, continuing with existing shapes: {exc}")
        return model


def _node_records(model: Any, tensors: Mapping[str, TensorInfo]) -> List[NodeRecord]:
    records: List[NodeRecord] = []
    for i, node in enumerate(model.graph.node):
        attrs = {a.name: _attr_to_python(a) for a in node.attribute}
        in_shapes = {name: tensors[name].shape for name in node.input if name in tensors}
        out_shapes = {name: tensors[name].shape for name in node.output if name in tensors}
        records.append(
            NodeRecord(
                index=i,
                name=node.name or f"<{node.op_type}_{i}>",
                op_type=node.op_type,
                namespace=_namespace(node.name),
                inputs=list(node.input),
                outputs=list(node.output),
                input_shapes=in_shapes,
                output_shapes=out_shapes,
                attrs=attrs,
            )
        )
    return records


def _largest_tensors(tensors: Iterable[TensorInfo], limit: int) -> List[TensorInfo]:
    known = [t for t in tensors if t.numel is not None]
    return sorted(known, key=lambda t: int(t.numel or 0), reverse=True)[:limit]


def _find_chains(records: Sequence[NodeRecord], chain_specs: Sequence[Tuple[str, ...]], limit: int = 50) -> Dict[str, List[List[str]]]:
    producer: Dict[str, NodeRecord] = {}
    consumers: Dict[str, List[NodeRecord]] = defaultdict(list)
    for rec in records:
        for out in rec.outputs:
            producer[out] = rec
        for inp in rec.inputs:
            consumers[inp].append(rec)

    found: Dict[str, List[List[str]]] = {"->".join(spec): [] for spec in chain_specs}
    for spec in chain_specs:
        key = "->".join(spec)
        for rec in records:
            if rec.op_type != spec[0]:
                continue
            paths = [[rec]]
            for expected in spec[1:]:
                new_paths = []
                for path in paths:
                    last = path[-1]
                    next_nodes: List[NodeRecord] = []
                    for out in last.outputs:
                        next_nodes.extend(consumers.get(out, []))
                    for nxt in next_nodes:
                        if nxt.op_type == expected:
                            new_paths.append(path + [nxt])
                paths = new_paths
                if not paths:
                    break
            for path in paths:
                found[key].append([f"{p.index}:{p.name}:{p.op_type}" for p in path])
                if len(found[key]) >= limit:
                    break
            if len(found[key]) >= limit:
                break
    return found


def _initializer_duplicate_groups(model: Any, limit: int = 20) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, Tuple[int, ...], bytes], List[str]] = defaultdict(list)
    for init in model.graph.initializer:
        # Hash only reasonably small tensors to avoid expensive work on huge weights.
        size = 1
        for d in init.dims:
            size *= int(d)
        if size > 4096:
            continue
        try:
            arr = numpy_helper.to_array(init)
            key = (str(arr.dtype), tuple(int(d) for d in arr.shape), arr.tobytes())
        except Exception:
            key = (str(init.data_type), tuple(int(d) for d in init.dims), bytes(init.raw_data))
        groups[key].append(init.name)

    out = []
    for (dtype, shape, _), names in groups.items():
        if len(names) > 1:
            out.append({"dtype": dtype, "shape": list(shape), "count": len(names), "names": names[:50]})
    return sorted(out, key=lambda x: x["count"], reverse=True)[:limit]


def _recommendations(records: Sequence[NodeRecord], op_counts: Mapping[str, int]) -> List[str]:
    recs: List[str] = []
    if op_counts.get("GatherElements", 0) + op_counts.get("Gather", 0) > 100:
        recs.append(
            "High Gather/GatherElements count: inspect PAF sampling/indexing and try a semantics-preserving batched gather rewrite."
        )
    if op_counts.get("Where", 0) > 100:
        recs.append(
            "High Where count: inspect mask/logical subgraphs and test equivalent mask algebra or Cast/Where cleanup."
        )
    if op_counts.get("Cast", 0) > 50:
        recs.append("High Cast count: remove redundant FP16/FP32/int casts around masks, TopK indices, and constants where safe.")
    if op_counts.get("Reshape", 0) + op_counts.get("Unsqueeze", 0) + op_counts.get("Squeeze", 0) > 200:
        recs.append("Many shape-only ops: run ONNX simplification/constant folding and remove adjacent shape chains.")
    for node in records:
        if node.op_type == "TopK":
            in_shape = next(iter(node.input_shapes.values()), [])
            if any(isinstance(d, int) and d >= 1_000_000 for d in in_shape):
                recs.append(
                    f"TopK node {node.name!r} sees a very large input shape {in_shape}; this is likely a major MIGraphX kernel hotspot."
                )
    if not recs:
        recs.append("No obvious structural hotspot detected by heuristics; inspect PFTRACE kernel groups and ONNX node timings.")
    return recs


def analyze(path: Path, top_limit: int = 30) -> GraphAnalysis:
    original = onnx.load(str(path))
    model = _infer_shapes(original)
    tensors = _shape_map(model)
    records = _node_records(model, tensors)

    op_counts = Counter(rec.op_type for rec in records)
    namespace_counts = Counter(rec.namespace for rec in records)

    graph_input_names = {x.name for x in model.graph.input}
    initializer_names = {x.name for x in model.graph.initializer}
    real_inputs = [name for name in graph_input_names if name not in initializer_names]

    input_infos = [tensors.get(name, TensorInfo(name, [], "", None)) for name in real_inputs]
    output_infos = [tensors.get(x.name, TensorInfo(x.name, [], "", None)) for x in model.graph.output]

    output_tensor_names = {name for rec in records for name in rec.outputs}
    intermediate_tensors = [tensors[name] for name in output_tensor_names if name in tensors]

    init_infos = [tensors[x.name] for x in model.graph.initializer if x.name in tensors]

    chain_specs = [
        ("Cast", "Cast"),
        ("Reshape", "Reshape"),
        ("Unsqueeze", "Concat", "Reshape"),
        ("Greater", "Cast"),
        ("Equal", "Cast"),
        ("Greater", "And", "Where"),
        ("Equal", "And", "Where"),
    ]

    return GraphAnalysis(
        model_path=str(path),
        ir_version=int(model.ir_version),
        opset_imports={op.domain or "ai.onnx": int(op.version) for op in model.opset_import},
        num_nodes=len(records),
        num_initializers=len(model.graph.initializer),
        num_inputs=len(input_infos),
        num_outputs=len(output_infos),
        op_counts=dict(op_counts.most_common()),
        namespace_counts=dict(namespace_counts.most_common()),
        inputs=input_infos,
        outputs=output_infos,
        topk_nodes=[rec for rec in records if rec.op_type == "TopK"],
        gather_nodes=[rec for rec in records if rec.op_type in {"Gather", "GatherElements"}],
        mask_logic_nodes=[rec for rec in records if rec.op_type in {"Where", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal", "And", "Or", "Not"}],
        cast_nodes=[rec for rec in records if rec.op_type == "Cast"],
        reshape_like_nodes=[rec for rec in records if rec.op_type in {"Reshape", "Transpose", "Unsqueeze", "Squeeze", "Concat"}],
        largest_intermediate_tensors=_largest_tensors(intermediate_tensors, top_limit),
        largest_initializers=_largest_tensors(init_infos, top_limit),
        suspicious_chains=_find_chains(records, chain_specs),
        duplicate_initializer_groups=_initializer_duplicate_groups(model),
        recommendations=_recommendations(records, op_counts),
    )


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _format_tensor(t: TensorInfo) -> str:
    numel = "?" if t.numel is None else f"{t.numel:,}"
    return f"`{t.name}` | `{t.dtype}` | `{t.shape}` | {numel}"


def write_markdown(analysis: GraphAnalysis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# ONNX graph analysis: `{Path(analysis.model_path).name}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Nodes: **{analysis.num_nodes:,}**")
    lines.append(f"- Initializers: **{analysis.num_initializers:,}**")
    lines.append(f"- Inputs: **{analysis.num_inputs}**")
    lines.append(f"- Outputs: **{analysis.num_outputs}**")
    lines.append(f"- IR version: `{analysis.ir_version}`")
    lines.append(f"- Opsets: `{analysis.opset_imports}`")
    lines.append("")

    lines.append("## Top op counts")
    lines.append("")
    lines.append("| Op | Count |")
    lines.append("|---|---:|")
    for op, count in list(analysis.op_counts.items())[:40]:
        lines.append(f"| `{op}` | {count:,} |")
    lines.append("")

    lines.append("## Namespace counts")
    lines.append("")
    lines.append("| Namespace | Count |")
    lines.append("|---|---:|")
    for ns, count in list(analysis.namespace_counts.items())[:40]:
        lines.append(f"| `{ns}` | {count:,} |")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append("| Name | Dtype | Shape | Numel |")
    lines.append("|---|---|---|---:|")
    for t in analysis.inputs:
        lines.append("| " + _format_tensor(t) + " |")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append("| Name | Dtype | Shape | Numel |")
    lines.append("|---|---|---|---:|")
    for t in analysis.outputs:
        lines.append("| " + _format_tensor(t) + " |")
    lines.append("")

    def node_table(title: str, nodes: Sequence[NodeRecord], limit: int = 40) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| # | Name | Op | Inputs | Outputs | Attrs |")
        lines.append("|---:|---|---|---|---|---|")
        for n in nodes[:limit]:
            attrs = json.dumps(n.attrs, ensure_ascii=False)[:300]
            inputs = "<br>".join(f"`{k}` {v}" for k, v in n.input_shapes.items())
            outputs = "<br>".join(f"`{k}` {v}" for k, v in n.output_shapes.items())
            lines.append(f"| {n.index} | `{n.name}` | `{n.op_type}` | {inputs} | {outputs} | `{attrs}` |")
        if len(nodes) > limit:
            lines.append(f"| ... | ... | ... | ... | ... | truncated, total {len(nodes):,} |")
        lines.append("")

    node_table("TopK nodes", analysis.topk_nodes)
    node_table("Gather / GatherElements nodes", analysis.gather_nodes)
    node_table("Mask / logical nodes", analysis.mask_logic_nodes)
    node_table("Cast nodes", analysis.cast_nodes)

    lines.append("## Largest intermediate tensors")
    lines.append("")
    lines.append("| Name | Dtype | Shape | Numel |")
    lines.append("|---|---|---|---:|")
    for t in analysis.largest_intermediate_tensors:
        lines.append("| " + _format_tensor(t) + " |")
    lines.append("")

    lines.append("## Suspicious chains")
    lines.append("")
    for key, chains in analysis.suspicious_chains.items():
        lines.append(f"### `{key}`")
        if not chains:
            lines.append("No chains found.")
        else:
            for chain in chains[:20]:
                lines.append("- " + " -> ".join(f"`{x}`" for x in chain))
        lines.append("")

    lines.append("## Duplicate small initializer groups")
    lines.append("")
    if not analysis.duplicate_initializer_groups:
        lines.append("No duplicate small initializers detected.")
    else:
        for group in analysis.duplicate_initializer_groups:
            lines.append(f"- dtype={group['dtype']} shape={group['shape']} count={group['count']} names={group['names'][:10]}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for rec in analysis.recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze an ONNX graph for MIGraphX optimization work.")
    p.add_argument("onnx", type=Path, help="Path to ONNX model.")
    p.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    p.add_argument("--markdown", "--md", type=Path, default=None, help="Optional Markdown output path.")
    p.add_argument("--top-limit", type=int, default=30, help="Number of largest tensors to report.")
    p.add_argument("--print-summary", action="store_true", help="Print compact summary to stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(args.onnx, top_limit=args.top_limit)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(_to_jsonable(analysis), indent=2, sort_keys=False), encoding="utf-8")
        print(f"[OK] wrote JSON: {args.json}")

    if args.markdown:
        write_markdown(analysis, args.markdown)
        print(f"[OK] wrote Markdown: {args.markdown}")

    if args.print_summary or not args.json and not args.markdown:
        print(json.dumps({
            "model": analysis.model_path,
            "num_nodes": analysis.num_nodes,
            "op_counts_top20": dict(list(analysis.op_counts.items())[:20]),
            "namespace_counts": analysis.namespace_counts,
            "num_topk": len(analysis.topk_nodes),
            "num_gather": len(analysis.gather_nodes),
            "num_mask_logic": len(analysis.mask_logic_nodes),
            "recommendations": analysis.recommendations,
        }, indent=2))


if __name__ == "__main__":
    main()
