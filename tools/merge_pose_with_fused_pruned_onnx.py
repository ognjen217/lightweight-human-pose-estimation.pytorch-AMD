#!/usr/bin/env python3
"""
Merge pose model ONNX with fused-pruned postprocess ONNX.

Goal
----
Create a monolithic ONNX graph:

    image/input tensor
      -> pose model
      -> adapter Slice nodes
      -> fused_pruned postprocess graph, replicated per batch sample
      -> 6 small pruned outputs

This is the ONNX-level step needed before compiling one MIGraphX .mxr:
    CPU -> GPU(merged pose + postprocess) -> CPU

Why graph replication for B=2?
------------------------------
The current fused-pruned postprocess ONNX tail was originally written for a
single sample. In particular, its pruning tail reshapes pair_scores to
[19, K*K] and applies TopK per limb. Therefore the safest first batched
version is to slice each pose output sample and feed a separate copy of the
fused-pruned subgraph. Outputs are then stacked/concatenated to include a
leading batch dimension.

This is intentionally conservative and easier to validate than rewriting the
postprocess graph to be fully batch-aware in one pass.

Example:
  PYTHONPATH=. python tools/merge_pose_with_fused_pruned_onnx.py \
    --pose-onnx models/fp16_refinment1.onnx \
    --fused-pruned-onnx models/fused_postprocess_pruned_cache/fused_cubic_topk_fullres_paf_pruned_68x121_to_1080x1920_k20_m20_thr0p1_r6_separable_ham0p75_p8_min0p05_sr0p8_pam0p75_mp0p0.onnx \
    --batch-size 2 \
    --output-onnx models/merged_pose_fused_pruned/pose_fused_pruned_b2_1080x1920.onnx
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import onnx
from onnx import TensorProto, helper, shape_inference


PRUNED_OUTPUT_BASE_NAMES = [
    "top_scores",
    "top_indices",
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]


def _dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def _dims(vi: Any) -> List[Any]:
    shape = vi.type.tensor_type.shape
    out = []
    for d in shape.dim:
        if d.dim_value:
            out.append(int(d.dim_value))
        elif d.dim_param:
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def _dtype(vi: Any) -> int:
    return int(vi.type.tensor_type.elem_type)


def _vi_map(model: onnx.ModelProto) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        m[vi.name] = vi
    return m


def _shape_map(model: onnx.ModelProto) -> Dict[str, Tuple[int, List[Any]]]:
    out: Dict[str, Tuple[int, List[Any]]] = {}
    for name, vi in _vi_map(model).items():
        try:
            out[name] = (_dtype(vi), _dims(vi))
        except Exception:
            pass
    return out


def _find_by_name(values: Iterable[Any], name: str) -> Any:
    for v in values:
        if v.name == name:
            return v
    raise KeyError(name)


def _guess_pose_outputs(
    pose: onnx.ModelProto,
    heatmaps_name: str = "",
    pafs_name: str = "",
    combined_name: str = "",
) -> Tuple[str, Optional[str], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Return (heatmaps_or_combined_name, pafs_name_or_none, heatmap_channel_range, paf_channel_range).

    If separate outputs are detected:
      returns heatmap_name, paf_name, None, None.

    If a combined output [B,57,H,W] is detected:
      returns combined_name, None, (0,19), (19,57) by default.

    Notes:
      - We choose the last matching output to prefer final refinement outputs.
      - Heatmap can be 18 or 19 channels; adapter will later slice to fused input C.
    """
    outputs = list(pose.graph.output)
    smap = _shape_map(pose)

    if heatmaps_name and pafs_name:
        return heatmaps_name, pafs_name, None, None

    if combined_name:
        return combined_name, None, (0, 19), (19, 57)

    heat_candidates: List[str] = []
    paf_candidates: List[str] = []
    combined_candidates: List[str] = []

    for out in outputs:
        name = out.name
        dims = smap.get(name, (None, []))[1]
        if len(dims) != 4:
            continue
        c = dims[1]
        if c in (18, 19):
            heat_candidates.append(name)
        elif c == 38:
            paf_candidates.append(name)
        elif c == 57:
            combined_candidates.append(name)

    if heatmaps_name:
        heat_candidates = [heatmaps_name]
    if pafs_name:
        paf_candidates = [pafs_name]

    if heat_candidates and paf_candidates:
        return heat_candidates[-1], paf_candidates[-1], None, None

    if combined_candidates:
        return combined_candidates[-1], None, (0, 19), (19, 57)

    raise RuntimeError(
        "Could not auto-detect pose heatmaps/PAFs outputs. "
        "Pass --pose-heatmaps-output and --pose-pafs-output, or --pose-combined-output.\n"
        f"Pose outputs: {[(o.name, smap.get(o.name, ('?', []))[1]) for o in outputs]}"
    )


def _find_fused_inputs(fused: onnx.ModelProto, heatmaps_name: str = "", pafs_name: str = "") -> Tuple[str, str]:
    inputs = list(fused.graph.input)

    if heatmaps_name and pafs_name:
        return heatmaps_name, pafs_name

    by_name = {x.name: x for x in inputs}
    if not heatmaps_name:
        if "heatmaps" in by_name:
            heatmaps_name = "heatmaps"
        else:
            for x in inputs:
                if "heat" in x.name.lower():
                    heatmaps_name = x.name
                    break
    if not pafs_name:
        if "pafs" in by_name:
            pafs_name = "pafs"
        elif "paf" in by_name:
            pafs_name = "paf"
        else:
            for x in inputs:
                if "paf" in x.name.lower():
                    pafs_name = x.name
                    break

    if not heatmaps_name or not pafs_name:
        raise RuntimeError(f"Could not detect fused inputs. Inputs: {[x.name for x in inputs]}")

    return heatmaps_name, pafs_name


def _find_output_containing(model: onnx.ModelProto, token: str) -> str:
    for out in model.graph.output:
        if token in out.name:
            return out.name
    raise RuntimeError(f"Could not find fused output containing {token!r}. Outputs: {[o.name for o in model.graph.output]}")


def _const_i64(graph: onnx.GraphProto, name: str, values: Sequence[int]) -> None:
    import numpy as np
    from onnx import numpy_helper

    graph.initializer.append(numpy_helper.from_array(np.asarray(list(values), dtype=np.int64), name=name))


def _make_slice_node(
    graph: onnx.GraphProto,
    input_name: str,
    output_name: str,
    *,
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Sequence[int],
    steps: Optional[Sequence[int]] = None,
    node_name: str,
) -> None:
    if steps is None:
        steps = [1] * len(starts)
    prefix = node_name.replace("/", "_")
    starts_name = f"{prefix}_starts"
    ends_name = f"{prefix}_ends"
    axes_name = f"{prefix}_axes"
    steps_name = f"{prefix}_steps"
    _const_i64(graph, starts_name, starts)
    _const_i64(graph, ends_name, ends)
    _const_i64(graph, axes_name, axes)
    _const_i64(graph, steps_name, steps)
    graph.node.append(
        helper.make_node(
            "Slice",
            [input_name, starts_name, ends_name, axes_name, steps_name],
            [output_name],
            name=node_name,
        )
    )


def _prefix_name(name: str, prefix: str, external_map: Dict[str, str]) -> str:
    if name == "":
        return ""
    if name in external_map:
        return external_map[name]
    return f"{prefix}{name}"


def _append_prefixed_fused_graph(
    merged_graph: onnx.GraphProto,
    fused: onnx.ModelProto,
    *,
    prefix: str,
    fused_heatmaps_input: str,
    fused_pafs_input: str,
    heatmaps_source: str,
    pafs_source: str,
) -> Dict[str, str]:
    """
    Append a copy of the fused graph with all internal names prefixed.
    Returns map from original fused output names to new prefixed names.
    """
    external_map = {
        fused_heatmaps_input: heatmaps_source,
        fused_pafs_input: pafs_source,
    }

    # Initializers
    for init in fused.graph.initializer:
        new_init = copy.deepcopy(init)
        new_init.name = _prefix_name(init.name, prefix, external_map)
        merged_graph.initializer.append(new_init)

    # Value infos are not required for execution, but useful for debugging.
    for vi in fused.graph.value_info:
        new_vi = copy.deepcopy(vi)
        new_vi.name = _prefix_name(vi.name, prefix, external_map)
        merged_graph.value_info.append(new_vi)

    # Nodes
    for node in fused.graph.node:
        new_node = copy.deepcopy(node)
        new_node.name = f"{prefix}{node.name}" if node.name else ""
        new_node.input[:] = [_prefix_name(x, prefix, external_map) for x in node.input]
        new_node.output[:] = [_prefix_name(x, prefix, external_map) for x in node.output]
        merged_graph.node.append(new_node)

    out_map: Dict[str, str] = {}
    for out in fused.graph.output:
        out_map[out.name] = _prefix_name(out.name, prefix, external_map)
    return out_map


def _needs_unsqueeze_for_batch(output_shape: List[Any]) -> bool:
    # If fused output already includes leading batch dim 1, concat directly.
    # Otherwise add an explicit leading sample dim.
    if output_shape and output_shape[0] == 1:
        return False
    return True


def _add_batched_outputs(
    graph: onnx.GraphProto,
    sample_outputs: List[Dict[str, str]],
    fused_output_info: Dict[str, Tuple[int, List[Any]]],
    *,
    batch_size: int,
    output_prefix: str,
) -> List[str]:
    """
    Convert per-sample fused outputs into 6 final graph outputs with leading batch axis.
    """
    _const_i64(graph, "merged_unsqueeze_axis0", [0])
    final_names: List[str] = []

    for base in PRUNED_OUTPUT_BASE_NAMES:
        # Find original fused output key containing this base token.
        original_name = None
        for k in sample_outputs[0].keys():
            if base in k:
                original_name = k
                break
        if original_name is None:
            raise RuntimeError(f"Could not find fused output for token {base}")

        dtype, shape = fused_output_info.get(original_name, (TensorProto.FLOAT, []))
        prepared: List[str] = []

        for i, out_map in enumerate(sample_outputs):
            sample_out = out_map[original_name]
            if _needs_unsqueeze_for_batch(shape):
                unsq = f"{output_prefix}/sample{i}/{base}_batched"
                graph.node.append(
                    helper.make_node(
                        "Unsqueeze",
                        [sample_out, "merged_unsqueeze_axis0"],
                        [unsq],
                        name=f"{output_prefix}/sample{i}/unsqueeze_{base}",
                    )
                )
                prepared.append(unsq)
            else:
                prepared.append(sample_out)

        final_name = f"{output_prefix}/{base}"
        if batch_size == 1:
            graph.node.append(helper.make_node("Identity", [prepared[0]], [final_name], name=f"{output_prefix}/identity_{base}"))
        else:
            graph.node.append(helper.make_node("Concat", prepared, [final_name], name=f"{output_prefix}/concat_{base}", axis=0))

        # Compute output shape best-effort.
        if shape:
            if _needs_unsqueeze_for_batch(shape):
                out_shape = [batch_size] + list(shape)
            else:
                out_shape = list(shape)
                out_shape[0] = batch_size
        else:
            out_shape = None

        graph.output.append(helper.make_tensor_value_info(final_name, dtype, out_shape))
        final_names.append(final_name)

    return final_names


def merge_pose_with_fused_pruned(
    *,
    pose_onnx: str | Path,
    fused_pruned_onnx: str | Path,
    output_onnx: str | Path,
    batch_size: int,
    pose_heatmaps_output: str = "",
    pose_pafs_output: str = "",
    pose_combined_output: str = "",
    fused_heatmaps_input: str = "",
    fused_pafs_input: str = "",
    fused_heatmap_channels: int = 18,
    infer_shapes: bool = True,
    use_external_data: bool = False,
) -> Path:
    pose_onnx = Path(pose_onnx)
    fused_pruned_onnx = Path(fused_pruned_onnx)
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    pose = onnx.load(str(pose_onnx), load_external_data=True)
    fused = onnx.load(str(fused_pruned_onnx), load_external_data=True)

    if infer_shapes:
        try:
            pose = shape_inference.infer_shapes(pose)
        except Exception as e:
            print(f"[warn] pose shape inference failed: {e}")
        try:
            fused = shape_inference.infer_shapes(fused)
        except Exception as e:
            print(f"[warn] fused shape inference failed: {e}")

    pose_heat, pose_paf, heat_range, paf_range = _guess_pose_outputs(
        pose,
        heatmaps_name=pose_heatmaps_output,
        pafs_name=pose_pafs_output,
        combined_name=pose_combined_output,
    )
    fused_heat, fused_paf = _find_fused_inputs(fused, fused_heatmaps_input, fused_pafs_input)

    pose_shape_info = _shape_map(pose)
    fused_shape_info = _shape_map(fused)

    print("[merge] pose heatmaps/combined:", pose_heat, pose_shape_info.get(pose_heat))
    print("[merge] pose pafs:             ", pose_paf, pose_shape_info.get(pose_paf) if pose_paf else None)
    print("[merge] fused inputs:          ", fused_heat, fused_paf)

    # Start from pose graph.
    merged = copy.deepcopy(pose)
    graph = merged.graph

    # Remove original pose outputs; final graph outputs are fused-pruned outputs.
    del graph.output[:]

    sample_output_maps: List[Dict[str, str]] = []

    for i in range(batch_size):
        sample_prefix = f"merged/sample{i}"

        if pose_paf is None:
            # Combined output, e.g. [B,57,H,W].
            sample_combined = f"{sample_prefix}/combined_b1"
            _make_slice_node(
                graph,
                pose_heat,
                sample_combined,
                starts=[i],
                ends=[i + 1],
                axes=[0],
                node_name=f"{sample_prefix}/slice_batch_combined",
            )
            raw_heat = f"{sample_prefix}/heatmaps_raw"
            raw_paf = f"{sample_prefix}/pafs_raw"
            hs, he = heat_range or (0, 19)
            ps, pe = paf_range or (19, 57)
            _make_slice_node(
                graph,
                sample_combined,
                raw_heat,
                starts=[hs],
                ends=[he],
                axes=[1],
                node_name=f"{sample_prefix}/slice_combined_heatmaps",
            )
            _make_slice_node(
                graph,
                sample_combined,
                raw_paf,
                starts=[ps],
                ends=[pe],
                axes=[1],
                node_name=f"{sample_prefix}/slice_combined_pafs",
            )
        else:
            raw_heat = f"{sample_prefix}/heatmaps_raw_b1"
            raw_paf = f"{sample_prefix}/pafs_raw_b1"
            _make_slice_node(
                graph,
                pose_heat,
                raw_heat,
                starts=[i],
                ends=[i + 1],
                axes=[0],
                node_name=f"{sample_prefix}/slice_batch_heatmaps",
            )
            _make_slice_node(
                graph,
                pose_paf,
                raw_paf,
                starts=[i],
                ends=[i + 1],
                axes=[0],
                node_name=f"{sample_prefix}/slice_batch_pafs",
            )

        # Adapt heatmap channels to fused input, usually 19 -> 18.
        heat_shape = pose_shape_info.get(pose_heat, (None, []))[1]
        heat_c = None
        if len(heat_shape) >= 2 and isinstance(heat_shape[1], int):
            heat_c = int(heat_shape[1])
        if heat_c is not None and heat_c > fused_heatmap_channels:
            adapted_heat = f"{sample_prefix}/heatmaps_c{fused_heatmap_channels}"
            _make_slice_node(
                graph,
                raw_heat,
                adapted_heat,
                starts=[0],
                ends=[fused_heatmap_channels],
                axes=[1],
                node_name=f"{sample_prefix}/slice_heatmap_channels",
            )
        else:
            adapted_heat = raw_heat

        # Append prefixed fused-pruned graph for this sample.
        out_map = _append_prefixed_fused_graph(
            graph,
            fused,
            prefix=f"fused_s{i}/",
            fused_heatmaps_input=fused_heat,
            fused_pafs_input=fused_paf,
            heatmaps_source=adapted_heat,
            pafs_source=raw_paf,
        )
        sample_output_maps.append(out_map)

    final_outputs = _add_batched_outputs(
        graph,
        sample_output_maps,
        fused_shape_info,
        batch_size=batch_size,
        output_prefix="merged_outputs",
    )

    # Harmonize opset imports: take max version per domain.
    opsets: Dict[str, int] = {}
    for model in (pose, fused):
        for o in model.opset_import:
            domain = o.domain or ""
            opsets[domain] = max(opsets.get(domain, 0), int(o.version))
    del merged.opset_import[:]
    for domain, version in sorted(opsets.items()):
        merged.opset_import.append(helper.make_operatorsetid(domain, version))

    # Keep producer metadata.
    merged.producer_name = "merge_pose_with_fused_pruned_onnx.py"
    merged.producer_version = "1"

    print("[merge] checking merged model...")
    onnx.checker.check_model(merged)

    if use_external_data:
        data_name = output_onnx.name + ".data"
        onnx.save_model(
            merged,
            str(output_onnx),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(merged, str(output_onnx))

    debug = {
        "pose_onnx": str(pose_onnx),
        "fused_pruned_onnx": str(fused_pruned_onnx),
        "output_onnx": str(output_onnx),
        "batch_size": batch_size,
        "pose_heatmaps_or_combined_output": pose_heat,
        "pose_pafs_output": pose_paf,
        "combined_heat_range": heat_range,
        "combined_paf_range": paf_range,
        "fused_heatmaps_input": fused_heat,
        "fused_pafs_input": fused_paf,
        "final_outputs": final_outputs,
        "pose_outputs": [(o.name, pose_shape_info.get(o.name)) for o in pose.graph.output],
        "fused_outputs": [(o.name, fused_shape_info.get(o.name)) for o in fused.graph.output],
    }
    output_onnx.with_suffix(".merge_debug.json").write_text(json.dumps(debug, indent=2))
    print(f"[merge] saved: {output_onnx}")
    print(f"[merge] debug: {output_onnx.with_suffix('.merge_debug.json')}")
    return output_onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-onnx", required=True)
    ap.add_argument("--fused-pruned-onnx", required=True)
    ap.add_argument("--output-onnx", required=True)
    ap.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8], required=True)

    ap.add_argument("--pose-heatmaps-output", default="")
    ap.add_argument("--pose-pafs-output", default="")
    ap.add_argument("--pose-combined-output", default="")
    ap.add_argument("--fused-heatmaps-input", default="")
    ap.add_argument("--fused-pafs-input", default="")
    ap.add_argument("--fused-heatmap-channels", type=int, default=18)
    ap.add_argument("--no-shape-infer", action="store_true")
    ap.add_argument("--use-external-data", action="store_true")
    args = ap.parse_args()

    merge_pose_with_fused_pruned(
        pose_onnx=args.pose_onnx,
        fused_pruned_onnx=args.fused_pruned_onnx,
        output_onnx=args.output_onnx,
        batch_size=args.batch_size,
        pose_heatmaps_output=args.pose_heatmaps_output,
        pose_pafs_output=args.pose_pafs_output,
        pose_combined_output=args.pose_combined_output,
        fused_heatmaps_input=args.fused_heatmaps_input,
        fused_pafs_input=args.fused_pafs_input,
        fused_heatmap_channels=args.fused_heatmap_channels,
        infer_shapes=not args.no_shape_infer,
        use_external_data=args.use_external_data,
    )


if __name__ == "__main__":
    main()
