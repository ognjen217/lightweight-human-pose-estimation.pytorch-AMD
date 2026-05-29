#!/usr/bin/env python3
"""
Compile a fused postprocess MIGraphX head:

    heatmaps [1,18,Hm,Wm]
    pafs     [1,38,Hm,Wm]
        ↓
    manual cubic heatmap resize + NMS + TopK
        ↓
    full-res-like cubic PAF pair scoring
        ↓
    outputs:
      pair_scores [1,19,K,K]
      pair_valid  [1,19,K,K]
      top_scores  [1,18,K]
      top_indices [1,18,K]

Why this exists
---------------
The previous Python runtime executed two separate postprocess MXR programs:

    manual_cubic_nms_topk.mxr
    paf_fullres_pair_scorer.mxr

which caused:
    GPU -> CPU top_scores/top_indices
    CPU -> GPU top_scores/top_indices + pafs

This fused compiler merges the two exported ONNX graphs into one ONNX graph and
then compiles one .mxr.  The handoff from TopK to PAF scoring then happens inside
one MIGraphX program, i.e. without a Python/CPU round-trip between these two
postprocess GPU stages.

Important limitation
--------------------
This does NOT yet fuse pose_model.mxr with the postprocess head.  Standard
MIGraphX Python `program.run(np.ndarray)` still returns pose outputs to host.
To achieve full CPU -> GPU -> GPU -> GPU -> CPU, either:
  1) fuse the original pose ONNX model with this fused postprocess ONNX, or
  2) implement a C++/HIP device-buffer runtime that chains compiled programs.

This file implements the safest first step: fused postprocess head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def fused_head_name(
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    *,
    topk: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    heatmap_cubic_a: float,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
    paf_cubic_a: float,
) -> str:
    return (
        "fused_cubic_topk_fullres_paf_"
        f"{int(in_h)}x{int(in_w)}_to_{int(full_h)}x{int(full_w)}_"
        f"k{int(topk)}_thr{_safe_float_token(threshold)}_r{int(nms_radius)}_{nms_impl}_"
        f"ha{_safe_float_token(heatmap_cubic_a)}_"
        f"p{int(points_per_limb)}_min{_safe_float_token(min_paf_score)}_"
        f"sr{_safe_float_token(success_ratio_thr)}_pa{_safe_float_token(paf_cubic_a)}"
    )


def _load_onnx(path: Path):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def _save_onnx(model, path: Path) -> None:
    import onnx
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _graph_inputs(model) -> List[str]:
    initializer_names = {i.name for i in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def _graph_outputs(model) -> List[str]:
    return [o.name for o in model.graph.output]


def _replace_graph_outputs(model, output_names: List[str]) -> None:
    """Restrict graph outputs to exactly output_names, reusing value_info when possible."""
    import onnx
    from onnx import helper, TensorProto

    vi = {}
    for value in list(model.graph.output) + list(model.graph.value_info) + list(model.graph.input):
        vi[value.name] = value

    del model.graph.output[:]
    for name in output_names:
        if name in vi:
            model.graph.output.append(vi[name])
        else:
            # Fallback shape is unknown float tensor. MIGraphX usually infers it.
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))


def _rename_external_inputs(model, rename: dict) -> None:
    """Rename graph external inputs and all node references."""
    for graph_input in model.graph.input:
        if graph_input.name in rename:
            graph_input.name = rename[graph_input.name]
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]


def build_fused_postprocess_onnx(
    *,
    manual_onnx: Path,
    paf_onnx: Path,
    fused_onnx: Path,
    keep_debug_json: bool = True,
) -> Tuple[Path, dict]:
    """Merge manual TopK ONNX and PAF scorer ONNX into one ONNX model."""
    import onnx
    from onnx import compose

    m_topk = _load_onnx(manual_onnx)
    m_paf = _load_onnx(paf_onnx)

    topk_inputs = _graph_inputs(m_topk)
    topk_outputs = _graph_outputs(m_topk)
    paf_inputs = _graph_inputs(m_paf)
    paf_outputs = _graph_outputs(m_paf)

    if len(topk_inputs) != 1:
        raise RuntimeError(f"Expected manual TopK ONNX to have 1 external input, got {topk_inputs}")
    if len(topk_outputs) < 2:
        raise RuntimeError(f"Expected manual TopK ONNX to have at least 2 outputs, got {topk_outputs}")
    if len(paf_inputs) < 3:
        raise RuntimeError(f"Expected PAF scorer ONNX to have at least 3 external inputs, got {paf_inputs}")
    if len(paf_outputs) < 2:
        raise RuntimeError(f"Expected PAF scorer ONNX to have at least 2 outputs, got {paf_outputs}")

    # PAF scorer input names should be top_scores/top_indices/pafs, but keep this robust.
    paf_top_scores = next((x for x in paf_inputs if "top_scores" in x), paf_inputs[0])
    paf_top_indices = next((x for x in paf_inputs if "top_indices" in x), paf_inputs[1])
    paf_pafs = next((x for x in paf_inputs if x.endswith("pafs") or x == "pafs" or "paf" in x), paf_inputs[2])

    # Prefix both graphs to avoid name collisions.
    m_topk_p = compose.add_prefix(m_topk, "topk/")
    m_paf_p = compose.add_prefix(m_paf, "paf/")

    topk_input_p = "topk/" + topk_inputs[0]
    top_scores_p = "topk/" + topk_outputs[0]
    top_indices_p = "topk/" + topk_outputs[1]

    paf_top_scores_p = "paf/" + paf_top_scores
    paf_top_indices_p = "paf/" + paf_top_indices
    paf_pafs_p = "paf/" + paf_pafs
    paf_outputs_p = ["paf/" + x for x in paf_outputs[:2]]

    merged = compose.merge_models(
        m_topk_p,
        m_paf_p,
        io_map=[
            (top_scores_p, paf_top_scores_p),
            (top_indices_p, paf_top_indices_p),
        ],
    )

    # Rename external inputs to clean names:
    #   topk/heatmaps -> heatmaps
    #   paf/pafs      -> pafs
    _rename_external_inputs(
        merged,
        {
            topk_input_p: "heatmaps",
            paf_pafs_p: "pafs",
        },
    )

    # Keep final PAF scorer outputs and also expose TopK outputs for CPU adapter.
    final_outputs = [
        paf_outputs_p[0],    # pair_scores
        paf_outputs_p[1],    # pair_valid
        top_scores_p,        # top_scores
        top_indices_p,       # top_indices
    ]
    _replace_graph_outputs(merged, final_outputs)

    # Clean names for graph IO metadata can be nice, but avoid renaming node internals now.
    onnx.checker.check_model(merged)
    _save_onnx(merged, fused_onnx)

    info = {
        "manual_onnx": str(manual_onnx),
        "paf_onnx": str(paf_onnx),
        "fused_onnx": str(fused_onnx),
        "topk_inputs": topk_inputs,
        "topk_outputs": topk_outputs,
        "paf_inputs": paf_inputs,
        "paf_outputs": paf_outputs,
        "fused_inputs": _graph_inputs(merged),
        "fused_outputs": _graph_outputs(merged),
        "io_map": [
            [top_scores_p, paf_top_scores_p],
            [top_indices_p, paf_top_indices_p],
        ],
    }
    if keep_debug_json:
        fused_onnx.with_suffix(".debug.json").write_text(json.dumps(info, indent=2))
    return fused_onnx, info


def compile_fused_postprocess_head(
    *,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    output_dir: str | Path = "models/fused_postprocess_cache",
    parts_dir: str | Path = "",
    topk: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    nms_impl: str = "separable",
    heatmap_cubic_a: float = -0.75,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
    paf_cubic_a: float = -0.75,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = True,
) -> Path:
    from modules.migraphx_manual_cubic_topk_compiler import (
        compile_manual_cubic_nms_topk_head,
        head_name as manual_head_name,
    )
    from modules.migraphx_paf_fullres_pair_scorer_compiler import (
        compile_paf_fullres_pair_scorer_head,
        head_name as paf_head_name,
    )
    import migraphx  # type: ignore

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(parts_dir) if parts_dir else output_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    name = fused_head_name(
        in_h,
        in_w,
        full_h,
        full_w,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        heatmap_cubic_a=heatmap_cubic_a,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        paf_cubic_a=paf_cubic_a,
    )
    fused_onnx = output_dir / f"{name}.onnx"
    fused_mxr = output_dir / f"{name}.mxr"

    if fused_mxr.exists() and not force:
        print(f"[fused-post] exists, skipping: {fused_mxr}")
        return fused_mxr

    print("[fused-post] exporting component ONNX files")
    # These calls also compile component MXRs, but we primarily need the ONNX files.
    compile_manual_cubic_nms_topk_head(
        in_h=in_h,
        in_w=in_w,
        out_h=full_h,
        out_w=full_w,
        output_dir=parts_dir,
        channels=18,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=heatmap_cubic_a,
        opset=opset,
        exhaustive_tune=False,
        force=force,
        keep_onnx=True,
    )
    compile_paf_fullres_pair_scorer_head(
        paf_h=in_h,
        paf_w=in_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=parts_dir,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        cubic_a=paf_cubic_a,
        opset=opset,
        exhaustive_tune=False,
        force=force,
        keep_onnx=True,
    )

    manual_onnx = parts_dir / f"{manual_head_name(in_h, in_w, full_h, full_w, topk=topk, threshold=threshold, nms_radius=nms_radius, nms_impl=nms_impl, cubic_a=heatmap_cubic_a)}.onnx"
    paf_onnx = parts_dir / f"{paf_head_name(in_h, in_w, full_h, full_w, topk=topk, points_per_limb=points_per_limb, min_paf_score=min_paf_score, success_ratio_thr=success_ratio_thr, cubic_a=paf_cubic_a)}.onnx"

    if not manual_onnx.exists():
        raise FileNotFoundError(f"Manual TopK ONNX not found: {manual_onnx}")
    if not paf_onnx.exists():
        raise FileNotFoundError(f"PAF scorer ONNX not found: {paf_onnx}")

    print("[fused-post] merging ONNX graphs")
    _, info = build_fused_postprocess_onnx(
        manual_onnx=manual_onnx,
        paf_onnx=paf_onnx,
        fused_onnx=fused_onnx,
    )
    print("[fused-post] fused inputs:", info["fused_inputs"])
    print("[fused-post] fused outputs:", info["fused_outputs"])

    print(f"[fused-post] compiling MIGraphX GPU target: {fused_onnx.name} -> {fused_mxr.name}")
    program = migraphx.parse_onnx(str(fused_onnx))
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    migraphx.save(program, str(fused_mxr))

    if not keep_onnx:
        try:
            fused_onnx.unlink()
        except FileNotFoundError:
            pass

    print(f"[fused-post] saved: {fused_mxr}")
    return fused_mxr


def compile_for_video(
    *,
    video: str | Path,
    target_width: int = 968,
    target_height: int = 544,
    stride: int = 8,
    output_dir: str | Path = "models/fused_postprocess_cache",
    parts_dir: str | Path = "",
    topk: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    nms_impl: str = "separable",
    heatmap_cubic_a: float = -0.75,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
    paf_cubic_a: float = -0.75,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = True,
) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {video}")

    full_h, full_w = frame.shape[:2]
    in_h = int(target_height) // int(stride)
    in_w = int(target_width) // int(stride)

    print(f"[fused-post] video full-res shape: {full_h}x{full_w}; low-res shape: {in_h}x{in_w}")
    return compile_fused_postprocess_head(
        in_h=in_h,
        in_w=in_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=output_dir,
        parts_dir=parts_dir,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        heatmap_cubic_a=heatmap_cubic_a,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        paf_cubic_a=paf_cubic_a,
        opset=opset,
        exhaustive_tune=exhaustive_tune,
        force=force,
        keep_onnx=keep_onnx,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile fused manual TopK + fullres PAF scorer postprocess MXR.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--shape", nargs=4, type=int, metavar=("IN_H", "IN_W", "FULL_H", "FULL_W"))

    p.add_argument("--output-dir", default="models/fused_postprocess_cache")
    p.add_argument("--parts-dir", default="")
    p.add_argument("--target-width", type=int, default=968)
    p.add_argument("--target-height", type=int, default=544)
    p.add_argument("--stride", type=int, default=8)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)

    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.video:
        compile_for_video(
            video=args.video,
            target_width=args.target_width,
            target_height=args.target_height,
            stride=args.stride,
            output_dir=args.output_dir,
            parts_dir=args.parts_dir,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            heatmap_cubic_a=args.heatmap_cubic_a,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            paf_cubic_a=args.paf_cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )
    else:
        in_h, in_w, full_h, full_w = args.shape
        compile_fused_postprocess_head(
            in_h=in_h,
            in_w=in_w,
            full_h=full_h,
            full_w=full_w,
            output_dir=args.output_dir,
            parts_dir=args.parts_dir,
            topk=args.topk,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            nms_impl=args.nms_impl,
            heatmap_cubic_a=args.heatmap_cubic_a,
            points_per_limb=args.points_per_limb,
            min_paf_score=args.min_paf_score,
            success_ratio_thr=args.success_ratio_thr,
            paf_cubic_a=args.paf_cubic_a,
            opset=args.opset,
            exhaustive_tune=args.exhaustive_tune,
            force=args.force,
            keep_onnx=args.keep_onnx,
        )


if __name__ == "__main__":
    main()
