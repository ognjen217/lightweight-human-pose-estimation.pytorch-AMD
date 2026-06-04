#!/usr/bin/env python3
"""
Compile fused_postprocess_v2 / pruned pair head.

Base fused head outputs:
  pair_scores [*,19,K,K]
  pair_valid  [*,19,K,K]
  top_scores  [*,18,K]
  top_indices [*,18,K]

This compiler appends an ONNX TopM pruning tail:

  pair_scores/pair_valid -> per-limb TopM pairs

New outputs:
  top_scores
  top_indices
  limb_top_pair_a_idx   [19,M]
  limb_top_pair_b_idx   [19,M]
  limb_top_pair_score   [19,M]
  limb_top_pair_valid   [19,M]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def pruned_head_name(
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    *,
    topk: int,
    limb_topm: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    heatmap_cubic_a: float,
    points_per_limb: int,
    min_paf_score: float,
    success_ratio_thr: float,
    paf_cubic_a: float,
    min_pair_score: float = 0.0,
) -> str:
    return (
        "fused_cubic_topk_fullres_paf_pruned_"
        f"{int(in_h)}x{int(in_w)}_to_{int(full_h)}x{int(full_w)}_"
        f"k{int(topk)}_m{int(limb_topm)}_thr{_safe_float_token(threshold)}_"
        f"r{int(nms_radius)}_{nms_impl}_"
        f"ha{_safe_float_token(heatmap_cubic_a)}_"
        f"p{int(points_per_limb)}_min{_safe_float_token(min_paf_score)}_"
        f"sr{_safe_float_token(success_ratio_thr)}_pa{_safe_float_token(paf_cubic_a)}_"
        f"mp{_safe_float_token(min_pair_score)}"
    )


def _find_output_name(model, token: str) -> str:
    for out in model.graph.output:
        if token in out.name:
            return out.name
    raise RuntimeError(f"Could not find output containing token {token!r}. Outputs: {[o.name for o in model.graph.output]}")


def _replace_outputs(model, output_names_and_types):
    from onnx import helper

    vi = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        vi[value.name] = value

    del model.graph.output[:]
    for name, dtype, shape in output_names_and_types:
        if name in vi and shape is None:
            model.graph.output.append(vi[name])
        else:
            model.graph.output.append(helper.make_tensor_value_info(name, dtype, shape))


def append_pruning_tail(
    fused_onnx: str | Path,
    pruned_onnx: str | Path,
    *,
    topk: int = 20,
    limb_topm: int = 20,
    num_limbs: int = 19,
    min_pair_score: float = 0.0,
    keep_pair_debug_outputs: bool = False,
) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    fused_onnx = Path(fused_onnx)
    pruned_onnx = Path(pruned_onnx)
    model = onnx.load(str(fused_onnx))
    onnx.checker.check_model(model)

    pair_scores = _find_output_name(model, "pair_scores")
    pair_valid = _find_output_name(model, "pair_valid")
    top_scores = _find_output_name(model, "top_scores")
    top_indices = _find_output_name(model, "top_indices")

    def const_tensor(name, arr):
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(arr), name=name))

    flat_dim = int(topk) * int(topk)
    const_tensor("prune_shape_lkk", np.asarray([int(num_limbs), flat_dim], dtype=np.int64))
    const_tensor("prune_topm_const", np.asarray([int(limb_topm)], dtype=np.int64))
    const_tensor("prune_k_float", np.asarray(float(topk), dtype=np.float32))
    const_tensor("prune_neg_inf", np.asarray(-1.0e9, dtype=np.float32))
    const_tensor("prune_zero", np.asarray(0.0, dtype=np.float32))
    const_tensor("prune_min_pair_score", np.asarray(float(min_pair_score), dtype=np.float32))

    nodes = [
        helper.make_node("Reshape", [pair_scores, "prune_shape_lkk"], ["prune_pair_scores_flat"], name="prune/reshape_scores"),
        helper.make_node("Reshape", [pair_valid, "prune_shape_lkk"], ["prune_pair_valid_flat_raw"], name="prune/reshape_valid"),
        helper.make_node("Cast", ["prune_pair_valid_flat_raw"], ["prune_pair_valid_flat"], name="prune/cast_valid_float", to=TensorProto.FLOAT),
        helper.make_node("Greater", ["prune_pair_valid_flat", "prune_zero"], ["prune_pair_valid_bool"], name="prune/valid_gt_zero"),
        helper.make_node("Where", ["prune_pair_valid_bool", "prune_pair_scores_flat", "prune_neg_inf"], ["prune_masked_scores"], name="prune/mask_scores"),
        helper.make_node(
            "TopK",
            ["prune_masked_scores", "prune_topm_const"],
            ["limb_top_pair_score", "limb_top_pair_flat_idx"],
            name="prune/topm_pairs",
            axis=1,
            largest=1,
            sorted=1,
        ),
        helper.make_node("Greater", ["limb_top_pair_score", "prune_min_pair_score"], ["limb_top_pair_valid_bool"], name="prune/topm_valid"),
        helper.make_node("Cast", ["limb_top_pair_valid_bool"], ["limb_top_pair_valid"], name="prune/topm_valid_float", to=TensorProto.FLOAT),
        helper.make_node("Cast", ["limb_top_pair_flat_idx"], ["prune_flat_idx_float"], name="prune/cast_flat_float", to=TensorProto.FLOAT),
        helper.make_node("Div", ["prune_flat_idx_float", "prune_k_float"], ["prune_a_float_raw"], name="prune/a_div"),
        helper.make_node("Floor", ["prune_a_float_raw"], ["prune_a_float"], name="prune/a_floor"),
        helper.make_node("Mul", ["prune_a_float", "prune_k_float"], ["prune_a_times_k"], name="prune/a_mul_k"),
        helper.make_node("Sub", ["prune_flat_idx_float", "prune_a_times_k"], ["prune_b_float"], name="prune/b_sub"),
        helper.make_node("Cast", ["prune_a_float"], ["limb_top_pair_a_idx"], name="prune/cast_a_i64", to=TensorProto.INT64),
        helper.make_node("Cast", ["prune_b_float"], ["limb_top_pair_b_idx"], name="prune/cast_b_i64", to=TensorProto.INT64),
    ]
    model.graph.node.extend(nodes)

    outputs = [
        (top_scores, TensorProto.FLOAT, None),
        (top_indices, TensorProto.INT64, None),
        ("limb_top_pair_a_idx", TensorProto.INT64, [int(num_limbs), int(limb_topm)]),
        ("limb_top_pair_b_idx", TensorProto.INT64, [int(num_limbs), int(limb_topm)]),
        ("limb_top_pair_score", TensorProto.FLOAT, [int(num_limbs), int(limb_topm)]),
        ("limb_top_pair_valid", TensorProto.FLOAT, [int(num_limbs), int(limb_topm)]),
    ]
    if keep_pair_debug_outputs:
        outputs = [(pair_scores, TensorProto.FLOAT, None), (pair_valid, TensorProto.FLOAT, None)] + outputs

    _replace_outputs(model, outputs)
    pruned_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(pruned_onnx))
    pruned_onnx.with_suffix(".debug.json").write_text(
        json.dumps(
            {
                "base_fused_onnx": str(fused_onnx),
                "pair_scores": pair_scores,
                "pair_valid": pair_valid,
                "top_scores": top_scores,
                "top_indices": top_indices,
                "limb_topm": limb_topm,
                "topk": topk,
                "outputs": [x[0] for x in outputs],
            },
            indent=2,
        )
    )
    return pruned_onnx


def compile_pruned_fused_postprocess_head(
    *,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    output_dir: str | Path = "models/fused_postprocess_pruned_cache",
    parts_dir: str | Path = "",
    topk: int = 20,
    limb_topm: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 6,
    nms_impl: str = "separable",
    heatmap_cubic_a: float = -0.75,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
    paf_cubic_a: float = -0.75,
    min_pair_score: float = 0.0,
    opset: int = 18,
    exhaustive_tune: bool = False,
    force: bool = False,
    keep_onnx: bool = True,
) -> Path:
    import migraphx  # type: ignore
    from modules.migraphx_fused_postprocess_compiler import compile_fused_postprocess_head, fused_head_name

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(parts_dir) if parts_dir else output_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    name = pruned_head_name(
        in_h,
        in_w,
        full_h,
        full_w,
        topk=topk,
        limb_topm=limb_topm,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        heatmap_cubic_a=heatmap_cubic_a,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        paf_cubic_a=paf_cubic_a,
        min_pair_score=min_pair_score,
    )
    mxr_path = output_dir / f"{name}.mxr"
    onnx_path = output_dir / f"{name}.onnx"
    if mxr_path.exists() and not force:
        print(f"[fused-pruned] exists, skipping: {mxr_path}")
        return mxr_path

    print("[fused-pruned] compiling/checking base fused postprocess ONNX")
    compile_fused_postprocess_head(
        in_h=in_h,
        in_w=in_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=parts_dir,
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
        exhaustive_tune=False,
        force=force,
        keep_onnx=True,
    )

    base_name = fused_head_name(
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
    base_onnx = parts_dir / f"{base_name}.onnx"
    if not base_onnx.exists():
        raise FileNotFoundError(f"Expected base fused ONNX not found: {base_onnx}")

    print(f"[fused-pruned] appending TopM pruning tail: K={topk}, M={limb_topm}")
    append_pruning_tail(base_onnx, onnx_path, topk=topk, limb_topm=limb_topm, min_pair_score=min_pair_score)

    print(f"[fused-pruned] compiling MIGraphX GPU target: {onnx_path.name} -> {mxr_path.name}")
    program = migraphx.parse_onnx(str(onnx_path))
    program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
    migraphx.save(program, str(mxr_path))

    if not keep_onnx:
        try:
            onnx_path.unlink()
        except FileNotFoundError:
            pass
    print(f"[fused-pruned] saved: {mxr_path}")
    return mxr_path


def compile_for_video(**kwargs) -> Path:
    import cv2

    video = kwargs.pop("video")
    target_width = int(kwargs.pop("target_width", 968))
    target_height = int(kwargs.pop("target_height", 544))
    stride = int(kwargs.pop("stride", 8))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {video}")

    full_h, full_w = frame.shape[:2]
    in_h = target_height // stride
    in_w = target_width // stride
    print(f"[fused-pruned] video full-res shape: {full_h}x{full_w}; low-res shape: {in_h}x{in_w}")
    return compile_pruned_fused_postprocess_head(in_h=in_h, in_w=in_w, full_h=full_h, full_w=full_w, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile fused postprocess v2 with per-limb TopM pruning.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--shape", nargs=4, type=int, metavar=("IN_H", "IN_W", "FULL_H", "FULL_W"))
    p.add_argument("--output-dir", default="models/fused_postprocess_pruned_cache")
    p.add_argument("--parts-dir", default="")
    p.add_argument("--target-width", type=int, default=968)
    p.add_argument("--target-height", type=int, default=544)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limb-topm", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--min-pair-score", type=float, default=0.0)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common = dict(
        output_dir=args.output_dir,
        parts_dir=args.parts_dir,
        topk=args.topk,
        limb_topm=args.limb_topm,
        threshold=args.threshold,
        nms_radius=args.nms_radius,
        nms_impl=args.nms_impl,
        heatmap_cubic_a=args.heatmap_cubic_a,
        points_per_limb=args.points_per_limb,
        min_paf_score=args.min_paf_score,
        success_ratio_thr=args.success_ratio_thr,
        paf_cubic_a=args.paf_cubic_a,
        min_pair_score=args.min_pair_score,
        opset=args.opset,
        exhaustive_tune=args.exhaustive_tune,
        force=args.force,
        keep_onnx=args.keep_onnx,
    )
    if args.video:
        compile_for_video(video=args.video, target_width=args.target_width, target_height=args.target_height, stride=args.stride, **common)
    else:
        in_h, in_w, full_h, full_w = args.shape
        compile_pruned_fused_postprocess_head(in_h=in_h, in_w=in_w, full_h=full_h, full_w=full_w, **common)


if __name__ == "__main__":
    main()
