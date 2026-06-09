#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import onnx


def load_tool_module(path: str | Path, module_name: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inspect_video_hw(video: str | Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from video: {video}")
    h, w = frame.shape[:2]
    return int(h), int(w)


def count_nodes(path: str | Path) -> dict:
    model = onnx.load(str(path))
    hist = {}
    for node in model.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    return {
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "op_histogram": dict(sorted(hist.items())),
    }


def optimize_onnx_static(
    input_onnx: str | Path,
    output_onnx: str | Path,
    *,
    simplify: bool = True,
    infer_shapes: bool = True,
) -> Path:
    input_onnx = Path(input_onnx)
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[onnx-opt] load: {input_onnx}")
    model = onnx.load(str(input_onnx))
    onnx.checker.check_model(model)

    before = len(model.graph.node)

    if infer_shapes:
        try:
            print("[onnx-opt] shape inference")
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            print(f"[onnx-opt] shape inference skipped: {exc}")

    try:
        import onnxoptimizer

        passes = [
            "eliminate_identity",
            "eliminate_deadend",
            "eliminate_nop_dropout",
            "eliminate_nop_flatten",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "eliminate_nop_transpose",
            "eliminate_unused_initializer",
            "extract_constant_to_initializer",
            "fuse_consecutive_concats",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
        ]
        print("[onnx-opt] onnxoptimizer")
        model = onnxoptimizer.optimize(model, passes)
    except Exception as exc:
        print(f"[onnx-opt] onnxoptimizer skipped: {exc}")

    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            print("[onnx-opt] onnxsim simplify")
            simplified, ok = onnx_simplify(model)
            if ok:
                model = simplified
            else:
                print("[onnx-opt] onnxsim returned ok=False; keeping previous graph")
        except Exception as exc:
            print(f"[onnx-opt] onnxsim skipped: {exc}")

    onnx.checker.check_model(model)
    after = len(model.graph.node)

    onnx.save(model, str(output_onnx))
    print(f"[onnx-opt] saved: {output_onnx}")
    print(f"[onnx-opt] nodes: {before} -> {after}")
    return output_onnx


def compile_migraphx(
    onnx_path: str | Path,
    mxr_path: str | Path,
    *,
    batch_size: int,
    input_name: str,
    target_height: int,
    target_width: int,
    exhaustive_tune: bool,
    quantize_fp16: bool,
) -> Path:
    import migraphx

    onnx_path = Path(onnx_path)
    mxr_path = Path(mxr_path)
    mxr_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape = [int(batch_size), 3, int(target_height), int(target_width)]
    print(f"[migraphx] parse: {onnx_path}")
    print(f"[migraphx] map_input_dims: {input_name} -> {input_shape}")

    program = migraphx.parse_onnx(
        str(onnx_path),
        map_input_dims={input_name: input_shape},
    )

    if quantize_fp16:
        print("[migraphx] quantize_fp16")
        migraphx.quantize_fp16(program)

    print(f"[migraphx] compile gpu exhaustive_tune={bool(exhaustive_tune)}")
    t0 = time.time()
    program.compile(
        migraphx.get_target("gpu"),
        exhaustive_tune=bool(exhaustive_tune),
    )

    migraphx.save(program, str(mxr_path))
    print(f"[migraphx] saved: {mxr_path}")
    print(f"[migraphx] compile_elapsed_s: {time.time() - t0:.2f}")
    return mxr_path


def main():
    p = argparse.ArgumentParser(
        description="Generate optimized pose+fused-pruned batch-aware ONNX and compile MXR."
    )

    p.add_argument("--pose-onnx", default="models/fp16_refinment1.onnx")
    p.add_argument("--video", default="cctv_1280x720_24fps_3.mp4")
    p.add_argument("--out-dir", default="models/merged_pose_fused_pruned_batchaware_onnxopt")
    p.add_argument("--name-suffix", default="onnxopt")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--target-width", type=int, default=968)
    p.add_argument("--target-height", type=int, default=544)
    p.add_argument("--stride", type=int, default=8)

    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limb-topm", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--heatmap-cubic-a", type=float, default=-0.75)
    p.add_argument("--paf-cubic-a", type=float, default=-0.75)
    p.add_argument("--points-per-limb", type=int, default=8)
    p.add_argument("--min-paf-score", type=float, default=0.05)
    p.add_argument("--success-ratio-thr", type=float, default=0.8)
    p.add_argument("--min-pair-score", type=float, default=0.0)

    p.add_argument("--input-name", default="input")
    p.add_argument("--pose-heatmaps-output", default="stage_heatmaps")
    p.add_argument("--pose-pafs-output", default="stage_pafs")

    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--quantize-fp16", action="store_true")
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()

    repo_root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_h, full_w = inspect_video_hw(args.video)
    in_h = int(args.target_height) // int(args.stride)
    in_w = int(args.target_width) // int(args.stride)

    print(f"[shape] low-res: {in_h}x{in_w}")
    print(f"[shape] full-res: {full_h}x{full_w}")
    print(f"[shape] batch: {args.batch_size}")

    from modules.migraphx_fused_postprocess_pruned_compiler import (
        compile_pruned_fused_postprocess_head,
    )

    post_dir = out_dir / "post_pruned"
    parts_dir = post_dir / "_parts"

    print("[step 1] generate fused-pruned postprocess ONNX/MXR")
    post_mxr = compile_pruned_fused_postprocess_head(
        in_h=in_h,
        in_w=in_w,
        full_h=full_h,
        full_w=full_w,
        output_dir=post_dir,
        parts_dir=parts_dir,
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
        batch_size=args.batch_size,
        exhaustive_tune=False,
        force=args.force,
        keep_onnx=True,
    )

    post_onnx = Path(post_mxr).with_suffix(".onnx")
    if not post_onnx.exists():
        raise FileNotFoundError(f"Expected postprocess ONNX not found: {post_onnx}")

    print("[step 2] optimize fused-pruned postprocess ONNX")
    post_onnx_opt = post_onnx.with_name(post_onnx.stem + "_optimized.onnx")
    optimize_onnx_static(
        post_onnx,
        post_onnx_opt,
        simplify=not args.no_simplify,
    )

    merge_tool = load_tool_module(
        repo_root / "tools" / "compile_merged_pose_batchaware_fused_pruned.py",
        "compile_merged_pose_batchaware_fused_pruned_local",
    )

    base_name = (
        f"pose_fused_pruned_batchaware_b{args.batch_size}_"
        f"{full_h}x{full_w}_"
        f"k{args.topk}_m{args.limb_topm}_p{args.points_per_limb}_"
        f"thr{str(args.threshold).replace('.', 'p')}_"
        f"r{args.nms_radius}_{args.nms_impl}_{args.name_suffix}"
    )

    merged_raw_onnx = out_dir / f"{base_name}_raw.onnx"
    merged_opt_onnx = out_dir / f"{base_name}.onnx"
    merged_mxr = out_dir / f"{base_name}.mxr"

    print("[step 3] merge pose ONNX + optimized postprocess ONNX")
    merge_tool.merge_pose_post(
        pose_onnx=args.pose_onnx,
        post_onnx=post_onnx_opt,
        output_onnx=merged_raw_onnx,
        batch_size=args.batch_size,
        input_name=args.input_name,
        pose_heatmaps_output=args.pose_heatmaps_output,
        pose_pafs_output=args.pose_pafs_output,
    )

    print("[step 4] optimize merged pose+postprocess ONNX")
    optimize_onnx_static(
        merged_raw_onnx,
        merged_opt_onnx,
        simplify=not args.no_simplify,
    )

    print("[step 5] compile optimized merged ONNX -> MXR")
    compile_migraphx(
        merged_opt_onnx,
        merged_mxr,
        batch_size=args.batch_size,
        input_name=args.input_name,
        target_height=args.target_height,
        target_width=args.target_width,
        exhaustive_tune=args.exhaustive_tune,
        quantize_fp16=args.quantize_fp16,
    )

    meta = {
        "pose_onnx": str(args.pose_onnx),
        "video": str(args.video),
        "full_hw": [full_h, full_w],
        "lowres_hw": [in_h, in_w],
        "batch_size": args.batch_size,
        "topk": args.topk,
        "limb_topm": args.limb_topm,
        "points_per_limb": args.points_per_limb,
        "threshold": args.threshold,
        "nms_radius": args.nms_radius,
        "nms_impl": args.nms_impl,
        "heatmap_cubic_a": args.heatmap_cubic_a,
        "paf_cubic_a": args.paf_cubic_a,
        "min_paf_score": args.min_paf_score,
        "success_ratio_thr": args.success_ratio_thr,
        "min_pair_score": args.min_pair_score,
        "post_onnx": str(post_onnx),
        "post_onnx_optimized": str(post_onnx_opt),
        "merged_raw_onnx": str(merged_raw_onnx),
        "merged_onnx": str(merged_opt_onnx),
        "merged_mxr": str(merged_mxr),
        "post_onnx_stats": count_nodes(post_onnx),
        "post_onnx_optimized_stats": count_nodes(post_onnx_opt),
        "merged_raw_onnx_stats": count_nodes(merged_raw_onnx),
        "merged_onnx_stats": count_nodes(merged_opt_onnx),
        "exhaustive_tune": bool(args.exhaustive_tune),
        "quantize_fp16": bool(args.quantize_fp16),
    }

    meta_path = merged_mxr.with_suffix(".build.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDONE")
    print(f"merged ONNX: {merged_opt_onnx}")
    print(f"merged MXR:  {merged_mxr}")
    print(f"metadata:    {meta_path}")


if __name__ == "__main__":
    main()
