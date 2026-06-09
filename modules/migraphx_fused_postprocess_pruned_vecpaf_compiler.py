#!/usr/bin/env python3
"""Compile fused-pruned postprocess head with vectorized two-channel PAF sampling.

This is an accuracy-safe experimental variant of the regular
migraphx_fused_postprocess_pruned_compiler.py path. It keeps the same scoring,
TopK/TopM, thresholds, cubic interpolation, and output contract, but it uses the
PAF scorer's vectorized sampling path so PAF X/Y channels are sampled together.

The generated cache name includes `_vecpaf` to avoid mixing it with the baseline
fused-pruned head.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


def _safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def _supports_kw(func: Callable[..., Any], kw: str) -> bool:
    try:
        return kw in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    filtered = {k: v for k, v in kwargs.items() if _supports_kw(func, k)}
    return func(*args, **filtered)


def _compile_onnx_to_mxr(onnx_path: str | Path, mxr_path: str | Path, *, exhaustive_tune: bool = False) -> None:
    onnx_path = Path(onnx_path)
    mxr_path = Path(mxr_path)
    mxr_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import migraphx  # type: ignore
    except ModuleNotFoundError:
        migraphx = None

    if migraphx is not None and hasattr(migraphx, "parse_onnx"):
        program = migraphx.parse_onnx(str(onnx_path))
        program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
        migraphx.save(program, str(mxr_path))
        return

    driver = shutil.which("migraphx-driver") or "/opt/rocm/bin/migraphx-driver"
    if not Path(driver).exists() and shutil.which(driver) is None:
        raise RuntimeError(
            "Python migraphx is unavailable and migraphx-driver was not found. "
            "Run `source rocm721/activate_rocm721.sh` or check ROCm/MIGraphX install."
        )

    if exhaustive_tune:
        print("[warning] exhaustive_tune is ignored by migraphx-driver fallback")
    cmd = [driver, "compile", str(onnx_path), "--onnx", "--gpu", "--binary", "-o", str(mxr_path)]
    print("[migraphx-fallback] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def vecpaf_pruned_head_name(
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
    batch_size: int = 1,
) -> str:
    batch_token = "" if int(batch_size) == 1 else f"_b{int(batch_size)}"
    return (
        "fused_cubic_topk_fullres_paf_pruned_vecpaf_"
        f"{int(in_h)}x{int(in_w)}_to_{int(full_h)}x{int(full_w)}{batch_token}_"
        f"k{int(topk)}_m{int(limb_topm)}_thr{_safe_float_token(threshold)}_"
        f"r{int(nms_radius)}_{nms_impl}_"
        f"ha{_safe_float_token(heatmap_cubic_a)}_"
        f"p{int(points_per_limb)}_min{_safe_float_token(min_paf_score)}_"
        f"sr{_safe_float_token(success_ratio_thr)}_pa{_safe_float_token(paf_cubic_a)}_"
        f"mp{_safe_float_token(min_pair_score)}"
    )


def compile_pruned_fused_postprocess_vecpaf_head(
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
    batch_size: int = 1,
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
        _run_export_subprocess as export_paf_fullres_pair_scorer_onnx,
        onnx_path as paf_scorer_onnx_path,
    )
    from modules.migraphx_fused_postprocess_compiler import build_fused_postprocess_onnx, fused_head_name
    from modules.migraphx_fused_postprocess_pruned_compiler import append_pruning_tail

    batch_size = int(batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(parts_dir) if parts_dir else output_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    name = vecpaf_pruned_head_name(
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
        batch_size=batch_size,
    )
    final_onnx = output_dir / f"{name}.onnx"
    final_mxr = output_dir / f"{name}.mxr"
    if final_mxr.exists() and not force:
        print(f"[fused-pruned-vecpaf] exists, skipping: {final_mxr}")
        return final_mxr

    print(f"[fused-pruned-vecpaf] compiling/checking manual TopK component B={batch_size}")
    _call_with_supported_kwargs(
        compile_manual_cubic_nms_topk_head,
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
        batch_size=batch_size,
    )

    manual_base_name = _call_with_supported_kwargs(
        manual_head_name,
        in_h,
        in_w,
        full_h,
        full_w,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=heatmap_cubic_a,
        batch_size=batch_size,
    )
    manual_onnx = parts_dir / f"{manual_base_name}.onnx"
    if not manual_onnx.exists():
        raise FileNotFoundError(f"Manual TopK ONNX not found: {manual_onnx}")

    paf_onnx = paf_scorer_onnx_path(
        parts_dir,
        in_h,
        in_w,
        full_h,
        full_w,
        batch_size=batch_size,
        topk=topk,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        cubic_a=paf_cubic_a,
        vectorized_sampling=True,
    )
    print(f"[fused-pruned-vecpaf] exporting vectorized PAF scorer ONNX: {paf_onnx.name}")
    export_paf_fullres_pair_scorer_onnx(
        output_onnx=paf_onnx,
        paf_h=in_h,
        paf_w=in_w,
        full_h=full_h,
        full_w=full_w,
        topk=topk,
        batch_size=batch_size,
        points_per_limb=points_per_limb,
        min_paf_score=min_paf_score,
        success_ratio_thr=success_ratio_thr,
        cubic_a=paf_cubic_a,
        vectorized_sampling=True,
        opset=opset,
    )

    fused_base_name = fused_head_name(
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
        batch_size=batch_size,
    )
    # Keep base fused ONNX separate from regular fused cache by adding vecpaf suffix.
    fused_onnx = parts_dir / f"{fused_base_name}_vecpaf.onnx"

    print("[fused-pruned-vecpaf] merging manual TopK + vectorized PAF scorer ONNX graphs")
    _, info = build_fused_postprocess_onnx(
        manual_onnx=manual_onnx,
        paf_onnx=paf_onnx,
        fused_onnx=fused_onnx,
    )
    print("[fused-pruned-vecpaf] fused inputs:", info["fused_inputs"])
    print("[fused-pruned-vecpaf] fused outputs:", info["fused_outputs"])

    print(f"[fused-pruned-vecpaf] appending TopM pruning tail: B={batch_size}, K={topk}, M={limb_topm}")
    append_pruning_tail(
        fused_onnx,
        final_onnx,
        topk=topk,
        limb_topm=limb_topm,
        min_pair_score=min_pair_score,
        batch_size=batch_size,
    )

    print(f"[fused-pruned-vecpaf] compiling MIGraphX GPU target: {final_onnx.name} -> {final_mxr.name}")
    _compile_onnx_to_mxr(final_onnx, final_mxr, exhaustive_tune=bool(exhaustive_tune))

    if not keep_onnx:
        for path in (paf_onnx, fused_onnx, final_onnx):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    print(f"[fused-pruned-vecpaf] saved: {final_mxr}")
    return final_mxr


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
    print(f"[fused-pruned-vecpaf] video full-res shape: {full_h}x{full_w}; low-res shape: {in_h}x{in_w}")
    return compile_pruned_fused_postprocess_vecpaf_head(in_h=in_h, in_w=in_w, full_h=full_h, full_w=full_w, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile fused-pruned postprocess head with vectorized PAF sampling.")
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
    p.add_argument("--batch-size", type=int, default=1)
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
        batch_size=args.batch_size,
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
        compile_pruned_fused_postprocess_vecpaf_head(in_h=in_h, in_w=in_w, full_h=full_h, full_w=full_w, **common)


if __name__ == "__main__":
    main()
