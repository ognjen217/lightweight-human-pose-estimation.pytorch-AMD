#!/usr/bin/env python3
"""Utilities for compiling MIGraphX postprocess heads on demand.

The validation scripts use these helpers to keep the validation entry points
simple while still supporting per-shape postprocess heads:

* manual cubic heatmap NMS+TopK
* fused heatmap TopK + full-res-like PAF scorer
* fused-pruned / merged_fused_pruned TopM pair head

The functions are intentionally conservative: if a head already exists and
``force`` is false, the underlying compiler returns the cached .mxr path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, MutableSet, Sequence, Tuple


FUSED_MODE = "mx_fused_cubic_topk_fullres_paf"
PRUNED_MODE = "mx_fused_cubic_topk_fullres_paf_pruned"
MANUAL_MODE = "migraphx_manual_cubic_nms_topk"
NMS_MODES = {"migraphx_nms", "migraphx_nms_k20"}

MERGED_FUSED_PRUNED_ALIASES = {
    "merged_fused_pruned",
    "merged-fused-pruned",
    "mx_merged_fused_pruned",
    "mx-merged-fused-pruned",
}


def normalize_validation_variant_name(name: str) -> str:
    """Normalize local validation aliases before calling postprocessing.normalize_mode."""
    value = str(name).strip()
    if value in MERGED_FUSED_PRUNED_ALIASES:
        return PRUNED_MODE
    return value


def needs_manual_head(variants: Sequence[str]) -> bool:
    return MANUAL_MODE in set(variants)


def needs_fused_head(variants: Sequence[str]) -> bool:
    return FUSED_MODE in set(variants)


def needs_pruned_head(variants: Sequence[str]) -> bool:
    return PRUNED_MODE in set(variants)


def _args_get(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _fused_pruned_heatmap_mode(args: Any) -> str:
    return str(_args_get(args, "fused_pruned_heatmap_mode", "full-res"))


def _smart_cache_suffix(args: Any) -> str:
    return (
        f"smartfullres_sp{int(_args_get(args, 'smart_proposals', 64))}"
        f"_lr{int(_args_get(args, 'smart_local_radius', 8))}"
        f"_lnms{int(_args_get(args, 'smart_lowres_nms_radius', 1))}"
    )


def _fused_pruned_cache_dir(args: Any) -> str:
    configured = str(_args_get(args, "fused_pruned_postprocess_cache_dir", "models/fused_postprocess_pruned_cache"))
    # If the user keeps the normal default cache dir but selects smart-full-res,
    # redirect to a smart-specific cache to avoid overwriting full-res heads with
    # the same legacy resolver filename.
    if _fused_pruned_heatmap_mode(args) != "full-res" and configured == "models/fused_postprocess_pruned_cache":
        return f"models/fused_postprocess_pruned_cache_{_smart_cache_suffix(args)}"
    return configured


def postprocess_extra_from_args(args: Any) -> dict:
    """Build PostprocessConfig.extra for MIGraphX fused/manual/pruned paths."""
    heatmap_mode = _fused_pruned_heatmap_mode(args)
    smart_proposals = int(_args_get(args, "smart_proposals", 64))
    smart_local_radius = int(_args_get(args, "smart_local_radius", 8))
    smart_lowres_nms_radius = int(_args_get(args, "smart_lowres_nms_radius", 1))
    pruned_cache_dir = _fused_pruned_cache_dir(args)
    return {
        "gpu_compute_dtype": _args_get(args, "gpu_compute_dtype", "float32"),
        "nms_impl": _args_get(args, "nms_impl", "separable"),
        "prealloc_resize_buffers": bool(_args_get(args, "prealloc_resize_buffers", False)),
        # Manual cubic TopK head.
        "manual_cubic_topk": int(_args_get(args, "manual_cubic_topk", _args_get(args, "max_keypoints", 20))),
        "manual_cubic_threshold": float(_args_get(args, "manual_cubic_threshold", _args_get(args, "threshold", 0.1))),
        "manual_cubic_nms_radius": int(_args_get(args, "manual_cubic_nms_radius", _args_get(args, "nms_radius_fullres", 6))),
        "manual_cubic_nms_impl": str(_args_get(args, "manual_cubic_nms_impl", _args_get(args, "nms_impl", "separable"))),
        "manual_cubic_a": float(_args_get(args, "manual_cubic_a", -0.75)),
        "heatmap_cubic_a": float(_args_get(args, "manual_cubic_a", -0.75)),
        # Fused full-res-like PAF head.
        "fused_postprocess_cache_dir": _args_get(args, "fused_postprocess_cache_dir", "models/fused_postprocess_cache"),
        "migraphx_fused_postprocess_cache_dir": _args_get(args, "fused_postprocess_cache_dir", "models/fused_postprocess_cache"),
        "fused_postprocess_mxr": _args_get(args, "fused_postprocess_mxr", ""),
        "migraphx_fused_postprocess_mxr": _args_get(args, "fused_postprocess_mxr", ""),
        "paf_cubic_a": float(_args_get(args, "paf_cubic_a", -0.75)),
        # Fused-pruned / merged_fused_pruned head.
        "fused_pruned_postprocess_cache_dir": pruned_cache_dir,
        "migraphx_fused_pruned_postprocess_cache_dir": pruned_cache_dir,
        "fused_postprocess_pruned_cache_dir": pruned_cache_dir,
        "fused_pruned_postprocess_mxr": _args_get(args, "fused_pruned_postprocess_mxr", ""),
        "migraphx_fused_pruned_postprocess_mxr": _args_get(args, "fused_pruned_postprocess_mxr", ""),
        "limb_topm": int(_args_get(args, "limb_topm", 20)),
        "fused_limb_topm": int(_args_get(args, "limb_topm", 20)),
        "min_pair_score": float(_args_get(args, "min_pair_score", 0.0)),
        # Experimental heatmap candidate generator inside fused-pruned heads.
        "fused_pruned_heatmap_mode": heatmap_mode,
        "heatmap_mode": heatmap_mode,
        "smart_proposals": smart_proposals,
        "smart_local_radius": smart_local_radius,
        "smart_lowres_nms_radius": smart_lowres_nms_radius,
    }


def ensure_video_postprocess_heads(args: Any, variants: Sequence[str], video_path: str | Path) -> None:
    """Compile fixed-video-shape postprocess heads when requested/missing."""
    if not bool(_args_get(args, "compile_missing_postprocess_heads", False)):
        return

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video to inspect shape: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first video frame for postprocess head compile: {video_path}")

    full_h, full_w = frame.shape[:2]
    in_h = int(_args_get(args, "target_height", 544)) // int(_args_get(args, "stride", 8))
    in_w = int(_args_get(args, "target_width", 968)) // int(_args_get(args, "stride", 8))
    ensure_shape_postprocess_heads(args, variants, (in_h, in_w), (full_h, full_w), compiled=set())


def ensure_shape_postprocess_heads(
    args: Any,
    variants: Sequence[str],
    heatmaps_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
    *,
    compiled: MutableSet[Tuple[Any, ...]] | None = None,
) -> None:
    """Compile heads for one low-res/full-res shape if variants require them."""
    if not bool(_args_get(args, "compile_missing_postprocess_heads", False)):
        return

    compiled = compiled if compiled is not None else set()
    in_h, in_w = int(heatmaps_hw[0]), int(heatmaps_hw[1])
    full_h, full_w = int(original_hw[0]), int(original_hw[1])
    force = bool(_args_get(args, "force_compile_postprocess_heads", False))
    keep_onnx = bool(_args_get(args, "keep_postprocess_onnx", False))
    variants_set = set(variants)

    heatmap_mode = _fused_pruned_heatmap_mode(args)
    smart_proposals = int(_args_get(args, "smart_proposals", 64))
    smart_local_radius = int(_args_get(args, "smart_local_radius", 8))
    smart_lowres_nms_radius = int(_args_get(args, "smart_lowres_nms_radius", 1))
    pruned_cache_dir = _fused_pruned_cache_dir(args)

    common_key = (
        in_h,
        in_w,
        full_h,
        full_w,
        int(_args_get(args, "max_keypoints", 20)),
        float(_args_get(args, "threshold", 0.1)),
        int(_args_get(args, "nms_radius_fullres", 6)),
        str(_args_get(args, "nms_impl", "separable")),
        float(_args_get(args, "manual_cubic_a", -0.75)),
        int(_args_get(args, "points_per_limb", 8)),
        float(_args_get(args, "min_paf_score", 0.05)),
        float(_args_get(args, "success_ratio_thr", 0.8)),
        float(_args_get(args, "paf_cubic_a", -0.75)),
    )

    if MANUAL_MODE in variants_set and not _args_get(args, "migraphx_manual_cubic_topk_mxr", ""):
        key = (MANUAL_MODE,) + common_key
        if key not in compiled:
            from modules.migraphx_manual_cubic_topk_compiler import compile_manual_cubic_nms_topk_head

            compile_manual_cubic_nms_topk_head(
                in_h=in_h,
                in_w=in_w,
                out_h=full_h,
                out_w=full_w,
                output_dir=_args_get(args, "migraphx_manual_cubic_topk_cache_dir", "models/manual_cubic_nms_topk_cache"),
                channels=18,
                topk=int(_args_get(args, "manual_cubic_topk", _args_get(args, "max_keypoints", 20))),
                threshold=float(_args_get(args, "manual_cubic_threshold", _args_get(args, "threshold", 0.1))),
                nms_radius=int(_args_get(args, "manual_cubic_nms_radius", _args_get(args, "nms_radius_fullres", 6))),
                nms_impl=str(_args_get(args, "manual_cubic_nms_impl", _args_get(args, "nms_impl", "separable"))),
                cubic_a=float(_args_get(args, "manual_cubic_a", -0.75)),
                force=force,
                keep_onnx=keep_onnx,
            )
            compiled.add(key)

    if FUSED_MODE in variants_set and not _args_get(args, "fused_postprocess_mxr", ""):
        key = (FUSED_MODE,) + common_key
        if key not in compiled:
            from modules.migraphx_fused_postprocess_compiler import compile_fused_postprocess_head

            compile_fused_postprocess_head(
                in_h=in_h,
                in_w=in_w,
                full_h=full_h,
                full_w=full_w,
                output_dir=_args_get(args, "fused_postprocess_cache_dir", "models/fused_postprocess_cache"),
                topk=int(_args_get(args, "max_keypoints", 20)),
                threshold=float(_args_get(args, "threshold", 0.1)),
                nms_radius=int(_args_get(args, "nms_radius_fullres", 6)),
                nms_impl=str(_args_get(args, "nms_impl", "separable")),
                heatmap_cubic_a=float(_args_get(args, "manual_cubic_a", -0.75)),
                points_per_limb=int(_args_get(args, "points_per_limb", 8)),
                min_paf_score=float(_args_get(args, "min_paf_score", 0.05)),
                success_ratio_thr=float(_args_get(args, "success_ratio_thr", 0.8)),
                paf_cubic_a=float(_args_get(args, "paf_cubic_a", -0.75)),
                force=force,
                keep_onnx=keep_onnx,
            )
            compiled.add(key)

    if PRUNED_MODE in variants_set and not _args_get(args, "fused_pruned_postprocess_mxr", ""):
        key = (
            PRUNED_MODE,
            heatmap_mode,
            smart_proposals,
            smart_local_radius,
            smart_lowres_nms_radius,
            int(_args_get(args, "limb_topm", 20)),
            float(_args_get(args, "min_pair_score", 0.0)),
        ) + common_key
        if key not in compiled:
            import shutil
            from modules.migraphx_fused_postprocess_pruned_compiler import (
                compile_pruned_fused_postprocess_head,
                pruned_head_name,
            )

            compiled_path = compile_pruned_fused_postprocess_head(
                in_h=in_h,
                in_w=in_w,
                full_h=full_h,
                full_w=full_w,
                output_dir=pruned_cache_dir,
                topk=int(_args_get(args, "max_keypoints", 20)),
                limb_topm=int(_args_get(args, "limb_topm", 20)),
                threshold=float(_args_get(args, "threshold", 0.1)),
                nms_radius=int(_args_get(args, "nms_radius_fullres", 6)),
                nms_impl=str(_args_get(args, "nms_impl", "separable")),
                heatmap_cubic_a=float(_args_get(args, "manual_cubic_a", -0.75)),
                points_per_limb=int(_args_get(args, "points_per_limb", 8)),
                min_paf_score=float(_args_get(args, "min_paf_score", 0.05)),
                success_ratio_thr=float(_args_get(args, "success_ratio_thr", 0.8)),
                paf_cubic_a=float(_args_get(args, "paf_cubic_a", -0.75)),
                min_pair_score=float(_args_get(args, "min_pair_score", 0.0)),
                batch_size=1,
                heatmap_mode=heatmap_mode,
                smart_proposals=smart_proposals,
                smart_local_radius=smart_local_radius,
                smart_lowres_nms_radius=smart_lowres_nms_radius,
                force=force,
                keep_onnx=keep_onnx,
            )

            # modules.postprocessing resolves pruned heads through the legacy
            # full-res filename. For smart mode, keep the smart-tokened file for
            # provenance and also write a resolver alias inside the smart cache.
            if heatmap_mode != "full-res":
                alias_name = pruned_head_name(
                    in_h,
                    in_w,
                    full_h,
                    full_w,
                    topk=int(_args_get(args, "max_keypoints", 20)),
                    limb_topm=int(_args_get(args, "limb_topm", 20)),
                    threshold=float(_args_get(args, "threshold", 0.1)),
                    nms_radius=int(_args_get(args, "nms_radius_fullres", 6)),
                    nms_impl=str(_args_get(args, "nms_impl", "separable")),
                    heatmap_cubic_a=float(_args_get(args, "manual_cubic_a", -0.75)),
                    points_per_limb=int(_args_get(args, "points_per_limb", 8)),
                    min_paf_score=float(_args_get(args, "min_paf_score", 0.05)),
                    success_ratio_thr=float(_args_get(args, "success_ratio_thr", 0.8)),
                    paf_cubic_a=float(_args_get(args, "paf_cubic_a", -0.75)),
                    min_pair_score=float(_args_get(args, "min_pair_score", 0.0)),
                    batch_size=1,
                    heatmap_mode="full-res",
                )
                alias_path = Path(pruned_cache_dir) / f"{alias_name}.mxr"
                if Path(compiled_path).resolve() != alias_path.resolve():
                    alias_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(compiled_path), str(alias_path))
                    print(f"[autocompile] smart resolver alias: {alias_path} -> {compiled_path}")

            compiled.add(key)
