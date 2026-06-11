"""Postprocess mode normalization and merged pose/fused-pruned helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .utils import Timer


MERGED_POSE_FUSED_PRUNED_ALIASES = {
    "mx_merged_pose_fused_pruned",
    "mx-merged-pose-fused-pruned",
    "merged_pose_fused_pruned",
    "merged-pose-fused-pruned",
    "pose_fused_pruned",
    "pose-fused-pruned",
    "pose_fused_pruned_b1",
    "pose-fused-pruned-b1",
    "merged_pose_fused_pruned_b1",
    "merged-pose-fused-pruned-b1",
}


def is_merged_pose_fused_pruned_variant(user_variant: str) -> bool:
    key = str(user_variant or "").strip().lower().replace(" ", "-")
    key_dash = key.replace("_", "-")
    key_underscore = key.replace("-", "_")
    return key in MERGED_POSE_FUSED_PRUNED_ALIASES or key_dash in MERGED_POSE_FUSED_PRUNED_ALIASES or key_underscore in MERGED_POSE_FUSED_PRUNED_ALIASES


def _select_merged_batch_tensor(arr: Any, item_index: int, real_batch_size: int) -> np.ndarray:
    """Return the per-frame slice from a merged pose+pruned output tensor.

    The B=1 model that was inspected returns:
        top_scores       [1, 18, 20]
        top_indices      [1, 18, 20]
        pair a/b/score   [1, 19, 20]
        pair valid       [1, 19, 20]

    Keep the leading singleton batch axis for B=1 because the existing pruned
    assembly helpers already accept that shape.  For future B>1 merged models,
    take item_index:item_index+1 so each queued item still looks like B=1.
    """
    x = np.asarray(arr)
    if x.ndim >= 3 and x.shape[0] >= real_batch_size:
        return np.ascontiguousarray(x[item_index : item_index + 1])
    if real_batch_size == 1:
        return np.ascontiguousarray(x)
    raise ValueError(
        f"Merged output cannot be split for item {item_index}/{real_batch_size}: "
        f"shape={x.shape}"
    )


def build_merged_pose_fused_pruned_items_from_batch(
    *,
    batch_items: Sequence[Dict[str, Any]],
    results: Any,
    infer_done_ts: float,
    inference_ms_total: float,
    decode_ms_total: float,
    queue_wait_times_ms: Sequence[float],
) -> List[Dict[str, Any]]:
    """Convert merged pose+fused-pruned MXR outputs to per-frame queue items.

    The merged MXR already includes pose inference, heatmap TopK, full-res PAF
    scoring and per-limb TopM pruning.  Therefore postprocess workers should
    receive only the six small pruned tensors and run CPU pose assembly.
    """
    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) != 6:
        raise RuntimeError(
            "Merged pose+fused-pruned model must return exactly 6 outputs: "
            "top_scores, top_indices, limb_top_pair_a_idx, limb_top_pair_b_idx, "
            f"limb_top_pair_score, limb_top_pair_valid. Got {len(results)} outputs."
        )

    top_scores, top_indices, a_idx, b_idx, pair_score, pair_valid = [np.asarray(x) for x in results]
    n = len(batch_items)
    out_items: List[Dict[str, Any]] = []
    for i, item in enumerate(batch_items):
        out_item = {
            "camera_id": int(item["camera_id"]),
            "frame_id": int(item["frame_id"]),
            "source": item["source"],
            "capture_ts": float(item["capture_ts"]),
            "preprocess_done_ts": float(item["preprocess_done_ts"]),
            "infer_done_ts": float(infer_done_ts),
            "original_hw": tuple(item["original_hw"]),
            "preprocess_ms": float(item["preprocess_ms"]),
            "queue_pre_to_infer_ms": float(queue_wait_times_ms[i]),
            "inference_ms": float(inference_ms_total) / float(max(1, n)),
            "decode_ms": float(decode_ms_total) / float(max(1, n)),
            "batch_inference_ms": float(inference_ms_total),
            "batch_decode_ms": float(decode_ms_total),
            "migraphx_batch_size": int(n),
            "merged_pose_fused_pruned_precomputed": True,
            "fused_pruned_precomputed": True,
            "merged_pose_fused_pruned_mx_ms": float(inference_ms_total) / float(max(1, n)),
            "fused_pruned_mx_ms": float(inference_ms_total) / float(max(1, n)),
            "fused_pruned_top_scores": _select_merged_batch_tensor(top_scores, i, n).astype(np.float32, copy=False),
            "fused_pruned_top_indices": _select_merged_batch_tensor(top_indices, i, n).astype(np.int64, copy=False),
            "fused_pruned_a_idx": _select_merged_batch_tensor(a_idx, i, n).astype(np.int64, copy=False),
            "fused_pruned_b_idx": _select_merged_batch_tensor(b_idx, i, n).astype(np.int64, copy=False),
            "fused_pruned_pair_score": _select_merged_batch_tensor(pair_score, i, n).astype(np.float32, copy=False),
            "fused_pruned_pair_valid": _select_merged_batch_tensor(pair_valid, i, n).astype(np.float32, copy=False),
        }
        if "frame_bgr" in item:
            out_item["frame_bgr"] = item["frame_bgr"]
        out_items.append(out_item)
    return out_items


def _squeeze_leading_batch_axis(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        return np.ascontiguousarray(arr[0])
    return np.ascontiguousarray(arr)


def postprocess_precomputed_merged_pose_fused_pruned_item(
    *,
    item: Dict[str, Any],
    threshold: float,
    min_pair_score: float = 0.0,
):
    """CPU-only tail for a merged pose+fused-pruned MXR output item."""
    from modules.postprocessing import PostprocessOutput
    from modules.mx_pair_assembly_pruned import assemble_poses_from_pruned_pairs

    top_scores = item["fused_pruned_top_scores"]
    top_indices = item["fused_pruned_top_indices"]
    a_idx = item["fused_pruned_a_idx"]
    b_idx = item["fused_pruned_b_idx"]
    pair_score = item["fused_pruned_pair_score"]
    pair_valid = item["fused_pruned_pair_valid"]

    with Timer() as t_cpu:
        try:
            poses, kpts, asm_times = assemble_poses_from_pruned_pairs(
                top_scores,
                top_indices,
                a_idx,
                b_idx,
                pair_score,
                pair_valid,
                full_width=int(item["original_hw"][1]),
                threshold=float(threshold),
                min_pair_score=float(min_pair_score),
                return_timing=True,
            )
        except Exception:
            # Some helper versions expect pair tensors without a leading B=1 axis.
            poses, kpts, asm_times = assemble_poses_from_pruned_pairs(
                _squeeze_leading_batch_axis(top_scores),
                _squeeze_leading_batch_axis(top_indices),
                _squeeze_leading_batch_axis(a_idx),
                _squeeze_leading_batch_axis(b_idx),
                _squeeze_leading_batch_axis(pair_score),
                _squeeze_leading_batch_axis(pair_valid),
                full_width=int(item["original_hw"][1]),
                threshold=float(threshold),
                min_pair_score=float(min_pair_score),
                return_timing=True,
            )

    mx_ms = float(item.get("merged_pose_fused_pruned_mx_ms", item.get("fused_pruned_mx_ms", 0.0)) or 0.0)
    timings: Dict[str, float] = {
        "merged_pose_fused_pruned_mx_in_infer": mx_ms,
        "fused_pruned_mx_in_infer": mx_ms,
        "pruned_cpu_tail": float(t_cpu.ms),
        "topk_adapter": float(asm_times.get("topk_adapter", 0.0)),
        "mx_assembly_total": float(asm_times.get("mx_assembly_total", t_cpu.ms)),
        "group_pose": float(t_cpu.ms),
        "group_keypoints": float(t_cpu.ms),
        "group_total": float(t_cpu.ms),
        # For stream summary, post_ms should represent only the CPU tail because
        # the GPU merged work has already been accounted for as inference_ms.
        "total_postprocess": float(t_cpu.ms),
    }
    for k, v in asm_times.items():
        try:
            timings[str(k)] = float(v)
        except Exception:
            pass

    return PostprocessOutput(
        np.asarray(poses, dtype=np.float32),
        np.asarray(kpts, dtype=np.float32),
        timings,
    )

def resolve_registry_mode(user_mode: str) -> Tuple[str, str, bool]:
    """Map public CLI variant to the actual mode used by postprocess_from_maps.

    postprocess_from_maps intentionally rejects *_two_process aliases because in
    speed/accuracy validators those are handled by a special runner. In this
    script the process split is already provided by the architecture, so the
    worker maps the alias back to the underlying map-based registry mode.
    """
    if is_merged_pose_fused_pruned_variant(user_mode):
        return "mx_merged_pose_fused_pruned", "mx_merged_pose_fused_pruned", False

    from modules.postprocessing import normalize_mode

    canonical = normalize_mode(user_mode)
    if canonical == "gpu_nms_fullres_two_process":
        return canonical, "gpu_nms_fullres_cpu_group", True
    if canonical == "gpu_nms_lowres_two_process":
        return canonical, "gpu_nms_lowres_cpu_group", True
    if canonical == "cpu_k20_fast_two_process":
        return canonical, "optimized_batch_k20_fast", False
    return canonical, canonical, canonical.startswith("gpu")


def select_migraphx_nms_mxr_for_hw(
    *,
    original_hw: Tuple[int, int],
    migraphx_nms_mxr: str = "",
    migraphx_nms_cache_dir: str = "",
) -> str:
    """Resolve the compiled MIGraphX NMS head for a full-resolution frame.

    Video streams have constant frame resolution, so normally one cached
    heatmap_nms_head_<H>x<W>.mxr file is enough for the whole run.
    """
    if migraphx_nms_mxr:
        return migraphx_nms_mxr

    if not migraphx_nms_cache_dir:
        return ""

    h, w = int(original_hw[0]), int(original_hw[1])
    return str(Path(migraphx_nms_cache_dir) / f"heatmap_nms_head_{h}x{w}.mxr")


def compile_migraphx_nms_for_stream_if_requested(args, sources: Sequence[str]) -> None:
    if not getattr(args, "compile_migraphx_nms", False):
        return

    cache_dir = getattr(args, "migraphx_nms_cache_dir", "") or "models/nms_fullres_cache"
    video = sources[0] if sources else ""
    if not video:
        raise RuntimeError("Cannot compile MIGraphX NMS head: no input video source found.")

    from modules.migraphx_compiler import compile_nms_cache_for_video

    print(f"[MX-NMS] compiling stream NMS head from video: {video}", flush=True)
    compile_nms_cache_for_video(
        video=video,
        output_dir=cache_dir,
        force=bool(getattr(args, "force_compile_migraphx_nms", False)),
        keep_onnx=bool(getattr(args, "keep_migraphx_nms_onnx", False)),
        exhaustive_tune=bool(getattr(args, "exhaustive_tune_migraphx_nms", False)),
    )
