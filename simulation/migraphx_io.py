"""MIGraphX input batching and output decoding helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .shared_memory import _input_tensor_from_item


def cast_for_migraphx(expected_dtype: str, tensor: np.ndarray) -> np.ndarray:
    if "half" in expected_dtype:
        return np.ascontiguousarray(tensor.astype(np.float16, copy=False))
    # MIGraphX Python inputs are safest as fp32 for fp32/bf16 cases.
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False))


def _as_migraphx_output_array(x: Any) -> np.ndarray:
    """Convert MIGraphX output argument/list item to numpy without assuming layout."""
    return np.asarray(x)


def decode_migraphx_batch_outputs(
    results: Any,
    out_h: int,
    out_w: int,
    output_dtype: str,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode MIGraphX model outputs for batched inference.

    Expected model output for input Bx3xHxW:
        heatmaps: B x 19 x out_h x out_w
        pafs:     B x 38 x out_h x out_w

    Returns:
        heatmaps: B x out_h x out_w x 19
        pafs:     B x out_h x out_w x 38
    """
    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) < 2:
        raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")

    heat_raw = _as_migraphx_output_array(results[-2])
    paf_raw = _as_migraphx_output_array(results[-1])

    heat_per_sample = 19 * out_h * out_w
    paf_per_sample = 38 * out_h * out_w

    if batch_size is None:
        if heat_raw.size % heat_per_sample != 0:
            raise RuntimeError(
                f"Cannot infer heatmap batch size from output size={heat_raw.size}, "
                f"per_sample={heat_per_sample}, raw_shape={heat_raw.shape}"
            )
        batch_size = int(heat_raw.size // heat_per_sample)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"Invalid decoded batch size: {batch_size}")

    if heat_raw.ndim == 4:
        if heat_raw.shape[0] == batch_size and heat_raw.shape[1] == 19:
            heat_bchw = heat_raw
        elif heat_raw.shape[0] == batch_size and heat_raw.shape[-1] == 19:
            heat_bchw = np.transpose(heat_raw, (0, 3, 1, 2))
        else:
            heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)
    elif heat_raw.ndim == 3:
        if heat_raw.shape[0] == 19:
            heat_bchw = heat_raw.reshape(1, 19, out_h, out_w)
        elif heat_raw.shape[-1] == 19:
            heat_bchw = np.transpose(heat_raw, (2, 0, 1)).reshape(1, 19, out_h, out_w)
        else:
            heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)
    else:
        heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)

    if paf_raw.ndim == 4:
        if paf_raw.shape[0] == batch_size and paf_raw.shape[1] == 38:
            paf_bchw = paf_raw
        elif paf_raw.shape[0] == batch_size and paf_raw.shape[-1] == 38:
            paf_bchw = np.transpose(paf_raw, (0, 3, 1, 2))
        else:
            paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)
    elif paf_raw.ndim == 3:
        if paf_raw.shape[0] == 38:
            paf_bchw = paf_raw.reshape(1, 38, out_h, out_w)
        elif paf_raw.shape[-1] == 38:
            paf_bchw = np.transpose(paf_raw, (2, 0, 1)).reshape(1, 38, out_h, out_w)
        else:
            paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)
    else:
        paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)

    heat_bhwc = np.moveaxis(heat_bchw, 1, -1)
    paf_bhwc = np.moveaxis(paf_bchw, 1, -1)

    if output_dtype == "float16":
        return (
            np.ascontiguousarray(heat_bhwc, dtype=np.float16),
            np.ascontiguousarray(paf_bhwc, dtype=np.float16),
        )
    return (
        np.ascontiguousarray(heat_bhwc, dtype=np.float32),
        np.ascontiguousarray(paf_bhwc, dtype=np.float32),
    )


def decode_migraphx_outputs(results: Any, out_h: int, out_w: int, output_dtype: str) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible single-frame decode. Returns HxWx19 and HxWx38."""
    heat_bhwc, paf_bhwc = decode_migraphx_batch_outputs(
        results,
        out_h,
        out_w,
        output_dtype,
        batch_size=1,
    )
    return heat_bhwc[0], paf_bhwc[0]


def make_migraphx_input_batch(
    items: Sequence[Dict[str, Any]],
    expected_dtype: str,
    compiled_batch_size: int,
    shared_input_slots: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Build Bx3xHxW input for MIGraphX.

    Each camera item contains input_tensor shaped 1x3xHxW.
    If compiled_batch_size > number of real items, pad by repeating the last
    frame. This is useful for static batch MXR files such as b4/b8 models.
    """
    if not items:
        raise RuntimeError("Cannot build MIGraphX batch from empty item list.")

    actual_batch_size = len(items)
    tensors = []
    for item in items:
        x = _input_tensor_from_item(item, shared_input_slots)
        if x.ndim != 4:
            raise ValueError(f"input_tensor must be 4D, got shape={x.shape}")
        if x.shape[0] != 1:
            raise ValueError(f"Each queued item must be one frame, got shape={x.shape}")
        tensors.append(x)

    batch = np.concatenate(tensors, axis=0)

    compiled_batch_size = max(1, int(compiled_batch_size))
    if compiled_batch_size > actual_batch_size:
        pad_count = compiled_batch_size - actual_batch_size
        pad = np.repeat(batch[-1:, ...], pad_count, axis=0)
        batch = np.concatenate([batch, pad], axis=0)

    return cast_for_migraphx(expected_dtype, batch), actual_batch_size


def build_inference_output_items_from_batch(
    *,
    batch_items: Sequence[Dict[str, Any]],
    heatmaps_bhwc: np.ndarray,
    pafs_bhwc: np.ndarray,
    infer_done_ts: float,
    inference_ms_total: float,
    decode_ms_total: float,
    queue_wait_times_ms: Sequence[float],
) -> List[Dict[str, Any]]:
    """Split batched MIGraphX outputs back into per-frame pipeline items."""
    out_items: List[Dict[str, Any]] = []
    n = len(batch_items)
    if n <= 0:
        return out_items

    for i, item in enumerate(batch_items):
        out_item = {
            "camera_id": int(item["camera_id"]),
            "frame_id": int(item["frame_id"]),
            "source": item["source"],
            "capture_ts": float(item["capture_ts"]),
            "preprocess_done_ts": float(item["preprocess_done_ts"]),
            "infer_done_ts": infer_done_ts,
            "original_hw": tuple(item["original_hw"]),
            "preprocess_ms": float(item["preprocess_ms"]),
            "queue_pre_to_infer_ms": float(queue_wait_times_ms[i]),
            "inference_ms": float(inference_ms_total) / float(n),
            "decode_ms": float(decode_ms_total) / float(n),
            "batch_inference_ms": float(inference_ms_total),
            "batch_decode_ms": float(decode_ms_total),
            "migraphx_batch_size": int(n),
            "heatmaps": np.ascontiguousarray(heatmaps_bhwc[i]),
            "pafs": np.ascontiguousarray(pafs_bhwc[i]),
        }
        if "frame_bgr" in item:
            out_item["frame_bgr"] = item["frame_bgr"]
        out_items.append(out_item)
    return out_items
