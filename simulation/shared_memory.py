"""Shared-memory slot helpers for camera, inference, and postprocess transport."""

from __future__ import annotations

import os
import sys
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _dtype_from_name(name: str):
    return np.float16 if str(name) == "float16" else np.float32


def _default_heatmap_channels() -> int:
    """Return shared heatmap channel count for the active stream variant.

    The legacy stream models return 19 heatmap channels, including the background
    channel.  The split pose adapter used by the host-mediated
    MXR1 -> hip_host_smart -> MXR2 stream path returns 18 body-keypoint channels.

    Shared map buffers must match the decoded output shape exactly; otherwise the
    inference worker falls back to Queue/pickle transport and reports
    shared_map_misses for every frame.
    """
    raw = os.environ.get("STREAM_SHARED_HEATMAP_CHANNELS", "").strip()
    if raw:
        value = int(raw)
        if value not in (18, 19):
            raise ValueError(f"STREAM_SHARED_HEATMAP_CHANNELS must be 18 or 19, got {value}")
        return value

    # The split HIP smart integration is injected from the root wrapper and sets
    # the variant through normal CLI args.  Keep this fallback local so callers do
    # not have to thread an extra parameter through the whole simulator runner.
    argv = " ".join(sys.argv).strip().lower().replace("-", "_")
    if "split_hip_host_smart" in argv or "split_hip_smart" in argv or "mxr1_hip_smart_mxr2" in argv:
        return 18
    return 19


def create_shared_map_buffers(
    num_slots: int,
    out_h: int,
    out_w: int,
    dtype_name: str,
    heatmap_channels: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[shared_memory.SharedMemory]]:
    dtype = _dtype_from_name(dtype_name)
    heat_c = int(_default_heatmap_channels() if heatmap_channels is None else heatmap_channels)
    if heat_c not in (18, 19):
        raise ValueError(f"heatmap_channels must be 18 or 19, got {heat_c}")
    heat_shape = (int(out_h), int(out_w), heat_c)
    paf_shape = (int(out_h), int(out_w), 38)
    heat_nbytes = int(np.prod(heat_shape) * np.dtype(dtype).itemsize)
    paf_nbytes = int(np.prod(paf_shape) * np.dtype(dtype).itemsize)
    descs: List[Dict[str, Any]] = []
    handles: List[shared_memory.SharedMemory] = []
    for slot_id in range(max(0, int(num_slots))):
        heat_shm = shared_memory.SharedMemory(create=True, size=heat_nbytes)
        paf_shm = shared_memory.SharedMemory(create=True, size=paf_nbytes)
        handles.extend([heat_shm, paf_shm])
        descs.append({
            "slot_id": slot_id,
            "dtype": np.dtype(dtype).name,
            "heat_shape": heat_shape,
            "paf_shape": paf_shape,
            "heat_channels": heat_c,
            "heat_name": heat_shm.name,
            "paf_name": paf_shm.name,
        })
    return descs, handles


def close_shared_map_buffers(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def open_shared_map_buffers(descs: Optional[Sequence[Dict[str, Any]]]):
    if not descs:
        return {}, []
    slots: Dict[int, Dict[str, Any]] = {}
    handles = []
    for desc in descs:
        heat_shm = shared_memory.SharedMemory(name=desc["heat_name"])
        paf_shm = shared_memory.SharedMemory(name=desc["paf_name"])
        handles.extend([heat_shm, paf_shm])
        dtype = np.dtype(desc["dtype"])
        slots[int(desc["slot_id"])] = {
            "heat": np.ndarray(tuple(desc["heat_shape"]), dtype=dtype, buffer=heat_shm.buf),
            "paf": np.ndarray(tuple(desc["paf_shape"]), dtype=dtype, buffer=paf_shm.buf),
        }
    return slots, handles


def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass


def create_shared_input_buffers(
    num_slots: int,
    target_h: int,
    target_w: int,
    dtype_name: str = "float32",
) -> Tuple[List[Dict[str, Any]], List[shared_memory.SharedMemory]]:
    """Create shared-memory slots for preprocessed 1x3xHxW input tensors."""
    dtype = _dtype_from_name(dtype_name)
    shape = (1, 3, int(target_h), int(target_w))
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    descs: List[Dict[str, Any]] = []
    handles: List[shared_memory.SharedMemory] = []
    for slot_id in range(max(0, int(num_slots))):
        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        handles.append(shm)
        descs.append({
            "slot_id": slot_id,
            "dtype": np.dtype(dtype).name,
            "shape": shape,
            "input_name": shm.name,
        })
    return descs, handles


def open_shared_input_buffers(descs: Optional[Sequence[Dict[str, Any]]]):
    if not descs:
        return {}, []
    slots: Dict[int, Dict[str, Any]] = {}
    handles = []
    for desc in descs:
        shm = shared_memory.SharedMemory(name=desc["input_name"])
        handles.append(shm)
        dtype = np.dtype(desc["dtype"])
        slots[int(desc["slot_id"])] = {
            "input": np.ndarray(tuple(desc["shape"]), dtype=dtype, buffer=shm.buf),
        }
    return slots, handles


def close_shared_input_buffers(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def _input_tensor_from_item(
    item: Dict[str, Any],
    shared_input_slots: Optional[Dict[int, Dict[str, Any]]] = None,
) -> np.ndarray:
    """Return queued item tensor, either from shared memory or old Queue payload."""
    if shared_input_slots and "shared_input_slot" in item:
        slot_id = int(item["shared_input_slot"])
        return np.asarray(shared_input_slots[slot_id]["input"])
    return np.asarray(item["input_tensor"])


def release_shared_slot_from_item(item: Any, free_q) -> None:
    if not isinstance(item, dict) or free_q is None:
        return
    slot_id = item.get("shared_map_slot")
    if slot_id is None:
        return
    try:
        free_q.put_nowait(int(slot_id))
    except Exception:
        pass
