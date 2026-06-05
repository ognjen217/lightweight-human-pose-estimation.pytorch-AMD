#!/usr/bin/env python3
"""
Patch simulate_10_camera_stream.py to use shared-memory input tensors in
latest-buffer mode.

The old latest-mode camera workers enqueue the full preprocessed NumPy tensor
through multiprocessing.Queue. For 1x3x544x968 float32 this is about 6 MB per
frame. With many cameras this creates heavy pickling/copying/memory-bandwidth
pressure before MIGraphX ever runs.

After this patch, when --shared-input-slots is enabled:

    camera process:
        preprocess frame -> write tensor into shared memory -> enqueue metadata

    inference process:
        dequeue metadata -> read shared-memory tensor view -> build MXR batch

The default remains unchanged when --shared-input-slots=0.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HELPERS = '''

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
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence for {label}, found {count}.")
    return text.replace(old, new, 1)


def replace_once_in_region(
    text: str,
    start_marker: str,
    end_marker: str,
    old: str,
    new: str,
    label: str,
) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    region = text[start:end]
    count = region.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence for {label} in region, found {count}.")
    region = region.replace(old, new, 1)
    return text[:start] + region + text[end:]


def patch_file(path: Path) -> bool:
    text = path.read_text()
    original = text

    if "create_shared_input_buffers" in text:
        print("[OK] shared-input optimization already appears to be applied.")
        return False

    # 1) Add shared-input helper functions near existing shared-map helpers.
    text = replace_once(
        text,
        '''def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
''',
        '''def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
''' + HELPERS + "\n",
        "insert shared input helpers",
    )

    # 2) Teach MIGraphX input batch builder to read shared-memory tensors.
    text = replace_once(
        text,
        '''def make_migraphx_input_batch(
    items: Sequence[Dict[str, Any]],
    expected_dtype: str,
    compiled_batch_size: int,
) -> Tuple[np.ndarray, int]:
''',
        '''def make_migraphx_input_batch(
    items: Sequence[Dict[str, Any]],
    expected_dtype: str,
    compiled_batch_size: int,
    shared_input_slots: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, int]:
''',
        "extend make_migraphx_input_batch signature",
    )
    text = replace_once(
        text,
        '''        x = np.asarray(item["input_tensor"])
''',
        '''        x = _input_tensor_from_item(item, shared_input_slots)
''',
        "read item tensor through helper",
    )

    # 3) Patch latest camera worker only.
    text = replace_once_in_region(
        text,
        "def camera_preprocess_latest_worker(",
        "def inference_latest_worker(",
        '''    keep_frame_for_output: bool = False,
    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    """Camera worker that maintains a newest-frame-only slot for its camera."""
    try:
        import cv2
''',
        '''    keep_frame_for_output: bool = False,
    shared_input_descs: Optional[Sequence[Dict[str, Any]]] = None,
    shared_input_dtype: str = "float32",
    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    """Camera worker that maintains a newest-frame-only slot for its camera."""
    shared_input_slots = {}
    shared_input_handles = []
    try:
        import cv2
        shared_input_slots, shared_input_handles = open_shared_input_buffers(shared_input_descs)
''',
        "extend latest camera worker signature",
    )
    text = replace_once_in_region(
        text,
        "def camera_preprocess_latest_worker(",
        "def inference_latest_worker(",
        '''            if keep_frame_for_output:
                item["frame_bgr"] = frame.copy()
            replaced_before_infer += latest_put(q, item)
''',
        '''            if keep_frame_for_output:
                item["frame_bgr"] = frame.copy()
            if shared_input_slots:
                slot = shared_input_slots[int(camera_id)]["input"]
                if slot.shape != tensor.shape:
                    raise ValueError(f"shared input slot shape mismatch: {slot.shape}!={tensor.shape}")
                np.copyto(slot, tensor.astype(slot.dtype, copy=False), casting="same_kind")
                item.pop("input_tensor", None)
                item["shared_input_slot"] = int(camera_id)
            replaced_before_infer += latest_put(q, item)
''',
        "write latest camera tensor into shared slot",
    )
    text = replace_once_in_region(
        text,
        "def camera_preprocess_latest_worker(",
        "def inference_latest_worker(",
        '''    except Exception:
        try:
            camera_done[camera_id] = 1
        except Exception:
            pass
        error_q.put({"stage": "camera_preprocess", "camera_id": camera_id, "traceback": traceback.format_exc()})


''',
        '''    except Exception:
        try:
            camera_done[camera_id] = 1
        except Exception:
            pass
        error_q.put({"stage": "camera_preprocess", "camera_id": camera_id, "traceback": traceback.format_exc()})
    finally:
        close_shared_map_views(shared_input_handles)


''',
        "close latest camera shared input views",
    )

    # 4) Patch latest inference worker only.
    text = replace_once_in_region(
        text,
        "def inference_latest_worker(",
        "def postprocess_latest_worker(",
        '''    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,


    migraphx_batch_size: int = 1,
''',
        '''    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,
    shared_input_descs: Optional[Sequence[Dict[str, Any]]] = None,


    migraphx_batch_size: int = 1,
''',
        "extend latest inference signature",
    )
    text = replace_once_in_region(
        text,
        "def inference_latest_worker(",
        "def postprocess_latest_worker(",
        '''        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        shared_map_misses = 0
''',
        '''        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        shared_input_slots, shared_input_handles = open_shared_input_buffers(shared_input_descs)
        shared_map_misses = 0
''',
        "open shared input slots in latest inference",
    )
    text = replace_once_in_region(
        text,
        "def inference_latest_worker(",
        "def postprocess_latest_worker(",
        '''                compiled_batch_size=configured_batch_size,
            )

            with Timer() as t_inf:
''',
        '''                compiled_batch_size=configured_batch_size,
                shared_input_slots=shared_input_slots,
            )

            with Timer() as t_inf:
''',
        "pass shared input slots to latest batch builder",
    )
    text = replace_once_in_region(
        text,
        "def inference_latest_worker(",
        "def postprocess_latest_worker(",
        '''        close_shared_map_views(shared_handles)
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
''',
        '''        close_shared_map_views(shared_handles)
        close_shared_map_views(shared_input_handles)
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
''',
        "close latest inference shared input views",
    )

    # 5) Create and wire shared input buffers in run_latest.
    text = replace_once_in_region(
        text,
        "def run_latest(args)",
        "def run(args)",
        '''    shared_map_descs: List[Dict[str, Any]] = []
    shared_map_handles: List[shared_memory.SharedMemory] = []
    free_map_slots = None
''',
        '''    shared_input_descs: List[Dict[str, Any]] = []
    shared_input_handles: List[shared_memory.SharedMemory] = []
    shared_input_slots_arg = int(getattr(args, "shared_input_slots", 0))
    if 0 < shared_input_slots_arg < int(args.num_cameras):
        raise ValueError("--shared-input-slots must be 0 or at least --num-cameras for one stable slot per camera")
    if shared_input_slots_arg > 0:
        shared_input_descs, shared_input_handles = create_shared_input_buffers(
            shared_input_slots_arg, args.target_height, args.target_width, args.shared_input_dtype
        )

    shared_map_descs: List[Dict[str, Any]] = []
    shared_map_handles: List[shared_memory.SharedMemory] = []
    free_map_slots = None
''',
        "create shared input buffers in run_latest",
    )
    text = replace_once_in_region(
        text,
        "def run_latest(args)",
        "def run(args)",
        '''                keep_frame_for_output=bool(args.grid_video),
                trace_log_every=args.trace_log_every,
''',
        '''                keep_frame_for_output=bool(args.grid_video),
                shared_input_descs=shared_input_descs,
                shared_input_dtype=args.shared_input_dtype,
                trace_log_every=args.trace_log_every,
''',
        "pass shared input descs to latest camera worker",
    )
    text = replace_once_in_region(
        text,
        "def run_latest(args)",
        "def run(args)",
        '''                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,


                migraphx_batch_size=args.migraphx_batch_size,
''',
        '''                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,
                shared_input_descs=shared_input_descs,


                migraphx_batch_size=args.migraphx_batch_size,
''',
        "pass shared input descs to latest inference worker",
    )
    text = replace_once_in_region(
        text,
        "def run_latest(args)",
        "def run(args)",
        '''        close_shared_map_buffers(shared_map_handles)
''',
        '''        close_shared_map_buffers(shared_map_handles)
        close_shared_input_buffers(shared_input_handles)
''',
        "close shared input buffers in parent",
    )
    text = replace_once_in_region(
        text,
        "def run_latest(args)",
        "def run(args)",
        '''    summary["shared_map_slots"] = getattr(args, "shared_map_slots", 0)
''',
        '''    summary["shared_map_slots"] = getattr(args, "shared_map_slots", 0)
    summary["shared_input_slots"] = getattr(args, "shared_input_slots", 0)
    summary["shared_input_dtype"] = getattr(args, "shared_input_dtype", "float32")
''',
        "add shared input fields to latest summary",
    )

    # 6) Add CLI flags.
    text = replace_once(
        text,
        '''    parser.add_argument(
        "--shared-map-slots",
        type=int,
        default=0,
        help="Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.",
    )
''',
        '''    parser.add_argument(
        "--shared-map-slots",
        type=int,
        default=0,
        help="Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.",
    )
    parser.add_argument(
        "--shared-input-slots",
        type=int,
        default=0,
        help=(
            "Latest-mode only: preallocate this many shared-memory preprocessed input slots "
            "between camera/preprocess and inference. Use --num-cameras so each camera "
            "has one stable slot. 0 keeps old Queue pickle/copy behavior."
        ),
    )
    parser.add_argument(
        "--shared-input-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="dtype used for shared camera->inference input slots. float32 matches current preprocess output.",
    )
''',
        "add CLI flags",
    )

    if text == original:
        raise RuntimeError("Patch made no changes.")

    backup = path.with_suffix(path.suffix + ".bak_shared_input")
    backup.write_text(original)
    path.write_text(text)
    print(f"[OK] patched {path}")
    print(f"[OK] backup written to {backup}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="simulate_10_camera_stream.py",
        help="Path to simulate_10_camera_stream.py from repo root.",
    )
    args = parser.parse_args()
    patch_file(Path(args.path))


if __name__ == "__main__":
    main()
