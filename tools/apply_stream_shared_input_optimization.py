#!/usr/bin/env python3
"""
Patch simulate_10_camera_stream.py to use shared-memory input tensors in
latest-buffer mode.

Why this exists
---------------
The B4 MXR model benchmarks at ~85 ms/batch in isolation, but the full
10-camera simulator reported ~800 ms/batch. The likely culprit is that each
camera process currently sends a full preprocessed NumPy tensor through
multiprocessing.Queue. For 544x968 float32 NCHW input, that is ~6 MB per frame
and causes heavy pickling/copying/memory-bandwidth pressure.

This patch keeps the existing queue/scheduler logic, but changes latest-mode
camera -> inference transport when --shared-input-slots is enabled:

    camera process:
        preprocess frame -> write tensor into shared memory slot -> queue metadata

    inference process:
        queue metadata -> read tensor view from shared memory -> build MIGraphX batch

The old behavior remains the default when --shared-input-slots=0.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HELPERS = r'''

def create_shared_input_buffers(
    num_slots: int,
    target_h: int,
    target_w: int,
    dtype_name: str = "float32",
) -> Tuple[List[Dict[str, Any]], List[shared_memory.SharedMemory]]:
    """Create preprocessed input tensor shared-memory slots.

    Slots are shaped as 1x3xHxW because each camera publishes one already
    preprocessed frame at a time. In latest mode, one slot per camera is usually
    enough and keeps the queue item metadata-only.
    """
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


def _input_tensor_from_item(item: Dict[str, Any], shared_input_slots: Optional[Dict[int, Dict[str, Any]]] = None) -> np.ndarray:
    """Return the preprocessed 1x3xHxW tensor for an item.

    This keeps backwards compatibility with old Queue-pickled items while
    allowing metadata-only items to point at shared-memory slots.
    """
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


def patch_file(path: Path) -> bool:
    text = path.read_text()
    original = text

    if "create_shared_input_buffers" in text:
        print("[OK] shared-input optimization already appears to be applied.")
        return False

    # 1) Add shared-input helper functions near existing shared-map helpers.
    text = replace_once(
        text,
        "def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:\n"
        "    for shm in handles:\n"
        "        try:\n"
        "            shm.close()\n"
        "        except Exception:\n"
        "            pass\n",
        "def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:\n"
        "    for shm in handles:\n"
        "        try:\n"
        "            shm.close()\n"
        "        except Exception:\n"
        "            pass\n" + HELPERS + "\n",
        "insert shared input helpers",
    )

    # 2) Teach MIGraphX input batch builder to read shared-memory tensors.
    text = replace_once(
        text,
        "def make_migraphx_input_batch(\n"
        "    items: Sequence[Dict[str, Any]],\n"
        "    expected_dtype: str,\n"
        "    compiled_batch_size: int,\n"
        ") -> Tuple[np.ndarray, int]:\n",
        "def make_migraphx_input_batch(\n"
        "    items: Sequence[Dict[str, Any]],\n"
        "    expected_dtype: str,\n"
        "    compiled_batch_size: int,\n"
        "    shared_input_slots: Optional[Dict[int, Dict[str, Any]]] = None,\n"
        ") -> Tuple[np.ndarray, int]:\n",
        "extend make_migraphx_input_batch signature",
    )
    text = replace_once(
        text,
        "        x = np.asarray(item[\"input_tensor\"])\n",
        "        x = _input_tensor_from_item(item, shared_input_slots)\n",
        "read item tensor through shared input helper",
    )

    # 3) Add optional shared input descs to latest camera worker.
    text = replace_once(
        text,
        "    keep_frame_for_output: bool = False,\n"
        "    trace_log_every: int = 0,\n"
        "    roctx_enabled: bool = False,\n"
        ") -> None:\n"
        "    \"\"\"Camera worker that maintains a newest-frame-only slot for its camera.\"\"\"\n"
        "    try:\n"
        "        import cv2\n",
        "    keep_frame_for_output: bool = False,\n"
        "    shared_input_descs: Optional[Sequence[Dict[str, Any]]] = None,\n"
        "    shared_input_dtype: str = \"float32\",\n"
        "    trace_log_every: int = 0,\n"
        "    roctx_enabled: bool = False,\n"
        ") -> None:\n"
        "    \"\"\"Camera worker that maintains a newest-frame-only slot for its camera.\"\"\"\n"
        "    shared_input_slots = {}\n"
        "    shared_input_handles = []\n"
        "    try:\n"
        "        import cv2\n"
        "        shared_input_slots, shared_input_handles = open_shared_input_buffers(shared_input_descs)\n",
        "extend camera_preprocess_latest_worker signature and open shm",
    )

    text = replace_once(
        text,
        "                item[\"frame_bgr\"] = frame.copy()\n"
        "            replaced_before_infer += latest_put(q, item)\n",
        "                item[\"frame_bgr\"] = frame.copy()\n"
        "            if shared_input_slots:\n"
        "                slot = shared_input_slots[int(camera_id)][\"input\"]\n"
        "                if slot.shape != tensor.shape:\n"
        "                    raise ValueError(f\"shared input slot shape mismatch: {slot.shape}!={tensor.shape}\")\n"
        "                np.copyto(slot, tensor.astype(slot.dtype, copy=False), casting=\"same_kind\")\n"
        "                item.pop(\"input_tensor\", None)\n"
        "                item[\"shared_input_slot\"] = int(camera_id)\n"
        "            replaced_before_infer += latest_put(q, item)\n",
        "write camera tensor to shared input slot before enqueue",
    )

    text = replace_once(
        text,
        "    except Exception:\n"
        "        try:\n"
        "            camera_done[camera_id] = 1\n"
        "        except Exception:\n"
        "            pass\n"
        "        error_q.put({\"stage\": \"camera_preprocess\", \"camera_id\": camera_id, \"traceback\": traceback.format_exc()})\n"
        "\n\n"
        "def inference_latest_worker(",
        "    except Exception:\n"
        "        try:\n"
        "            camera_done[camera_id] = 1\n"
        "        except Exception:\n"
        "            pass\n"
        "        error_q.put({\"stage\": \"camera_preprocess\", \"camera_id\": camera_id, \"traceback\": traceback.format_exc()})\n"
        "    finally:\n"
        "        close_shared_map_views(shared_input_handles)\n"
        "\n\n"
        "def inference_latest_worker(",
        "close camera shared input views",
    )

    # 4) Add optional shared input descs to latest inference worker.
    text = replace_once(
        text,
        "    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,\n"
        "    free_map_slots=None,\n"
        "\n\n"
        "    migraphx_batch_size: int = 1,\n",
        "    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,\n"
        "    free_map_slots=None,\n"
        "    shared_input_descs: Optional[Sequence[Dict[str, Any]]] = None,\n"
        "\n\n"
        "    migraphx_batch_size: int = 1,\n",
        "extend inference_latest_worker signature",
    )
    text = replace_once(
        text,
        "        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)\n"
        "        shared_map_misses = 0\n",
        "        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)\n"
        "        shared_input_slots, shared_input_handles = open_shared_input_buffers(shared_input_descs)\n"
        "        shared_map_misses = 0\n",
        "open shared input slots in inference worker",
    )
    text = replace_once(
        text,
        "                compiled_batch_size=configured_batch_size,\n"
        "            )\n"
        "\n"
        "            with Timer() as t_inf:\n",
        "                compiled_batch_size=configured_batch_size,\n"
        "                shared_input_slots=shared_input_slots,\n"
        "            )\n"
        "\n"
        "            with Timer() as t_inf:\n",
        "pass shared input slots to make_migraphx_input_batch",
    )
    text = replace_once(
        text,
        "        close_shared_map_views(shared_handles)\n"
        "        print(\n"
        "            f\"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} \"\n",
        "        close_shared_map_views(shared_handles)\n"
        "        close_shared_map_views(shared_input_handles)\n"
        "        print(\n"
        "            f\"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} \"\n",
        "close inference shared input views on normal exit",
    )

    # 5) Create shared input buffers in run_latest and wire them into workers.
    text = replace_once(
        text,
        "    shared_map_descs: List[Dict[str, Any]] = []\n"
        "    shared_map_handles: List[shared_memory.SharedMemory] = []\n"
        "    free_map_slots = None\n",
        "    shared_input_descs: List[Dict[str, Any]] = []\n"
        "    shared_input_handles: List[shared_memory.SharedMemory] = []\n"
        "    if getattr(args, \"shared_input_slots\", 0) > 0:\n"
        "        shared_input_descs, shared_input_handles = create_shared_input_buffers(\n"
        "            int(args.shared_input_slots), args.target_height, args.target_width, args.shared_input_dtype\n"
        "        )\n"
        "\n"
        "    shared_map_descs: List[Dict[str, Any]] = []\n"
        "    shared_map_handles: List[shared_memory.SharedMemory] = []\n"
        "    free_map_slots = None\n",
        "create shared input buffers in run_latest",
    )
    text = replace_once(
        text,
        "                keep_frame_for_output=bool(args.grid_video),\n"
        "                trace_log_every=args.trace_log_every,\n",
        "                keep_frame_for_output=bool(args.grid_video),\n"
        "                shared_input_descs=shared_input_descs,\n"
        "                shared_input_dtype=args.shared_input_dtype,\n"
        "                trace_log_every=args.trace_log_every,\n",
        "pass shared input descs to camera worker",
    )
    text = replace_once(
        text,
        "                shared_map_descs=shared_map_descs,\n"
        "                free_map_slots=free_map_slots,\n"
        "\n\n"
        "                migraphx_batch_size=args.migraphx_batch_size,\n",
        "                shared_map_descs=shared_map_descs,\n"
        "                free_map_slots=free_map_slots,\n"
        "                shared_input_descs=shared_input_descs,\n"
        "\n\n"
        "                migraphx_batch_size=args.migraphx_batch_size,\n",
        "pass shared input descs to inference worker",
    )
    text = replace_once(
        text,
        "        close_shared_map_buffers(shared_map_handles)\n",
        "        close_shared_map_buffers(shared_map_handles)\n"
        "        close_shared_input_buffers(shared_input_handles)\n",
        "close shared input buffers in parent",
    )
    text = replace_once(
        text,
        "    summary[\"shared_map_slots\"] = getattr(args, \"shared_map_slots\", 0)\n",
        "    summary[\"shared_map_slots\"] = getattr(args, \"shared_map_slots\", 0)\n"
        "    summary[\"shared_input_slots\"] = getattr(args, \"shared_input_slots\", 0)\n"
        "    summary[\"shared_input_dtype\"] = getattr(args, \"shared_input_dtype\", \"float32\")\n",
        "add shared input fields to summary",
    )

    # 6) Add CLI args.
    text = replace_once(
        text,
        "    parser.add_argument(\n"
        "        \"--shared-map-slots\",\n"
        "        type=int,\n"
        "        default=0,\n"
        "        help=\"Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.\",\n"
        "    )\n",
        "    parser.add_argument(\n"
        "        \"--shared-map-slots\",\n"
        "        type=int,\n"
        "        default=0,\n"
        "        help=\"Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--shared-input-slots\",\n"
        "        type=int,\n"
        "        default=0,\n"
        "        help=(\n"
        "            \"Latest-mode only: preallocate this many shared-memory preprocessed input slots \\"\n"
        "            \"between camera/preprocess and inference. Use --num-cameras to give each camera \\"\n"
        "            \"one stable slot. 0 keeps old Queue pickle/copy behavior.\"\n"
        "        ),\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--shared-input-dtype\",\n"
        "        choices=[\"float32\", \"float16\"],\n"
        "        default=\"float32\",\n"
        "        help=\"dtype used for shared camera->inference input slots. float32 matches current preprocess output.\",\n"
        "    )\n",
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
