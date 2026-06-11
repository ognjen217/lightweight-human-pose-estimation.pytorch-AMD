"""Camera/preprocess worker process entrypoints."""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..camera import preprocess_frame
from ..queues import latest_put
from ..shared_memory import close_shared_map_views, open_shared_input_buffers
from ..system import configure_child_cpu_runtime
from ..tracing import RocTxTracer, trace_print
from ..utils import Timer, mean, percentile


def camera_preprocess_worker(
    *,
    camera_id: int,
    video_path: str,
    out_q,
    stats_q,
    error_q,
    stop_event,
    target_w: int,
    target_h: int,
    max_frames: int,
    duration_s: float,
    realtime: bool,
    camera_fps: float,
    queue_policy: str,
    keep_frame_for_output: bool = False,
    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    try:
        import cv2


        tracer = RocTxTracer(roctx_enabled, f"camera:{camera_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")


        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Camera {camera_id}: cannot find video {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {camera_id}: could not open video {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = float(camera_fps if camera_fps > 0 else (source_fps if source_fps > 0 else 24.0))
        period_s = 1.0 / fps if fps > 0 else 0.0
        next_frame_deadline = time.perf_counter()

        attempted = 0
        enqueued = 0
        dropped = 0
        loops = 0
        preprocess_times: List[float] = []
        t_worker_start = time.perf_counter()

        while not stop_event.is_set():
            if max_frames > 0 and attempted >= max_frames:
                break
            if duration_s > 0 and (time.perf_counter() - t_worker_start) >= duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                loops += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            attempted += 1
            capture_ts = time.perf_counter()
            original_h, original_w = frame.shape[:2]

            with Timer() as t_pre:
                with tracer.range("preprocess_frame"):
                    tensor = preprocess_frame(frame, target_w, target_h)
            preprocess_times.append(t_pre.ms)

            item = {
                "camera_id": camera_id,
                "frame_id": attempted,
                "source": video_path,
                "capture_ts": capture_ts,
                "preprocess_done_ts": time.perf_counter(),
                "original_hw": (int(original_h), int(original_w)),
                "preprocess_ms": float(t_pre.ms),
                "input_tensor": tensor,
            }
            if keep_frame_for_output:
                item["frame_bgr"] = frame.copy()

            if queue_policy == "block":
                out_q.put(item)
                enqueued += 1
            else:
                try:
                    out_q.put_nowait(item)
                    enqueued += 1
                except py_queue.Full:
                    dropped += 1

            trace_print(
                trace_log_every,
                attempted,
                f"[TRACE camera:{camera_id} pid={os.getpid()}] "
                f"frame={attempted} preprocess={t_pre.ms:.2f}ms enqueued={enqueued} dropped={dropped}",
            )

            if realtime and period_s > 0:
                next_frame_deadline += period_s
                sleep_s = next_frame_deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -period_s:
                    # If the pipeline falls behind heavily, resync instead of sleeping forever later.
                    next_frame_deadline = time.perf_counter()

        cap.release()
        stats_q.put(
            {
                "stage": "camera_preprocess",
                "camera_id": camera_id,
                "source": video_path,
                "attempted": attempted,
                "enqueued": enqueued,
                "dropped": dropped,
                "loops": loops,
                "avg_preprocess_ms": mean(preprocess_times),
                "p95_preprocess_ms": percentile(preprocess_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )

    except Exception:
        error_q.put({"stage": "camera_preprocess", "camera_id": camera_id, "traceback": traceback.format_exc()})

def camera_preprocess_latest_worker(
    *,
    camera_id: int,
    video_path: str,
    out_queues: Sequence[Any],
    camera_done,
    stats_q,
    error_q,
    stop_event,
    target_w: int,
    target_h: int,
    max_frames: int,
    duration_s: float,
    realtime: bool,
    camera_fps: float,
    keep_frame_for_output: bool = False,
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


        tracer = RocTxTracer(roctx_enabled, f"camera:{camera_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")


        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Camera {camera_id}: cannot find video {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {camera_id}: could not open video {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = float(camera_fps if camera_fps > 0 else (source_fps if source_fps > 0 else 24.0))
        period_s = 1.0 / fps if fps > 0 else 0.0
        next_frame_deadline = time.perf_counter()

        attempted = 0
        published = 0
        replaced_before_infer = 0
        loops = 0
        preprocess_times: List[float] = []
        t_worker_start = time.perf_counter()

        q = out_queues[camera_id]

        while not stop_event.is_set():
            if max_frames > 0 and attempted >= max_frames:
                break
            if duration_s > 0 and (time.perf_counter() - t_worker_start) >= duration_s:
                break

            ret, frame = cap.read()
            if not ret:
                loops += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            attempted += 1
            capture_ts = time.perf_counter()
            original_h, original_w = frame.shape[:2]

            with Timer() as t_pre:
                with tracer.range("preprocess_frame"):
                    tensor = preprocess_frame(frame, target_w, target_h)
            preprocess_times.append(t_pre.ms)

            item = {
                "camera_id": camera_id,
                "frame_id": attempted,
                "source": video_path,
                "capture_ts": capture_ts,
                "preprocess_done_ts": time.perf_counter(),
                "original_hw": (int(original_h), int(original_w)),
                "preprocess_ms": float(t_pre.ms),
                "input_tensor": tensor,
            }
            if keep_frame_for_output:
                item["frame_bgr"] = frame.copy()
            if shared_input_slots:
                slot = shared_input_slots[int(camera_id)]["input"]
                if slot.shape != tensor.shape:
                    raise ValueError(f"shared input slot shape mismatch: {slot.shape}!={tensor.shape}")
                np.copyto(slot, tensor.astype(slot.dtype, copy=False), casting="same_kind")
                item.pop("input_tensor", None)
                item["shared_input_slot"] = int(camera_id)
            replaced_before_infer += latest_put(q, item)
            published += 1
            trace_print(
                trace_log_every,
                attempted,
                f"[TRACE camera:{camera_id} pid={os.getpid()} latest] "
                f"frame={attempted} preprocess={t_pre.ms:.2f}ms published={published} replaced={replaced_before_infer}",
            )

            if realtime and period_s > 0:
                next_frame_deadline += period_s
                sleep_s = next_frame_deadline - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif sleep_s < -period_s:
                    next_frame_deadline = time.perf_counter()

        cap.release()
        camera_done[camera_id] = 1
        stats_q.put(
            {
                "stage": "camera_preprocess",
                "buffer_mode": "latest",
                "camera_id": camera_id,
                "source": video_path,
                "attempted": attempted,
                "published": published,
                "enqueued": published,
                "dropped": replaced_before_infer,
                "replaced_before_infer": replaced_before_infer,
                "loops": loops,
                "avg_preprocess_ms": mean(preprocess_times),
                "p95_preprocess_ms": percentile(preprocess_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )

    except Exception:
        try:
            camera_done[camera_id] = 1
        except Exception:
            pass
        error_q.put({"stage": "camera_preprocess", "camera_id": camera_id, "traceback": traceback.format_exc()})
    finally:
        close_shared_map_views(shared_input_handles)
