#!/usr/bin/env python3
"""
simulate_10_camera_stream.py

Multi-camera live-feed simulator for the lightweight-human-pose-estimation.pytorch-AMD
MIGraphX + postprocessing pipeline.

Architecture
------------
Camera/preprocess workers:
    - one process per simulated camera by default
    - read one of the CCTV videos in a loop
    - resize/normalize/transpose frames into NCHW float32 tensors
    - push frames into either FIFO queue mode or newest-frame-only per-camera slots

MIGraphX inference workers:
    - separate process group that imports/uses MIGraphX only
    - loads the .mxr model
    - casts preprocessed tensors to the model input dtype
    - runs inference
    - decodes heatmaps/PAFs to low-resolution HWC arrays
    - pushes decoded maps into FIFO queues or newest-frame-only per-camera postprocess slots
    - in latest mode, optional backpressure prevents inference from producing
      another result for a camera while that camera already has a pending
      postprocess result

Postprocess workers:
    - separate process group that imports modules.postprocessing
    - if the selected variant uses Torch/GPU, Torch ROCm is initialized only here
    - calls postprocess_from_maps(...) with the selected variant/config
    - returns timing/stat rows to the parent process

This layout intentionally keeps MIGraphX and PyTorch ROCm in different Python
processes when GPU postprocessing is selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue as py_queue
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp

import numpy as np


DEFAULT_VIDEO_CYCLE = [
    "cctv_1280x720_24fps_1.mp4",
    "cctv_1280x720_24fps_original.mp4",
    "cctv_1280x720_24fps_3.mp4",
    "cctv_1280x720_24fps_2.mp4",
]


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *args):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def ensure_parent(path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def camera_sources(num_cameras: int, videos: Sequence[str]) -> List[str]:
    """Return the default 10-camera mapping requested in the prompt.

    For 10 cameras:
      0 -> video 1
      1 -> video 2
      2 -> video 3
      3 -> original
      4 -> video 1
      5 -> video 2
      6 -> video 3
      7 -> original
      8 -> video 3
      9 -> original

    For more than 10 cameras, continue round-robin.
    """
    if len(videos) < 4:
        raise ValueError("At least four video paths are required.")

    out: List[str] = []
    for cam_id in range(num_cameras):
        if cam_id == 8:
            out.append(videos[2])
        elif cam_id == 9:
            out.append(videos[3])
        else:
            out.append(videos[cam_id % len(videos)])
    return out


def preprocess_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    import cv2

    img = cv2.resize(frame, (target_w, target_h))
    img = (img.astype(np.float32) - 128.0) / 256.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(img, dtype=np.float32)


def cast_for_migraphx(expected_dtype: str, tensor: np.ndarray) -> np.ndarray:
    if "half" in expected_dtype:
        return np.ascontiguousarray(tensor.astype(np.float16, copy=False))
    # MIGraphX Python inputs are safest as fp32 for fp32/bf16 cases.
    return np.ascontiguousarray(tensor.astype(np.float32, copy=False))


def decode_migraphx_outputs(results: Any, out_h: int, out_w: int, output_dtype: str) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) < 2:
        raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")

    heatmaps = np.asarray(results[-2], dtype=np.float32).reshape(19, out_h, out_w)
    pafs = np.asarray(results[-1], dtype=np.float32).reshape(38, out_h, out_w)

    heatmaps = np.moveaxis(heatmaps, 0, -1)
    pafs = np.moveaxis(pafs, 0, -1)

    if output_dtype == "float16":
        return (
            np.ascontiguousarray(heatmaps, dtype=np.float16),
            np.ascontiguousarray(pafs, dtype=np.float16),
        )
    return (
        np.ascontiguousarray(heatmaps, dtype=np.float32),
        np.ascontiguousarray(pafs, dtype=np.float32),
    )


def resolve_registry_mode(user_mode: str) -> Tuple[str, str, bool]:
    """Map public CLI variant to the actual mode used by postprocess_from_maps.

    postprocess_from_maps intentionally rejects *_two_process aliases because in
    speed/accuracy validators those are handled by a special runner. In this
    script the process split is already provided by the architecture, so the
    worker maps the alias back to the underlying map-based registry mode.
    """
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




# ---------------------------------------------------------------------------
# Optional security-monitor grid video output
# ---------------------------------------------------------------------------
def draw_poses_on_frame(frame: np.ndarray, pose_entries: np.ndarray, all_keypoints: np.ndarray) -> None:
    """Draw skeletons returned by postprocess_from_maps on a BGR frame in-place."""
    if pose_entries is None or all_keypoints is None or len(all_keypoints) == 0:
        return

    try:
        import cv2
        from modules.keypoints import BODY_PARTS_KPT_IDS
    except Exception:
        return

    for pose in pose_entries:
        for part_id in range(len(BODY_PARTS_KPT_IDS)):
            kpt_a_id = pose[BODY_PARTS_KPT_IDS[part_id][0]]
            kpt_b_id = pose[BODY_PARTS_KPT_IDS[part_id][1]]
            if kpt_a_id != -1 and kpt_b_id != -1:
                kpt_a = all_keypoints[int(kpt_a_id)]
                kpt_b = all_keypoints[int(kpt_b_id)]
                cv2.line(
                    frame,
                    (int(kpt_a[0]), int(kpt_a[1])),
                    (int(kpt_b[0]), int(kpt_b[1])),
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

        for kpt_id in pose[:-2]:
            if kpt_id != -1:
                kpt = all_keypoints[int(kpt_id)]
                cv2.circle(
                    frame,
                    (int(kpt[0]), int(kpt[1])),
                    3,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA,
                )


def make_monitor_grid_frame(
    *,
    latest_frames: Dict[int, Dict[str, Any]],
    num_cameras: int,
    grid_rows: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
    camera_sources_: Sequence[str],
) -> np.ndarray:
    """Create one BGR 4x4-like monitor frame from latest per-camera outputs."""
    import cv2

    grid = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for cam_id in range(min(num_cameras, grid_rows * grid_cols)):
        r = cam_id // grid_cols
        c = cam_id % grid_cols
        y0 = r * cell_h
        x0 = c * cell_w
        tile = grid[y0 : y0 + cell_h, x0 : x0 + cell_w]
        packet = latest_frames.get(cam_id)

        if packet is not None and packet.get("frame_bgr") is not None:
            frame = packet["frame_bgr"]
            try:
                resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                tile[:] = resized
            except Exception:
                tile[:] = 0
        else:
            tile[:] = 18
            cv2.putText(tile, "NO SIGNAL", (16, cell_h // 2), font, 0.8, (120, 120, 120), 2, cv2.LINE_AA)

        # Dark label strip for readability.
        cv2.rectangle(tile, (0, 0), (cell_w, 42), (0, 0, 0), -1)
        source_name = Path(camera_sources_[cam_id]).name if cam_id < len(camera_sources_) else ""
        if packet is None:
            label1 = f"CAM {cam_id:02d}"
            label2 = source_name
        else:
            label1 = (
                f"CAM {cam_id:02d}  f={int(packet.get('frame_id', 0))}  "
                f"poses={int(packet.get('num_poses', 0))}"
            )
            label2 = f"e2e={safe_float(packet.get('e2e_ms', 0.0)):.0f}ms  {source_name}"

        cv2.putText(tile, label1[:48], (8, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, label2[:58], (8, 34), font, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (70, 70, 70), 1)

    return grid


def grid_video_writer_worker(
    *,
    grid_q,
    output_path: str,
    num_cameras: int,
    grid_rows: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
    fps: float,
    codec: str,
    camera_sources_: Sequence[str],
    stop_event,
    stats_q,
    error_q,
) -> None:
    """Write a single security-monitor-style concatenated grid video.

    The writer receives already-drawn per-camera frames from postprocess workers,
    keeps the newest frame per camera, and writes a fixed-rate grid video.

    Important for MP4: the process must exit cleanly and call
    VideoWriter.release(); otherwise ffprobe reports "moov atom not found".
    """
    writer = None
    released = False
    try:
        import cv2

        ensure_parent(output_path)
        fps = float(fps if fps > 0 else 10.0)
        period_s = 1.0 / fps
        grid_w = int(grid_cols * cell_w)
        grid_h = int(grid_rows * cell_h)
        fourcc_text = (codec or "mp4v")[:4]
        if len(fourcc_text) < 4:
            fourcc_text = "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open grid video writer: {output_path}")

        latest_frames: Dict[int, Dict[str, Any]] = {}
        packets_received = 0
        frames_written = 0
        first_packet_ts: Optional[float] = None
        t0 = time.perf_counter()
        next_write_ts = t0

        print(
            f"[GRID] Writing monitor video: {output_path} "
            f"({grid_cols}x{grid_rows}, {grid_w}x{grid_h}, {fps:.2f} FPS)",
            flush=True,
        )

        # Write one initial frame immediately. This makes the output container
        # valid even if no postprocess packet ever arrives, and it also makes
        # debugging easier because ffprobe/ffmpeg can still open the file.
        initial_frame = make_monitor_grid_frame(
            latest_frames=latest_frames,
            num_cameras=num_cameras,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            cell_w=cell_w,
            cell_h=cell_h,
            camera_sources_=camera_sources_,
        )
        writer.write(initial_frame)
        frames_written += 1

        while True:
            drained = 0
            while drained < 256:
                try:
                    packet = grid_q.get_nowait()
                except py_queue.Empty:
                    break
                cam_id = int(packet.get("camera_id", -1))
                if 0 <= cam_id < num_cameras:
                    latest_frames[cam_id] = packet
                    packets_received += 1
                    if first_packet_ts is None:
                        first_packet_ts = time.perf_counter()
                drained += 1

            now = time.perf_counter()
            if latest_frames and now >= next_write_ts:
                frame = make_monitor_grid_frame(
                    latest_frames=latest_frames,
                    num_cameras=num_cameras,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    camera_sources_=camera_sources_,
                )
                writer.write(frame)
                frames_written += 1
                next_write_ts += period_s
                if next_write_ts < now - period_s:
                    next_write_ts = now + period_s

            if stop_event.is_set():
                # Drain remaining packets once, write a final frame, and exit.
                if grid_q.empty():
                    if latest_frames:
                        frame = make_monitor_grid_frame(
                            latest_frames=latest_frames,
                            num_cameras=num_cameras,
                            grid_rows=grid_rows,
                            grid_cols=grid_cols,
                            cell_w=cell_w,
                            cell_h=cell_h,
                            camera_sources_=camera_sources_,
                        )
                        writer.write(frame)
                        frames_written += 1
                    break

            time.sleep(0.002)

        writer.release()
        released = True
        wall_s = time.perf_counter() - t0
        stats_q.put(
            {
                "stage": "grid_video_writer",
                "output_path": output_path,
                "packets_received": packets_received,
                "frames_written": frames_written,
                "fps": fps,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "cell_w": cell_w,
                "cell_h": cell_h,
                "wall_s": wall_s,
            }
        )
        print(
            f"[GRID] Done. packets={packets_received} frames_written={frames_written} output={output_path}",
            flush=True,
        )

    except Exception:
        error_q.put({"stage": "grid_video_writer", "traceback": traceback.format_exc()})
    finally:
        # Always try to finalize the container. Without this, MP4 output may
        # exist but be unreadable because the moov atom was never written.
        try:
            if writer is not None and not released:
                writer.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------
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
) -> None:
    try:
        import cv2

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


def inference_worker(
    *,
    worker_id: int,
    model_path: str,
    in_q,
    out_q,
    stats_q,
    error_q,
    target_w: int,
    target_h: int,
    stride: int,
    shared_dtype: str,
) -> None:
    try:
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)

        out_h = target_h // stride
        out_w = target_w // stride
        processed = 0
        inference_times: List[float] = []
        decode_times: List[float] = []
        queue_wait_times: List[float] = []
        skipped_due_backpressure = 0
        backpressure_idle_loops = 0
        t_worker_start = time.perf_counter()

        while True:
            item = in_q.get()
            if item is None:
                break

            infer_start = time.perf_counter()
            queue_wait_times.append((infer_start - float(item.get("preprocess_done_ts", infer_start))) * 1000.0)

            input_tensor = cast_for_migraphx(expected_dtype, item["input_tensor"])

            with Timer() as t_inf:
                results = model.run({"input": input_tensor})
            inference_times.append(t_inf.ms)

            with Timer() as t_dec:
                heatmaps, pafs = decode_migraphx_outputs(results, out_h, out_w, shared_dtype)
            decode_times.append(t_dec.ms)

            out_item = {
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "capture_ts": float(item["capture_ts"]),
                "preprocess_done_ts": float(item["preprocess_done_ts"]),
                "infer_done_ts": time.perf_counter(),
                "original_hw": tuple(item["original_hw"]),
                "preprocess_ms": float(item["preprocess_ms"]),
                "queue_pre_to_infer_ms": queue_wait_times[-1],
                "inference_ms": float(t_inf.ms),
                "decode_ms": float(t_dec.ms),
                "heatmaps": heatmaps,
                "pafs": pafs,
            }
            if "frame_bgr" in item:
                out_item["frame_bgr"] = item["frame_bgr"]
            out_q.put(out_item)
            processed += 1

        stats_q.put(
            {
                "stage": "inference",
                "worker_id": worker_id,
                "processed": processed,
                "avg_queue_pre_to_infer_ms": mean(queue_wait_times),
                "p95_queue_pre_to_infer_ms": percentile(queue_wait_times, 95),
                "avg_inference_ms": mean(inference_times),
                "p95_inference_ms": percentile(inference_times, 95),
                "avg_decode_ms": mean(decode_times),
                "p95_decode_ms": percentile(decode_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(f"[INFER:{worker_id}] Done. processed={processed}", flush=True)

    except Exception:
        error_q.put({"stage": "inference", "worker_id": worker_id, "traceback": traceback.format_exc()})


def postprocess_worker(
    *,
    worker_id: int,
    user_variant: str,
    in_q,
    result_q,
    stats_q,
    error_q,
    torch_device: str,
    require_gpu: bool,
    max_keypoints: int,
    threshold: float,
    nms_radius_fullres: int,
    nms_radius_lowres: int,
    nms_impl: str,
    gpu_compute_dtype: str,
    grid_q=None,
    render_output: bool = False,
    migraphx_nms_mxr: str = "",
    migraphx_nms_cache_dir: str = "",
) -> None:
    try:
        canonical, registry_mode, wants_torch = resolve_registry_mode(user_variant)

        if wants_torch:
            import torch

            print(f"[POST:{worker_id}] Initializing PyTorch ROCm/CUDA for {canonical}...", flush=True)
            print(f"[POST:{worker_id}] torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
            if torch_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested --torch-device cuda, but torch.cuda.is_available() is False")
            if torch_device == "cuda":
                warm = torch.empty((1,), device="cuda")
                warm += 1
                torch.cuda.synchronize()
                print(f"[POST:{worker_id}] Torch GPU name: {torch.cuda.get_device_name(0)}", flush=True)

        from modules.postprocessing import PostprocessConfig, postprocess_from_maps

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=torch_device,
            require_gpu=bool(require_gpu and wants_torch and torch_device == "cuda"),
            extra={
                "gpu_compute_dtype": gpu_compute_dtype,
                "nms_impl": nms_impl,
                "migraphx_nms_mxr": migraphx_nms_mxr,
                "migraphx_nms_cache_dir": migraphx_nms_cache_dir,
            },
        )

        # MIGraphX NMS postprocess resolver in modules.postprocessing expects
        # these as direct config attributes, not only inside config.extra.
        config.migraphx_nms_mxr = migraphx_nms_mxr
        config.migraphx_nms_cache_dir = migraphx_nms_cache_dir

        print(
            f"[POST:{worker_id}] user_variant={canonical} registry_mode={registry_mode} "
            f"nms_impl={nms_impl} gpu_dtype={gpu_compute_dtype}",
            flush=True,
        )

        processed = 0
        post_times: List[float] = []
        queue_wait_times: List[float] = []
        e2e_times: List[float] = []
        t_worker_start = time.perf_counter()

        while True:
            item = in_q.get()
            if item is None:
                break

            post_start = time.perf_counter()
            queue_wait_ms = (post_start - float(item.get("infer_done_ts", post_start))) * 1000.0
            queue_wait_times.append(queue_wait_ms)

            if registry_mode in {"migraphx_nms", "migraphx_nms_k20"}:
                selected_mxr = select_migraphx_nms_mxr_for_hw(
                    original_hw=tuple(item["original_hw"]),
                    migraphx_nms_mxr=migraphx_nms_mxr,
                    migraphx_nms_cache_dir=migraphx_nms_cache_dir,
                )
                if not selected_mxr or not Path(selected_mxr).exists():
                    raise FileNotFoundError(
                        "Missing MIGraphX NMS .mxr for stream resolution. "
                        f"original_hw={tuple(item['original_hw'])}, expected={selected_mxr}. "
                        "Run: python -m modules.migraphx_compiler --video <video> "
                        "--output-dir models/nms_fullres_cache"
                    )
                config.extra["migraphx_nms_mxr"] = selected_mxr

            out = postprocess_from_maps(
                registry_mode,
                item["heatmaps"],
                item["pafs"],
                tuple(item["original_hw"]),
                config=config,
            )
            post_done = time.perf_counter()

            timings = dict(out.timings)
            post_ms = float(timings.get("total_postprocess", (post_done - post_start) * 1000.0))
            e2e_ms = (post_done - float(item["capture_ts"])) * 1000.0
            post_times.append(post_ms)
            e2e_times.append(e2e_ms)

            row: Dict[str, Any] = {
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "variant": canonical,
                "registry_mode": registry_mode,
                "post_worker_id": worker_id,
                "preprocess_ms": float(item["preprocess_ms"]),
                "queue_pre_to_infer_ms": float(item["queue_pre_to_infer_ms"]),
                "inference_ms": float(item["inference_ms"]),
                "decode_ms": float(item["decode_ms"]),
                "queue_infer_to_post_ms": float(queue_wait_ms),
                "post_ms": post_ms,
                "e2e_ms": e2e_ms,
                "num_poses": int(len(out.pose_entries)) if out.pose_entries is not None else 0,
                "num_keypoints": int(len(out.all_keypoints)) if out.all_keypoints is not None else 0,
            }
            for key, value in timings.items():
                row[f"timing_{key}"] = safe_float(value)

            if render_output and grid_q is not None and "frame_bgr" in item:
                frame_out = item["frame_bgr"].copy()
                draw_poses_on_frame(frame_out, out.pose_entries, out.all_keypoints)
                packet = {
                    "camera_id": int(item["camera_id"]),
                    "frame_id": int(item["frame_id"]),
                    "source": item["source"],
                    "frame_bgr": frame_out,
                    "e2e_ms": e2e_ms,
                    "post_ms": post_ms,
                    "num_poses": row["num_poses"],
                    "num_keypoints": row["num_keypoints"],
                }
                try:
                    grid_q.put_nowait(packet)
                except py_queue.Full:
                    pass

            result_q.put(row)
            processed += 1

        stats_q.put(
            {
                "stage": "postprocess",
                "worker_id": worker_id,
                "variant": canonical,
                "registry_mode": registry_mode,
                "processed": processed,
                "avg_queue_infer_to_post_ms": mean(queue_wait_times),
                "p95_queue_infer_to_post_ms": percentile(queue_wait_times, 95),
                "avg_post_ms": mean(post_times),
                "p95_post_ms": percentile(post_times, 95),
                "avg_e2e_ms": mean(e2e_times),
                "p95_e2e_ms": percentile(e2e_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(f"[POST:{worker_id}] Done. processed={processed}", flush=True)

    except Exception:
        error_q.put({"stage": "postprocess", "worker_id": worker_id, "traceback": traceback.format_exc()})



# ---------------------------------------------------------------------------
# Latest-frame buffer worker processes
# ---------------------------------------------------------------------------
def latest_put(q, item) -> int:
    """Put newest item into a maxsize=1 queue, replacing the previous item if needed.

    Returns 1 when an older item had to be discarded.
    """
    dropped = 0
    try:
        q.put_nowait(item)
        return dropped
    except py_queue.Full:
        pass

    try:
        q.get_nowait()
        dropped = 1
    except py_queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except py_queue.Full:
        # Rare race if another producer filled it; keep newest semantics by dropping one more.
        try:
            q.get_nowait()
            dropped = 1
        except py_queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except py_queue.Full:
            dropped = 1
    return dropped


def all_done(done_flags) -> bool:
    try:
        return all(bool(v) for v in done_flags[:])
    except Exception:
        return False


def all_queues_empty(queues: Sequence[Any]) -> bool:
    for q in queues:
        try:
            if not q.empty():
                return False
        except Exception:
            return False
    return True


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
) -> None:
    """Camera worker that maintains a newest-frame-only slot for its camera."""
    try:
        import cv2

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
            replaced_before_infer += latest_put(q, item)
            published += 1

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


def inference_latest_worker(
    *,
    worker_id: int,
    model_path: str,
    in_queues: Sequence[Any],
    out_queues: Sequence[Any],
    camera_done,
    infer_done,
    post_pending,
    backpressure: bool,
    stats_q,
    error_q,
    target_w: int,
    target_h: int,
    stride: int,
    shared_dtype: str,
    poll_sleep_s: float = 0.001,
    migraphx_nms_mxr: str = "",
    migraphx_nms_cache_dir: str = "",
) -> None:
    """Round-robin MIGraphX worker over newest-frame slots, one slot per camera.

    Backpressure mode prevents wasteful inference: a camera is eligible for
    inference only when it does not already have a queued/in-flight
    postprocess result. This keeps MIGraphX output count close to final output
    count instead of repeatedly overwriting unprocessed heatmap/PAF maps.
    """
    try:
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)

        out_h = target_h // stride
        out_w = target_w // stride
        ncam = len(in_queues)
        next_cam = worker_id % max(1, ncam)

        processed = 0
        replaced_before_post = 0
        skipped_due_backpressure = 0
        backpressure_idle_loops = 0
        inference_times: List[float] = []
        decode_times: List[float] = []
        queue_wait_times: List[float] = []
        t_worker_start = time.perf_counter()

        while True:
            item = None
            scanned = 0
            skipped_this_scan = 0
            while scanned < ncam:
                cam_id = next_cam
                next_cam = (next_cam + 1) % ncam
                scanned += 1

                if backpressure and post_pending is not None and bool(post_pending[cam_id]):
                    skipped_due_backpressure += 1
                    skipped_this_scan += 1
                    continue

                try:
                    item = in_queues[cam_id].get_nowait()
                    break
                except py_queue.Empty:
                    continue

            if item is None:
                if all_done(camera_done) and all_queues_empty(in_queues):
                    break
                if skipped_this_scan > 0:
                    backpressure_idle_loops += 1
                time.sleep(poll_sleep_s)
                continue

            infer_start = time.perf_counter()
            queue_wait_times.append((infer_start - float(item.get("preprocess_done_ts", infer_start))) * 1000.0)

            input_tensor = cast_for_migraphx(expected_dtype, item["input_tensor"])

            with Timer() as t_inf:
                results = model.run({"input": input_tensor})
            inference_times.append(t_inf.ms)

            with Timer() as t_dec:
                heatmaps, pafs = decode_migraphx_outputs(results, out_h, out_w, shared_dtype)
            decode_times.append(t_dec.ms)

            out_item = {
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "capture_ts": float(item["capture_ts"]),
                "preprocess_done_ts": float(item["preprocess_done_ts"]),
                "infer_done_ts": time.perf_counter(),
                "original_hw": tuple(item["original_hw"]),
                "preprocess_ms": float(item["preprocess_ms"]),
                "queue_pre_to_infer_ms": queue_wait_times[-1],
                "inference_ms": float(t_inf.ms),
                "decode_ms": float(t_dec.ms),
                "heatmaps": heatmaps,
                "pafs": pafs,
            }
            if "frame_bgr" in item:
                out_item["frame_bgr"] = item["frame_bgr"]
            cam_id = int(item["camera_id"])
            if backpressure and post_pending is not None:
                # Mark this camera as having an outstanding postprocess result
                # before publishing the item, so another inference worker cannot
                # pick the same camera in the small publication window.
                post_pending[cam_id] = 1
                try:
                    out_queues[cam_id].put_nowait(out_item)
                except py_queue.Full:
                    # Should be rare when post_pending is respected. Keep the
                    # script robust and preserve newest semantics if the queue
                    # state and flag ever get out of sync.
                    replaced_before_post += latest_put(out_queues[cam_id], out_item)
            else:
                replaced_before_post += latest_put(out_queues[cam_id], out_item)
            processed += 1

        infer_done[worker_id] = 1
        stats_q.put(
            {
                "stage": "inference",
                "buffer_mode": "latest",
                "worker_id": worker_id,
                "processed": processed,
                "replaced_before_post": replaced_before_post,
                "backpressure_enabled": bool(backpressure),
                "skipped_due_backpressure": skipped_due_backpressure,
                "backpressure_idle_loops": backpressure_idle_loops,
                "avg_queue_pre_to_infer_ms": mean(queue_wait_times),
                "p95_queue_pre_to_infer_ms": percentile(queue_wait_times, 95),
                "avg_inference_ms": mean(inference_times),
                "p95_inference_ms": percentile(inference_times, 95),
                "avg_decode_ms": mean(decode_times),
                "p95_decode_ms": percentile(decode_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(
            f"[INFER:{worker_id}] Done. processed={processed} "
            f"replaced_before_post={replaced_before_post} "
            f"backpressure_skips={skipped_due_backpressure}",
            flush=True,
        )

    except Exception:
        try:
            infer_done[worker_id] = 1
        except Exception:
            pass
        error_q.put({"stage": "inference", "worker_id": worker_id, "traceback": traceback.format_exc()})


def postprocess_latest_worker(
    *,
    worker_id: int,
    user_variant: str,
    in_queues: Sequence[Any],
    infer_done,
    post_pending,
    result_q,
    stats_q,
    error_q,
    torch_device: str,
    require_gpu: bool,
    max_keypoints: int,
    threshold: float,
    nms_radius_fullres: int,
    nms_radius_lowres: int,
    nms_impl: str,
    gpu_compute_dtype: str,
    grid_q=None,
    render_output: bool = False,
    poll_sleep_s: float = 0.001,
    migraphx_nms_mxr: str = "",
    migraphx_nms_cache_dir: str = "",
) -> None:
    """Round-robin postprocess worker over newest decoded-map slots, one slot per camera."""
    try:
        canonical, registry_mode, wants_torch = resolve_registry_mode(user_variant)

        if wants_torch:
            import torch

            print(f"[POST:{worker_id}] Initializing PyTorch ROCm/CUDA for {canonical}...", flush=True)
            print(f"[POST:{worker_id}] torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
            if torch_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested --torch-device cuda, but torch.cuda.is_available() is False")
            if torch_device == "cuda":
                warm = torch.empty((1,), device="cuda")
                warm += 1
                torch.cuda.synchronize()
                print(f"[POST:{worker_id}] Torch GPU name: {torch.cuda.get_device_name(0)}", flush=True)

        from modules.postprocessing import PostprocessConfig, postprocess_from_maps

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=torch_device,
            require_gpu=bool(require_gpu and wants_torch and torch_device == "cuda"),
            extra={
                "gpu_compute_dtype": gpu_compute_dtype,
                "nms_impl": nms_impl,
                "migraphx_nms_mxr": migraphx_nms_mxr,
                "migraphx_nms_cache_dir": migraphx_nms_cache_dir,
            },
        )

        # MIGraphX NMS postprocess resolver in modules.postprocessing expects
        # these as direct config attributes, not only inside config.extra.
        config.migraphx_nms_mxr = migraphx_nms_mxr
        config.migraphx_nms_cache_dir = migraphx_nms_cache_dir

        print(
            f"[POST:{worker_id}] user_variant={canonical} registry_mode={registry_mode} "
            f"nms_impl={nms_impl} gpu_dtype={gpu_compute_dtype}",
            flush=True,
        )

        ncam = len(in_queues)
        next_cam = worker_id % max(1, ncam)
        processed = 0
        post_times: List[float] = []
        queue_wait_times: List[float] = []
        e2e_times: List[float] = []
        t_worker_start = time.perf_counter()

        while True:
            item = None
            scanned = 0
            while scanned < ncam:
                cam_id = next_cam
                next_cam = (next_cam + 1) % ncam
                scanned += 1
                try:
                    item = in_queues[cam_id].get_nowait()
                    break
                except py_queue.Empty:
                    continue

            if item is None:
                if all_done(infer_done) and all_queues_empty(in_queues):
                    break
                time.sleep(poll_sleep_s)
                continue

            post_start = time.perf_counter()
            queue_wait_ms = (post_start - float(item.get("infer_done_ts", post_start))) * 1000.0
            queue_wait_times.append(queue_wait_ms)

            if registry_mode in {"migraphx_nms", "migraphx_nms_k20"}:
                selected_mxr = select_migraphx_nms_mxr_for_hw(
                    original_hw=tuple(item["original_hw"]),
                    migraphx_nms_mxr=migraphx_nms_mxr,
                    migraphx_nms_cache_dir=migraphx_nms_cache_dir,
                )
                if not selected_mxr or not Path(selected_mxr).exists():
                    raise FileNotFoundError(
                        "Missing MIGraphX NMS .mxr for stream resolution. "
                        f"original_hw={tuple(item['original_hw'])}, expected={selected_mxr}. "
                        "Run: python -m modules.migraphx_compiler --video <video> "
                        "--output-dir models/nms_fullres_cache"
                    )
                config.extra["migraphx_nms_mxr"] = selected_mxr

            out = postprocess_from_maps(
                registry_mode,
                item["heatmaps"],
                item["pafs"],
                tuple(item["original_hw"]),
                config=config,
            )
            post_done = time.perf_counter()

            timings = dict(out.timings)
            post_ms = float(timings.get("total_postprocess", (post_done - post_start) * 1000.0))
            e2e_ms = (post_done - float(item["capture_ts"])) * 1000.0
            post_times.append(post_ms)
            e2e_times.append(e2e_ms)

            row: Dict[str, Any] = {
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "variant": canonical,
                "registry_mode": registry_mode,
                "post_worker_id": worker_id,
                "preprocess_ms": float(item["preprocess_ms"]),
                "queue_pre_to_infer_ms": float(item["queue_pre_to_infer_ms"]),
                "inference_ms": float(item["inference_ms"]),
                "decode_ms": float(item["decode_ms"]),
                "queue_infer_to_post_ms": float(queue_wait_ms),
                "post_ms": post_ms,
                "e2e_ms": e2e_ms,
                "num_poses": int(len(out.pose_entries)) if out.pose_entries is not None else 0,
                "num_keypoints": int(len(out.all_keypoints)) if out.all_keypoints is not None else 0,
            }
            for key, value in timings.items():
                row[f"timing_{key}"] = safe_float(value)

            if render_output and grid_q is not None and "frame_bgr" in item:
                # In latest-buffer mode the previous version forgot to publish
                # drawn frames to the grid writer, so the writer received zero
                # packets and MP4 output was left invalid/empty.
                frame_out = item["frame_bgr"].copy()
                draw_poses_on_frame(frame_out, out.pose_entries, out.all_keypoints)
                packet = {
                    "camera_id": int(item["camera_id"]),
                    "frame_id": int(item["frame_id"]),
                    "source": item["source"],
                    "frame_bgr": frame_out,
                    "e2e_ms": e2e_ms,
                    "post_ms": post_ms,
                    "num_poses": row["num_poses"],
                    "num_keypoints": row["num_keypoints"],
                }
                try:
                    grid_q.put_nowait(packet)
                except py_queue.Full:
                    # Grid output is only visualization; never block the benchmark.
                    pass

            result_q.put(row)
            if post_pending is not None:
                post_pending[int(item["camera_id"])] = 0
            processed += 1

        stats_q.put(
            {
                "stage": "postprocess",
                "buffer_mode": "latest",
                "worker_id": worker_id,
                "variant": canonical,
                "registry_mode": registry_mode,
                "processed": processed,
                "avg_queue_infer_to_post_ms": mean(queue_wait_times),
                "p95_queue_infer_to_post_ms": percentile(queue_wait_times, 95),
                "avg_post_ms": mean(post_times),
                "p95_post_ms": percentile(post_times, 95),
                "avg_e2e_ms": mean(e2e_times),
                "p95_e2e_ms": percentile(e2e_times, 95),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(f"[POST:{worker_id}] Done. processed={processed}", flush=True)

    except Exception:
        error_q.put({"stage": "postprocess", "worker_id": worker_id, "traceback": traceback.format_exc()})

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]], stage_stats: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    total = len(rows)
    cameras = sorted({int(r["camera_id"]) for r in rows})

    summary: Dict[str, Any] = {
        "total_processed_frames": total,
        "wall_s": wall_s,
        "aggregate_output_fps": total / wall_s if wall_s > 0 else 0.0,
        "active_cameras": len(cameras),
        "avg_output_fps_per_camera": (total / wall_s / len(cameras)) if wall_s > 0 and cameras else 0.0,
        "avg_preprocess_ms": mean([r["preprocess_ms"] for r in rows]),
        "avg_queue_pre_to_infer_ms": mean([r["queue_pre_to_infer_ms"] for r in rows]),
        "avg_inference_ms": mean([r["inference_ms"] for r in rows]),
        "avg_decode_ms": mean([r["decode_ms"] for r in rows]),
        "avg_queue_infer_to_post_ms": mean([r["queue_infer_to_post_ms"] for r in rows]),
        "avg_post_ms": mean([r["post_ms"] for r in rows]),
        "avg_e2e_ms": mean([r["e2e_ms"] for r in rows]),
        "p95_e2e_ms": percentile([r["e2e_ms"] for r in rows], 95),
        "p95_post_ms": percentile([r["post_ms"] for r in rows], 95),
        "stage_stats": stage_stats,
    }

    by_cam: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cam[int(row["camera_id"])].append(row)

    summary["per_camera"] = []
    for cam_id in sorted(by_cam):
        cam_rows = by_cam[cam_id]
        summary["per_camera"].append(
            {
                "camera_id": cam_id,
                "frames": len(cam_rows),
                "fps": len(cam_rows) / wall_s if wall_s > 0 else 0.0,
                "avg_e2e_ms": mean([r["e2e_ms"] for r in cam_rows]),
                "p95_e2e_ms": percentile([r["e2e_ms"] for r in cam_rows], 95),
                "avg_post_ms": mean([r["post_ms"] for r in cam_rows]),
                "source": cam_rows[0].get("source", ""),
            }
        )

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 150)
    print("10-CAMERA STREAM SIMULATION SUMMARY")
    print("=" * 150)
    print(f"Processed frames:          {summary['total_processed_frames']}")
    print(f"Wall time:                 {summary['wall_s']:.2f} s")
    print(f"Aggregate output FPS:      {summary['aggregate_output_fps']:.2f}")
    print(f"Avg output FPS / camera:   {summary['avg_output_fps_per_camera']:.2f}")
    print(f"Avg preprocess:            {summary['avg_preprocess_ms']:.2f} ms")
    print(f"Avg queue pre->infer:      {summary['avg_queue_pre_to_infer_ms']:.2f} ms")
    print(f"Avg inference:             {summary['avg_inference_ms']:.2f} ms")
    print(f"Avg decode:                {summary['avg_decode_ms']:.2f} ms")
    print(f"Avg queue infer->post:     {summary['avg_queue_infer_to_post_ms']:.2f} ms")
    print(f"Avg postprocess:           {summary['avg_post_ms']:.2f} ms")
    print(f"Avg E2E latency:           {summary['avg_e2e_ms']:.2f} ms")
    print(f"P95 E2E latency:           {summary['p95_e2e_ms']:.2f} ms")

    print("\nPer-camera output:")
    print(f"{'cam':>3} {'frames':>8} {'fps':>8} {'avg_e2e':>10} {'p95_e2e':>10} {'avg_post':>10} source")
    print("-" * 150)
    for cam in summary["per_camera"]:
        print(
            f"{cam['camera_id']:>3} {cam['frames']:>8} {cam['fps']:>8.2f} "
            f"{cam['avg_e2e_ms']:>10.2f} {cam['p95_e2e_ms']:>10.2f} "
            f"{cam['avg_post_ms']:>10.2f} {cam['source']}"
        )
    print("=" * 150)


def write_detailed_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path or not rows:
        return
    ensure_parent(path)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    preferred = [
        "camera_id",
        "frame_id",
        "source",
        "variant",
        "registry_mode",
        "post_worker_id",
        "preprocess_ms",
        "queue_pre_to_infer_ms",
        "inference_ms",
        "decode_ms",
        "queue_infer_to_post_ms",
        "post_ms",
        "e2e_ms",
        "num_poses",
        "num_keypoints",
    ]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Detailed CSV saved: {path}")


def write_summary_json(path: str, summary: Dict[str, Any]) -> None:
    if not path:
        return
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Summary JSON saved: {path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_queue(args) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")

    videos = args.videos or DEFAULT_VIDEO_CYCLE
    sources = camera_sources(args.num_cameras, videos)

    # Validate variant in the parent process without touching Torch CUDA.
    canonical, registry_mode, wants_torch = resolve_registry_mode(args.variant)
    compile_migraphx_nms_for_stream_if_requested(args, sources)

    pre_q = ctx.Queue(maxsize=max(1, int(args.preprocess_queue_size)))
    post_q = ctx.Queue(maxsize=max(1, int(args.postprocess_queue_size)))
    result_q = ctx.Queue()
    stats_q = ctx.Queue()
    error_q = ctx.Queue()
    stop_event = ctx.Event()
    grid_q = ctx.Queue(maxsize=max(1, int(args.grid_queue_size))) if args.grid_video else None
    grid_stop_event = ctx.Event() if args.grid_video else None

    camera_procs = []
    grid_procs = []
    infer_procs = []
    post_procs = []

    print("\nStarting multi-camera stream simulation")
    print("---------------------------------------")
    print(f"Variant:       {canonical}")
    print(f"Registry mode: {registry_mode}")
    print(f"Torch needed:  {wants_torch}")
    print(f"Model:         {args.model}")
    print(f"Cameras:       {args.num_cameras}")
    print(f"Infer workers: {args.infer_workers}")
    print(f"Post workers:  {args.post_workers}")
    print(f"Queue policy:  {args.queue_policy}")
    print(f"Buffer mode:   queue")
    print(f"Realtime:      {args.realtime}")
    if args.grid_video:
        print(f"Grid video:    {args.grid_video} ({args.grid_cols}x{args.grid_rows})")
    print("Camera sources:")
    for cam_id, src in enumerate(sources):
        print(f"  cam {cam_id:02d}: {src}")

    if args.grid_video:
        p = ctx.Process(
            target=grid_video_writer_worker,
            kwargs=dict(
                grid_q=grid_q,
                output_path=args.grid_video,
                num_cameras=args.num_cameras,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                cell_w=args.grid_cell_width,
                cell_h=args.grid_cell_height,
                fps=args.grid_video_fps,
                codec=args.grid_video_codec,
                camera_sources_=sources,
                stop_event=grid_stop_event,
                stats_q=stats_q,
                error_q=error_q,
            ),
            name="grid_video_writer",
        )
        p.start()
        grid_procs.append(p)

    for cam_id, src in enumerate(sources):
        p = ctx.Process(
            target=camera_preprocess_worker,
            kwargs=dict(
                camera_id=cam_id,
                video_path=src,
                out_q=pre_q,
                stats_q=stats_q,
                error_q=error_q,
                stop_event=stop_event,
                target_w=args.target_width,
                target_h=args.target_height,
                max_frames=args.frames_per_camera,
                duration_s=args.duration_s,
                realtime=args.realtime,
                camera_fps=args.camera_fps,
                queue_policy=args.queue_policy,
                keep_frame_for_output=bool(args.grid_video),
            ),
            name=f"camera_preprocess_{cam_id}",
        )
        p.start()
        camera_procs.append(p)

    for worker_id in range(args.infer_workers):
        p = ctx.Process(
            target=inference_worker,
            kwargs=dict(
                worker_id=worker_id,
                model_path=args.model,
                in_q=pre_q,
                out_q=post_q,
                stats_q=stats_q,
                error_q=error_q,
                target_w=args.target_width,
                target_h=args.target_height,
                stride=args.stride,
                shared_dtype=args.shared_dtype,
            ),
            name=f"migraphx_inference_{worker_id}",
        )
        p.start()
        infer_procs.append(p)

    for worker_id in range(args.post_workers):
        p = ctx.Process(
            target=postprocess_worker,
            kwargs=dict(
                worker_id=worker_id,
                user_variant=args.variant,
                in_q=post_q,
                result_q=result_q,
                stats_q=stats_q,
                error_q=error_q,
                torch_device="cuda" if args.torch_device == "auto" and wants_torch else args.torch_device,
                require_gpu=args.require_gpu,
                max_keypoints=args.max_keypoints,
                threshold=args.threshold,
                nms_radius_fullres=args.nms_radius_fullres,
                nms_radius_lowres=args.nms_radius_lowres,
                nms_impl=args.nms_impl,
                gpu_compute_dtype=args.gpu_compute_dtype,
                grid_q=grid_q,
                render_output=bool(args.grid_video),
                migraphx_nms_mxr=args.migraphx_nms_mxr,
                migraphx_nms_cache_dir=args.migraphx_nms_cache_dir,
            ),
            name=f"postprocess_{worker_id}",
        )
        p.start()
        post_procs.append(p)

    rows: List[Dict[str, Any]] = []
    stage_stats: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    sent_infer_stop = False
    sent_post_stop = False
    last_progress_print = 0

    try:
        while True:
            while not error_q.empty():
                err = error_q.get()
                raise RuntimeError(f"Worker failed: {err.get('stage')} {err.get('worker_id', err.get('camera_id', ''))}\n{err.get('traceback')}")

            # Drain result rows.
            while True:
                try:
                    row = result_q.get_nowait()
                except py_queue.Empty:
                    break
                rows.append(row)
                if args.print_every > 0 and len(rows) - last_progress_print >= args.print_every:
                    last_progress_print = len(rows)
                    elapsed = time.perf_counter() - t0
                    fps = len(rows) / elapsed if elapsed > 0 else 0.0
                    print(f"Processed output frames: {len(rows)} | elapsed={elapsed:.1f}s | aggregate FPS={fps:.2f}", flush=True)

            # Once all camera workers finish, close inference input.
            if not sent_infer_stop and all(not p.is_alive() for p in camera_procs):
                for _ in infer_procs:
                    pre_q.put(None)
                sent_infer_stop = True

            # Once inference workers finish, close postprocess input.
            if sent_infer_stop and not sent_post_stop and all(not p.is_alive() for p in infer_procs):
                for _ in post_procs:
                    post_q.put(None)
                sent_post_stop = True

            # Done when postprocess workers finish and queues have been drained.
            if sent_post_stop and all(not p.is_alive() for p in post_procs):
                break

            time.sleep(0.05)

        if args.grid_video and grid_stop_event is not None:
            grid_stop_event.set()
            for p in grid_procs:
                p.join(timeout=5.0)

        # Drain any remaining rows/stats.
        if args.grid_video and grid_stop_event is not None:
            grid_stop_event.set()
            for p in grid_procs:
                p.join(timeout=5.0)

        while True:
            try:
                rows.append(result_q.get_nowait())
            except py_queue.Empty:
                break

        while True:
            try:
                stage_stats.append(stats_q.get_nowait())
            except py_queue.Empty:
                break

        for p in camera_procs + infer_procs + post_procs + grid_procs:
            p.join(timeout=2.0)

    except KeyboardInterrupt:
        print("Interrupted; stopping workers...", flush=True)
        stop_event.set()
        raise
    finally:
        if args.grid_video and grid_stop_event is not None:
            grid_stop_event.set()
        for p in camera_procs + infer_procs + post_procs + grid_procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    wall_s = time.perf_counter() - t0
    summary = summarize(rows, stage_stats, wall_s)
    summary["variant"] = canonical
    summary["registry_mode"] = registry_mode
    summary["model"] = args.model
    summary["num_cameras"] = args.num_cameras
    summary["infer_workers"] = args.infer_workers
    summary["post_workers"] = args.post_workers
    summary["queue_policy"] = args.queue_policy
    summary["buffer_mode"] = "queue"
    summary["grid_video"] = args.grid_video
    summary["grid_rows"] = args.grid_rows if args.grid_video else 0
    summary["grid_cols"] = args.grid_cols if args.grid_video else 0
    summary["realtime"] = args.realtime
    summary["camera_sources"] = sources

    print_summary(summary)
    write_detailed_csv(args.detailed_csv, rows)
    write_summary_json(args.summary_json, summary)
    return summary


def run_latest(args) -> Dict[str, Any]:
    """Run the pipeline with newest-frame-only slots per camera between stages."""
    ctx = mp.get_context("spawn")

    videos = args.videos or DEFAULT_VIDEO_CYCLE
    sources = camera_sources(args.num_cameras, videos)

    canonical, registry_mode, wants_torch = resolve_registry_mode(args.variant)
    compile_migraphx_nms_for_stream_if_requested(args, sources)

    pre_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_cameras)]
    post_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_cameras)]
    result_q = ctx.Queue()
    stats_q = ctx.Queue()
    error_q = ctx.Queue()
    stop_event = ctx.Event()
    grid_q = ctx.Queue(maxsize=max(1, int(args.grid_queue_size))) if args.grid_video else None
    grid_stop_event = ctx.Event() if args.grid_video else None
    camera_done = ctx.Array("b", [0] * args.num_cameras)
    infer_done = ctx.Array("b", [0] * args.infer_workers)
    post_pending = ctx.Array("b", [0] * args.num_cameras)
    backpressure_enabled = not bool(args.disable_backpressure)

    camera_procs = []
    infer_procs = []
    post_procs = []
    grid_procs = []

    print("\nStarting multi-camera stream simulation")
    print("---------------------------------------")
    print(f"Variant:       {canonical}")
    print(f"Registry mode: {registry_mode}")
    print(f"Torch needed:  {wants_torch}")
    print(f"Model:         {args.model}")
    print(f"Cameras:       {args.num_cameras}")
    print(f"Infer workers: {args.infer_workers}")
    print(f"Post workers:  {args.post_workers}")
    print(f"Buffer mode:   latest")
    print(f"Backpressure:  {'enabled' if backpressure_enabled else 'disabled'}")
    print(f"Realtime:      {args.realtime}")
    if args.grid_video:
        print(f"Grid video:    {args.grid_video} ({args.grid_cols}x{args.grid_rows})")
    print("Camera sources:")
    for cam_id, src in enumerate(sources):
        print(f"  cam {cam_id:02d}: {src}")

    if args.grid_video:
        p = ctx.Process(
            target=grid_video_writer_worker,
            kwargs=dict(
                grid_q=grid_q,
                output_path=args.grid_video,
                num_cameras=args.num_cameras,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                cell_w=args.grid_cell_width,
                cell_h=args.grid_cell_height,
                fps=args.grid_video_fps,
                codec=args.grid_video_codec,
                camera_sources_=sources,
                stop_event=grid_stop_event,
                stats_q=stats_q,
                error_q=error_q,
            ),
            name="grid_video_writer",
        )
        p.start()
        grid_procs.append(p)

    for cam_id, src in enumerate(sources):
        p = ctx.Process(
            target=camera_preprocess_latest_worker,
            kwargs=dict(
                camera_id=cam_id,
                video_path=src,
                out_queues=pre_queues,
                camera_done=camera_done,
                stats_q=stats_q,
                error_q=error_q,
                stop_event=stop_event,
                target_w=args.target_width,
                target_h=args.target_height,
                max_frames=args.frames_per_camera,
                duration_s=args.duration_s,
                realtime=args.realtime,
                camera_fps=args.camera_fps,
                keep_frame_for_output=bool(args.grid_video),
            ),
            name=f"camera_preprocess_latest_{cam_id}",
        )
        p.start()
        camera_procs.append(p)

    for worker_id in range(args.infer_workers):
        p = ctx.Process(
            target=inference_latest_worker,
            kwargs=dict(
                worker_id=worker_id,
                model_path=args.model,
                in_queues=pre_queues,
                out_queues=post_queues,
                camera_done=camera_done,
                infer_done=infer_done,
                post_pending=post_pending,
                backpressure=backpressure_enabled,
                stats_q=stats_q,
                error_q=error_q,
                target_w=args.target_width,
                target_h=args.target_height,
                stride=args.stride,
                shared_dtype=args.shared_dtype,
            ),
            name=f"migraphx_inference_latest_{worker_id}",
        )
        p.start()
        infer_procs.append(p)

    for worker_id in range(args.post_workers):
        p = ctx.Process(
            target=postprocess_latest_worker,
            kwargs=dict(
                worker_id=worker_id,
                user_variant=args.variant,
                in_queues=post_queues,
                infer_done=infer_done,
                post_pending=post_pending,
                result_q=result_q,
                stats_q=stats_q,
                error_q=error_q,
                torch_device="cuda" if args.torch_device == "auto" and wants_torch else args.torch_device,
                require_gpu=args.require_gpu,
                max_keypoints=args.max_keypoints,
                threshold=args.threshold,
                nms_radius_fullres=args.nms_radius_fullres,
                nms_radius_lowres=args.nms_radius_lowres,
                nms_impl=args.nms_impl,
                gpu_compute_dtype=args.gpu_compute_dtype,
                grid_q=grid_q,
                render_output=bool(args.grid_video),
                migraphx_nms_mxr=args.migraphx_nms_mxr,
                migraphx_nms_cache_dir=args.migraphx_nms_cache_dir,
            ),
            name=f"postprocess_latest_{worker_id}",
        )
        p.start()
        post_procs.append(p)

    rows: List[Dict[str, Any]] = []
    stage_stats: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    last_progress_print = 0

    try:
        while True:
            while not error_q.empty():
                err = error_q.get()
                raise RuntimeError(
                    f"Worker failed: {err.get('stage')} {err.get('worker_id', err.get('camera_id', ''))}\n"
                    f"{err.get('traceback')}"
                )

            while True:
                try:
                    row = result_q.get_nowait()
                except py_queue.Empty:
                    break
                rows.append(row)
                if args.print_every > 0 and len(rows) - last_progress_print >= args.print_every:
                    last_progress_print = len(rows)
                    elapsed = time.perf_counter() - t0
                    fps = len(rows) / elapsed if elapsed > 0 else 0.0
                    print(f"Processed output frames: {len(rows)} | elapsed={elapsed:.1f}s | aggregate FPS={fps:.2f}", flush=True)

            # Done when every post worker has exited; their exit condition is all inference workers done + post slots empty.
            if all(not p.is_alive() for p in post_procs):
                break

            # If camera workers have exited but an inference worker is stuck, this check still allows errors to surface.
            time.sleep(0.05)

        if args.grid_video and grid_stop_event is not None:
            # Stop the grid writer only after all postprocess workers have exited
            # and all result rows have been emitted. Join it before the finally
            # block so it can call VideoWriter.release() instead of being
            # terminated mid-write.
            grid_stop_event.set()
            for p in grid_procs:
                p.join(timeout=15.0)
                if p.is_alive():
                    raise RuntimeError("Grid video writer did not finish cleanly; output video may be incomplete.")

        while True:
            try:
                rows.append(result_q.get_nowait())
            except py_queue.Empty:
                break

        while True:
            try:
                stage_stats.append(stats_q.get_nowait())
            except py_queue.Empty:
                break

        for p in camera_procs + infer_procs + post_procs + grid_procs:
            p.join(timeout=2.0)

    except KeyboardInterrupt:
        print("Interrupted; stopping workers...", flush=True)
        stop_event.set()
        raise
    finally:
        if args.grid_video and grid_stop_event is not None:
            grid_stop_event.set()
        for p in camera_procs + infer_procs + post_procs + grid_procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    wall_s = time.perf_counter() - t0
    summary = summarize(rows, stage_stats, wall_s)
    summary["variant"] = canonical
    summary["registry_mode"] = registry_mode
    summary["model"] = args.model
    summary["num_cameras"] = args.num_cameras
    summary["infer_workers"] = args.infer_workers
    summary["post_workers"] = args.post_workers
    summary["queue_policy"] = args.queue_policy
    summary["buffer_mode"] = "latest"
    summary["grid_video"] = args.grid_video
    summary["grid_rows"] = args.grid_rows if args.grid_video else 0
    summary["grid_cols"] = args.grid_cols if args.grid_video else 0
    summary["backpressure_enabled"] = backpressure_enabled
    summary["realtime"] = args.realtime
    summary["camera_sources"] = sources

    print_summary(summary)
    write_detailed_csv(args.detailed_csv, rows)
    write_summary_json(args.summary_json, summary)
    return summary


def run(args) -> Dict[str, Any]:
    if getattr(args, "buffer_mode", "latest") == "latest":
        return run_latest(args)
    return run_queue(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate 10 live camera streams through preprocess -> MIGraphX -> postprocess pipeline."
    )
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr")
    parser.add_argument(
        "--variant",
        default="gpu_nms_fullres_two_process",
        help=(
            "Postprocess variant. Examples: standard, optimized_batch_k20_fast, "
            "lowres_cpu_group, gpu_nms_fullres_two_process, gpu_nms_lowres_two_process, "
            "migraphx-nms, migraphx-nms-k20."
        ),
    )
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEO_CYCLE)
    parser.add_argument("--num-cameras", type=int, default=10)
    parser.add_argument("--frames-per-camera", type=int, default=100, help="0 means run until interrupted/duration.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Optional wall-clock duration per camera. 0 disables duration limit.")
    parser.add_argument("--realtime", action="store_true", help="Throttle each simulated camera to --camera-fps.")
    parser.add_argument("--camera-fps", type=float, default=24.0)
    parser.add_argument("--queue-policy", choices=["drop", "block"], default="drop")
    parser.add_argument("--buffer-mode", choices=["latest", "queue"], default="latest", help="latest keeps one newest-frame slot per camera between stages; queue preserves the original FIFO queues.")
    parser.add_argument(
        "--disable-backpressure",
        action="store_true",
        help=(
            "Only used with --buffer-mode latest. By default, inference skips a camera "
            "while that camera already has a queued/in-flight postprocess result. "
            "This flag disables that guard and restores overwrite-before-post behavior."
        ),
    )

    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=1)
    parser.add_argument("--preprocess-queue-size", type=int, default=30)
    parser.add_argument("--postprocess-queue-size", type=int, default=30)

    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--shared-dtype", choices=["float32", "float16"], default="float32")

    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--migraphx-nms-mxr",
        default="",
        help="Optional explicit compiled MIGraphX NMS .mxr path for migraphx-nms variants.",
    )
    parser.add_argument(
        "--migraphx-nms-cache-dir",
        default="models/nms_fullres_cache",
        help="Directory containing heatmap_nms_head_<H>x<W>.mxr files.",
    )
    parser.add_argument(
        "--compile-migraphx-nms",
        action="store_true",
        help="Compile the stream-resolution MIGraphX NMS head before starting the stream.",
    )
    parser.add_argument("--force-compile-migraphx-nms", action="store_true")
    parser.add_argument("--keep-migraphx-nms-onnx", action="store_true")
    parser.add_argument("--exhaustive-tune-migraphx-nms", action="store_true")

    parser.add_argument(
        "--grid-video",
        default="",
        help=(
            "Optional output path for a single security-monitor-style grid video. "
            "When set, postprocessed frames are drawn and concatenated into one video."
        ),
    )
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--grid-cell-width", type=int, default=480)
    parser.add_argument("--grid-cell-height", type=int, default=270)
    parser.add_argument("--grid-video-fps", type=float, default=10.0)
    parser.add_argument("--grid-video-codec", default="mp4v")
    parser.add_argument("--grid-queue-size", type=int, default=256)

    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--detailed-csv", default="outputs/stream_10cam_detailed.csv")
    parser.add_argument("--summary-json", default="outputs/stream_10cam_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())