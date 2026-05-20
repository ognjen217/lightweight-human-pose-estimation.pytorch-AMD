#!/usr/bin/env python3
"""
Two-process support for modules.postprocessing.

This module is intentionally separate from modules.postprocessing so the public
post-processing registry can expose a two-process runner without forcing the
single-process MIGraphX path to import or initialize PyTorch/ROCm.

Process layout
--------------
Process A: imports MIGraphX only, reads video, preprocesses frames, runs model,
           decodes low-res heatmaps/PAFs, writes them to shared-memory slots.
Process B: imports Torch only when GPU postprocess is selected, reads shared
           heatmaps/PAFs, calls modules.postprocessing.postprocess_from_maps().
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory
from queue import Empty
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


TimingRow = Dict[str, Any]


def parse_np_dtype(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}. Use float16 or float32.")


def create_shared_array(shape: Tuple[int, ...], dtype=np.float32):
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(0)
    return shm, arr


def attach_shared_array(shm_name: str, shape: Tuple[int, ...], dtype=np.float32):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def preprocess_frame(frame: np.ndarray, target_w: int, target_h: int, expected_dtype: str) -> np.ndarray:
    import cv2

    img = cv2.resize(frame, (target_w, target_h))
    img = (img.astype(np.float32) - 128.0) / 256.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    img = np.ascontiguousarray(img)
    if "half" in expected_dtype:
        return img.astype(np.float16)
    return img.astype(np.float32)


def decode_outputs(results: Any, out_h: int, out_w: int, output_dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) < 2:
        raise ValueError("MIGraphX results must contain at least heatmaps and PAFs.")

    heatmaps = np.asarray(results[-2], dtype=np.float32).reshape(19, out_h, out_w)
    pafs = np.asarray(results[-1], dtype=np.float32).reshape(38, out_h, out_w)
    heatmaps = np.moveaxis(heatmaps, 0, -1)
    pafs = np.moveaxis(pafs, 0, -1)

    if output_dtype == np.float16:
        return (
            np.ascontiguousarray(heatmaps, dtype=np.float16),
            np.ascontiguousarray(pafs, dtype=np.float16),
        )
    return (
        np.ascontiguousarray(heatmaps, dtype=np.float32),
        np.ascontiguousarray(pafs, dtype=np.float32),
    )


def inference_worker(
    *,
    video_path: str,
    model_path: str,
    target_w: int,
    target_h: int,
    stride: int,
    total_frames: int,
    heatmap_shm_name: str,
    paf_shm_name: str,
    heatmap_shape: Tuple[int, ...],
    paf_shape: Tuple[int, ...],
    free_q,
    full_q,
    error_q,
    shared_dtype_name: str,
) -> None:
    """MIGraphX-only process."""
    try:
        import cv2
        import migraphx

        out_h = target_h // stride
        out_w = target_w // stride
        shared_dtype = parse_np_dtype(shared_dtype_name)

        heatmap_shm, heatmap_buf = attach_shared_array(heatmap_shm_name, heatmap_shape, shared_dtype)
        paf_shm, paf_buf = attach_shared_array(paf_shm_name, paf_shape, shared_dtype)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Cannot find video: {video_path}")

        print("[INFER] Loading MIGraphX model...")
        model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER] Model loaded. Expected dtype: {expected_dtype}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if total_frames is not None and frame_idx > total_frames:
                break

            slot = free_q.get()
            original_h, original_w = frame.shape[:2]

            t0 = time.perf_counter()
            inp = preprocess_frame(frame, target_w, target_h, expected_dtype)
            pre_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            results = model.run({"input": inp})
            infer_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            heatmaps, pafs = decode_outputs(results, out_h, out_w, output_dtype=shared_dtype)
            decode_ms = (time.perf_counter() - t0) * 1000.0

            heatmap_buf[slot, :, :, :] = heatmaps
            paf_buf[slot, :, :, :] = pafs

            full_q.put(
                {
                    "slot": slot,
                    "frame_idx": frame_idx,
                    "original_hw": (original_h, original_w),
                    "preprocess": pre_ms,
                    "inference": infer_ms,
                    "decode": decode_ms,
                }
            )

        cap.release()
        full_q.put(None)
        heatmap_shm.close()
        paf_shm.close()
        print("[INFER] Done.")

    except Exception:
        error_q.put(("inference_worker", traceback.format_exc()))
        try:
            full_q.put(None)
        except Exception:
            pass


def _worker_mode_to_registry_mode(worker_mode: str) -> str:
    if worker_mode == "cpu-k20-fast":
        return "optimized_batch_k20_fast"
    if worker_mode == "gpu-nms-fullres":
        return "gpu_nms_fullres_cpu_group"
    if worker_mode == "gpu-nms-lowres":
        return "gpu_nms_lowres_cpu_group"
    raise ValueError(
        f"Unknown two-process worker mode: {worker_mode}. "
        "Use cpu-k20-fast, gpu-nms-fullres, or gpu-nms-lowres."
    )


def postprocess_worker(
    *,
    worker_mode: str,
    torch_device: str,
    max_keypoints: int,
    threshold: float,
    shared_dtype_name: str,
    gpu_compute_dtype: str,
    nms_radius_fullres: int,
    nms_radius_lowres: int,
    nms_impl: str,
    heatmap_shm_name: str,
    paf_shm_name: str,
    heatmap_shape: Tuple[int, ...],
    paf_shape: Tuple[int, ...],
    free_q,
    full_q,
    result_q,
    error_q,
) -> None:
    """Postprocess-only process. Torch is imported only here for GPU modes."""
    try:
        registry_mode = _worker_mode_to_registry_mode(worker_mode)
        wants_torch = registry_mode.startswith("gpu")

        if wants_torch:
            import torch

            print("[POST] Initializing PyTorch ROCm/CUDA in isolated postprocess process...")
            print(f"[POST] torch.cuda.is_available(): {torch.cuda.is_available()}")
            if torch_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested torch_device=cuda, but torch.cuda.is_available() is False")
            if torch_device == "cuda":
                warm = torch.empty((1,), device="cuda")
                warm += 1
                torch.cuda.synchronize()
                print(f"[POST] Torch GPU name: {torch.cuda.get_device_name(0)}")

        from modules.postprocessing import PostprocessConfig, postprocess_from_maps

        shared_dtype = parse_np_dtype(shared_dtype_name)
        heatmap_shm, heatmap_buf = attach_shared_array(heatmap_shm_name, heatmap_shape, shared_dtype)
        paf_shm, paf_buf = attach_shared_array(paf_shm_name, paf_shape, shared_dtype)

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=torch_device,
            require_gpu=wants_torch and torch_device == "cuda",
            extra={"gpu_compute_dtype": gpu_compute_dtype, "nms_impl": nms_impl},
        )

        print(f"[POST] Worker mode: {worker_mode}")
        print(f"[POST] Registry mode: {registry_mode}")
        print(f"[POST] Shared dtype: {shared_dtype_name}")
        print(f"[POST] GPU compute dtype: {gpu_compute_dtype}")

        while True:
            item = full_q.get()
            if item is None:
                break

            slot = int(item["slot"])
            frame_idx = int(item["frame_idx"])
            original_hw = tuple(item["original_hw"])

            heatmaps = heatmap_buf[slot]
            pafs = paf_buf[slot]

            t0 = time.perf_counter()
            try:
                out = postprocess_from_maps(registry_mode, heatmaps, pafs, original_hw, config=config)
            finally:
                # The slot must only be released after postprocess finishes reading it.
                free_q.put(slot)

            post_worker_wall_ms = (time.perf_counter() - t0) * 1000.0
            row = dict(out.timings)
            row.update(
                {
                    "frame_idx": frame_idx,
                    "preprocess": float(item.get("preprocess", 0.0)),
                    "inference": float(item.get("inference", 0.0)),
                    "decode": float(item.get("decode", 0.0)),
                    "post_worker_wall_ms": post_worker_wall_ms,
                    "e2e": float(item.get("preprocess", 0.0))
                    + float(item.get("inference", 0.0))
                    + float(item.get("decode", 0.0))
                    + float(out.timings.get("total_postprocess", 0.0)),
                }
            )
            result_q.put(row)

        heatmap_shm.close()
        paf_shm.close()
        result_q.put(None)
        print("[POST] Done.")

    except Exception:
        error_q.put(("postprocess_worker", traceback.format_exc()))
        try:
            result_q.put(None)
        except Exception:
            pass


def summarize_rows(rows: List[TimingRow], total_wall_ms: float, variant_name: str) -> Dict[str, Any]:
    return {
        "variant": variant_name,
        "frames": len(rows),
        "preprocess_ms": mean([r.get("preprocess", 0.0) for r in rows]),
        "inference_ms": mean([r.get("inference", 0.0) for r in rows]),
        "decode_ms": mean([r.get("decode", 0.0) for r in rows]),
        "hm_resize_ms": mean([r.get("resize_heatmaps", 0.0) for r in rows]),
        "paf_resize_ms": mean([r.get("resize_pafs", 0.0) for r in rows]),
        "extract_ms": mean([r.get("extract_keypoints", 0.0) for r in rows]),
        "group_ms": mean([r.get("group_keypoints", 0.0) for r in rows]),
        "post_avg_ms": mean([r.get("total_postprocess", 0.0) for r in rows]),
        "post_p50_ms": percentile([r.get("total_postprocess", 0.0) for r in rows], 50),
        "post_p95_ms": percentile([r.get("total_postprocess", 0.0) for r in rows], 95),
        "e2e_avg_ms": mean([r.get("e2e", 0.0) for r in rows]),
        "e2e_p95_ms": percentile([r.get("e2e", 0.0) for r in rows], 95),
        "e2e_fps": 1000.0 / mean([r.get("e2e", 0.0) for r in rows]) if mean([r.get("e2e", 0.0) for r in rows]) > 0 else 0.0,
        "pipeline_wall_s": total_wall_ms / 1000.0,
        "pipeline_fps": len(rows) / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0,
    }


def print_row(row: TimingRow) -> None:
    print(
        f"{int(row['frame_idx']):6d} | "
        f"pre={row.get('preprocess', 0.0):7.2f} | "
        f"infer={row.get('inference', 0.0):7.2f} | "
        f"dec={row.get('decode', 0.0):6.2f} | "
        f"hm_resize={row.get('resize_heatmaps', 0.0):7.2f} | "
        f"paf_resize={row.get('resize_pafs', 0.0):7.2f} | "
        f"extract={row.get('extract_keypoints', 0.0):8.2f} | "
        f"group={row.get('group_keypoints', 0.0):7.2f} | "
        f"post={row.get('total_postprocess', 0.0):8.2f}"
    )


def run_two_process_pipeline(
    *,
    video_path: str,
    model_path: str,
    variant_name: str,
    worker_mode: str,
    target_width: int = 968,
    target_height: int = 544,
    stride: int = 8,
    max_frames: int = 100,
    warmup_frames: int = 5,
    slots: int = 3,
    print_every: int = 10,
    torch_device: str = "cuda",
    shared_dtype: str = "float32",
    gpu_compute_dtype: str = "float32",
    max_keypoints: int = 20,
    threshold: float = 0.1,
    nms_radius_fullres: int = 6,
    nms_radius_lowres: int = 1,
    nms_impl: str = "2d",
    collect_rows: bool = True,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")

    out_h = target_height // stride
    out_w = target_width // stride
    heatmap_shape = (slots, out_h, out_w, 19)
    paf_shape = (slots, out_h, out_w, 38)
    shared_np_dtype = parse_np_dtype(shared_dtype)

    heatmap_shm, _ = create_shared_array(heatmap_shape, shared_np_dtype)
    paf_shm, _ = create_shared_array(paf_shape, shared_np_dtype)

    free_q = ctx.Queue(maxsize=slots)
    full_q = ctx.Queue(maxsize=slots)
    result_q = ctx.Queue()
    error_q = ctx.Queue()

    for i in range(slots):
        free_q.put(i)

    total_frames_to_process = int(max_frames) + int(warmup_frames)

    infer_p = ctx.Process(
        target=inference_worker,
        kwargs=dict(
            video_path=video_path,
            model_path=model_path,
            target_w=target_width,
            target_h=target_height,
            stride=stride,
            total_frames=total_frames_to_process,
            heatmap_shm_name=heatmap_shm.name,
            paf_shm_name=paf_shm.name,
            heatmap_shape=heatmap_shape,
            paf_shape=paf_shape,
            free_q=free_q,
            full_q=full_q,
            error_q=error_q,
            shared_dtype_name=shared_dtype,
        ),
        name="migraphx_inference_process",
    )

    post_p = ctx.Process(
        target=postprocess_worker,
        kwargs=dict(
            worker_mode=worker_mode,
            torch_device=torch_device,
            max_keypoints=max_keypoints,
            threshold=threshold,
            shared_dtype_name=shared_dtype,
            gpu_compute_dtype=gpu_compute_dtype,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            nms_impl=nms_impl,
            heatmap_shm_name=heatmap_shm.name,
            paf_shm_name=paf_shm.name,
            heatmap_shape=heatmap_shape,
            paf_shape=paf_shape,
            free_q=free_q,
            full_q=full_q,
            result_q=result_q,
            error_q=error_q,
        ),
        name="postprocess_process",
    )

    print("Two-process postprocessing through modules.postprocessing")
    print("--------------------------------------------------------")
    print(f"Variant:     {variant_name}")
    print(f"Worker mode: {worker_mode}")
    print(f"Video:       {video_path}")
    print(f"Model:       {model_path}")
    print(f"Warmup:      {warmup_frames}")
    print(f"Measured:    {max_frames}")
    print(f"Slots:       {slots}")
    print(f"Shared dtype:{shared_dtype}")
    print(f"Torch device:{torch_device}")
    print(f"Output dims: {out_h} x {out_w}")

    all_rows: List[TimingRow] = []
    measured_rows: List[TimingRow] = []
    t_total = time.perf_counter()

    try:
        infer_p.start()
        post_p.start()

        while True:
            while not error_q.empty():
                who, tb = error_q.get()
                raise RuntimeError(f"{who} failed:\n{tb}")

            try:
                item = result_q.get(timeout=0.5)
            except Empty:
                if not infer_p.is_alive() and not post_p.is_alive():
                    break
                continue

            if item is None:
                break

            all_rows.append(item)
            if int(item["frame_idx"]) > warmup_frames:
                measured_rows.append(item)
                if len(measured_rows) == 1 or (print_every > 0 and len(measured_rows) % print_every == 0):
                    print_row(item)

            if len(measured_rows) >= max_frames:
                # Inference worker was already bounded to warmup+measured frames.
                pass

        infer_p.join(timeout=5)
        post_p.join(timeout=5)

        while not error_q.empty():
            who, tb = error_q.get()
            raise RuntimeError(f"{who} failed:\n{tb}")

    finally:
        total_wall_ms = (time.perf_counter() - t_total) * 1000.0
        if infer_p.is_alive():
            infer_p.terminate()
        if post_p.is_alive():
            post_p.terminate()
        heatmap_shm.close()
        paf_shm.close()
        try:
            heatmap_shm.unlink()
        except FileNotFoundError:
            pass
        try:
            paf_shm.unlink()
        except FileNotFoundError:
            pass

    measured_rows = measured_rows[:max_frames]
    if not measured_rows:
        raise RuntimeError("No measured rows produced by two-process pipeline.")

    summary = summarize_rows(measured_rows, total_wall_ms, variant_name)
    return {
        "variant": variant_name,
        "worker_mode": worker_mode,
        "rows": measured_rows if collect_rows else [],
        "all_rows": all_rows if collect_rows else [],
        "summary": summary,
        "wall_ms": total_wall_ms,
    }
