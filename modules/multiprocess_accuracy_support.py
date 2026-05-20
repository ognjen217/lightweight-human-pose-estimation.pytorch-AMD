#!/usr/bin/env python3
"""
Two-process support for COCO accuracy validation.

This module is intentionally small and does not import MIGraphX.  The main
accuracy_validation.py process owns MIGraphX inference.  GPU post-processing
workers spawned from this module own PyTorch/ROCm.  This keeps MIGraphX and
PyTorch GPU contexts in separate Python processes.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from queue import Empty
from typing import Any, Dict, Tuple


@dataclass
class AccuracyWorkerHandle:
    variant: str
    process: Any
    input_q: Any
    output_q: Any
    error_q: Any
    next_seq: int = 0


def _worker_mode_to_registry_mode(worker_mode: str) -> str:
    if worker_mode == "gpu-nms-fullres":
        return "gpu_nms_fullres_cpu_group"
    if worker_mode == "gpu-nms-lowres":
        return "gpu_nms_lowres_cpu_group"
    if worker_mode == "cpu-k20-fast":
        return "optimized_batch_k20_fast"
    raise ValueError(
        f"Unknown two-process worker mode: {worker_mode}. "
        "Use gpu-nms-fullres, gpu-nms-lowres, or cpu-k20-fast."
    )


def accuracy_postprocess_worker(
    *,
    variant_name: str,
    worker_mode: str,
    torch_device: str,
    max_keypoints: int,
    threshold: float,
    nms_radius_fullres: int,
    nms_radius_lowres: int,
    nms_impl: str,
    gpu_compute_dtype: str,
    input_q,
    output_q,
    error_q,
) -> None:
    """Postprocess-only worker for COCO validation."""
    try:
        registry_mode = _worker_mode_to_registry_mode(worker_mode)
        wants_torch = registry_mode.startswith("gpu")

        if wants_torch:
            import torch

            print(f"[ACC-POST:{variant_name}] Initializing PyTorch ROCm/CUDA in isolated process...")
            print(f"[ACC-POST:{variant_name}] torch.cuda.is_available(): {torch.cuda.is_available()}")
            if torch_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested torch_device=cuda, but torch.cuda.is_available() is False")
            if torch_device == "cuda":
                warm = torch.empty((1,), device="cuda")
                warm += 1
                torch.cuda.synchronize()
                print(f"[ACC-POST:{variant_name}] Torch GPU name: {torch.cuda.get_device_name(0)}")

        from modules.postprocessing import PostprocessConfig, postprocess_from_maps

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=torch_device,
            require_gpu=wants_torch and torch_device == "cuda",
            extra={"gpu_compute_dtype": gpu_compute_dtype, "nms_impl": nms_impl},
        )

        print(f"[ACC-POST:{variant_name}] Worker mode: {worker_mode}")
        print(f"[ACC-POST:{variant_name}] Registry mode: {registry_mode}")
        print(f"[ACC-POST:{variant_name}] NMS impl: {nms_impl}")
        print(f"[ACC-POST:{variant_name}] GPU compute dtype: {gpu_compute_dtype}")

        while True:
            item = input_q.get()
            if item is None:
                break

            seq = int(item["seq"])
            image_id = int(item["image_id"])
            heatmaps = item["heatmaps"]
            pafs = item["pafs"]
            original_hw = tuple(item["original_hw"])

            t0 = time.perf_counter()
            out = postprocess_from_maps(registry_mode, heatmaps, pafs, original_hw, config=config)
            post_worker_wall_ms = (time.perf_counter() - t0) * 1000.0

            output_q.put(
                {
                    "seq": seq,
                    "image_id": image_id,
                    "variant": variant_name,
                    "pose_entries": out.pose_entries,
                    "all_keypoints": out.all_keypoints,
                    "timings": dict(out.timings),
                    "post_worker_wall_ms": post_worker_wall_ms,
                }
            )

        output_q.put(None)
        print(f"[ACC-POST:{variant_name}] Done.")

    except Exception:
        error_q.put((variant_name, traceback.format_exc()))
        try:
            output_q.put(None)
        except Exception:
            pass


def start_accuracy_postprocess_worker(
    *,
    variant_name: str,
    worker_mode: str,
    torch_device: str,
    max_keypoints: int,
    threshold: float,
    nms_radius_fullres: int,
    nms_radius_lowres: int,
    nms_impl: str,
    gpu_compute_dtype: str,
    queue_size: int = 2,
) -> AccuracyWorkerHandle:
    ctx = mp.get_context("spawn")
    input_q = ctx.Queue(maxsize=max(1, int(queue_size)))
    output_q = ctx.Queue(maxsize=max(1, int(queue_size)))
    error_q = ctx.Queue()

    process = ctx.Process(
        target=accuracy_postprocess_worker,
        kwargs=dict(
            variant_name=variant_name,
            worker_mode=worker_mode,
            torch_device=torch_device,
            max_keypoints=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            nms_impl=nms_impl,
            gpu_compute_dtype=gpu_compute_dtype,
            input_q=input_q,
            output_q=output_q,
            error_q=error_q,
        ),
        name=f"accuracy_postprocess_{variant_name}",
    )
    process.start()
    return AccuracyWorkerHandle(
        variant=variant_name,
        process=process,
        input_q=input_q,
        output_q=output_q,
        error_q=error_q,
    )


def _raise_worker_error_if_any(handle: AccuracyWorkerHandle) -> None:
    if not handle.error_q.empty():
        variant, tb = handle.error_q.get()
        raise RuntimeError(f"Accuracy postprocess worker failed for {variant}:\n{tb}")


def run_accuracy_postprocess_item(
    handle: AccuracyWorkerHandle,
    *,
    image_id: int,
    heatmaps,
    pafs,
    original_hw: Tuple[int, int],
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    _raise_worker_error_if_any(handle)

    seq = handle.next_seq
    handle.next_seq += 1
    handle.input_q.put(
        {
            "seq": seq,
            "image_id": int(image_id),
            "heatmaps": heatmaps,
            "pafs": pafs,
            "original_hw": tuple(original_hw),
        }
    )

    deadline = time.perf_counter() + float(timeout_s)
    while True:
        _raise_worker_error_if_any(handle)
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Timed out waiting for accuracy worker {handle.variant} on image_id={image_id}")
        try:
            result = handle.output_q.get(timeout=0.25)
        except Empty:
            if not handle.process.is_alive():
                _raise_worker_error_if_any(handle)
                raise RuntimeError(f"Accuracy worker {handle.variant} exited before returning result.")
            continue
        if result is None:
            _raise_worker_error_if_any(handle)
            raise RuntimeError(f"Accuracy worker {handle.variant} stopped unexpectedly.")
        if int(result.get("seq", -1)) == seq:
            return result
        raise RuntimeError(
            f"Accuracy worker {handle.variant} returned out-of-order seq={result.get('seq')} expected={seq}."
        )


def stop_accuracy_postprocess_worker(handle: AccuracyWorkerHandle, timeout_s: float = 10.0) -> None:
    try:
        handle.input_q.put(None)
    except Exception:
        pass
    try:
        handle.process.join(timeout=timeout_s)
    except Exception:
        pass
    if handle.process.is_alive():
        handle.process.terminate()
        handle.process.join(timeout=2.0)
