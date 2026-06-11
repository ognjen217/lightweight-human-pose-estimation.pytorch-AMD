"""Postprocess worker process entrypoints."""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..grid_video import draw_poses_on_frame
from ..postprocess_modes import (
    postprocess_precomputed_merged_pose_fused_pruned_item,
    resolve_registry_mode,
    select_migraphx_nms_mxr_for_hw,
)
from ..queues import all_done, all_queues_empty
from ..shared_memory import close_shared_map_views, open_shared_map_buffers, release_shared_slot_from_item
from ..system import configure_child_cpu_runtime
from ..tracing import RocTxTracer, trace_print
from ..utils import mean, percentile, safe_float


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
    prealloc_resize_buffers: bool = False,


    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    try:
        tracer = RocTxTracer(roctx_enabled, f"post:{worker_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")

        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
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
            require_gpu=bool(require_gpu and wants_torch),
            extra={
                "gpu_compute_dtype": gpu_compute_dtype,
                "nms_impl": nms_impl,
                "migraphx_nms_mxr": migraphx_nms_mxr,
                "migraphx_nms_cache_dir": migraphx_nms_cache_dir,
                "prealloc_resize_buffers": bool(prealloc_resize_buffers),
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

            with tracer.range(f"postprocess_{registry_mode}"):
                if registry_mode == "mx_merged_pose_fused_pruned" and item.get("merged_pose_fused_pruned_precomputed"):
                    out = postprocess_precomputed_merged_pose_fused_pruned_item(
                        item=item,
                        threshold=threshold,
                        min_pair_score=0.0,
                    )
                elif registry_mode == "mx_merged_pose_fused_pruned":
                    raise RuntimeError(
                        "mx_merged_pose_fused_pruned expects precomputed merged outputs from the inference worker. "
                        "Use --model <merged_b1.mxr> --variant mx_merged_pose_fused_pruned --migraphx-batch-size 1."
                    )
                else:
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
                "post_done_ts": post_done,
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
            trace_print(
                trace_log_every,
                processed,
                f"[TRACE post:{worker_id} pid={os.getpid()}] "
                f"processed={processed} cam={row['camera_id']} frame={row['frame_id']} "
                f"queue={queue_wait_ms:.2f}ms post={post_ms:.2f}ms e2e={e2e_ms:.2f}ms poses={row['num_poses']}",
            )

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

def postprocess_latest_worker(
    *,
    worker_id: int,
    user_variant: str,
    in_queues: Sequence[Any],
    infer_done,
    post_pending,
    last_processed_ts=None,
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
    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,
    prealloc_resize_buffers: bool = False,
    gpu_nms_batch_size: int = 1,
    gpu_nms_batch_timeout_ms: float = 0.0,


    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    """Round-robin postprocess worker over newest decoded-map slots, one slot per camera."""

    try:
        tracer = RocTxTracer(roctx_enabled, f"post:{worker_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")

        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
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

        from modules.postprocessing import PostprocessConfig, postprocess_from_maps, postprocess_gpu_nms_fullres_batch

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=torch_device,
            require_gpu=bool(require_gpu and wants_torch),
            extra={
                "gpu_compute_dtype": gpu_compute_dtype,
                "nms_impl": nms_impl,
                "migraphx_nms_mxr": migraphx_nms_mxr,
                "migraphx_nms_cache_dir": migraphx_nms_cache_dir,
                "prealloc_resize_buffers": bool(prealloc_resize_buffers),
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

        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        batch_size = max(1, int(gpu_nms_batch_size))
        batch_timeout_s = max(0.0, float(gpu_nms_batch_timeout_ms)) / 1000.0
        use_gpu_nms_batch = registry_mode == "gpu_nms_fullres_cpu_group" and batch_size > 1

        def _maps_for(batch_item):
            slot_id = batch_item.get("shared_map_slot") if isinstance(batch_item, dict) else None
            if slot_id is not None and int(slot_id) in shared_slots:
                slot = shared_slots[int(slot_id)]
                return slot["heat"], slot["paf"]
            return batch_item["heatmaps"], batch_item["pafs"]

        def _release_item(batch_item) -> None:
            release_shared_slot_from_item(batch_item, free_map_slots)

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

            batch_items = [item]
            if use_gpu_nms_batch:
                deadline = time.perf_counter() + batch_timeout_s
                while len(batch_items) < batch_size:
                    extra = None
                    scanned = 0
                    while scanned < ncam:
                        cam_id = next_cam
                        next_cam = (next_cam + 1) % ncam
                        scanned += 1
                        try:
                            extra = in_queues[cam_id].get_nowait()
                            break
                        except py_queue.Empty:
                            continue
                    if extra is not None:
                        batch_items.append(extra)
                        continue
                    if batch_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                        break
                    time.sleep(min(poll_sleep_s, max(0.0, deadline - time.perf_counter())))

            post_start = time.perf_counter()

            try:
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



                with tracer.range(f"postprocess_{registry_mode}_batch{len(batch_items)}"):
                    if use_gpu_nms_batch and len(batch_items) > 1:
                        batch_inputs = []
                        for bi in batch_items:
                            hm, pf = _maps_for(bi)
                            batch_inputs.append((hm, pf, tuple(bi["original_hw"])))
                        batch_outputs = postprocess_gpu_nms_fullres_batch(batch_inputs, config=config)
                    else:
                        batch_outputs = []
                        for bi in batch_items:
                            if registry_mode == "mx_merged_pose_fused_pruned" and bi.get("merged_pose_fused_pruned_precomputed"):
                                batch_outputs.append(
                                    postprocess_precomputed_merged_pose_fused_pruned_item(
                                        item=bi,
                                        threshold=threshold,
                                        min_pair_score=0.0,
                                    )
                                )
                            elif registry_mode == "mx_merged_pose_fused_pruned":
                                raise RuntimeError(
                                    "mx_merged_pose_fused_pruned expects precomputed merged outputs from the inference worker. "
                                    "Use --model <merged_b1.mxr> --variant mx_merged_pose_fused_pruned --migraphx-batch-size 1."
                                )
                            else:
                                hm, pf = _maps_for(bi)
                                batch_outputs.append(
                                    postprocess_from_maps(
                                        registry_mode,
                                        hm,
                                        pf,
                                        tuple(bi["original_hw"]),
                                        config=config,
                                    )
                                )


                if registry_mode == "mx_merged_pose_fused_pruned":
                    batch_outputs = []
                    for bi in batch_items:
                        if not bi.get("merged_pose_fused_pruned_precomputed"):
                            raise RuntimeError(
                                "mx_merged_pose_fused_pruned expects precomputed merged outputs from the inference worker. "
                                "Use --model <merged_b1.mxr> --variant mx_merged_pose_fused_pruned --migraphx-batch-size 1."
                            )
                        batch_outputs.append(
                            postprocess_precomputed_merged_pose_fused_pruned_item(
                                item=bi,
                                threshold=threshold,
                                min_pair_score=0.0,
                            )
                        )
                elif use_gpu_nms_batch and len(batch_items) > 1:
                    batch_inputs = []
                    for bi in batch_items:
                        hm, pf = _maps_for(bi)
                        batch_inputs.append((hm, pf, tuple(bi["original_hw"])))
                    batch_outputs = postprocess_gpu_nms_fullres_batch(batch_inputs, config=config)
                else:
                    batch_outputs = []
                    for bi in batch_items:
                        if registry_mode == "mx_merged_pose_fused_pruned":
                            if not bi.get("merged_pose_fused_pruned_precomputed"):
                                raise RuntimeError(
                                    "mx_merged_pose_fused_pruned expects precomputed merged outputs from the inference worker. "
                                    "Use --model <merged_b1.mxr> --variant mx_merged_pose_fused_pruned --migraphx-batch-size 1."
                                )
                            batch_outputs.append(
                                postprocess_precomputed_merged_pose_fused_pruned_item(
                                    item=bi,
                                    threshold=threshold,
                                    min_pair_score=0.0,
                                )
                            )
                            continue

                        hm, pf = _maps_for(bi)
                        batch_outputs.append(
                            postprocess_from_maps(
                                registry_mode,
                                hm,
                                pf,
                                tuple(bi["original_hw"]),
                                config=config,
                            )
                        )



            except Exception:
                for bi in batch_items:
                    _release_item(bi)
                raise

            for bi, out in zip(batch_items, batch_outputs):
                post_done = time.perf_counter()
                queue_wait_ms = (post_start - float(bi.get("infer_done_ts", post_start))) * 1000.0
                queue_wait_times.append(queue_wait_ms)

                timings = dict(out.timings)
                post_ms = float(timings.get("total_postprocess", (post_done - post_start) * 1000.0))
                e2e_ms = (post_done - float(bi["capture_ts"])) * 1000.0
                post_times.append(post_ms)
                e2e_times.append(e2e_ms)

                row: Dict[str, Any] = {
                    "camera_id": int(bi["camera_id"]),
                    "frame_id": int(bi["frame_id"]),
                    "source": bi["source"],
                    "variant": canonical,
                    "registry_mode": registry_mode,
                    "post_worker_id": worker_id,
                    "preprocess_ms": float(bi["preprocess_ms"]),
                    "queue_pre_to_infer_ms": float(bi["queue_pre_to_infer_ms"]),
                    "inference_ms": float(bi["inference_ms"]),
                    "decode_ms": float(bi["decode_ms"]),
                    "queue_infer_to_post_ms": float(queue_wait_ms),
                    "post_ms": post_ms,
                    "e2e_ms": e2e_ms,
                    "post_done_ts": post_done,
                    "num_poses": int(len(out.pose_entries)) if out.pose_entries is not None else 0,
                    "num_keypoints": int(len(out.all_keypoints)) if out.all_keypoints is not None else 0,
                }
                for key, value in timings.items():
                    row[f"timing_{key}"] = safe_float(value)

                if render_output and grid_q is not None and "frame_bgr" in bi:
                    frame_out = bi["frame_bgr"].copy()
                    draw_poses_on_frame(frame_out, out.pose_entries, out.all_keypoints)
                    packet = {
                        "camera_id": int(bi["camera_id"]),
                        "frame_id": int(bi["frame_id"]),
                        "source": bi["source"],
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
                cam_done_id = int(bi["camera_id"])
                if post_pending is not None:
                    post_pending[cam_done_id] = 0
                if last_processed_ts is not None:
                    last_processed_ts[cam_done_id] = time.perf_counter()
                _release_item(bi)
                processed += 1


                trace_print(
                    trace_log_every,
                    processed,
                    f"[TRACE post:{worker_id} pid={os.getpid()} latest] "
                    f"processed={processed} cam={row['camera_id']} frame={row['frame_id']} "
                    f"batch={len(batch_items)} queue={queue_wait_ms:.2f}ms post={post_ms:.2f}ms e2e={e2e_ms:.2f}ms poses={row['num_poses']}",
                )



        close_shared_map_views(shared_handles)
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
