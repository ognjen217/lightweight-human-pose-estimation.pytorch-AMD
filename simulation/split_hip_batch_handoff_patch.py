"""Batch-handoff runtime patch for split HIP2 stream variants.

This patch builds on ``simulation.split_hip_smart_patch`` and adds an
experimental low-latency handoff mode:

    split_hip2_batch_handoff
        MXR1 Bx image inference -> one batch packet -> HIP smart heatmap TopK
        -> HIP2 PAF pruning -> CPU assembly.

The previous ``split_hip2_host_smart`` path splits the MXR1 batch back into
per-camera queue items and the postprocess stage has to rebuild a batch.  This
mode preserves the already-formed MXR1 batch across the inference->postprocess
boundary, so postprocess runs on the same real batch instead of re-batching.
"""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from simulation import split_hip_smart_patch as split_base

_HANDOFF_ALIASES = {
    "split_hip2_batch_handoff",
    "split-hip2-batch-handoff",
    "split_hip2_host_smart_handoff",
    "split-hip2-host-smart-handoff",
    "split_hip2_handoff",
    "split-hip2-handoff",
    "mxr1_hip2_batch_handoff",
    "mxr1-hip2-batch-handoff",
}

_ORIGINALS: Dict[str, Any] = {}


def _variant_key(user_variant: str) -> str:
    return str(user_variant or "").strip().lower().replace(" ", "-")


def _is_handoff_variant(user_variant: str) -> bool:
    key = _variant_key(user_variant)
    return key in _HANDOFF_ALIASES or key.replace("-", "_") in _HANDOFF_ALIASES or key.replace("_", "-") in _HANDOFF_ALIASES


def _canonical_handoff_variant(_user_variant: str) -> str:
    return "split_hip2_batch_handoff"


def _split_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    return int(raw)


def _split_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _packet_items(packet_or_item: Any) -> List[Dict[str, Any]]:
    if isinstance(packet_or_item, dict) and packet_or_item.get("split_batch_packet"):
        return list(packet_or_item.get("batch_items", []))
    if isinstance(packet_or_item, dict):
        return [packet_or_item]
    return []


def _release_packet(packet_or_item: Any, free_map_slots=None, post_pending=None) -> int:
    """Release shared-map slots for a dropped item/packet and clear pending flags."""
    from simulation.shared_memory import release_shared_slot_from_item

    released = 0
    for item in _packet_items(packet_or_item):
        try:
            release_shared_slot_from_item(item, free_map_slots)
        finally:
            try:
                if post_pending is not None:
                    post_pending[int(item["camera_id"])] = 0
            except Exception:
                pass
        released += 1
    return released


def inference_latest_worker_patched(
    *,
    worker_id: int,
    model_path: str,
    in_queues: Sequence[Any],
    out_queues: Sequence[Any],
    camera_done,
    infer_done,
    post_pending,
    backpressure_mode: str = "strict",
    max_pending_age_ms: float = 300.0,
    post_pending_ts=None,
    last_processed_ts=None,
    target_period_s: float = 0.0,
    stats_q=None,
    error_q=None,
    target_w: int = 968,
    target_h: int = 544,
    stride: int = 8,
    shared_dtype: str = "float32",
    poll_sleep_s: float = 0.001,
    migraphx_nms_mxr: str = "",
    migraphx_nms_cache_dir: str = "",
    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,
    shared_input_descs: Optional[Sequence[Dict[str, Any]]] = None,
    migraphx_batch_size: int = 1,
    migraphx_batch_timeout_ms: float = 0.0,
    collector_coalesce: bool = True,
    merged_pose_fused_pruned: bool = False,
    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    if os.environ.get("STREAM_SPLIT_HIP_BATCH_HANDOFF", "0").strip() != "1":
        return _ORIGINALS["inference_latest_worker"](
            worker_id=worker_id,
            model_path=model_path,
            in_queues=in_queues,
            out_queues=out_queues,
            camera_done=camera_done,
            infer_done=infer_done,
            post_pending=post_pending,
            backpressure_mode=backpressure_mode,
            max_pending_age_ms=max_pending_age_ms,
            post_pending_ts=post_pending_ts,
            last_processed_ts=last_processed_ts,
            target_period_s=target_period_s,
            stats_q=stats_q,
            error_q=error_q,
            target_w=target_w,
            target_h=target_h,
            stride=stride,
            shared_dtype=shared_dtype,
            poll_sleep_s=poll_sleep_s,
            migraphx_nms_mxr=migraphx_nms_mxr,
            migraphx_nms_cache_dir=migraphx_nms_cache_dir,
            shared_map_descs=shared_map_descs,
            free_map_slots=free_map_slots,
            shared_input_descs=shared_input_descs,
            migraphx_batch_size=migraphx_batch_size,
            migraphx_batch_timeout_ms=migraphx_batch_timeout_ms,
            collector_coalesce=collector_coalesce,
            merged_pose_fused_pruned=merged_pose_fused_pruned,
            trace_log_every=trace_log_every,
            roctx_enabled=roctx_enabled,
        )

    try:
        import migraphx  # type: ignore

        from simulation.migraphx_io import (
            build_inference_output_items_from_batch,
            decode_migraphx_batch_outputs,
            make_migraphx_input_batch,
        )
        from simulation.queues import all_done, all_queues_empty, latest_put_with_dropped
        from simulation.shared_memory import (
            close_shared_map_views,
            open_shared_input_buffers,
            open_shared_map_buffers,
            release_shared_slot_from_item,
        )
        from simulation.system import configure_child_cpu_runtime
        from simulation.tracing import RocTxTracer, trace_print
        from simulation.utils import Timer, mean, percentile

        tracer = RocTxTracer(roctx_enabled, f"infer:{worker_id}:pid:{os.getpid()}:split_hip2_batch_handoff")
        tracer.mark("worker_start")
        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        with tracer.range("migraphx_load"):
            model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)
        print(
            f"[INFER:{worker_id}] MIGraphX inference batch size={int(migraphx_batch_size)} "
            f"timeout={float(migraphx_batch_timeout_ms):.2f} ms collector_coalesce={bool(collector_coalesce)} "
            "handoff=split_batch_packet",
            flush=True,
        )

        if merged_pose_fused_pruned:
            raise RuntimeError("split_hip2_batch_handoff expects MXR1 heatmaps/PAFs, not merged fused-pruned outputs")

        out_h = target_h // stride
        out_w = target_w // stride
        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        shared_input_slots, shared_input_handles = open_shared_input_buffers(shared_input_descs)
        shared_map_misses = 0
        ncam = len(in_queues)
        next_cam = worker_id % max(1, ncam)
        configured_batch_size = max(1, int(migraphx_batch_size))
        batch_timeout_s = max(0.0, float(migraphx_batch_timeout_ms)) / 1000.0

        processed = 0
        batch_runs = 0
        packet_runs = 0
        batch_sizes_seen: List[int] = []
        replaced_before_post = 0
        skipped_due_backpressure = 0
        soft_overrides = 0
        throttle_skips = 0
        backpressure_idle_loops = 0
        inference_times: List[float] = []
        decode_times: List[float] = []
        queue_wait_times: List[float] = []
        stale_records_discarded_pre_batch = 0
        stale_discards_per_batch: List[int] = []
        batches_with_stale_discards = 0
        t_worker_start = time.perf_counter()

        def _camera_is_eligible(cam_id: int, count_skip: bool = True) -> bool:
            nonlocal skipped_due_backpressure, soft_overrides, throttle_skips

            if target_period_s > 0.0 and last_processed_ts is not None:
                last_ts = float(last_processed_ts[cam_id])
                if last_ts > 0.0 and (time.perf_counter() - last_ts) < target_period_s:
                    if count_skip:
                        throttle_skips += 1
                    return False

            if backpressure_mode != "off" and post_pending is not None and bool(post_pending[cam_id]):
                if backpressure_mode == "soft" and post_pending_ts is not None:
                    age_ms = (time.perf_counter() - float(post_pending_ts[cam_id])) * 1000.0
                    if age_ms <= max_pending_age_ms:
                        if count_skip:
                            skipped_due_backpressure += 1
                        return False
                    if count_skip:
                        soft_overrides += 1
                    return True
                if count_skip:
                    skipped_due_backpressure += 1
                return False

            return True

        def _item_preprocess_ts(item: Dict[str, Any]) -> float:
            return float(item.get("preprocess_done_ts", item.get("capture_ts", 0.0)))

        def _drain_latest_for_camera(cam_id: int) -> Optional[Dict[str, Any]]:
            nonlocal stale_records_discarded_pre_batch
            if not collector_coalesce:
                try:
                    return in_queues[cam_id].get_nowait()
                except py_queue.Empty:
                    return None

            newest = None
            while True:
                try:
                    candidate = in_queues[cam_id].get_nowait()
                except py_queue.Empty:
                    break
                if newest is not None:
                    stale_records_discarded_pre_batch += 1
                newest = candidate
            return newest

        def _add_or_replace_batch_item(
            batch_items: List[Dict[str, Any]],
            batch_index_by_camera: Dict[int, int],
            candidate: Dict[str, Any],
        ) -> bool:
            nonlocal stale_records_discarded_pre_batch
            cam_id = int(candidate["camera_id"])
            existing_idx = batch_index_by_camera.get(cam_id)
            if existing_idx is None:
                batch_index_by_camera[cam_id] = len(batch_items)
                batch_items.append(candidate)
                return True
            stale_records_discarded_pre_batch += 1
            if _item_preprocess_ts(candidate) >= _item_preprocess_ts(batch_items[existing_idx]):
                batch_items[existing_idx] = candidate
            return False

        def _get_next_item() -> tuple[Optional[Dict[str, Any]], int]:
            nonlocal next_cam
            scanned = 0
            skipped_this_scan = 0
            while scanned < ncam:
                cam_id = next_cam
                next_cam = (next_cam + 1) % ncam
                scanned += 1

                before_bp = skipped_due_backpressure
                before_thr = throttle_skips
                if not _camera_is_eligible(cam_id, count_skip=True):
                    if skipped_due_backpressure > before_bp or throttle_skips > before_thr:
                        skipped_this_scan += 1
                    continue

                item = _drain_latest_for_camera(cam_id)
                if item is not None:
                    return item, skipped_this_scan

            return None, skipped_this_scan

        def _copy_item_to_shared_map(out_item: Dict[str, Any]) -> None:
            nonlocal shared_map_misses
            if shared_slots and free_map_slots is not None:
                slot_id = None
                try:
                    slot_id = int(free_map_slots.get_nowait())
                    slot = shared_slots[slot_id]
                    if slot["heat"].shape != out_item["heatmaps"].shape or slot["paf"].shape != out_item["pafs"].shape:
                        raise ValueError(
                            f"shared-map slot shape mismatch: heat {slot['heat'].shape}!={out_item['heatmaps'].shape}, "
                            f"paf {slot['paf'].shape}!={out_item['pafs'].shape}"
                        )
                    np.copyto(slot["heat"], out_item.pop("heatmaps"), casting="same_kind")
                    np.copyto(slot["paf"], out_item.pop("pafs"), casting="same_kind")
                    out_item["shared_map_slot"] = slot_id
                except py_queue.Empty:
                    shared_map_misses += 1
                except Exception:
                    shared_map_misses += 1
                    if slot_id is not None:
                        try:
                            free_map_slots.put_nowait(slot_id)
                        except Exception:
                            pass

        while True:
            item, skipped_this_scan = _get_next_item()

            if item is None:
                if all_done(camera_done) and all_queues_empty(in_queues):
                    break
                if skipped_this_scan > 0 or throttle_skips > 0:
                    backpressure_idle_loops += 1
                time.sleep(poll_sleep_s)
                continue

            stale_before_batch = stale_records_discarded_pre_batch
            batch_items = [item]
            batch_index_by_camera = {int(item["camera_id"]): 0}

            if configured_batch_size > 1:
                deadline = time.perf_counter() + batch_timeout_s
                while len(batch_items) < configured_batch_size:
                    extra, _ = _get_next_item()
                    if extra is not None:
                        if collector_coalesce:
                            _add_or_replace_batch_item(batch_items, batch_index_by_camera, extra)
                        else:
                            batch_items.append(extra)
                        continue
                    if batch_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                        break
                    time.sleep(min(poll_sleep_s, max(0.0, deadline - time.perf_counter())))

            if collector_coalesce and batch_items:
                for cam_id in list(batch_index_by_camera.keys()):
                    if not _camera_is_eligible(cam_id, count_skip=False):
                        continue
                    latest = _drain_latest_for_camera(cam_id)
                    if latest is not None:
                        _add_or_replace_batch_item(batch_items, batch_index_by_camera, latest)

            batch_stale_discards = stale_records_discarded_pre_batch - stale_before_batch
            stale_discards_per_batch.append(batch_stale_discards)
            if batch_stale_discards > 0:
                batches_with_stale_discards += 1

            infer_start = time.perf_counter()
            batch_queue_wait_ms = [
                (infer_start - float(bi.get("preprocess_done_ts", infer_start))) * 1000.0
                for bi in batch_items
            ]
            queue_wait_times.extend(batch_queue_wait_ms)

            input_batch, actual_batch_size = make_migraphx_input_batch(
                batch_items,
                expected_dtype=expected_dtype,
                compiled_batch_size=configured_batch_size,
                shared_input_slots=shared_input_slots,
            )

            with Timer() as t_inf:
                with tracer.range(f"migraphx_run_batch{actual_batch_size}"):
                    results = model.run({"input": input_batch})
            inference_times.append(t_inf.ms)
            batch_runs += 1
            batch_sizes_seen.append(actual_batch_size)

            with Timer() as t_dec:
                with tracer.range("decode_migraphx_outputs"):
                    heatmaps_bhwc, pafs_bhwc = decode_migraphx_batch_outputs(
                        results,
                        out_h,
                        out_w,
                        shared_dtype,
                        batch_size=input_batch.shape[0],
                    )
                    infer_done_ts = time.perf_counter()
                    out_items = build_inference_output_items_from_batch(
                        batch_items=batch_items,
                        heatmaps_bhwc=heatmaps_bhwc,
                        pafs_bhwc=pafs_bhwc,
                        infer_done_ts=infer_done_ts,
                        inference_ms_total=t_inf.ms,
                        decode_ms_total=0.0,
                        queue_wait_times_ms=batch_queue_wait_ms,
                    )
            decode_times.append(t_dec.ms)
            for out_item in out_items:
                out_item["decode_ms"] = float(t_dec.ms) / float(max(1, len(out_items)))
                out_item["batch_decode_ms"] = float(t_dec.ms)
                out_item["collector_stale_records_discarded_pre_batch"] = int(batch_stale_discards)
                out_item["collector_stale_records_discarded_pre_batch_cumulative"] = int(stale_records_discarded_pre_batch)
                _copy_item_to_shared_map(out_item)

            trace_print(
                trace_log_every,
                batch_runs,
                f"[TRACE infer:{worker_id} pid={os.getpid()} latest handoff] "
                f"batch_run={batch_runs} actual_batch={actual_batch_size} "
                f"queue_avg={mean(batch_queue_wait_ms):.2f}ms infer={t_inf.ms:.2f}ms "
                f"decode={t_dec.ms:.2f}ms stale_pre_batch={batch_stale_discards}",
            )

            packet = {
                "split_batch_packet": True,
                "variant": "split_hip2_batch_handoff",
                "batch_items": out_items,
                "batch_size": int(len(out_items)),
                "compiled_batch_size": int(configured_batch_size),
                "infer_done_ts": float(infer_done_ts),
                "batch_inference_ms": float(t_inf.ms),
                "batch_decode_ms": float(t_dec.ms),
            }

            q_idx = (batch_runs - 1) % max(1, len(out_queues))
            dropped_packet = latest_put_with_dropped(out_queues[q_idx], packet)
            if dropped_packet is packet:
                replaced_before_post += _release_packet(packet, free_map_slots=free_map_slots, post_pending=post_pending)
                continue
            if dropped_packet is not None:
                replaced_before_post += _release_packet(dropped_packet, free_map_slots=free_map_slots, post_pending=post_pending)

            if backpressure_mode != "off" and post_pending is not None:
                now = time.perf_counter()
                for out_item in out_items:
                    cam_id = int(out_item["camera_id"])
                    post_pending[cam_id] = 1
                    if post_pending_ts is not None:
                        post_pending_ts[cam_id] = now
            processed += len(out_items)
            packet_runs += 1

        infer_done[worker_id] = 1
        stats_q.put(  # type: ignore[union-attr]
            {
                "stage": "inference",
                "buffer_mode": "latest",
                "worker_id": worker_id,
                "processed": processed,
                "batch_runs": batch_runs,
                "batch_packets": packet_runs,
                "split_batch_handoff": True,
                "avg_real_batch_size": mean(batch_sizes_seen),
                "p95_real_batch_size": percentile(batch_sizes_seen, 95),
                "configured_migraphx_batch_size": configured_batch_size,
                "migraphx_batch_timeout_ms": float(migraphx_batch_timeout_ms),
                "replaced_before_post": replaced_before_post,
                "collector_coalesce": bool(collector_coalesce),
                "stale_records_discarded_pre_batch": int(stale_records_discarded_pre_batch),
                "batches_with_stale_records_pre_batch": int(batches_with_stale_discards),
                "avg_stale_records_discarded_pre_batch_per_batch": mean(stale_discards_per_batch),
                "p95_stale_records_discarded_pre_batch_per_batch": percentile(stale_discards_per_batch, 95),
                "shared_map_misses": shared_map_misses,
                "backpressure_mode": backpressure_mode,
                "backpressure_enabled": backpressure_mode != "off",
                "skipped_due_backpressure": skipped_due_backpressure,
                "soft_backpressure_overrides": soft_overrides,
                "throttle_skips": throttle_skips,
                "backpressure_idle_loops": backpressure_idle_loops,
                "avg_queue_pre_to_infer_ms": mean(queue_wait_times),
                "p95_queue_pre_to_infer_ms": percentile(queue_wait_times, 95),
                "avg_inference_ms": mean(inference_times),
                "p95_inference_ms": percentile(inference_times, 95),
                "avg_decode_ms": mean(decode_times),
                "p95_decode_ms": percentile(decode_times, 95),
                "merged_pose_fused_pruned": False,
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        close_shared_map_views(shared_handles)
        close_shared_map_views(shared_input_handles)
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
            f"packets={packet_runs} avg_real_batch={mean(batch_sizes_seen):.2f} "
            f"replaced_before_post={replaced_before_post} backpressure_skips={skipped_due_backpressure} "
            f"soft_overrides={soft_overrides} throttle_skips={throttle_skips} "
            f"stale_pre_batch={stale_records_discarded_pre_batch}",
            flush=True,
        )

    except Exception:
        try:
            infer_done[worker_id] = 1
        except Exception:
            pass
        if error_q is not None:
            error_q.put({"stage": "inference", "worker_id": worker_id, "traceback": traceback.format_exc()})


def postprocess_latest_worker_patched(
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
    if not _is_handoff_variant(user_variant):
        return _ORIGINALS["postprocess_latest_worker"](
            worker_id=worker_id,
            user_variant=user_variant,
            in_queues=in_queues,
            infer_done=infer_done,
            post_pending=post_pending,
            last_processed_ts=last_processed_ts,
            result_q=result_q,
            stats_q=stats_q,
            error_q=error_q,
            torch_device=torch_device,
            require_gpu=require_gpu,
            max_keypoints=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            nms_impl=nms_impl,
            gpu_compute_dtype=gpu_compute_dtype,
            grid_q=grid_q,
            render_output=render_output,
            poll_sleep_s=poll_sleep_s,
            migraphx_nms_mxr=migraphx_nms_mxr,
            migraphx_nms_cache_dir=migraphx_nms_cache_dir,
            shared_map_descs=shared_map_descs,
            free_map_slots=free_map_slots,
            prealloc_resize_buffers=prealloc_resize_buffers,
            gpu_nms_batch_size=gpu_nms_batch_size,
            gpu_nms_batch_timeout_ms=gpu_nms_batch_timeout_ms,
            trace_log_every=trace_log_every,
            roctx_enabled=roctx_enabled,
        )

    canonical = registry_mode = _canonical_handoff_variant(user_variant)
    try:
        from simulation.grid_video import draw_poses_on_frame
        from simulation.queues import all_done, all_queues_empty
        from simulation.shared_memory import close_shared_map_views, open_shared_map_buffers, release_shared_slot_from_item
        from simulation.system import configure_child_cpu_runtime
        from simulation.tracing import RocTxTracer, trace_print
        from simulation.utils import mean, percentile

        tracer = RocTxTracer(roctx_enabled, f"post:{worker_id}:pid:{os.getpid()}:{canonical}")
        tracer.mark("worker_start")
        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        runtime = split_base._load_split_runtime(use_hip2=True)
        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)

        nqueues = len(in_queues)
        next_q = worker_id % max(1, nqueues)
        processed = 0
        packets_processed = 0
        post_times: List[float] = []
        queue_wait_times: List[float] = []
        e2e_times: List[float] = []
        packet_sizes: List[int] = []
        t_worker_start = time.perf_counter()

        print(
            f"[POST:{worker_id}] {canonical} pair_backend=hip2 handoff=batch_packet "
            f"sp={_split_env_int('STREAM_SPLIT_HIP_SMART_PROPOSALS', 32)} "
            f"lr={_split_env_int('STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS', 4)}",
            flush=True,
        )

        def _get_next_packet():
            nonlocal next_q
            scanned = 0
            while scanned < nqueues:
                qid = next_q
                next_q = (next_q + 1) % max(1, nqueues)
                scanned += 1
                try:
                    return in_queues[qid].get_nowait()
                except py_queue.Empty:
                    continue
            return None

        def _maps_for(batch_item: Dict[str, Any]):
            slot_id = batch_item.get("shared_map_slot") if isinstance(batch_item, dict) else None
            if slot_id is not None and int(slot_id) in shared_slots:
                slot = shared_slots[int(slot_id)]
                return slot["heat"], slot["paf"]
            return batch_item["heatmaps"], batch_item["pafs"]

        while True:
            packet = _get_next_packet()
            if packet is None:
                if all_done(infer_done) and all_queues_empty(in_queues):
                    break
                time.sleep(poll_sleep_s)
                continue

            if not isinstance(packet, dict) or not packet.get("split_batch_packet"):
                raise RuntimeError(f"{canonical} expected split_batch_packet, got {type(packet)}")

            batch_items = list(packet.get("batch_items", []))
            if not batch_items:
                continue
            packet_sizes.append(len(batch_items))

            post_start = time.perf_counter()
            try:
                map_pairs = [_maps_for(bi) for bi in batch_items]
                with tracer.range(f"postprocess_{canonical}_packet{len(batch_items)}"):
                    batch_outputs = split_base._run_split_hip_smart_batch(
                        batch_items=batch_items,
                        map_pairs=map_pairs,
                        runtime=runtime,
                        threshold=threshold,
                        use_hip2=True,
                    )
            except Exception:
                for bi in batch_items:
                    release_shared_slot_from_item(bi, free_map_slots)
                raise

            post_done_batch = time.perf_counter()
            for bi, out in zip(batch_items, batch_outputs):
                # Use the same completion timestamp for every frame in the handoff packet;
                # all of them become available only after the packet postprocess finishes.
                row = split_base._make_row(
                    bi=bi,
                    out=out,
                    canonical=canonical,
                    registry_mode=registry_mode,
                    worker_id=worker_id,
                    post_start=post_start,
                    post_done=post_done_batch,
                )
                row["split_batch_handoff"] = 1
                row["split_handoff_packet_size"] = int(len(batch_items))
                row["split_handoff_queue_id"] = int(packet.get("queue_id", -1))
                row["batch_post_wall_ms"] = (post_done_batch - post_start) * 1000.0
                post_times.append(row["post_ms"])
                queue_wait_times.append(row["queue_infer_to_post_ms"])
                e2e_times.append(row["e2e_ms"])

                if render_output and grid_q is not None and "frame_bgr" in bi:
                    frame_out = bi["frame_bgr"].copy()
                    draw_poses_on_frame(frame_out, out.pose_entries, out.all_keypoints)
                    try:
                        grid_q.put_nowait(
                            {
                                "camera_id": row["camera_id"],
                                "frame_id": row["frame_id"],
                                "source": row["source"],
                                "frame_bgr": frame_out,
                                "e2e_ms": row["e2e_ms"],
                                "post_ms": row["post_ms"],
                                "num_poses": row["num_poses"],
                                "num_keypoints": row["num_keypoints"],
                            }
                        )
                    except py_queue.Full:
                        pass

                result_q.put(row)
                cam_done_id = int(bi["camera_id"])
                if post_pending is not None:
                    post_pending[cam_done_id] = 0
                if last_processed_ts is not None:
                    last_processed_ts[cam_done_id] = time.perf_counter()
                release_shared_slot_from_item(bi, free_map_slots)
                processed += 1
                trace_print(
                    trace_log_every,
                    processed,
                    f"[TRACE post:{worker_id} pid={os.getpid()} latest handoff] "
                    f"processed={processed} cam={row['camera_id']} frame={row['frame_id']} "
                    f"packet={len(batch_items)} post={row['post_ms']:.2f}ms e2e={row['e2e_ms']:.2f}ms",
                )
            packets_processed += 1

        close_shared_map_views(shared_handles)
        stats_q.put(
            {
                "stage": "postprocess",
                "buffer_mode": "latest",
                "worker_id": worker_id,
                "variant": canonical,
                "registry_mode": registry_mode,
                "processed": processed,
                "split_batch_handoff": True,
                "handoff_packets_processed": packets_processed,
                "avg_handoff_packet_size": mean(packet_sizes),
                "p95_handoff_packet_size": percentile(packet_sizes, 95),
                "avg_queue_infer_to_post_ms": mean(queue_wait_times),
                "p95_queue_infer_to_post_ms": percentile(queue_wait_times, 95),
                "avg_post_ms": mean(post_times),
                "p95_post_ms": percentile(post_times, 95),
                "avg_e2e_ms": mean(e2e_times),
                "p95_e2e_ms": percentile(e2e_times, 95),
                "split_pair_backend": "hip2",
                "split_mxr2": "",
                "split_mxr2_batch_size": _split_env_int("STREAM_SPLIT_HIP_BATCH_SIZE", 4),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(f"[POST:{worker_id}] Done. processed={processed} packets={packets_processed}", flush=True)
    except Exception:
        error_q.put({"stage": "postprocess", "worker_id": worker_id, "traceback": traceback.format_exc()})


def _patch_resolve_registry_mode() -> None:
    import simulation.postprocess_modes as modes
    import simulation.runner as runner
    import simulation.workers.postprocess as post_mod

    previous = modes.resolve_registry_mode
    _ORIGINALS.setdefault("resolve_registry_mode", previous)

    def resolve_registry_mode_patched(user_mode: str):
        if _is_handoff_variant(user_mode):
            canonical = _canonical_handoff_variant(user_mode)
            return canonical, canonical, False
        return previous(user_mode)

    modes.resolve_registry_mode = resolve_registry_mode_patched
    runner.resolve_registry_mode = resolve_registry_mode_patched
    post_mod.resolve_registry_mode = resolve_registry_mode_patched


def _patch_run() -> None:
    import simulation.cli as cli
    import simulation.runner as runner

    previous_run = runner.run
    original_runner_run = split_base._ORIGINALS.get("runner_run", previous_run)
    _ORIGINALS.setdefault("runner_run", previous_run)

    def run_patched(args):
        if not _is_handoff_variant(getattr(args, "variant", "")):
            os.environ.pop("STREAM_SPLIT_HIP_BATCH_HANDOFF", None)
            return previous_run(args)

        os.environ["STREAM_SPLIT_HIP_BATCH_HANDOFF"] = "1"
        os.environ["STREAM_SHARED_HEATMAP_CHANNELS"] = "18"
        os.environ["STREAM_SPLIT_HIP_MXR2"] = str(getattr(args, "split_mxr2", split_base._split_env_path()))
        os.environ["STREAM_SPLIT_HIP_BATCH_SIZE"] = str(int(getattr(args, "split_mxr2_batch_size", 4)))
        os.environ["STREAM_SPLIT_HIP_TIMEOUT_MS"] = str(float(getattr(args, "split_batch_timeout_ms", 4.0)))
        os.environ["STREAM_SPLIT_HIP_PAF_BACKEND"] = str(getattr(args, "split_paf_backend", "hip_host"))
        os.environ["STREAM_SPLIT_HIP_SMART_PROPOSALS"] = str(int(getattr(args, "smart_proposals", 32)))
        os.environ["STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS"] = str(int(getattr(args, "smart_local_radius", 4)))
        os.environ["STREAM_SPLIT_HIP_SMART_LOWRES_NMS_RADIUS"] = str(int(getattr(args, "smart_lowres_nms_radius", 1)))
        os.environ["STREAM_SPLIT_HIP_TOPK"] = str(int(getattr(args, "max_keypoints", 20)))
        os.environ["STREAM_SPLIT_HIP_LIMB_TOPM"] = str(int(getattr(args, "max_keypoints", 20)))
        return original_runner_run(args)

    runner.run = run_patched
    cli.run = run_patched


def _patch_workers() -> None:
    import simulation.runner as runner
    import simulation.workers.inference as infer_mod
    import simulation.workers.postprocess as post_mod

    _ORIGINALS.setdefault("inference_latest_worker", infer_mod.inference_latest_worker)
    _ORIGINALS.setdefault("postprocess_latest_worker", post_mod.postprocess_latest_worker)

    infer_mod.inference_latest_worker = inference_latest_worker_patched
    runner.inference_latest_worker = inference_latest_worker_patched
    post_mod.postprocess_latest_worker = postprocess_latest_worker_patched
    runner.postprocess_latest_worker = postprocess_latest_worker_patched


def apply_patch() -> None:
    if _ORIGINALS.get("applied"):
        return

    split_base.apply_patch()

    import simulation.cli  # noqa: F401
    import simulation.runner  # noqa: F401
    import simulation.workers.inference  # noqa: F401
    import simulation.workers.postprocess  # noqa: F401

    _patch_resolve_registry_mode()
    _patch_run()
    _patch_workers()
    _ORIGINALS["applied"] = True
