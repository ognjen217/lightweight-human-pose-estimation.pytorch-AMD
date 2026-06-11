"""MIGraphX inference worker process entrypoints."""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..migraphx_io import (
    build_inference_output_items_from_batch,
    decode_migraphx_batch_outputs,
    make_migraphx_input_batch,
)
from ..postprocess_modes import build_merged_pose_fused_pruned_items_from_batch
from ..queues import all_done, all_queues_empty, collect_queue_batch, latest_put_with_dropped
from ..shared_memory import (
    close_shared_map_views,
    open_shared_input_buffers,
    open_shared_map_buffers,
    release_shared_slot_from_item,
)
from ..system import configure_child_cpu_runtime
from ..tracing import RocTxTracer, trace_print
from ..utils import Timer, mean, percentile


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
    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,
    migraphx_batch_size: int = 1,
    migraphx_batch_timeout_ms: float = 0.0,
    merged_pose_fused_pruned: bool = False,
    trace_log_every: int = 0,
    roctx_enabled: bool = False,
) -> None:
    try:


        tracer = RocTxTracer(roctx_enabled, f"infer:{worker_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")


        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        with tracer.range("migraphx_load"):
            model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)
        print(
            f"[INFER:{worker_id}] MIGraphX inference batch size={int(migraphx_batch_size)} "
            f"timeout={float(migraphx_batch_timeout_ms):.2f} ms",
            flush=True,
        )
        if merged_pose_fused_pruned:
            print(
                f"[INFER:{worker_id}] Running MERGED pose+fused-pruned MXR; "
                "postprocess workers will run CPU-only pruned assembly.",
                flush=True,
            )

        out_h = target_h // stride
        out_w = target_w // stride
        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        shared_map_misses = 0
        processed = 0
        batch_runs = 0
        batch_sizes_seen: List[int] = []
        inference_times: List[float] = []
        decode_times: List[float] = []
        queue_wait_times: List[float] = []
        t_worker_start = time.perf_counter()

        while True:
            item = in_q.get()
            if item is None:
                break

            batch_items, saw_stop = collect_queue_batch(
                first_item=item,
                in_q=in_q,
                batch_size=migraphx_batch_size,
                batch_timeout_ms=migraphx_batch_timeout_ms,
            )

            infer_start = time.perf_counter()
            batch_queue_wait_ms = [
                (infer_start - float(bi.get("preprocess_done_ts", infer_start))) * 1000.0
                for bi in batch_items
            ]
            queue_wait_times.extend(batch_queue_wait_ms)

            input_batch, actual_batch_size = make_migraphx_input_batch(
                batch_items,
                expected_dtype=expected_dtype,
                compiled_batch_size=migraphx_batch_size,
            )

            with Timer() as t_inf:
                with tracer.range(f"migraphx_run_batch{actual_batch_size}"):
                    results = model.run({"input": input_batch})
            inference_times.append(t_inf.ms)
            batch_runs += 1
            batch_sizes_seen.append(actual_batch_size)

            with Timer() as t_dec:
                if merged_pose_fused_pruned:
                    with tracer.range("decode_merged_pose_fused_pruned_outputs"):
                        out_items = build_merged_pose_fused_pruned_items_from_batch(
                            batch_items=batch_items,
                            results=results,
                            infer_done_ts=time.perf_counter(),
                            inference_ms_total=t_inf.ms,
                            decode_ms_total=0.0,  # filled immediately below after Timer exits
                            queue_wait_times_ms=batch_queue_wait_ms,
                        )
                else:
                    with tracer.range("decode_migraphx_outputs"):
                        heatmaps_bhwc, pafs_bhwc = decode_migraphx_batch_outputs(
                            results,
                            out_h,
                            out_w,
                            shared_dtype,
                            batch_size=input_batch.shape[0],
                        )
                        out_items = build_inference_output_items_from_batch(
                            batch_items=batch_items,
                            heatmaps_bhwc=heatmaps_bhwc,
                            pafs_bhwc=pafs_bhwc,
                            infer_done_ts=time.perf_counter(),
                            inference_ms_total=t_inf.ms,
                            decode_ms_total=0.0,  # filled immediately below after Timer exits
                            queue_wait_times_ms=batch_queue_wait_ms,
                        )
            decode_times.append(t_dec.ms)
            for _out_item in out_items:
                _out_item["decode_ms"] = float(t_dec.ms) / float(max(1, len(out_items)))
                _out_item["batch_decode_ms"] = float(t_dec.ms)


            trace_print(
                trace_log_every,
                batch_runs,
                f"[TRACE infer:{worker_id} pid={os.getpid()}] "
                f"batch_run={batch_runs} actual_batch={actual_batch_size} "
                f"queue_avg={mean(batch_queue_wait_ms):.2f}ms infer={t_inf.ms:.2f}ms decode={t_dec.ms:.2f}ms",
            )

            for out_item in out_items:
                if out_item.get("merged_pose_fused_pruned_precomputed") or out_item.get("fused_pruned_precomputed"):
                    out_item.pop("heatmaps", None)
                    out_item.pop("pafs", None)
                elif shared_slots and free_map_slots is not None:
                    slot_id = None
                    try:
                        slot_id = int(free_map_slots.get(timeout=0.05))
                        slot = shared_slots[slot_id]
                        if slot["heat"].shape != out_item["heatmaps"].shape or slot["paf"].shape != out_item["pafs"].shape:
                            raise ValueError(
                                f"shared-map slot shape mismatch: "
                                f"heat {slot['heat'].shape}!={out_item['heatmaps'].shape}, "
                                f"paf {slot['paf'].shape}!={out_item['pafs'].shape}"
                            )
                        np.copyto(slot["heat"], out_item.pop("heatmaps"), casting="same_kind")
                        np.copyto(slot["paf"], out_item.pop("pafs"), casting="same_kind")
                        out_item["shared_map_slot"] = slot_id
                    except Exception:
                        shared_map_misses += 1
                        if slot_id is not None:
                            try:
                                free_map_slots.put_nowait(slot_id)
                            except Exception:
                                pass
                out_q.put(out_item)
                processed += 1

            if saw_stop:
                break

        close_shared_map_views(shared_handles)
        stats_q.put(
            {
                "stage": "inference",
                "worker_id": worker_id,
                "processed": processed,
                "batch_runs": batch_runs,
                "avg_real_batch_size": mean(batch_sizes_seen),
                "p95_real_batch_size": percentile(batch_sizes_seen, 95),
                "configured_migraphx_batch_size": int(migraphx_batch_size),
                "migraphx_batch_timeout_ms": float(migraphx_batch_timeout_ms),
                "shared_map_misses": shared_map_misses,
                "avg_queue_pre_to_infer_ms": mean(queue_wait_times),
                "p95_queue_pre_to_infer_ms": percentile(queue_wait_times, 95),
                "avg_inference_ms": mean(inference_times),
                "p95_inference_ms": percentile(inference_times, 95),
                "avg_decode_ms": mean(decode_times),
                "p95_decode_ms": percentile(decode_times, 95),
                "merged_pose_fused_pruned": bool(merged_pose_fused_pruned),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
            f"avg_real_batch={mean(batch_sizes_seen):.2f}",
            flush=True,
        )

    except Exception:
        error_q.put({"stage": "inference", "worker_id": worker_id, "traceback": traceback.format_exc()})

def inference_latest_worker(
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

    """Round-robin MIGraphX worker over newest-frame slots, one slot per camera.

    backpressure_mode:
      "off"    – never skip (maximum throughput, results may be overwritten).
      "strict" – skip camera while post_pending flag is set (original behaviour).
      "soft"   – skip only while pending AND result is fresher than
                 max_pending_age_ms; allows re-inference when a result is stale.
    target_period_s > 0 enables per-camera rate throttling: a camera is not
    eligible for inference until at least target_period_s seconds have elapsed
    since its last completed postprocess result.
    """
    try:
        tracer = RocTxTracer(roctx_enabled, f"infer:{worker_id}:pid:{os.getpid()}")
        tracer.mark("worker_start")

        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        with tracer.range("migraphx_load"):
            model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)
        print(
            f"[INFER:{worker_id}] MIGraphX inference batch size={int(migraphx_batch_size)} "
            f"timeout={float(migraphx_batch_timeout_ms):.2f} ms "
            f"collector_coalesce={bool(collector_coalesce)}",
            flush=True,
        )
        if merged_pose_fused_pruned:
            print(
                f"[INFER:{worker_id}] Running MERGED pose+fused-pruned MXR in latest worker; "
                "postprocess workers will run CPU-only pruned assembly.",
                flush=True,
            )

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
            """Return newest queued metadata for a camera, discarding older records.

            In the current latest path each camera queue is normally maxsize=1,
            so this is usually a no-op. It still makes the collector robust to
            races, future queue-size changes, and duplicate camera records staged
            while a batch is being assembled.
            """
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
            """Add candidate or replace an older staged item from same camera.

            Returns True only when real batch size increased.
            """
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

        def _get_next_item() -> Tuple[Optional[Dict[str, Any]], int]:
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
                # Last-moment coalescing: while waiting for B4/B8 fill, a selected
                # camera may publish a newer metadata record. Replace the staged
                # record before input batch assembly.
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
                if merged_pose_fused_pruned:
                    with tracer.range("decode_merged_pose_fused_pruned_outputs"):
                        out_items = build_merged_pose_fused_pruned_items_from_batch(
                            batch_items=batch_items,
                            results=results,
                            infer_done_ts=time.perf_counter(),
                            inference_ms_total=t_inf.ms,
                            decode_ms_total=0.0,  # filled immediately below after Timer exits
                            queue_wait_times_ms=batch_queue_wait_ms,
                        )
                else:
                    with tracer.range("decode_migraphx_outputs"):
                        heatmaps_bhwc, pafs_bhwc = decode_migraphx_batch_outputs(
                            results,
                            out_h,
                            out_w,
                            shared_dtype,
                            batch_size=input_batch.shape[0],
                        )
                        out_items = build_inference_output_items_from_batch(
                            batch_items=batch_items,
                            heatmaps_bhwc=heatmaps_bhwc,
                            pafs_bhwc=pafs_bhwc,
                            infer_done_ts=time.perf_counter(),
                            inference_ms_total=t_inf.ms,
                            decode_ms_total=0.0,  # filled immediately below after Timer exits
                            queue_wait_times_ms=batch_queue_wait_ms,
                        )
            decode_times.append(t_dec.ms)
            for _out_item in out_items:
                _out_item["decode_ms"] = float(t_dec.ms) / float(max(1, len(out_items)))
                _out_item["batch_decode_ms"] = float(t_dec.ms)


            trace_print(
                trace_log_every,
                batch_runs,
                f"[TRACE infer:{worker_id} pid={os.getpid()} latest] "
                f"batch_run={batch_runs} actual_batch={actual_batch_size} "
                f"queue_avg={mean(batch_queue_wait_ms):.2f}ms infer={t_inf.ms:.2f}ms "
                f"decode={t_dec.ms:.2f}ms stale_pre_batch={batch_stale_discards}",
            )

            for out_item in out_items:
                cam_id = int(out_item["camera_id"])
                out_item["collector_stale_records_discarded_pre_batch"] = int(batch_stale_discards)
                out_item["collector_stale_records_discarded_pre_batch_cumulative"] = int(
                    stale_records_discarded_pre_batch
                )

                if out_item.get("merged_pose_fused_pruned_precomputed") or out_item.get("fused_pruned_precomputed"):
                    out_item.pop("heatmaps", None)
                    out_item.pop("pafs", None)
                elif shared_slots and free_map_slots is not None:
                    slot_id = None
                    try:
                        slot_id = int(free_map_slots.get_nowait())
                        slot = shared_slots[slot_id]
                        if slot["heat"].shape != out_item["heatmaps"].shape or slot["paf"].shape != out_item["pafs"].shape:
                            raise ValueError(
                                f"shared-map slot shape mismatch: "
                                f"heat {slot['heat'].shape}!={out_item['heatmaps'].shape}, "
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

                if backpressure_mode != "off" and post_pending is not None:
                    post_pending[cam_id] = 1
                    if post_pending_ts is not None:
                        post_pending_ts[cam_id] = time.perf_counter()
                    try:
                        out_queues[cam_id].put_nowait(out_item)
                    except py_queue.Full:
                        dropped_item = latest_put_with_dropped(out_queues[cam_id], out_item)
                        if dropped_item is not None:
                            replaced_before_post += 1
                            release_shared_slot_from_item(dropped_item, free_map_slots)
                else:
                    dropped_item = latest_put_with_dropped(out_queues[cam_id], out_item)
                    if dropped_item is not None:
                        replaced_before_post += 1
                        release_shared_slot_from_item(dropped_item, free_map_slots)
                processed += 1


        infer_done[worker_id] = 1
        stats_q.put( # type: ignore
            {
                "stage": "inference",
                "buffer_mode": "latest",
                "worker_id": worker_id,
                "processed": processed,
                "batch_runs": batch_runs,
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
                "merged_pose_fused_pruned": bool(merged_pose_fused_pruned),
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        close_shared_map_views(shared_handles)
        close_shared_map_views(shared_input_handles)
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
            f"avg_real_batch={mean(batch_sizes_seen):.2f} replaced_before_post={replaced_before_post} "
            f"backpressure_skips={skipped_due_backpressure} soft_overrides={soft_overrides} "
            f"throttle_skips={throttle_skips} stale_pre_batch={stale_records_discarded_pre_batch}",
            flush=True,
        )

    except Exception:
        try:
            infer_done[worker_id] = 1
        except Exception:
            pass
        error_q.put({"stage": "inference", "worker_id": worker_id, "traceback": traceback.format_exc()})
