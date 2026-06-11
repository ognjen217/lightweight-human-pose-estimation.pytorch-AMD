"""Main queue/latest stream orchestration."""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List

import multiprocessing as mp

from .camera import camera_sources
from .defaults import DEFAULT_VIDEO_CYCLE
from .grid_video import grid_video_writer_worker
from .postprocess_modes import (
    compile_migraphx_nms_for_stream_if_requested,
    is_merged_pose_fused_pruned_variant,
    resolve_registry_mode,
)
from .reporting import apply_warmup_filter, print_summary, summarize, write_detailed_csv, write_summary_json
from .shared_memory import (
    close_shared_input_buffers,
    close_shared_map_buffers,
    create_shared_input_buffers,
    create_shared_map_buffers,
)
from .system import (
    SysMonitor,
    _process_pid_groups,
    _register_processes,
    configure_worker_thread_env,
    pin_stream_processes,
    print_affinity_report,
    print_system_profile,
)
from .workers.inference import inference_latest_worker, inference_worker
from .workers.postprocess import postprocess_latest_worker, postprocess_worker
from .workers.preprocess import camera_preprocess_latest_worker, camera_preprocess_worker


def run_queue(args) -> Dict[str, Any]:


    if getattr(args, "allow_ptrace_attach", False):
        os.environ["STREAM_ALLOW_PTRACE_ATTACH"] = "1"
    else:
        os.environ.pop("STREAM_ALLOW_PTRACE_ATTACH", None)
    configure_worker_thread_env(args.worker_threads)
    ctx = mp.get_context(getattr(args, "mp_start_method", "spawn"))


    configure_worker_thread_env(args.worker_threads)
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

    shared_map_descs: List[Dict[str, Any]] = []
    shared_map_handles: List[shared_memory.SharedMemory] = []
    free_map_slots = None
    if getattr(args, "shared_map_slots", 0) > 0:
        out_h = args.target_height // args.stride
        out_w = args.target_width // args.stride
        shared_map_descs, shared_map_handles = create_shared_map_buffers(
            int(args.shared_map_slots), out_h, out_w, args.shared_dtype
        )
        free_map_slots = ctx.Queue(maxsize=int(args.shared_map_slots))
        for slot_id in range(int(args.shared_map_slots)):
            free_map_slots.put(slot_id)

    camera_procs = []
    grid_procs = []
    infer_procs = []
    post_procs = []
    monitor = SysMonitor(args.profile_interval_s) if args.profile_system else None
    system_profile: Dict[str, Any] = {}
    if monitor is not None:
        monitor.start()

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
                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,
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
                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,


                migraphx_batch_size=args.migraphx_batch_size,
                migraphx_batch_timeout_ms=args.migraphx_batch_timeout_ms,
                merged_pose_fused_pruned=(registry_mode == "mx_merged_pose_fused_pruned"),
                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,


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
                torch_device=args.torch_device,
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
                prealloc_resize_buffers=args.prealloc_resize_buffers,


                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,


            ),
            name=f"postprocess_{worker_id}",
        )
        p.start()
        post_procs.append(p)

    pid_groups = _process_pid_groups(camera_procs, infer_procs, post_procs, grid_procs)
    pin_stream_processes(pid_groups, args)
    _register_processes(monitor, pid_groups)
    if args.pin_cpus or args.report_affinity:
        print_affinity_report(pid_groups)

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
        if monitor is not None:
            system_profile = monitor.stop()


        close_shared_map_buffers(shared_map_handles)



    wall_s = time.perf_counter() - t0
    summary_rows, warmup_info = apply_warmup_filter(rows, args)
    summary = summarize(summary_rows, stage_stats, wall_s)
    summary.update(warmup_info)
    summary["variant"] = canonical
    summary["registry_mode"] = registry_mode
    summary["model"] = args.model
    summary["num_cameras"] = args.num_cameras
    summary["infer_workers"] = args.infer_workers
    summary["post_workers"] = args.post_workers
    summary["mp_start_method"] = getattr(args, "mp_start_method", "spawn")
    summary["queue_policy"] = args.queue_policy
    summary["buffer_mode"] = "queue"
    summary["shared_map_slots"] = getattr(args, "shared_map_slots", 0)
    summary["migraphx_batch_size"] = getattr(args, "migraphx_batch_size", 1)
    summary["migraphx_batch_timeout_ms"] = getattr(args, "migraphx_batch_timeout_ms", 0.0)
    summary["merged_pose_fused_pruned"] = (registry_mode == "mx_merged_pose_fused_pruned")
    summary["roctx"] = bool(getattr(args, "roctx", False))
    summary["trace_log_every"] = int(getattr(args, "trace_log_every", 0) or 0)
    summary["allow_ptrace_attach"] = bool(getattr(args, "allow_ptrace_attach", False))
    summary["grid_video"] = args.grid_video
    summary["grid_rows"] = args.grid_rows if args.grid_video else 0
    summary["grid_cols"] = args.grid_cols if args.grid_video else 0
    summary["realtime"] = args.realtime
    summary["camera_sources"] = sources
    if system_profile:
        summary["system_profile"] = system_profile

    print_summary(summary)
    print_system_profile(system_profile)
    write_detailed_csv(args.detailed_csv, summary_rows)
    write_summary_json(args.summary_json, summary)
    return summary


def run_latest(args) -> Dict[str, Any]:
    """Run the pipeline with newest-frame-only slots per camera between stages."""


    if getattr(args, "allow_ptrace_attach", False):
        os.environ["STREAM_ALLOW_PTRACE_ATTACH"] = "1"
    else:
        os.environ.pop("STREAM_ALLOW_PTRACE_ATTACH", None)
    configure_worker_thread_env(args.worker_threads)
    ctx = mp.get_context(getattr(args, "mp_start_method", "spawn"))


    configure_worker_thread_env(args.worker_threads)
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
    post_pending_ts = ctx.Array("d", [0.0] * args.num_cameras)
    last_processed_ts = ctx.Array("d", [0.0] * args.num_cameras)
    # --disable-backpressure is a legacy alias for --backpressure-mode off.
    backpressure_mode = "off" if bool(getattr(args, "disable_backpressure", False)) else args.backpressure_mode
    backpressure_enabled = backpressure_mode != "off"
    target_period_s = (
        1.0 / args.target_output_fps_per_camera
        if getattr(args, "target_output_fps_per_camera", 0.0) > 0.0
        else 0.0
    )

    shared_input_descs: List[Dict[str, Any]] = []
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
    if getattr(args, "shared_map_slots", 0) > 0:
        out_h = args.target_height // args.stride
        out_w = args.target_width // args.stride
        shared_map_descs, shared_map_handles = create_shared_map_buffers(
            int(args.shared_map_slots), out_h, out_w, args.shared_dtype
        )
        free_map_slots = ctx.Queue(maxsize=int(args.shared_map_slots))
        for slot_id in range(int(args.shared_map_slots)):
            free_map_slots.put(slot_id)

    camera_procs = []
    infer_procs = []
    post_procs = []
    grid_procs = []
    monitor = SysMonitor(args.profile_interval_s) if args.profile_system else None
    system_profile: Dict[str, Any] = {}
    if monitor is not None:
        monitor.start()

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
    print(f"Backpressure:  {backpressure_mode}")
    if backpressure_mode == "soft":
        print(f"Max pending:   {args.max_pending_age_ms:.0f} ms")
    if target_period_s > 0.0:
        print(f"Target FPS/cam:{args.target_output_fps_per_camera:.2f}  (period={target_period_s*1000:.0f} ms)")
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
                shared_input_descs=shared_input_descs,
                shared_input_dtype=args.shared_input_dtype,
                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,
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
                backpressure_mode=backpressure_mode,
                max_pending_age_ms=args.max_pending_age_ms,
                post_pending_ts=post_pending_ts,
                last_processed_ts=last_processed_ts,
                target_period_s=target_period_s,
                stats_q=stats_q,
                error_q=error_q,
                target_w=args.target_width,
                target_h=args.target_height,
                stride=args.stride,
                shared_dtype=args.shared_dtype,
                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,
                shared_input_descs=shared_input_descs,


                migraphx_batch_size=args.migraphx_batch_size,
                migraphx_batch_timeout_ms=args.migraphx_batch_timeout_ms,
                merged_pose_fused_pruned=(registry_mode == "mx_merged_pose_fused_pruned"),
                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,


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
                last_processed_ts=last_processed_ts,
                result_q=result_q,
                stats_q=stats_q,
                error_q=error_q,
                torch_device=args.torch_device,
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
                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,
                prealloc_resize_buffers=args.prealloc_resize_buffers,
                gpu_nms_batch_size=args.gpu_nms_batch_size,
                gpu_nms_batch_timeout_ms=args.gpu_nms_batch_timeout_ms,


                trace_log_every=args.trace_log_every,
                roctx_enabled=args.roctx,


            ),
            name=f"postprocess_latest_{worker_id}",
        )
        p.start()
        post_procs.append(p)

    pid_groups = _process_pid_groups(camera_procs, infer_procs, post_procs, grid_procs)
    pin_stream_processes(pid_groups, args)
    _register_processes(monitor, pid_groups)
    if args.pin_cpus or args.report_affinity:
        print_affinity_report(pid_groups)

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
        if monitor is not None:
            system_profile = monitor.stop()
        close_shared_map_buffers(shared_map_handles)
        close_shared_input_buffers(shared_input_handles)

    wall_s = time.perf_counter() - t0
    summary_rows, warmup_info = apply_warmup_filter(rows, args)
    summary = summarize(summary_rows, stage_stats, wall_s)
    summary.update(warmup_info)
    summary["variant"] = canonical
    summary["registry_mode"] = registry_mode
    summary["model"] = args.model
    summary["num_cameras"] = args.num_cameras
    summary["infer_workers"] = args.infer_workers
    summary["post_workers"] = args.post_workers
    summary["mp_start_method"] = getattr(args, "mp_start_method", "spawn")
    summary["queue_policy"] = args.queue_policy
    summary["buffer_mode"] = "latest"
    summary["grid_video"] = args.grid_video
    summary["grid_rows"] = args.grid_rows if args.grid_video else 0
    summary["grid_cols"] = args.grid_cols if args.grid_video else 0
    summary["backpressure_mode"] = backpressure_mode
    summary["backpressure_enabled"] = backpressure_enabled
    summary["max_pending_age_ms"] = args.max_pending_age_ms if backpressure_mode == "soft" else None
    summary["target_output_fps_per_camera"] = getattr(args, "target_output_fps_per_camera", 0.0)
    summary["shared_map_slots"] = getattr(args, "shared_map_slots", 0)
    summary["shared_input_slots"] = getattr(args, "shared_input_slots", 0)
    summary["shared_input_dtype"] = getattr(args, "shared_input_dtype", "float32")


    summary["migraphx_batch_size"] = getattr(args, "migraphx_batch_size", 1)
    summary["migraphx_batch_timeout_ms"] = getattr(args, "migraphx_batch_timeout_ms", 0.0)
    summary["merged_pose_fused_pruned"] = (registry_mode == "mx_merged_pose_fused_pruned")
    summary["prealloc_resize_buffers"] = bool(getattr(args, "prealloc_resize_buffers", False))
    summary["gpu_nms_batch_size"] = getattr(args, "gpu_nms_batch_size", 1)
    summary["gpu_nms_batch_timeout_ms"] = getattr(args, "gpu_nms_batch_timeout_ms", 0.0)
    summary["roctx"] = bool(getattr(args, "roctx", False))
    summary["trace_log_every"] = int(getattr(args, "trace_log_every", 0) or 0)
    summary["allow_ptrace_attach"] = bool(getattr(args, "allow_ptrace_attach", False))

    summary["prealloc_resize_buffers"] = bool(getattr(args, "prealloc_resize_buffers", False))
    summary["gpu_nms_batch_size"] = getattr(args, "gpu_nms_batch_size", 1)
    summary["gpu_nms_batch_timeout_ms"] = getattr(args, "gpu_nms_batch_timeout_ms", 0.0)


    summary["prealloc_resize_buffers"] = bool(getattr(args, "prealloc_resize_buffers", False))
    summary["gpu_nms_batch_size"] = getattr(args, "gpu_nms_batch_size", 1)
    summary["gpu_nms_batch_timeout_ms"] = getattr(args, "gpu_nms_batch_timeout_ms", 0.0)

    summary["realtime"] = args.realtime
    summary["camera_sources"] = sources
    if system_profile:
        summary["system_profile"] = system_profile

    print_summary(summary)
    print_system_profile(system_profile)
    write_detailed_csv(args.detailed_csv, summary_rows)
    write_summary_json(args.summary_json, summary)
    return summary


def run(args) -> Dict[str, Any]:
    if getattr(args, "buffer_mode", "latest") == "latest":
        return run_latest(args)
    return run_queue(args)
