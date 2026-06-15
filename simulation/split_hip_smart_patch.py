"""Runtime integration for the split MXR1 -> HIP smart TopK -> MXR2 stream variant.

This module is imported by the root ``simulate_camera_stream.py`` wrapper before
``simulation.cli.main`` is executed.  It keeps the existing modular simulator
unchanged for all current variants and adds a new stream variant:

    split_hip_host_smart

The pipeline is host-mediated by design in this stage:

    MXR1 inference worker -> postprocess worker:
        heatmaps -> native HIP smart TopK -> MXR2 PAF pruning -> CPU assembly

The smart HIP defaults match the validated speed candidate:
    smart_proposals=32, smart_local_radius=4, smart_lowres_nms_radius=1

The reducer thread count is build-time controlled by tools/build_heatmap_topk_hip.sh
via HIP_TOPK_SMART_THREADS, defaulting to 64.
"""

from __future__ import annotations

import os
import queue as py_queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SPLIT_ALIASES = {
    "split_hip_host_smart",
    "split-hip-host-smart",
    "split_hip_smart",
    "split-hip-smart",
    "mxr1_hip_smart_mxr2",
    "mxr1-hip-smart-mxr2",
    "mxr1_hip_host_smart_mxr2",
    "mxr1-hip-host-smart-mxr2",
}

_MXR2_OUTPUT_NAMES = [
    "limb_top_pair_a_idx",
    "limb_top_pair_b_idx",
    "limb_top_pair_score",
    "limb_top_pair_valid",
]

_ORIGINALS: Dict[str, Any] = {}


def _is_split_variant(user_variant: str) -> bool:
    key = str(user_variant or "").strip().lower().replace(" ", "-")
    return key in _SPLIT_ALIASES or key.replace("-", "_") in _SPLIT_ALIASES or key.replace("_", "-") in _SPLIT_ALIASES


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


def _split_env_path() -> str:
    path = os.environ.get("STREAM_SPLIT_HIP_MXR2", "").strip()
    if not path:
        path = "models/split_paf_pruning_from_topk/split_paf_pruning_from_topk_b4_68x121_to_1080x1920_k20_m20_p8_min0p05_sr0p8_pam0p75_mp0p0.mxr"
    return path


def _as_numpy(x: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x))


def _migraphx_argument(arr: np.ndarray):
    import migraphx  # type: ignore

    return migraphx.argument(np.ascontiguousarray(arr))


def _run_mxr2(program: Any, pafs_bchw: np.ndarray, top_scores: np.ndarray, top_indices: np.ndarray) -> Dict[str, np.ndarray]:
    outputs = program.run(
        {
            "pafs": _migraphx_argument(np.ascontiguousarray(pafs_bchw.astype(np.float32, copy=False))),
            "top_scores": _migraphx_argument(np.ascontiguousarray(top_scores.astype(np.float32, copy=False))),
            "top_indices": _migraphx_argument(np.ascontiguousarray(top_indices.astype(np.int64, copy=False))),
        }
    )
    outputs = [_as_numpy(o) for o in outputs]
    if len(outputs) != len(_MXR2_OUTPUT_NAMES):
        raise RuntimeError(f"MXR2 returned {len(outputs)} outputs, expected {len(_MXR2_OUTPUT_NAMES)}")
    return dict(zip(_MXR2_OUTPUT_NAMES, outputs))


def _heatmap_to_chw(heatmaps: np.ndarray) -> np.ndarray:
    arr = np.asarray(heatmaps)
    if arr.ndim != 3:
        raise ValueError(f"Expected single-frame heatmaps as HWC/CHW rank-3 tensor, got {arr.shape}")
    if arr.shape[-1] in (18, 19):
        arr = np.moveaxis(arr[..., :18], -1, 0)
    elif arr.shape[0] in (18, 19):
        arr = arr[:18]
    else:
        raise ValueError(f"Cannot identify heatmap channel axis in shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _paf_to_chw(pafs: np.ndarray) -> np.ndarray:
    arr = np.asarray(pafs)
    if arr.ndim != 3:
        raise ValueError(f"Expected single-frame PAFs as HWC/CHW rank-3 tensor, got {arr.shape}")
    if arr.shape[-1] == 38:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] == 38:
        pass
    else:
        raise ValueError(f"Cannot identify PAF channel axis in shape={arr.shape}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _pad_batch(arr: np.ndarray, compiled_batch_size: int) -> Tuple[np.ndarray, int]:
    real_n = int(arr.shape[0])
    compiled = max(1, int(compiled_batch_size))
    if compiled < real_n:
        compiled = real_n
    if compiled > real_n:
        pad = np.repeat(arr[-1:, ...], compiled - real_n, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return np.ascontiguousarray(arr), real_n


def _slice_batched(arr: np.ndarray, i: int) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim >= 3 and x.shape[0] > i:
        return np.ascontiguousarray(x[i : i + 1])
    return np.ascontiguousarray(x)


def _load_split_runtime():
    import migraphx  # type: ignore

    from modules.external_heatmap_topk import HeatmapTopKConfig, run_external_heatmap_topk
    from modules.mx_pair_assembly_pruned import assemble_poses_from_pruned_pairs
    from modules.postprocessing import PostprocessOutput

    mxr2_path = _split_env_path()
    if not Path(mxr2_path).exists():
        raise FileNotFoundError(f"Missing --split-mxr2 MXR2 model: {mxr2_path}")
    mxr2 = migraphx.load(mxr2_path)
    return mxr2, HeatmapTopKConfig, run_external_heatmap_topk, assemble_poses_from_pruned_pairs, PostprocessOutput


def _run_split_hip_smart_batch(
    *,
    batch_items: Sequence[Dict[str, Any]],
    map_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    mxr2: Any,
    HeatmapTopKConfig: Any,
    run_external_heatmap_topk: Any,
    assemble_poses_from_pruned_pairs: Any,
    PostprocessOutput: Any,
    threshold: float,
) -> List[Any]:
    if not batch_items:
        return []

    compiled_batch_size = _split_env_int("STREAM_SPLIT_HIP_BATCH_SIZE", 4)
    smart_proposals = _split_env_int("STREAM_SPLIT_HIP_SMART_PROPOSALS", 32)
    smart_local_radius = _split_env_int("STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS", 4)
    smart_lowres_nms_radius = _split_env_int("STREAM_SPLIT_HIP_SMART_LOWRES_NMS_RADIUS", 1)

    heat_bchw = np.stack([_heatmap_to_chw(hm) for hm, _pf in map_pairs], axis=0)
    paf_bchw = np.stack([_paf_to_chw(pf) for _hm, pf in map_pairs], axis=0)
    heat_bchw, real_n = _pad_batch(heat_bchw, compiled_batch_size)
    paf_bchw, _ = _pad_batch(paf_bchw, compiled_batch_size)

    cfg = HeatmapTopKConfig(
        batch_size=int(heat_bchw.shape[0]),
        in_h=int(heat_bchw.shape[2]),
        in_w=int(heat_bchw.shape[3]),
        full_h=int(batch_items[0]["original_hw"][0]),
        full_w=int(batch_items[0]["original_hw"][1]),
        channels=18,
        topk=_split_env_int("STREAM_SPLIT_HIP_TOPK", 20),
        threshold=float(threshold),
        smart_proposals=smart_proposals,
        smart_local_radius=smart_local_radius,
        smart_lowres_nms_radius=smart_lowres_nms_radius,
    )

    t0 = time.perf_counter()
    top_scores, top_indices = run_external_heatmap_topk(heat_bchw, cfg, backend="hip_host_smart")
    t1 = time.perf_counter()
    mxr2_out = _run_mxr2(mxr2, paf_bchw, top_scores, top_indices)
    t2 = time.perf_counter()

    heatmap_ms_total = (t1 - t0) * 1000.0
    mxr2_ms_total = (t2 - t1) * 1000.0
    outputs = []
    for i, item in enumerate(batch_items[:real_n]):
        t_asm0 = time.perf_counter()
        poses, keypoints, asm_times = assemble_poses_from_pruned_pairs(
            _slice_batched(top_scores, i),
            _slice_batched(top_indices, i),
            _slice_batched(mxr2_out["limb_top_pair_a_idx"], i),
            _slice_batched(mxr2_out["limb_top_pair_b_idx"], i),
            _slice_batched(mxr2_out["limb_top_pair_score"], i),
            _slice_batched(mxr2_out["limb_top_pair_valid"], i),
            full_width=int(item["original_hw"][1]),
            threshold=float(threshold),
            min_pair_score=0.0,
            return_timing=True,
        )
        asm_ms = (time.perf_counter() - t_asm0) * 1000.0
        valid_topk = float(np.sum(_slice_batched(top_scores, i) > -1.0e8))
        limb_valid = float(np.sum(_slice_batched(mxr2_out["limb_top_pair_valid"], i) > 0.5))
        per_frame_gpu_ms = (heatmap_ms_total + mxr2_ms_total) / float(max(1, real_n))
        timings: Dict[str, float] = {
            "split_smart_heatmap": heatmap_ms_total / float(max(1, real_n)),
            "split_smart_heatmap_batch": heatmap_ms_total,
            "split_mxr2": mxr2_ms_total / float(max(1, real_n)),
            "split_mxr2_batch": mxr2_ms_total,
            "split_cpu_assembly": asm_ms,
            "split_total_batch": heatmap_ms_total + mxr2_ms_total,
            "split_real_batch_size": float(real_n),
            "split_compiled_batch_size": float(heat_bchw.shape[0]),
            "valid_topk_count": valid_topk,
            "limb_valid_count": limb_valid,
            "total_postprocess": per_frame_gpu_ms + asm_ms,
        }
        for k, v in dict(asm_times).items():
            try:
                timings[str(k)] = float(v)
            except Exception:
                pass
        outputs.append(PostprocessOutput(np.asarray(poses, dtype=np.float32), np.asarray(keypoints, dtype=np.float32), timings))
    return outputs


def _make_row(*, bi: Dict[str, Any], out: Any, canonical: str, registry_mode: str, worker_id: int, post_start: float, post_done: float) -> Dict[str, Any]:
    from simulation.utils import safe_float

    queue_wait_ms = (post_start - float(bi.get("infer_done_ts", post_start))) * 1000.0
    timings = dict(out.timings)
    post_ms = float(timings.get("total_postprocess", (post_done - post_start) * 1000.0))
    e2e_ms = (post_done - float(bi["capture_ts"])) * 1000.0
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
        "collector_stale_records_discarded_pre_batch": int(bi.get("collector_stale_records_discarded_pre_batch", 0)),
        "collector_stale_records_discarded_pre_batch_cumulative": int(bi.get("collector_stale_records_discarded_pre_batch_cumulative", 0)),
    }
    for key, value in timings.items():
        row[f"timing_{key}"] = safe_float(value)
    return row


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
    if not _is_split_variant(user_variant):
        return _ORIGINALS["postprocess_latest_worker"](
            worker_id=worker_id, user_variant=user_variant, in_queues=in_queues, infer_done=infer_done,
            post_pending=post_pending, last_processed_ts=last_processed_ts, result_q=result_q,
            stats_q=stats_q, error_q=error_q, torch_device=torch_device, require_gpu=require_gpu,
            max_keypoints=max_keypoints, threshold=threshold, nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres, nms_impl=nms_impl, gpu_compute_dtype=gpu_compute_dtype,
            grid_q=grid_q, render_output=render_output, poll_sleep_s=poll_sleep_s,
            migraphx_nms_mxr=migraphx_nms_mxr, migraphx_nms_cache_dir=migraphx_nms_cache_dir,
            shared_map_descs=shared_map_descs, free_map_slots=free_map_slots,
            prealloc_resize_buffers=prealloc_resize_buffers, gpu_nms_batch_size=gpu_nms_batch_size,
            gpu_nms_batch_timeout_ms=gpu_nms_batch_timeout_ms, trace_log_every=trace_log_every,
            roctx_enabled=roctx_enabled,
        )

    try:
        from simulation.grid_video import draw_poses_on_frame
        from simulation.queues import all_done, all_queues_empty
        from simulation.shared_memory import close_shared_map_views, open_shared_map_buffers, release_shared_slot_from_item
        from simulation.system import configure_child_cpu_runtime
        from simulation.tracing import RocTxTracer, trace_print
        from simulation.utils import mean, percentile

        tracer = RocTxTracer(roctx_enabled, f"post:{worker_id}:pid:{os.getpid()}:split_hip_smart")
        tracer.mark("worker_start")
        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        mxr2, HeatmapTopKConfig, run_external_heatmap_topk, assemble_poses_from_pruned_pairs, PostprocessOutput = _load_split_runtime()

        canonical = registry_mode = "split_hip_host_smart"
        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
        batch_size = max(1, _split_env_int("STREAM_SPLIT_HIP_BATCH_SIZE", 4))
        batch_timeout_s = max(0.0, _split_env_float("STREAM_SPLIT_HIP_TIMEOUT_MS", 4.0)) / 1000.0
        ncam = len(in_queues)
        next_cam = worker_id % max(1, ncam)
        processed = 0
        post_times: List[float] = []
        queue_wait_times: List[float] = []
        e2e_times: List[float] = []
        t_worker_start = time.perf_counter()
        print(
            f"[POST:{worker_id}] split_hip_host_smart mxr2={_split_env_path()} batch={batch_size} "
            f"timeout={batch_timeout_s*1000:.2f}ms sp={_split_env_int('STREAM_SPLIT_HIP_SMART_PROPOSALS', 32)} "
            f"lr={_split_env_int('STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS', 4)}",
            flush=True,
        )

        def _get_next():
            nonlocal next_cam
            scanned = 0
            while scanned < ncam:
                cam_id = next_cam
                next_cam = (next_cam + 1) % ncam
                scanned += 1
                try:
                    return in_queues[cam_id].get_nowait()
                except py_queue.Empty:
                    continue
            return None

        def _maps_for(batch_item):
            slot_id = batch_item.get("shared_map_slot") if isinstance(batch_item, dict) else None
            if slot_id is not None and int(slot_id) in shared_slots:
                slot = shared_slots[int(slot_id)]
                return slot["heat"], slot["paf"]
            return batch_item["heatmaps"], batch_item["pafs"]

        while True:
            item = _get_next()
            if item is None:
                if all_done(infer_done) and all_queues_empty(in_queues):
                    break
                time.sleep(poll_sleep_s)
                continue

            batch_items = [item]
            deadline = time.perf_counter() + batch_timeout_s
            while len(batch_items) < batch_size:
                extra = _get_next()
                if extra is not None:
                    batch_items.append(extra)
                    continue
                if batch_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                    break
                time.sleep(min(poll_sleep_s, max(0.0, deadline - time.perf_counter())))

            post_start = time.perf_counter()
            try:
                map_pairs = [_maps_for(bi) for bi in batch_items]
                with tracer.range(f"postprocess_split_hip_host_smart_batch{len(batch_items)}"):
                    batch_outputs = _run_split_hip_smart_batch(
                        batch_items=batch_items,
                        map_pairs=map_pairs,
                        mxr2=mxr2,
                        HeatmapTopKConfig=HeatmapTopKConfig,
                        run_external_heatmap_topk=run_external_heatmap_topk,
                        assemble_poses_from_pruned_pairs=assemble_poses_from_pruned_pairs,
                        PostprocessOutput=PostprocessOutput,
                        threshold=threshold,
                    )
            except Exception:
                for bi in batch_items:
                    release_shared_slot_from_item(bi, free_map_slots)
                raise

            for bi, out in zip(batch_items, batch_outputs):
                post_done = time.perf_counter()
                row = _make_row(bi=bi, out=out, canonical=canonical, registry_mode=registry_mode, worker_id=worker_id, post_start=post_start, post_done=post_done)
                post_times.append(row["post_ms"])
                queue_wait_times.append(row["queue_infer_to_post_ms"])
                e2e_times.append(row["e2e_ms"])
                if render_output and grid_q is not None and "frame_bgr" in bi:
                    frame_out = bi["frame_bgr"].copy()
                    draw_poses_on_frame(frame_out, out.pose_entries, out.all_keypoints)
                    try:
                        grid_q.put_nowait({"camera_id": row["camera_id"], "frame_id": row["frame_id"], "source": row["source"], "frame_bgr": frame_out, "e2e_ms": row["e2e_ms"], "post_ms": row["post_ms"], "num_poses": row["num_poses"], "num_keypoints": row["num_keypoints"]})
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
                trace_print(trace_log_every, processed, f"[TRACE post:{worker_id} pid={os.getpid()} latest split] processed={processed} cam={row['camera_id']} frame={row['frame_id']} batch={len(batch_items)} post={row['post_ms']:.2f}ms e2e={row['e2e_ms']:.2f}ms")

        close_shared_map_views(shared_handles)
        stats_q.put({"stage": "postprocess", "buffer_mode": "latest", "worker_id": worker_id, "variant": canonical, "registry_mode": registry_mode, "processed": processed, "avg_queue_infer_to_post_ms": mean(queue_wait_times), "p95_queue_infer_to_post_ms": percentile(queue_wait_times, 95), "avg_post_ms": mean(post_times), "p95_post_ms": percentile(post_times, 95), "avg_e2e_ms": mean(e2e_times), "p95_e2e_ms": percentile(e2e_times, 95), "split_mxr2": _split_env_path(), "split_mxr2_batch_size": batch_size, "wall_s": time.perf_counter() - t_worker_start})
        print(f"[POST:{worker_id}] Done. processed={processed}", flush=True)
    except Exception:
        error_q.put({"stage": "postprocess", "worker_id": worker_id, "traceback": traceback.format_exc()})


def postprocess_worker_patched(**kwargs) -> None:
    # Queue mode support uses the same latest-compatible implementation by adapting
    # the single shared input queue into per-camera semantics would be more invasive.
    # For non-latest stream runs, fall back to the existing worker unless the split
    # variant is requested, in which case fail loudly with a clear message.
    user_variant = kwargs.get("user_variant", "")
    if not _is_split_variant(user_variant):
        return _ORIGINALS["postprocess_worker"](**kwargs)
    error_q = kwargs.get("error_q")
    worker_id = kwargs.get("worker_id", 0)
    try:
        raise RuntimeError("split_hip_host_smart is currently implemented for --buffer-mode latest. Use --buffer-mode latest.")
    except Exception:
        if error_q is not None:
            error_q.put({"stage": "postprocess", "worker_id": worker_id, "traceback": traceback.format_exc()})


def _patch_resolve_registry_mode():
    import simulation.postprocess_modes as modes
    import simulation.runner as runner
    import simulation.workers.postprocess as worker_mod

    original = modes.resolve_registry_mode
    _ORIGINALS.setdefault("resolve_registry_mode", original)

    def resolve_registry_mode_patched(user_mode: str):
        if _is_split_variant(user_mode):
            return "split_hip_host_smart", "split_hip_host_smart", False
        return original(user_mode)

    modes.resolve_registry_mode = resolve_registry_mode_patched
    runner.resolve_registry_mode = resolve_registry_mode_patched
    worker_mod.resolve_registry_mode = resolve_registry_mode_patched


def _patch_cli_and_run():
    import simulation.cli as cli
    import simulation.runner as runner

    original_build_parser = cli.build_parser
    original_run = runner.run
    _ORIGINALS.setdefault("build_parser", original_build_parser)
    _ORIGINALS.setdefault("runner_run", original_run)

    def build_parser_patched():
        parser = original_build_parser()
        existing = {a.dest for a in parser._actions}
        if "split_mxr2" not in existing:
            parser.add_argument("--split-mxr2", default=_split_env_path(), help="MXR2 model for --variant split_hip_host_smart.")
            parser.add_argument("--split-mxr2-batch-size", type=int, default=4, help="Compiled/static batch size of --split-mxr2. Default: 4.")
            parser.add_argument("--split-batch-timeout-ms", type=float, default=4.0, help="Maximum wait to fill a split MXR2 postprocess batch. Default: 4 ms.")
            parser.add_argument("--smart-proposals", type=int, default=32, help="hip_host_smart proposals per keypoint type. Default: 32.")
            parser.add_argument("--smart-local-radius", type=int, default=4, help="hip_host_smart full-res local refinement radius. Default: 4.")
            parser.add_argument("--smart-lowres-nms-radius", type=int, default=1, help="hip_host_smart low-res NMS radius. Default: 1.")
        return parser

    def run_patched(args):
        os.environ["STREAM_SPLIT_HIP_MXR2"] = str(getattr(args, "split_mxr2", _split_env_path()))
        os.environ["STREAM_SPLIT_HIP_BATCH_SIZE"] = str(int(getattr(args, "split_mxr2_batch_size", 4)))
        os.environ["STREAM_SPLIT_HIP_TIMEOUT_MS"] = str(float(getattr(args, "split_batch_timeout_ms", 4.0)))
        os.environ["STREAM_SPLIT_HIP_SMART_PROPOSALS"] = str(int(getattr(args, "smart_proposals", 32)))
        os.environ["STREAM_SPLIT_HIP_SMART_LOCAL_RADIUS"] = str(int(getattr(args, "smart_local_radius", 4)))
        os.environ["STREAM_SPLIT_HIP_SMART_LOWRES_NMS_RADIUS"] = str(int(getattr(args, "smart_lowres_nms_radius", 1)))
        os.environ["STREAM_SPLIT_HIP_TOPK"] = str(int(getattr(args, "max_keypoints", 20)))
        return original_run(args)

    cli.build_parser = build_parser_patched
    runner.run = run_patched
    cli.run = run_patched


def _patch_workers():
    import simulation.runner as runner
    import simulation.workers.postprocess as worker_mod

    _ORIGINALS.setdefault("postprocess_latest_worker", worker_mod.postprocess_latest_worker)
    _ORIGINALS.setdefault("postprocess_worker", worker_mod.postprocess_worker)
    worker_mod.postprocess_latest_worker = postprocess_latest_worker_patched
    worker_mod.postprocess_worker = postprocess_worker_patched
    runner.postprocess_latest_worker = postprocess_latest_worker_patched
    runner.postprocess_worker = postprocess_worker_patched


def apply_patch() -> None:
    if _ORIGINALS.get("applied"):
        return
    import simulation.cli  # noqa: F401 - force import before monkey patching shared module globals
    import simulation.runner  # noqa: F401
    import simulation.workers.postprocess  # noqa: F401

    _patch_resolve_registry_mode()
    _patch_cli_and_run()
    _patch_workers()
    _ORIGINALS["applied"] = True
