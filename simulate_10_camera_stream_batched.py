#!/usr/bin/env python3
"""
simulate_camera_stream.py

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
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
from multiprocessing import shared_memory

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



def _dtype_from_name(name: str):
    return np.float16 if str(name) == "float16" else np.float32


def create_shared_map_buffers(num_slots: int, out_h: int, out_w: int, dtype_name: str) -> Tuple[List[Dict[str, Any]], List[shared_memory.SharedMemory]]:
    dtype = _dtype_from_name(dtype_name)
    heat_shape = (out_h, out_w, 19)
    paf_shape = (out_h, out_w, 38)
    heat_nbytes = int(np.prod(heat_shape) * np.dtype(dtype).itemsize)
    paf_nbytes = int(np.prod(paf_shape) * np.dtype(dtype).itemsize)
    descs: List[Dict[str, Any]] = []
    handles: List[shared_memory.SharedMemory] = []
    for slot_id in range(max(0, int(num_slots))):
        heat_shm = shared_memory.SharedMemory(create=True, size=heat_nbytes)
        paf_shm = shared_memory.SharedMemory(create=True, size=paf_nbytes)
        handles.extend([heat_shm, paf_shm])
        descs.append({
            "slot_id": slot_id,
            "dtype": np.dtype(dtype).name,
            "heat_shape": heat_shape,
            "paf_shape": paf_shape,
            "heat_name": heat_shm.name,
            "paf_name": paf_shm.name,
        })
    return descs, handles


def close_shared_map_buffers(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def open_shared_map_buffers(descs: Optional[Sequence[Dict[str, Any]]]):
    if not descs:
        return {}, []
    slots: Dict[int, Dict[str, Any]] = {}
    handles = []
    for desc in descs:
        heat_shm = shared_memory.SharedMemory(name=desc["heat_name"])
        paf_shm = shared_memory.SharedMemory(name=desc["paf_name"])
        handles.extend([heat_shm, paf_shm])
        dtype = np.dtype(desc["dtype"])
        slots[int(desc["slot_id"])] = {
            "heat": np.ndarray(tuple(desc["heat_shape"]), dtype=dtype, buffer=heat_shm.buf),
            "paf": np.ndarray(tuple(desc["paf_shape"]), dtype=dtype, buffer=paf_shm.buf),
        }
    return slots, handles


def close_shared_map_views(handles: Sequence[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        try:
            shm.close()
        except Exception:
            pass


def latest_put_with_dropped(q, item):
    try:
        q.put_nowait(item)
        return None
    except py_queue.Full:
        pass
    dropped = None
    try:
        dropped = q.get_nowait()
    except py_queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except py_queue.Full:
        try:
            newer = q.get_nowait()
            dropped = newer if dropped is None else dropped
        except py_queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except py_queue.Full:
            return item
    return dropped


def release_shared_slot_from_item(item: Any, free_q) -> None:
    if not isinstance(item, dict) or free_q is None:
        return
    slot_id = item.get("shared_map_slot")
    if slot_id is None:
        return
    try:
        free_q.put_nowait(int(slot_id))
    except Exception:
        pass


def _fmt_mb(kb: float) -> str:
    return f"{kb / 1024.0:.1f}M"


def _bar(pct: float, width: int = 30) -> str:
    filled = max(0, min(width, int(round(width * pct / 100.0))))
    return "█" * filled + "░" * (width - filled)


def _read_proc_stat() -> List[List[int]]:
    rows: List[List[int]] = []
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                if parts[0] == "cpu" or not parts[0][3:].isdigit():
                    continue
                rows.append([int(x) for x in parts[1:8]])
    except Exception:
        pass
    return rows


def _cpu_pct(prev: Sequence[int], cur: Sequence[int]) -> float:
    prev_idle = prev[3] + prev[4]
    cur_idle = cur[3] + cur[4]
    prev_total = sum(prev)
    cur_total = sum(cur)
    total_delta = cur_total - prev_total
    idle_delta = cur_idle - prev_idle
    if total_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))


def _clock_ticks() -> int:
    try:
        return int(os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                   if isinstance(os.sysconf_names, dict) else os.sysconf("SC_CLK_TCK"))
    except Exception:
        return 100


def _read_pid_cpu_ticks(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            data = f.read()
        tail = data[data.rfind(")") + 2:].split()
        return int(tail[11]) + int(tail[12])
    except Exception:
        return None


def _read_pid_affinity(pid: int) -> str:
    try:
        cpus = sorted(os.sched_getaffinity(pid))
        if len(cpus) > 16:
            return f"[{cpus[0]}..{cpus[-1]}] ({len(cpus)})"
        return "[" + ",".join(str(c) for c in cpus) + "]"
    except Exception:
        return "?"


def _read_pid_memory_kb(pid: int) -> Dict[str, int]:
    out = {
        "rss_kb": 0,
        "vms_kb": 0,
        "hwm_kb": 0,
        "rss_anon_kb": 0,
        "rss_file_kb": 0,
        "rss_shmem_kb": 0,
    }
    mapping = {
        "VmRSS:": "rss_kb",
        "VmSize:": "vms_kb",
        "VmHWM:": "hwm_kb",
        "RssAnon:": "rss_anon_kb",
        "RssFile:": "rss_file_kb",
        "RssShmem:": "rss_shmem_kb",
    }
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in mapping:
                    out[mapping[parts[0]]] = int(parts[1])
    except Exception:
        pass
    return out


def _read_gpu_busy_pct() -> Optional[float]:
    candidates = [
        "/sys/class/drm/card1/device/gpu_busy_percent",
        "/sys/class/drm/card0/device/gpu_busy_percent",
    ]
    for p in candidates:
        try:
            with open(p, "r") as f:
                return float(f.read().strip())
        except Exception:
            continue
    return None


def _read_gpu_vram_mb() -> Optional[float]:
    candidates = [
        "/sys/class/drm/card1/device/mem_info_vram_used",
        "/sys/class/drm/card0/device/mem_info_vram_used",
    ]
    for p in candidates:
        try:
            with open(p, "r") as f:
                return float(f.read().strip()) / (1024.0 * 1024.0)
        except Exception:
            continue
    return None


class SysMonitor:
    """Lightweight parent-side process/CPU/GPU monitor for stream runs."""

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = max(0.05, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pid_groups: Dict[int, str] = {}
        self._pid_affinity: Dict[int, str] = {}
        self._prev_core = _read_proc_stat()
        self._prev_pid_ticks: Dict[int, int] = {}
        self._prev_time = time.perf_counter()
        self.core_samples: List[List[float]] = []
        self.gpu_samples: List[float] = []
        self.vram_samples: List[float] = []
        self.pid_cpu_samples: Dict[int, List[float]] = defaultdict(list)
        self.pid_mem_samples: Dict[int, List[Dict[str, int]]] = defaultdict(list)

    def register_pids(self, group: str, pids: Sequence[int]) -> None:
        with self._lock:
            for pid in pids:
                if pid:
                    pid = int(pid)
                    self._pid_groups[pid] = group
                    self._pid_affinity[pid] = _read_pid_affinity(pid)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stream-sys-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summary()

    def _run(self) -> None:
        hz = _clock_ticks()
        while not self._stop.wait(self.interval_s):
            now = time.perf_counter()
            dt = max(1e-6, now - self._prev_time)

            cur_core = _read_proc_stat()
            if self._prev_core and cur_core and len(self._prev_core) == len(cur_core):
                self.core_samples.append([_cpu_pct(a, b) for a, b in zip(self._prev_core, cur_core)])
            self._prev_core = cur_core

            gpu = _read_gpu_busy_pct()
            if gpu is not None:
                self.gpu_samples.append(gpu)
            vram = _read_gpu_vram_mb()
            if vram is not None:
                self.vram_samples.append(vram)

            with self._lock:
                pids = dict(self._pid_groups)

            for pid in pids:
                ticks = _read_pid_cpu_ticks(pid)
                if ticks is None:
                    continue
                prev = self._prev_pid_ticks.get(pid)
                if prev is not None:
                    self.pid_cpu_samples[pid].append(max(0.0, 100.0 * (ticks - prev) / hz / dt))
                self._prev_pid_ticks[pid] = ticks
                mem = _read_pid_memory_kb(pid)
                if mem.get("rss_kb", 0) or mem.get("vms_kb", 0):
                    self.pid_mem_samples[pid].append(mem)

            self._prev_time = now

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            groups = dict(self._pid_groups)
            affinities = dict(self._pid_affinity)

        core_avg: List[float] = []
        if self.core_samples:
            arr = np.asarray(self.core_samples, dtype=np.float64)
            core_avg = [float(x) for x in np.mean(arr, axis=0)]

        pids: Dict[int, Dict[str, Any]] = {}
        for pid, group in groups.items():
            cpu = self.pid_cpu_samples.get(pid, [])
            mem = self.pid_mem_samples.get(pid, [])
            row: Dict[str, Any] = {
                "group": group,
                "cpu_avg_pct": mean(cpu),
                "affinity": affinities.get(pid) or _read_pid_affinity(pid),
                "samples": len(cpu),
            }
            if mem:
                for key in ["rss_kb", "vms_kb", "hwm_kb", "rss_anon_kb", "rss_file_kb", "rss_shmem_kb"]:
                    vals = [m.get(key, 0) for m in mem]
                    row[f"{key}_avg"] = float(np.mean(vals))
                    row[f"{key}_peak"] = int(max(vals))
                row["mem_samples"] = len(mem)
            else:
                row["mem_samples"] = 0
            pids[pid] = row

        return {
            "monitor_interval_s": self.interval_s,
            "cpu_core_avg_pct": core_avg,
            "gpu_avg_pct": mean(self.gpu_samples),
            "gpu_peak_pct": max(self.gpu_samples) if self.gpu_samples else 0.0,
            "gpu_idle_pct": (100.0 * sum(1 for x in self.gpu_samples if x < 5.0) / len(self.gpu_samples)) if self.gpu_samples else 0.0,
            "vram_avg_mb": mean(self.vram_samples),
            "vram_peak_mb": max(self.vram_samples) if self.vram_samples else 0.0,
            "gpu_samples": len(self.gpu_samples),
            "pids": pids,
        }


def _process_pid_groups(
    camera_procs: Sequence[mp.Process],
    infer_procs: Sequence[mp.Process],
    post_procs: Sequence[mp.Process],
    grid_procs: Sequence[mp.Process],
) -> Dict[str, List[int]]:
    return {
        "camera": [p.pid for p in camera_procs if p.pid],
        "inference": [p.pid for p in infer_procs if p.pid],
        "postprocess": [p.pid for p in post_procs if p.pid],
        "grid": [p.pid for p in grid_procs if p.pid],
    }


def _register_processes(monitor: Optional[SysMonitor], pid_groups: Dict[str, List[int]]) -> None:
    if monitor is None:
        return
    for group, pids in pid_groups.items():
        monitor.register_pids(group, pids)


def _pin_pid(pid: int, cpus: Sequence[int]) -> None:
    if not pid or not cpus:
        return
    os.sched_setaffinity(int(pid), set(int(c) for c in cpus))


def _pin_process_threads(pid: int, cpus: Sequence[int]) -> int:
    if not pid or not cpus:
        return 0
    task_dir = Path(f"/proc/{int(pid)}/task")
    if not task_dir.exists():
        return 0
    pinned = 0
    cpu_set = set(int(c) for c in cpus)
    for task in task_dir.iterdir():
        if not task.name.isdigit():
            continue
        try:
            os.sched_setaffinity(int(task.name), cpu_set)
            pinned += 1
        except Exception:
            pass
    return pinned


def configure_worker_thread_env(num_threads: int) -> None:
    n = str(max(1, int(num_threads)))
    os.environ["STREAM_WORKER_THREADS"] = n
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[env_name] = n
    os.environ.setdefault("OMP_PROC_BIND", "true")
    os.environ.setdefault("OMP_PLACES", "cores")


def configure_child_cpu_runtime(num_threads: int = 1) -> None:
    configure_worker_thread_env(num_threads)
    try:
        import cv2
        cv2.setNumThreads(max(0, int(num_threads)))
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max(1, int(num_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def pin_stream_processes(pid_groups: Dict[str, List[int]], args) -> None:
    """Pin each camera, inference, and postprocess worker to distinct CPUs."""
    if not getattr(args, "pin_cpus", False):
        return
    camera_base = int(args.pin_camera_base)
    inference_base = int(args.pin_inference_base)
    post_base = int(args.pin_post_base)

    assignments: Dict[int, List[int]] = {}
    for idx, pid in enumerate(pid_groups.get("camera", [])):
        assignments[int(pid)] = [camera_base + idx]
    for idx, pid in enumerate(pid_groups.get("inference", [])):
        assignments[int(pid)] = [inference_base + idx]
    for idx, pid in enumerate(pid_groups.get("postprocess", [])):
        assignments[int(pid)] = [post_base + idx]

    for pid, cpus in assignments.items():
        _pin_pid(pid, cpus)
        if getattr(args, "pin_all_threads", False):
            _pin_process_threads(pid, cpus)


def print_affinity_report(pid_groups: Dict[str, List[int]]) -> None:
    print("\n[CPU AFFINITY]")
    print(f"{'Group':<14} {'PID':>8} CPUs")
    print("-" * 52)
    for group, pids in pid_groups.items():
        if not pids:
            continue
        for pid in pids:
            print(f"{group:<14} {pid:>8} {_read_pid_affinity(pid)}")


def print_system_profile(stats: Dict[str, Any]) -> None:
    if not stats:
        return
    print("\n" + "=" * 150)
    print("SYSTEM / PROCESS PROFILE")
    print("=" * 150)

    cores = stats.get("cpu_core_avg_pct") or []
    if cores:
        print("\nCPU utilization po jezgru:")
        print(f"{'core':>6} {'avg%':>8}  bar")
        print("-" * 50)
        for idx, pct in enumerate(cores[:32]):
            print(f"cpu{idx:02d} {pct:>7.1f}%  {_bar(pct)}")
        if len(cores) > 32:
            print(f"... ({len(cores) - 32} dodatnih logical CPU jezgara skriveno)")

    pids = stats.get("pids") or {}
    if pids:
        print("\nProcesi:")
        print(
            f"{'PID':>8} {'Group':<14} {'CPU%':>7} {'RSS avg':>9} {'RSS peak':>9} "
            f"{'VMS avg':>9} {'HWM':>9}  Affinity"
        )
        print("-" * 120)
        for pid, row in sorted(pids.items(), key=lambda item: (item[1].get("group", ""), int(item[0]))):
            rss_avg = row.get("rss_kb_avg", 0.0)
            rss_peak = row.get("rss_kb_peak", 0.0)
            vms_avg = row.get("vms_kb_avg", 0.0)
            hwm = row.get("hwm_kb_peak", 0.0)
            print(
                f"{int(pid):>8} {row.get('group', ''):<14} {row.get('cpu_avg_pct', 0.0):>6.1f}% "
                f"{_fmt_mb(rss_avg):>9} {_fmt_mb(rss_peak):>9} {_fmt_mb(vms_avg):>9} {_fmt_mb(hwm):>9}  "
                f"{row.get('affinity', '?')}"
            )

        print("\nMemory breakdown po procesu (avg):")
        print(f"{'PID':>8} {'Group':<14} {'Anon':>9} {'File':>9} {'Shmem':>9} {'Samples':>8}")
        print("-" * 76)
        for pid, row in sorted(pids.items(), key=lambda item: (item[1].get("group", ""), int(item[0]))):
            print(
                f"{int(pid):>8} {row.get('group', ''):<14} "
                f"{_fmt_mb(row.get('rss_anon_kb_avg', 0.0)):>9} "
                f"{_fmt_mb(row.get('rss_file_kb_avg', 0.0)):>9} "
                f"{_fmt_mb(row.get('rss_shmem_kb_avg', 0.0)):>9} "
                f"{row.get('mem_samples', 0):>8}"
            )

    gpu_samples = int(stats.get("gpu_samples", 0) or 0)
    if gpu_samples:
        print("\nGPU:")
        print(f"Avg GPU%:   {stats.get('gpu_avg_pct', 0.0):6.1f}%  {_bar(stats.get('gpu_avg_pct', 0.0))}")
        print(f"Peak GPU%:  {stats.get('gpu_peak_pct', 0.0):6.1f}%")
        print(f"GPU idle:   {stats.get('gpu_idle_pct', 0.0):6.1f}%  (samples < 5%)")
        print(f"VRAM avg:   {stats.get('vram_avg_mb', 0.0):6.1f} MB")
        print(f"VRAM peak:  {stats.get('vram_peak_mb', 0.0):6.1f} MB")
        print(f"Samples:    {gpu_samples}")
    print("=" * 150)

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


def _as_migraphx_output_array(x: Any) -> np.ndarray:
    """Convert MIGraphX output argument/list item to numpy without assuming layout."""
    return np.asarray(x)


def decode_migraphx_batch_outputs(
    results: Any,
    out_h: int,
    out_w: int,
    output_dtype: str,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode MIGraphX model outputs for batched inference.

    Expected model output for input Bx3xHxW:
        heatmaps: B x 19 x out_h x out_w
        pafs:     B x 38 x out_h x out_w

    Returns:
        heatmaps: B x out_h x out_w x 19
        pafs:     B x out_h x out_w x 38
    """
    if not isinstance(results, (list, tuple)):
        results = list(results)
    if len(results) < 2:
        raise RuntimeError("MIGraphX model must return at least heatmaps and PAFs.")

    heat_raw = _as_migraphx_output_array(results[-2])
    paf_raw = _as_migraphx_output_array(results[-1])

    heat_per_sample = 19 * out_h * out_w
    paf_per_sample = 38 * out_h * out_w

    if batch_size is None:
        if heat_raw.size % heat_per_sample != 0:
            raise RuntimeError(
                f"Cannot infer heatmap batch size from output size={heat_raw.size}, "
                f"per_sample={heat_per_sample}, raw_shape={heat_raw.shape}"
            )
        batch_size = int(heat_raw.size // heat_per_sample)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"Invalid decoded batch size: {batch_size}")

    if heat_raw.ndim == 4:
        if heat_raw.shape[0] == batch_size and heat_raw.shape[1] == 19:
            heat_bchw = heat_raw
        elif heat_raw.shape[0] == batch_size and heat_raw.shape[-1] == 19:
            heat_bchw = np.transpose(heat_raw, (0, 3, 1, 2))
        else:
            heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)
    elif heat_raw.ndim == 3:
        if heat_raw.shape[0] == 19:
            heat_bchw = heat_raw.reshape(1, 19, out_h, out_w)
        elif heat_raw.shape[-1] == 19:
            heat_bchw = np.transpose(heat_raw, (2, 0, 1)).reshape(1, 19, out_h, out_w)
        else:
            heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)
    else:
        heat_bchw = heat_raw.reshape(batch_size, 19, out_h, out_w)

    if paf_raw.ndim == 4:
        if paf_raw.shape[0] == batch_size and paf_raw.shape[1] == 38:
            paf_bchw = paf_raw
        elif paf_raw.shape[0] == batch_size and paf_raw.shape[-1] == 38:
            paf_bchw = np.transpose(paf_raw, (0, 3, 1, 2))
        else:
            paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)
    elif paf_raw.ndim == 3:
        if paf_raw.shape[0] == 38:
            paf_bchw = paf_raw.reshape(1, 38, out_h, out_w)
        elif paf_raw.shape[-1] == 38:
            paf_bchw = np.transpose(paf_raw, (2, 0, 1)).reshape(1, 38, out_h, out_w)
        else:
            paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)
    else:
        paf_bchw = paf_raw.reshape(batch_size, 38, out_h, out_w)

    heat_bhwc = np.moveaxis(heat_bchw, 1, -1)
    paf_bhwc = np.moveaxis(paf_bchw, 1, -1)

    if output_dtype == "float16":
        return (
            np.ascontiguousarray(heat_bhwc, dtype=np.float16),
            np.ascontiguousarray(paf_bhwc, dtype=np.float16),
        )
    return (
        np.ascontiguousarray(heat_bhwc, dtype=np.float32),
        np.ascontiguousarray(paf_bhwc, dtype=np.float32),
    )


def decode_migraphx_outputs(results: Any, out_h: int, out_w: int, output_dtype: str) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible single-frame decode. Returns HxWx19 and HxWx38."""
    heat_bhwc, paf_bhwc = decode_migraphx_batch_outputs(
        results,
        out_h,
        out_w,
        output_dtype,
        batch_size=1,
    )
    return heat_bhwc[0], paf_bhwc[0]


def make_migraphx_input_batch(
    items: Sequence[Dict[str, Any]],
    expected_dtype: str,
    compiled_batch_size: int,
) -> Tuple[np.ndarray, int]:
    """
    Build Bx3xHxW input for MIGraphX.

    Each camera item contains input_tensor shaped 1x3xHxW.
    If compiled_batch_size > number of real items, pad by repeating the last
    frame. This is useful for static batch MXR files such as b4/b8 models.
    """
    if not items:
        raise RuntimeError("Cannot build MIGraphX batch from empty item list.")

    actual_batch_size = len(items)
    tensors = []
    for item in items:
        x = np.asarray(item["input_tensor"])
        if x.ndim != 4:
            raise ValueError(f"input_tensor must be 4D, got shape={x.shape}")
        if x.shape[0] != 1:
            raise ValueError(f"Each queued item must be one frame, got shape={x.shape}")
        tensors.append(x)

    batch = np.concatenate(tensors, axis=0)

    compiled_batch_size = max(1, int(compiled_batch_size))
    if compiled_batch_size > actual_batch_size:
        pad_count = compiled_batch_size - actual_batch_size
        pad = np.repeat(batch[-1:, ...], pad_count, axis=0)
        batch = np.concatenate([batch, pad], axis=0)

    return cast_for_migraphx(expected_dtype, batch), actual_batch_size


def build_inference_output_items_from_batch(
    *,
    batch_items: Sequence[Dict[str, Any]],
    heatmaps_bhwc: np.ndarray,
    pafs_bhwc: np.ndarray,
    infer_done_ts: float,
    inference_ms_total: float,
    decode_ms_total: float,
    queue_wait_times_ms: Sequence[float],
) -> List[Dict[str, Any]]:
    """Split batched MIGraphX outputs back into per-frame pipeline items."""
    out_items: List[Dict[str, Any]] = []
    n = len(batch_items)
    if n <= 0:
        return out_items

    for i, item in enumerate(batch_items):
        out_item = {
            "camera_id": int(item["camera_id"]),
            "frame_id": int(item["frame_id"]),
            "source": item["source"],
            "capture_ts": float(item["capture_ts"]),
            "preprocess_done_ts": float(item["preprocess_done_ts"]),
            "infer_done_ts": infer_done_ts,
            "original_hw": tuple(item["original_hw"]),
            "preprocess_ms": float(item["preprocess_ms"]),
            "queue_pre_to_infer_ms": float(queue_wait_times_ms[i]),
            "inference_ms": float(inference_ms_total) / float(n),
            "decode_ms": float(decode_ms_total) / float(n),
            "batch_inference_ms": float(inference_ms_total),
            "batch_decode_ms": float(decode_ms_total),
            "migraphx_batch_size": int(n),
            "heatmaps": np.ascontiguousarray(heatmaps_bhwc[i]),
            "pafs": np.ascontiguousarray(pafs_bhwc[i]),
        }
        if "frame_bgr" in item:
            out_item["frame_bgr"] = item["frame_bgr"]
        out_items.append(out_item)
    return out_items


def collect_queue_batch(
    *,
    first_item: Dict[str, Any],
    in_q,
    batch_size: int,
    batch_timeout_ms: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect a small batch from a FIFO queue. Returns (items, saw_stop_token)."""
    batch_size = max(1, int(batch_size))
    timeout_s = max(0.0, float(batch_timeout_ms)) / 1000.0
    batch_items = [first_item]
    saw_stop = False

    if batch_size <= 1:
        return batch_items, saw_stop

    deadline = time.perf_counter() + timeout_s
    while len(batch_items) < batch_size:
        try:
            if timeout_s > 0.0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                item = in_q.get(timeout=remaining)
            else:
                item = in_q.get_nowait()
        except py_queue.Empty:
            break

        if item is None:
            saw_stop = True
            break
        batch_items.append(item)

    return batch_items, saw_stop


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
    shared_map_descs: Optional[Sequence[Dict[str, Any]]] = None,
    free_map_slots=None,
    migraphx_batch_size: int = 1,
    migraphx_batch_timeout_ms: float = 0.0,
) -> None:
    try:
        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)
        print(
            f"[INFER:{worker_id}] MIGraphX inference batch size={int(migraphx_batch_size)} "
            f"timeout={float(migraphx_batch_timeout_ms):.2f} ms",
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
                results = model.run({"input": input_batch})
            inference_times.append(t_inf.ms)
            batch_runs += 1
            batch_sizes_seen.append(actual_batch_size)

            with Timer() as t_dec:
                heatmaps_bhwc, pafs_bhwc = decode_migraphx_batch_outputs(
                    results,
                    out_h,
                    out_w,
                    shared_dtype,
                    batch_size=input_batch.shape[0],
                )
            decode_times.append(t_dec.ms)

            out_items = build_inference_output_items_from_batch(
                batch_items=batch_items,
                heatmaps_bhwc=heatmaps_bhwc,
                pafs_bhwc=pafs_bhwc,
                infer_done_ts=time.perf_counter(),
                inference_ms_total=t_inf.ms,
                decode_ms_total=t_dec.ms,
                queue_wait_times_ms=batch_queue_wait_ms,
            )

            for out_item in out_items:
                if shared_slots and free_map_slots is not None:
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
) -> None:
    try:
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
            require_gpu=bool(require_gpu and wants_torch and torch_device == "cuda"),
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
    migraphx_batch_size: int = 1,
    migraphx_batch_timeout_ms: float = 0.0,
) -> None:
    """Round-robin MIGraphX worker over newest-frame slots, with optional batched inference."""
    try:
        configure_child_cpu_runtime(int(os.environ.get("STREAM_WORKER_THREADS", "1")))
        import migraphx

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find model: {model_path}")

        print(f"[INFER:{worker_id}] Loading MIGraphX model: {model_path}", flush=True)
        model = migraphx.load(model_path)
        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        print(f"[INFER:{worker_id}] Model loaded. Expected dtype: {expected_dtype}", flush=True)
        print(
            f"[INFER:{worker_id}] MIGraphX inference batch size={int(migraphx_batch_size)} "
            f"timeout={float(migraphx_batch_timeout_ms):.2f} ms",
            flush=True,
        )

        out_h = target_h // stride
        out_w = target_w // stride
        shared_slots, shared_handles = open_shared_map_buffers(shared_map_descs)
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

                try:
                    return in_queues[cam_id].get_nowait(), skipped_this_scan
                except py_queue.Empty:
                    continue
            return None, skipped_this_scan

        while True:
            item, skipped_this_scan = _get_next_item()

            if item is None:
                if all_done(camera_done) and all_queues_empty(in_queues):
                    break
                if skipped_this_scan > 0:
                    backpressure_idle_loops += 1
                time.sleep(poll_sleep_s)
                continue

            batch_items = [item]
            if configured_batch_size > 1:
                deadline = time.perf_counter() + batch_timeout_s
                while len(batch_items) < configured_batch_size:
                    extra, _ = _get_next_item()
                    if extra is not None:
                        batch_items.append(extra)
                        continue
                    if batch_timeout_s <= 0.0 or time.perf_counter() >= deadline:
                        break
                    time.sleep(min(poll_sleep_s, max(0.0, deadline - time.perf_counter())))

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
            )

            with Timer() as t_inf:
                results = model.run({"input": input_batch})
            inference_times.append(t_inf.ms)
            batch_runs += 1
            batch_sizes_seen.append(actual_batch_size)

            with Timer() as t_dec:
                heatmaps_bhwc, pafs_bhwc = decode_migraphx_batch_outputs(
                    results,
                    out_h,
                    out_w,
                    shared_dtype,
                    batch_size=input_batch.shape[0],
                )
            decode_times.append(t_dec.ms)

            out_items = build_inference_output_items_from_batch(
                batch_items=batch_items,
                heatmaps_bhwc=heatmaps_bhwc,
                pafs_bhwc=pafs_bhwc,
                infer_done_ts=time.perf_counter(),
                inference_ms_total=t_inf.ms,
                decode_ms_total=t_dec.ms,
                queue_wait_times_ms=batch_queue_wait_ms,
            )

            for out_item in out_items:
                cam_id = int(out_item["camera_id"])

                if shared_slots and free_map_slots is not None:
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
        stats_q.put(
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
                "wall_s": time.perf_counter() - t_worker_start,
            }
        )
        close_shared_map_views(shared_handles)
        print(
            f"[INFER:{worker_id}] Done. processed={processed} batch_runs={batch_runs} "
            f"avg_real_batch={mean(batch_sizes_seen):.2f} replaced_before_post={replaced_before_post} "
            f"backpressure_skips={skipped_due_backpressure} soft_overrides={soft_overrides} "
            f"throttle_skips={throttle_skips}",
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
) -> None:
    """Round-robin postprocess worker over newest decoded-map slots, one slot per camera."""
    try:
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
            require_gpu=bool(require_gpu and wants_torch and torch_device == "cuda"),
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

                if use_gpu_nms_batch and len(batch_items) > 1:
                    batch_inputs = []
                    for bi in batch_items:
                        hm, pf = _maps_for(bi)
                        batch_inputs.append((hm, pf, tuple(bi["original_hw"])))
                    batch_outputs = postprocess_gpu_nms_fullres_batch(batch_inputs, config=config)
                else:
                    batch_outputs = []
                    for bi in batch_items:
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

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def apply_warmup_filter(rows: List[Dict[str, Any]], args) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    warmup_s = max(0.0, float(getattr(args, "warmup_s", 0.0) or 0.0))
    warmup_frames = max(0, int(getattr(args, "warmup_output_frames", 0) or 0))
    if not rows or (warmup_s <= 0.0 and warmup_frames <= 0):
        return rows, {
            "warmup_s": warmup_s,
            "warmup_output_frames": warmup_frames,
            "raw_total_processed_frames": len(rows),
            "warmup_discarded_frames": 0,
        }

    ordered = sorted(rows, key=lambda r: (safe_float(r.get("post_done_ts", 0.0)), int(r.get("camera_id", 0)), int(r.get("frame_id", 0))))
    filtered = ordered
    if warmup_s > 0.0:
        ts_values = [safe_float(r.get("post_done_ts", 0.0)) for r in ordered if safe_float(r.get("post_done_ts", 0.0)) > 0.0]
        if ts_values:
            cutoff = min(ts_values) + warmup_s
            filtered = [r for r in filtered if safe_float(r.get("post_done_ts", 0.0)) >= cutoff]
    if warmup_frames > 0:
        filtered = filtered[warmup_frames:]

    return filtered, {
        "warmup_s": warmup_s,
        "warmup_output_frames": warmup_frames,
        "raw_total_processed_frames": len(rows),
        "warmup_discarded_frames": len(rows) - len(filtered),
    }


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
    if summary.get("warmup_discarded_frames", 0):
        print(
            f"Warmup discarded:         {summary['warmup_discarded_frames']} / "
            f"{summary.get('raw_total_processed_frames', summary['total_processed_frames'])} frames"
        )
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
        "post_done_ts",
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
                prealloc_resize_buffers=args.prealloc_resize_buffers,
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
    summary["queue_policy"] = args.queue_policy
    summary["buffer_mode"] = "queue"
    summary["shared_map_slots"] = getattr(args, "shared_map_slots", 0)
    summary["migraphx_batch_size"] = getattr(args, "migraphx_batch_size", 1)
    summary["migraphx_batch_timeout_ms"] = getattr(args, "migraphx_batch_timeout_ms", 0.0)
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
                migraphx_batch_size=args.migraphx_batch_size,
                migraphx_batch_timeout_ms=args.migraphx_batch_timeout_ms,
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
                shared_map_descs=shared_map_descs,
                free_map_slots=free_map_slots,
                prealloc_resize_buffers=args.prealloc_resize_buffers,
                gpu_nms_batch_size=args.gpu_nms_batch_size,
                gpu_nms_batch_timeout_ms=args.gpu_nms_batch_timeout_ms,
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
    summary["migraphx_batch_size"] = getattr(args, "migraphx_batch_size", 1)
    summary["migraphx_batch_timeout_ms"] = getattr(args, "migraphx_batch_timeout_ms", 0.0)
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
        help="Legacy alias for --backpressure-mode off.",
    )
    parser.add_argument(
        "--backpressure-mode",
        choices=["off", "strict", "soft"],
        default="strict",
        help=(
            "Backpressure policy for --buffer-mode latest. "
            "'off': never skip cameras (max throughput, results may be overwritten). "
            "'strict': skip a camera while its post_pending flag is set (original behaviour). "
            "'soft': skip only while the pending result is fresher than --max-pending-age-ms; "
            "allows re-inference once a result has been sitting too long. "
            "--disable-backpressure is a legacy alias for 'off'."
        ),
    )
    parser.add_argument(
        "--max-pending-age-ms",
        type=float,
        default=300.0,
        help=(
            "Used with --backpressure-mode soft. A camera whose pending postprocess result "
            "is older than this threshold (ms) is eligible for re-inference even though "
            "post_pending is still set. Prevents slow post workers from starving cameras. "
            "Default: 300 ms."
        ),
    )
    parser.add_argument(
        "--target-output-fps-per-camera",
        type=float,
        default=0.0,
        help=(
            "When > 0, the inference scheduler skips cameras that were fully postprocessed "
            "more recently than 1/fps seconds ago. Useful to cap per-camera processing rate "
            "and ensure fair share across cameras in mixed-difficulty scenes. 0 = disabled."
        ),
    )

    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=1)
    parser.add_argument(
        "--migraphx-batch-size",
        type=int,
        default=1,
        help=(
            "Batch size used by MIGraphX inference workers. Use 1 for old behavior. "
            "For static batch MXR models, set this to the compiled batch size, e.g. 2/4/8."
        ),
    )
    parser.add_argument(
        "--migraphx-batch-timeout-ms",
        type=float,
        default=0.0,
        help=(
            "Maximum time an inference worker waits to fill a MIGraphX batch. "
            "Use a small value such as 2-8 ms for live simulation."
        ),
    )
    parser.add_argument("--preprocess-queue-size", type=int, default=30)
    parser.add_argument("--postprocess-queue-size", type=int, default=30)

    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--shared-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--shared-map-slots",
        type=int,
        default=0,
        help="Latest-mode only: preallocate this many shared-memory heatmap/PAF slots between inference and postprocess. 0 keeps Queue pickle/copy.",
    )

    parser.add_argument("--torch-device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6)
    parser.add_argument("--nms-radius-lowres", type=int, default=1)
    parser.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    parser.add_argument("--gpu-compute-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--prealloc-resize-buffers",
        action="store_true",
        help="Reuse persistent cv2.resize dst buffers inside each postprocess worker when supported by OpenCV.",
    )
    parser.add_argument(
        "--gpu-nms-batch-size",
        type=int,
        default=1,
        help="Latest-mode gpu_nms_fullres_two_process only: batch this many frames per post worker for Torch max_pool NMS. 1 disables batching.",
    )
    parser.add_argument(
        "--gpu-nms-batch-timeout-ms",
        type=float,
        default=0.0,
        help="Maximum wait to fill a gpu_nms batch before running it. Keep small, e.g. 2-8 ms, for live feeds.",
    )
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

    parser.add_argument("--pin-cpus", action="store_true", help="Pin each camera, inference worker, and postprocess worker to distinct CPU cores.")
    parser.add_argument("--pin-camera-base", type=int, default=0, help="First CPU core for camera workers when --pin-cpus is set.")
    parser.add_argument("--pin-inference-base", type=int, default=10, help="First CPU core for inference workers when --pin-cpus is set.")
    parser.add_argument("--pin-post-base", type=int, default=12, help="First CPU core for postprocess workers when --pin-cpus is set.")
    parser.add_argument("--pin-all-threads", action="store_true", help="Also pin existing native threads under /proc/<pid>/task for each worker after startup.")
    parser.add_argument("--worker-threads", type=int, default=1, help="Set OpenCV/OpenMP/OpenBLAS/NumExpr/PyTorch CPU thread pools per worker. Default: 1.")
    parser.add_argument("--warmup-s", type=float, default=0.0, help="Discard output rows whose postprocess completion is within this many seconds of the first output row.")
    parser.add_argument("--warmup-output-frames", type=int, default=0, help="Discard this many additional earliest output rows before computing the summary.")

    parser.add_argument(
        "--profile-system",
        action="store_true",
        help="Collect parent-side per-PID CPU/memory, affinity, per-core CPU, GPU busy, and VRAM stats.",
    )
    parser.add_argument(
        "--profile-interval-s",
        type=float,
        default=0.1,
        help="Sampling interval for --profile-system. Default: 0.1 s.",
    )
    parser.add_argument(
        "--report-affinity",
        action="store_true",
        help="Print worker CPU affinity after all child processes are started.",
    )

    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--detailed-csv", default="outputs/stream_10cam_detailed.csv")
    parser.add_argument("--summary-json", default="outputs/stream_10cam_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())