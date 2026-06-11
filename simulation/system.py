"""Process, CPU affinity, thread-pool, and lightweight system monitor helpers."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import multiprocessing as mp

import numpy as np

from .tracing import allow_ptrace_attach_if_requested
from .utils import mean


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


    allow_ptrace_attach_if_requested()


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
