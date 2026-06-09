#!/usr/bin/env python3
"""
profile_pipeline.py
===================
Detaljno profilisanje AMD ROCm pipeline-a za 10-kamera simulaciju.

Mjeri:
  1. Vremenski breakdown po stage-u (sub-stage rezolucija)
  2. Koji CPU core opterećuje svaki worker-process
  3. Koliko opterećuje svaki core (% utilization, per-PID i per-core)
  4. GPU% utilization tokom rada
  5. HIP pinned-memory vs. pageable latency benchmark
  6. ROCTx markeri za rocprofv3 GPU timeline
  7. Preporuke za CPU affinity pinovanje

Pokretanje:
  # Osnovno profilisanje:
  python tools/profile_pipeline.py --model models/pose_model1_fp16_ref1.mxr --frames 60

  # Sa rocprofv3 GPU trace-om (HIP kernel timeline):
  rocprofv3 --hip-trace --hsa-trace --kernel-trace \\
      -o rocprof_out -- \\
      python tools/profile_pipeline.py --model models/pose_model1_fp16_ref1.mxr --frames 30

  # Sa rocprof-sys (Omnitrace) - system-wide multi-process profiling:
  rocprof-sys-sample --sampling-freq 200 -- \\
      python tools/profile_pipeline.py --model models/pose_model1_fp16_ref1.mxr --frames 30

  # Sa CPU affinity piniranjem:
  python tools/profile_pipeline.py --pin-cpus --model models/pose_model1_fp16_ref1.mxr --frames 60

  # Hip pinned-memory latency test (bez modela):
  python tools/profile_pipeline.py --hip-pin-bench-only
"""

from __future__ import annotations

import argparse
import ctypes
import math
import multiprocessing as mp
import os
import queue as py_queue
import struct
import time
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 1.  /proc čitanje - bez psutil zavisnosti
# ---------------------------------------------------------------------------

def _read_proc_stat() -> Dict[str, List[int]]:
    """Čita /proc/stat → {cpu: [user, nice, sys, idle, iowait, irq, softirq, ...]}"""
    result: Dict[str, List[int]] = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                parts = line.split()
                if parts[0].startswith("cpu"):
                    result[parts[0]] = [int(x) for x in parts[1:]]
    except Exception:
        pass
    return result


def _cpu_pct(s0: List[int], s1: List[int]) -> float:
    """Izračunava CPU% između dva očitavanja /proc/stat."""
    if not s0 or not s1:
        return 0.0
    idle0 = s0[3] + (s0[4] if len(s0) > 4 else 0)
    idle1 = s1[3] + (s1[4] if len(s1) > 4 else 0)
    total0 = sum(s0)
    total1 = sum(s1)
    dt = total1 - total0
    if dt <= 0:
        return 0.0
    return max(0.0, min(100.0, (1.0 - (idle1 - idle0) / dt) * 100.0))


def _read_pid_stat(pid: int) -> Optional[Tuple[int, int]]:
    """Čita /proc/<pid>/stat → (utime, stime) u clock ticks."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        # Polje 14 i 15 (0-indexed 13, 14) su utime i stime.
        # Treba skočiti iza zagrada jer comm može sadržati razmake.
        rp = data.rfind(")")
        fields = data[rp + 2:].split()
        utime = int(fields[11])
        stime = int(fields[12])
        return utime, stime
    except Exception:
        return None


def _get_clock_ticks() -> int:
    try:
        import os
        return os.sysconf("SC_CLK_TCK")
    except Exception:
        return 100


def _read_pid_cpu_affinity(pid: int) -> List[int]:
    """Čita CPU affinity za PID koristeći sched_getaffinity."""
    try:
        return sorted(os.sched_getaffinity(pid))
    except Exception:
        return []


def _read_pid_memory_kb(pid: int) -> Dict[str, int]:
    """Čita osnovne memory metrike iz /proc/<pid>/status u KiB."""
    wanted = {"VmRSS", "VmSize", "VmHWM", "RssAnon", "RssFile", "RssShmem"}
    result: Dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key not in wanted:
                    continue
                parts = rest.strip().split()
                if parts:
                    result[key] = int(parts[0])
    except Exception:
        pass
    return result


def _read_gpu_busy_pct() -> float:
    """Čita GPU% iz sysfs - najbrži pristup, bez CLI spawna."""
    paths = [
        "/sys/class/drm/card1/device/gpu_busy_percent",
        "/sys/class/drm/card0/device/gpu_busy_percent",
    ]
    for p in paths:
        try:
            return float(Path(p).read_text().strip())
        except Exception:
            continue
    return -1.0


def _read_gpu_vram_mb() -> float:
    """Čita VRAM korišćenje u MB iz sysfs."""
    paths = [
        "/sys/class/drm/card1/device/mem_info_vram_used",
        "/sys/class/drm/card0/device/mem_info_vram_used",
    ]
    for p in paths:
        try:
            return float(Path(p).read_text().strip()) / (1024 ** 2)
        except Exception:
            continue
    return -1.0


# ---------------------------------------------------------------------------
# 2.  SysMonitor - pozadinski thread za sistem monitoring
# ---------------------------------------------------------------------------

class SysMonitor:
    """
    Prati CPU (per-core) i GPU utilization u pozadini.
    Počni sa start(), pauziraj polling dok traje warmup, zatim resume(),
    na kraju stop() → vrati ProfileStats dict.
    """

    def __init__(self, poll_interval_s: float = 0.1):
        self._interval = poll_interval_s
        self._pids: Dict[str, List[int]] = {}  # ime_grupe → [pid1, pid2, ...]
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # počni pauzirano
        self._thread: Optional[threading.Thread] = None

        # Sakupljeni podaci
        self._gpu_samples: List[float] = []
        self._vram_samples: List[float] = []
        self._core_samples: Dict[str, List[float]] = defaultdict(list)  # "cpu0" → [pct, ...]
        self._pid_cpu_ticks: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)  # pid → [(ts, utick, stick)]
        self._pid_mem_samples: Dict[int, List[Dict[str, int]]] = defaultdict(list)
        self._pid_affinity: Dict[int, List[int]] = {}
        self._clock_ticks = _get_clock_ticks()

    def register_pids(self, group: str, pids: List[int]) -> None:
        self._pids[group] = pids
        for pid in pids:
            aff = _read_pid_cpu_affinity(pid)
            self._pid_affinity[pid] = aff

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SysMonitor")
        self._thread.start()

    def resume(self) -> None:
        """Počni skupljati uzorke (pozovi nakon warmup-a)."""
        self._paused.clear()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        return self._compute_stats()

    def _loop(self) -> None:
        prev_stat = _read_proc_stat()
        while not self._stop.is_set():
            time.sleep(self._interval)
            if self._paused.is_set():
                prev_stat = _read_proc_stat()  # resetuj baseline tokom pauze
                continue

            # GPU
            gpu_pct = _read_gpu_busy_pct()
            if gpu_pct >= 0:
                self._gpu_samples.append(gpu_pct)
            vram = _read_gpu_vram_mb()
            if vram >= 0:
                self._vram_samples.append(vram)

            # Per-core CPU
            curr_stat = _read_proc_stat()
            for key in curr_stat:
                if key == "cpu":
                    continue
                pct = _cpu_pct(prev_stat.get(key, []), curr_stat.get(key, []))
                self._core_samples[key].append(pct)
            prev_stat = curr_stat

            # Per-PID CPU ticks
            ts = time.monotonic()
            for pids in self._pids.values():
                for pid in pids:
                    ticks = _read_pid_stat(pid)
                    if ticks:
                        self._pid_cpu_ticks[pid].append((ts, ticks[0], ticks[1]))
                    mem = _read_pid_memory_kb(pid)
                    if mem:
                        self._pid_mem_samples[pid].append(mem)

    def _compute_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        # GPU
        g = self._gpu_samples
        stats["gpu_avg_pct"] = _mean(g)
        stats["gpu_p95_pct"] = _pctile(g, 95)
        stats["gpu_peak_pct"] = max(g) if g else 0.0
        stats["gpu_idle_pct"] = (sum(1 for v in g if v < 5) / len(g) * 100.0) if g else 0.0
        stats["vram_avg_mb"] = _mean(self._vram_samples)
        stats["gpu_samples"] = len(g)

        # Per-core CPU
        core_avgs: Dict[str, float] = {}
        for core, samples in self._core_samples.items():
            core_avgs[core] = _mean(samples)
        stats["core_avg_pct"] = core_avgs

        # Per-PID CPU% average
        pid_stats: Dict[int, Dict[str, Any]] = {}
        for group, pids in self._pids.items():
            for pid in pids:
                samples = self._pid_cpu_ticks.get(pid, [])
                if len(samples) < 2:
                    cpu_pct_avg = 0.0
                else:
                    s0 = samples[0]
                    s1 = samples[-1]
                    dt_s = s1[0] - s0[0]
                    dt_ticks = (s1[1] + s1[2]) - (s0[1] + s0[2])
                    cpu_pct_avg = (dt_ticks / self._clock_ticks / dt_s * 100.0) if dt_s > 0 else 0.0
                mem_samples = self._pid_mem_samples.get(pid, [])
                rss_vals = [m.get("VmRSS", 0) for m in mem_samples]
                vms_vals = [m.get("VmSize", 0) for m in mem_samples]
                hwm_vals = [m.get("VmHWM", 0) for m in mem_samples]
                anon_vals = [m.get("RssAnon", 0) for m in mem_samples]
                file_vals = [m.get("RssFile", 0) for m in mem_samples]
                shmem_vals = [m.get("RssShmem", 0) for m in mem_samples]
                pid_stats[pid] = {
                    "group": group,
                    "cpu_pct": cpu_pct_avg,
                    "affinity": _read_pid_cpu_affinity(pid) or self._pid_affinity.get(pid, []),
                    "rss_avg_mb": _mean(rss_vals) / 1024.0,
                    "rss_peak_mb": (max(rss_vals) / 1024.0) if rss_vals else 0.0,
                    "vms_avg_mb": _mean(vms_vals) / 1024.0,
                    "hwm_peak_mb": (max(hwm_vals) / 1024.0) if hwm_vals else 0.0,
                    "anon_avg_mb": _mean(anon_vals) / 1024.0,
                    "file_avg_mb": _mean(file_vals) / 1024.0,
                    "shmem_avg_mb": _mean(shmem_vals) / 1024.0,
                    "mem_samples": len(mem_samples),
                }
        stats["pid_stats"] = pid_stats
        stats["pid_groups"] = {g: pids for g, pids in self._pids.items()}

        return stats


# ---------------------------------------------------------------------------
# 3.  ROCTx wrapper (libroctx64.so) - markeri za rocprofv3 timeline
# ---------------------------------------------------------------------------

class RocTx:
    """
    Tanak ctypes wrapper za libroctx64.
    Koristi se u inference worker-u (isti process koji radi HIP pozive)
    da markira regione vidljive u rocprofv3 timeline-u.

    Primjer:
        roctx = RocTx()
        with roctx.range("migraphx_inference"):
            results = model.run(...)
    """

    _lib: Optional[Any] = None
    _loaded: bool = False

    @classmethod
    def _load(cls) -> bool:
        if cls._loaded:
            return cls._lib is not None
        cls._loaded = True
        for name in ["libroctx64.so", "libroctx64.so.4",
                     "/opt/rocm/lib/libroctx64.so"]:
            try:
                lib = ctypes.CDLL(name)
                lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
                lib.roctxRangePushA.restype = ctypes.c_int
                lib.roctxRangePop.argtypes = []
                lib.roctxRangePop.restype = None
                lib.roctxMarkA.argtypes = [ctypes.c_char_p]
                lib.roctxMarkA.restype = None
                cls._lib = lib
                return True
            except Exception:
                continue
        return False

    def push(self, name: str) -> None:
        if self._load() and self._lib:
            self._lib.roctxRangePushA(name.encode())

    def pop(self) -> None:
        if self._load() and self._lib:
            self._lib.roctxRangePop()

    def mark(self, name: str) -> None:
        if self._load() and self._lib:
            self._lib.roctxMarkA(name.encode())

    class _Range:
        def __init__(self, roctx: "RocTx", name: str):
            self._r = roctx
            self._n = name

        def __enter__(self):
            self._r.push(self._n)
            return self

        def __exit__(self, *_):
            self._r.pop()

    def range(self, name: str) -> "_Range":
        return self._Range(self, name)


# ---------------------------------------------------------------------------
# 4.  HIP Pinned Memory benchmark
# ---------------------------------------------------------------------------

class HipPinnedMemBench:
    """
    Mjeri razliku u latenciji između:
      - Regular (pageable) host memory → device (H2D)
      - Pinned (page-locked) host memory → device (H2D)

    Pinned memory eliminira jedan korak DMA mapiranja i može smanjiti
    latenciju za 15-40% za manje tensore (< 4 MB).
    """

    # HIP constants
    hipSuccess = 0
    hipMemcpyHostToDevice = 1
    hipHostMallocDefault = 0
    hipHostMallocPortable = 1

    def __init__(self):
        self._hip: Optional[Any] = None
        self._available = False
        self._load()

    def _load(self) -> None:
        for name in ["libamdhip64.so", "libamdhip64.so.7",
                     "/opt/rocm/lib/libamdhip64.so"]:
            try:
                lib = ctypes.CDLL(name)
                # hipMalloc(void** ptr, size_t size)
                lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
                lib.hipMalloc.restype = ctypes.c_int
                # hipFree(void* ptr)
                lib.hipFree.argtypes = [ctypes.c_void_p]
                lib.hipFree.restype = ctypes.c_int
                # hipHostMalloc(void** ptr, size_t size, unsigned int flags)
                lib.hipHostMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint
                ]
                lib.hipHostMalloc.restype = ctypes.c_int
                # hipHostFree(void* ptr)
                lib.hipHostFree.argtypes = [ctypes.c_void_p]
                lib.hipHostFree.restype = ctypes.c_int
                # hipMemcpy(dst, src, size, kind)
                lib.hipMemcpy.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_size_t, ctypes.c_int
                ]
                lib.hipMemcpy.restype = ctypes.c_int
                # hipDeviceSynchronize()
                lib.hipDeviceSynchronize.argtypes = []
                lib.hipDeviceSynchronize.restype = ctypes.c_int
                self._hip = lib
                self._available = True
                return
            except Exception:
                continue

    def run(self, tensor_shape: Tuple[int, ...] = (1, 3, 544, 968),
            dtype_bytes: int = 4, repeats: int = 50) -> Dict[str, Any]:
        """Pokreće benchmark. Vraća dict sa mjerenjima."""
        import numpy as np

        if not self._available:
            return {"available": False, "reason": "libamdhip64 nije učitana"}

        lib = self._hip
        size_bytes = int(np.prod(tensor_shape)) * dtype_bytes
        print(f"\n[HIP-PIN] Benchmark: tensor {tensor_shape} dtype_bytes={dtype_bytes} "
              f"size={size_bytes/1024:.1f} KB repeats={repeats}", flush=True)

        # Alociraj GPU bafer
        gpu_ptr = ctypes.c_void_p()
        ret = lib.hipMalloc(ctypes.byref(gpu_ptr), size_bytes)
        if ret != self.hipSuccess:
            return {"available": True, "error": f"hipMalloc failed: {ret}"}

        results: Dict[str, Any] = {"available": True, "size_bytes": size_bytes}

        # --- Pageable (regular) memory ---
        host_pageable = np.random.rand(*tensor_shape).astype(np.float32)
        host_ptr_pag = host_pageable.ctypes.data_as(ctypes.c_void_p)
        times_pageable = []
        for _ in range(repeats):
            lib.hipDeviceSynchronize()
            t0 = time.perf_counter()
            lib.hipMemcpy(gpu_ptr, host_ptr_pag, size_bytes, self.hipMemcpyHostToDevice)
            lib.hipDeviceSynchronize()
            times_pageable.append((time.perf_counter() - t0) * 1000.0)
        times_pageable = times_pageable[5:]  # odbaci warmup

        # --- Pinned memory ---
        pin_ptr = ctypes.c_void_p()
        ret = lib.hipHostMalloc(ctypes.byref(pin_ptr), size_bytes, self.hipHostMallocDefault)
        if ret != self.hipSuccess:
            lib.hipFree(gpu_ptr)
            return {"available": True, "error": f"hipHostMalloc failed: {ret}"}

        # Kopiraj podatke u pinned bafer
        ctypes.memmove(pin_ptr, host_pageable.ctypes.data_as(ctypes.c_void_p), size_bytes)

        times_pinned = []
        for _ in range(repeats):
            lib.hipDeviceSynchronize()
            t0 = time.perf_counter()
            lib.hipMemcpy(gpu_ptr, pin_ptr, size_bytes, self.hipMemcpyHostToDevice)
            lib.hipDeviceSynchronize()
            times_pinned.append((time.perf_counter() - t0) * 1000.0)
        times_pinned = times_pinned[5:]

        lib.hipHostFree(pin_ptr)
        lib.hipFree(gpu_ptr)

        results.update({
            "pageable_avg_ms": _mean(times_pageable),
            "pageable_p95_ms": _pctile(times_pageable, 95),
            "pageable_min_ms": min(times_pageable) if times_pageable else 0,
            "pinned_avg_ms": _mean(times_pinned),
            "pinned_p95_ms": _pctile(times_pinned, 95),
            "pinned_min_ms": min(times_pinned) if times_pinned else 0,
            "speedup_pct": (
                (_mean(times_pageable) - _mean(times_pinned)) / _mean(times_pageable) * 100.0
                if _mean(times_pageable) > 0 else 0.0
            ),
        })
        return results


# ---------------------------------------------------------------------------
# 5.  CPU Affinity Manager
# ---------------------------------------------------------------------------

class CpuAffinityManager:
    """
    Topologija: 32 CPU-a, 16 fizičkih jezgara × 2 HT nit
    Fizičko jezgro K → CPUs K i K+16

    Preporučeni raspored (10 kamera, 1 infer, 1-2 post):
      cameras  0-9  → fizička jezgra 0-9   (CPUs 0-9)
      infer  worker → fizičko jezgro 10     (CPU 10)
      post workers  → fizička jezgra 11-15  (CPUs 11-15)
      (HT parovi 16-31 ostaju slobodni ili za OS scheduler)
    """

    NCPU: int = 32
    NPHYS: int = 16  # fizička jezgra

    @staticmethod
    def physical_to_logical(phys_core: int) -> List[int]:
        """Vraća CPU-ove koji odgovaraju fizičkom jezgru (uključujući HT)."""
        # Na ovoj mašini: phys K → CPU K i CPU K+16
        return [phys_core, phys_core + 16]

    @classmethod
    def pin_camera_workers(cls, pids: List[int]) -> Dict[int, List[int]]:
        """Pini svaku kameru na vlastito fizičko jezgro."""
        pinning: Dict[int, List[int]] = {}
        for idx, pid in enumerate(pids):
            core = idx % 10  # fizička jezgra 0-9
            cpus = [core]    # samo primarna nit, ne HT
            try:
                os.sched_setaffinity(pid, cpus)
                pinning[pid] = cpus
            except PermissionError:
                # Potreban je CAP_SYS_NICE ili isti user; prikazati upozorenje
                pinning[pid] = []
            except Exception:
                pinning[pid] = []
        return pinning

    @classmethod
    def pin_inference_workers(cls, pids: List[int]) -> Dict[int, List[int]]:
        """Pini inference worker-e na jezgra 10-11 (minimalna CPU konkurencija)."""
        pinning: Dict[int, List[int]] = {}
        for idx, pid in enumerate(pids):
            core = 10 + (idx % 4)
            cpus = [core]
            try:
                os.sched_setaffinity(pid, cpus)
                pinning[pid] = cpus
            except Exception:
                pinning[pid] = []
        return pinning

    @classmethod
    def pin_postprocess_workers(cls, pids: List[int]) -> Dict[int, List[int]]:
        """Pini post worker-e na jezgra 12-15 (4 jezgra za CPU-intensive NMS)."""
        pinning: Dict[int, List[int]] = {}
        for idx, pid in enumerate(pids):
            # 4 dostupna jezgra za post radnike
            cpus = list(range(12, 16))
            try:
                os.sched_setaffinity(pid, cpus)
                pinning[pid] = cpus
            except Exception:
                pinning[pid] = []
        return pinning

    @classmethod
    def report_current_affinity(cls, pid_groups: Dict[str, List[int]]) -> None:
        """Ispisuje trenutni CPU affinity za sve praćene procese."""
        print("\n[CPU AFFINITY]")
        print(f"{'Grupa':<20} {'PID':>8} {'CPUs'}")
        print("-" * 60)
        for group, pids in pid_groups.items():
            for pid in pids:
                aff = _read_pid_cpu_affinity(pid)
                aff_str = ",".join(map(str, aff)) if aff else "N/A"
                print(f"  {group:<18} {pid:>8} [{aff_str}]")


# ---------------------------------------------------------------------------
# 6.  Profiled worker funkcije (prošireni workers sa sub-stage timing-om)
# ---------------------------------------------------------------------------

def _mean(v: Sequence) -> float:
    vals = [float(x) for x in v if x is not None and not math.isnan(float(x))]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _pctile(v: Sequence, q: float) -> float:
    vals = sorted(float(x) for x in v if x is not None and not math.isnan(float(x)))
    if not vals:
        return 0.0
    idx = (len(vals) - 1) * q / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def camera_preprocess_worker_profiled(
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
    pin_cpu: Optional[int] = None,
) -> None:
    """
    Profilisana verzija camera_preprocess_worker.
    Mjeri sub-stage timing: cap.read, cv2.resize, normalize, transpose.
    """
    try:
        import cv2
        import numpy as np

        if pin_cpu is not None:
            try:
                os.sched_setaffinity(0, [pin_cpu])
            except Exception:
                pass

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Camera {camera_id}: nema videa {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {camera_id}: ne može otvoriti {video_path}")

        attempted = 0
        t_worker_start = time.perf_counter()

        # Sub-stage tajmeri
        t_read: List[float] = []
        t_resize: List[float] = []
        t_norm: List[float] = []
        t_transpose: List[float] = []
        t_total_pre: List[float] = []

        while not stop_event.is_set():
            if max_frames > 0 and attempted >= max_frames:
                break

            with _Timer() as tr:
                ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            t_read.append(tr.ms)

            capture_ts = time.perf_counter()
            original_h, original_w = frame.shape[:2]
            t0_total = time.perf_counter()

            with _Timer() as tres:
                resized = cv2.resize(frame, (target_w, target_h))
            t_resize.append(tres.ms)

            with _Timer() as tnorm:
                normed = (resized.astype(np.float32) - 128.0) / 256.0
            t_norm.append(tnorm.ms)

            with _Timer() as ttr:
                tensor = np.ascontiguousarray(normed.transpose(2, 0, 1)[np.newaxis, ...],
                                               dtype=np.float32)
            t_transpose.append(ttr.ms)

            t_total_pre.append((time.perf_counter() - t0_total) * 1000.0)
            attempted += 1

            item = {
                "camera_id": camera_id,
                "frame_id": attempted,
                "source": video_path,
                "capture_ts": capture_ts,
                "preprocess_done_ts": time.perf_counter(),
                "original_hw": (int(original_h), int(original_w)),
                "preprocess_ms": t_total_pre[-1],
                "input_tensor": tensor,
                # Sub-stage breakdown:
                "sub_read_ms": t_read[-1],
                "sub_resize_ms": t_resize[-1],
                "sub_norm_ms": t_norm[-1],
                "sub_transpose_ms": t_transpose[-1],
            }
            try:
                out_q.put_nowait(item)
            except py_queue.Full:
                pass

        cap.release()
        wall_s = time.perf_counter() - t_worker_start
        stats_q.put({
            "stage": "camera_preprocess",
            "camera_id": camera_id,
            "source": video_path,
            "attempted": attempted,
            "wall_s": wall_s,
            "fps_input": attempted / wall_s if wall_s > 0 else 0.0,
            "avg_read_ms": _mean(t_read),
            "avg_resize_ms": _mean(t_resize),
            "avg_norm_ms": _mean(t_norm),
            "avg_transpose_ms": _mean(t_transpose),
            "avg_preprocess_ms": _mean(t_total_pre),
            "p95_preprocess_ms": _pctile(t_total_pre, 95),
            "p95_read_ms": _pctile(t_read, 95),
            "p95_resize_ms": _pctile(t_resize, 95),
            "pid": os.getpid(),
            "cpu_affinity": _read_pid_cpu_affinity(os.getpid()),
        })

    except Exception:
        error_q.put({"stage": "camera_preprocess", "camera_id": camera_id,
                     "traceback": traceback.format_exc()})


def inference_worker_profiled(
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
    pin_cpu: Optional[int] = None,
    use_roctx: bool = True,
) -> None:
    """
    Profilisana verzija inference_worker.
    Sub-stage: dtype_cast, model.run (GPU), decode_outputs.
    ROCTx markeri za rocprofv3 timeline.
    """
    try:
        import migraphx
        import numpy as np

        if pin_cpu is not None:
            try:
                os.sched_setaffinity(0, [pin_cpu])
            except Exception:
                pass

        roctx = RocTx() if use_roctx else None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model nije pronađen: {model_path}")

        print(f"[INFER-PROF:{worker_id}] PID={os.getpid()} "
              f"affinity={_read_pid_cpu_affinity(os.getpid())} "
              f"Učitavam model: {model_path}", flush=True)

        if roctx:
            roctx.mark("migraphx_load_start")
        model = migraphx.load(model_path)
        if roctx:
            roctx.mark("migraphx_load_done")

        expected_dtype = str(model.get_parameter_shapes()["input"].type())
        out_h = target_h // stride
        out_w = target_w // stride

        processed = 0
        t_cast: List[float] = []
        t_infer: List[float] = []
        t_decode: List[float] = []
        t_queue_wait: List[float] = []
        t_worker_start = time.perf_counter()

        while True:
            item = in_q.get()
            if item is None:
                break

            infer_start = time.perf_counter()
            t_queue_wait.append((infer_start - float(item.get("preprocess_done_ts", infer_start))) * 1000.0)

            # Sub-stage 1: dtype cast (CPU memory op)
            with _Timer() as tc:
                if roctx:
                    roctx.push("dtype_cast")
                if "half" in expected_dtype:
                    inp = np.ascontiguousarray(item["input_tensor"].astype(np.float16, copy=False))
                else:
                    inp = np.ascontiguousarray(item["input_tensor"].astype(np.float32, copy=False))
                if roctx:
                    roctx.pop()
            t_cast.append(tc.ms)

            # Sub-stage 2: GPU inference (H2D + kernel + D2H)
            with _Timer() as ti:
                if roctx:
                    roctx.push("migraphx_run")
                results = model.run({"input": inp})
                if roctx:
                    roctx.pop()
            t_infer.append(ti.ms)

            # Sub-stage 3: decode outputs (CPU)
            with _Timer() as td:
                if roctx:
                    roctx.push("decode_outputs")
                if not isinstance(results, (list, tuple)):
                    results = list(results)
                heatmaps = np.asarray(results[-2], dtype=np.float32).reshape(19, out_h, out_w)
                pafs = np.asarray(results[-1], dtype=np.float32).reshape(38, out_h, out_w)
                heatmaps = np.moveaxis(heatmaps, 0, -1)
                pafs = np.moveaxis(pafs, 0, -1)
                if roctx:
                    roctx.pop()
            t_decode.append(td.ms)

            out_q.put({
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "capture_ts": float(item["capture_ts"]),
                "preprocess_done_ts": float(item["preprocess_done_ts"]),
                "infer_done_ts": time.perf_counter(),
                "original_hw": tuple(item["original_hw"]),
                "preprocess_ms": float(item["preprocess_ms"]),
                "sub_read_ms": float(item.get("sub_read_ms", 0.0)),
                "sub_resize_ms": float(item.get("sub_resize_ms", 0.0)),
                "sub_norm_ms": float(item.get("sub_norm_ms", 0.0)),
                "sub_transpose_ms": float(item.get("sub_transpose_ms", 0.0)),
                "queue_pre_to_infer_ms": t_queue_wait[-1],
                "sub_cast_ms": float(tc.ms),
                "inference_ms": float(ti.ms),
                "decode_ms": float(td.ms),
                "heatmaps": heatmaps,
                "pafs": pafs,
            })
            processed += 1

        stats_q.put({
            "stage": "inference",
            "worker_id": worker_id,
            "processed": processed,
            "avg_queue_pre_to_infer_ms": _mean(t_queue_wait),
            "p95_queue_pre_to_infer_ms": _pctile(t_queue_wait, 95),
            "avg_cast_ms": _mean(t_cast),
            "avg_inference_ms": _mean(t_infer),
            "p95_inference_ms": _pctile(t_infer, 95),
            "avg_decode_ms": _mean(t_decode),
            "p95_decode_ms": _pctile(t_decode, 95),
            "wall_s": time.perf_counter() - t_worker_start,
            "pid": os.getpid(),
            "cpu_affinity": _read_pid_cpu_affinity(os.getpid()),
        })
        print(f"[INFER-PROF:{worker_id}] Gotovo. processed={processed}", flush=True)

    except Exception:
        error_q.put({"stage": "inference", "worker_id": worker_id,
                     "traceback": traceback.format_exc()})


def postprocess_worker_profiled(
    *,
    worker_id: int,
    variant: str,
    in_q,
    result_q,
    stats_q,
    error_q,
    torch_device: str = "cpu",
    max_keypoints: int = 20,
    threshold: float = 0.1,
    nms_radius_fullres: int = 6,
    nms_radius_lowres: int = 1,
    nms_impl: str = "2d",
    gpu_compute_dtype: str = "float32",
    require_gpu: bool = False,
    pin_cpu: Optional[int] = None,
    use_roctx: bool = True,
) -> None:
    """
    Profilisana verzija postprocess_worker.
    Prikuplja timing iz out.timings dict (već ima sub-breakdown).
    """
    try:
        for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(env_name, "1")

        affinity_error = ""
        if pin_cpu is not None:
            try:
                os.sched_setaffinity(0, pin_cpu if isinstance(pin_cpu, list) else [pin_cpu])
            except Exception as exc:
                affinity_error = f"{type(exc).__name__}: {exc}"
                error_q.put({
                    "stage": "postprocess",
                    "worker_id": worker_id,
                    "warning": f"sched_setaffinity failed: {affinity_error}",
                })

        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass

        from modules.postprocessing import (
            PostprocessConfig,
            is_two_process_mode,
            normalize_mode,
            postprocess_from_maps,
            two_process_worker_mode,
        )

        canonical_variant = normalize_mode(variant)
        registry_variant = canonical_variant
        if is_two_process_mode(canonical_variant):
            worker_mode = two_process_worker_mode(canonical_variant)
            if worker_mode == "gpu-nms-fullres":
                registry_variant = "gpu_nms_fullres_cpu_group"
            elif worker_mode == "gpu-nms-lowres":
                registry_variant = "gpu_nms_lowres_cpu_group"
            elif worker_mode == "cpu-k20-fast":
                registry_variant = "optimized_batch_k20_fast"
            else:
                raise ValueError(f"Unsupported two-process worker mode: {worker_mode}")

        wants_gpu_post = str(registry_variant).startswith("gpu_")
        resolved_torch_device = "cuda" if torch_device == "auto" and wants_gpu_post else torch_device

        roctx = RocTx() if use_roctx else None

        config = PostprocessConfig(
            max_keypoints_per_type=max_keypoints,
            threshold=threshold,
            nms_radius_fullres=nms_radius_fullres,
            nms_radius_lowres=nms_radius_lowres,
            torch_device=resolved_torch_device,
            require_gpu=bool(require_gpu or (wants_gpu_post and resolved_torch_device == "cuda")),
            extra={"gpu_compute_dtype": gpu_compute_dtype, "nms_impl": nms_impl},
        )

        processed = 0
        t_post: List[float] = []
        t_e2e: List[float] = []
        t_queue: List[float] = []
        timing_buckets: Dict[str, List[float]] = defaultdict(list)
        t_worker_start = time.perf_counter()

        while True:
            item = in_q.get()
            if item is None:
                break

            post_start = time.perf_counter()
            t_queue.append((post_start - float(item.get("infer_done_ts", post_start))) * 1000.0)

            if roctx:
                roctx.push(f"postprocess_{variant}")

            out = postprocess_from_maps(
                registry_variant, item["heatmaps"], item["pafs"],
                tuple(item["original_hw"]), config=config,
            )

            if roctx:
                roctx.pop()

            post_done = time.perf_counter()
            timings = dict(out.timings)
            post_ms = float(timings.get("total_postprocess", (post_done - post_start) * 1000.0))
            e2e_ms = (post_done - float(item["capture_ts"])) * 1000.0
            t_post.append(post_ms)
            t_e2e.append(e2e_ms)

            for k, v in timings.items():
                timing_buckets[k].append(float(v))

            row = {
                "camera_id": int(item["camera_id"]),
                "frame_id": int(item["frame_id"]),
                "source": item["source"],
                "variant": variant,
                "preprocess_ms": float(item["preprocess_ms"]),
                "sub_read_ms": float(item.get("sub_read_ms", 0.0)),
                "sub_resize_ms": float(item.get("sub_resize_ms", 0.0)),
                "sub_norm_ms": float(item.get("sub_norm_ms", 0.0)),
                "sub_transpose_ms": float(item.get("sub_transpose_ms", 0.0)),
                "sub_cast_ms": float(item.get("sub_cast_ms", 0.0)),
                "queue_pre_to_infer_ms": float(item["queue_pre_to_infer_ms"]),
                "inference_ms": float(item["inference_ms"]),
                "decode_ms": float(item["decode_ms"]),
                "queue_infer_to_post_ms": float(t_queue[-1]),
                "post_ms": post_ms,
                "e2e_ms": e2e_ms,
                "num_poses": int(len(out.pose_entries)) if out.pose_entries is not None else 0,
            }
            for k, v in timings.items():
                row[f"t_{k}"] = float(v)

            result_q.put(row)
            processed += 1

        timing_avgs = {k: _mean(vs) for k, vs in timing_buckets.items()}
        stats_q.put({
            "stage": "postprocess",
            "worker_id": worker_id,
            "variant": variant,
            "registry_variant": registry_variant,
            "processed": processed,
            "avg_queue_ms": _mean(t_queue),
            "avg_post_ms": _mean(t_post),
            "p95_post_ms": _pctile(t_post, 95),
            "avg_e2e_ms": _mean(t_e2e),
            "p95_e2e_ms": _pctile(t_e2e, 95),
            "wall_s": time.perf_counter() - t_worker_start,
            "timing_breakdown": timing_avgs,
            "pid": os.getpid(),
            "cpu_affinity": _read_pid_cpu_affinity(os.getpid()),
            "affinity_error": affinity_error,
        })
        print(f"[POST-PROF:{worker_id}] Gotovo. processed={processed}", flush=True)

    except Exception:
        error_q.put({"stage": "postprocess", "worker_id": worker_id,
                     "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# 7.  Glavni profiling runner
# ---------------------------------------------------------------------------

DEFAULT_VIDEOS = [
    "cctv_1280x720_24fps_1.mp4",
    "cctv_1280x720_24fps_original.mp4",
    "cctv_1280x720_24fps_3.mp4",
    "cctv_1280x720_24fps_2.mp4",
]


def _camera_sources(n: int, videos: List[str]) -> List[str]:
    return [videos[i % len(videos)] for i in range(n)]


def run_profiled(args) -> None:
    ctx = mp.get_context("spawn")

    videos = args.videos or DEFAULT_VIDEOS
    for v in videos[:args.num_cameras]:
        if not os.path.exists(v):
            print(f"[UPOZORENJE] Video nije pronađen: {v}")

    sources = _camera_sources(args.num_cameras, videos)
    variant = getattr(args, "variant", "cpu")

    print("\n" + "=" * 70)
    print("PIPELINE PROFILING - AMD ROCm")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Variant:        {variant}")
    print(f"  Kamere:         {args.num_cameras}")
    print(f"  Frames/kamera:  {args.frames}")
    print(f"  CPU pinovanje:  {args.pin_cpus}")
    print(f"  ROCTx markeri:  {args.roctx}")
    print(f"  Veličina queue: preprocess={args.pre_q_size} postprocess={args.post_q_size}")

    # Ograniči implicitno threadovanje u child procesima prije spawn-a.
    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_name, "1")

    # Queues
    pre_q = ctx.Queue(maxsize=args.pre_q_size)
    post_q = ctx.Queue(maxsize=args.post_q_size)
    result_q = ctx.Queue()
    stats_q = ctx.Queue()
    error_q = ctx.Queue()
    stop_event = ctx.Event()

    # SysMonitor (u main procesu)
    monitor = SysMonitor(poll_interval_s=0.1)
    monitor.start()

    # Pokretanje camera workers
    camera_procs = []
    for cam_id, src in enumerate(sources):
        pin = cam_id if args.pin_cpus else None
        p = ctx.Process(
            target=camera_preprocess_worker_profiled,
            kwargs=dict(
                camera_id=cam_id, video_path=src,
                out_q=pre_q, stats_q=stats_q, error_q=error_q,
                stop_event=stop_event,
                target_w=args.target_width, target_h=args.target_height,
                max_frames=args.frames, pin_cpu=pin,
            ),
            name=f"cam_{cam_id}",
        )
        p.start()
        camera_procs.append(p)

    # Pokretanje inference workers
    infer_procs = []
    for wid in range(args.infer_workers):
        pin = 10 + wid if args.pin_cpus else None
        p = ctx.Process(
            target=inference_worker_profiled,
            kwargs=dict(
                worker_id=wid, model_path=args.model,
                in_q=pre_q, out_q=post_q,
                stats_q=stats_q, error_q=error_q,
                target_w=args.target_width, target_h=args.target_height,
                stride=args.stride, shared_dtype="float32",
                pin_cpu=pin, use_roctx=args.roctx,
            ),
            name=f"infer_{wid}",
        )
        p.start()
        infer_procs.append(p)

    # Pokretanje postprocess workers
    post_procs = []
    for wid in range(args.post_workers):
        pin = list(range(12, 16)) if args.pin_cpus else None
        p = ctx.Process(
            target=postprocess_worker_profiled,
            kwargs=dict(
                worker_id=wid, variant=variant,
                in_q=post_q, result_q=result_q,
                stats_q=stats_q, error_q=error_q,
                torch_device=args.torch_device,
                max_keypoints=args.max_keypoints,
                threshold=args.threshold,
                nms_radius_fullres=args.nms_radius_fullres,
                nms_radius_lowres=args.nms_radius_lowres,
                nms_impl=args.nms_impl,
                gpu_compute_dtype=args.gpu_compute_dtype,
                require_gpu=args.require_gpu,
                pin_cpu=pin, use_roctx=args.roctx,
            ),
            name=f"post_{wid}",
        )
        p.start()
        post_procs.append(p)

    # Registruj PIDs u monitor
    time.sleep(0.3)  # sačekaj da se procesi pokrenu
    camera_pids = [p.pid for p in camera_procs if p.pid]
    infer_pids = [p.pid for p in infer_procs if p.pid]
    post_pids = [p.pid for p in post_procs if p.pid]

    # Parent-side pinning je dodatni guard: ako child pinning ne uspije ili se desi prerano,
    # ovdje hvatamo stvarne PID-ove i odmah ih ponovo ograničavamo.
    if args.pin_cpus:
        CpuAffinityManager.pin_camera_workers(camera_pids)
        CpuAffinityManager.pin_inference_workers(infer_pids)
        CpuAffinityManager.pin_postprocess_workers(post_pids)

    monitor.register_pids("camera", camera_pids)
    monitor.register_pids("inference", infer_pids)
    monitor.register_pids("postprocess", post_pids)

    # Opciono: prikaži affinity
    if args.pin_cpus or args.report_affinity:
        CpuAffinityManager.report_current_affinity({
            "camera": camera_pids,
            "inference": infer_pids,
            "postprocess": post_pids,
        })

    # Warmup period (ne skupljaj uzorke)
    print(f"\n[PROFILER] Warmup {args.warmup_s}s ...", flush=True)
    time.sleep(args.warmup_s)
    monitor.resume()  # počni skupljati uzorke
    print("[PROFILER] Skupljam uzorke ...", flush=True)

    rows: List[Dict[str, Any]] = []
    stage_stats_list: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    sent_infer_stop = False
    sent_post_stop = False
    aborted = False
    last_progress_s = -1  # guard: štampa progres jednom po sekundi, ne 20×
    # Hard timeout: frames × ~50ms/frame + 30s buffer, minimum 60s
    hard_timeout_s = max(60.0, args.frames * args.num_cameras * 0.05 + 30.0)

    all_procs = camera_procs + infer_procs + post_procs

    try:
        while True:
            elapsed = time.perf_counter() - t0

            # --- Hard timeout ---
            if elapsed > hard_timeout_s:
                print(f"\n[PROFILER] Hard timeout ({hard_timeout_s:.0f}s) — zaustavljam.", flush=True)
                stop_event.set()
                aborted = True
                break

            # --- Watchdog: otkrij workera koji je pao sa greškom (exitcode != 0) ---
            if not aborted:
                for p in all_procs:
                    if not p.is_alive() and p.exitcode not in (None, 0):
                        print(f"\n[PROFILER] Worker '{p.name}' pao (exitcode={p.exitcode}). "
                              f"Zaustavljam sve.", flush=True)
                        stop_event.set()
                        aborted = True
                        for _ in infer_procs:
                            try: pre_q.put_nowait(None)
                            except Exception: pass
                        for _ in post_procs:
                            try: post_q.put_nowait(None)
                            except Exception: pass
                        break

            if aborted:
                # Daj workerima 3s da se ugase normalno, pa ih ubij
                deadline = time.perf_counter() + 3.0
                while time.perf_counter() < deadline:
                    if all(not p.is_alive() for p in all_procs):
                        break
                    time.sleep(0.05)
                break

            # --- Drain errors ---
            while not error_q.empty():
                err = error_q.get()
                print(f"\n[GREŠKA] Worker {err.get('stage')} {err.get('worker_id','')}: "
                      f"{err.get('traceback','')}", flush=True)

            # --- Drain results ---
            while True:
                try:
                    rows.append(result_q.get_nowait())
                except py_queue.Empty:
                    break

            # --- Shutdown sekvenca ---
            if not sent_infer_stop and all(not p.is_alive() for p in camera_procs):
                for _ in infer_procs:
                    pre_q.put(None)
                sent_infer_stop = True

            if sent_infer_stop and not sent_post_stop and all(not p.is_alive() for p in infer_procs):
                for _ in post_procs:
                    post_q.put(None)
                sent_post_stop = True

            if sent_post_stop and all(not p.is_alive() for p in post_procs):
                break

            # Progres jednom po sekundi (ne 20× po sekundi)
            cur_s = int(elapsed)
            if cur_s > 0 and cur_s != last_progress_s and cur_s % 5 == 0:
                last_progress_s = cur_s
                alive = sum(1 for p in all_procs if p.is_alive())
                print(f"[PROFILER] {cur_s}s  rows={len(rows)}  workers_alive={alive}/{len(all_procs)}",
                      flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        stop_event.set()
        print("\n[PROFILER] Prekinuto Ctrl+C.", flush=True)
    finally:
        stop_event.set()
        for p in all_procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=3.0)
            # Force-kill ako još živi
            if p.is_alive():
                p.kill()
                p.join(timeout=1.0)

    wall_s = time.perf_counter() - t0

    while True:
        try:
            rows.append(result_q.get_nowait())
        except py_queue.Empty:
            break
    while True:
        try:
            stage_stats_list.append(stats_q.get_nowait())
        except py_queue.Empty:
            break
    for p in camera_procs + infer_procs + post_procs:
        p.join(timeout=1.0)

    sys_stats = monitor.stop()
    print_profile_report(rows, stage_stats_list, sys_stats, wall_s, args)


# ---------------------------------------------------------------------------
# 8.  Reportovanje
# ---------------------------------------------------------------------------

def _sparkline(values: List[float], width: int = 40) -> str:
    if not values:
        return " " * width
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return blocks[3] * width
    step = max(1, len(values) // width)
    sampled = values[::step][:width]
    return "".join(blocks[min(7, int((v - lo) / (hi - lo + 1e-9) * 8))] for v in sampled)


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _bottleneck(rows: List[Dict]) -> str:
    if not rows:
        return "N/A"
    stages = {
        "cap.read": "sub_read_ms",
        "cv2.resize": "sub_resize_ms",
        "normalize": "sub_norm_ms",
        "dtype_cast": "sub_cast_ms",
        "inference (GPU)": "inference_ms",
        "decode_outputs": "decode_ms",
        "postprocess": "post_ms",
        "queue pre→infer": "queue_pre_to_infer_ms",
        "queue infer→post": "queue_infer_to_post_ms",
    }
    avgs = {name: _mean([r.get(key, 0.0) for r in rows]) for name, key in stages.items()}
    worst = max(avgs, key=avgs.get)
    return f"{worst}  ({avgs[worst]:.2f} ms avg)"


def print_profile_report(
    rows: List[Dict],
    stage_stats: List[Dict],
    sys_stats: Dict,
    wall_s: float,
    args: Any,
) -> None:
    total = len(rows)
    agg_fps = total / wall_s if wall_s > 0 else 0.0

    print("\n" + "═" * 80)
    print("  PIPELINE PROFILING REPORT")
    print("═" * 80)
    print(f"  Ukupno frejmova:     {total}")
    print(f"  Wall time:           {wall_s:.2f} s")
    print(f"  Aggregate FPS:       {agg_fps:.2f}")
    print(f"  FPS/kamera:          {agg_fps / args.num_cameras:.2f}")

    if not rows:
        print("\n  ⚠ Nema rezultata. Provjeri model path i varijant.")
        return

    # --- Stage timing tabela ---
    print("\n" + "─" * 80)
    print("  1. TIMING PO STAGE-U  (sub-stage razlučivost)")
    print("─" * 80)
    stages_def = [
        ("cap.read  (I/O decode)",  "sub_read_ms"),
        ("cv2.resize",              "sub_resize_ms"),
        ("normalize (NumPy)",       "sub_norm_ms"),
        ("transpose/contiguous",    "sub_transpose_ms"),
        ("queue pre→infer  (čekanje)", "queue_pre_to_infer_ms"),
        ("dtype cast (CPU)",        "sub_cast_ms"),
        ("model.run (GPU)",         "inference_ms"),
        ("decode outputs (CPU)",    "decode_ms"),
        ("queue infer→post (čekanje)", "queue_infer_to_post_ms"),
        ("postprocess (NMS+PAF)",   "post_ms"),
        ("E2E latencija",           "e2e_ms"),
    ]
    print(f"  {'Stage':<32} {'avg':>8} {'p50':>8} {'p95':>8}  bar (avg/100ms)")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8}  {'─'*22}")
    for name, key in stages_def:
        vals = [r[key] for r in rows if key in r]
        avg = _mean(vals)
        p50 = _pctile(vals, 50)
        p95 = _pctile(vals, 95)
        bar = _bar(min(avg, 100.0), width=22)
        marker = " ◄ USKO GRLO" if avg == _mean([_mean([r[k] for r in rows if k in r])
                                                  for _, k in stages_def[:-2]]) else ""
        print(f"  {name:<32} {avg:>7.2f}ms {p50:>7.2f}ms {p95:>7.2f}ms  {bar}{marker}")

    worst = _bottleneck(rows)
    print(f"\n  ▶ Najsporiji stage:  {worst}")

    # --- CPU utilization ---
    print("\n" + "─" * 80)
    print("  2. CPU UTILIZACIJA PO JEZGRU  (tokom profiling perioda)")
    print("─" * 80)
    core_avgs = sys_stats.get("core_avg_pct", {})
    ncores_show = 16  # prikaži prvih 16 (fizička jezgra)
    print(f"  {'Jezgro':<8} {'Avg%':>6}  bar")
    for i in range(min(ncores_show, 32)):
        key = f"cpu{i}"
        pct = core_avgs.get(key, 0.0)
        bar = _bar(pct, width=30)
        print(f"  cpu{i:02d}    {pct:>5.1f}%  {bar}")

    if len(core_avgs) > ncores_show:
        print(f"  ... (HT parovi cpu{ncores_show}-cpu31 skriveni, uglavnom ≈ isti nivo)")

    # --- Per-PID CPU ---
    print("\n" + "─" * 80)
    print("  3. CPU UTILIZACIJA PO PROCESU")
    print("─" * 80)
    pid_stats = sys_stats.get("pid_stats", {})
    if pid_stats:
        print(
            f"  {'PID':>8} {'Grupa':<18} {'CPU%':>7} "
            f"{'RSS avg':>9} {'RSS peak':>9} {'VMS avg':>9} {'HWM':>9}  {'Affinity'}"
        )
        print(f"  {'─'*8} {'─'*18} {'─'*7} {'─'*9} {'─'*9} {'─'*9} {'─'*9}  {'─'*30}")
        for pid, info in sorted(pid_stats.items(), key=lambda x: x[1]["group"]):
            aff = ",".join(map(str, info["affinity"])) if info["affinity"] else "N/A"
            print(
                f"  {pid:>8} {info['group']:<18} {info['cpu_pct']:>6.1f}% "
                f"{info.get('rss_avg_mb', 0.0):>8.1f}M "
                f"{info.get('rss_peak_mb', 0.0):>8.1f}M "
                f"{info.get('vms_avg_mb', 0.0):>8.1f}M "
                f"{info.get('hwm_peak_mb', 0.0):>8.1f}M  [{aff}]"
            )

        print("\n  Memory breakdown po procesu (avg):")
        print(f"  {'PID':>8} {'Grupa':<18} {'Anon':>9} {'File':>9} {'Shmem':>9} {'Samples':>8}")
        print(f"  {'─'*8} {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*8}")
        for pid, info in sorted(pid_stats.items(), key=lambda x: x[1]["group"]):
            print(
                f"  {pid:>8} {info['group']:<18} "
                f"{info.get('anon_avg_mb', 0.0):>8.1f}M "
                f"{info.get('file_avg_mb', 0.0):>8.1f}M "
                f"{info.get('shmem_avg_mb', 0.0):>8.1f}M "
                f"{info.get('mem_samples', 0):>8}"
            )
    else:
        print("  (nema per-PID podataka - kratak run)")

    # Stage stats iz workers
    cam_stats = [s for s in stage_stats if s.get("stage") == "camera_preprocess"]
    infer_stats = [s for s in stage_stats if s.get("stage") == "inference"]
    post_stats_list = [s for s in stage_stats if s.get("stage") == "postprocess"]

    if cam_stats:
        print("\n  Camera workers (iz stats_q):")
        for s in cam_stats:
            aff = ",".join(map(str, s.get("cpu_affinity", []))) or "N/A"
            print(f"    cam{s.get('camera_id','?'):02d} PID={s.get('pid','?')} "
                  f"aff=[{aff}] "
                  f"read={s.get('avg_read_ms',0):.2f}ms "
                  f"resize={s.get('avg_resize_ms',0):.2f}ms "
                  f"norm={s.get('avg_norm_ms',0):.2f}ms "
                  f"fps_in={s.get('fps_input',0):.1f}")

    if infer_stats:
        print("\n  Inference workers (iz stats_q):")
        for s in infer_stats:
            aff = ",".join(map(str, s.get("cpu_affinity", []))) or "N/A"
            print(f"    infer{s.get('worker_id','?')} PID={s.get('pid','?')} "
                  f"aff=[{aff}] "
                  f"cast={s.get('avg_cast_ms',0):.2f}ms "
                  f"gpu_run={s.get('avg_inference_ms',0):.2f}ms "
                  f"decode={s.get('avg_decode_ms',0):.2f}ms "
                  f"queue={s.get('avg_queue_pre_to_infer_ms',0):.2f}ms")

    if post_stats_list:
        print("\n  Postprocess workers (iz stats_q):")
        for s in post_stats_list:
            aff = ",".join(map(str, s.get("cpu_affinity", []))) or "N/A"
            aff_note = f" affinity_error={s.get('affinity_error')}" if s.get("affinity_error") else ""
            print(f"    post{s.get('worker_id','?')} PID={s.get('pid','?')} "
                  f"aff=[{aff}] "
                  f"post={s.get('avg_post_ms',0):.2f}ms "
                  f"e2e={s.get('avg_e2e_ms',0):.2f}ms"
                  f"{aff_note}")
            tb = s.get("timing_breakdown", {})
            if tb:
                count_keys = {"group_pairs_total", "group_valid_limbs", "group_connections"}
                parts = []
                for k, v in sorted(tb.items()):
                    if k in count_keys:
                        parts.append(f"{k}={v:.0f}")
                    else:
                        parts.append(f"{k}={v:.2f}ms")
                print("      → Timing breakdown: " + "  ".join(parts))

    # --- GPU utilization ---
    print("\n" + "─" * 80)
    print("  4. GPU UTILIZACIJA (AMD ROCm)")
    print("─" * 80)
    gpu_avg = sys_stats.get("gpu_avg_pct", -1.0)
    gpu_peak = sys_stats.get("gpu_peak_pct", -1.0)
    gpu_idle = sys_stats.get("gpu_idle_pct", -1.0)
    vram_avg = sys_stats.get("vram_avg_mb", -1.0)
    n_samples = sys_stats.get("gpu_samples", 0)

    if n_samples == 0 or gpu_avg < 0:
        print("  ⚠ GPU sysfs nije dostupan ili nema uzoraka.")
        print("    Provjeri: cat /sys/class/drm/card1/device/gpu_busy_percent")
    else:
        bar_gpu = _bar(gpu_avg, width=30)
        print(f"  Avg GPU%:   {gpu_avg:>6.1f}%  {bar_gpu}")
        print(f"  Peak GPU%:  {gpu_peak:>6.1f}%")
        print(f"  GPU idle:   {gpu_idle:>6.1f}%  (uzorci < 5%)")
        if vram_avg > 0:
            print(f"  VRAM avg:   {vram_avg:>6.1f} MB")
        print(f"  Uzoraka:    {n_samples} (~{n_samples*0.1:.0f}s praćenja)")
        print(f"  Sparkline:  {_sparkline(sys_stats.get('_gpu_raw', [gpu_avg]*n_samples))}")

        if gpu_avg < 30.0:
            print("\n  ⚠ GPU je nisko iskorišćen (<30%).")
            print("    → Pipeline je vjerovatno CPU-bound (preprocessing ili postprocessing).")
            print("    → Povećaj broj camera/inference workers ili optimizuj preprocessing.")
        elif gpu_avg > 85.0:
            print("\n  ✓ GPU je visoko iskorišćen (>85%) - dobro!")
        else:
            print(f"\n  → GPU iskorištenost {gpu_avg:.0f}% - umjerena. Pogledaj queue čekanje.")

    # --- Preporuke ---
    print("\n" + "─" * 80)
    print("  5. PREPORUKE ZA OPTIMIZACIJU RESURSA")
    print("─" * 80)

    avg_pre = _mean([r.get("preprocess_ms", 0) for r in rows])
    avg_infer = _mean([r.get("inference_ms", 0) for r in rows])
    avg_post = _mean([r.get("post_ms", 0) for r in rows])
    avg_q_pre = _mean([r.get("queue_pre_to_infer_ms", 0) for r in rows])
    avg_q_post = _mean([r.get("queue_infer_to_post_ms", 0) for r in rows])

    recs: List[str] = []

    if avg_post > avg_infer * 1.5:
        recs.append(
            f"[A] POSTPROCESS je usko grlo ({avg_post:.1f}ms > inference {avg_infer:.1f}ms).\n"
            f"     → Dodaj više post workers: --post-workers 2 ili 3\n"
            f"     → Ili prebaci na brži varijant (cpu_k20_fast / migraphx_nms)"
        )

    if avg_q_pre > 20.0:
        recs.append(
            f"[B] Queue pre→infer čekanje je visoko ({avg_q_pre:.1f}ms).\n"
            f"     → Preprocess radnici ne mogu nahraniti inference fast enough.\n"
            f"     → Provjeri CPU affinity kamere - možda dele cores s post workers.\n"
            f"     → Povećaj --preprocess-queue-size ili dodaj više camera cores."
        )

    if avg_q_post > 20.0:
        recs.append(
            f"[C] Queue infer→post čekanje je visoko ({avg_q_post:.1f}ms).\n"
            f"     → Post workers ne stižu za inferencom.\n"
            f"     → Povećaj --post-workers."
        )

    if gpu_avg >= 0 and gpu_avg < 40.0 and avg_infer > 0:
        recs.append(
            f"[D] GPU iskorištenost je niska ({gpu_avg:.0f}%) ali inference traje {avg_infer:.1f}ms.\n"
            f"     → GPU provodi dosta vremena čekajući CPU (H2D copy, CPU stall).\n"
            f"     → Razmotr pinned memory za input tensor (vidi HIP test ispod).\n"
            f"     → Ili async model.run() sa migraphx async API (ako dostupno)."
        )

    if not args.pin_cpus:
        recs.append(
            "[E] CPU affinity NIJE postavljen.\n"
            "     → Pokreni sa --pin-cpus da pinirate kamere (cpu0-9), inference (cpu10),\n"
            "       postprocess (cpu12-15) na različita fizička jezgra.\n"
            "     → Izbjegavate context switch overhead između grupa."
        )
    else:
        recs.append(
            "[E] CPU affinity JE postavljen (--pin-cpus).\n"
            "     → Kamere: cpu0-9, inference: cpu10, postprocess: cpu12-15\n"
            "     → Provjeri tabelu 'CPU utilization po jezgru' gore."
        )

    avg_resize = _mean([r.get("sub_resize_ms", 0) for r in rows])
    if avg_resize > 5.0:
        recs.append(
            f"[F] cv2.resize je spor ({avg_resize:.1f}ms).\n"
            f"     → Koristi INTER_LINEAR (default) umjesto INTER_AREA za manje tačan ali brži resize.\n"
            f"     → Ili preprocess na GPU sa HIP/ROCm kernel-om."
        )

    for i, rec in enumerate(recs):
        print(f"\n  {rec}")

    if not recs:
        print("\n  ✓ Pipeline izgleda dobro balansirano. Nema kritičnih preporuka.")

    # --- ROCTx / rocprofv3 instrukcije ---
    print("\n" + "─" * 80)
    print("  6. ROCTx / rocprofv3 INSTRUKCIJE")
    print("─" * 80)
    print("""
  ROCTx markeri su aktivni u inference_worker_profiled (--roctx flag).
  Za GPU kernel timeline vizualizaciju:

  # Snimi HIP trace (kernel-level GPU timeline):
  rocprofv3 --hip-trace --hsa-trace --kernel-trace \\
      -o output/rocprof_out \\
      -- python tools/profile_pipeline.py \\
         --model models/pose_model1_fp16_ref1.mxr \\
         --frames 20 --num-cameras 5

  # Pregledaj rezultate:
  ls output/rocprof_out/
  # CSV fajlovi: hip_api_trace.csv, kernel_trace.csv, hsa_api_trace.csv

  # Za system-wide multi-process profiling (Omnitrace):
  rocprof-sys-sample --sampling-freq 200 --pid <pid_inference_worker> \\
      -- python tools/profile_pipeline.py --frames 30

  # Za hardware performance counters (GPU occupancy, memory bandwidth):
  rocprofv3-avail --list-counters  # pogledaj dostupne brojače
  rocprofv3 --pmc SQ_WAVES,FETCH_SIZE,WRITE_SIZE \\
      -- python tools/profile_pipeline.py --frames 20

  # rocm-smi monitoring tokom runa (u drugom terminalu):
  watch -n 0.5 rocm-smi --showuse --showmemuse
""")

    # --- HIP pinned memory instrukcije ---
    print("─" * 80)
    print("  7. HIP PINNED MEMORY (za inference input tensor)")
    print("─" * 80)
    print("""
  Pokretanje samo HIP benchmark-a:
    python tools/profile_pipeline.py --hip-pin-bench-only

  Kako koristiti pinned memory u inference worker-u:
    import ctypes
    hip = ctypes.CDLL("libamdhip64.so")
    pin_ptr = ctypes.c_void_p()
    hip.hipHostMalloc(ctypes.byref(pin_ptr), tensor_size_bytes, 0)
    # kopiranje numpy podataka u pin_ptr bafer
    # prosljeđivanje pin_ptr direktno migraphx.run() umjesto numpy array-a
    # (zahtijeva migraphx Python API koji prima ctypes pointer)

  Napomena: MIGraphX Python API prima numpy array - automatski radi H2D.
  Pinned memory pomaže samo ako imate direktan pristup HIP baferima.
  Alternativa: koristiti migraphx async API (rocprofv3 profil pomoći identifikovati gdje H2D traje).
""")

    print("═" * 80)
    print("  Kraj profiling izvještaja")
    print("═" * 80)


# ---------------------------------------------------------------------------
# 9.  argparse + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detaljno profilisanje AMD ROCm pipeline-a za 10-kamera simulaciju.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="models/pose_model1_fp16_ref1.mxr",
                        help="Path do .mxr modela")
    parser.add_argument("--variant", default="cpu",
                        help="Postprocess varijant: cpu, standard, optimized_batch_k20_fast, ...")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--num-cameras", type=int, default=10)
    parser.add_argument("--frames", type=int, default=60,
                        help="Frames po kameri za profiling run")
    parser.add_argument("--infer-workers", type=int, default=1)
    parser.add_argument("--post-workers", type=int, default=1)
    parser.add_argument("--target-width", type=int, default=968)
    parser.add_argument("--target-height", type=int, default=544)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--torch-device", default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-keypoints", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--nms-radius-fullres", type=int, default=6,
                        help="Full-res GPU NMS radius for gpu_nms_* variants")
    parser.add_argument("--nms-radius-lowres", type=int, default=1,
                        help="Low-res GPU NMS radius for gpu_nms_* variants")
    parser.add_argument("--nms-impl", default="2d", choices=["2d", "separable"],
                        help="Torch GPU NMS implementation")
    parser.add_argument("--gpu-compute-dtype", default="float32", choices=["float32", "float16"],
                        help="Torch GPU NMS compute dtype")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Fail if a gpu_nms variant cannot run on ROCm/CUDA")
    parser.add_argument("--pre-q-size", type=int, default=20)
    parser.add_argument("--post-q-size", type=int, default=20)
    parser.add_argument("--warmup-s", type=float, default=2.0,
                        help="Warmup period u sekundama (ne skupljaj uzorke)")

    parser.add_argument("--pin-cpus", action="store_true",
                        help="Pini procese na dedicirane CPU cores (kamera→0-9, infer→10, post→12-15)")
    parser.add_argument("--report-affinity", action="store_true",
                        help="Ispiši CPU affinity za svaki process")
    parser.add_argument("--roctx", action="store_true",
                        help="Aktiviraj ROCTx markere za rocprofv3 timeline")
    parser.add_argument("--hip-pin-bench-only", action="store_true",
                        help="Pokreni samo HIP pinned memory benchmark (bez pipeline-a)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Samo HIP benchmark
    if args.hip_pin_bench_only:
        print("=== HIP Pinned Memory Benchmark ===")
        bench = HipPinnedMemBench()
        result = bench.run(
            tensor_shape=(1, 3, args.target_height, args.target_width),
            dtype_bytes=2,  # fp16 kao što koristi model
            repeats=100,
        )
        if not result.get("available"):
            print(f"HIP nije dostupan: {result.get('reason', 'nepoznato')}")
        elif "error" in result:
            print(f"Greška: {result['error']}")
        else:
            print(f"\n  Tensor shape: (1, 3, {args.target_height}, {args.target_width})")
            print(f"  Size:         {result['size_bytes']/1024:.1f} KB")
            print(f"\n  Pageable memory (regular malloc):")
            print(f"    avg = {result['pageable_avg_ms']:.3f} ms")
            print(f"    p95 = {result['pageable_p95_ms']:.3f} ms")
            print(f"    min = {result['pageable_min_ms']:.3f} ms")
            print(f"\n  Pinned memory (hipHostMalloc):")
            print(f"    avg = {result['pinned_avg_ms']:.3f} ms")
            print(f"    p95 = {result['pinned_p95_ms']:.3f} ms")
            print(f"    min = {result['pinned_min_ms']:.3f} ms")
            speedup = result["speedup_pct"]
            icon = "✓" if speedup > 5.0 else "~"
            print(f"\n  {icon} Ubrzanje pinned vs pageable: {speedup:.1f}%")
            if speedup > 10.0:
                print("  → Pinned memory daje značajno ubrzanje!")
                print("    Korisno implementovati za inference input buffer.")
            elif speedup > 3.0:
                print("  → Malo ubrzanje - vrijedi za visoko-frekventne streame.")
            else:
                print("  → Minimalna razlika - pageable memory je dovoljno za ovaj tensor size.")
        return

    run_profiled(args)


if __name__ == "__main__":
    main()
