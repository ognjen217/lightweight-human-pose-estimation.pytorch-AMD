"""Lightweight memory tracking helpers for Python heap and GPU VRAM.

Usage
-----
    from modules.memory_tracker import start_tracing, stop_tracing, mem_checkpoint

    start_tracing()

    with mem_checkpoint("model load"):
        model = migraphx.load(path)

    with mem_checkpoint("inference"):
        results = model.run({"input": inp})

    stop_tracing()
"""

from __future__ import annotations

import json
import subprocess
import tracemalloc
from contextlib import contextmanager
from typing import Generator, Optional

import torch


def start_tracing(depth: int = 25) -> None:
    """Start tracemalloc with the given call-stack depth."""
    tracemalloc.start(depth)


def stop_tracing() -> None:
    tracemalloc.stop()


def _gpu_allocated_bytes() -> int:
    # is_initialized() does NOT trigger CUDA init — safe to call before migraphx.load().
    # is_available() may initialize the ROCm context on some driver versions, which would
    # conflict with MIGraphX claiming the GPU, so we avoid it here.
    if torch.cuda.is_initialized():
        return torch.cuda.memory_allocated()
    return 0


def _rocm_vram_used_mb() -> Optional[float]:
    """Query rocm-smi for total GPU VRAM in use (all processes, all models)."""
    try:
        raw = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(raw)
        for card_data in data.values():
            used = card_data.get("VRAM Total Used Memory (B)")
            if used is not None:
                return int(used) / 1e6
    except Exception:
        return None
    return None


@contextmanager
def mem_checkpoint(
    label: str,
    *,
    enabled: bool = True,
    top_n: int = 8,
    group_by: str = "lineno",
    show_rocm: bool = False,
) -> Generator[None, None, None]:
    """Context manager that prints Python heap and GPU VRAM changes.

    Parameters
    ----------
    label:      Human-readable name printed in the report header.
    enabled:    Set to False to make this a no-op (keeps the call sites in code
                but disables output when not profiling).
    top_n:      How many Python heap allocation sites to show.
    group_by:   tracemalloc grouping: "lineno", "filename", or "traceback".
    show_rocm:  Also query rocm-smi for whole-GPU VRAM delta (slower, ~10 ms).
    """
    if not enabled:
        yield
        return

    tracing = tracemalloc.is_tracing()
    snap_before = tracemalloc.take_snapshot() if tracing else None
    gpu_before = _gpu_allocated_bytes()
    rocm_before = _rocm_vram_used_mb() if show_rocm else None

    yield

    gpu_after = _gpu_allocated_bytes()
    rocm_after = _rocm_vram_used_mb() if show_rocm else None

    _print_report(
        label=label,
        snap_before=snap_before,
        gpu_before=gpu_before,
        gpu_after=gpu_after,
        rocm_before=rocm_before,
        rocm_after=rocm_after,
        top_n=top_n,
        group_by=group_by,
    )


def _print_report(
    label: str,
    snap_before,
    gpu_before: int,
    gpu_after: int,
    rocm_before: Optional[float],
    rocm_after: Optional[float],
    top_n: int,
    group_by: str,
) -> None:
    sep = "─" * 60
    print(f"\n┌{sep}")
    print(f"│ [MEM] {label}")
    print(f"├{sep}")

    # GPU (torch allocator)
    delta_mb = (gpu_after - gpu_before) / 1e6
    total_mb = gpu_after / 1e6
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    sign = "+" if delta_mb >= 0 else ""
    print(f"│  Torch GPU allocated : {total_mb:7.2f} MB  (delta: {sign}{delta_mb:.2f} MB)")
    print(f"│  Torch GPU peak      : {peak_mb:7.2f} MB")

    # rocm-smi (whole GPU, includes MIGraphX model)
    if rocm_before is not None and rocm_after is not None:
        rocm_delta = rocm_after - rocm_before
        sign = "+" if rocm_delta >= 0 else ""
        print(f"│  ROCm VRAM total     : {rocm_after:7.2f} MB  (delta: {sign}{rocm_delta:.2f} MB)")

    # Python heap diff
    if snap_before is not None:
        snap_after = tracemalloc.take_snapshot()
        diffs = snap_after.compare_to(snap_before, group_by)
        nonzero = [d for d in diffs if d.size_diff != 0]
        if nonzero:
            print(f"│  Python heap top {top_n} changes ({group_by}):")
            for d in nonzero[:top_n]:
                sign = "+" if d.size_diff >= 0 else ""
                size_kb = d.size_diff / 1024
                print(f"│    {sign}{size_kb:+8.1f} KB  ({d.count_diff:+5d} obj)  {d.traceback[0]}")
        else:
            print("│  Python heap: no changes detected")
    else:
        print("│  Python heap: tracemalloc not active (call start_tracing() first)")

    print(f"└{sep}")


def print_gpu_summary(header: str = "GPU memory summary") -> None:
    """Print current + peak GPU memory. Call once at end of session."""
    if not torch.cuda.is_initialized():
        print(f"[MEM] {header}: CUDA not initialized (CPU-only session)")
        return
    allocated_mb = torch.cuda.memory_allocated() / 1e6
    reserved_mb = torch.cuda.memory_reserved() / 1e6
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(
        f"\n[MEM] {header}\n"
        f"  allocated : {allocated_mb:.2f} MB\n"
        f"  reserved  : {reserved_mb:.2f} MB\n"
        f"  peak      : {peak_mb:.2f} MB"
    )
