"""Optional worker profiling, ROCTx tracing, and trace-print helpers."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any


def maybe_profile_worker(stage: str, worker_id: Any, fn, *args, **kwargs):
    profile_dir = os.environ.get("STREAM_CPROFILE_DIR", "")
    if not profile_dir:
        return fn(*args, **kwargs)

    import cProfile
    import pstats

    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    out_prof = Path(profile_dir) / f"{stage}_{worker_id}_pid{pid}.prof"
    out_txt = Path(profile_dir) / f"{stage}_{worker_id}_pid{pid}.txt"

    profiler = cProfile.Profile()
    try:
        return profiler.runcall(fn, *args, **kwargs)
    finally:
        profiler.dump_stats(str(out_prof))
        with open(out_txt, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs().sort_stats("cumtime").print_stats(80)

class RocTxTracer:
    """Tiny optional ROCTx wrapper; no-op when roctx is unavailable or disabled."""

    def __init__(self, enabled: bool, prefix: str = ""):
        self.enabled = bool(enabled)
        self.prefix = prefix
        self._roctx = None
        self._roctx_push = None
        self._roctx_pop = None
        if self.enabled:
            try:
                import roctx # pyright: ignore[reportMissingImports]
                self._roctx = roctx
                self._roctx_push = getattr(roctx, "push", None) or getattr(roctx, "rangePush", None)
                self._roctx_pop = getattr(roctx, "pop", None) or getattr(roctx, "rangePop", None)
                if self._roctx_push is None or self._roctx_pop is None:
                    self.enabled = False
            except Exception:
                self.enabled = False

    def label(self, name: str) -> str:
        return f"{self.prefix}:{name}" if self.prefix else name

    @contextlib.contextmanager
    def range(self, name: str):
        if self.enabled and self._roctx_push is not None and self._roctx_pop is not None:
            self._roctx_push(self.label(name))
            try:
                yield
            finally:
                self._roctx_pop()
        else:
            yield

    def mark(self, name: str) -> None:
        if self.enabled and self._roctx is not None:
            try:
                self._roctx.mark(self.label(name))
            except Exception:
                pass


def trace_print(every: int, count: int, msg: str) -> None:
    if every > 0 and count > 0 and count % every == 0:
        print(msg, flush=True)


def allow_ptrace_attach_if_requested() -> None:
    if os.environ.get("STREAM_ALLOW_PTRACE_ATTACH") != "1":
        return
    try:
        import ctypes
        PR_SET_PTRACER = 0x59616D61
        PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
        if rc != 0:
            err = ctypes.get_errno()
            print(f"[TRACE pid={os.getpid()}] prctl(PR_SET_PTRACER_ANY) failed errno={err}", flush=True)
    except Exception as exc:
        print(f"[TRACE pid={os.getpid()}] ptrace attach opt-in failed: {exc}", flush=True)
