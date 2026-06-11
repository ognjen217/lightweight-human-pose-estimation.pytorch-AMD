"""Modular multi-camera stream simulation package."""

from __future__ import annotations

from .cli import parse_args
from .defaults import DEFAULT_VIDEO_CYCLE
from .runner import run, run_latest, run_queue


__all__ = ["DEFAULT_VIDEO_CYCLE", "parse_args", "run", "run_queue", "run_latest"]
