#!/usr/bin/env python3
from __future__ import annotations

import accuracy_validation_core as core


def _patched_check(variants):
    rest = [v for v in variants if v != core.SPLIT_HIP_SMART_MODE]
    bad = [v for v in rest if v.startswith("gpu_") and not core.is_two_process_mode(v))]
    if bad:
        raise RuntimeError(f"Single-process Torch GPU postprocess variants are not supported here: {bad}")
    if any(core.is_two_process_mode(v) for v in rest):
        raise RuntimeError("Two-process postprocess variants are not supported by this validation entry point.")


core._assert_accuracy_variants_are_migrarhx_safe = _patched_check

if __name__ == "__main__":
    core.validate_accuracy(core.parse_args())
