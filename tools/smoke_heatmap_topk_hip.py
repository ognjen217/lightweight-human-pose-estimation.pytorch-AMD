#!/usr/bin/env python3
"""Smoke-test the native HIP heatmap TopK shared-library ABI.

The current native library is a scaffold and should return
HIP_TOPK_NOT_IMPLEMENTED for valid dummy pointers.  This script verifies that the
library can be loaded and the ABI can be called from Python.
"""

from __future__ import annotations

import argparse

try:
    from modules.external_heatmap_topk_hip import (
        HIP_TOPK_NOT_IMPLEMENTED,
        HipHeatmapTopKBackend,
        HipHeatmapTopKShape,
    )
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from modules.external_heatmap_topk_hip import (
        HIP_TOPK_NOT_IMPLEMENTED,
        HipHeatmapTopKBackend,
        HipHeatmapTopKShape,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test HIP heatmap TopK backend ABI.")
    p.add_argument("--lib", default="", help="Path to libheatmap_topk_hip.so. Empty = default build path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backend = HipHeatmapTopKBackend(args.lib or None)
    shape = HipHeatmapTopKShape()

    # Valid non-null dummy pointers.  The scaffold validates pointers/shapes but
    # does not dereference pointers before returning NOT_IMPLEMENTED.
    status = backend.run_raw(
        heatmaps_ptr=1,
        top_scores_ptr=2,
        top_indices_ptr=3,
        shape=shape,
        hip_stream_ptr=0,
        raise_on_error=False,
    )
    print("library:", backend.path)
    print("status:", status, backend.status_string(status))
    if status != HIP_TOPK_NOT_IMPLEMENTED:
        raise SystemExit(f"Unexpected status from scaffold: {status} {backend.status_string(status)}")
    print("ABI smoke test passed")


if __name__ == "__main__":
    main()
