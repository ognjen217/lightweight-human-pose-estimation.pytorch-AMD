#!/usr/bin/env python3
"""Runtime wrapper for fused manual TopK + full-res PAF scorer MXR."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class MIGraphXFusedPostprocess:
    """Runs fused postprocess MXR.

    Inputs:
        heatmaps_nchw: [1,18,H,W] float32
        pafs_nchw:     [1,38,H,W] float32

    Returns:
        pair_scores, pair_valid, top_scores, top_indices
    """

    def __init__(self, mxr_path: str | Path):
        import migraphx  # type: ignore

        self.path = str(mxr_path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.program = migraphx.load(self.path)

    def run(self, heatmaps_nchw, pafs_nchw) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        heatmaps_nchw = np.ascontiguousarray(heatmaps_nchw, dtype=np.float32)
        pafs_nchw = np.ascontiguousarray(pafs_nchw, dtype=np.float32)

        result = self.program.run(
            {
                "heatmaps": heatmaps_nchw,
                "pafs": pafs_nchw,
            }
        )
        if not isinstance(result, (list, tuple)):
            result = list(result)

        if len(result) < 4:
            raise RuntimeError(f"Expected 4 outputs from fused postprocess, got {len(result)}")

        pair_scores = np.asarray(result[0], dtype=np.float32)
        pair_valid = np.asarray(result[1], dtype=np.float32)
        top_scores = np.asarray(result[2], dtype=np.float32)
        top_indices = np.asarray(result[3], dtype=np.float32)
        return pair_scores, pair_valid, top_scores, top_indices
