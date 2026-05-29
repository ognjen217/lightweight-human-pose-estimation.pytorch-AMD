#!/usr/bin/env python3
"""Runtime wrapper for fused_postprocess_v2 / pruned pair MXR."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class MIGraphXFusedPostprocessPruned:
    def __init__(self, mxr_path: str | Path):
        import migraphx  # type: ignore
        self.path = str(mxr_path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.program = migraphx.load(self.path)

    def run(self, heatmaps_nchw, pafs_nchw):
        heatmaps_nchw = np.ascontiguousarray(heatmaps_nchw, dtype=np.float32)
        pafs_nchw = np.ascontiguousarray(pafs_nchw, dtype=np.float32)
        result = self.program.run({"heatmaps": heatmaps_nchw, "pafs": pafs_nchw})
        if not isinstance(result, (list, tuple)):
            result = list(result)
        if len(result) < 6:
            raise RuntimeError(f"Expected 6 outputs from pruned fused postprocess, got {len(result)}")
        return (
            np.asarray(result[0], dtype=np.float32),
            np.asarray(result[1]),
            np.asarray(result[2], dtype=np.int64),
            np.asarray(result[3], dtype=np.int64),
            np.asarray(result[4], dtype=np.float32),
            np.asarray(result[5], dtype=np.float32),
        )
