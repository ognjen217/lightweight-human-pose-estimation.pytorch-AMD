#!/usr/bin/env python3
"""Runtime wrapper for compiled MIGraphX full-res cubic PAF pair scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


class MIGraphXPAFFullResPairScorer:
    def __init__(self, mxr_path: str | Path):
        import migraphx  # type: ignore

        self.path = str(mxr_path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.program = migraphx.load(self.path)

    def run(self, top_scores, top_indices, pafs_nchw) -> Tuple[np.ndarray, np.ndarray]:
        top_scores = np.ascontiguousarray(top_scores, dtype=np.float32)
        top_indices = np.ascontiguousarray(top_indices, dtype=np.float32)
        pafs_nchw = np.ascontiguousarray(pafs_nchw, dtype=np.float32)

        result = self.program.run(
            {
                "top_scores": top_scores,
                "top_indices": top_indices,
                "pafs": pafs_nchw,
            }
        )
        if not isinstance(result, (list, tuple)):
            result = list(result)
        pair_scores = np.asarray(result[0], dtype=np.float32)
        pair_valid = np.asarray(result[1], dtype=np.float32)
        return pair_scores, pair_valid
