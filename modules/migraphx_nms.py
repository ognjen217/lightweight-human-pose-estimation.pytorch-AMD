"""Runtime wrapper for a compiled MIGraphX heatmap NMS head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MIGraphXNMSHead:
    """Small wrapper around a compiled heatmap NMS .mxr program.

    The wrapped program expects dense NCHW heatmaps and returns a dense NCHW
    peak mask. Candidate list extraction remains outside MIGraphX.
    """

    def __init__(self, mxr_path: str, input_name: str = "heatmaps") -> None:
        self.mxr_path = str(mxr_path)
        self.input_name = input_name

        if not Path(self.mxr_path).exists():
            raise FileNotFoundError(f"MIGraphX NMS .mxr file not found: {self.mxr_path}")

        import migraphx  # type: ignore

        self._migraphx: Any = migraphx
        self.program = migraphx.load(self.mxr_path)

    def run(self, heatmaps_nchw: np.ndarray) -> np.ndarray:
        heatmaps_nchw = np.ascontiguousarray(heatmaps_nchw, dtype=np.float32)

        if heatmaps_nchw.ndim != 4:
            raise ValueError(
                "MIGraphXNMSHead expects 4D NCHW input, "
                f"got shape {heatmaps_nchw.shape}"
            )
        if heatmaps_nchw.shape[0] != 1:
            raise ValueError(
                "This experimental wrapper expects batch size 1, "
                f"got shape {heatmaps_nchw.shape}"
            )

        result = self.program.run({
            self.input_name: self._migraphx.argument(heatmaps_nchw)
        })

        if not result:
            raise RuntimeError("MIGraphX NMS program returned no outputs")

        peak_mask = np.asarray(result[0], dtype=np.float32)
        peak_mask = peak_mask.reshape(heatmaps_nchw.shape)
        return peak_mask
