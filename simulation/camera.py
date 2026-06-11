"""Camera source mapping and frame preprocessing."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def camera_sources(num_cameras: int, videos: Sequence[str]) -> List[str]:
    """Return the default 10-camera mapping requested in the prompt.

    For 10 cameras:
      0 -> video 1
      1 -> video 2
      2 -> video 3
      3 -> original
      4 -> video 1
      5 -> video 2
      6 -> video 3
      7 -> original
      8 -> video 3
      9 -> original

    For more than 10 cameras, continue round-robin.
    """
    if len(videos) < 4:
        raise ValueError("At least four video paths are required.")

    out: List[str] = []
    for cam_id in range(num_cameras):
        if cam_id == 8:
            out.append(videos[2])
        elif cam_id == 9:
            out.append(videos[3])
        else:
            out.append(videos[cam_id % len(videos)])
    return out


def preprocess_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    import cv2

    img = cv2.resize(frame, (target_w, target_h))
    img = (img.astype(np.float32) - 128.0) / 256.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(img, dtype=np.float32)
