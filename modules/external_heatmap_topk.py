"""External heatmap resize/NMS/TopK backends for split MXR experiments.

These helpers intentionally preserve the same output contract as the heatmap
branch inside the fused-pruned MIGraphX graph:

    heatmaps [B,18,68,121] -> top_scores [B,18,K], top_indices [B,18,K]

The HIP variants are staged experiments.  `hip_host` is the stable dense
correctness baseline, `hip_host_fused` is the rejected dense-fusion experiment,
and `hip_host_smart` is the E4 smart-full-res candidate backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tools.export_batchaware_fused_pruned_postprocess import BatchAwareFusedPrunedPostprocess
except ModuleNotFoundError:  # pragma: no cover - useful when imported from tools/ scripts
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from tools.export_batchaware_fused_pruned_postprocess import BatchAwareFusedPrunedPostprocess


HeatmapBackendName = Literal["torch_manual", "torch_bicubic", "hip_host", "hip_host_fused", "hip_host_smart"]


@dataclass(frozen=True)
class HeatmapTopKConfig:
    batch_size: int
    in_h: int = 68
    in_w: int = 121
    full_h: int = 1080
    full_w: int = 1920
    channels: int = 18
    topk: int = 20
    threshold: float = 0.1
    nms_radius: int = 6
    nms_impl: str = "separable"
    cubic_a: float = -0.75
    smart_proposals: int = 32
    smart_local_radius: int = 4
    smart_lowres_nms_radius: int = 1


def torch_device_summary() -> str:
    cuda_built = False
    try:
        cuda_built = bool(torch.backends.cuda.is_built())
    except Exception:
        cuda_built = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    return (
        f"torch={getattr(torch, '__version__', 'unknown')} "
        f"hip={hip_version} cuda={cuda_version} "
        f"cuda_built={cuda_built} cuda_available={cuda_available}"
    )


def _select_device(device: str | None = None) -> torch.device:
    requested = (device or "").strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested in {"rocm", "hip"}:
        requested = "cuda"
    if requested.startswith("cuda"):
        try:
            cuda_built = bool(torch.backends.cuda.is_built())
        except Exception:
            cuda_built = False
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if not cuda_built or not cuda_available:
            raise RuntimeError(
                "Requested PyTorch GPU device, but this Python environment does not expose a usable "
                "PyTorch CUDA/ROCm backend. On ROCm builds, torch.version.hip should be non-null and "
                f"torch.cuda.is_available() should be True. Diagnostics: {torch_device_summary()}"
            )
    return torch.device(requested)


def _to_numpy_pair(scores: torch.Tensor, indices: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    indices_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.ascontiguousarray(scores_np), np.ascontiguousarray(indices_np)


def _validate_heatmaps(heatmaps: np.ndarray, cfg: HeatmapTopKConfig) -> np.ndarray:
    arr = np.asarray(heatmaps)
    expected = (int(cfg.batch_size), int(cfg.channels), int(cfg.in_h), int(cfg.in_w))
    if tuple(arr.shape) != expected:
        raise ValueError(f"Expected heatmaps shape {expected}, got {tuple(arr.shape)}")
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def torch_manual_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    arr = _validate_heatmaps(heatmaps, cfg)
    dev = _select_device(device)
    module = BatchAwareFusedPrunedPostprocess(
        batch_size=int(cfg.batch_size),
        in_h=int(cfg.in_h),
        in_w=int(cfg.in_w),
        full_h=int(cfg.full_h),
        full_w=int(cfg.full_w),
        topk=int(cfg.topk),
        limb_topm=20,
        threshold=float(cfg.threshold),
        nms_radius=int(cfg.nms_radius),
        nms_impl=str(cfg.nms_impl),
        heatmap_cubic_a=float(cfg.cubic_a),
        points_per_limb=8,
        min_paf_score=0.05,
        success_ratio_thr=0.8,
        paf_cubic_a=float(cfg.cubic_a),
        min_pair_score=0.0,
    ).eval().to(dev)
    x = torch.from_numpy(arr).to(dev, non_blocking=False)
    with torch.no_grad():
        scores, indices = module.topk_heatmaps(x)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    return _to_numpy_pair(scores, indices)


def torch_bicubic_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    arr = _validate_heatmaps(heatmaps, cfg)
    dev = _select_device(device)
    x = torch.from_numpy(arr).to(dev, non_blocking=False)
    with torch.no_grad():
        hm = F.interpolate(x, size=(int(cfg.full_h), int(cfg.full_w)), mode="bicubic", align_corners=False, antialias=False)
        r = int(cfg.nms_radius)
        k = 2 * r + 1
        if cfg.nms_impl == "2d":
            pooled = F.max_pool2d(hm, kernel_size=k, stride=1, padding=r)
        elif cfg.nms_impl == "separable":
            pooled = F.max_pool2d(hm, kernel_size=(k, 1), stride=1, padding=(r, 0))
            pooled = F.max_pool2d(pooled, kernel_size=(1, k), stride=1, padding=(0, r))
        else:
            raise RuntimeError(f"Unsupported nms_impl={cfg.nms_impl}")
        peaks = (hm == pooled) & (hm > float(cfg.threshold))
        masked = torch.where(peaks, hm, torch.full_like(hm, -1.0e9))
        flat = masked.flatten(start_dim=2)
        scores, indices = torch.topk(flat, k=int(cfg.topk), dim=2, largest=True, sorted=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    return _to_numpy_pair(scores, indices)


def _check_hip_cfg(cfg: HeatmapTopKConfig) -> None:
    if cfg.nms_impl not in {"separable", "2d"}:
        raise RuntimeError(f"Unsupported nms_impl={cfg.nms_impl}")
    if abs(float(cfg.cubic_a) - (-0.75)) > 1.0e-6:
        raise RuntimeError("HIP backend currently implements heatmap cubic_a=-0.75 only")


def _hip_shape(cfg: HeatmapTopKConfig):
    from modules.external_heatmap_topk_hip import HipHeatmapTopKShape

    return HipHeatmapTopKShape(
        batch=int(cfg.batch_size),
        channels=int(cfg.channels),
        in_h=int(cfg.in_h),
        in_w=int(cfg.in_w),
        full_h=int(cfg.full_h),
        full_w=int(cfg.full_w),
        topk=int(cfg.topk),
        threshold=float(cfg.threshold),
        nms_radius=int(cfg.nms_radius),
    )


def _hip_smart_shape(cfg: HeatmapTopKConfig):
    from modules.external_heatmap_topk_hip import HipHeatmapTopKSmartShape

    return HipHeatmapTopKSmartShape(
        batch=int(cfg.batch_size),
        channels=int(cfg.channels),
        in_h=int(cfg.in_h),
        in_w=int(cfg.in_w),
        full_h=int(cfg.full_h),
        full_w=int(cfg.full_w),
        topk=int(cfg.topk),
        threshold=float(cfg.threshold),
        lowres_nms_radius=int(cfg.smart_lowres_nms_radius),
        smart_proposals=int(cfg.smart_proposals),
        smart_local_radius=int(cfg.smart_local_radius),
    )


def hip_host_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    del device
    _check_hip_cfg(cfg)
    arr = _validate_heatmaps(heatmaps, cfg)
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend

    return HipHeatmapTopKBackend().run_host(arr, _hip_shape(cfg))


def hip_host_fused_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    del device
    _check_hip_cfg(cfg)
    arr = _validate_heatmaps(heatmaps, cfg)
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend

    return HipHeatmapTopKBackend().run_host_fused(arr, _hip_shape(cfg))


def hip_host_smart_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    del device
    _check_hip_cfg(cfg)
    arr = _validate_heatmaps(heatmaps, cfg)
    from modules.external_heatmap_topk_hip import HipHeatmapTopKBackend

    return HipHeatmapTopKBackend().run_host_smart(arr, _hip_smart_shape(cfg))


def run_external_heatmap_topk(heatmaps: np.ndarray, cfg: HeatmapTopKConfig, *, backend: HeatmapBackendName = "torch_manual", device: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if backend == "torch_manual":
        return torch_manual_heatmap_topk(heatmaps, cfg, device=device)
    if backend == "torch_bicubic":
        return torch_bicubic_heatmap_topk(heatmaps, cfg, device=device)
    if backend == "hip_host":
        return hip_host_heatmap_topk(heatmaps, cfg, device=device)
    if backend == "hip_host_fused":
        return hip_host_fused_heatmap_topk(heatmaps, cfg, device=device)
    if backend == "hip_host_smart":
        return hip_host_smart_heatmap_topk(heatmaps, cfg, device=device)
    raise ValueError(f"Unsupported external heatmap backend: {backend}")
