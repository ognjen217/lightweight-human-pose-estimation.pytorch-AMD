#!/usr/bin/env python3
"""
Export, compile, and optionally benchmark heatmap TopK ablation heads.

The goal is to decompose the validated accuracy-preserving heatmap branch:

    heatmaps [B,18,68,121]
      -> manual bicubic resize to full resolution
      -> separable/2D MaxPool NMS
      -> full-resolution peak mask
      -> TopK over H*W

This script does not change the production model. It creates diagnostic heads
that keep the same parameters and progressively stop after each stage so the
cost of the full-resolution heatmap branch can be measured in isolation.

Modes:
    resize_only             resize heatmaps to [B,C,full_h,full_w]
    resize_pool             resize + NMS pooling
    resize_pool_mask        resize + NMS pooling + peak masking
    resize_pool_mask_topk   full current heatmap branch ending in TopK

Example:
    python tools/export_heatmap_topk_ablation_heads.py \
      --shape 68 121 1080 1920 \
      --batch-size 4 \
      --channels 18 \
      --topk 20 \
      --threshold 0.1 \
      --nms-radius 6 \
      --nms-impl separable \
      --compile \
      --benchmark \
      --output-dir models/heatmap_topk_ablation_b4 \
      --report-json outputs/heatmap_topk_ablation_b4/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = (
    "resize_only",
    "resize_pool",
    "resize_pool_mask",
    "resize_pool_mask_topk",
)


def safe_float_token(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def cubic_weights_and_indices(in_size: int, out_size: int, a: float = -0.75):
    """Return four cubic interpolation index/weight vectors.

    This mirrors the current manual-cubic exporter so the diagnostic branch uses
    the same sampling convention as the validated model.
    """
    scale = float(in_size) / float(out_size)
    dst = torch.arange(out_size, dtype=torch.float32)
    src = (dst + 0.5) * scale - 0.5
    base = torch.floor(src).to(torch.int64)

    idxs = []
    weights = []
    aa = float(a)
    for off in (-1, 0, 1, 2):
        raw_idx = base + int(off)
        dist = torch.abs(src - raw_idx.to(torch.float32))
        dist2 = dist * dist
        dist3 = dist2 * dist
        w1 = (aa + 2.0) * dist3 - (aa + 3.0) * dist2 + 1.0
        w2 = aa * dist3 - 5.0 * aa * dist2 + 8.0 * aa * dist - 4.0 * aa
        w = torch.where(dist <= 1.0, w1, torch.where(dist < 2.0, w2, torch.zeros_like(dist)))
        idx = torch.clamp(raw_idx, 0, int(in_size) - 1)
        idxs.append(idx.to(torch.int64))
        weights.append(w.to(torch.float32))
    return torch.stack(idxs, dim=0), torch.stack(weights, dim=0)


class HeatmapTopKAblationHead(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        in_h: int,
        in_w: int,
        full_h: int,
        full_w: int,
        channels: int = 18,
        topk: int = 20,
        threshold: float = 0.1,
        nms_radius: int = 6,
        nms_impl: str = "separable",
        cubic_a: float = -0.75,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unsupported mode={mode!r}; expected one of {MODES}")
        if nms_impl not in {"2d", "separable"}:
            raise ValueError("nms_impl must be '2d' or 'separable'")

        self.mode = str(mode)
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        self.full_h = int(full_h)
        self.full_w = int(full_w)
        self.channels = int(channels)
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.nms_radius = int(nms_radius)
        self.nms_impl = str(nms_impl)

        x_idx, x_w = cubic_weights_and_indices(self.in_w, self.full_w, a=float(cubic_a))
        y_idx, y_w = cubic_weights_and_indices(self.in_h, self.full_h, a=float(cubic_a))
        self.register_buffer("x_idx", x_idx, persistent=True)
        self.register_buffer("x_w", x_w, persistent=True)
        self.register_buffer("y_idx", y_idx, persistent=True)
        self.register_buffer("y_w", y_w, persistent=True)

    def manual_cubic_resize_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        x0 = torch.index_select(heatmaps, 3, self.x_idx[0]) * self.x_w[0].view(1, 1, 1, self.full_w)
        x1 = torch.index_select(heatmaps, 3, self.x_idx[1]) * self.x_w[1].view(1, 1, 1, self.full_w)
        x2 = torch.index_select(heatmaps, 3, self.x_idx[2]) * self.x_w[2].view(1, 1, 1, self.full_w)
        x3 = torch.index_select(heatmaps, 3, self.x_idx[3]) * self.x_w[3].view(1, 1, 1, self.full_w)
        tmp = x0 + x1 + x2 + x3

        y0 = torch.index_select(tmp, 2, self.y_idx[0]) * self.y_w[0].view(1, 1, self.full_h, 1)
        y1 = torch.index_select(tmp, 2, self.y_idx[1]) * self.y_w[1].view(1, 1, self.full_h, 1)
        y2 = torch.index_select(tmp, 2, self.y_idx[2]) * self.y_w[2].view(1, 1, self.full_h, 1)
        y3 = torch.index_select(tmp, 2, self.y_idx[3]) * self.y_w[3].view(1, 1, self.full_h, 1)
        return y0 + y1 + y2 + y3

    def nms_pool(self, hm: torch.Tensor) -> torch.Tensor:
        r = self.nms_radius
        k = 2 * r + 1
        if self.nms_impl == "2d":
            return F.max_pool2d(hm, kernel_size=k, stride=1, padding=r)
        pooled = F.max_pool2d(hm, kernel_size=(k, 1), stride=1, padding=(r, 0))
        pooled = F.max_pool2d(pooled, kernel_size=(1, k), stride=1, padding=(0, r))
        return pooled

    def mask_peaks(self, hm: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        peaks = (hm == pooled) & (hm > self.threshold)
        return torch.where(peaks, hm, torch.full_like(hm, -1.0e9))

    def forward(self, heatmaps: torch.Tensor):
        hm = self.manual_cubic_resize_heatmaps(heatmaps)
        if self.mode == "resize_only":
            return hm

        pooled = self.nms_pool(hm)
        if self.mode == "resize_pool":
            return pooled

        masked = self.mask_peaks(hm, pooled)
        if self.mode == "resize_pool_mask":
            return masked

        flat = masked.flatten(start_dim=2)
        return torch.topk(flat, k=self.topk, dim=2, largest=True, sorted=True)


@dataclass
class ExportedHead:
    mode: str
    onnx: str
    mxr: Optional[str]


@dataclass
class BenchmarkResult:
    mode: str
    model: str
    input_name: str
    input_shape: List[int]
    input_dtype: str
    runs: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def head_name(
    *,
    mode: str,
    batch_size: int,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    channels: int,
    topk: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    cubic_a: float,
) -> str:
    return (
        "heatmap_topk_ablation_"
        f"{mode}_"
        f"b{int(batch_size)}_c{int(channels)}_"
        f"{int(in_h)}x{int(in_w)}_to_{int(full_h)}x{int(full_w)}_"
        f"k{int(topk)}_thr{safe_float_token(threshold)}_"
        f"r{int(nms_radius)}_{nms_impl}_a{safe_float_token(cubic_a)}"
    )


def export_onnx(
    *,
    mode: str,
    output_dir: Path,
    batch_size: int,
    in_h: int,
    in_w: int,
    full_h: int,
    full_w: int,
    channels: int,
    topk: int,
    threshold: float,
    nms_radius: int,
    nms_impl: str,
    cubic_a: float,
    opset: int,
    force: bool,
) -> Path:
    name = head_name(
        mode=mode,
        batch_size=batch_size,
        in_h=in_h,
        in_w=in_w,
        full_h=full_h,
        full_w=full_w,
        channels=channels,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=cubic_a,
    )
    onnx_path = output_dir / f"{name}.onnx"
    if onnx_path.exists() and not force:
        print(f"[export] exists, skipping: {onnx_path}")
        return onnx_path

    model = HeatmapTopKAblationHead(
        mode=mode,
        in_h=in_h,
        in_w=in_w,
        full_h=full_h,
        full_w=full_w,
        channels=channels,
        topk=topk,
        threshold=threshold,
        nms_radius=nms_radius,
        nms_impl=nms_impl,
        cubic_a=cubic_a,
    ).eval()

    x = torch.randn(int(batch_size), int(channels), int(in_h), int(in_w), dtype=torch.float32)
    output_names = ["output"]
    if mode == "resize_pool_mask_topk":
        output_names = ["top_scores", "top_indices"]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] mode={mode} -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,),
            str(onnx_path),
            input_names=["heatmaps"],
            output_names=output_names,
            opset_version=int(opset),
            do_constant_folding=True,
        )
    return onnx_path


def compile_onnx_to_mxr(onnx_path: Path, mxr_path: Path, *, exhaustive_tune: bool = False, force: bool = False) -> Path:
    if mxr_path.exists() and not force:
        print(f"[compile] exists, skipping: {mxr_path}")
        return mxr_path

    mxr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import migraphx  # type: ignore
    except ModuleNotFoundError:
        migraphx = None

    if migraphx is not None and hasattr(migraphx, "parse_onnx"):
        print(f"[compile] Python MIGraphX: {onnx_path.name} -> {mxr_path.name}")
        program = migraphx.parse_onnx(str(onnx_path))
        t0 = time.time()
        program.compile(migraphx.get_target("gpu"), exhaustive_tune=bool(exhaustive_tune))
        migraphx.save(program, str(mxr_path))
        print(f"[compile] saved: {mxr_path} elapsed_s={time.time() - t0:.2f}")
        return mxr_path

    driver = shutil.which("migraphx-driver") or "/opt/rocm/bin/migraphx-driver"
    if not Path(driver).exists() and shutil.which(driver) is None:
        raise RuntimeError("Python migraphx is unavailable and migraphx-driver was not found.")
    if exhaustive_tune:
        print("[warning] exhaustive_tune is ignored by migraphx-driver fallback")
    cmd = [driver, "compile", str(onnx_path), "--onnx", "--gpu", "--binary", "-o", str(mxr_path)]
    print("[compile-fallback] " + " ".join(cmd))
    subprocess.check_call(cmd)
    return mxr_path


def benchmark_mxr(mxr_path: Path, *, runs: int, warmup: int, seed: int, mode: str) -> BenchmarkResult:
    import migraphx  # type: ignore

    program = migraphx.load(str(mxr_path))
    shapes = program.get_parameter_shapes()
    if not shapes:
        raise RuntimeError(f"No parameters found in MXR: {mxr_path}")
    input_name = list(shapes.keys())[0]
    shape = shapes[input_name]
    lens = tuple(int(x) for x in shape.lens())
    stype = str(shape.type()).lower()
    if "half" in stype or "float16" in stype:
        dtype = np.float16
    elif "double" in stype:
        dtype = np.float64
    else:
        dtype = np.float32

    x = np.ascontiguousarray(np.random.default_rng(seed).standard_normal(lens).astype(dtype))
    arg = {input_name: migraphx.argument(x)}

    for _ in range(int(warmup)):
        program.run(arg)

    times: List[float] = []
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        program.run(arg)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return BenchmarkResult(
        mode=mode,
        model=str(mxr_path),
        input_name=str(input_name),
        input_shape=[int(x) for x in lens],
        input_dtype=str(dtype),
        runs=int(runs),
        avg_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
    )


def parse_modes(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(MODES)
    modes = [x.strip() for x in text.split(",") if x.strip()]
    invalid = [m for m in modes if m not in MODES]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid modes: {invalid}; valid={MODES}")
    return modes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export/compile/benchmark heatmap TopK ablation heads.")
    p.add_argument("--shape", nargs=4, type=int, metavar=("IN_H", "IN_W", "FULL_H", "FULL_W"), default=[68, 121, 1080, 1920])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--cubic-a", type=float, default=-0.75)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--modes", type=parse_modes, default=list(MODES), help="Comma-separated modes or 'all'.")
    p.add_argument("--output-dir", type=Path, default=Path("models/heatmap_topk_ablation"))
    p.add_argument("--compile", action="store_true", help="Compile exported ONNX heads to MXR.")
    p.add_argument("--exhaustive-tune", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="Benchmark compiled MXR heads. Implies --compile.")
    p.add_argument("--runs", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--report-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_h, in_w, full_h, full_w = [int(x) for x in args.shape]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    do_compile = bool(args.compile or args.benchmark)
    exported: List[ExportedHead] = []
    benchmarks: List[BenchmarkResult] = []

    for mode in args.modes:
        onnx_path = export_onnx(
            mode=mode,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            in_h=in_h,
            in_w=in_w,
            full_h=full_h,
            full_w=full_w,
            channels=int(args.channels),
            topk=int(args.topk),
            threshold=float(args.threshold),
            nms_radius=int(args.nms_radius),
            nms_impl=str(args.nms_impl),
            cubic_a=float(args.cubic_a),
            opset=int(args.opset),
            force=bool(args.force),
        )
        mxr_path: Optional[Path] = None
        if do_compile:
            mxr_path = onnx_path.with_suffix(".mxr")
            compile_onnx_to_mxr(onnx_path, mxr_path, exhaustive_tune=bool(args.exhaustive_tune), force=bool(args.force))
        exported.append(ExportedHead(mode=mode, onnx=str(onnx_path), mxr=str(mxr_path) if mxr_path else None))

        if args.benchmark:
            assert mxr_path is not None
            result = benchmark_mxr(mxr_path, runs=int(args.runs), warmup=int(args.warmup), seed=int(args.seed), mode=mode)
            benchmarks.append(result)
            print("[benchmark] " + json.dumps(asdict(result), indent=2))

    report = {
        "config": {
            "shape": [in_h, in_w, full_h, full_w],
            "batch_size": int(args.batch_size),
            "channels": int(args.channels),
            "topk": int(args.topk),
            "threshold": float(args.threshold),
            "nms_radius": int(args.nms_radius),
            "nms_impl": str(args.nms_impl),
            "cubic_a": float(args.cubic_a),
            "opset": int(args.opset),
            "modes": list(args.modes),
        },
        "exported": [asdict(x) for x in exported],
        "benchmarks": [asdict(x) for x in benchmarks],
    }

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[report] wrote: {args.report_json}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
