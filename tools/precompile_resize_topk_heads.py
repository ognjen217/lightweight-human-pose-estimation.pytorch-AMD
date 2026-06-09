#!/usr/bin/env python3
"""Precompile MIGraphX resize+NMS+TopK heads for COCO image shapes.

This intentionally compiles each shape in a separate Python subprocess so the
main accuracy-validation process does not repeatedly call MIGraphX parse/compile
while also running the main pose model.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Iterable, List, Tuple

import cv2

from datasets.coco import CocoValDataset
from modules.migraphx_resize_topk_compiler import head_name


def _padded_shape_and_pad(h: int, w: int, *, base_height: int, base_width: int, stride: int) -> Tuple[int, int, List[int]]:
    ratio = min(float(base_height) / float(h), float(base_width) / float(w))
    # Match cv2.resize(..., fx=ratio, fy=ratio) sizing as closely as possible by
    # performing a tiny resize-free equivalent with OpenCV-compatible rounding.
    # Python round() is banker's rounding, so use actual resize dimensions via cv2
    # in collect_shapes where the image is available.
    raise RuntimeError("internal helper should not be called directly")


def _shape_for_image(img, *, base_height: int, base_width: int, stride: int) -> Tuple[int, int, int, int]:
    orig_h, orig_w = img.shape[:2]
    ratio = min(float(base_height) / float(orig_h), float(base_width) / float(orig_w))

    # Use cv2.resize exactly like accuracy_validation.prepare_coco_input so shape
    # rounding matches the real validation path.
    resized = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    scaled_h, scaled_w = resized.shape[:2]

    min_h = int(math.ceil(float(scaled_h) / float(stride)) * stride)
    min_w = int(max(base_width, scaled_w))
    min_w = int(math.ceil(float(min_w) / float(stride)) * stride)

    pad_top = int(math.floor((min_h - scaled_h) / 2.0))
    pad_left = int(math.floor((min_w - scaled_w) / 2.0))
    pad_bottom = int(min_h - scaled_h - math.floor((min_h - scaled_h) / 2.0))
    pad_right = int(min_w - scaled_w - math.floor((min_w - scaled_w) / 2.0))

    full_low_h = int(base_height // stride)
    full_low_w = int(base_width // stride)

    # accuracy_validation.run_model_on_image uses pad // stride for crop indices.
    top = pad_top // stride
    left = pad_left // stride
    bottom = pad_bottom // stride
    right = pad_right // stride

    low_h = full_low_h - top - bottom
    low_w = full_low_w - left - right
    if low_h <= 0 or low_w <= 0:
        raise RuntimeError(
            f"Invalid low-res shape for image {orig_h}x{orig_w}: low={low_h}x{low_w}, pad={[pad_top, pad_left, pad_bottom, pad_right]}"
        )

    return int(low_h), int(low_w), int(orig_h), int(orig_w)


def collect_shapes(args: argparse.Namespace) -> List[Tuple[int, int, int, int]]:
    dataset = CocoValDataset(args.labels, args.images_folder)
    shapes = []
    processed = 0
    for idx, sample in enumerate(dataset):  # type: ignore
        if idx < args.skip_images:
            continue
        if args.max_images is not None and processed >= args.max_images:
            break
        shapes.append(
            _shape_for_image(
                sample["img"],
                base_height=args.base_height,
                base_width=args.base_width,
                stride=args.stride,
            )
        )
        processed += 1
    return sorted(set(shapes))


def compile_one(args: argparse.Namespace, shape: Tuple[int, int, int, int], idx: int, total: int) -> None:
    in_h, in_w, out_h, out_w = shape
    output_dir = Path(args.output_dir)
    name = head_name(
        in_h,
        in_w,
        out_h,
        out_w,
        topk=args.topk,
        resize_mode=args.resize_mode,
        nms_impl=args.nms_impl,
    )
    mxr = output_dir / f"{name}.mxr"
    if mxr.exists() and not args.force:
        print(f"[precompile {idx}/{total}] exists: {mxr}")
        return

    print(f"[precompile {idx}/{total}] {in_h}x{in_w} -> {out_h}x{out_w}")
    code = r'''
from pathlib import Path
from modules.migraphx_resize_topk_compiler import compile_resize_nms_topk_head
compile_resize_nms_topk_head(
    in_h=IN_H,
    in_w=IN_W,
    out_h=OUT_H,
    out_w=OUT_W,
    output_dir=OUTPUT_DIR,
    channels=CHANNELS,
    topk=TOPK,
    threshold=THRESHOLD,
    nms_radius=NMS_RADIUS,
    nms_impl=NMS_IMPL,
    resize_mode=RESIZE_MODE,
    force=FORCE,
    keep_onnx=KEEP_ONNX,
    exhaustive_tune=EXHAUSTIVE_TUNE,
)
'''
    replacements = {
        "IN_H": repr(in_h),
        "IN_W": repr(in_w),
        "OUT_H": repr(out_h),
        "OUT_W": repr(out_w),
        "OUTPUT_DIR": repr(str(output_dir)),
        "CHANNELS": repr(args.channels),
        "TOPK": repr(args.topk),
        "THRESHOLD": repr(args.threshold),
        "NMS_RADIUS": repr(args.nms_radius),
        "NMS_IMPL": repr(args.nms_impl),
        "RESIZE_MODE": repr(args.resize_mode),
        "FORCE": repr(bool(args.force)),
        "KEEP_ONNX": repr(bool(args.keep_onnx)),
        "EXHAUSTIVE_TUNE": repr(bool(args.exhaustive_tune)),
    }
    for k, v in replacements.items():
        code = code.replace(k, v)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompile resize+NMS+TopK MIGraphX heads for COCO validation.")
    p.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    p.add_argument("--images-folder", default="coco/val2017")
    p.add_argument("--output-dir", default="models/resize_nms_topk_coco_cache")
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--skip-images", type=int, default=0)
    p.add_argument("--base-height", type=int, default=544)
    p.add_argument("--base-width", type=int, default=968)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-radius", type=int, default=6)
    p.add_argument("--nms-impl", choices=["2d", "separable"], default="separable")
    p.add_argument("--resize-mode", choices=["nearest", "bilinear", "bicubic"], default="bilinear")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-onnx", action="store_true")
    p.add_argument("--exhaustive-tune", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shapes = collect_shapes(args)
    print(f"[precompile] unique heads needed: {len(shapes)}")
    for idx, shape in enumerate(shapes, start=1):
        compile_one(args, shape, idx, len(shapes))
    print("[precompile] done")


if __name__ == "__main__":
    main()
