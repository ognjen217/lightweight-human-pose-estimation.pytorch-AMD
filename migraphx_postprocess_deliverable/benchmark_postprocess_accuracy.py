#!/usr/bin/env python3
"""
benchmark_postprocess_accuracy.py

Accuracy + timing benchmark for lightweight-human-pose-estimation.pytorch-AMD
postprocessing variants.

Compares:
  - standard
  - optimized / optimized_batch_k20_findnonzero_v1
  - gpu-nms
  - migraphx-nms
  - migraphx-nms-k20

Main outputs:
  - COCO keypoints JSON per variant
  - CSV summary with AP / AP50 / AP75 / AR + timing metrics

Example:

python benchmark_postprocess_accuracy.py \
  --images /path/to/coco/val2017 \
  --annotations /path/to/coco/annotations/person_keypoints_val2017.json \
  --model pose_model1_fp16_ref1.mxr \
  --migraphx-nms-mxr models/heatmap_nms_head.mxr \
  --variants standard optimized_batch_k20_findnonzero_v1 gpu-nms migraphx-nms migraphx-nms-k20 \
  --max-images 500 \
  --warmup-images 5 \
  --output-dir accuracy_bench_results

Notes:
  - This script assumes that video_val.py exposes PoseEstimator.
  - It tries to call engine.postprocess_by_mode(...) if available.
  - If not available, it falls back to known method names.
  - It does NOT draw or write videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
# Compatibility fix for older pycocotools with NumPy >= 1.24
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required for COCO AP/AR evaluation.\n"
        "Install it with:\n"
        "  pip install pycocotools\n"
    ) from exc


# Import PoseEstimator from local repo.
try:
    from video_val import PoseEstimator
except ImportError as exc:
    raise SystemExit(
        "Could not import PoseEstimator from video_val.py.\n"
        "Run this script from the repository root / deliverable directory."
    ) from exc


# Existing OpenPose-style internal format usually has 18 keypoints:
#   0 nose
#   1 neck
#   2 right shoulder
#   3 right elbow
#   4 right wrist
#   5 left shoulder
#   6 left elbow
#   7 left wrist
#   8 right hip
#   9 right knee
#   10 right ankle
#   11 left hip
#   12 left knee
#   13 left ankle
#   14 right eye
#   15 left eye
#   16 right ear
#   17 left ear
#
# COCO order has 17 keypoints:
#   0 nose
#   1 left eye
#   2 right eye
#   3 left ear
#   4 right ear
#   5 left shoulder
#   6 right shoulder
#   7 left elbow
#   8 right elbow
#   9 left wrist
#   10 right wrist
#   11 left hip
#   12 right hip
#   13 left knee
#   14 right knee
#   15 left ankle
#   16 right ankle
#
# Map COCO index -> OpenPose 18 index.
COCO_TO_OPENPOSE_18 = [
    0,   # nose
    15,  # left eye
    14,  # right eye
    17,  # left ear
    16,  # right ear
    5,   # left shoulder
    2,   # right shoulder
    6,   # left elbow
    3,   # right elbow
    7,   # left wrist
    4,   # right wrist
    11,  # left hip
    8,   # right hip
    12,  # left knee
    9,   # right knee
    13,  # left ankle
    10,  # right ankle
]


DEFAULT_VARIANTS = [
    "standard",
    "optimized_batch_k20_findnonzero_v1",
    "gpu-nms",
    "migraphx-nms",
    "migraphx-nms-k20",
]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def normalize_timings(timings: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Normalize timing dictionary keys across older/newer video_val.py versions."""
    timings = timings or {}

    def get(*names: str) -> float:
        for name in names:
            if name in timings:
                try:
                    return float(timings[name])
                except Exception:
                    return 0.0
        return 0.0

    return {
        "decode": get("decode"),
        "resize_heatmaps": get("resize_heatmaps", "hm_resize"),
        "resize_pafs": get("resize_pafs", "paf_resize"),
        "mx_nms": get("mx_nms"),
        "extract_keypoints": get("extract_keypoints", "extract"),
        "extract_from_mask": get("extract_from_mask", "mask_ext"),
        "group_keypoints": get("group_keypoints", "group", "group_total"),
        "total_postprocess": get("total_postprocess", "post_total"),
    }


def call_postprocess(
    engine: Any,
    raw_results: Any,
    image_hw: Tuple[int, int],
    variant: str,
    migraphx_nms_mxr: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Calls whichever postprocess API exists in current video_val.py.

    Preferred:
      engine.postprocess_by_mode(results, hw, mode=..., migraphx_nms_mxr=...)

    Fallbacks:
      - standard -> engine.postprocess
      - optimized / optimized_batch_k20_findnonzero_v1 -> engine.postprocess_optimized
      - gpu-nms -> engine.postprocess_hybrid_gpu_nms
      - migraphx-nms / migraphx-nms-k20 -> engine.postprocess_migraphx_nms
    """
    if hasattr(engine, "postprocess_by_mode"):
        # Different video_val.py revisions expose slightly different
        # postprocess_by_mode signatures. Pass only kwargs that exist.
        import inspect

        fn = engine.postprocess_by_mode
        params = inspect.signature(fn).parameters
        kwargs = {}

        if "mode" in params:
            kwargs["mode"] = variant
        elif "postprocess_mode" in params:
            kwargs["postprocess_mode"] = variant

        if "migraphx_nms_mxr" in params:
            kwargs["migraphx_nms_mxr"] = migraphx_nms_mxr
        elif "migraphx_nms_mxr_path" in params:
            kwargs["migraphx_nms_mxr_path"] = migraphx_nms_mxr
        elif "nms_mxr_path" in params:
            kwargs["nms_mxr_path"] = migraphx_nms_mxr

        if kwargs:
            out = fn(raw_results, image_hw, **kwargs)
        else:
            out = fn(raw_results, image_hw, variant)
    elif hasattr(engine, "run_postprocess"):
        out = engine.run_postprocess(
            raw_results,
            image_hw,
            mode=variant,
            migraphx_nms_mxr=migraphx_nms_mxr,
        )
    else:
        if variant == "standard":
            out = engine.postprocess(raw_results, image_hw)
        elif variant in {"optimized", "optimized_batch_k20_findnonzero_v1"}:
            out = engine.postprocess_optimized(raw_results, image_hw)
        elif variant == "gpu-nms":
            out = engine.postprocess_hybrid_gpu_nms(raw_results, image_hw)
        elif variant in {"migraphx-nms", "migraphx-nms-k20"}:
            if not hasattr(engine, "postprocess_migraphx_nms"):
                raise AttributeError(
                    "video_val.py does not expose postprocess_migraphx_nms "
                    "or postprocess_by_mode. Update video_val.py first."
                )
            max_kpts = 20 if variant.endswith("-k20") else None
            out = engine.postprocess_migraphx_nms(
                raw_results,
                image_hw,
                migraphx_nms_mxr=migraphx_nms_mxr,
                max_kpts=max_kpts,
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

    if not isinstance(out, tuple):
        raise TypeError(f"Postprocess output for {variant} is not a tuple: {type(out)}")

    if len(out) == 2:
        poses, keypoints = out
        timings = {}
    elif len(out) == 3:
        poses, keypoints, timings = out
    else:
        raise ValueError(f"Unexpected postprocess output length for {variant}: {len(out)}")

    return np.asarray(poses), np.asarray(keypoints), normalize_timings(timings)


def pose_to_coco_detection(
    image_id: int,
    pose: np.ndarray,
    all_keypoints: np.ndarray,
    min_score: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Convert one internal pose entry to COCO detection dictionary.

    Internal pose_entry usually has:
      first 18 values: global keypoint ids, -1 if absent
      [-2]: pose score
      [-1]: number of detected keypoints

    COCO result keypoints:
      [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17]
    """
    if pose.size < 20 or all_keypoints.size == 0:
        return None

    coco_keypoints: List[float] = []
    visible_count = 0
    score_sum = 0.0

    for openpose_idx in COCO_TO_OPENPOSE_18:
        if openpose_idx >= len(pose):
            coco_keypoints.extend([0.0, 0.0, 0.0])
            continue

        global_id = int(pose[openpose_idx])
        if global_id < 0 or global_id >= len(all_keypoints):
            coco_keypoints.extend([0.0, 0.0, 0.0])
            continue

        x, y, kpt_score = all_keypoints[global_id][:3]
        kpt_score = float(kpt_score)

        # COCO detection results convention:
        # v can be 0/1/2. For predictions, 2 is commonly used for present keypoints.
        coco_keypoints.extend([float(x), float(y), 2.0])
        visible_count += 1
        score_sum += kpt_score

    if visible_count == 0:
        return None

    pose_score = float(pose[-2]) if pose.size >= 20 else score_sum / max(visible_count, 1)
    score = pose_score / max(float(pose[-1]), 1.0) if pose.size >= 20 else score_sum / visible_count

    if score < min_score:
        return None

    return {
        "image_id": int(image_id),
        "category_id": 1,
        "keypoints": coco_keypoints,
        "score": float(score),
    }


def evaluate_coco_keypoints(
    coco_gt: COCO,
    detections: List[Dict[str, Any]],
    image_ids: Sequence[int],
    json_path: Path,
) -> Dict[str, float]:
    """Save detections and run COCO keypoint evaluation."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(detections, f)

    if len(detections) == 0:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APM": 0.0,
            "APL": 0.0,
            "AR": 0.0,
            "AR50": 0.0,
            "AR75": 0.0,
            "ARM": 0.0,
            "ARL": 0.0,
        }

    coco_dt = coco_gt.loadRes(str(json_path))
    evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
    evaluator.params.imgIds = list(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APM": float(stats[3]),
        "APL": float(stats[4]),
        "AR": float(stats[5]),
        "AR50": float(stats[6]),
        "AR75": float(stats[7]),
        "ARM": float(stats[8]),
        "ARL": float(stats[9]),
    }


def load_eval_images(coco: COCO, images_dir: Path, max_images: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load COCO images that have person annotations.
    This avoids evaluating images with no person labels unless you explicitly modify this.
    """
    person_cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=person_cat_ids))
    if max_images is not None and max_images > 0:
        img_ids = img_ids[:max_images]

    images = coco.loadImgs(img_ids)

    existing = []
    missing = 0
    for info in images:
        path = images_dir / info["file_name"]
        if path.exists():
            existing.append(info)
        else:
            missing += 1

    if missing:
        print(f"[warn] Missing {missing} image files under {images_dir}")

    return existing



def set_migraphx_nms_for_shape(
    engine: Any,
    cache_dir: Optional[Path],
    image_hw: Tuple[int, int],
) -> Optional[Path]:
    """
    Select a full-resolution MIGraphX NMS head for the current image shape.

    The cache is expected to contain files named:
      heatmap_nms_head_<H>x<W>.mxr

    Example:
      image_hw=(426, 640)
      -> models/nms_fullres_cache/heatmap_nms_head_426x640.mxr

    The function keeps loaded MIGraphXNMSHead objects in memory, so repeated
    dimensions do not reload/parse the same MXR again.
    """
    if cache_dir is None:
        return None

    h, w = int(image_hw[0]), int(image_hw[1])
    mxr_path = cache_dir / f"heatmap_nms_head_{h}x{w}.mxr"

    if not mxr_path.exists():
        raise FileNotFoundError(
            "Missing MIGraphX NMS head for current image shape.\n"
            f"  image shape: {h}x{w}\n"
            f"  expected:    {mxr_path}\n\n"
            "Generate the missing cache file with, for example:\n"
            "  python tools/compile_coco_nms_heads.py \\\n"
            "    --annotations coco/annotations/person_keypoints_val2017.json \\\n"
            "    --limit 1000 \\\n"
            f"    --output-dir {cache_dir}\n"
        )

    if not hasattr(engine, "_benchmark_migraphx_nms_cache"):
        engine._benchmark_migraphx_nms_cache = {}
        engine._benchmark_migraphx_nms_current_shape = None

    key = (h, w)
    cache = engine._benchmark_migraphx_nms_cache

    if key not in cache:
        from modules.migraphx_nms import MIGraphXNMSHead

        print(f"[mx-nms-cache] loading {h}x{w}: {mxr_path}")
        cache[key] = MIGraphXNMSHead(str(mxr_path), input_name="heatmaps")

    if getattr(engine, "_benchmark_migraphx_nms_current_shape", None) != key:
        # Current video_val.py uses self._migraphx_nms in postprocess_migraphx_nms().
        engine._migraphx_nms = cache[key]
        engine._benchmark_migraphx_nms_current_shape = key
        engine._benchmark_migraphx_nms_current_path = str(mxr_path)

    return mxr_path

def run_variant(
    variant: str,
    engine: Any,
    coco_gt: COCO,
    images: Sequence[Dict[str, Any]],
    images_dir: Path,
    output_dir: Path,
    migraphx_nms_mxr: Optional[str],
    migraphx_nms_cache_dir: Optional[Path],
    warmup_images: int,
    min_pose_score: float,
) -> Dict[str, Any]:
    print("\n" + "=" * 120)
    print(f"Running accuracy benchmark variant: {variant}")
    print("=" * 120)

    detections: List[Dict[str, Any]] = []
    image_ids: List[int] = []

    timing_values: Dict[str, List[float]] = {
        "preprocess": [],
        "infer": [],
        "decode": [],
        "resize_heatmaps": [],
        "resize_pafs": [],
        "mx_nms": [],
        "extract_keypoints": [],
        "extract_from_mask": [],
        "group_keypoints": [],
        "total_postprocess": [],
        "total_frame": [],
    }

    processed = 0

    for idx, img_info in enumerate(images, start=1):
        image_id = int(img_info["id"])
        image_path = images_dir / img_info["file_name"]

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[warn] Could not read image: {image_path}")
            continue

        frame_start = time.perf_counter()

        t0 = time.perf_counter()
        input_tensor = engine.preprocess(frame)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw_results = engine.model.run({"input": input_tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0

        # For full-res MIGraphX NMS on COCO, each image size may require
        # a different statically compiled MXR head. Select it per image.
        if variant in {"migraphx-nms", "migraphx-nms-k20"}:
            set_migraphx_nms_for_shape(
                engine=engine,
                cache_dir=migraphx_nms_cache_dir,
                image_hw=frame.shape[:2],
            )

        poses, keypoints, post_times = call_postprocess(
            engine=engine,
            raw_results=raw_results,
            image_hw=frame.shape[:2],
            variant=variant,
            migraphx_nms_mxr=migraphx_nms_mxr,
        )

        total_frame_ms = (time.perf_counter() - frame_start) * 1000.0

        # Skip warmup frames from timings and detections by default.
        # This keeps cold-start kernels / compilation effects out of AP comparison and timing means.
        if idx <= warmup_images:
            if idx == warmup_images:
                print(f"[info] Warmup complete after {warmup_images} image(s).")
            continue

        image_ids.append(image_id)

        for pose in poses:
            det = pose_to_coco_detection(
                image_id=image_id,
                pose=np.asarray(pose),
                all_keypoints=np.asarray(keypoints),
                min_score=min_pose_score,
            )
            if det is not None:
                detections.append(det)

        timing_values["preprocess"].append(preprocess_ms)
        timing_values["infer"].append(infer_ms)
        timing_values["total_frame"].append(total_frame_ms)

        for key in [
            "decode",
            "resize_heatmaps",
            "resize_pafs",
            "mx_nms",
            "extract_keypoints",
            "extract_from_mask",
            "group_keypoints",
            "total_postprocess",
        ]:
            timing_values[key].append(float(post_times.get(key, 0.0)))

        processed += 1

        if processed == 1 or processed % 50 == 0:
            print(
                f"[{variant}] processed={processed:5d} "
                f"image_id={image_id} "
                f"detections={len(detections):6d} "
                f"post={timing_values['total_postprocess'][-1]:8.2f} ms "
                f"frame={total_frame_ms:8.2f} ms"
            )

    if not image_ids:
        raise RuntimeError(f"No images evaluated for variant {variant}.")

    json_path = output_dir / f"coco_keypoints_{variant}.json"
    metrics = evaluate_coco_keypoints(coco_gt, detections, image_ids, json_path)

    summary: Dict[str, Any] = {
        "variant": variant,
        "num_images": len(image_ids),
        "num_detections": len(detections),
        **metrics,
    }

    for key, values in timing_values.items():
        summary[f"{key}_mean_ms"] = mean(values)
        summary[f"{key}_p95_ms"] = percentile(values, 95)

    print(f"[done] {variant}")
    print(
        f"AP={summary['AP']:.4f} AP50={summary['AP50']:.4f} "
        f"AP75={summary['AP75']:.4f} AR={summary['AR']:.4f} "
        f"post={summary['total_postprocess_mean_ms']:.2f} ms "
        f"frame={summary['total_frame_mean_ms']:.2f} ms"
    )

    return summary


def write_summary_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "variant",
        "num_images",
        "num_detections",
        "AP",
        "AP50",
        "AP75",
        "APM",
        "APL",
        "AR",
        "AR50",
        "AR75",
        "ARM",
        "ARL",
        "preprocess_mean_ms",
        "infer_mean_ms",
        "decode_mean_ms",
        "resize_heatmaps_mean_ms",
        "resize_pafs_mean_ms",
        "mx_nms_mean_ms",
        "extract_keypoints_mean_ms",
        "extract_from_mask_mean_ms",
        "group_keypoints_mean_ms",
        "total_postprocess_mean_ms",
        "total_frame_mean_ms",
        "total_frame_p95_ms",
        "total_postprocess_p95_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_compact_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nComparison summary:")
    print(
        f"{'variant':36s} "
        f"{'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'AR':>7s} "
        f"{'pre':>8s} {'infer':>8s} {'mx_nms':>8s} "
        f"{'extract':>9s} {'mask_ext':>9s} {'group':>8s} "
        f"{'post':>8s} {'frame':>8s} {'p95':>8s}"
    )
    print("-" * 150)

    for row in rows:
        print(
            f"{row['variant']:36s} "
            f"{row['AP']:7.4f} {row['AP50']:7.4f} {row['AP75']:7.4f} {row['AR']:7.4f} "
            f"{row['preprocess_mean_ms']:8.2f} "
            f"{row['infer_mean_ms']:8.2f} "
            f"{row['mx_nms_mean_ms']:8.2f} "
            f"{row['extract_keypoints_mean_ms']:9.2f} "
            f"{row['extract_from_mask_mean_ms']:9.2f} "
            f"{row['group_keypoints_mean_ms']:8.2f} "
            f"{row['total_postprocess_mean_ms']:8.2f} "
            f"{row['total_frame_mean_ms']:8.2f} "
            f"{row['total_frame_p95_ms']:8.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare COCO accuracy and timing of postprocess variants."
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to COCO val2017 images directory.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to COCO person_keypoints_val2017.json.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MIGraphX pose model .mxr.",
    )
    parser.add_argument(
        "--migraphx-nms-mxr",
        default="models/heatmap_nms_head.mxr",
        help="Path to one compiled MIGraphX heatmap NMS head .mxr. Used when --migraphx-nms-cache-dir is not set.",
    )
    parser.add_argument(
        "--migraphx-nms-cache-dir",
        default=None,
        help=(
            "Directory with per-resolution full-res MIGraphX NMS heads named "
            "heatmap_nms_head_<H>x<W>.mxr. If set, migraphx-nms variants "
            "select the correct MXR per image shape."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"Postprocess variants to compare. Default: {' '.join(DEFAULT_VARIANTS)}",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of COCO images. 0 means all available person images.",
    )
    parser.add_argument(
        "--warmup-images",
        type=int,
        default=0,
        help="Number of initial images to run but exclude from timings and AP.",
    )
    parser.add_argument(
        "--min-pose-score",
        type=float,
        default=0.0,
        help="Drop predicted poses below this score before COCO eval.",
    )
    parser.add_argument(
        "--output-dir",
        default="accuracy_bench_results",
        help="Directory for JSON detections and CSV summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images)
    annotations_path = Path(args.annotations)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    migraphx_nms_cache_dir = Path(args.migraphx_nms_cache_dir) if args.migraphx_nms_cache_dir else None

    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not annotations_path.exists():
        raise SystemExit(f"Annotations file not found: {annotations_path}")
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if migraphx_nms_cache_dir is not None and not migraphx_nms_cache_dir.exists():
        raise SystemExit(f"MIGraphX NMS cache dir not found: {migraphx_nms_cache_dir}")

    max_images = None if args.max_images <= 0 else args.max_images

    print("Loading COCO annotations...")
    coco_gt = COCO(str(annotations_path))

    print("Loading image list...")
    images = load_eval_images(coco_gt, images_dir, max_images=max_images)
    if not images:
        raise SystemExit("No valid COCO images found.")

    print(f"Images selected: {len(images)}")
    print(f"Variants: {' '.join(args.variants)}")
    print(f"Output directory: {output_dir}")

    print("Loading PoseEstimator...")
    engine = PoseEstimator(str(model_path))

    if any(v in {"migraphx-nms", "migraphx-nms-k20"} for v in args.variants):
        if migraphx_nms_cache_dir is not None:
            print(f"Using MIGraphX NMS full-res cache dir: {migraphx_nms_cache_dir}")
        else:
            if hasattr(engine, "load_migraphx_nms"):
                engine.load_migraphx_nms(args.migraphx_nms_mxr)
                print(f"Loaded static MIGraphX NMS head: {args.migraphx_nms_mxr}")

    rows: List[Dict[str, Any]] = []

    for variant in args.variants:
        if variant in {"migraphx-nms", "migraphx-nms-k20"}:
            if migraphx_nms_cache_dir is None and not Path(args.migraphx_nms_mxr).exists():
                print(
                    f"[warn] Skipping {variant}: MIGraphX NMS MXR not found: "
                    f"{args.migraphx_nms_mxr}"
                )
                continue

        row = run_variant(
            variant=variant,
            engine=engine,
            coco_gt=coco_gt,
            images=images,
            images_dir=images_dir,
            output_dir=output_dir,
            migraphx_nms_mxr=args.migraphx_nms_mxr,
            migraphx_nms_cache_dir=migraphx_nms_cache_dir,
            warmup_images=args.warmup_images,
            min_pose_score=args.min_pose_score,
        )
        rows.append(row)

    csv_path = output_dir / "postprocess_accuracy_summary.csv"
    write_summary_csv(rows, csv_path)
    print_compact_table(rows)
    print(f"\nSaved summary CSV: {csv_path}")
    print(f"Saved detection JSON files under: {output_dir}")


if __name__ == "__main__":
    main()