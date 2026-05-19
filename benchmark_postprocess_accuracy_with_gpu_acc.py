#!/usr/bin/env python3
"""
Benchmark post-processing variants for lightweight-human-pose-estimation.pytorch-AMD.

Variants included:
  - standard: full-resolution resize + original extract_keypoints + group_keypoints
  - fast: low-resolution postprocess + final keypoint coordinate scaling
  - k20_fast: full-resolution resize + batched keypoint extraction K=20 + group_keypoints_fast
  - gpu: experimental PyTorch GPU-heavy postprocess path

Measures:
  - COCO keypoint AP/AR for each variant
  - postprocess-only latency
  - sparse ROCm power samples during postprocess
  - estimated postprocess energy = avg sampled GPU power * accumulated postprocess wall time

Notes:
  - This isolates postprocess timing as much as possible. Inference is still executed once per image,
    then all postprocess variants are run on the same heatmap/PAF outputs.
  - Power sampling through rocm-smi is relatively slow and sparse; treat energy values as comparative,
    not lab-grade measurements.
  - The GPU variant uses PyTorch ROCm/CUDA if torch.cuda.is_available(); otherwise it falls back to CPU tensors.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import migraphx
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.coco import CocoValDataset
from val import normalize, pad_width
from modules.keypoints import (
    BODY_PARTS_KPT_IDS,
    BODY_PARTS_PAF_IDS,
    connections_nms,
    extract_keypoints,
    extract_keypoints_batch,
    group_keypoints,
    group_keypoints_fast,
)

try:
    from modules.keypoints import extract_keypoints_batch_cv2
except Exception:
    extract_keypoints_batch_cv2 = None


BASE_HEIGHT = 544
BASE_WIDTH = 968
STRIDE = 8
COCO_KPT_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


@dataclass
class VariantMeter:
    name: str
    detections: List[dict] = field(default_factory=list)
    times_ms: List[float] = field(default_factory=list)
    power_w_samples: List[float] = field(default_factory=list)

    def add_time(self, dt_ms: float) -> None:
        self.times_ms.append(float(dt_ms))

    def add_power(self, power_w: float) -> None:
        if power_w > 0:
            self.power_w_samples.append(float(power_w))

    def summary(self) -> Dict[str, float]:
        arr = np.asarray(self.times_ms, dtype=np.float64)
        if arr.size == 0:
            avg_ms = median_ms = p95_ms = total_s = fps_post = 0.0
        else:
            avg_ms = float(np.mean(arr))
            median_ms = float(np.median(arr))
            p95_ms = float(np.percentile(arr, 95))
            total_s = float(np.sum(arr) / 1000.0)
            fps_post = float(1000.0 / avg_ms) if avg_ms > 0 else 0.0

        avg_power = float(np.mean(self.power_w_samples)) if self.power_w_samples else 0.0
        est_energy_j = avg_power * total_s if avg_power > 0 else 0.0
        fps_per_w = fps_post / avg_power if avg_power > 0 else 0.0

        return {
            "images": int(arr.size),
            "avg_post_ms": avg_ms,
            "median_post_ms": median_ms,
            "p95_post_ms": p95_ms,
            "post_fps": fps_post,
            "total_post_s": total_s,
            "avg_gpu_power_w": avg_power,
            "estimated_post_energy_j": est_energy_j,
            "post_fps_per_watt": fps_per_w,
            "power_samples": len(self.power_w_samples),
        }


# -----------------------------------------------------------------------------
# MIGraphX helpers
# -----------------------------------------------------------------------------

def load_migraphx_model(args):
    compiled_model_path = args.model
    if os.path.exists(compiled_model_path):
        print(f"--- Loading compiled model: {compiled_model_path} ---")
        return migraphx.load(compiled_model_path)

    if not args.onnx:
        raise FileNotFoundError(
            f"Compiled model not found: {compiled_model_path}. "
            "Pass --onnx to compile a model, or pass a valid --model .mxr path."
        )

    print(f"--- Compiling ONNX model: {args.onnx} ---")
    model = migraphx.parse_onnx(args.onnx)
    if args.quantization == "fp16":
        migraphx.quantize_fp16(model)
    elif args.quantization == "bf16":
        migraphx.quantize_bf16(model)
    elif args.quantization == "int8":
        target = migraphx.get_target("gpu")
        migraphx.quantize_int8(model, target, [])

    model.compile(migraphx.get_target("gpu"), exhaustive_tune=args.exhaustive_tune)
    migraphx.save(model, compiled_model_path)
    print(f"--- Saved compiled model: {compiled_model_path} ---")
    return model


def run_inference(model, tensor_input: torch.Tensor):
    n_input = tensor_input.cpu().contiguous().numpy()
    param_type = str(model.get_parameter_shapes()["input"].type())

    if "half" in param_type:
        n_input = n_input.astype(np.float16)
    elif "bfloat" in param_type:
        n_input = n_input.astype(np.float32)
    elif "float" in param_type:
        n_input = n_input.astype(np.float32)

    return model.run({"input": np.ascontiguousarray(n_input)})


def infer_migraphx_outputs(
    model,
    img: np.ndarray,
    pad_value=(0, 0, 0),
    img_mean=(128, 128, 128),
    img_scale=1 / 256,
):
    normed_img = normalize(img, img_mean, img_scale)
    orig_h, orig_w, _ = normed_img.shape
    ratio = min(BASE_HEIGHT / orig_h, BASE_WIDTH / orig_w)

    scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    padded_img, pad = pad_width(scaled_img, STRIDE, pad_value, [BASE_HEIGHT, BASE_WIDTH])

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    raw_results = run_inference(model, tensor_img)

    # The original val path uses final two outputs as heatmaps and PAFs.
    heatmaps = np.transpose(np.asarray(raw_results[-2]).squeeze().astype(np.float32), (1, 2, 0))
    pafs = np.transpose(np.asarray(raw_results[-1]).squeeze().astype(np.float32), (1, 2, 0))

    scaled_pad = [p // STRIDE for p in pad]
    heatmaps = heatmaps[
        scaled_pad[0] : heatmaps.shape[0] - scaled_pad[2],
        scaled_pad[1] : heatmaps.shape[1] - scaled_pad[3],
        :,
    ]
    pafs = pafs[
        scaled_pad[0] : pafs.shape[0] - scaled_pad[2],
        scaled_pad[1] : pafs.shape[1] - scaled_pad[3],
        :,
    ]

    return heatmaps, pafs, orig_w, orig_h


# -----------------------------------------------------------------------------
# Power helpers
# -----------------------------------------------------------------------------

def get_gpu_power_rocm_smi() -> float:
    """Return current GPU socket/package power in W, or 0.0 if unavailable."""
    try:
        raw = subprocess.check_output(
            ["rocm-smi", "--showpower", "--json"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        ).decode("utf-8")
        data = json.loads(raw)
        # Usually card0, but keep it generic.
        for card_data in data.values():
            if isinstance(card_data, dict):
                for key, value in card_data.items():
                    if "Power" in key and "W" in key:
                        return float(str(value).split()[0])
        return 0.0
    except Exception:
        return 0.0


def sync_if_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# COCO helpers
# -----------------------------------------------------------------------------

def coco_eval_stats(gt_file_path: str, dt_file_path: str) -> Dict[str, float]:
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, "keypoints")
    result.evaluate()
    result.accumulate()
    result.summarize()

    keys = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
    return {key: float(value) for key, value in zip(keys, result.stats.copy())}


def build_coco_detections(image_id: int, pose_entries: np.ndarray, all_keypoints: np.ndarray) -> List[dict]:
    coco_result = []
    if pose_entries is None or len(pose_entries) == 0:
        return coco_result

    all_keypoints = np.asarray(all_keypoints, dtype=np.float32)

    for pose_entry in pose_entries:
        if len(pose_entry) == 0:
            continue

        keypoints = [0] * 17 * 3
        person_score = float(pose_entry[-2])
        position_id = -1

        for keypoint_id in pose_entry[:-2]:
            position_id += 1
            if position_id == 1:  # skip neck, not in COCO keypoints
                continue

            cx, cy, visibility = 0.0, 0.0, 0
            if keypoint_id != -1:
                kp = all_keypoints[int(keypoint_id)]
                cx = float(kp[0] + 0.5)
                cy = float(kp[1] + 0.5)
                visibility = 1

            coco_idx = COCO_KPT_MAP[position_id]
            if coco_idx >= 0:
                keypoints[coco_idx * 3 + 0] = cx
                keypoints[coco_idx * 3 + 1] = cy
                keypoints[coco_idx * 3 + 2] = visibility

        score = person_score * max(0.0, float(pose_entry[-1] - 1))
        coco_result.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "score": float(score),
            }
        )

    return coco_result


# -----------------------------------------------------------------------------
# CPU postprocess variants
# -----------------------------------------------------------------------------

def postprocess_standard(heatmaps, pafs, original_hw, stride=STRIDE):
    orig_h, orig_w = original_hw
    heatmaps = cv2.resize(heatmaps, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    all_keypoints_by_type = []
    total_keypoints_num = 0
    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx],
            all_keypoints_by_type,
            total_keypoints_num,
        )

    return group_keypoints(all_keypoints_by_type, pafs)


def postprocess_fast(heatmaps, pafs, original_hw, stride=STRIDE):
    orig_h, orig_w = original_hw
    out_h, out_w = heatmaps.shape[:2]

    all_keypoints_by_type = []
    total_keypoints_num = 0
    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx],
            all_keypoints_by_type,
            total_keypoints_num,
        )

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

    scale_x = orig_w / out_w
    scale_y = orig_h / out_h
    for kpt in all_keypoints:
        kpt[0] *= scale_x
        kpt[1] *= scale_y

    return pose_entries, all_keypoints


def postprocess_k20_fast(heatmaps, pafs, original_hw, stride=STRIDE):
    orig_h, orig_w = original_hw
    heatmaps = cv2.resize(heatmaps, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    extractor = extract_keypoints_batch_cv2 if extract_keypoints_batch_cv2 is not None else extract_keypoints_batch
    all_keypoints_by_type, _ = extractor(heatmaps[:, :, :18], max_keypoints_per_type=20)
    return group_keypoints_fast(all_keypoints_by_type, pafs)


# -----------------------------------------------------------------------------
# Experimental GPU postprocess variant
# -----------------------------------------------------------------------------

def extract_keypoints_gpu_nms(
    heatmaps: np.ndarray,
    max_keypoints_per_type: int = 20,
    threshold: float = 0.1,
    nms_radius: int = 1,
):
    """Torch max_pool2d NMS on heatmaps, returns CPU OpenPose-style keypoint lists."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hm = torch.as_tensor(heatmaps[:, :, :18], dtype=torch.float32, device=device)
    hm = hm.permute(2, 0, 1).unsqueeze(0).contiguous()  # 1 x 18 x H x W

    kernel = 2 * nms_radius + 1
    pooled = torch.nn.functional.max_pool2d(hm, kernel_size=kernel, stride=1, padding=nms_radius)
    peaks = (hm == pooled) & (hm > threshold)

    all_keypoints_by_type = []
    total_keypoints_num = 0

    for kpt_idx in range(18):
        ys, xs = torch.nonzero(peaks[0, kpt_idx], as_tuple=True)
        if xs.numel() == 0:
            all_keypoints_by_type.append([])
            continue

        scores = hm[0, kpt_idx, ys, xs]
        k = min(max_keypoints_per_type, int(scores.numel()))
        vals, order = torch.topk(scores, k=k, largest=True, sorted=True)
        xs = xs[order]
        ys = ys[order]

        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()
        vals_np = vals.detach().cpu().numpy()

        keypoints = []
        for i in range(k):
            keypoints.append(
                (
                    float(xs_np[i]),
                    float(ys_np[i]),
                    float(vals_np[i]),
                    int(total_keypoints_num + i),
                )
            )

        all_keypoints_by_type.append(keypoints)
        total_keypoints_num += k

    return all_keypoints_by_type, total_keypoints_num


def score_paf_connections_gpu(
    all_keypoints_by_type,
    pafs: np.ndarray,
    points_per_limb: int = 8,
    min_paf_score: float = 0.05,
    success_ratio_thr: float = 0.8,
):
    """Torch-vectorized PAF candidate scoring; greedy connection NMS remains on CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pafs_t = torch.as_tensor(pafs, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()
    paf_h = pafs_t.shape[1]
    paf_w = pafs_t.shape[2]
    grid = torch.linspace(0.0, 1.0, points_per_limb, device=device).view(1, points_per_limb, 1)

    connections_by_part = []

    for part_id, paf_ids in enumerate(BODY_PARTS_PAF_IDS):
        kpts_a_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]], dtype=np.float32)
        kpts_b_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]], dtype=np.float32)

        n = len(kpts_a_np)
        m = len(kpts_b_np)
        if n == 0 or m == 0:
            connections_by_part.append([])
            continue

        kpts_a = torch.as_tensor(kpts_a_np[:, :2], device=device, dtype=torch.float32)
        kpts_b = torch.as_tensor(kpts_b_np[:, :2], device=device, dtype=torch.float32)

        vec_raw = (kpts_b[:, None, :] - kpts_a[None, :, :]).reshape(-1, 1, 2)
        vec_norm = torch.linalg.norm(vec_raw, dim=-1, keepdim=True)
        valid_vec = vec_norm.reshape(-1) > 1e-6
        if not bool(valid_vec.any().item()):
            connections_by_part.append([])
            continue

        pair_ids = torch.nonzero(valid_vec, as_tuple=False).reshape(-1)
        vec_raw_valid = vec_raw[valid_vec]
        vec_norm_valid = vec_norm[valid_vec]

        b_pair_idx = torch.div(pair_ids, n, rounding_mode="floor")
        a_pair_idx = pair_ids - b_pair_idx * n

        a_points = kpts_a[a_pair_idx].reshape(-1, 1, 2)
        points = torch.round(a_points + vec_raw_valid * grid).long()
        x = points[..., 0].reshape(-1).clamp_(0, paf_w - 1)
        y = points[..., 1].reshape(-1).clamp_(0, paf_h - 1)

        paf_x_id, paf_y_id = int(paf_ids[0]), int(paf_ids[1])
        field_x = pafs_t[paf_x_id, y, x]
        field_y = pafs_t[paf_y_id, y, x]
        field = torch.stack((field_x, field_y), dim=-1).reshape(-1, points_per_limb, 2)

        vec = vec_raw_valid / (vec_norm_valid + 1e-6)
        affinity_scores_per_point = (field * vec).sum(dim=-1)
        valid_affinity_scores = affinity_scores_per_point > min_paf_score
        valid_num = valid_affinity_scores.sum(dim=1)

        affinity_scores = (
            affinity_scores_per_point * valid_affinity_scores.float()
        ).sum(dim=1) / (valid_num.float() + 1e-6)
        success_ratio = valid_num.float() / float(points_per_limb)

        valid_limb_local = torch.nonzero(
            (affinity_scores > 0) & (success_ratio > success_ratio_thr),
            as_tuple=False,
        ).reshape(-1)
        if valid_limb_local.numel() == 0:
            connections_by_part.append([])
            continue

        valid_limbs = pair_ids[valid_limb_local]
        b_idx_t = torch.div(valid_limbs, n, rounding_mode="floor")
        a_idx_t = valid_limbs - b_idx_t * n
        scores_t = affinity_scores[valid_limb_local]

        a_idx = a_idx_t.detach().cpu().numpy().astype(np.int32)
        b_idx = b_idx_t.detach().cpu().numpy().astype(np.int32)
        scores = scores_t.detach().cpu().numpy().astype(np.float32)

        a_idx, b_idx, scores = connections_nms(a_idx, b_idx, scores)
        connections = list(
            zip(
                kpts_a_np[a_idx, 3].astype(np.int32),
                kpts_b_np[b_idx, 3].astype(np.int32),
                scores,
            )
        )
        connections_by_part.append(connections)

    return connections_by_part


def assemble_pose_entries_from_connections(all_keypoints_by_type, connections_by_part, pose_entry_size=20):
    non_empty = [np.asarray(k, dtype=np.float32) for k in all_keypoints_by_type if len(k) > 0]
    if non_empty:
        all_keypoints = np.concatenate(non_empty, axis=0).astype(np.float32)
    else:
        return np.empty((0, pose_entry_size), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    pose_entries = []

    for part_id, connections in enumerate(connections_by_part):
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [np.ones(pose_entry_size, dtype=np.float32) * -1 for _ in range(len(connections))]
            for i, conn in enumerate(connections):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = conn[0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = conn[1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = np.sum(all_keypoints[[conn[0], conn[1]], 2]) + conn[2]
            continue

        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

        if part_id == 17 or part_id == 18:
            for conn in connections:
                for pose in pose_entries:
                    if pose[kpt_a_id] == conn[0] and pose[kpt_b_id] == -1:
                        pose[kpt_b_id] = conn[1]
                    elif pose[kpt_b_id] == conn[1] and pose[kpt_a_id] == -1:
                        pose[kpt_a_id] = conn[0]
            continue

        for conn in connections:
            num = 0
            for pose in pose_entries:
                if pose[kpt_a_id] == conn[0]:
                    pose[kpt_b_id] = conn[1]
                    num += 1
                    pose[-1] += 1
                    pose[-2] += all_keypoints[conn[1], 2] + conn[2]

            if num == 0:
                pose_entry = np.ones(pose_entry_size, dtype=np.float32) * -1
                pose_entry[kpt_a_id] = conn[0]
                pose_entry[kpt_b_id] = conn[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[[conn[0], conn[1]], 2]) + conn[2]
                pose_entries.append(pose_entry)

    filtered_entries = []
    for pose in pose_entries:
        if pose[-1] < 3:
            continue
        if pose[-2] / pose[-1] < 0.2:
            continue
        filtered_entries.append(pose)

    return np.asarray(filtered_entries, dtype=np.float32), all_keypoints


def postprocess_gpu(heatmaps, pafs, original_hw, stride=STRIDE):
    orig_h, orig_w = original_hw
    out_h, out_w = heatmaps.shape[:2]

    all_keypoints_by_type, _ = extract_keypoints_gpu_nms(
        heatmaps,
        max_keypoints_per_type=20,
        threshold=0.1,
        nms_radius=1,
    )
    connections_by_part = score_paf_connections_gpu(
        all_keypoints_by_type,
        pafs,
        points_per_limb=8,
        min_paf_score=0.05,
        success_ratio_thr=0.8,
    )
    pose_entries, all_keypoints = assemble_pose_entries_from_connections(all_keypoints_by_type, connections_by_part)

    scale_x = orig_w / out_w
    scale_y = orig_h / out_h
    if len(all_keypoints) > 0:
        all_keypoints[:, 0] *= scale_x
        all_keypoints[:, 1] *= scale_y

    return pose_entries, all_keypoints


# -----------------------------------------------------------------------------
# Main benchmark loop
# -----------------------------------------------------------------------------

def build_variants(selected: Sequence[str]) -> List[Tuple[str, Callable]]:
    registry = {
        "standard": postprocess_standard,
        "fast": postprocess_fast,
        "k20_fast": postprocess_k20_fast,
        "gpu": postprocess_gpu,
    }
    variants = []
    for name in selected:
        if name not in registry:
            raise ValueError(f"Unknown variant '{name}'. Available: {', '.join(registry)}")
        variants.append((name, registry[name]))
    return variants


def benchmark_variants(args) -> Dict[str, dict]:
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_migraphx_model(args)
    dataset = CocoValDataset(args.labels, args.images_folder)
    variants = build_variants(args.variants)
    meters = {name: VariantMeter(name) for name, _ in variants}

    total_images = len(dataset)
    target_count = args.max_images if args.max_images is not None else max(0, total_images - args.skip_images)

    print("\n--- Postprocess benchmark configuration ---")
    print(f"images_folder: {args.images_folder}")
    print(f"labels:        {args.labels}")
    print(f"variants:      {', '.join(args.variants)}")
    print(f"skip_images:   {args.skip_images}")
    print(f"max_images:    {args.max_images}")
    print(f"power_every:   every {args.power_every} processed image(s) per variant")
    print(f"torch device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

    processed = 0
    for i, sample in enumerate(dataset):  # type: ignore
        if i < args.skip_images:
            continue
        if args.max_images is not None and processed >= args.max_images:
            break

        file_name = sample["file_name"]
        img = sample["img"]
        image_id = int(file_name[0 : file_name.rfind(".")])

        if processed % args.progress_every == 0:
            print(f"  processing image {processed + 1}/{target_count}: {file_name}")

        heatmaps, pafs, orig_w, orig_h = infer_migraphx_outputs(model, img)
        original_hw = (orig_h, orig_w)

        for variant_name, post_fn in variants:
            sync_if_gpu()
            t0 = time.perf_counter()
            pose_entries, all_keypoints = post_fn(heatmaps, pafs, original_hw, STRIDE)
            sync_if_gpu()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            meter = meters[variant_name]
            meter.add_time(dt_ms)
            meter.detections.extend(build_coco_detections(image_id, pose_entries, all_keypoints))

            if args.power_every > 0 and processed % args.power_every == 0:
                meter.add_power(get_gpu_power_rocm_smi())

        processed += 1

    results = {}

    for name, meter in meters.items():
        detections_path = os.path.join(args.output_dir, f"detections_{name}.json")
        with open(detections_path, "w") as f:
            json.dump(meter.detections, f)

        print(f"\n--- COCO eval for variant: {name} ---")
        metrics = coco_eval_stats(args.labels, detections_path)
        summary = meter.summary()
        results[name] = {**summary, **metrics, "detections_path": detections_path}

    write_reports(args.output_dir, results)
    return results


def write_reports(output_dir: str, results: Dict[str, dict]) -> None:
    summary_json = os.path.join(output_dir, "postprocess_accuracy_energy_summary.json")
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)

    summary_csv = os.path.join(output_dir, "postprocess_accuracy_energy_summary.csv")
    fieldnames = [
        "variant",
        "images",
        "avg_post_ms",
        "median_post_ms",
        "p95_post_ms",
        "post_fps",
        "avg_gpu_power_w",
        "estimated_post_energy_j",
        "post_fps_per_watt",
        "AP",
        "AP50",
        "AP75",
        "APm",
        "APl",
        "AR",
        "AR50",
        "AR75",
        "ARm",
        "ARl",
        "detections_path",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant, values in results.items():
            row = {"variant": variant}
            row.update(values)
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("\nComparison summary:")
    print(
        f"{'variant':<14} {'AP':>6} {'AP50':>6} {'AP75':>6} "
        f"{'AR':>6} {'ms':>9} {'p95':>9} {'W':>8} {'J':>10} {'FPS/W':>8}"
    )
    print("-" * 94)
    for variant, values in results.items():
        print(
            f"{variant:<14} "
            f"{values['AP']:>6.3f} {values['AP50']:>6.3f} {values['AP75']:>6.3f} "
            f"{values['AR']:>6.3f} "
            f"{values['avg_post_ms']:>9.2f} {values['p95_post_ms']:>9.2f} "
            f"{values['avg_gpu_power_w']:>8.2f} "
            f"{values['estimated_post_energy_j']:>10.2f} "
            f"{values['post_fps_per_watt']:>8.3f}"
        )

    print(f"\nSaved summary JSON: {summary_json}")
    print(f"Saved summary CSV:  {summary_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenPose postprocessing variants for accuracy, latency, power, and energy."
    )
    parser.add_argument("--model", default="pose_model1_fp16_ref1.mxr", help="Path to compiled MIGraphX .mxr model")
    parser.add_argument("--onnx", default="", help="Optional ONNX path if --model does not exist")
    parser.add_argument("--quantization", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--exhaustive-tune", action="store_true", help="Use MIGraphX exhaustive_tune when compiling ONNX")
    parser.add_argument("--labels", default="coco/annotations/person_keypoints_val2017.json")
    parser.add_argument("--images-folder", default="coco/val2017/")
    parser.add_argument("--output-dir", default="outputs/postprocess_accuracy_energy")
    parser.add_argument("--max-images", type=int, default=5000, help="Number of COCO images to evaluate")
    parser.add_argument("--skip-images", type=int, default=0, help="Skip first N images from dataset")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--power-every", type=int, default=10, help="Sample rocm-smi every N processed images per variant")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard", "fast", "k20_fast", "gpu"],
        choices=["standard", "fast", "k20_fast", "gpu"],
        help="Postprocess variants to test",
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark_variants(parse_args())
