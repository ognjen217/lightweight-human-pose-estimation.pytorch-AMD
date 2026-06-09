#!/usr/bin/env python3
"""
Diagnose CPU/GPU post-processing variants from cached heatmap/PAF tensors.
No MIGraphX import here, so PyTorch ROCm can own the GPU cleanly.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# Compatibility fix for old pycocotools with NumPy >= 1.20
if not hasattr(np, "float"):
    np.float = float

import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from modules.keypoints_gpu_variant import (
        extract_keypoints,
        group_keypoints,
        group_keypoints_fast,
        extract_keypoints_batch,
        extract_keypoints_batch_cv2,
        connections_nms,
        BODY_PARTS_KPT_IDS,
        BODY_PARTS_PAF_IDS,
    )
except Exception:
    from modules.keypoints import (
        extract_keypoints,
        group_keypoints,
        group_keypoints_fast,
        extract_keypoints_batch,
        extract_keypoints_batch_cv2,
        connections_nms,
        BODY_PARTS_KPT_IDS,
        BODY_PARTS_PAF_IDS,
    )


def coco_eval_stats(gt_file_path, dt_file_path):
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, 'keypoints')
    result.evaluate()
    result.accumulate()
    result.summarize()
    keys = ['AP', 'AP50', 'AP75', 'APm', 'APl', 'AR', 'AR50', 'AR75', 'ARm', 'ARl']
    return dict(zip(keys, result.stats.copy()))


def build_coco_detections(image_id, pose_entries, all_keypoints):
    coco_result = []
    if all_keypoints is None or len(all_keypoints) == 0:
        return coco_result
    all_keypoints = np.asarray(all_keypoints, dtype=np.float32)
    to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        person_score = float(pose_entries[n][-2])
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:
                continue
            cx, cy, visibility = 0.0, 0.0, 0
            if keypoint_id != -1:
                cx, cy = all_keypoints[int(keypoint_id), 0:2]
                cx = float(cx + 0.5)
                cy = float(cy + 0.5)
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        score = person_score * max(0, (float(pose_entries[n][-1]) - 1))
        coco_result.append({'image_id': int(image_id), 'category_id': 1, 'keypoints': keypoints, 'score': float(score)})
    return coco_result


def assemble_pose_entries_from_connections(all_keypoints_by_type, connections_by_part, pose_entry_size=20):
    non_empty = [np.asarray(k, dtype=np.float32) for k in all_keypoints_by_type if len(k) > 0]
    if non_empty:
        all_keypoints = np.concatenate(non_empty, axis=0)
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
                pose_entries[i][-2] = np.sum(all_keypoints[[int(conn[0]), int(conn[1])], 2]) + conn[2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for conn in connections:
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == conn[0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = conn[1]
                    elif pose_entries[j][kpt_b_id] == conn[1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = conn[0]
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for conn in connections:
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == conn[0]:
                        pose_entries[j][kpt_b_id] = conn[1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[int(conn[1]), 2] + conn[2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size, dtype=np.float32) * -1
                    pose_entry[kpt_a_id] = conn[0]
                    pose_entry[kpt_b_id] = conn[1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[[int(conn[0]), int(conn[1])], 2]) + conn[2]
                    pose_entries.append(pose_entry)

    filtered = []
    for p in pose_entries:
        if p[-1] < 3:
            continue
        if p[-2] / p[-1] < 0.2:
            continue
        filtered.append(p)
    return np.asarray(filtered, dtype=np.float32), all_keypoints


def postprocess_k20_fast(heatmaps, pafs, original_hw):
    orig_h, orig_w = original_hw
    heatmaps = cv2.resize(heatmaps.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    all_kpts, _ = extract_keypoints_batch_cv2(heatmaps[:, :, :18], max_keypoints_per_type=20)
    return group_keypoints_fast(all_kpts, pafs)


def postprocess_lowres_cpu_group(heatmaps, pafs, original_hw):
    orig_h, orig_w = original_hw
    out_h, out_w = heatmaps.shape[:2]
    all_kpts, _ = extract_keypoints_batch_cv2(heatmaps.astype(np.float32)[:, :, :18], max_keypoints_per_type=20)
    poses, kpts = group_keypoints_fast(all_kpts, pafs.astype(np.float32))
    if len(kpts) > 0:
        kpts[:, 0] *= orig_w / out_w
        kpts[:, 1] *= orig_h / out_h
    return poses, kpts


def gpu_extract_keypoints(heatmaps, device, max_keypoints_per_type=20, threshold=0.1, nms_radius=6):
    # heatmaps expected H x W x C, can be full-res or low-res
    hm = torch.as_tensor(heatmaps[:, :, :18], device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    pooled = F.max_pool2d(hm, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)
    peaks = (hm == pooled) & (hm > threshold)
    all_kpts = []
    total = 0
    for k in range(18):
        coords = torch.nonzero(peaks[0, k], as_tuple=False)
        if coords.numel() == 0:
            all_kpts.append([])
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        scores = hm[0, k, ys, xs]
        keep = min(max_keypoints_per_type, int(scores.numel()))
        top_scores, order = torch.topk(scores, k=keep, largest=True, sorted=True)
        xs_np = xs[order].detach().cpu().numpy()
        ys_np = ys[order].detach().cpu().numpy()
        sc_np = top_scores.detach().cpu().numpy()
        pts = [(float(xs_np[i]), float(ys_np[i]), float(sc_np[i]), total + i) for i in range(keep)]
        all_kpts.append(pts)
        total += len(pts)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return all_kpts, total


def postprocess_gpu_nms_fullres_cpu_group(heatmaps, pafs, original_hw, device):
    orig_h, orig_w = original_hw
    hm_full = cv2.resize(heatmaps.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    paf_full = cv2.resize(pafs.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    # full-res CPU path used nms_radius=6 in current extract_keypoints_batch_cv2
    all_kpts, _ = gpu_extract_keypoints(hm_full, device, nms_radius=6)
    return group_keypoints_fast(all_kpts, paf_full)


def score_paf_connections_gpu(all_keypoints_by_type, pafs, device, points_per_limb=8, min_paf_score=0.05, success_ratio_thr=0.8):
    pafs_t = torch.as_tensor(pafs, device=device, dtype=torch.float32).permute(2, 0, 1).contiguous()
    paf_h, paf_w = pafs.shape[:2]
    grid = torch.arange(points_per_limb, device=device, dtype=torch.float32).view(1, points_per_limb, 1)
    connections_by_part = []
    for part_id, paf_ids in enumerate(BODY_PARTS_PAF_IDS):
        kpts_a_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]], dtype=np.float32)
        kpts_b_np = np.asarray(all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]], dtype=np.float32)
        n, m = len(kpts_a_np), len(kpts_b_np)
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
        b_pair_idx = torch.div(pair_ids, n, rounding_mode='floor')
        a_pair_idx = pair_ids - b_pair_idx * n
        steps = vec_raw_valid / float(points_per_limb - 1)
        a_points = kpts_a[a_pair_idx].reshape(-1, 1, 2)
        points = torch.round(steps * grid + a_points).long()
        x = points[..., 0].reshape(-1).clamp(0, paf_w - 1)
        y = points[..., 1].reshape(-1).clamp(0, paf_h - 1)
        paf_x_id, paf_y_id = int(paf_ids[0]), int(paf_ids[1])
        field = torch.stack((pafs_t[paf_x_id, y, x], pafs_t[paf_y_id, y, x]), dim=-1).reshape(-1, points_per_limb, 2)
        vec = vec_raw_valid / (vec_norm_valid + 1e-6)
        scores_per_point = (field * vec).sum(dim=-1)
        valid_scores = scores_per_point > min_paf_score
        valid_num = valid_scores.sum(dim=1)
        affinity = (scores_per_point * valid_scores.float()).sum(dim=1) / (valid_num.float() + 1e-6)
        success_ratio = valid_num.float() / float(points_per_limb)
        valid_limb_local = torch.nonzero((affinity > 0) & (success_ratio > success_ratio_thr), as_tuple=False).reshape(-1)
        if valid_limb_local.numel() == 0:
            connections_by_part.append([])
            continue
        valid_limbs = pair_ids[valid_limb_local]
        b_idx_t = torch.div(valid_limbs, n, rounding_mode='floor')
        a_idx_t = valid_limbs - b_idx_t * n
        a_idx = a_idx_t.detach().cpu().numpy().astype(np.int32)
        b_idx = b_idx_t.detach().cpu().numpy().astype(np.int32)
        scores = affinity[valid_limb_local].detach().cpu().numpy().astype(np.float32)
        a_idx, b_idx, scores = connections_nms(a_idx, b_idx, scores)
        connections = list(zip(kpts_a_np[a_idx, 3].astype(np.int32), kpts_b_np[b_idx, 3].astype(np.int32), scores))
        connections_by_part.append(connections)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return connections_by_part

def postprocess_standard(heatmaps, pafs, original_hw):
    """
    Original standard full-resolution CPU post-processing:
    1. Resize heatmaps to original image size
    2. Resize PAFs to original image size
    3. Run original per-keypoint extract_keypoints()
    4. Run original group_keypoints()
    """
    orig_h, orig_w = original_hw

    heatmaps = cv2.resize(
        heatmaps.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC,
    )

    pafs = cv2.resize(
        pafs.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC,
    )

    all_keypoints_by_type = []
    total_keypoints_num = 0

    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx],
            all_keypoints_by_type,
            total_keypoints_num,
        )

    return group_keypoints(all_keypoints_by_type, pafs)

def postprocess_gpu_fullres_paf(heatmaps, pafs, original_hw, device):
    orig_h, orig_w = original_hw
    hm_full = cv2.resize(heatmaps.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    paf_full = cv2.resize(pafs.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    all_kpts, _ = extract_keypoints_batch_cv2(hm_full[:, :, :18], max_keypoints_per_type=20)
    connections = score_paf_connections_gpu(all_kpts, paf_full, device)
    return assemble_pose_entries_from_connections(all_kpts, connections)


def postprocess_gpu_lowres_paf(heatmaps, pafs, original_hw, device):
    orig_h, orig_w = original_hw
    out_h, out_w = heatmaps.shape[:2]
    all_kpts, _ = gpu_extract_keypoints(heatmaps.astype(np.float32), device, nms_radius=1)
    connections = score_paf_connections_gpu(all_kpts, pafs.astype(np.float32), device)
    poses, kpts = assemble_pose_entries_from_connections(all_kpts, connections)
    if len(kpts) > 0:
        kpts[:, 0] *= orig_w / out_w
        kpts[:, 1] *= orig_h / out_h
    return poses, kpts


def evaluate_variant(name, fn, items, cache_dir, labels, output_dir):
    detections = []
    times = []
    for idx, item in enumerate(items, 1):
        data = np.load(cache_dir / item['cache'], allow_pickle=False)
        heatmaps = data['heatmaps'].astype(np.float32)
        pafs = data['pafs'].astype(np.float32)
        image_id = int(data['image_id'])
        orig_w = int(data['orig_w'])
        orig_h = int(data['orig_h'])
        t0 = time.perf_counter()
        poses, kpts = fn(heatmaps, pafs, (orig_h, orig_w))
        times.append((time.perf_counter() - t0) * 1000.0)
        detections.extend(build_coco_detections(image_id, poses, kpts))
        if idx % 20 == 0 or idx == 1:
            print(f'  {name}: {idx}/{len(items)}')
    out_path = output_dir / f'detections_{name}.json'
    with open(out_path, 'w') as f:
        json.dump(detections, f)
    stats = coco_eval_stats(labels, str(out_path))
    stats['avg_ms'] = float(np.mean(times)) if times else 0.0
    stats['p95_ms'] = float(np.percentile(times, 95)) if times else 0.0
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default='outputs/cached_migraphx_val_5000')
    ap.add_argument('--labels', default=None)
    ap.add_argument('--output-dir', default='outputs/diagnose_cached_postprocess')
    ap.add_argument(
    '--variants',
    nargs='+',
    default=[
        'standard',
        'k20_fast',
        'lowres_cpu_group',
        'gpu_nms_fullres_cpu_group',
        'gpu_fullres_paf',
        'gpu_lowres_paf',
    ],
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / 'manifest.json', 'r') as f:
        manifest = json.load(f)
    labels = args.labels or manifest['labels']
    items = manifest['items']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Torch postprocess device: {device}')
    if device.type == 'cuda':
        print(f'Torch GPU: {torch.cuda.get_device_name(0)}')

    variant_map = {
        'standard': lambda h, p, hw: postprocess_standard(h, p, hw),
        'k20_fast': lambda h, p, hw: postprocess_k20_fast(h, p, hw),
        'lowres_cpu_group': lambda h, p, hw: postprocess_lowres_cpu_group(h, p, hw),
        'gpu_nms_fullres_cpu_group': lambda h, p, hw: postprocess_gpu_nms_fullres_cpu_group(h, p, hw, device),
        'gpu_fullres_paf': lambda h, p, hw: postprocess_gpu_fullres_paf(h, p, hw, device),
        'gpu_lowres_paf': lambda h, p, hw: postprocess_gpu_lowres_paf(h, p, hw, device),
    }

    results = {}
    for name in args.variants:
        print(f'\nEvaluating variant: {name}')
        results[name] = evaluate_variant(name, variant_map[name], items, cache_dir, labels, output_dir)

    print('\nComparison summary:')
    print(f'{"variant":<30} {"AP":>6} {"AP50":>6} {"AP75":>6} {"AR":>6} {"ms":>9} {"p95":>9}')
    print('-' * 82)
    for name, s in results.items():
        print(f'{name:<30} {s["AP"]:6.3f} {s["AP50"]:6.3f} {s["AP75"]:6.3f} {s["AR"]:6.3f} {s["avg_ms"]:9.2f} {s["p95_ms"]:9.2f}')

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
