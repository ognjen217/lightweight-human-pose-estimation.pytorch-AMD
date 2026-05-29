#!/usr/bin/env python3
"""CPU final pose assembly from pruned per-limb pair lists."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


BODY_PARTS_KPT_IDS = np.array(
    [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17],
    ],
    dtype=np.int32,
)


def _sq2(arr, name):
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 2:
        raise ValueError(f"{name} should squeeze to 2D, got {arr.shape}")
    return arr


def topk_to_keypoints_pruned(top_scores, top_indices, *, full_width: int, threshold: float = 0.1, num_keypoint_types: int = 18):
    scores = _sq2(top_scores, "top_scores").astype(np.float32)
    indices = _sq2(top_indices, "top_indices").astype(np.int64)
    keypoints_by_type: List[List[int]] = [[] for _ in range(num_keypoint_types)]
    all_keypoints = []
    gid = 0
    for t in range(min(num_keypoint_types, scores.shape[0])):
        for k in range(scores.shape[1]):
            score = float(scores[t, k])
            if not np.isfinite(score) or score <= float(threshold):
                continue
            flat = int(indices[t, k])
            if flat < 0:
                continue
            x = float(flat % int(full_width))
            y = float(flat // int(full_width))
            keypoints_by_type[t].append(gid)
            all_keypoints.append([x, y, score, float(gid)])
            gid += 1
    arr = np.asarray(all_keypoints, dtype=np.float32) if all_keypoints else np.zeros((0, 4), dtype=np.float32)
    return keypoints_by_type, arr


def assemble_poses_from_pruned_pairs(
    top_scores,
    top_indices,
    limb_top_pair_a_idx,
    limb_top_pair_b_idx,
    limb_top_pair_score,
    limb_top_pair_valid,
    *,
    full_width: int,
    threshold: float = 0.1,
    min_pair_score: float = 0.0,
    min_keypoints: int = 3,
    min_avg_score: float = 0.2,
    return_timing: bool = False,
):
    import time
    t0 = time.perf_counter()
    keypoints_by_type, all_keypoints = topk_to_keypoints_pruned(top_scores, top_indices, full_width=full_width, threshold=threshold)
    t_adapter = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()

    a_idx = _sq2(limb_top_pair_a_idx, "a_idx").astype(np.int64)
    b_idx = _sq2(limb_top_pair_b_idx, "b_idx").astype(np.int64)
    score = _sq2(limb_top_pair_score, "pair_score").astype(np.float32)
    valid = _sq2(limb_top_pair_valid, "pair_valid").astype(np.float32)

    pose_entries: List[np.ndarray] = []

    def find_pose_with(kpt_id: int, part_id: int) -> int:
        for i, entry in enumerate(pose_entries):
            if int(entry[part_id]) == int(kpt_id):
                return i
        return -1

    for limb_id, (part_a, part_b) in enumerate(BODY_PARTS_KPT_IDS):
        if limb_id >= a_idx.shape[0]:
            break
        used_a, used_b = set(), set()
        order = np.argsort(-score[limb_id])
        for j in order:
            if valid[limb_id, j] <= 0:
                continue
            s = float(score[limb_id, j])
            if not np.isfinite(s) or s <= float(min_pair_score):
                continue
            la, lb = int(a_idx[limb_id, j]), int(b_idx[limb_id, j])
            if la in used_a or lb in used_b or la < 0 or lb < 0:
                continue
            if la >= len(keypoints_by_type[part_a]) or lb >= len(keypoints_by_type[part_b]):
                continue
            ga, gb = int(keypoints_by_type[part_a][la]), int(keypoints_by_type[part_b][lb])
            used_a.add(la)
            used_b.add(lb)

            pa = find_pose_with(ga, int(part_a))
            pb = find_pose_with(gb, int(part_b))

            if pa < 0 and pb < 0:
                entry = -np.ones(20, dtype=np.float32)
                entry[part_a] = ga
                entry[part_b] = gb
                entry[-2] = float(all_keypoints[ga, 2]) + float(all_keypoints[gb, 2]) + s
                entry[-1] = 2
                pose_entries.append(entry)
            elif pa >= 0 and pb < 0:
                entry = pose_entries[pa]
                if entry[part_b] < 0:
                    entry[part_b] = gb
                    entry[-2] += float(all_keypoints[gb, 2]) + s
                    entry[-1] += 1
            elif pa < 0 and pb >= 0:
                entry = pose_entries[pb]
                if entry[part_a] < 0:
                    entry[part_a] = ga
                    entry[-2] += float(all_keypoints[ga, 2]) + s
                    entry[-1] += 1
            elif pa != pb:
                ea, eb = pose_entries[pa], pose_entries[pb]
                if not np.any((ea[:18] >= 0) & (eb[:18] >= 0)):
                    mask = eb[:18] >= 0
                    ea[:18][mask] = eb[:18][mask]
                    ea[-2] += eb[-2] + s
                    ea[-1] += eb[-1]
                    pose_entries.pop(pb)

    filtered = []
    for entry in pose_entries:
        cnt = float(entry[-1])
        score_sum = float(entry[-2])
        if cnt >= int(min_keypoints) and (score_sum / max(cnt, 1.0)) >= float(min_avg_score):
            filtered.append(entry)

    pose_arr = np.vstack(filtered).astype(np.float32) if filtered else np.zeros((0, 20), dtype=np.float32)
    t_asm = (time.perf_counter() - t1) * 1000.0
    timings = {"topk_adapter": t_adapter, "mx_assembly_total": t_asm, "group_keypoints": t_asm, "group_total": t_asm}
    if return_timing:
        return pose_arr, all_keypoints, timings
    return pose_arr, all_keypoints
