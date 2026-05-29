#!/usr/bin/env python3
"""
CPU pose assembly using MXR-computed PAF pair scores.

This module replaces:
  full-res PAF resize + PAF sampling + affinity calculation

with:
  pair_scores/pair_valid produced by modules.migraphx_paf_pair_scorer

It still performs greedy connection NMS and pose assembly on CPU because those
steps are dynamic/list-oriented and are not good first targets for ONNX/MIGraphX.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS, connections_nms


def topk_to_keypoint_lists(
    top_scores: np.ndarray,
    top_indices: np.ndarray,
    *,
    full_width: int,
    threshold: float = 0.1,
    num_keypoint_types: int = 18,
) -> Tuple[List[List[Tuple[int, int, float, int]]], np.ndarray]:
    """Convert fixed [1,18,K] TopK outputs to the existing keypoint-list format."""
    scores = np.asarray(top_scores).reshape(num_keypoint_types, -1)
    indices = np.asarray(top_indices).reshape(num_keypoint_types, -1)

    all_keypoints_by_type: List[List[Tuple[int, int, float, int]]] = []
    all_keypoints: List[Tuple[int, int, float, int]] = []
    next_id = 0

    for kpt_type in range(num_keypoint_types):
        kpts: List[Tuple[int, int, float, int]] = []
        for k in range(scores.shape[1]):
            score = float(scores[kpt_type, k])
            if score <= threshold or score < -1.0e8:
                continue
            flat = int(indices[kpt_type, k])
            if flat < 0:
                continue
            x = int(flat % int(full_width))
            y = int(flat // int(full_width))
            item = (x, y, score, next_id)
            kpts.append(item)
            all_keypoints.append(item)
            next_id += 1
        all_keypoints_by_type.append(kpts)

    if all_keypoints:
        all_keypoints_arr = np.asarray(all_keypoints, dtype=np.float32)
    else:
        all_keypoints_arr = np.empty((0, 4), dtype=np.float32)
    return all_keypoints_by_type, all_keypoints_arr


def group_keypoints_from_mx_pair_scores(
    all_keypoints_by_type: Sequence[Sequence[Sequence[float]]],
    pair_scores: np.ndarray,
    pair_valid: np.ndarray,
    *,
    pose_entry_size: int = 20,
    min_pair_score: float = 0.0,
    return_timing: bool = False,
):
    """Assemble poses using precomputed [19,K,K] limb scores.

    pair_scores convention:
        pair_scores[part_id, a_idx, b_idx]
    where a_idx indexes BODY_PARTS_KPT_IDS[part_id][0], and b_idx indexes
    BODY_PARTS_KPT_IDS[part_id][1].
    """
    t_total = time.perf_counter()
    timings: Dict[str, float] = {
        "mx_assembly_prepare": 0.0,
        "mx_assembly_connections": 0.0,
        "mx_assembly_nms": 0.0,
        "mx_assembly_pose": 0.0,
        "mx_assembly_filter": 0.0,
        "mx_assembly_total": 0.0,
        "mx_assembly_connections_total": 0,
    }

    t0 = time.perf_counter()
    all_keypoints_by_type_np = [
        np.asarray(kpts, dtype=np.float32) for kpts in all_keypoints_by_type
    ]
    non_empty = [kpts for kpts in all_keypoints_by_type_np if len(kpts) > 0]
    all_keypoints = np.concatenate(non_empty, axis=0) if non_empty else np.empty((0, 4), dtype=np.float32)

    ps = np.asarray(pair_scores, dtype=np.float32)
    pv = np.asarray(pair_valid, dtype=np.float32)
    if ps.ndim == 4:
        ps = ps[0]
    if pv.ndim == 4:
        pv = pv[0]
    timings["mx_assembly_prepare"] += (time.perf_counter() - t0) * 1000.0

    pose_entries: List[np.ndarray] = []

    for part_id in range(len(BODY_PARTS_KPT_IDS)):
        t0 = time.perf_counter()
        kpt_a_type, kpt_b_type = BODY_PARTS_KPT_IDS[part_id]
        kpts_a = all_keypoints_by_type_np[kpt_a_type]
        kpts_b = all_keypoints_by_type_np[kpt_b_type]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            timings["mx_assembly_connections"] += (time.perf_counter() - t0) * 1000.0
            continue

        score_mat = ps[part_id, :n, :m]
        valid_mat = pv[part_id, :n, :m] > 0.5
        valid_mat = valid_mat & (score_mat > float(min_pair_score))

        a_idx, b_idx = np.where(valid_mat)
        if len(a_idx) == 0:
            timings["mx_assembly_connections"] += (time.perf_counter() - t0) * 1000.0
            continue

        affinity_scores = score_mat[a_idx, b_idx].astype(np.float32)
        timings["mx_assembly_connections"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        a_idx, b_idx, affinity_scores = connections_nms(a_idx.astype(np.int32), b_idx.astype(np.int32), affinity_scores)
        timings["mx_assembly_nms"] += (time.perf_counter() - t0) * 1000.0

        connections = list(
            zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores,
            )
        )
        timings["mx_assembly_connections_total"] += len(connections)
        if len(connections) == 0:
            continue

        t0 = time.perf_counter()
        if part_id == 0:
            pose_entries = [
                np.ones(pose_entry_size, dtype=np.float32) * -1
                for _ in range(len(connections))
            ]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = (
                    np.sum(all_keypoints[[connections[i][0], connections[i][1]], 2])
                    + connections[i][2]
                )

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                        pose_entries[j][kpt_a_id] = connections[i][0]

        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                if num == 0:
                    pose_entry = np.ones(pose_entry_size, dtype=np.float32) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = (
                        np.sum(all_keypoints[[connections[i][0], connections[i][1]], 2])
                        + connections[i][2]
                    )
                    pose_entries.append(pose_entry)

        timings["mx_assembly_pose"] += (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    filtered_entries = []
    for entry in pose_entries:
        if entry[-1] < 3:
            continue
        if entry[-2] / entry[-1] < 0.2:
            continue
        filtered_entries.append(entry)
    pose_entries_arr = np.asarray(filtered_entries, dtype=np.float32)
    timings["mx_assembly_filter"] += (time.perf_counter() - t0) * 1000.0

    timings["mx_assembly_total"] = (time.perf_counter() - t_total) * 1000.0

    if return_timing:
        return pose_entries_arr, all_keypoints, timings
    return pose_entries_arr, all_keypoints
