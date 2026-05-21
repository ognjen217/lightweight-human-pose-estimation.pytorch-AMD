import math
import numpy as np
from operator import itemgetter
from numba import njit

BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])


def extract_keypoints_batch(heatmaps, max_keypoints_per_type=20):
    import cv2
    import numpy as np

    threshold = 0.1
    nms_radius = 6

    h, w, num_kpts = heatmaps.shape

    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    all_keypoints_by_type = []
    total_keypoints_num = 0

    # cv2.dilate podržava višekanalnu sliku: H x W x C
    heatmaps_dilated = cv2.dilate(heatmaps, kernel)

    peaks_binary = (heatmaps == heatmaps_dilated) & (heatmaps > threshold)

    for kpt_idx in range(num_kpts):
        ys, xs = np.nonzero(peaks_binary[:, :, kpt_idx])

        if len(xs) == 0:
            all_keypoints_by_type.append([])
            continue

        scores = heatmaps[ys, xs, kpt_idx]

        order = np.argsort(scores)[::-1]

        if len(order) > max_keypoints_per_type:
            order = order[:max_keypoints_per_type]

        xs = xs[order]
        ys = ys[order]
        scores = scores[order]

        keypoints_with_score_and_id = []

        for i in range(len(xs)):
            keypoints_with_score_and_id.append(
                (
                    int(xs[i]),
                    int(ys[i]),
                    float(scores[i]),
                    total_keypoints_num + i
                )
            )

        all_keypoints_by_type.append(keypoints_with_score_and_id)
        total_keypoints_num += len(keypoints_with_score_and_id)

    return all_keypoints_by_type, total_keypoints_num


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    import cv2
    import numpy as np

    threshold = 0.1
    nms_radius = 6
    max_keypoints_per_type = 20

    h, w = heatmap.shape

    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    heatmap_dilated = cv2.dilate(heatmap, kernel)

    peaks_binary = (heatmap == heatmap_dilated) & (heatmap > threshold)

    ys, xs = np.nonzero(peaks_binary)

    if len(xs) == 0:
        all_keypoints.append([])
        return 0

    scores = heatmap[ys, xs]

    order = np.argsort(scores)[::-1]

    if len(order) > max_keypoints_per_type:
        order = order[:max_keypoints_per_type]

    xs = xs[order]
    ys = ys[order]
    scores = scores[order]

    keypoints_with_score_and_id = []

    for i in range(len(xs)):
        keypoints_with_score_and_id.append(
            (
                int(xs[i]),
                int(ys[i]),
                float(scores[i]),
                total_keypoint_num + i
            )
        )

    all_keypoints.append(keypoints_with_score_and_id)

    return len(keypoints_with_score_and_id)

def connections_nms(a_idx, b_idx, affinity_scores):
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]

def extract_keypoints_batch_cv2(heatmaps, max_keypoints_per_type=20):
    import cv2
    import numpy as np

    threshold = 0.1
    nms_radius = 6

    h, w, num_kpts = heatmaps.shape

    kernel_size = 2 * nms_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    all_keypoints_by_type = []
    total_keypoints_num = 0

    heatmaps_dilated = cv2.dilate(heatmaps, kernel)

    peaks_binary = (heatmaps == heatmaps_dilated) & (heatmaps > threshold)

    for kpt_idx in range(num_kpts):
        mask = peaks_binary[:, :, kpt_idx].astype(np.uint8)

        pts = cv2.findNonZero(mask)

        if pts is None:
            all_keypoints_by_type.append([])
            continue

        pts = pts.reshape(-1, 2)

        xs = pts[:, 0]
        ys = pts[:, 1]

        scores = heatmaps[ys, xs, kpt_idx]

        order = np.argsort(scores)[::-1]

        if len(order) > max_keypoints_per_type:
            order = order[:max_keypoints_per_type]

        xs = xs[order]
        ys = ys[order]
        scores = scores[order]

        keypoints_with_score_and_id = []

        for i in range(len(xs)):
            keypoints_with_score_and_id.append(
                (
                    int(xs[i]),
                    int(ys[i]),
                    float(scores[i]),
                    total_keypoints_num + i
                )
            )

        all_keypoints_by_type.append(keypoints_with_score_and_id)
        total_keypoints_num += len(keypoints_with_score_and_id)

    return all_keypoints_by_type, total_keypoints_num

def group_keypoints(
    all_keypoints_by_type,
    pafs,
    pose_entry_size=20,
    min_paf_score=0.05,
    points_per_limb=8,
    success_ratio_thr=0.8,
    debug_timing=False
):
    import time
    import numpy as np

    t_total = time.perf_counter()

    tm_prepare = 0.0
    tm_pairs = 0.0
    tm_sample = 0.0
    tm_affinity = 0.0
    tm_nms = 0.0
    tm_pose = 0.0
    tm_filter = 0.0

    total_pairs = 0
    total_valid_limbs = 0
    total_connections = 0

    pose_entries = []

    # -------------------------------------------------
    # 1. Prepare all keypoints
    # -------------------------------------------------
    t0 = time.perf_counter()

    all_keypoints = np.array(
        [item for sublist in all_keypoints_by_type for item in sublist],
        dtype=np.float32
    )

    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    all_keypoints_by_type = [
        np.asarray(keypoints, dtype=np.float32)
        for keypoints in all_keypoints_by_type
    ]

    paf_h, paf_w = pafs.shape[:2]

    tm_prepare += time.perf_counter() - t0

    # -------------------------------------------------
    # 2. Process every body part / limb
    # -------------------------------------------------
    for part_id in range(len(BODY_PARTS_PAF_IDS)):

        t0 = time.perf_counter()

        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]

        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]

        n = len(kpts_a)
        m = len(kpts_b)

        if n == 0 or m == 0:
            tm_prepare += time.perf_counter() - t0
            continue

        total_pairs += n * m

        tm_prepare += time.perf_counter() - t0

        # -------------------------------------------------
        # 3. Create all candidate limb vectors
        # -------------------------------------------------
        t0 = time.perf_counter()

        a = kpts_a[:, :2]  # shape: n x 2
        b = kpts_b[:, :2]  # shape: m x 2

        # shape: m x n x 2 -> flattened to (m*n) x 1 x 2
        vec_raw = (b[:, None, :] - a[None, :, :]).reshape(-1, 1, 2)

        vec_norm = np.linalg.norm(
            vec_raw,
            ord=2,
            axis=-1,
            keepdims=True
        )

        valid_vec = vec_norm.reshape(-1) > 1e-6

        if not np.any(valid_vec):
            tm_pairs += time.perf_counter() - t0
            continue

        pair_ids = np.nonzero(valid_vec)[0]

        vec_raw_valid = vec_raw[valid_vec]
        vec_norm_valid = vec_norm[valid_vec]

        # pair_id = b_idx * n + a_idx
        b_pair_idx, a_pair_idx = np.divmod(pair_ids, n)

        tm_pairs += time.perf_counter() - t0

        # -------------------------------------------------
        # 4. Sample points along each candidate limb
        # -------------------------------------------------
        t0 = time.perf_counter()

        steps = vec_raw_valid / float(points_per_limb - 1)

        a_points = a[a_pair_idx].reshape(-1, 1, 2)

        points = steps * grid + a_points
        points = np.rint(points).astype(np.int32)

        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Safety clipping, da ne izađe van slike
        x = np.clip(x, 0, paf_w - 1)
        y = np.clip(y, 0, paf_h - 1)

        tm_sample += time.perf_counter() - t0

        # -------------------------------------------------
        # 5. Compute PAF affinity score
        # -------------------------------------------------
        t0 = time.perf_counter()

        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)

        vec = vec_raw_valid / (vec_norm_valid + 1e-6)

        affinity_scores_per_point = (field * vec).sum(-1)

        valid_affinity_scores = affinity_scores_per_point > min_paf_score

        valid_num = valid_affinity_scores.sum(axis=1)

        affinity_scores = (
            affinity_scores_per_point * valid_affinity_scores
        ).sum(axis=1) / (valid_num + 1e-6)

        success_ratio = valid_num / float(points_per_limb)

        valid_limb_local = np.where(
            np.logical_and(
                affinity_scores > 0,
                success_ratio > success_ratio_thr
            )
        )[0]

        total_valid_limbs += len(valid_limb_local)

        if len(valid_limb_local) == 0:
            tm_affinity += time.perf_counter() - t0
            continue

        valid_limbs = pair_ids[valid_limb_local]

        b_idx, a_idx = np.divmod(valid_limbs, n)

        affinity_scores = affinity_scores[valid_limb_local]

        tm_affinity += time.perf_counter() - t0

        # -------------------------------------------------
        # 6. NMS over candidate connections
        # -------------------------------------------------
        t0 = time.perf_counter()

        a_idx, b_idx, affinity_scores = connections_nms(
            a_idx,
            b_idx,
            affinity_scores
        )

        connections = list(
            zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores
            )
        )

        total_connections += len(connections)

        if len(connections) == 0:
            tm_nms += time.perf_counter() - t0
            continue

        tm_nms += time.perf_counter() - t0

        # -------------------------------------------------
        # 7. Assemble pose entries
        # -------------------------------------------------
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
                    np.sum(
                        all_keypoints[
                            [connections[i][0], connections[i][1]],
                            2
                        ]
                    )
                    + connections[i][2]
                )

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            for i in range(len(connections)):
                for j in range(len(pose_entries)):

                    if (
                        pose_entries[j][kpt_a_id] == connections[i][0]
                        and pose_entries[j][kpt_b_id] == -1
                    ):
                        pose_entries[j][kpt_b_id] = connections[i][1]

                    elif (
                        pose_entries[j][kpt_b_id] == connections[i][1]
                        and pose_entries[j][kpt_a_id] == -1
                    ):
                        pose_entries[j][kpt_a_id] = connections[i][0]

            tm_pose += time.perf_counter() - t0
            continue

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

                        pose_entries[j][-2] += (
                            all_keypoints[connections[i][1], 2]
                            + connections[i][2]
                        )

                if num == 0:
                    pose_entry = np.ones(
                        pose_entry_size,
                        dtype=np.float32
                    ) * -1

                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]

                    pose_entry[-1] = 2

                    pose_entry[-2] = (
                        np.sum(
                            all_keypoints[
                                [connections[i][0], connections[i][1]],
                                2
                            ]
                        )
                        + connections[i][2]
                    )

                    pose_entries.append(pose_entry)

        tm_pose += time.perf_counter() - t0

    # -------------------------------------------------
    # 8. Filter weak pose entries
    # -------------------------------------------------
    t0 = time.perf_counter()

    filtered_entries = []

    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3:
            continue

        if pose_entries[i][-2] / pose_entries[i][-1] < 0.2:
            continue

        filtered_entries.append(pose_entries[i])

    pose_entries = np.asarray(filtered_entries, dtype=np.float32)

    tm_filter += time.perf_counter() - t0

    # -------------------------------------------------
    # 9. Optional debug timing
    # -------------------------------------------------
    if debug_timing:
        total_ms = (time.perf_counter() - t_total) * 1000.0

        print(
            f"[group] total={total_ms:.2f} ms | "
            f"prepare={tm_prepare * 1000:.2f} | "
            f"pairs={tm_pairs * 1000:.2f} | "
            f"sample={tm_sample * 1000:.2f} | "
            f"affinity={tm_affinity * 1000:.2f} | "
            f"nms={tm_nms * 1000:.2f} | "
            f"pose={tm_pose * 1000:.2f} | "
            f"filter={tm_filter * 1000:.2f} | "
            f"pairs_total={total_pairs} | "
            f"valid_limbs={total_valid_limbs} | "
            f"connections={total_connections}"
        )

    return pose_entries, all_keypoints

def group_keypoints_fast(
    all_keypoints_by_type,
    pafs,
    pose_entry_size=20,
    min_paf_score=0.05,
    points_per_limb=8,
    success_ratio_thr=0.8,
    debug_timing=False,
    return_timing=False
):
    import time
    import numpy as np

    t_total = time.perf_counter()

    tm_prepare = 0.0
    tm_pairs = 0.0
    tm_sample = 0.0
    tm_affinity = 0.0
    tm_nms = 0.0
    tm_pose = 0.0
    tm_filter = 0.0

    total_pairs = 0
    total_valid_limbs = 0
    total_connections = 0

    pose_entries = []

    # -------------------------------------------------
    # 1. Prepare all keypoints
    # -------------------------------------------------
    t0 = time.perf_counter()

    all_keypoints_by_type = [
        np.asarray(keypoints, dtype=np.float32)
        for keypoints in all_keypoints_by_type
    ]

    non_empty_keypoints = [
        keypoints
        for keypoints in all_keypoints_by_type
        if len(keypoints) > 0
    ]

    if non_empty_keypoints:
        all_keypoints = np.concatenate(non_empty_keypoints, axis=0)
    else:
        all_keypoints = np.empty((0, 4), dtype=np.float32)

    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    paf_h, paf_w = pafs.shape[:2]

    tm_prepare += time.perf_counter() - t0

    # -------------------------------------------------
    # 2. Process every body part / limb
    # -------------------------------------------------
    for part_id in range(len(BODY_PARTS_PAF_IDS)):

        # -------------------------------------------------
        # 2.1 Prepare only indices, do NOT slice full PAF map
        # -------------------------------------------------
        t0 = time.perf_counter()

        paf_x_id, paf_y_id = BODY_PARTS_PAF_IDS[part_id]

        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]

        n = len(kpts_a)
        m = len(kpts_b)

        if n == 0 or m == 0:
            tm_prepare += time.perf_counter() - t0
            continue

        total_pairs += n * m

        tm_prepare += time.perf_counter() - t0

        # -------------------------------------------------
        # 3. Create all candidate limb vectors
        # -------------------------------------------------
        t0 = time.perf_counter()

        a = kpts_a[:, :2]
        b = kpts_b[:, :2]

        vec_raw = (b[:, None, :] - a[None, :, :]).reshape(-1, 1, 2)

        vec_norm = np.linalg.norm(
            vec_raw,
            ord=2,
            axis=-1,
            keepdims=True
        )

        valid_vec = vec_norm.reshape(-1) > 1e-6

        if not np.any(valid_vec):
            tm_pairs += time.perf_counter() - t0
            continue

        pair_ids = np.nonzero(valid_vec)[0]

        vec_raw_valid = vec_raw[valid_vec]
        vec_norm_valid = vec_norm[valid_vec]

        b_pair_idx, a_pair_idx = np.divmod(pair_ids, n)

        tm_pairs += time.perf_counter() - t0

        # -------------------------------------------------
        # 4. Sample points along each candidate limb
        # -------------------------------------------------
        t0 = time.perf_counter()

        steps = vec_raw_valid / float(points_per_limb - 1)

        a_points = a[a_pair_idx].reshape(-1, 1, 2)

        points = steps * grid + a_points
        points = np.rint(points).astype(np.int32)

        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        x = np.clip(x, 0, paf_w - 1)
        y = np.clip(y, 0, paf_h - 1)

        tm_sample += time.perf_counter() - t0

        # -------------------------------------------------
        # 5. Compute PAF affinity score
        # -------------------------------------------------
        t0 = time.perf_counter()

        # OLD SLOW VERSION:
        # part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        # field = part_pafs[y, x].reshape(-1, points_per_limb, 2)

        # NEW FASTER VERSION:
        # Read only sampled PAF values directly from original pafs.
        # This avoids copying full H x W x 2 PAF map for every limb.
        field = np.empty((x.shape[0], 2), dtype=np.float32)

        field[:, 0] = pafs[y, x, paf_x_id]
        field[:, 1] = pafs[y, x, paf_y_id]

        field = field.reshape(-1, points_per_limb, 2)

        vec = vec_raw_valid / (vec_norm_valid + 1e-6)

        affinity_scores_per_point = (field * vec).sum(-1)

        valid_affinity_scores = affinity_scores_per_point > min_paf_score

        valid_num = valid_affinity_scores.sum(axis=1)

        affinity_scores = (
            affinity_scores_per_point * valid_affinity_scores
        ).sum(axis=1) / (valid_num + 1e-6)

        success_ratio = valid_num / float(points_per_limb)

        valid_limb_local = np.where(
            np.logical_and(
                affinity_scores > 0,
                success_ratio > success_ratio_thr
            )
        )[0]

        total_valid_limbs += len(valid_limb_local)

        if len(valid_limb_local) == 0:
            tm_affinity += time.perf_counter() - t0
            continue

        valid_limbs = pair_ids[valid_limb_local]

        b_idx, a_idx = np.divmod(valid_limbs, n)

        affinity_scores = affinity_scores[valid_limb_local]

        tm_affinity += time.perf_counter() - t0

        # -------------------------------------------------
        # 6. NMS over candidate connections
        # -------------------------------------------------
        t0 = time.perf_counter()

        a_idx, b_idx, affinity_scores = connections_nms(
            a_idx,
            b_idx,
            affinity_scores
        )

        connections = list(
            zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores
            )
        )

        total_connections += len(connections)

        if len(connections) == 0:
            tm_nms += time.perf_counter() - t0
            continue

        tm_nms += time.perf_counter() - t0

        # -------------------------------------------------
        # 7. Assemble pose entries
        # -------------------------------------------------
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
                    np.sum(
                        all_keypoints[
                            [connections[i][0], connections[i][1]],
                            2
                        ]
                    )
                    + connections[i][2]
                )

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            for i in range(len(connections)):
                for j in range(len(pose_entries)):

                    if (
                        pose_entries[j][kpt_a_id] == connections[i][0]
                        and pose_entries[j][kpt_b_id] == -1
                    ):
                        pose_entries[j][kpt_b_id] = connections[i][1]

                    elif (
                        pose_entries[j][kpt_b_id] == connections[i][1]
                        and pose_entries[j][kpt_a_id] == -1
                    ):
                        pose_entries[j][kpt_a_id] = connections[i][0]

            tm_pose += time.perf_counter() - t0
            continue

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

                        pose_entries[j][-2] += (
                            all_keypoints[connections[i][1], 2]
                            + connections[i][2]
                        )

                if num == 0:
                    pose_entry = np.ones(
                        pose_entry_size,
                        dtype=np.float32
                    ) * -1

                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]

                    pose_entry[-1] = 2

                    pose_entry[-2] = (
                        np.sum(
                            all_keypoints[
                                [connections[i][0], connections[i][1]],
                                2
                            ]
                        )
                        + connections[i][2]
                    )

                    pose_entries.append(pose_entry)

        tm_pose += time.perf_counter() - t0

    # -------------------------------------------------
    # 8. Filter weak pose entries
    # -------------------------------------------------
    t0 = time.perf_counter()

    filtered_entries = []

    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3:
            continue

        if pose_entries[i][-2] / pose_entries[i][-1] < 0.2:
            continue

        filtered_entries.append(pose_entries[i])

    pose_entries = np.asarray(filtered_entries, dtype=np.float32)

    tm_filter += time.perf_counter() - t0

    # -------------------------------------------------
    # 9. Timing summary
    # -------------------------------------------------
    total_ms = (time.perf_counter() - t_total) * 1000.0

    group_times = {
        "group_total": total_ms,
        "group_prepare": tm_prepare * 1000.0,
        "group_pairs": tm_pairs * 1000.0,
        "group_sample": tm_sample * 1000.0,
        "group_affinity": tm_affinity * 1000.0,
        "group_nms": tm_nms * 1000.0,
        "group_pose": tm_pose * 1000.0,
        "group_filter": tm_filter * 1000.0,
        "group_pairs_total": total_pairs,
        "group_valid_limbs": total_valid_limbs,
        "group_connections": total_connections,
    }

    if debug_timing:
        print(
            f"[group] total={group_times['group_total']:.2f} ms | "
            f"prepare={group_times['group_prepare']:.2f} | "
            f"pairs={group_times['group_pairs']:.2f} | "
            f"sample={group_times['group_sample']:.2f} | "
            f"affinity={group_times['group_affinity']:.2f} | "
            f"nms={group_times['group_nms']:.2f} | "
            f"pose={group_times['group_pose']:.2f} | "
            f"filter={group_times['group_filter']:.2f} | "
            f"pairs_total={group_times['group_pairs_total']} | "
            f"valid_limbs={group_times['group_valid_limbs']} | "
            f"connections={group_times['group_connections']}"
        )

    if return_timing:
        return pose_entries, all_keypoints, group_times

    return pose_entries, all_keypoints

@njit(cache=True)
def _score_paf_connections_numba(
    kpts_a,
    kpts_b,
    part_pafs,
    points_per_limb,
    min_paf_score,
    success_ratio_thr
):
    n = kpts_a.shape[0]
    m = kpts_b.shape[0]

    paf_h = part_pafs.shape[0]
    paf_w = part_pafs.shape[1]

    max_pairs = n * m

    out_a_idx = np.empty(max_pairs, dtype=np.int32)
    out_b_idx = np.empty(max_pairs, dtype=np.int32)
    out_scores = np.empty(max_pairs, dtype=np.float32)

    count = 0

    for b_i in range(m):
        bx = kpts_b[b_i, 0]
        by = kpts_b[b_i, 1]

        for a_i in range(n):
            ax = kpts_a[a_i, 0]
            ay = kpts_a[a_i, 1]

            dx = bx - ax
            dy = by - ay

            norm = (dx * dx + dy * dy) ** 0.5

            if norm <= 1e-6:
                continue

            vx = dx / (norm + 1e-6)
            vy = dy / (norm + 1e-6)

            valid_num = 0
            score_sum = 0.0

            for p in range(points_per_limb):
                alpha = p / (points_per_limb - 1)

                x = int(np.rint(ax + dx * alpha))
                y = int(np.rint(ay + dy * alpha))

                if x < 0:
                    x = 0
                elif x >= paf_w:
                    x = paf_w - 1

                if y < 0:
                    y = 0
                elif y >= paf_h:
                    y = paf_h - 1

                paf_x = part_pafs[y, x, 0]
                paf_y = part_pafs[y, x, 1]

                paf_score = paf_x * vx + paf_y * vy

                if paf_score > min_paf_score:
                    valid_num += 1
                    score_sum += paf_score

            affinity_score = score_sum / (valid_num + 1e-6)
            success_ratio = valid_num / points_per_limb

            if affinity_score > 0.0 and success_ratio > success_ratio_thr:
                out_a_idx[count] = a_i
                out_b_idx[count] = b_i
                out_scores[count] = affinity_score
                count += 1

    return out_a_idx[:count], out_b_idx[:count], out_scores[:count]


def group_keypoints_numba(
    all_keypoints_by_type,
    pafs,
    pose_entry_size=20,
    min_paf_score=0.05,
    points_per_limb=8,
    success_ratio_thr=0.8
):
    pose_entries = []

    all_keypoints = np.array(
        [item for sublist in all_keypoints_by_type for item in sublist],
        dtype=np.float32
    )

    all_keypoints_by_type = [
        np.asarray(keypoints, dtype=np.float32)
        for keypoints in all_keypoints_by_type
    ]

    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]

        n = len(kpts_a)
        m = len(kpts_b)

        if n == 0 or m == 0:
            continue

        a_idx, b_idx, affinity_scores = _score_paf_connections_numba(
            kpts_a,
            kpts_b,
            part_pafs,
            points_per_limb,
            min_paf_score,
            success_ratio_thr
        )

        if len(a_idx) == 0:
            continue

        a_idx, b_idx, affinity_scores = connections_nms(
            a_idx,
            b_idx,
            affinity_scores
        )

        connections = list(
            zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores
            )
        )

        if len(connections) == 0:
            continue

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
                    np.sum(
                        all_keypoints[
                            [connections[i][0], connections[i][1]],
                            2
                        ]
                    )
                    + connections[i][2]
                )

        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if (
                        pose_entries[j][kpt_a_id] == connections[i][0]
                        and pose_entries[j][kpt_b_id] == -1
                    ):
                        pose_entries[j][kpt_b_id] = connections[i][1]

                    elif (
                        pose_entries[j][kpt_b_id] == connections[i][1]
                        and pose_entries[j][kpt_a_id] == -1
                    ):
                        pose_entries[j][kpt_a_id] = connections[i][0]

            continue

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

                        pose_entries[j][-2] += (
                            all_keypoints[connections[i][1], 2]
                            + connections[i][2]
                        )

                if num == 0:
                    pose_entry = np.ones(
                        pose_entry_size,
                        dtype=np.float32
                    ) * -1

                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]

                    pose_entry[-1] = 2

                    pose_entry[-2] = (
                        np.sum(
                            all_keypoints[
                                [connections[i][0], connections[i][1]],
                                2
                            ]
                        )
                        + connections[i][2]
                    )

                    pose_entries.append(pose_entry)

    filtered_entries = []

    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3:
            continue

        if pose_entries[i][-2] / pose_entries[i][-1] < 0.2:
            continue

        filtered_entries.append(pose_entries[i])

    pose_entries = np.asarray(filtered_entries, dtype=np.float32)

    return pose_entries, all_keypoints

def extract_keypoints_from_peak_mask(
    heatmaps,
    peak_mask,
    max_candidates_per_part=None,
    num_keypoint_types=18,
):
    """Convert a dense peak mask into OpenPose-style keypoint candidates.

    Parameters
    ----------
    heatmaps : np.ndarray
        Heatmaps in either HWC or NCHW/NHWC batch format. Scores are read from
        this tensor.
    peak_mask : np.ndarray
        Dense peak mask matching heatmaps spatially. Non-zero values are treated
        as selected local maxima.
    max_candidates_per_part : Optional[int]
        If set, keep only the highest-scoring K candidates per keypoint type.
        Use 20 for compatibility with the existing optimized extraction path.
    num_keypoint_types : int
        Number of human keypoint channels to convert. Defaults to 18, leaving
        the background channel unused when C=19.

    Returns
    -------
    tuple[list[list[tuple]], int]
        Same candidate format as extract_keypoints_batch_cv2(): per-type lists
        of (x, y, score, global_id), plus total candidate count.
    """
    import numpy as np

    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    peak_mask = np.asarray(peak_mask)

    # Normalize common layouts to H x W x C.
    if heatmaps.ndim == 4:
        if heatmaps.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for heatmaps, got {heatmaps.shape}")
        # NCHW -> HWC when channel dimension is second.
        if heatmaps.shape[1] <= 64:
            heatmaps = np.moveaxis(heatmaps[0], 0, -1)
        else:  # NHWC
            heatmaps = heatmaps[0]
    elif heatmaps.ndim != 3:
        raise ValueError(f"Expected heatmaps as HWC or batched 4D tensor, got {heatmaps.shape}")

    if peak_mask.ndim == 4:
        if peak_mask.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for peak_mask, got {peak_mask.shape}")
        if peak_mask.shape[1] <= 64:
            peak_mask = np.moveaxis(peak_mask[0], 0, -1)
        else:
            peak_mask = peak_mask[0]
    elif peak_mask.ndim != 3:
        raise ValueError(f"Expected peak_mask as HWC or batched 4D tensor, got {peak_mask.shape}")

    if heatmaps.shape[:2] != peak_mask.shape[:2]:
        raise ValueError(
            f"Spatial shape mismatch: heatmaps {heatmaps.shape}, peak_mask {peak_mask.shape}"
        )

    channels = min(int(num_keypoint_types), heatmaps.shape[2], peak_mask.shape[2])
    all_keypoints_by_type = []
    total_keypoints_num = 0

    for kpt_idx in range(channels):
        ys, xs = np.nonzero(peak_mask[:, :, kpt_idx] > 0)

        if len(xs) == 0:
            all_keypoints_by_type.append([])
            continue

        scores = heatmaps[ys, xs, kpt_idx]
        order = np.argsort(scores)[::-1]

        if max_candidates_per_part is not None and len(order) > max_candidates_per_part:
            order = order[:max_candidates_per_part]

        xs = xs[order]
        ys = ys[order]
        scores = scores[order]

        keypoints = []
        for i in range(len(xs)):
            keypoints.append(
                (
                    int(xs[i]),
                    int(ys[i]),
                    float(scores[i]),
                    total_keypoints_num + i,
                )
            )

        all_keypoints_by_type.append(keypoints)
        total_keypoints_num += len(keypoints)

    # Preserve the expected 18-list structure even if a smaller tensor is tested.
    while len(all_keypoints_by_type) < int(num_keypoint_types):
        all_keypoints_by_type.append([])

    return all_keypoints_by_type, total_keypoints_num
