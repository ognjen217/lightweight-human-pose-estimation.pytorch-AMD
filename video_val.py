import cv2
import numpy as np
import time
import os
import torch
from modules.keypoints import (
    extract_keypoints,
    group_keypoints,
    extract_keypoints_batch,
    extract_keypoints_batch_cv2
)
import torch.nn.functional as F
import migraphx


class PoseEstimator:
    def __init__(self, onnx_path, target_dim=(968, 544), stride=8):
        self.w, self.h = target_dim
        self.stride = stride
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cannot find {onnx_path} in {os.getcwd()}")
        self.model = self._load_model(onnx_path)
        self.expected_dtype = str(self.model.get_parameter_shapes()['input'].type())
        
    def _load_model(self, mxr_path):
        return migraphx.load(mxr_path)
        
    def _load_model(self, mxr_path):
        return migraphx.load(mxr_path)

    def _load_model_onnx(self, onnx_path):
        print(f"--- Compiling {onnx_path} ---")
        model = migraphx.parse_onnx(onnx_path)
        migraphx.quantize_fp16(model)
        model.compile(migraphx.get_target("gpu"))
        return model

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = (img.astype(np.float32) - 128) / 256.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        img = np.ascontiguousarray(img)
        if 'half' in self.expected_dtype:
            img = img.astype(np.float16)
        else:
            img = img.astype(np.float32)
        return img

    def postprocess(self, results, original_hw):
        orig_h, orig_w = original_hw
        
        heatmaps = np.array(results[0]).astype(np.float32).reshape(1, 19, self.h//self.stride, self.w//self.stride)
        pafs = np.array(results[1]).astype(np.float32).reshape(1, 38, self.h//self.stride, self.w//self.stride)
        
        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
        pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
        
        heatmaps = cv2.resize(heatmaps, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        all_kpts = []
        total = 0
        for i in range(18):
            total += extract_keypoints(heatmaps[:,:,i], all_kpts, total)

        pose_entries, all_keypoints = group_keypoints(
            all_kpts,
            pafs,
            points_per_limb=8
        )
        return pose_entries, all_keypoints

    def postprocess_fast(self, results, original_hw):
        orig_h, orig_w = original_hw

        out_h = self.h // self.stride
        out_w = self.w // self.stride

        # 1. Decode network outputs
        heatmaps = np.array(results[0]).reshape(1, 19, out_h, out_w)
        pafs     = np.array(results[1]).reshape(1, 38, out_h, out_w)

        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))  # H x W x 19
        pafs     = np.transpose(pafs.squeeze(),     (1, 2, 0)) # H x W x 38

        # 2. Extract keypoints (NO resize here)
        all_kpts = []
        total_kpts = 0

        for kpt_id in range(18):
            num = extract_keypoints(
                heatmaps[:, :, kpt_id],
                all_kpts,
                total_kpts
            )
            total_kpts += num

        # 3. Group keypoints using PAFs (still output resolution)
        pose_entries, all_keypoints = group_keypoints(
            all_kpts,
            pafs,
            points_per_limb=8,
            debug_timing=True
        )

        # 4. Scale keypoints to original image size
        scale_x = orig_w / out_w
        scale_y = orig_h / out_h

        for kpt in all_keypoints:
            kpt[0] *= scale_x  # x
            kpt[1] *= scale_y  # y

        return pose_entries, all_keypoints
    
    def postprocess_optimized(self, results, original_hw):
        timings = {}

        total_start = time.perf_counter()

        orig_h, orig_w = original_hw
        out_h = self.h // self.stride
        out_w = self.w // self.stride

        # -------------------------------------------------
        # 1. Decode / reshape
        # -------------------------------------------------
        t0 = time.perf_counter()

        heatmaps = np.asarray(
            results[0],
            dtype=np.float32
        ).reshape(19, out_h, out_w)

        pafs = np.asarray(
            results[1],
            dtype=np.float32
        ).reshape(38, out_h, out_w)

        heatmaps = np.moveaxis(heatmaps, 0, -1)
        pafs = np.moveaxis(pafs, 0, -1)

        timings["decode"] = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------
        # 2. Resize heatmaps
        # -------------------------------------------------
        t0 = time.perf_counter()

        heatmaps = cv2.resize(
            heatmaps,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )

        heatmaps = np.ascontiguousarray(heatmaps, dtype=np.float32)

        timings["resize_heatmaps"] = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------
        # 3. Resize PAFs
        # -------------------------------------------------
        t0 = time.perf_counter()

        pafs = cv2.resize(
            pafs,
            (orig_w, orig_h),
            interpolation=cv2.INTER_CUBIC
        )

        pafs = np.ascontiguousarray(pafs, dtype=np.float32)

        timings["resize_pafs"] = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------
        # 4. Extract keypoints
        # -------------------------------------------------
        t0 = time.perf_counter()

        all_kpts = []
        total = 0

        all_kpts, total = extract_keypoints_batch_cv2(
            heatmaps[:, :, :18],
            max_keypoints_per_type=20,
        )

        #print("KPTS PER TYPE:", [len(x) for x in all_kpts], "TOTAL:", total)

        timings["extract_keypoints"] = (time.perf_counter() - t0) * 1000

        # -------------------------------------------------
        # 5. Group keypoints
        # -------------------------------------------------
        t0 = time.perf_counter()

        poses, kpts = group_keypoints(
            all_kpts,
            pafs,
            points_per_limb=8
        )

        timings["group_keypoints"] = (time.perf_counter() - t0) * 1000

        timings["total_postprocess"] = (
            time.perf_counter() - total_start
        ) * 1000

        return poses, kpts, timings
    
class Profiler:
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); self.ms = (self.end - self.start) * 1000


def run_benchmarked_session(video_path, model_path):
    engine = PoseEstimator(model_path)
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        'output_pose_benchmarked_2_optimized.mp4',
        fourcc,
        fps,
        (width, height)
    )

    print(
        f"{'Pre':>8} | "
        f"{'Infer':>8} | "
        f"{'Decode':>8} | "
        f"{'HM Resize':>10} | "
        f"{'PAF Resize':>11} | "
        f"{'Extract':>9} | "
        f"{'Group':>8} | "
        f"{'Post Total':>11}"
    )

    print("-" * 95)

    BODY_PARTS_KPT_IDS = [
        [1, 2], [1, 5], [2, 3], [3, 4],
        [5, 6], [6, 7], [1, 8], [8, 9],
        [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
    ]

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------------
        # PREPROCESS
        # ---------------------------------------------
        with Profiler() as pre_p:
            input_tensor = engine.preprocess(frame)

        # ---------------------------------------------
        # INFERENCE
        # ---------------------------------------------
        with Profiler() as infer_p:
            raw_results = engine.model.run({
                'input': input_tensor
            })

        # ---------------------------------------------
        # POSTPROCESS
        # ---------------------------------------------
        pose_entries, all_keypoints, post_times = \
            engine.postprocess_optimized(
                raw_results,
                frame.shape[:2],

            )

        # ---------------------------------------------
        # DRAW
        # ---------------------------------------------
        for pose in pose_entries:

            for part_id in range(len(BODY_PARTS_KPT_IDS)):

                kpt_a_id = pose[
                    BODY_PARTS_KPT_IDS[part_id][0]
                ]

                kpt_b_id = pose[
                    BODY_PARTS_KPT_IDS[part_id][1]
                ]

                if kpt_a_id != -1 and kpt_b_id != -1:

                    kpt_a = all_keypoints[int(kpt_a_id)]
                    kpt_b = all_keypoints[int(kpt_b_id)]

                    cv2.line(
                        frame,
                        (int(kpt_a[0]), int(kpt_a[1])),
                        (int(kpt_b[0]), int(kpt_b[1])),
                        (0, 255, 0),
                        2
                    )

            for kpt_id in pose[:-2]:

                if kpt_id != -1:

                    kpt = all_keypoints[int(kpt_id)]

                    cv2.circle(
                        frame,
                        (int(kpt[0]), int(kpt[1])),
                        3,
                        (0, 255, 0),
                        -1
                    )

        out.write(frame)

        # ---------------------------------------------
        # PRINT
        # ---------------------------------------------
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:

            print(
                f"{pre_p.ms:8.2f} | "
                f"{infer_p.ms:8.2f} | "
                f"{post_times['decode']:8.2f} | "
                f"{post_times['resize_heatmaps']:10.2f} | "
                f"{post_times['resize_pafs']:11.2f} | "
                f"{post_times['extract_keypoints']:9.2f} | "
                f"{post_times['group_keypoints']:8.2f} | "
                f"{post_times['total_postprocess']:11.2f}"
            )

    cap.release()
    out.release()

    print(
        "\n--- Done! Video saved as "
        "output_pose_benchmarked_3_optimized.mp4 ---"
    )

if __name__ == "__main__":
    # Ensure this matches your ACTUAL file name from your 'ls' command
    run_benchmarked_session("cctv_1280x720_24fps_2.mp4", "pose_model1_fp16_ref1.mxr")