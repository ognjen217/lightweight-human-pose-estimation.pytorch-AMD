import cv2
import numpy as np
import time
import os
import torch
from modules.keypoints import extract_keypoints, group_keypoints
import torch.nn.functional as F
import migraphx

class PoseEstimator:
    def __init__(self, onnx_path, target_dim=(968, 544), stride=8):
        self.w, self.h = target_dim
        self.stride = stride
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Cannot find {onnx_path} in {os.getcwd()}")
        self.model = self._load_model(onnx_path)
        
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
        return np.ascontiguousarray(img).astype(np.float32)

    def postprocess(self, results, original_hw):
        orig_h, orig_w = original_hw
        
        heatmaps = np.array(results[0]).reshape(1, 19, self.h//self.stride, self.w//self.stride)
        pafs = np.array(results[1]).reshape(1, 38, self.h//self.stride, self.w//self.stride)
        
        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
        pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
        
        heatmaps = cv2.resize(heatmaps, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        all_kpts = []
        total = 0
        for i in range(18):
            total += extract_keypoints(heatmaps[:,:,i], all_kpts, total)

        poses, kpts = group_keypoints(all_kpts, pafs)
        return poses, kpts

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
        pose_entries, all_kpts = group_keypoints(all_kpts, pafs)

        # 4. Scale keypoints to original image size
        scale_x = orig_w / out_w
        scale_y = orig_h / out_h

        for kpt in all_kpts:
            kpt[0] *= scale_x  # x
            kpt[1] *= scale_y  # y

        return pose_entries, all_kpts

class Profiler:
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *args): self.end = time.perf_counter(); self.ms = (self.end - self.start) * 1000

def run_benchmarked_session(video_path, model_path):
    engine = PoseEstimator(model_path)
    cap = cv2.VideoCapture(video_path)
    # SETUP VIDEO WRITER
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_pose_benchmarked.mp4', fourcc, fps, (width, height))

    print(f"{'Pre (ms)':<10} | {'Inference (ms)':<15} | {'Post (ms)':<10}")
    print("-" * 45)

    BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        with Profiler() as pre_p:
            input_tensor = engine.preprocess(frame)

        with Profiler() as infer_p:
            raw_results = engine.model.run({'input': input_tensor})

        with Profiler() as post_p:
            pose_entries, all_keypoints = engine.postprocess(raw_results, frame.shape[:2])
            
            # DRAWING (Keep this inside post-process timing or separate it)
            for pose in pose_entries:
                for part_id in range(len(BODY_PARTS_KPT_IDS)):
                    kpt_a_id = pose[BODY_PARTS_KPT_IDS[part_id][0]]
                    kpt_b_id = pose[BODY_PARTS_KPT_IDS[part_id][1]]
                    if kpt_a_id != -1 and kpt_b_id != -1:
                        kpt_a, kpt_b = all_keypoints[int(kpt_a_id)], all_keypoints[int(kpt_b_id)]
                        cv2.line(frame, (int(kpt_a[0]), int(kpt_a[1])), (int(kpt_b[0]), int(kpt_b[1])), (0, 255, 0), 2)
                for kpt_id in pose[:-2]:
                    if kpt_id != -1:
                        kpt = all_keypoints[int(kpt_id)]
                        cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)

        out.write(frame)

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
            print(f"{pre_p.ms:>8.2f} | {infer_p.ms:>13.2f} | {post_p.ms:>8.2f}")

    cap.release()
    out.release()
    print("\n--- Done! Video saved as output_pose_benchmarked.mp4 ---")

if __name__ == "__main__":
    # Ensure this matches your ACTUAL file name from your 'ls' command
    run_benchmarked_session("cctv_1280x720_24fps.mp4", "pose_model1_fp16_ref1.mxr")