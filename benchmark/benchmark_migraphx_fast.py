import torch
import time
import subprocess
import numpy as np
import json
import cv2
import torch.profiler
import csv
import argparse
import os
import math

import migraphx

from datasets.coco import CocoValDataset
from modules.keypoints import extract_keypoints, group_keypoints
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


FINAL_RESULTS = []

base_height = 544
base_width = 968
stride = 8


class MIGraphXWrapper:
    def __init__(self, model):
        self.model = model
        self.input_name = "input"
        self.expected_type = str(
            self.model.get_parameter_shapes()[self.input_name].type()
        )
        print(f"DEBUG: MIGraphX expects input type: {self.expected_type}")

    def eval(self):
        return self

    def __call__(self, x):
        n_input = x.cpu().numpy()

        if "half" in self.expected_type:
            n_input = n_input.astype(np.float16)
        elif "float" in self.expected_type:
            n_input = n_input.astype(np.float32)

        n_input = np.ascontiguousarray(n_input)

        results = self.model.run({
            self.input_name: n_input
        })

        return [
            torch.from_numpy(np.array(res)).float()
            for res in results
        ]


def normalize(img, img_mean=(128, 128, 128), img_scale=1 / 256):
    img = np.asarray(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape

    h = min(min_dims[0], h)

    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride

    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))  # top
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))  # left
    pad.append(int(min_dims[0] - h - pad[0]))             # bottom
    pad.append(int(min_dims[1] - w - pad[1]))             # right

    padded_img = cv2.copyMakeBorder(
        img,
        pad[0],
        pad[2],
        pad[1],
        pad[3],
        cv2.BORDER_CONSTANT,
        value=pad_value
    )

    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []

    to_coco_map = [
        0, -1, 6, 8, 10, 5, 7, 9,
        12, 14, 16, 11, 13, 15, 2, 1, 4, 3
    ]

    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue

        keypoints = [0] * 17 * 3

        person_score = pose_entries[n][-2]
        position_id = -1

        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1

            # COCO does not have neck keypoint
            if position_id == 1:
                continue

            cx, cy, score, visibility = 0, 0, 0, 0

            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1

            coco_id = to_coco_map[position_id]

            keypoints[coco_id * 3 + 0] = cx
            keypoints[coco_id * 3 + 1] = cy
            keypoints[coco_id * 3 + 2] = visibility

        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, pose_entries[n][-1] - 1))

    return coco_keypoints, scores


def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = "keypoints"

    print(f"Running test for {annotation_type} results.")

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def infer_fast(
    net,
    img,
    base_height=544,
    base_width=968,
    stride=8,
    quantization_type="fp16",
    pad_value=(0, 0, 0),
    img_mean=(128, 128, 128),
    img_scale=1 / 256
):
    """
    This replaces the standard val.py infer().

    Difference:
    - It does NOT resize heatmaps and PAFs back to image resolution.
    - It returns low-resolution network outputs directly.
    """

    normed_img = normalize(img, img_mean, img_scale)

    orig_h, orig_w, _ = normed_img.shape

    ratio = min(
        base_height / orig_h,
        base_width / orig_w
    )

    scaled_img = cv2.resize(
        normed_img,
        (0, 0),
        fx=ratio,
        fy=ratio,
        interpolation=cv2.INTER_LINEAR
    )

    scaled_h, scaled_w, _ = scaled_img.shape

    min_dims = [base_height, base_width]

    padded_img, pad = pad_width(
        scaled_img,
        stride,
        pad_value,
        min_dims
    )

    if quantization_type in ["int8", "mixed_fp32", "mixed_fp16"]:
        device = torch.device("cpu")
        input_dtype = torch.float32

        if quantization_type == "mixed_fp16":
            input_dtype = torch.float16
    else:
        device = torch.device("cuda")

        if quantization_type == "fp16":
            input_dtype = torch.float16
        elif quantization_type == "bf16":
            input_dtype = torch.bfloat16
        else:
            input_dtype = torch.float32

    tensor_img = torch.from_numpy(padded_img)
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
    tensor_img = tensor_img.to(device).to(input_dtype)

    with torch.no_grad():
        stages_output = net(tensor_img)

    stage_heatmaps = stages_output[-2]
    stage_pafs = stages_output[-1]

    heatmaps = stage_heatmaps.squeeze().float().cpu().numpy()
    pafs = stage_pafs.squeeze().float().cpu().numpy()

    heatmaps = np.transpose(heatmaps, (1, 2, 0))  # H x W x 19
    pafs = np.transpose(pafs, (1, 2, 0))          # H x W x 38

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "scaled_h": scaled_h,
        "scaled_w": scaled_w,
        "pad": pad,
        "stride": stride
    }

    return heatmaps, pafs, meta


def postprocess_fast_coco(heatmaps, pafs, meta):
    """
    Fast postprocessing:

    - extract_keypoints() is done on low-resolution heatmaps
    - group_keypoints() is done on low-resolution PAFs
    - final keypoint coordinates are scaled back to original image space

    This is the same idea as your PoseEstimator.postprocess_fast().
    """

    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    scaled_h = meta["scaled_h"]
    scaled_w = meta["scaled_w"]

    pad_top, pad_left, pad_bottom, pad_right = meta["pad"]
    stride = meta["stride"]

    all_keypoints_by_type = []
    total_keypoints_num = 0

    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx],
            all_keypoints_by_type,
            total_keypoints_num
        )

    pose_entries, all_keypoints = group_keypoints(
        all_keypoints_by_type,
        pafs
    )

    if len(all_keypoints) > 0:
        scale_x = orig_w / scaled_w
        scale_y = orig_h / scaled_h

        for kpt in all_keypoints:
            # low-res feature coordinate -> padded input pixel coordinate
            x_padded = kpt[0] * stride
            y_padded = kpt[1] * stride

            # remove padding
            x_scaled = x_padded - pad_left
            y_scaled = y_padded - pad_top

            # scaled input image coordinate -> original image coordinate
            kpt[0] = x_scaled * scale_x
            kpt[1] = y_scaled * scale_y

    return pose_entries, all_keypoints


def evaluate_fast(
    labels,
    output_name,
    images_folder,
    net,
    multiscale=False,
    visualize=False,
    quantization_type="fp16"
):
    """
    COCO evaluation using fast postprocessing.
    """

    net = net.eval()

    if multiscale:
        print("WARNING: fast postprocessing version currently uses single-scale only.")

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []

    for i, sample in enumerate(dataset):
        file_name = sample["file_name"]
        img = sample["img"]

        if i % 20 == 0:
            print(f"Processing image {i + 1}/{len(dataset)}: {file_name}")

        heatmaps, pafs, meta = infer_fast(
            net=net,
            img=img,
            base_height=base_height,
            base_width=base_width,
            stride=stride,
            quantization_type=quantization_type
        )

        pose_entries, all_keypoints = postprocess_fast_coco(
            heatmaps,
            pafs,
            meta
        )

        coco_keypoints, scores = convert_to_coco_format(
            pose_entries,
            all_keypoints
        )

        image_id = int(file_name[0:file_name.rfind(".")])

        for idx in range(len(coco_keypoints)):
            coco_result.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": coco_keypoints[idx],
                "score": scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    x = int(keypoints[idx * 3])
                    y = int(keypoints[idx * 3 + 1])

                    if x > 0 and y > 0:
                        cv2.circle(
                            img,
                            (x, y),
                            3,
                            (255, 0, 255),
                            -1
                        )

            cv2.imshow("fast keypoints", img)
            key = cv2.waitKey()

            if key == 27:
                return

    print(f"\n--- Fast inference complete. Writing results to {output_name}... ---")

    with open(output_name, "w") as f:
        json.dump(coco_result, f, indent=4)

    print("--- Starting COCO Evaluation for FAST postprocessing ---")
    run_coco_eval(labels, output_name)
    print("--- Evaluation Finished! ---")


def load_model(args, device):
    onnx_path = "models/fp16_refinment1.onnx"

    compiled_model_path = (
        f"pose_model1_{args.quantization}_ref{args.num_refinement_stages}.mxr"
    )

    if os.path.exists(compiled_model_path):
        print(f"--- Loading pre-compiled model from {compiled_model_path} ---")
        return migraphx.load(compiled_model_path)

    print(f"--- Compiled model not found. Compiling {onnx_path} ---")

    model = migraphx.parse_onnx(onnx_path)
    target = migraphx.get_target("gpu")

    if args.quantization == "fp16":
        migraphx.quantize_fp16(model)
    elif args.quantization == "int8":
        migraphx.quantize_int8(model, target, [])
    elif args.quantization == "bf16":
        migraphx.quantize_bf16(model)

    model.compile(target, exhaustive_tune=True)

    print(f"--- Saving compiled model to {compiled_model_path} ---")
    migraphx.save(model, compiled_model_path)

    return model


def get_gpu_power():
    try:
        res = subprocess.check_output(
            ["rocm-smi", "--showpower", "--json"]
        ).decode("utf-8")

        data = json.loads(res)

        power_str = data["card0"]["Current Socket Graphics Package Power (W)"]
        return float(power_str)

    except Exception:
        return 0.0


def run_inference(model, tensor_input):
    n_input = tensor_input.detach().cpu().numpy()

    param_type = str(
        model.get_parameter_shapes()["input"].type()
    )

    if "half" in param_type:
        n_input = n_input.astype(np.float16)
    elif "bfloat" in param_type:
        n_input = n_input.astype(np.float32)
    elif "float" in param_type:
        n_input = n_input.astype(np.float32)

    n_input = np.ascontiguousarray(n_input)

    return model.run({
        "input": n_input
    })


def benchmark(
    args,
    iterations=100,
    warm_up=20,
    profiler=False,
    validate_accuracy=True
):
    device = torch.device("cuda")

    print(f"\n--- Benchmarking: {args.quantization.upper()} on {device} ---")

    net = load_model(args, device)

    param_type = str(net.get_parameter_shapes()["input"].type())

    if "half" in param_type:
        input_dtype = torch.float16
    elif "bfloat" in param_type:
        input_dtype = torch.bfloat16
    else:
        input_dtype = torch.float32

    dummy_input = torch.randn(
        1,
        3,
        base_height,
        base_width,
        dtype=input_dtype
    ).to("cuda")

    print("Warming up...")

    with torch.inference_mode():
        for _ in range(warm_up):
            _ = run_inference(net, dummy_input)
            torch.cuda.synchronize()

    if profiler:
        print("\n--- Running Profiler ---")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=5,
                active=5
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./log/pose_estimation"
            ),
            record_shapes=True,
            with_stack=True
        ) as prof:

            with torch.inference_mode():
                for _ in range(11):
                    _ = run_inference(net, dummy_input)
                    prof.step()

        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            )
        )

    if validate_accuracy:
        print("\n--- Starting Accuracy Validation with FAST postprocessing ---")

        wrapped_model = MIGraphXWrapper(net)

        output_json = (
            f"results_{args.quantization}_ref"
            f"{args.num_refinement_stages}_fast.json"
        )

        try:
            evaluate_fast(
                labels=args.labels,
                output_name=output_json,
                images_folder=args.images_folder,
                net=wrapped_model,
                quantization_type=args.quantization
            )

        except Exception as e:
            print(f"EVALUATION FAILED: {e}")

    input_array = dummy_input.detach().cpu().numpy()

    if input_array.dtype == np.float32 and "half" in param_type:
        input_array = input_array.astype(np.float16)

    input_array = np.ascontiguousarray(input_array)

    input_arg = migraphx.argument(input_array)

    run_params = {
        "input": input_arg
    }

    latencies = []
    powers = []

    start_benchmark = time.perf_counter()

    for i in range(iterations):
        iter_start = time.perf_counter()

        net.run(run_params)

        torch.cuda.synchronize()

        latencies.append(time.perf_counter() - iter_start)

        if i % 10 == 0:
            powers.append(get_gpu_power())

    total_bench_time = time.perf_counter() - start_benchmark

    avg_latency = np.mean(latencies) * 1000
    fps = 1.0 / np.mean(latencies)
    avg_power = np.mean(powers) if powers else 0

    res = {
        "Stages": args.num_refinement_stages,
        "Mode": args.quantization,
        "Latency (ms)": f"{avg_latency:.2f}",
        "Throughput (FPS)": f"{fps:.2f}",
        "Power (W)": f"{avg_power:.2f}" if device.type == "cuda" else "N/A",
        "Postprocess": "fast"
    }

    FINAL_RESULTS.append(res)

    print(f"DONE: {fps:.2f} FPS")
    print(f"Results for {args.quantization}:")
    print(f" - Avg Latency: {avg_latency:.2f} ms")
    print(f" - Throughput: {fps:.2f} FPS")

    if device.type == "cuda":
        print(f" - Avg Power: {avg_power:.2f} W")

        efficiency = fps / avg_power if avg_power > 0 else 0
        print(f" - Efficiency: {efficiency:.2f} FPS/Watt")
    else:
        print(" - Note: Power reading skipped for CPU mode.")


def create_args(mode, refinement_stages):
    ckpt = "models/checkpoint_iter_370000.pth"

    return argparse.Namespace(
        quantization=mode,
        num_refinement_stages=refinement_stages,
        checkpoint_path=ckpt,
        labels="coco/annotations/person_keypoints_val2017.json",
        images_folder="coco/val2017/"
    )


if __name__ == "__main__":
    target_refinements = [1]
    target_modes = ["fp16"]

    for ref in target_refinements:
        print(f"\n{'=' * 20}")
        print(f" TESTING REFINEMENT STAGES: {ref} ")
        print(f"{'=' * 20}")

        for mode in target_modes:
            try:
                args = create_args(mode, ref)

                benchmark(
                    args,
                    iterations=50,
                    profiler=False,
                    validate_accuracy=True
                )

            except Exception as e:
                print(f"FAILED [Mode: {mode}, Ref: {ref}]: {e}")

    print("\n" + "=" * 70)
    print(
        f"{'STAGES':<8} | "
        f"{'MODE':<10} | "
        f"{'LATENCY':<10} | "
        f"{'FPS':<10} | "
        f"{'Power (W)':<10} | "
        f"{'FPS/Watt':<10} | "
        f"{'POST':<8}"
    )
    print("-" * 70)

    for r in FINAL_RESULTS:
        fps = float(r.get("Throughput (FPS)", 0))
        power = float(r.get("Power (W)", 0))

        efficiency = fps / power if power > 0 else 0
        r["Efficiency (FPS/W)"] = round(efficiency, 2)

        print(
            f"{r['Stages']:<8} | "
            f"{r['Mode']:<10} | "
            f"{r['Latency (ms)']:<10} | "
            f"{fps:<10.2f} | "
            f"{power:<10.2f} | "
            f"{r['Efficiency (FPS/W)']:<10.2f} | "
            f"{r['Postprocess']:<8}"
        )

    if FINAL_RESULTS:
        with open("benchmark_results_fast_postprocess.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FINAL_RESULTS[0].keys()
            )

            writer.writeheader()
            writer.writerows(FINAL_RESULTS)

        print("\nResults saved to benchmark_results_fast_postprocess.csv")