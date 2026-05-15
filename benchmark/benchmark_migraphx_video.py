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

import migraphx
from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, extract_keypoints_batch, group_keypoints
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from val import evaluate, normalize, pad_width

FINAL_RESULTS = []
base_height = 544
base_width = 968

class MIGraphXWrapper:
    def __init__(self, model):
        self.model = model
        self.input_name = 'input'
        self.expected_type = str(self.model.get_parameter_shapes()[self.input_name].type())
        print(f"DEBUG: MIGraphX expects input type: {self.expected_type}")

    def eval(self): return self

    def __call__(self, x):
        n_input = x.cpu().numpy()
        
        if 'half' in self.expected_type:
            n_input = n_input.astype(np.float16)
        elif 'float' in self.expected_type:
            n_input = n_input.astype(np.float32)

        n_input = np.ascontiguousarray(n_input)
        results = self.model.run({self.input_name: n_input})
        return [torch.from_numpy(np.array(res)).float() for res in results]


def coco_eval_stats(gt_file_path, dt_file_path):
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)
    result = COCOeval(coco_gt, coco_dt, 'keypoints')
    result.evaluate()
    result.accumulate()
    result.summarize()

    keys = [
        'AP',
        'AP50',
        'AP75',
        'APm',
        'APl',
        'AR',
        'AR50',
        'AR75',
        'ARm',
        'ARl',
    ]

    return dict(zip(keys, result.stats.copy()))


def postprocess_standard(heatmaps, pafs, original_hw, stride=8):
    orig_h, orig_w = original_hw

    heatmaps = cv2.resize(heatmaps, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    all_keypoints_by_type = []
    total_keypoints_num = 0

    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx],
            all_keypoints_by_type,
            total_keypoints_num
        )

    return group_keypoints(all_keypoints_by_type, pafs)

def postprocess_fast(heatmaps, pafs, original_hw, stride=8):
    orig_h, orig_w = original_hw
    out_h, out_w = heatmaps.shape[:2]

    all_keypoints = []
    total_keypoints_num = 0
    for kpt_idx in range(18):
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints, pafs)
    
    scale_x = orig_w / out_w
    scale_y = orig_h / out_h
    for kpt in all_keypoints:
        kpt[0] *= scale_x
        kpt[1] *= scale_y
    
    return pose_entries, all_keypoints

def postprocess_optimized_batch(heatmaps, pafs, original_hw, stride=8):
    orig_h, orig_w = original_hw

    # -------------------------------------------------
    # 1. Resize heatmaps to original image size
    # -------------------------------------------------
    heatmaps = cv2.resize(
        heatmaps,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    # -------------------------------------------------
    # 2. Resize PAFs to original image size
    # -------------------------------------------------
    pafs = cv2.resize(
        pafs,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    # -------------------------------------------------
    # 3. Batch keypoint extraction
    # -------------------------------------------------
    all_keypoints_by_type, total_keypoints_num = extract_keypoints_batch(
        heatmaps[:, :, :18],
        max_keypoints_per_type=10
    )

    # -------------------------------------------------
    # 4. Group keypoints
    # -------------------------------------------------
    return group_keypoints(all_keypoints_by_type, pafs)

def postprocess_optimized_batch_k20(heatmaps, pafs, original_hw, stride=8):
    orig_h, orig_w = original_hw

    heatmaps = cv2.resize(
        heatmaps,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    pafs = cv2.resize(
        pafs,
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC
    )

    all_keypoints_by_type, total_keypoints_num = extract_keypoints_batch(
        heatmaps[:, :, :18],
        max_keypoints_per_type=20
    )

    return group_keypoints(all_keypoints_by_type, pafs)

def infer_migraphx_outputs(model, img, quantization_type='fp32', pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    stride = 8
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    ratio = min(base_height / height, base_width / width)

    scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
    min_dims = [base_height, base_width]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).contiguous()
    raw_results = run_inference(model, tensor_img)

    heatmaps = np.transpose(np.array(raw_results[-2]).squeeze().astype(np.float32), (1, 2, 0))
    pafs = np.transpose(np.array(raw_results[-1]).squeeze().astype(np.float32), (1, 2, 0))

    scaled_pad = [p // stride for p in pad]
    heatmaps = heatmaps[scaled_pad[0]:heatmaps.shape[0] - scaled_pad[2], scaled_pad[1]:heatmaps.shape[1] - scaled_pad[3], :]
    pafs = pafs[scaled_pad[0]:pafs.shape[0] - scaled_pad[2], scaled_pad[1]:pafs.shape[1] - scaled_pad[3], :]
    
    return heatmaps, pafs, width, height


def build_coco_detections(image_id, pose_entries, all_keypoints):
    coco_result = []
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:
                continue
            cx, cy, score, visibility = 0, 0, 0, 0
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))
    for idx in range(len(coco_keypoints)):
        coco_result.append({
            'image_id': image_id,
            'category_id': 1,
            'keypoints': coco_keypoints[idx],
            'score': scores[idx]
        })
    return coco_result


def evaluate_postprocessing_variants(model, labels, images_folder, variants, output_dir='outputs', quantization_type='fp32', max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    dataset = CocoValDataset(labels, images_folder)
    results = {}
    total_images = len(dataset)

    for name, post_fn in variants:
        detections = []
        print(f"\nEvaluating postprocessing variant: {name}")
        processed = 0
        for i, sample in enumerate(dataset):
            if i < 134:
                continue
            if max_images is not None and processed >= max_images:
                break
            file_name = sample['file_name']
            img = sample['img']
            image_id = int(file_name[0:file_name.rfind('.')])

            if processed % 20 == 0:
                target_count = max_images if max_images is not None else total_images - 134
                print(f"  processing image {processed + 1}/{target_count}: {file_name}")

            heatmaps, pafs, orig_w, orig_h = infer_migraphx_outputs(model, img, quantization_type)
            pose_entries, all_keypoints = post_fn(heatmaps, pafs, (orig_h, orig_w), 8)
            detections.extend(build_coco_detections(image_id, pose_entries, all_keypoints))
            processed += 1

        output_name = os.path.join(output_dir, f"detections_{name}.json")
        with open(output_name, 'w') as f:
            json.dump(detections, f, indent=4)

        results[name] = coco_eval_stats(labels, output_name)

    return results


def compare_postprocessing_on_datasets(model, dataset_configs, variants, output_dir='outputs', quantization_type='fp32', max_images=None):
    all_results = {}
    for ds_name, labels, images_folder in dataset_configs:
        ds_dir = os.path.join(output_dir, ds_name.replace(' ', '_'))
        print(f"\nComparing on dataset: {ds_name}")
        all_results[ds_name] = evaluate_postprocessing_variants(
            model, labels, images_folder, variants, ds_dir, quantization_type, max_images=max_images)

    print('\nComparison summary:')
    print('{:<20} {:<24} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}'.format(
        'dataset', 'variant', 'AP', 'AP50', 'AP75', 'AR', 'AR50', 'AR75'
    ))

    for ds_name, metrics in all_results.items():
        for variant_name, stats in metrics.items():
            print('{:<20} {:<24} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f}'.format(
                ds_name,
                variant_name,
                stats['AP'],
                stats['AP50'],
                stats['AP75'],
                stats['AR'],
                stats['AR50'],
                stats['AR75']
            ))
    return all_results

def print_comparison_results(results):
    print('\nAP/AR comparison results:')
    for ds_name, metrics in results.items():
        print(f'\nDataset: {ds_name}')
        for variant_name, stats in metrics.items():
            print(
                f'  {variant_name}: '
                f'AP={stats["AP"]:.3f}, '
                f'AP50={stats["AP50"]:.3f}, '
                f'AP75={stats["AP75"]:.3f}, '
                f'AR={stats["AR"]:.3f}, '
                f'AR50={stats["AR50"]:.3f}, '
                f'AR75={stats["AR75"]:.3f}'
            )

def load_model(args, device):
    onnx_path = "models/fp16_refinment1.onnx"

    compiled_model_path = f"pose_model1_{args.quantization}_ref{args.num_refinement_stages}.mxr"
    if os.path.exists(compiled_model_path):
        print(f"--- Loading pre-compiled model from {compiled_model_path} ---")
        return migraphx.load(compiled_model_path)

    print(f"--- Compiled model not found. Compiling {onnx_path} (Exhaustive Tune) ---")
    model = migraphx.parse_onnx(onnx_path)
    
    target = migraphx.get_target("gpu")

    if args.quantization == 'fp16':
        migraphx.quantize_fp16(model)
    elif args.quantization == 'int8':
        migraphx.quantize_int8(model, target, [])
    elif args.quantization == 'bf16':
        migraphx.quantize_bf16(model)
    
    model.compile(target, exhaustive_tune=True)
    
    print(f"--- Saving compiled model to {compiled_model_path} ---")
    migraphx.save(model, compiled_model_path)

    return model

def get_gpu_power():
    try:
        res = subprocess.check_output(['rocm-smi', '--showpower', '--json']).decode('utf-8')
        data = json.loads(res)
        power_str = data['card0']['Current Socket Graphics Package Power (W)']
        return float(power_str)
    except Exception:
        return 0.0

def run_inference(model, tensor_input):
    n_input = tensor_input.cpu().contiguous().numpy()
    param_type = str(model.get_parameter_shapes()['input'].type())

    if 'half' in param_type:
        n_input = n_input.astype(np.float16)
    elif 'bfloat' in param_type:
        n_input = n_input.astype(np.float32)
    elif 'float' in param_type:
        n_input = n_input.astype(np.float32)

    return model.run({'input': n_input})
    
def benchmark(args, iterations=100, warm_up=20, profiler=False, validate_accuracy=True):
    device = torch.device('cuda')

    print(f"\n--- Benchmarking: {args.quantization.upper()} on {device} ---")
    net = load_model(args, device)

    param_type = str(net.get_parameter_shapes()['input'].type())
    if 'half' in param_type:
        input_dtype = torch.float16
    elif 'bfloat' in param_type:
        input_dtype = torch.bfloat16
    else:
        input_dtype = torch.float32

    dummy_input = torch.randn(1, 3, base_height, base_width, dtype=input_dtype).to("cuda")

    print(f"Warming up...")
    with torch.inference_mode():
        for _ in range(warm_up):
            _ = run_inference(net, dummy_input)
            torch.cuda.synchronize()

    if profiler:
        print("\n--- Running Profiler ---")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/pose_estimation'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.inference_mode():
                for _ in range(11):  # Matches schedule (1+5+5)
                    _ = run_inference(net, dummy_input)
                    prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if validate_accuracy:
        print(f"\n--- Starting Accuracy Validation (COCO) ---")
        wrapped_model = MIGraphXWrapper(net)
        output_json = f"results_{args.quantization}_ref{args.num_refinement_stages}.json"
        
        try:
            evaluate(
                labels=args.labels, 
                output_name=output_json, 
                images_folder=args.images_folder, 
                net=wrapped_model, 
                quantization_type=args.quantization
            )
        except Exception as e:
            print(f"EVALUATION FAILED: {e}")

    input_arg = migraphx.argument(dummy_input.detach().cpu().numpy())
    run_params = {"input": input_arg}
    
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
        "Power (W)": f"{avg_power:.2f}" if device.type == 'cuda' else "N/A"
    }
    FINAL_RESULTS.append(res)
    
    print(f"DONE: {fps:.2f} FPS")
    print(f"Results for {args.quantization}:")
    print(f" - Avg Latency: {avg_latency:.2f} ms")
    print(f" - Throughput: {fps:.2f} FPS")
    if device.type == 'cuda':
        print(f" - Avg Power: {avg_power:.2f} W")
        print(f" - Efficiency: {fps/avg_power if avg_power > 0 else 0:.2f} FPS/Watt")
    else:
        print(f" - Note: Power reading skipped for CPU mode.")


def create_args(mode, refinement_stages):
    ckpt = "models/checkpoint_iter_370000.pth"
    return argparse.Namespace(
        quantization=mode,
        num_refinement_stages=refinement_stages,
        checkpoint_path=ckpt,
        labels='coco/annotations/person_keypoints_val2017.json',
        images_folder='coco/val2017/'
    )


if __name__ == '__main__':
    args = create_args('fp16', 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Running postprocessing comparison on {device} ---")

    net = load_model(args, device)
    variants = [
        ('standard', postprocess_standard),
        ('optimized_batch_k10', postprocess_optimized_batch),
        ('optimized_batch_k20', postprocess_optimized_batch_k20),
    ]

    results = compare_postprocessing_on_datasets(
        net,
        [('COCO val2017', args.labels, args.images_folder)],
        variants,
        output_dir='outputs',
        quantization_type=args.quantization,
        max_images=5000
    )

    print_comparison_results(results)
    print('\nDone: compared two postprocessing variants.')
