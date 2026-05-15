import torch
import time
import subprocess
import numpy as np
import json
import cv2
import torch.profiler
import csv
import argparse

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

FINAL_RESULTS = []
base_height = 544
base_width = 968

def calibrate_model(model, args):
    print("--- Starting Calibration (Static PTQ) ---")
    dataset = CocoValDataset(args.labels, args.images_folder)
    with torch.no_grad():
        for i in range(min(135, len(dataset))):
            img = dataset[i]['img']
            img_resized = cv2.resize(img, (base_width, base_height))
            img_mean = 128
            img_scale = 1/256
            normalized_img = (img.astype(np.float32) - img_mean) * img_scale
            input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).float()
            model(input_tensor)
            if i % 10 == 0:
                print(f"Calibrating image {i}/135...")
    print("--- Calibration Complete ---")

def load_model(args):
    net = PoseEstimationWithMobileNet(num_refinement_stages=args.num_refinement_stages)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    load_state(net, checkpoint)
    net.eval()
    net.fuse_model()
    if args.quantization in ['int8', 'mixed_fp32', 'mixed_fp16']:
        torch.backends.quantized.engine = 'fbgemm'
        net.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        if args.quantization in ['mixed_fp32', 'mixed_fp16']:
            layers_to_skip = [net.model[0], net.model[1], net.model[2]]
            net.is_mixed = True
            for layer in layers_to_skip:
                layer.qconfig = None 

        torch.quantization.prepare(net, inplace=True)
        calibrate_model(net, args)
        torch.quantization.convert(net, inplace=True)

        if args.quantization == 'mixed_fp16':
            print("--- Manually casting first 3 layers to FP16 ---")
            for layer in layers_to_skip:
                layer.half()
        print("Loaded calibrated INT8 weights successfully.")
        return net
    else:
        torch.backends.quantized.engine = 'fbgemm'
        net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        net = net.to(torch.device('cpu'))
        if args.quantization == 'fp16':
            net.half()
            print("Running on CPU in FP16.")
        elif args.quantization == 'bf16':
            net = net.to(torch.bfloat16)
            print("Running on CPU in BF16.")
        else:
            print("Running on CPU in FP32.")
        return net.eval()

def benchmark(args, iterations=100, warm_up=20, profiler=False):

    device = torch.device('cpu')

    input_dtype = torch.float32
    if args.quantization == 'fp16' or args.quantization == 'mixed_fp16':
        input_dtype = torch.float16
    elif args.quantization == 'bf16':
        input_dtype = torch.bfloat16

    print(f"\n--- Benchmarking: {args.quantization.upper()} on {device} ---")
    net = load_model(args)

    if (
        args.compile_model
        and args.quantization in ["fp32", "bf16"]
    ):
        print("Compiling model with torch.compile (CPU)...")
        net = torch.compile(
            net,
            backend="inductor",
            mode="max-autotune",
            fullgraph=True,
        )

    dummy_input = torch.randn(1, 3, base_height, base_width, dtype=input_dtype)

    print(f"Warming up...")
    with torch.no_grad():
        for _ in range(warm_up):
            _ = net(dummy_input)

    if profiler:
        print("\n--- Running Profiler ---")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/pose_estimation'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(11):  # Matches schedule (1+5+5)
                    _ = net(dummy_input)
                    prof.step()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    latencies = []
    powers = []
    start_benchmark = time.perf_counter()
    with torch.inference_mode():
        for i in range(iterations):
            iter_start = time.perf_counter()
            _ = net(dummy_input)
            latencies.append(time.perf_counter() - iter_start)
    
    total_bench_time = time.perf_counter() - start_benchmark
    avg_latency = np.mean(latencies) * 1000
    fps = 1.0 / np.mean(latencies)
    avg_power = 0
    
    res = {
        "Stages": args.num_refinement_stages,
        "Mode": args.quantization,
        "Latency (ms)": f"{avg_latency:.2f}",
        "Throughput (FPS)": f"{fps:.2f}",
        "Power (W)": f"{avg_power:.2f}" if device.type == 'cuda' else "N/A",
        "Compiled": args.compile_model if device.type == 'cuda' else "N/A"
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
        images_folder='coco/val2017/',
        compile_model=True,
    )

if __name__ == '__main__':
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    target_refinements = [1,2]
    target_modes = ['fp32','bf16','int8','mixed_fp16','mixed_fp32']

    for ref in target_refinements:
        print(f"\n{'='*20}")
        print(f" TESTING REFINEMENT STAGES: {ref} ")
        print(f"{'='*20}")
        for mode in target_modes:
            try:
                args = create_args(mode, ref)
                benchmark(args, iterations=50, profiler=False)
            except Exception as e:
                print(f"FAILED [Mode: {mode}, Ref: {ref}]: {e}")

    print("\n" + "="*50)
    print(f"{'STAGES':<8} | {'MODE':<10} | {'LATENCY':<10} | {'FPS':<10}")
    print("-" * 50)
    for r in FINAL_RESULTS:
        print(f"{r['Stages']:<8} | {r['Mode']:<10} | {r['Latency (ms)']:<10} | {r['Throughput (FPS)']:<10} | {r.get('Power (W)', 'N/A'):<10}")
    
    if FINAL_RESULTS:
        with open('benchmark_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_RESULTS[0].keys())
            writer.writeheader()
            writer.writerows(FINAL_RESULTS)
        print(f"\nResults saved to benchmark_results.csv")