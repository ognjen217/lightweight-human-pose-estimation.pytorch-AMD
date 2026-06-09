# Lightweight Human Pose Estimation on AMD (ROCm & MIGraphX)

This repository is an optimized version of the Lightweight OpenPose project, specifically tailored for the AMD ROCm ecosystem. By leveraging the AMD Strix Halo APU’s unified memory and MIGraphX inference engine, this implementation achieves real-time multi-person tracking suitable for edge deployment in high-traffic venues.

## Key Improvements (AMD Optimization)

Compared to the original repository, the following changes have been implemented:

1. ROCm 7.2+ Port: Migrated to Python 3.10+ and the latest PyTorch ROCm binaries.

2. MIGraphX Backend: Model compiled specifically for RDNA 3.5, jumping from ~8 FPS (Native) to ~215 FPS.

3. Zero-Copy Memory Opt: Reduced 6x redundant PCIe copies between CPU/GPU to 2x. Intermediate Heatmaps & PAFs remain in unified GPU memory.

4. Kernel Exhaustive Search: Identification of the fastest mathematical kernels specifically for the Strix Halo architecture.

5. Seamless Quantization: Implemented a unified codebase for FP32, FP16, BF16, and INT8 (both static and dynamic).

## Benchmark Results (Strix Halo)

Following extensive optimization, FP16 with 1 refinement stage emerged as the "sweet spot" for balancing throughput and precision on the Strix Halo architecture.

|Optimization Phase |Backend |Precision|Throughput (FPS)|Avg. Power (Watts)| 
|------------------|--------|---------|----------------|------------|
|Initial Port      |PyTorch |FP16     |8.02               |83.25W       |
|MIGraphX          |MIGraphX|FP16     |148.31            |62.63W       |
|Improved model perf|MIGraphX|FP16     |210.66            |55.44W       |
|Final (Exhaustive kernel)|MIGraphX|FP16     |215.45            |48.44W       |

### Key Observations:

Backend Impact: The initial PyTorch port using native kernels was severely bottlenecked (INT8 peaked at only 28.58 FPS). Migrating to MIGraphX provided an immediate 18x performance boost.

Quantization Choice: While FP32 performed respectably at 82.07 FPS, FP16 reached 148.31 FPS. Testing with INT8 and BF16 showed no significant gains in throughput over FP16, leading us to standardize on FP16 for the best balance of speed and stability.

IR & Memory Optimization: Profiling the Intermediate Representation (IR) revealed six redundant PCIe copy operations. By optimizing the data flow to retain heatmaps and Part Affinity Fields (PAFs) in GPU memory, we significantly reduced overhead.

```
for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(self.cat_op.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

final_results = [stages_output[-2], stages_output[-1]]
return [self.dequant(out) for out in final_results]
```

In the above code we can see that for each refinement stage we are appending a new value for heatmaps and pafs to the output, but we only need the latest one, so we can optimize the code to only keep the latest and that optimisation saves about 50FPS.

Kernel Tuning: We utilized MIOpen exhaustive search to tune convolution algorithms. By pre-selecting optimal tiling sizes and memory layouts for RDNA 3.5 Compute Units, we shaved an additional 1.5ms off the tail latency.

## Accuracy Analysis

Accuracy testing demonstrated that FP16 maintains parity with FP32, whereas INT8 suffers from significant precision degradation.

|Model Variant|Precision|AP @ 0.5:0.95|AR @ 0.5:0.95|Notes                       |
|-------------|---------|-------------|-------------|----------------------------|
|Refinement 1 |FP32     |0.436        |0.490        |Baseline                    |
|             |FP16     |0.428        |0.482        |< 1% drop from FP32         |
|             |INT8     |0.274        |0.313        |Significant precision loss  |
|Refinement 2 |FP32     |0.458        |0.513        |                            |
|             |FP16     |0.453        |0.508        |Optimal performance/accuracy|
|             |INT8     |0.296        |0.336        |                            |
|Refinement 3 |FP32     |0.461        |0.518        |Peak accuracy               |
|             |FP16     |0.456        |0.510        |                            |



Power Efficiency Final
|Precision         |Avg. Power (Watts)|Throughput (FPS)|Efficiency (FPS/Watt)|
|------------------|-------------|--------|-------------|
|FP16 (1 Stage)    |	48.44W	 | 215.45 | 4.45
|FP16 (2 Stages)   |	61.24W   | 207.02 |	3.38
|FP16 (3 Stages)   |	66.85W   | 196.92 | 2.95
|FP32 (1 Stage)    |   60.24W    | 95.75  | 1.59

### Performance Profiling

Below is the execution profile for FP16 (1 refinement step) processing five 968 × 544 images. The profile highlights the dominance of fused MLIR kernels (convolution + ReLU) and the minimized impact of Device-to-Host (DtoH) copies.

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  

                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us      20.963ms       107.10%      20.963ms       2.096ms            10  
                    mlir_convolution_broadcast_add_relu         0.00%       0.000us         0.00%       0.000us       0.000us      15.227ms        77.79%      15.227ms      78.085us           195  
                mlir_convolution_broadcast_add_relu_add         0.00%       0.000us         0.00%       0.000us       0.000us       2.101ms        10.73%       2.101ms      84.022us            25  
                                       mlir_convolution         0.00%       0.000us         0.00%       0.000us       0.000us     749.703us         3.83%     749.703us      24.990us            30  
                                          ProfilerStep*         3.52%     946.228us       100.00%      26.909ms       5.382ms       0.000us         0.00%     502.377us     100.475us             5  
                                               aten::to         0.05%      12.949us         8.95%       2.409ms     240.887us       0.000us         0.00%     502.377us      50.238us            10  
                                         aten::_to_copy         0.11%      30.589us         8.90%       2.396ms     479.183us       0.000us         0.00%     502.377us     100.475us             5  
                                            aten::copy_         0.13%      35.419us         8.67%       2.332ms     466.436us     502.377us         2.57%     502.377us     100.475us             5  
                           Memcpy DtoH (Device -> Host)         0.00%       0.000us         0.00%       0.000us       0.000us     502.377us         2.57%     502.377us     100.475us             5  
                      _migraphxgpudevicelauncherIZZN...         0.00%       0.000us         0.00%       0.000us       0.000us     389.653us         1.99%     389.653us      25.977us            15  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  



## Extended Investigation: End-to-End Video Pipeline Bottleneck

After the inference backend was optimized with MIGraphX, the next research step was to evaluate the complete video pipeline rather than the neural network inference stage in isolation. This is important because the previously reported throughput of approximately 215 FPS describes the optimized model execution path, while the real application also includes frame preprocessing, heatmap/PAF decoding, resizing, keypoint extraction, keypoint grouping, and optional drawing/video output.

The profiling of `video_val.py` showed that MIGraphX inference is no longer the dominant bottleneck. Instead, the main cost moved to CPU-side postprocessing, especially the keypoint extraction and pose grouping stages. In the measured FP16 refinement-stage-1 configuration, inference was approximately 8 ms per frame, while the original postprocessing path could take several hundred milliseconds per frame.

### Postprocessing Variants Tested

The following postprocessing strategies were implemented and benchmarked:

| Variant | Description | Expected Accuracy Behavior |
|---|---|---|
| `standard` | Original postprocess path: resize heatmaps, resize PAFs, run original `extract_keypoints`, then `group_keypoints`. | Reference behavior and accuracy baseline. |
| `fast_no_resize` | Skips heatmap and PAF resizing before grouping. Grouping is done at network output resolution and keypoints are scaled afterwards. | Extremely fast, but may reduce AP/AR because grouping is performed on lower-resolution maps. |
| `optimized_batch_k10` | Keeps original resize behavior, but replaces per-channel keypoint extraction with batched OpenCV-based extraction limited to 10 keypoints per type. | Faster than standard, but may lose accuracy when more people/keypoints are present. |
| `optimized_batch_k20` | Same batched extraction approach, but allows up to 20 keypoints per type. | Best accuracy-preserving optimization in the current tests. |

### Latest End-to-End Benchmark

The following benchmark was run on the FP16, one-refinement-stage MIGraphX model:

```bash
python benchmark_postprocess_models_report.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --warmup 20
```

Measured configuration:

- Model: `pose_model1_fp16_ref1.mxr`
- Precision: FP16
- Refinement stages: 1
- Measured frames: 100
- Warmup frames: 20
- Drawing and video writing: disabled

| Model | Variant | Preprocess (ms) | Inference (ms) | Postprocess (ms) | End-to-End (ms) | End-to-End FPS | Postprocess Speedup | E2E Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pose_model1_fp16_ref1.mxr` | `standard` | 3.77 | 7.99 | 499.80 | 511.56 | 1.95 | 1.00× | 1.00× |
| `pose_model1_fp16_ref1.mxr` | `fast_no_resize` | 3.77 | 7.99 | 3.25 | 15.01 | 66.64 | 153.77× | 34.09× |
| `pose_model1_fp16_ref1.mxr` | `optimized_batch_k10` | 3.77 | 7.99 | 334.66 | 346.42 | 2.89 | 1.49× | 1.48× |
| `pose_model1_fp16_ref1.mxr` | `optimized_batch_k20` | 3.77 | 7.99 | 334.25 | 346.01 | 2.89 | 1.50× | 1.48× |

The results show that the neural network inference step is already efficient: it takes only around 7.99 ms per frame. However, the full standard pipeline reaches only 1.95 FPS because the postprocessing stage dominates the runtime. The `fast_no_resize` variant demonstrates the upper bound of a lightweight postprocessing path, reaching 66.64 FPS end-to-end, but this result should not be interpreted as accuracy-preserving without COCO validation. The `optimized_batch_k20` variant is currently the best conservative optimization because it preserves the original resize-and-grouping logic while accelerating keypoint extraction.

### Accuracy Comparison of Postprocessing Variants

The postprocessing variants were also compared on COCO validation metrics:

| Dataset | Variant | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 | `standard` | 0.386 | 0.650 | 0.389 | 0.444 | 0.690 | 0.452 |
| COCO val2017 | `optimized_batch_k10` | 0.364 | 0.615 | 0.364 | 0.421 | 0.656 | 0.426 |
| COCO val2017 | `optimized_batch_k20` | 0.386 | 0.650 | 0.389 | 0.444 | 0.690 | 0.452 |

The `optimized_batch_k20` variant matches the standard postprocess accuracy in the tested COCO validation run, while `optimized_batch_k10` is faster but loses AP/AR because the maximum number of retained keypoints per type is too restrictive. Therefore, `optimized_batch_k20` is the best current candidate for the final accuracy-preserving pipeline.

### Energy and Power Metrics

The benchmark was extended with power and efficiency metrics so that models can be compared not only by latency and FPS, but also by energy efficiency. The extended script reports:

| Metric | Meaning |
|---|---|
| `avg_power_w` | Average GPU socket/package power sampled through `rocm-smi --showpower`. |
| `fps_per_watt` | End-to-end FPS divided by average GPU power. Higher is better. |
| `energy_j_per_frame` | Approximate joules consumed per processed frame. Lower is better. |
| `e2e_avg_ms` | Full pipeline latency: preprocess + inference + postprocess. |
| `e2e_fps` | Full pipeline throughput calculated from end-to-end latency. |
| `post_speedup_vs_standard` | Postprocess speedup compared to the standard postprocess path. |
| `e2e_speedup_vs_standard` | Full-pipeline speedup compared to the standard postprocess path. |

The formulas used are:

```python
e2e_fps = 1000.0 / e2e_avg_ms
fps_per_watt = e2e_fps / avg_power_w
energy_j_per_frame = avg_power_w * (e2e_avg_ms / 1000.0)
```

On the tested system, `rocm-smi --showpower` reports power in the following format:

```text
GPU[0] : Current Socket Graphics Package Power (W): 43.02
```

The benchmark power parser was updated to support this ROCm output format. If the parser cannot read a valid power value, the table prints `N/A` for `Power`, `FPS/W`, and `J/frame`, while latency and FPS are still reported normally.

### Updated Benchmark Script

The updated benchmark script is intended to compare both model variants and postprocess variants:

```bash
python benchmark_postprocess_models_report_powerfix.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --warmup 20
```

For comparing multiple MIGraphX models:

```bash
python benchmark_postprocess_models_report_powerfix.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --models pose_model1_fp16_ref1.mxr pose_model1_fp32_ref1.mxr pose_model1_int8_ref1.mxr \
  --frames 100 \
  --warmup 20 \
  --csv model_postprocess_benchmark_summary.csv \
  --md model_postprocess_benchmark_report.md \
  --detailed-csv model_postprocess_per_frame.csv
```

The summary table includes the following fields:

```text
model | variant | mode | ref | frames | pre | infer | post | e2e | FPS | Power | FPS/W | J/frame | post_spd | e2e_spd | Δe2e%
```

### Current Best Configuration

Based on the current tests, there are two relevant configurations depending on the goal:

| Goal | Recommended Variant | Reason |
|---|---|---|
| Best accuracy-preserving optimization | `optimized_batch_k20` | Matches standard AP/AR in the COCO validation run while reducing postprocess time from 499.80 ms to 334.25 ms. |
| Maximum speed experiment | `fast_no_resize` | Reduces end-to-end latency to 15.01 ms and reaches 66.64 FPS, but must be treated as a separate approximation because grouping is done at lower resolution. |

The current final recommendation is to use FP16 with one refinement stage and `optimized_batch_k20` as the stable accuracy-preserving postprocess configuration. Further acceleration should focus on optimizing `group_keypoints`, but replacing it with a faster grouping method may reduce AP/AR metrics and therefore requires a full COCO validation run after each change.

### Research Status After Postprocessing Tests

The optimization path can now be summarized in two layers:

1. **Model inference optimization**: ROCm + MIGraphX + FP16 + kernel exhaustive search moved inference from the original PyTorch bottleneck to an efficient model execution path.
2. **Application pipeline optimization**: After inference became fast, CPU-side postprocessing became the dominant bottleneck. Batched keypoint extraction improves the conservative pipeline, while low-resolution grouping provides a much faster but potentially less accurate alternative.

This means that the main remaining research direction is no longer neural network inference, but postprocessing algorithm design. The next promising step is to accelerate or redesign pose grouping while preserving COCO AP/AR metrics.


## Installation and Setup

1. Prepare COCO Dataset:\
Download from cocodataset.org [https://cocodataset.org/#home\]
Extract COCO 2017 into the <COCO_HOME> folder.

2. Environment:
```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements/requirements.txt
```
If `pycocotools` fails to build with `ModuleNotFoundError: No module named 'Cython'`, rerun:
```
    pip install --no-build-isolation -r requirements/requirements.txt
```
If COCO evaluation later fails in `pycocotools` with NumPy 2.x, install a compatible NumPy first:
```
    pip install "numpy<2.0"
    pip install -r requirements/requirements.txt
```
Then continue with:
```
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
    # Export ROCm Library Path (Crucial for MIGraphX/ROCm 7.2.0)
    export PYTHONPATH=$PYTHONPATH:/opt/rocm-7.2.0/lib
```
3. Example validation script:
```
    python3 val.py \
    --checkpoint-path models/checkpoint_iter_370000.pth \
    --labels coco/annotations/person_keypoints_val2017.json \
    --images-folder coco/val2017 \
    --num-refinement-stages 1 \
    --quantization fp32
```

## Citations and References

This project is based on:

    Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
    Daniil Osokin, 2018. arXiv:1811.12004

Original Project: osokin/lightweight-human-pose-estimation.pytorch