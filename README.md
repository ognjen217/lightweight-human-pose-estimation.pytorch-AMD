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
### Grouping Optimization: Removing Full PAF Channel Copies

After the batched keypoint extraction improvements, the remaining bottleneck was the pose grouping stage. Function-level timing inside `group_keypoints` showed that most of the grouping time was not spent in NMS or pose assembly, but in repeatedly slicing the full Part Affinity Field tensor for every limb pair.

The original grouping path used the following pattern for each body part:

```python
part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
```

This creates a full `H × W × 2` copy of the PAF channels for every limb before only a small number of sampled points are actually used. The optimized implementation avoids this full-channel copy and instead stores only the two PAF channel IDs, then gathers the required values directly at the sampled coordinates:

```python
paf_x_id, paf_y_id = BODY_PARTS_PAF_IDS[part_id]

field = np.empty((x.shape[0], 2), dtype=np.float32)
field[:, 0] = pafs[y, x, paf_x_id]
field[:, 1] = pafs[y, x, paf_y_id]
field = field.reshape(-1, points_per_limb, 2)
```

This change keeps the grouping algorithm equivalent, because the same PAF values are used for affinity scoring, but it avoids repeatedly copying large PAF maps. It directly targets the dominant cost observed in `group_keypoints` and makes the accuracy-preserving postprocess path significantly faster.

### Latest Benchmark After Grouping Improvements

The latest benchmark was run with `benchmark_postprocess_models_report_powerfix.py` on `cctv_1280x720_24fps_3.mp4`, using 5 warmup frames and 100 measured frames. Drawing and video writing were disabled so the benchmark isolates preprocess, inference, and postprocess cost.

| Model | Variant | Mode | Frames | Preprocess (ms) | Inference (ms) | Postprocess (ms) | End-to-End (ms) | FPS | Power (W) | FPS/W | J/frame | Postprocess Speedup | E2E Speedup | ΔE2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pose_model1_fp16_ref1.mxr` | `standard` | fp16 | 100 | 3.72 | 7.75 | 498.08 | 509.55 | 1.96 | 39.44 | 0.05 | 20.0966 | 1.00× | 1.00× | 0.00% |
| `pose_model1_fp16_ref1.mxr` | `fast_no_resize` | fp16 | 100 | 3.72 | 7.75 | 3.05 | 14.52 | 68.88 | 39.44 | 1.75 | 0.5726 | 163.45× | 35.10× | -97.15% |
| `pose_model1_fp16_ref1.mxr` | `optimized_batch_k20_fast` | fp16 | 100 | 3.72 | 7.75 | 123.94 | 135.41 | 7.38 | 39.44 | 0.19 | 5.3407 | 4.02× | 3.76× | -73.42% |
| `pose_model1_fp16_ref1.mxr` | `optimized_batch_k20` | fp16 | 100 | 3.72 | 7.75 | 336.03 | 347.50 | 2.88 | 39.44 | 0.07 | 13.7053 | 1.48× | 1.47× | -31.80% |

The key result is that `optimized_batch_k20_fast` reduces postprocessing from **498.08 ms** to **123.94 ms**, giving a **4.02× postprocess speedup** and a **3.76× end-to-end speedup** compared with the standard pipeline. End-to-end latency drops from **509.55 ms** to **135.41 ms**, while throughput improves from **1.96 FPS** to **7.38 FPS**.

Importantly, the COCO validation results show that the new optimized grouping path preserves the same AP/AR metrics as the standard path in this test:

| Dataset | Variant | AP | AP50 | AP75 | AR | AR50 | AR75 |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO val2017 | `standard` | 0.387 | 0.651 | 0.390 | 0.446 | 0.692 | 0.453 |
| COCO val2017 | `optimized_batch_k20_fast` | 0.387 | 0.651 | 0.390 | 0.446 | 0.692 | 0.453 |
| COCO val2017 | `optimized_batch_k20` | 0.387 | 0.651 | 0.390 | 0.446 | 0.692 | 0.453 |

Therefore, the current best accuracy-preserving configuration is now `optimized_batch_k20_fast`: it combines batched keypoint extraction with the faster grouping implementation that avoids full PAF channel slicing. The previous conservative `optimized_batch_k20` remains useful as a reference implementation, but the new grouping optimization provides a much larger speedup without reducing measured COCO AP/AR in the latest validation run.

### Current Best Configuration

Based on the current tests, there are two relevant configurations depending on the goal:

| Goal | Recommended Variant | Reason |
|---|---|---|
| Best accuracy-preserving optimization | `optimized_batch_k20_fast` | Matches standard AP/AR in the latest COCO validation run while reducing postprocess time from 498.08 ms to 123.94 ms. |
| Conservative accuracy-preserving reference | `optimized_batch_k20` | Keeps the original full-resolution grouping behavior and reduces postprocess time to 336.03 ms. |
| Maximum speed experiment | `fast_no_resize` | Reduces end-to-end latency to 14.52 ms and reaches 68.88 FPS, but must be treated as a separate approximation because grouping is done at lower resolution. |

The current final recommendation is to use FP16 with one refinement stage and `optimized_batch_k20_fast` as the stable accuracy-preserving postprocess configuration. The latest grouping optimization avoids full PAF channel copies and preserves AP/AR in the measured COCO validation run. Further grouping changes should still be validated on COCO after each modification.

### Research Status After Postprocessing Tests

The optimization path can now be summarized in two layers:

1. **Model inference optimization**: ROCm + MIGraphX + FP16 + kernel exhaustive search moved inference from the original PyTorch bottleneck to an efficient model execution path.
2. **Application pipeline optimization**: After inference became fast, CPU-side postprocessing became the dominant bottleneck. Batched keypoint extraction improves the conservative pipeline, while low-resolution grouping provides a much faster but potentially less accurate alternative.

This means that the main remaining research direction is no longer neural network inference, but postprocessing algorithm design. The next promising step is to accelerate or redesign pose grouping while preserving COCO AP/AR metrics.


## GPU-Accelerated Postprocessing Investigation

After the CPU postprocessing optimizations, the next experiment was to move selected postprocessing stages to the GPU using ROCm-backed PyTorch operations. The goal was not to rewrite the full OpenPose postprocessor with custom HIP kernels, but to identify which parts of the existing heatmap/PAF pipeline can benefit from GPU primitives while preserving COCO AP/AR.

The GPU work focused on three postprocessing stages:

| Stage | CPU Baseline | GPU / Hybrid Experiment | Purpose |
|---|---|---|---|
| Heatmap NMS / keypoint extraction | `extract_keypoints` or `extract_keypoints_batch_cv2` | Torch `max_pool2d` NMS on heatmaps | Detect local maxima/keypoint candidates in parallel. |
| PAF connection scoring | CPU affinity sampling inside grouping | Torch tensor-based PAF sampling and dot products | Score candidate limb connections on GPU. |
| Pose assembly / final filtering | CPU grouping logic | Mostly kept on CPU | Preserve existing pose-entry behavior and COCO-compatible output. |

The most important finding is that **partial GPU acceleration is better than moving the whole postprocessor to GPU**. In particular, moving only heatmap NMS/keypoint extraction to GPU while keeping the optimized CPU grouping path gives the best accuracy-preserving latency result in the cached COCO benchmark.

### GPU Postprocessing Variants

The following GPU and hybrid variants were tested in addition to the CPU baselines:

| Variant | Description | Accuracy Expectation |
|---|---|---|
| `standard` | Original full-resolution CPU postprocessing: resize heatmaps and PAFs, run original `extract_keypoints`, then original `group_keypoints`. | Accuracy baseline. |
| `k20_fast` | Full-resolution CPU path using batched K20 keypoint extraction and `group_keypoints_fast`. | Accuracy-preserving optimized CPU baseline. |
| `lowres_cpu_group` | Runs batched K20 extraction and fast grouping directly on low-resolution heatmaps/PAFs, then scales coordinates back to image size. | Very fast, but expected to lose AP/AR. |
| `gpu_nms_fullres_cpu_group` | Full-resolution resize, GPU heatmap NMS/keypoint extraction, then CPU `group_keypoints_fast`. | Main hybrid GPU candidate. |
| `gpu_fullres_paf` | Full-resolution CPU K20 keypoint extraction, GPU PAF connection scoring, then CPU pose assembly. | Accuracy-preserving, but may have GPU/CPU synchronization overhead. |
| `gpu_lowres_paf` | Low-resolution GPU NMS and GPU PAF scoring, then scales keypoints to image size. | Faster than full-resolution PAF scoring, but expected to lose AP/AR. |

### Cached COCO Benchmark: Accuracy and Postprocess Latency

To evaluate the postprocessing algorithms independently from the MIGraphX runtime, the COCO validation benchmark was split into two phases:

1. **MIGraphX cache generation**: MIGraphX inference is executed first and the resulting heatmaps/PAFs are saved for each COCO image.
2. **Cached postprocess evaluation**: the saved heatmaps/PAFs are loaded from disk and each CPU/GPU postprocessing variant is evaluated without importing or running MIGraphX in the same process.

This benchmark is useful for measuring the **algorithmic quality of postprocessing variants** because every variant receives the same network outputs. It also avoids runtime interference between MIGraphX and PyTorch ROCm, so PyTorch can use the GPU cleanly for the GPU-based postprocessing experiments.

Latest cached benchmark result:

| Variant | AP | AP50 | AP75 | AR | Avg Postprocess (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `standard` | 0.415 | 0.684 | 0.422 | 0.473 | 48.48 | 75.01 |
| `k20_fast` | 0.415 | 0.684 | 0.422 | 0.473 | 17.49 | 22.54 |
| `lowres_cpu_group` | 0.203 | 0.459 | 0.147 | 0.254 | 0.88 | 1.91 |
| `gpu_nms_fullres_cpu_group` | 0.415 | 0.684 | 0.422 | 0.473 | 10.99 | 15.03 |
| `gpu_fullres_paf` | 0.415 | 0.684 | 0.422 | 0.473 | 25.97 | 35.42 |
| `gpu_lowres_paf` | 0.232 | 0.530 | 0.169 | 0.291 | 5.62 | 11.12 |

The best cached benchmark result is `gpu_nms_fullres_cpu_group`. It preserves the exact same AP/AR as both `standard` and `k20_fast`, while reducing average postprocessing latency to **10.99 ms**.

| Comparison | Avg Postprocess Before | Avg Postprocess After | Speedup | Latency Reduction |
|---|---:|---:|---:|---:|
| `standard` → `gpu_nms_fullres_cpu_group` | 48.48 ms | 10.99 ms | 4.41× | 77.3% |
| `k20_fast` → `gpu_nms_fullres_cpu_group` | 17.49 ms | 10.99 ms | 1.59× | 37.2% |
| `gpu_fullres_paf` → `gpu_nms_fullres_cpu_group` | 25.97 ms | 10.99 ms | 2.36× | 57.7% |

This result shows that heatmap NMS is a good target for GPU acceleration. It is a highly parallel local-maximum operation, and Torch `max_pool2d` maps well to the GPU when PyTorch ROCm owns the GPU context during the cached postprocess run.

The accuracy results also show that GPU execution itself is not the source of AP/AR degradation. The full-resolution variants all preserve the same metrics:

| Variant | Resolution Used for NMS/Grouping | AP | AR | Accuracy Result |
|---|---|---:|---:|---|
| `standard` | Full resolution | 0.415 | 0.473 | Baseline |
| `k20_fast` | Full resolution | 0.415 | 0.473 | Same as baseline |
| `gpu_nms_fullres_cpu_group` | Full resolution | 0.415 | 0.473 | Same as baseline |
| `gpu_fullres_paf` | Full resolution | 0.415 | 0.473 | Same as baseline |
| `lowres_cpu_group` | Low resolution | 0.203 | 0.254 | Large accuracy drop |
| `gpu_lowres_paf` | Low resolution | 0.232 | 0.291 | Large accuracy drop |

Therefore, the main reason for the accuracy drop in the low-resolution GPU experiments is not GPU NMS, but the algorithmic change from full-resolution postprocessing to low-resolution postprocessing. Even when the final keypoint coordinates are scaled back to the original image size, low-resolution heatmaps and PAFs lose spatial detail that is important for keypoint localization and body-part association.

### Why Full GPU PAF Scoring Was Not the Best Variant

The `gpu_fullres_paf` variant also preserves AP/AR, but it is slower than `gpu_nms_fullres_cpu_group`:

| Variant | AP | AR | Avg Postprocess (ms) | Interpretation |
|---|---:|---:|---:|---|
| `gpu_nms_fullres_cpu_group` | 0.415 | 0.473 | 10.99 | Best accuracy-preserving latency. |
| `gpu_fullres_paf` | 0.415 | 0.473 | 25.97 | Accurate, but slower due to transfer/synchronization and per-limb overhead. |

Although PAF scoring is mathematically parallel, the current implementation still performs per-body-part loops, moves keypoint candidates between CPU and GPU, and performs connection NMS/pose assembly on CPU. These synchronization points reduce the benefit of GPU execution. A faster full-GPU PAF implementation would likely require a more complete redesign of grouping, not only a direct tensor translation of the existing CPU algorithm.

### Low-Resolution GPU and CPU Variants

The low-resolution variants are useful as speed experiments, but they are not currently accuracy-preserving:

| Variant | AP | AR | Avg Postprocess (ms) | Result |
|---|---:|---:|---:|---|
| `lowres_cpu_group` | 0.203 | 0.254 | 0.88 | Fastest, but accuracy drops heavily. |
| `gpu_lowres_paf` | 0.232 | 0.291 | 5.62 | Better AP/AR than low-res CPU grouping, but still far below full-res variants. |

The AP/AR loss comes from doing keypoint extraction and PAF grouping on low-resolution feature maps. Even if the final coordinates are scaled back to image size, the missing spatial detail reduces localization quality and body-part association quality.

### Single-Process Video Benchmark Note

A separate single-process video benchmark was also run through the CLI-style runner, where MIGraphX inference and PyTorch GPU postprocessing execute in the same Python process. In that benchmark, the GPU NMS path did **not** show the same improvement:

| Variant | Mode | Frames | Preprocess (ms) | Inference (ms) | Decode (ms) | HM Resize (ms) | PAF Resize (ms) | Extract (ms) | Group (ms) | Post Avg (ms) | p95 (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `standard` | `standard` | 100 | 3.69 | 8.43 | 1.07 | 4.01 | 8.23 | 294.88 | 217.75 | 525.95 | 536.56 | 1.90 |
| `fast_no_resize` | `fast` | 100 | 3.69 | 8.43 | 1.04 | 0.00 | 0.00 | 0.54 | 1.48 | 3.07 | 3.24 | 325.97 |
| `k20_standard_group` | `k20` | 100 | 3.69 | 8.43 | 0.98 | 4.01 | 8.04 | 108.92 | 217.72 | 339.68 | 347.70 | 2.94 |
| `k20_fast_cpu` | `k20-fast` | 100 | 3.69 | 8.43 | 1.04 | 3.91 | 7.79 | 109.04 | 1.81 | 123.61 | 127.49 | 8.09 |
| `lowres_cpu_group` | `lowres-cpu-group` | 100 | 3.69 | 8.43 | 1.01 | 0.00 | 0.00 | 0.71 | 1.40 | 3.13 | 3.31 | 319.43 |
| `gpu_nms_fullres_cpu_group` | `gpu-nms` | 100 | 3.69 | 8.43 | 0.97 | 3.75 | 7.75 | 530.68 | 1.93 | 545.09 | 566.44 | 1.83 |
| `gpu_fullres_paf` | `gpu-fullres-paf` | 100 | 3.69 | 8.43 | 1.05 | 3.76 | 8.00 | 110.15 | 75.15 | 198.14 | 206.04 | 5.05 |
| `gpu_lowres_paf` | `gpu-lowres-paf` | 100 | 3.69 | 8.43 | 1.04 | 0.00 | 0.00 | 2.19 | 3.29 | 6.53 | 7.47 | 153.19 |

The difference between the cached benchmark and the single-process video benchmark is important:

| Benchmark | What it measures | Runtime setup | Best use | Limitation |
|---|---|---|---|---|
| Cached COCO postprocess benchmark | Accuracy and latency of postprocessing variants on fixed heatmap/PAF tensors. | MIGraphX is not imported during postprocessing; PyTorch ROCm can own the GPU cleanly. | Best benchmark for judging postprocessing algorithm quality and AP/AR impact. | Does not include live video decode, preprocessing, MIGraphX inference, or single-process runtime interaction. |
| Single-process video benchmark | Current end-to-end CLI behavior on video frames. | MIGraphX inference and PyTorch GPU postprocessing run inside the same Python process. | Best benchmark for measuring current integration cost and real runner behavior. | GPU timings can be dominated by runtime interaction, synchronization, or transfer overhead. |

This discrepancy is expected on the tested ROCm setup. The cached benchmark intentionally avoids importing MIGraphX and PyTorch ROCm in the same process, while the video benchmark uses both runtimes together. The very high `extract` time for `gpu_nms_fullres_cpu_group` in the single-process run indicates that GPU NMS timing is dominated by runtime interaction, synchronization, or transfer overhead rather than by the NMS operation alone.

For that reason, the cached COCO benchmark is the preferred measurement for evaluating GPU postprocessing algorithm quality, while the single-process video benchmark remains useful for measuring the current CLI integration cost. In practice, this means that `gpu_nms_fullres_cpu_group` is the best algorithmic candidate, but deployment must still solve the MIGraphX + PyTorch ROCm runtime interaction if this path is used in a single live video process.

### Updated Recommendation After GPU Tests

Based on the current speed and accuracy results, the recommended postprocessing configurations are now:

| Goal | Recommended Variant | Reason |
|---|---|---|
| Best accuracy-preserving GPU/hybrid postprocess | `gpu_nms_fullres_cpu_group` | Matches `standard` AP/AR and gives the lowest full-resolution accuracy-preserving postprocess time in the cached COCO benchmark: 10.99 ms. |
| Best CPU-only accuracy-preserving postprocess | `k20_fast` / `optimized_batch_k20_fast` | Preserves AP/AR and avoids runtime interaction between MIGraphX and PyTorch ROCm. |
| Fastest experimental path | `lowres_cpu_group` | Runs below 1 ms in cached postprocess evaluation, but AP/AR loss is too large for accuracy-critical use. |
| Full GPU PAF research path | `gpu_fullres_paf` | Preserves AP/AR but is slower than GPU NMS hybrid, so it needs deeper grouping redesign before it is useful. |

The final practical conclusion is that **GPU NMS + full-resolution CPU fast grouping is the best tested accuracy-preserving postprocessing method**. It should be used as the main research direction for GPU-accelerated postprocessing, while the production video runner should account for the ROCm runtime interaction between MIGraphX and PyTorch. If both runtimes cannot share the GPU efficiently in one process, the deployment design should use either a two-process cache/streaming architecture or keep `k20_fast` as the stable single-process fallback.


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