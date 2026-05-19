# Lightweight Human Pose Estimation on AMD ROCm + MIGraphX

This repository contains an AMD ROCm/MIGraphX-oriented optimization of the Lightweight OpenPose pipeline. The current focus is no longer only neural network inference speed, but the complete end-to-end video pipeline: preprocessing, MIGraphX inference, heatmap/PAF decoding, full-resolution postprocessing, keypoint extraction, pose grouping, and runtime/power behavior.

A more detailed historical README with the full development trail and earlier benchmark notes is available here:

[https://github.com/ognjen217/lightweight-human-pose-estimation.pytorch-AMD/blob/gpu-accelerated-postprocessing/README.md](https://github.com/ognjen217/lightweight-human-pose-estimation.pytorch-AMD/blob/gpu-accelerated-postprocessing/README.md)

---

## Current Status

The model inference path has already been heavily optimized with MIGraphX, FP16 execution, memory-flow cleanup, and kernel tuning. After these changes, the main bottleneck moved from neural network inference to postprocessing.

The current optimization work therefore focuses on:

- full-resolution heatmap NMS / keypoint extraction,
- faster PAF-based grouping,
- CPU-only fallback paths,
- GPU/hybrid postprocessing,
- MIGraphX NMS as a compiled postprocess subgraph,
- end-to-end video latency and power efficiency.

---

## Model Inference Summary

The best inference configuration remains the FP16 one-refinement-stage MIGraphX model.

| Optimization Phase | Backend | Precision | Throughput | Avg. Power |
|---|---|---:|---:|---:|
| Initial PyTorch ROCm port | PyTorch | FP16 | ~8 FPS | ~83 W |
| MIGraphX backend | MIGraphX | FP16 | ~148 FPS | ~63 W |
| Improved model output path | MIGraphX | FP16 | ~211 FPS | ~55 W |
| Kernel-tuned final model | MIGraphX | FP16 | ~215 FPS | ~48 W |

Key observations:

- MIGraphX gives the largest performance improvement over the initial PyTorch ROCm path.
- FP16 provides the best speed/stability tradeoff.
- INT8 reduced accuracy too much and did not provide a useful throughput advantage.
- Removing redundant intermediate heatmap/PAF outputs improved inference throughput because only the final heatmaps and PAFs are needed by postprocessing.

---

## Why Postprocessing Became the Main Bottleneck

Once MIGraphX inference reached single-digit millisecond latency, the original OpenPose-style CPU postprocessing became the dominant cost. The original path performs full-resolution heatmap resize, full-resolution PAF resize, CPU keypoint extraction, and CPU pose grouping.

Earlier profiling showed that the standard CPU postprocess could take hundreds of milliseconds per frame, even when inference itself was already around 6–8 ms. Therefore, optimizing only the model was not enough for real end-to-end video performance.

---

## Postprocessing Variants

The main postprocessing variants investigated are:

| Variant | Description | Accuracy Status |
|---|---|---|
| `standard_cpu` / `standard` | Original CPU keypoint extraction and original CPU grouping. | Baseline. |
| `optimized_batch_k20_findnonzero_v1_cpu` | Best CPU-only path using batched K20/findNonZero extraction and fast grouping. | Accuracy-preserving in COCO tests. |
| `gpu-fullres-nms-cpu-group` / `gpu-nms` | Full-resolution GPU heatmap NMS/top-K extraction with CPU fast grouping. | Best speed/accuracy tradeoff. |
| `full-gpu` | GPU NMS plus GPU PAF affinity scoring, with final dynamic pose assembly still on CPU. | Accurate but slower than hybrid GPU NMS. |
| `migraphx-nms` | Full-resolution MIGraphX NMS head producing a dense peak mask, followed by CPU mask extraction and fast grouping. | Accuracy-safe, but not fastest. |
| `migraphx-nms-k20` | MIGraphX NMS with candidate limiting. | Slightly faster than full MIGraphX NMS, but loses AP/AR. |
| Low-resolution variants | Skip full-resolution postprocess and scale results back. | Very fast but not accuracy-preserving. |

---

## Latest Single-Process E2E Video Benchmark

This benchmark was run on a fixed 1280x720 video using 100 measured frames and 5 warmup frames. Drawing and video writing were disabled. The benchmark measures the real single-process path:

```text
frame read -> preprocess -> MIGraphX inference -> selected postprocess variant
```

Power was sampled through `rocm-smi`, so `J/frame` and `FPS/W` are estimates based on GPU package power, not full-system wall power.

| Variant | Postprocess | E2E Latency | FPS | Power | J/frame | FPS/W | E2E Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `standard_cpu` | 227.76 ms | 240.06 ms | 4.17 | 56.31 W | 13.52 | 0.074 | 1.00x |
| `optimized_batch_k20_findnonzero_v1_cpu` | 60.05 ms | 72.60 ms | 13.77 | 63.85 W | 4.64 | 0.216 | 3.31x |
| `gpu-fullres-nms-cpu-group` | 38.95 ms | 49.33 ms | 20.27 | 81.68 W | 4.03 | 0.248 | 4.87x |
| `full-gpu` | 46.51 ms | 57.64 ms | 17.35 | 82.06 W | 4.73 | 0.211 | 4.16x |
| `migraphx-nms` | 110.83 ms | 125.11 ms | 7.99 | 61.14 W | 7.65 | 0.131 | 1.92x |

### Single-Process E2E Interpretation

The best current single-process E2E variant is:

```text
gpu-fullres-nms-cpu-group
```

It reaches 20.27 FPS and provides the best energy efficiency among the tested variants. The main reason is that it moves the highly parallel heatmap NMS/top-K extraction step to the GPU while keeping the optimized CPU grouping path.

The best CPU-only fallback is:

```text
optimized_batch_k20_findnonzero_v1_cpu
```

It avoids PyTorch/MIGraphX GPU runtime interaction and still improves E2E latency from 240.06 ms to 72.60 ms.

The `full-gpu` variant is not the best final option because the final OpenPose pose assembly is dynamic and remains CPU-bound. Moving PAF scoring to GPU adds overhead and synchronization without outperforming the simpler hybrid GPU-NMS approach.

---

## COCO2017 Accuracy Benchmark

Accuracy was evaluated on COCO `val2017` keypoint validation using the first 1000 COCO person images, with 5 warmup images excluded from reported results. The benchmark therefore reports metrics over 995 evaluated images.

| Variant | Images | Detections | AP | AP50 | AP75 | AR | Postprocess | Frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `standard` | 995 | 3532 | 0.3208 | 0.5666 | 0.3080 | 0.3716 | 47.98 ms | 58.35 ms |
| `optimized_batch_k20_findnonzero_v1` | 995 | 3532 | 0.3208 | 0.5666 | 0.3080 | 0.3716 | 18.80 ms | 28.17 ms |
| `gpu-nms` | 995 | 3532 | 0.3208 | 0.5666 | 0.3080 | 0.3716 | 13.30 ms | 22.64 ms |
| `migraphx-nms` | 995 | 3867 | 0.3207 | 0.5642 | 0.3085 | 0.3721 | 19.76 ms | 30.47 ms |
| `migraphx-nms-k20` | 995 | 3638 | 0.3161 | 0.5625 | 0.3028 | 0.3675 | 17.72 ms | 27.66 ms |

### Accuracy Interpretation

The following variants are accuracy-preserving relative to the standard baseline:

```text
optimized_batch_k20_findnonzero_v1
gpu-nms
migraphx-nms
```

The best accuracy/speed tradeoff is:

```text
gpu-nms
```

It preserves AP/AR exactly relative to `standard` while reducing postprocess time from 47.98 ms to 13.30 ms.

The `migraphx-nms` variant is also accuracy-safe, with AP 0.3207 vs 0.3208 for `standard` and AR 0.3721 vs 0.3716. However, `migraphx-nms-k20` introduces a measurable accuracy drop, so it should not be treated as the safest final variant.

---

## MIGraphX NMS Findings

### Why COCO Requires Multiple MIGraphX NMS MXR Files

MIGraphX `.mxr` programs are compiled for a fixed input tensor shape. The full-resolution NMS head receives heatmaps in this layout:

```text
[1, 19, H, W]
```

For fixed-resolution video, one `.mxr` is enough because every frame has the same height and width. For example, a 1280x720 video can reuse a single NMS head:

```text
[1, 19, 720, 1280]
```

COCO is different because images have many different resolutions. To keep the validation full-resolution and avoid low-resolution AP/AR loss, the COCO benchmark used a per-resolution MXR cache:

```text
models/nms_fullres_cache/
  heatmap_nms_head_426x640.mxr
  heatmap_nms_head_427x640.mxr
  heatmap_nms_head_480x640.mxr
  ...
```

The benchmark selects the correct `.mxr` file for each image shape.

### MIGraphX NMS Accuracy Result

Full-resolution `migraphx-nms` was successfully validated on COCO. It preserved accuracy almost exactly:

| Variant | AP | AR |
|---|---:|---:|
| `standard` | 0.3208 | 0.3716 |
| `migraphx-nms` | 0.3207 | 0.3721 |

This confirms that the MIGraphX NMS approach is accuracy-safe when applied after full-resolution heatmap resizing and when the correct shape-specific `.mxr` is used.

### MIGraphX NMS Runtime Result

The main limitation is runtime. In the single-process 1280x720 video benchmark, `migraphx-nms` reached 125.11 ms E2E and 7.99 FPS, while the best hybrid GPU-NMS path reached 49.33 ms E2E and 20.27 FPS.

The detailed MIGraphX NMS breakdown was:

| Stage | Time |
|---|---:|
| Decode | 1.09 ms |
| Heatmap resize | 2.38 ms |
| PAF resize | 4.10 ms |
| MIGraphX NMS | 5.66 ms |
| CPU extract from mask | 59.79 ms |
| CPU fast grouping | 5.80 ms |
| Total postprocess | 110.83 ms |

The MIGraphX NMS operation itself is not the main bottleneck. The expensive part is CPU `extract_from_mask`, which converts the dense NMS mask into keypoint candidate lists. This stage dominates the MIGraphX path and removes most of the potential performance gain.

### MIGraphX NMS Conclusion

MIGraphX NMS is validated as an accuracy-safe full-resolution alternative, but it is not the fastest implementation in the current pipeline.

Current interpretation:

- `migraphx-nms` is useful as a proof that NMS can be represented and validated as a compiled MIGraphX subgraph.
- For COCO validation, per-resolution `.mxr` caching is required.
- For fixed-size video, one `.mxr` head is enough.
- The current bottleneck is not the compiled NMS operation, but CPU mask-to-keypoint extraction after the NMS head.
- `migraphx-nms-k20` should not be used as the safest final variant because it reduces AP/AR.
- Further MIGraphX NMS work only makes sense if `extract_from_mask` is replaced by a faster GPU/top-K or optimized CPU extraction path.

---

## Current Recommendation

| Goal | Recommended Variant | Reason |
|---|---|---|
| Best overall speed/accuracy tradeoff | `gpu-fullres-nms-cpu-group` / `gpu-nms` | Preserves COCO AP/AR and gives the best latency in both COCO timing and single-process video tests. |
| Best CPU-only fallback | `optimized_batch_k20_findnonzero_v1_cpu` | Good latency reduction without relying on PyTorch GPU postprocessing in the same process. |
| MIGraphX-specific validated path | `migraphx-nms` | Accuracy-safe, but currently slower due to CPU `extract_from_mask`. |
| Avoid as final accuracy-preserving path | `migraphx-nms-k20` | Introduces measurable AP/AR loss. |
| Avoid for accuracy-critical use | Low-resolution variants | Very fast, but lose too much AP/AR. |

The current final recommendation is to use the FP16 one-refinement-stage MIGraphX model together with full-resolution GPU NMS and CPU fast grouping. If runtime interaction between MIGraphX and PyTorch ROCm becomes problematic in deployment, the optimized CPU-only K20/findNonZero path should be kept as the stable fallback.

---

## Installation and Setup

1. Prepare COCO Dataset:

Download COCO 2017 from [https://cocodataset.org/#home](https://cocodataset.org/#home) and extract it into a local COCO folder.

2. Create the Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
```

If `pycocotools` fails to build with `ModuleNotFoundError: No module named 'Cython'`, rerun:

```bash
pip install --no-build-isolation -r requirements/requirements.txt
```

If COCO evaluation fails with NumPy 2.x, install a compatible NumPy first:

```bash
pip install "numpy<2.0"
pip install -r requirements/requirements.txt
```

Then install PyTorch ROCm and expose ROCm/MIGraphX libraries:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.2
export PYTHONPATH=$PYTHONPATH:/opt/rocm-7.2.0/lib
```

---

## Example Commands

Run the single-process E2E video benchmark:

```bash
python benchmark_singleprocess_e2e_power.py \
  --video migraphx_postprocess_deliverable/cctv_1280x720_24fps_original.mp4 \
  --model migraphx_postprocess_deliverable/pose_model1_fp16_ref1.mxr \
  --migraphx-nms-mxr migraphx_postprocess_deliverable/models/heatmap_nms_head.mxr \
  --variants standard_cpu optimized_batch_k20_findnonzero_v1_cpu gpu-fullres-nms-cpu-group full-gpu migraphx-nms \
  --frames 100 \
  --warmup 5 \
  --torch-device cuda
```

Run COCO postprocess accuracy benchmark:

```bash
python benchmark_postprocess_accuracy.py \
  --images migraphx_postprocess_deliverable/coco/val2017 \
  --annotations migraphx_postprocess_deliverable/coco/annotations/person_keypoints_val2017.json \
  --model migraphx_postprocess_deliverable/pose_model1_fp16_ref1.mxr \
  --migraphx-nms-cache-dir migraphx_postprocess_deliverable/models/nms_fullres_cache \
  --variants standard optimized_batch_k20_findnonzero_v1 gpu-nms migraphx-nms migraphx-nms-k20 \
  --max-images 1000 \
  --warmup-images 5
```

Compile full-resolution MIGraphX NMS heads for COCO shapes:

```bash
python tools/compile_coco_nms_heads.py \
  --annotations migraphx_postprocess_deliverable/coco/annotations/person_keypoints_val2017.json \
  --limit 1000 \
  --output-dir migraphx_postprocess_deliverable/models/nms_fullres_cache \
  --channels 19 \
  --threshold 0.1
```

---

## References

This project is based on:

```text
Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
Daniil Osokin, 2018. arXiv:1811.12004
```

Original project:

```text
osokin/lightweight-human-pose-estimation.pytorch
```
