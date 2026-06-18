# Lightweight Human Pose Estimation on AMD ROCm + MIGraphX

This repository is an AMD-focused optimization and validation fork of the Lightweight OpenPose human pose estimation pipeline. It keeps the original MobileNet-style pose model idea, but the runtime work has moved far beyond a basic PyTorch demo: the repo now contains MIGraphX model compilation, ROCm/HIP postprocessing kernels, stream simulators, accuracy validation, speed validation, and report tooling for repeatable multi-camera experiments.

The most important result is that model inference was not the final bottleneck. After MIGraphX made the neural network fast, the real work became optimizing heatmap decoding, PAF scoring, pose assembly, batching, queueing, and CPU/GPU scheduling.

## Headline Results

| Optimization step | Baseline | Optimized result | Speedup / impact |
|---|---:|---:|---|
| PyTorch FP16 inference -> MIGraphX FP16 inference | 8.02 FPS | 215.45 FPS | About 26.9x model-throughput speedup |
| Smart-full-res fused-pruned stream | 28.16 FPS | 50.57 FPS | 1.80x aggregate stream FPS |
| Smart-full-res accuracy | CPU AP 0.4321 | Smart AP 0.4324 | COCO subset AP parity |
| MXR2 PAF pruning -> HIP2 PAF pruning | 40.92 ms | 8.08 ms | About 5.1x backend speedup |
| Previous MXR2 split stream -> HIP2 split stream | 50.44 FPS | 75.92 FPS | +50.5% aggregate FPS |
| Previous MXR2 split stream E2E latency -> HIP2 split stream | 194.60 ms | 88.46 ms | -54.5% average E2E latency |
| Previous MXR2 split stream P95 latency -> HIP2 split stream | 217.44 ms | 109.42 ms | -49.7% P95 E2E latency |

The best confirmed stream configuration so far is:

```text
variant:            split_hip2_host_smart
pose adapter model:  models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr
MIGraphX batch:      B2
batch timeout:       2 ms
post workers:        2
source load:         10 cameras at 24 FPS source rate
duration:            130 seconds
result:              75.92 aggregate FPS
average E2E:         88.46 ms
P95 E2E:             109.42 ms
```

Earlier in the stream investigation, the B4 merged path collapsed to about 2.13 aggregate FPS in the full simulator even though the isolated model was healthy. CPU pinning, shared-memory handoff, smarter heatmap candidate generation, and the custom HIP2 PAF backend moved the system from that early failure mode to the current 75.92 FPS stream result.

## What This Repo Contains

The repository is organized around the full pose-estimation runtime, not only the neural network.

```text
.
|-- models/                     compiled or exported ONNX/MXR model artifacts
|-- modules/                    model, postprocess, MIGraphX, HIP, and pose logic
|-- simulation/                 modular multi-camera stream simulator
|-- tools/                      export, compile, compare, profile, and plot tools
|-- cpp/                        HIP/C++ kernels for heatmap TopK and PAF pruning
|-- benchmark/                  model and postprocess benchmark scripts/results
|-- reports/                    focused experiment reports and result summaries
|-- docs/                       longer background reports and notes
|-- outputs/                    generated CSV/JSON/video experiment outputs
|-- accuracy_validation.py      COCO-style accuracy validation wrapper
|-- speed_validation.py         video speed validation
|-- simulate_camera_stream.py   patched multi-camera stream entrypoint
+-- video_val.py                frame-level validation/debug entrypoint
```

## Runtime Pipeline

The optimized pipeline is easiest to understand as a sequence of stages:

```text
video frames
  -> preprocess to model input
  -> MIGraphX pose model or split pose-adapter model
  -> heatmaps + PAFs
  -> heatmap candidate extraction
  -> PAF pair scoring and pruning
  -> CPU pose assembly tail
  -> per-camera pose outputs, timing rows, optional grid video
```

The current best stream path splits the work this way:

```text
MXR1:
  image -> pose model -> adapter -> heatmaps + PAFs

HIP heatmap stage:
  low-res proposals -> full-res local refinement -> top_scores/top_indices

HIP2 PAF backend:
  full-resolution PAF sampling -> limb pair scores -> pruned pair tensors

CPU tail:
  assemble final pose entries from pruned pairs
```

This is why the recommended stream variant is `split_hip2_host_smart`.

## Practical Decision Guide

| Goal | Recommended path | Notes |
|---|---|---|
| Best current live stream result | `split_hip2_host_smart` | Use B2, 2 ms timeouts, HIP PAF backend, shared memory, CPU pinning. |
| Best visual/demo stream from the current experiments | `split_hip2_host_smart` with 8 cameras, 6 post workers, K15, threshold 0.15 | Produces a 2x4 grid video and detailed CSV. |
| Accuracy-preserving smart-full-res fused-pruned validation | `mx_fused_cubic_topk_fullres_paf_pruned` / `merged_fused_pruned` style paths | Validated on a 1000-image dominant-resolution COCO subset. |
| Historical accuracy-preserving GPU NMS runtime | `gpu_nms_fullres_two_process` | Useful baseline before split HIP2 work. |
| Historical MIGraphX-NMS proof of concept | `migraphx_nms`, `migraphx_nms_k20` | Shows compiled NMS feasibility, but not the fastest final path. |
| CPU fallback baseline | `optimized_batch_k20_fast` | Good when avoiding GPU postprocess dependencies. |

The `hip_fused_host` backend is available through `--split-paf-backend hip_fused_host`, but it is not the current recommended path. The unfused `hip_host` path interleaves better with MIGraphX inference and won the latest stream experiments.

## Setup

Install Python dependencies and make sure ROCm/MIGraphX are available:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt

python -c "import migraphx; print('MIGraphX OK')"
```

If MIGraphX is installed under `/opt/rocm` but Python cannot import it, add the ROCm library path to the active environment:

```bash
python - <<'PY2'
import site
from pathlib import Path

site_dir = Path(site.getsitepackages()[0])
(site_dir / "rocm-migraphx.pth").write_text("/opt/rocm/lib\n")
print(site_dir / "rocm-migraphx.pth")
PY2
```

The common validated input geometry is:

```text
model input:       3 x 544 x 968
low-res outputs:   heatmaps [18, 68, 121], PAFs [38, 68, 121]
stream videos:     1920 x 1080 or 1280 x 720 CCTV-style sources
```

## Model And Data Artifacts

Typical files used by the latest runs:

```text
models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr
models/fp16_refinment1.onnx
coco/val2017/
coco/annotations/person_keypoints_val2017.json
cctv_1280x720_24fps_1.mp4
cctv_1280x720_24fps_original.mp4
cctv_1280x720_24fps_2.mp4
cctv_1280x720_24fps_3.mp4
```

If the B2 split pose-adapter MXR is missing, export and compile it from the pose ONNX:

```bash
python tools/export_split_pose_adapter.py \
  --pose-onnx models/fp16_refinment1.onnx \
  --batch-size 2 \
  --output-onnx models/split_pose_adapter/pose_adapter_b2_1080x1920.onnx \
  --output-mxr models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr \
  --compile
```

## Rerun Recipes

### Clean Environment Before Stream Runs

Use this before reproducing stream experiments. It removes environment-level overrides so the command-line flags define the run.

```bash
unset STREAM_SPLIT_HIP_MIN_PAF_SCORE
unset STREAM_SPLIT_HIP_SUCCESS_RATIO_THR
unset STREAM_SPLIT_HIP_MIN_PAIR_SCORE
unset STREAM_SPLIT_HIP_POINTS_PER_LIMB

unset STREAM_POSE_MIN_PAIR_SCORE
unset STREAM_POSE_MIN_KEYPOINTS
unset STREAM_POSE_MIN_AVG_SCORE
unset STREAM_POSE_FRAME_HEIGHT
unset STREAM_POSE_MAX_LIMB_RATIO
unset STREAM_POSE_MAX_BBOX_WIDTH_RATIO
unset STREAM_POSE_MIN_BBOX_HEIGHT_RATIO
unset STREAM_POSE_MAX_BBOX_ASPECT
unset STREAM_POSE_REQUIRE_TORSO_ANCHOR
```

### Best Current 8-Camera Grid-Video Demo Run

This is the practical visual/demo run based on the latest local command shape. It is useful for producing a 2x4 monitor video plus detailed per-frame timing CSV. It is not the exact 130-second 10-camera benchmark that produced the 75.92 FPS headline result.

```bash
mkdir -p outputs/normal_more_workers

unset STREAM_SPLIT_HIP_MIN_PAF_SCORE
unset STREAM_SPLIT_HIP_SUCCESS_RATIO_THR
unset STREAM_SPLIT_HIP_MIN_PAIR_SCORE
unset STREAM_SPLIT_HIP_POINTS_PER_LIMB

unset STREAM_POSE_MIN_PAIR_SCORE
unset STREAM_POSE_MIN_KEYPOINTS
unset STREAM_POSE_MIN_AVG_SCORE
unset STREAM_POSE_FRAME_HEIGHT
unset STREAM_POSE_MAX_LIMB_RATIO
unset STREAM_POSE_MAX_BBOX_WIDTH_RATIO
unset STREAM_POSE_MIN_BBOX_HEIGHT_RATIO
unset STREAM_POSE_MAX_BBOX_ASPECT
unset STREAM_POSE_REQUIRE_TORSO_ANCHOR

python simulate_camera_stream.py \
  --model models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr \
  --variant split_hip2_host_smart \
  --migraphx-batch-size 2 \
  --migraphx-batch-timeout-ms 2 \
  --num-cameras 8 \
  --frames-per-camera 0 \
  --duration-s 60 \
  --realtime \
  --camera-fps 24 \
  --buffer-mode latest \
  --backpressure-mode soft \
  --infer-workers 1 \
  --post-workers 6 \
  --shared-input-slots 8 \
  --shared-map-slots 24 \
  --shared-dtype float32 \
  --shared-input-dtype float32 \
  --split-mxr2-batch-size 2 \
  --split-batch-timeout-ms 2 \
  --split-paf-backend hip_host \
  --smart-proposals 32 \
  --smart-local-radius 4 \
  --smart-lowres-nms-radius 1 \
  --max-keypoints 15 \
  --threshold 0.15 \
  --pin-cpus \
  --pin-camera-base 0 \
  --pin-inference-base 10 \
  --pin-post-base 12 \
  --grid-video outputs/normal_more_workers/b2_t2_p6_k15_thr015_default_paf_default_pose_grid_2x4.mp4 \
  --grid-rows 2 \
  --grid-cols 4 \
  --grid-cell-width 640 \
  --grid-cell-height 360 \
  --grid-video-fps 10 \
  --detailed-csv outputs/normal_more_workers/b2_t2_p6_k15_thr015_default_paf_default_pose_detailed.csv \
  --summary-json outputs/normal_more_workers/b2_t2_p6_k15_thr015_default_paf_default_pose_summary.json
```

### Best Confirmed 10-Camera Benchmark Run

This is the configuration that matches the final confirmed B2/t2/P2 stream result: 75.92 aggregate FPS, 88.46 ms average E2E latency, and 109.42 ms P95 E2E latency.

```bash
mkdir -p outputs/split_hip2_best

unset STREAM_SPLIT_HIP_MIN_PAF_SCORE
unset STREAM_SPLIT_HIP_SUCCESS_RATIO_THR
unset STREAM_SPLIT_HIP_MIN_PAIR_SCORE
unset STREAM_SPLIT_HIP_POINTS_PER_LIMB

unset STREAM_POSE_MIN_PAIR_SCORE
unset STREAM_POSE_MIN_KEYPOINTS
unset STREAM_POSE_MIN_AVG_SCORE
unset STREAM_POSE_FRAME_HEIGHT
unset STREAM_POSE_MAX_LIMB_RATIO
unset STREAM_POSE_MAX_BBOX_WIDTH_RATIO
unset STREAM_POSE_MIN_BBOX_HEIGHT_RATIO
unset STREAM_POSE_MAX_BBOX_ASPECT
unset STREAM_POSE_REQUIRE_TORSO_ANCHOR

python simulate_camera_stream.py \
  --model models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr \
  --variant split_hip2_host_smart \
  --migraphx-batch-size 2 \
  --migraphx-batch-timeout-ms 2 \
  --num-cameras 10 \
  --frames-per-camera 0 \
  --duration-s 130 \
  --realtime \
  --camera-fps 24 \
  --buffer-mode latest \
  --backpressure-mode soft \
  --infer-workers 1 \
  --post-workers 2 \
  --shared-input-slots 10 \
  --shared-map-slots 16 \
  --shared-dtype float32 \
  --shared-input-dtype float32 \
  --split-mxr2-batch-size 2 \
  --split-batch-timeout-ms 2 \
  --split-paf-backend hip_host \
  --smart-proposals 32 \
  --smart-local-radius 4 \
  --smart-lowres-nms-radius 1 \
  --max-keypoints 20 \
  --threshold 0.1 \
  --pin-cpus \
  --pin-camera-base 0 \
  --pin-inference-base 10 \
  --pin-post-base 12 \
  --detailed-csv outputs/split_hip2_best/b2_t2_p2_soft_130s_detailed.csv \
  --summary-json outputs/split_hip2_best/b2_t2_p2_soft_130s_summary.json
```

### Accuracy Validation With Auto-Compilation

Use `--compile-missing-postprocess-heads` for manual/fused/fused-pruned postprocess heads. Use `--split-mxr2-auto-compile` only for the older `split_hip_host_smart` MXR2 path. Split MXR2 is statically compiled for one selected full-resolution shape, so pair it with `--image-selection dominant-dimensions`.

```bash
python accuracy_validation.py \
  --models models/split_pose_adapter/pose_adapter_b2_1080x1920.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017 \
  --output-dir outputs/accuracy_validation_split_smart \
  --image-selection dominant-dimensions \
  --max-images 1000 \
  --validation-batch-size 2 \
  --variants split_hip_host_smart \
  --split-mxr2-auto-compile \
  --split-mxr2-batch-size 2 \
  --compile-missing-postprocess-heads \
  --max-keypoints 20 \
  --threshold 0.1 \
  --smart-proposals 32 \
  --smart-local-radius 4 \
  --smart-lowres-nms-radius 1
```

For smart-full-res fused-pruned accuracy validation, use the smart wrapper and the postprocess-head autocompiler:

```bash
python accuracy_validation_smart.py \
  --models pose_model1_fp16_ref1.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017 \
  --output-dir outputs/accuracy_validation_smart_fullres \
  --image-selection dominant-dimensions \
  --max-images 1000 \
  --variants merged_fused_pruned \
  --fused-pruned-heatmap-mode smart-full-res \
  --smart-proposals 64 \
  --smart-local-radius 8 \
  --smart-lowres-nms-radius 1 \
  --compile-missing-postprocess-heads
```

### Speed Validation With Auto-Compilation

`speed_validation.py` uses `--variants` plural. Use `--compile-missing-postprocess-heads` for manual/fused/fused-pruned heads. Add `--compile-migraphx-nms` only when the requested variants include `migraphx_nms` or `migraphx_nms_k20`.

```bash
python speed_validation.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --warmup 5 \
  --variants standard optimized_batch_k20_fast gpu_nms_fullres_two_process \
  --compile-missing-postprocess-heads \
  --csv outputs/speed_validation_summary.csv \
  --json outputs/speed_validation_summary.json
```

MIGraphX-NMS example with NMS-head compilation:

```bash
python speed_validation.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --warmup 5 \
  --variants migraphx_nms migraphx_nms_k20 \
  --compile-migraphx-nms \
  --migraphx-nms-cache-dir models/nms_fullres_cache \
  --csv outputs/speed_validation_migraphx_nms.csv \
  --json outputs/speed_validation_migraphx_nms.json
```

## Command-Line Parameters

This section documents the parser flags accepted by the three main entrypoints. The flags are grouped by purpose so the tables stay readable.

### `accuracy_validation.py`

`accuracy_validation.py` calls `accuracy_validation_core.parse_args()`.

#### Model, Data, And Output

| Flag | Default / choices | Meaning |
|---|---|---|
| `--models` | `pose_model1_fp16_ref1.mxr` | One or more MIGraphX model files to evaluate. |
| `--model` | none | Single-model alias stored as `single_model`. |
| `--onnx` | empty | ONNX source used when compiling a missing model MXR. |
| `--quantization` | `fp16`; choices `fp32`, `fp16`, `bf16`, `int8` | Quantization mode used for ONNX-to-MXR compilation. |
| `--exhaustive-tune` | false | Enable MIGraphX exhaustive tuning during compilation. |
| `--labels` | `coco/annotations/person_keypoints_val2017.json` | COCO keypoint annotation JSON. |
| `--images-folder` | `coco/val2017/` | Directory containing validation images. |
| `--output-dir` | `outputs/accuracy_validation` | Directory for detections and summary files. |
| `--max-images` | `5000` | Maximum selected images. |
| `--num-of-test-img` | none | Optional selected-image count override. |
| `--image-selection` | `sequential`; choices `sequential`, `dominant-dimensions` | Image subset selection strategy. |
| `--skip-images` | `0` | Skip this many selected images before validation. |
| `--progress-every` | `20` | Print progress after this many images. |
| `--power-every` | `10` | Power-sampling interval in images. |
| `--validation-batch-size` | `0` | Static validation batch size. `0` infers from model input. |

#### Geometry And Runtime

| Flag | Default / choices | Meaning |
|---|---|---|
| `--base-height` | `544` | Input canvas height before stride alignment. |
| `--base-width` | `968` | Input canvas width before stride alignment. |
| `--stride` | `8` | Model output stride. |
| `--variants` | `standard optimized_batch_k20_fast gpu_nms_fullres_two_process` | Postprocess variants or aliases to evaluate. |
| `--torch-device` | `auto`; choices `auto`, `cuda`, `cpu` | Torch device used by GPU/Torch postprocess paths. |
| `--require-gpu` | false | Fail if the requested runtime cannot use GPU. |

#### Postprocess Controls

| Flag | Default / choices | Meaning |
|---|---|---|
| `--max-keypoints` | `20` | Maximum keypoints retained per type for K-limited paths. |
| `--threshold` | `0.1` | Heatmap candidate threshold. |
| `--nms-radius-fullres` | `6` | Full-resolution NMS radius. |
| `--nms-radius-lowres` | `1` | Low-resolution NMS radius. |
| `--nms-impl` | `separable`; choices `2d`, `separable` | NMS implementation. |
| `--gpu-compute-dtype` | `float32`; choices `float32`, `float16` | GPU postprocess compute dtype. |
| `--points-per-limb` | `8` | Number of PAF samples along each limb. |
| `--min-paf-score` | `0.05` | Minimum per-point PAF score. |
| `--success-ratio-thr` | `0.8` | Minimum ratio of valid PAF samples for a limb pair. |

#### MIGraphX And Fused Head Controls

| Flag | Default / choices | Meaning |
|---|---|---|
| `--migraphx-nms-mxr` | empty | Explicit compiled MIGraphX NMS MXR. |
| `--migraphx-nms-cache-dir` | empty | Directory for cached MIGraphX NMS heads. |
| `--compile-missing-postprocess-heads` | false | Auto-compile missing manual/fused/fused-pruned postprocess heads. |
| `--force-compile-postprocess-heads` | false | Recompile heads even if cached files exist. |
| `--keep-postprocess-onnx` | false | Keep generated ONNX files for compiled postprocess heads. |
| `--migraphx-manual-cubic-topk-mxr` | empty | Explicit manual cubic resize/NMS/TopK MXR. |
| `--migraphx-manual-cubic-topk-cache-dir` | `models/manual_cubic_nms_topk_cache` | Cache directory for manual cubic TopK heads. |
| `--manual-cubic-topk` | `20` | TopK used by manual cubic TopK heads. |
| `--manual-cubic-threshold` | `0.1` | Threshold used by manual cubic TopK heads. |
| `--manual-cubic-nms-radius` | `6` | NMS radius for manual cubic TopK heads. |
| `--manual-cubic-nms-impl` | `separable`; choices `2d`, `separable` | NMS implementation for manual cubic heads. |
| `--manual-cubic-a` | `-0.75` | Cubic interpolation coefficient. |
| `--fused-postprocess-mxr` | empty | Explicit fused postprocess MXR. |
| `--fused-postprocess-cache-dir` | `models/fused_postprocess_cache` | Cache directory for fused postprocess heads. |
| `--fused-pruned-postprocess-mxr` | empty | Explicit fused-pruned postprocess MXR. |
| `--fused-pruned-postprocess-cache-dir` | `models/fused_postprocess_pruned_cache` | Cache directory for fused-pruned heads. |

#### Split Smart Controls

| Flag | Default / choices | Meaning |
|---|---|---|
| `--split-mxr2` | empty | Explicit MXR2 PAF-pruning model for `split_hip_host_smart`. |
| `--split-mxr2-cache-dir` | `models/split_paf_pruning_from_topk` | Cache for split MXR2 files. |
| `--split-mxr2-auto-compile` | false | Auto-export/compile split MXR2 for the selected dominant COCO shape. |
| `--split-mxr2-batch-size` | `4` | Static batch size of split MXR2. |
| `--smart-proposals` | `32` | Low-res proposals per keypoint type before local refinement. |
| `--smart-local-radius` | `4` | Full-res local refinement radius around each proposal. |
| `--smart-lowres-nms-radius` | `1` | Low-res NMS radius for smart proposal selection. |
| `--limb-topm` | `20` | Number of limb-pair candidates retained per limb type. |
| `--min-pair-score` | `0.0` | Minimum score for retained pruned limb pairs. |
| `--paf-cubic-a` | `-0.75` | Cubic interpolation coefficient for PAF sampling. |

#### `accuracy_validation_smart.py` Wrapper Add-Ons

`accuracy_validation_smart.py` reuses the core accuracy parser and pre-parses a few smart-full-res options before delegating to `accuracy_validation_core.parse_args()`.

| Flag | Default / choices | Meaning |
|---|---|---|
| `--fused-pruned-heatmap-mode` | `full-res`; choices `full-res`, `smart-full-res` | Select full-res or smart-full-res heatmap candidate generation for fused-pruned heads. |
| `--smart-proposals` | `64` | Low-res proposal count used by the smart-full-res wrapper. |
| `--smart-local-radius` | `8` | Full-res local refinement radius used by the smart-full-res wrapper. |
| `--smart-lowres-nms-radius` | `1` | Low-res NMS radius used by the smart-full-res wrapper. |

### `speed_validation.py`

Important: this script uses `--variants` plural.

#### Input, Output, And Geometry

| Flag | Default / choices | Meaning |
|---|---|---|
| `--video` | `cctv_1280x720_24fps_3.mp4` | Input video. |
| `--model` | `pose_model1_fp16_ref1.mxr` | Main MIGraphX pose model. |
| `--pose-postprocessing-merged-model` | empty | Merged pose+fused-pruned MXR. Defaults to `--model` for merged variants. |
| `--frames` | `100` | Measured frames. |
| `--warmup` | `5` | Warmup frames discarded before timing. |
| `--target-width` | `968` | Model input width. |
| `--target-height` | `544` | Model input height. |
| `--stride` | `8` | Model stride. |
| `--print-every` | `10` | Progress print interval. |
| `--csv` | `outputs/speed_validation_summary.csv` | Summary CSV output. |
| `--json` | `outputs/speed_validation_summary.json` | Summary JSON output. |

#### Runtime And Postprocess

| Flag | Default / choices | Meaning |
|---|---|---|
| `--variants` | `standard optimized_batch_k20_fast gpu_nms_fullres_two_process` | Postprocess variants or aliases. |
| `--torch-device` | `auto`; choices `auto`, `cuda`, `cpu` | Torch device for Torch/GPU postprocess variants. |
| `--require-gpu` | false | Fail if GPU is unavailable for GPU-required variants. |
| `--max-keypoints` | `20` | Maximum keypoints per type. |
| `--threshold` | `0.1` | Heatmap threshold. |
| `--nms-radius-fullres` | `6` | Full-res NMS radius. |
| `--nms-radius-lowres` | `1` | Low-res NMS radius. |
| `--points-per-limb` | `8` | PAF samples per limb. |
| `--min-paf-score` | `0.05` | Minimum per-sample PAF score. |
| `--success-ratio-thr` | `0.8` | Valid PAF sample ratio threshold. |
| `--two-process-slots` | `3` | Shared slots for two-process validation paths. |
| `--shared-dtype` | `float32`; choices `float32`, `float16` | Shared-memory dtype. |
| `--gpu-compute-dtype` | `float32`; choices `float32`, `float16` | GPU postprocess compute dtype. |
| `--nms-impl` | `separable`; choices `2d`, `separable` | NMS implementation. |
| `--prealloc-resize-buffers` | false | Reuse OpenCV resize destination buffers where supported. |

#### MIGraphX NMS

| Flag | Default / choices | Meaning |
|---|---|---|
| `--migraphx-nms-mxr` | empty | Explicit compiled NMS MXR. |
| `--migraphx-nms-cache-dir` | empty | Cache directory for compiled NMS heads. |
| `--compile-migraphx-nms` | false | Compile missing NMS head for the video shape. |
| `--force-compile-migraphx-nms` | false | Recompile NMS head even if present. |
| `--keep-migraphx-nms-onnx` | false | Keep generated NMS ONNX. |
| `--exhaustive-tune-migraphx-nms` | false | Use exhaustive tuning for NMS compilation. |

#### Manual/Fused/Fused-Pruned Heads

| Flag | Default / choices | Meaning |
|---|---|---|
| `--compile-missing-postprocess-heads` | false | Compile missing manual/fused/fused-pruned heads for the video shape. |
| `--force-compile-postprocess-heads` | false | Recompile cached heads. |
| `--keep-postprocess-onnx` | false | Keep generated postprocess ONNX files. |
| `--migraphx-manual-cubic-topk-mxr` | empty | Explicit manual cubic TopK MXR. |
| `--migraphx-manual-cubic-topk-cache-dir` | `models/manual_cubic_nms_topk_cache` | Manual cubic TopK cache. |
| `--manual-cubic-topk` | `20` | Manual cubic TopK value. |
| `--manual-cubic-threshold` | `0.1` | Manual cubic heatmap threshold. |
| `--manual-cubic-nms-radius` | `6` | Manual cubic NMS radius. |
| `--manual-cubic-nms-impl` | `separable`; choices `2d`, `separable` | Manual cubic NMS implementation. |
| `--manual-cubic-a` | `-0.75` | Cubic interpolation coefficient. |
| `--fused-postprocess-mxr` | empty | Explicit fused postprocess MXR. |
| `--fused-postprocess-cache-dir` | `models/fused_postprocess_cache` | Fused postprocess cache. |
| `--fused-pruned-postprocess-mxr` | empty | Explicit fused-pruned postprocess MXR. |
| `--fused-pruned-postprocess-cache-dir` | `models/fused_postprocess_pruned_cache` | Fused-pruned postprocess cache. |
| `--limb-topm` | `20` | TopM retained limb candidates. |
| `--min-pair-score` | `0.0` | Minimum retained pair score. |
| `--paf-cubic-a` | `-0.75` | Cubic coefficient for PAF sampling. |

### `simulate_camera_stream.py`

`simulate_camera_stream.py` imports `simulation.split_hip_fused_patch.apply_patch()` before building the CLI. The full CLI is therefore the base stream parser plus the split HIP smart patch and fused-backend choice.

#### Stream Inputs And Scheduling

| Flag | Default / choices | Meaning |
|---|---|---|
| `--model` | `pose_model1_fp16_ref1.mxr` | MIGraphX model or split pose-adapter MXR. |
| `--variant` | `gpu_nms_fullres_two_process` | Stream postprocess/runtime variant. |
| `--videos` | default CCTV cycle | Source videos used by simulated cameras. |
| `--num-cameras` | `10` | Number of simulated cameras. |
| `--frames-per-camera` | `100` | Frames per camera. `0` means run until duration/interruption. |
| `--duration-s` | `0.0` | Optional wall-clock duration. `0` disables duration limit. |
| `--realtime` | false | Throttle camera input to `--camera-fps`. |
| `--camera-fps` | `24.0` | Source FPS used by realtime camera simulation. |
| `--queue-policy` | `drop`; choices `drop`, `block` | Behavior when queues are full. |
| `--buffer-mode` | `latest`; choices `latest`, `queue` | Latest-frame slots or FIFO queues between stages. |
| `--disable-backpressure` | false | Legacy alias for `--backpressure-mode off`. |
| `--backpressure-mode` | `strict`; choices `off`, `strict`, `soft` | Backpressure policy in latest mode. |
| `--max-pending-age-ms` | `300.0` | Soft-backpressure freshness cutoff. |
| `--target-output-fps-per-camera` | `0.0` | Per-camera output throttle. `0` disables throttling. |

#### Worker And Batch Controls

| Flag | Default / choices | Meaning |
|---|---|---|
| `--infer-workers` | `1` | Number of inference workers. |
| `--post-workers` | `1` | Number of postprocess workers. |
| `--mp-start-method` | `spawn`; choices `spawn`, `fork`, `forkserver` | Multiprocessing start method. |
| `--migraphx-batch-size` | `1` | Static MIGraphX inference batch size. |
| `--migraphx-batch-timeout-ms` | `0.0` | Inference worker wait time to fill a batch. |
| `--collector-coalesce` / `--no-collector-coalesce` | enabled | Drain/coalesce newest records before batch assembly. |
| `--collector-policy` | `strict_timeout`; choices `strict_timeout`, `freshness_first`, `balanced_fill` | Latest-mode batch launch policy. |
| `--collector-freshness-budget-ms` | `0.0` | Early-launch freshness budget for collector policies. |
| `--collector-empty-scan-grace-ms` | `0.5` | Grace window for `balanced_fill`. |
| `--collector-min-early-batch-size` | `0` | Minimum early batch size for `balanced_fill`. |
| `--preprocess-queue-size` | `30` | Preprocess queue size. |
| `--postprocess-queue-size` | `30` | Postprocess queue size. |

#### Geometry, Shared Memory, And Device

| Flag | Default / choices | Meaning |
|---|---|---|
| `--target-width` | `968` | Model input width. |
| `--target-height` | `544` | Model input height. |
| `--stride` | `8` | Model stride. |
| `--shared-dtype` | `float32`; choices `float32`, `float16` | Shared heatmap/PAF map dtype. |
| `--shared-map-slots` | `0` | Shared heatmap/PAF slots between inference and postprocess. |
| `--shared-input-slots` | `0` | Shared preprocessed input slots between camera/preprocess and inference. |
| `--shared-input-dtype` | `float32`; choices `float32`, `float16` | Shared camera input dtype. |
| `--torch-device` | `auto`; choices `auto`, `cuda`, `cpu` | Torch device for Torch/GPU postprocess stages. |
| `--require-gpu` | false | Fail when GPU is unavailable. |

#### Postprocess And NMS

| Flag | Default / choices | Meaning |
|---|---|---|
| `--max-keypoints` | `20` | Maximum keypoints per type. |
| `--threshold` | `0.1` | Heatmap threshold. |
| `--nms-radius-fullres` | `6` | Full-res NMS radius. |
| `--nms-radius-lowres` | `1` | Low-res NMS radius. |
| `--nms-impl` | `separable`; choices `2d`, `separable` | NMS implementation. |
| `--gpu-compute-dtype` | `float32`; choices `float32`, `float16` | GPU compute dtype. |
| `--prealloc-resize-buffers` | false | Reuse resize output buffers when supported. |
| `--gpu-nms-batch-size` | `1` | Batch size for latest-mode GPU NMS post workers. |
| `--gpu-nms-batch-timeout-ms` | `0.0` | Wait time to fill GPU NMS postprocess batches. |

#### MIGraphX NMS

| Flag | Default / choices | Meaning |
|---|---|---|
| `--migraphx-nms-mxr` | empty | Explicit compiled MIGraphX NMS MXR. |
| `--migraphx-nms-cache-dir` | `models/nms_fullres_cache` | Cache directory for NMS heads. |
| `--compile-migraphx-nms` | false | Compile stream-resolution NMS head before the stream starts. |
| `--force-compile-migraphx-nms` | false | Recompile even if cached NMS head exists. |
| `--keep-migraphx-nms-onnx` | false | Keep generated NMS ONNX. |
| `--exhaustive-tune-migraphx-nms` | false | Use exhaustive tuning for NMS compilation. |

#### Split HIP Smart Patch

| Flag | Default / choices | Meaning |
|---|---|---|
| `--split-mxr2` | default split MXR2 path | MXR2 model for `split_hip_host_smart`; not required for `split_hip2_host_smart`. |
| `--split-mxr2-batch-size` | `4` | Static batch size of split postprocess stage. |
| `--split-batch-timeout-ms` | `4.0` | Wait time to fill split postprocess batch. |
| `--split-paf-backend` | `hip_host`; choices include `hip_host`, `hip_fused_host` | PAF pruning backend for `split_hip2_host_smart`. Use `hip_host` for best current results. |
| `--smart-proposals` | `32` | Smart heatmap proposals per keypoint type. |
| `--smart-local-radius` | `4` | Full-res local refinement radius. |
| `--smart-lowres-nms-radius` | `1` | Low-res NMS radius for smart proposals. |

Stream note: `--compile-migraphx-nms` applies to MIGraphX-NMS variants. The best `split_hip2_host_smart` path uses the B2 split pose-adapter MXR plus HIP PAF pruning, so it does not expose stream-side `--split-mxr2-auto-compile`.

#### Grid Video, Pinning, Profiling, And Outputs

| Flag | Default / choices | Meaning |
|---|---|---|
| `--grid-video` | empty | Optional output path for monitor-style grid video. |
| `--grid-rows` | `4` | Grid video rows. |
| `--grid-cols` | `4` | Grid video columns. |
| `--grid-cell-width` | `480` | Grid cell width. |
| `--grid-cell-height` | `270` | Grid cell height. |
| `--grid-video-fps` | `10.0` | Output grid video FPS. |
| `--grid-video-codec` | `mp4v` | OpenCV video codec. |
| `--grid-queue-size` | `256` | Grid writer queue size. |
| `--pin-cpus` | false | Pin camera, inference, and postprocess workers. |
| `--pin-camera-base` | `0` | First CPU core for camera workers. |
| `--pin-inference-base` | `10` | First CPU core for inference workers. |
| `--pin-post-base` | `12` | First CPU core for postprocess workers. |
| `--pin-all-threads` | false | Also pin native child threads. |
| `--worker-threads` | `1` | CPU thread-pool size per worker. |
| `--warmup-s` | `0.0` | Discard rows completed during initial warmup seconds. |
| `--warmup-output-frames` | `0` | Discard this many earliest output rows. |
| `--profile-system` | false | Sample parent-side CPU, memory, GPU, and VRAM stats. |
| `--profile-interval-s` | `0.1` | System profile sampling interval. |
| `--report-affinity` | false | Print worker CPU affinity. |
| `--roctx` | false | Emit ROCTx ranges for rocprofv3. |
| `--trace-log-every` | `0` | Print per-worker timing trace every N outputs. |
| `--allow-ptrace-attach` | false | Allow same-user profiler attach on ptrace-restricted systems. |
| `--print-every` | `100` | Stream progress print interval. |
| `--detailed-csv` | `outputs/stream_10cam_detailed.csv` | Per-frame detailed timing output. |
| `--summary-json` | `outputs/stream_10cam_summary.json` | Stream summary output. |

## Output Files

Common stream outputs:

```text
outputs/.../*_detailed.csv
outputs/.../*_summary.json
outputs/.../*_grid_*.mp4
```

The detailed CSV contains per-frame timing columns such as:

```text
camera_id
frame_id
source
variant
preprocess_ms
queue_pre_to_infer_ms
inference_ms
decode_ms
queue_infer_to_post_ms
post_ms
e2e_ms
num_poses
num_keypoints
timing_split_smart_heatmap
timing_split_hip2
timing_split_cpu_assembly
timing_total_postprocess
```

The summary JSON reports aggregate FPS, per-camera FPS, average and P95 E2E latency, average postprocess time, worker stats, batch stats, and optional system-profile metrics.

## Known Limitations

- The latest 75.92 FPS result is a stream-performance result, not a new full COCO AP claim by itself.
- Smart-full-res fused-pruned AP parity is validated on the documented 1000-image dominant-resolution COCO subset.
- The split MXR2 path is shape-specific and needs one selected full-resolution shape when auto-compiling for accuracy validation.
- The best split HIP2 stream path currently still has a CPU pose assembly tail.
- Full 10 x 24 FPS output is not reached by the current best configuration; the source cameras run at 24 FPS, while the system emits a lower-latency sampled live-monitoring stream.

## Main References Inside This Repo

| File | What to read it for |
|---|---|
| `reports/hip2_b2_and_fusion_experiment_report.md` | Latest HIP2, B2/t2, and final stream benchmark result. |
| `reports/smart_fullres_fused_pruned_report.md` | Smart-full-res accuracy and stream speedup. |
| `reports/split_hip_smart_stream_and_mxr2_profile_report.md` | Previous MXR2 split-stream result and bottleneck diagnosis. |
| `docs/deep-research-report.md` | Bridge report from stream collapse to focused MIGraphX pipeline tuning. |
| `docs/stream_simulation_grid_search_report.md` | Older multi-camera grid-search comparison. |
| `modules/postprocessing.py` | Canonical postprocess modes and shared postprocess implementation. |
| `simulation/cli.py` | Base stream simulator parser. |
| `simulation/split_hip_smart_patch.py` | Split HIP smart runtime patch and CLI additions. |
| `simulation/split_hip_fused_patch.py` | Optional fused HIP backend patch. |
