# Lightweight Human Pose Estimation on AMD ROCm + MIGraphX

This repository contains an AMD ROCm / MIGraphX-oriented optimization of the Lightweight OpenPose-style human pose estimation pipeline. The project started from the classical `lightweight-human-pose-estimation.pytorch` implementation and focused on making the full end-to-end pipeline usable for video workloads on AMD hardware.

The central finding is that, after MIGraphX accelerates neural network inference, the main bottleneck is no longer the model itself. The expensive part becomes postprocessing: heatmap resize, PAF resize, keypoint extraction, pose grouping, and the runtime architecture around those stages.

This README summarizes the final validated state of the work and explains how the repository is organized and used.

---

## High-level goal

The optimization work targets the full runtime path:

```text
frame/image
  -> preprocessing
  -> MIGraphX model inference
  -> decode heatmaps + PAFs
  -> postprocessing
       -> heatmap NMS / peak extraction
       -> PAF parsing / pose grouping
  -> final poses / detections
```

The work compares multiple optimization directions:

- CPU-side postprocessing optimizations
- GPU-based postprocessing experiments
- MIGraphX / compiled NMS experiments
- single-process and two-process runtime architectures
- final COCO accuracy validation and CCTV video speed validation

The separate live multi-camera simulation work exists as an additional system-level story and is not treated as the main conclusion path in this README.

---

## Final conclusions

The final validation shows three practical conclusions.

First, the original CPU postprocessing path is correct but too slow. It preserves reference accuracy, but in the final CCTV benchmark it reaches only about **4.34 FPS** end-to-end, with postprocessing alone taking about **219 ms/frame**.

Second, the best CPU fallback is the `k20_fast` / `findNonZero v1 + k20` direction. It keeps the same COCO AP as the standard baseline while reducing postprocessing and end-to-end latency substantially. This is the best path when the runtime should avoid additional GPU postprocessing dependencies.

Third, the best accuracy-preserving runtime direction is the full-resolution GPU-NMS path, especially in the two-process architecture. The final validated `gpu_nms_fullres_two_process` variant preserves the baseline COCO AP and gives the best practical speed/accuracy tradeoff.

### Recommended variants

| Use case | Recommended variant | Reason |
|---|---|---|
| Reference baseline | `standard` | Original CPU extract + CPU group path; used only as correctness and comparison baseline. |
| Best CPU-only fallback | `optimized_batch_k20_fast` / `cpu_k20_fast_two_process` | Preserves baseline AP and removes most of the CPU grouping bottleneck. |
| Best accuracy-preserving GPU runtime | `gpu_nms_fullres_two_process` | Preserves baseline AP and gives the strongest validated end-to-end runtime. |
| Fast but accuracy-degraded path | `gpu_nms_lowres_two_process` | Very fast, but AP drops too much for accuracy-critical deployment. |
| MIGraphX-specific NMS proof-of-concept | `migraphx_nms` / `migraphx_nms_k20` | Validates compiled NMS feasibility, but is not the fastest final path. |

---

## Final validation summary

Final validation was performed with two complementary test types:

- `accuracy_validation.py`: COCO val2017-style validation, reporting AP / AP50 / AP75 / AR and average latency.
- `speed_validation.py`: CCTV video validation over 100 frames, reporting stage-level timing, mean latency, p95 latency, FPS, and pipeline throughput where applicable.

### COCO-style accuracy and latency summary

| Variant | AP | AP50 | AR | Post avg | E2E avg | E2E FPS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `standard` | 0.3995 | 0.6706 | 0.4603 | 63.68 ms | 82.77 ms | 12.08 | Reference baseline |
| `optimized_batch_k10` | 0.3777 | 0.6343 | 0.4356 | 50.38 ms | 69.47 ms | 14.40 | Faster, but AP drop |
| `optimized_batch_k20` | 0.3995 | 0.6706 | 0.4603 | 49.82 ms | 68.91 ms | 14.51 | Accuracy-preserving |
| `optimized_batch_k20_fast` | 0.3995 | 0.6706 | 0.4603 | 20.14 ms | 39.23 ms | 25.49 | Best single-process CPU fallback |
| `cpu_k20_fast_two_process` | 0.3995 | 0.6706 | 0.4603 | 21.23 ms | 35.93 ms | 27.83 | CPU fallback in runtime form |
| `gpu_nms_fullres_two_process` | 0.3995 | 0.6706 | 0.4603 | 11.97 ms | 28.07 ms | 35.63 | Best accuracy-preserving runtime |
| `gpu_nms_lowres_two_process` | 0.2479 | 0.5419 | 0.3058 | 3.69 ms | 19.34 ms | 51.70 | Very fast, but degraded AP |
| `migraphx_nms` | 0.4061 | 0.6729 | 0.4661 | 23.29 ms | 37.63 ms | 26.57 | Accuracy-safe, not fastest |
| `migraphx_nms_k20` | 0.3995 | 0.6706 | 0.4603 | 19.54 ms | 31.85 ms | 31.40 | Accuracy-preserving MIGraphX NMS variant |
| `fast_no_resize` | 0.2184 | 0.4690 | 0.2673 | 1.73 ms | 20.81 ms | 48.05 | Too much AP loss |
| `lowres_cpu_group` | 0.2184 | 0.4690 | 0.2673 | 1.14 ms | 20.23 ms | 49.44 | Too much AP loss |

### CCTV speed-validation summary

| Variant | Post avg | E2E avg | E2E p95 | E2E FPS | Pipeline FPS | Status |
|---|---:|---:|---:|---:|---:|---|
| `standard` | 219.27 ms | 230.59 ms | 236.85 ms | 4.34 | - | Reference baseline |
| `optimized_batch_k20` | 154.71 ms | 166.03 ms | 172.35 ms | 6.02 | - | Extraction improved, grouping still expensive |
| `optimized_batch_k20_fast` | 61.67 ms | 72.99 ms | 79.27 ms | 13.70 | - | Best single-process CPU direction |
| `cpu_k20_fast_two_process` | 60.69 ms | 76.42 ms | 80.39 ms | 13.08 | 14.24 | CPU fallback in pipeline form |
| `gpu_nms_fullres_two_process` | 34.07 ms | 47.49 ms | 50.34 ms | 21.06 | 17.41 | Best practical accuracy-preserving runtime |
| `gpu_nms_lowres_two_process` | 8.28 ms | 17.94 ms | 21.40 ms | 55.74 | 32.11 | Fastest, but accuracy-degraded |
| `migraphx_nms` | 75.12 ms | 84.28 ms | 86.49 ms | 11.87 | - | Validated, but slower |
| `migraphx_nms_k20` | 75.66 ms | 84.73 ms | 87.57 ms | 11.80 | - | Similar runtime to `migraphx_nms` |
| `fast_no_resize` | 4.90 ms | 16.22 ms | 17.17 ms | 61.66 | - | Fast but too inaccurate |
| `lowres_cpu_group` | 3.93 ms | 15.25 ms | 16.16 ms | 65.58 | - | Fast but too inaccurate |

---

## Experiment logic

### 1. Baseline and motivation

The baseline path uses:

```text
CPU extract_keypoints + CPU group_keypoints
```

It provides the reference accuracy, but profiling showed that `extract` and `group` dominate runtime after MIGraphX accelerates inference. This motivated all following work.

### 2. CPU-side postprocessing optimizations

The CPU branch kept the original algorithmic idea but reduced extraction and grouping cost.

The logical sequence was:

1. Start from `standard`.
2. Try `optimized_batch_k10` to reduce the number of candidates.
3. Move to `optimized_batch_k20` when K10 caused AP loss.
4. Introduce `k20_fast` / `findNonZero v1 + k20` to reduce candidate extraction and grouping overhead.
5. Test more aggressive shortcuts such as `fast_no_resize` and `lowres_cpu_group` to measure the speed/accuracy boundary.

Conclusion: `optimized_batch_k20_fast` is the best CPU-only direction because it preserves baseline AP while giving a large latency improvement.

### 3. GPU postprocessing experiments

The GPU branch tried to move the most parallel part of postprocessing, heatmap NMS / peak extraction, to the GPU.

The useful path was:

```text
full-resolution GPU NMS + optimized CPU grouping
```

This gave a much better speed/accuracy tradeoff than trying to move too much of the full PAF/grouping work onto the GPU or using low-resolution grouping paths.

Conclusion: full-resolution GPU NMS is the best GPU-side accuracy-preserving direction.

### 4. Runtime architecture experiments

The runtime branch compared single-process execution with a two-process pipeline.

The two-process idea is:

```text
Process 1: frame input + preprocessing + MIGraphX inference
Process 2: postprocessing
Shared memory / queue: transfers decoded outputs between stages
```

This architecture is useful because it allows inference and postprocessing to overlap. The final practical runtime direction is:

```text
gpu_nms_fullres_two_process
```

Conclusion: the two-process architecture is valuable when paired with the full-resolution GPU-NMS path.

### 5. MIGraphX / compiled NMS experiments

The MIGraphX NMS branch checked whether NMS logic can be exported and compiled as a separate MIGraphX graph.

The supporting scripts are:

```text
export_heatmap_nms_head.py
compile_heatmap_nms_migraphx.py
migraphx_nms.py
test_migraphx_nms_sanity.py
benchmark_migraphx_postprocess.py
```

Conclusion: MIGraphX NMS is technically viable and accuracy-safe, but the current implementation is not the fastest overall path because mask-to-keypoint extraction and remaining CPU-side work still cost too much.

---

## Repository organization

The repository keeps the original Lightweight OpenPose structure and adds AMD/MIGraphX-specific runtime, postprocessing, and validation tools.

Typical structure:

```text
.
├── README.md
├── requirements/
│   └── requirements.txt
├── modules/
│   ├── keypoints.py
│   ├── keypoints_gpu_variant.py
│   ├── pose.py
│   ├── load_state.py
│   └── ...
├── models/
│   ├── *.onnx
│   ├── *.mxr
│   └── nms_fullres_cache/
├── outputs/
│   ├── accuracy_validation_*/
│   ├── speed_validation_*/
│   └── stream_*/
├── final_report/
│   ├── final_acc_summary.csv
│   ├── final_acc_summary.json
│   ├── speed_validation_all_merged.csv
│   └── speed_validation_all_merged.json
├── accuracy_validation.py
├── speed_validation.py
├── video_val.py
├── simulate_10_camera_stream.py
├── migraphx_nms.py
├── export_heatmap_nms_head.py
├── compile_heatmap_nms_migraphx.py
├── benchmark_migraphx_postprocess.py
├── test_migraphx_nms_sanity.py
└── README_migraphx_nms.md
```

### Important files

| File | Purpose |
|---|---|
| `accuracy_validation.py` | Runs COCO-style accuracy validation and exports AP/AR summaries. |
| `speed_validation.py` | Runs video speed validation and exports stage-level latency summaries. |
| `video_val.py` | Main video validation / debugging entrypoint for frame-level timing. |
| `simulate_10_camera_stream.py` | Multi-camera live-feed simulation; treated as a separate system story. |
| `modules/keypoints.py` | Original and optimized CPU keypoint extraction/grouping logic. |
| `modules/keypoints_gpu_variant.py` | GPU postprocessing variants and hybrid paths. |
| `migraphx_nms.py` | Runtime wrapper for compiled MIGraphX NMS path. |
| `export_heatmap_nms_head.py` | Exports the heatmap/NMS subgraph. |
| `compile_heatmap_nms_migraphx.py` | Compiles exported NMS graph into MIGraphX `.mxr`. |
| `benchmark_migraphx_postprocess.py` | Benchmarks MIGraphX-related postprocessing variants. |
| `test_migraphx_nms_sanity.py` | Sanity-checks compiled NMS behavior. |
| `final_report/` | Stores merged final accuracy and speed summaries. |
| `outputs/` | Stores generated detections, detailed timing CSVs, summaries, and optional videos. |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ognjen217/lightweight-human-pose-estimation.pytorch-AMD.git
cd lightweight-human-pose-estimation.pytorch-AMD
```

Use the branch that contains the optimization work:

```bash
git fetch --all
git checkout <branch-name>
```

### 2. Create and activate environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
```

If COCO tools or NumPy compatibility cause issues:

```bash
pip install "numpy<2.0"
pip install --no-build-isolation -r requirements/requirements.txt
```

### 3. Install ROCm / MIGraphX

MIGraphX is expected to be installed through the ROCm system packages. After installation, verify that Python can import MIGraphX:

```bash
python -c "import migraphx; print('MIGraphX OK')"
```

If the import fails while MIGraphX is installed under `/opt/rocm`, expose ROCm libraries to the virtual environment. One common workaround is to add the ROCm library path to a `.pth` file inside the active environment:

```bash
python - <<'PY'
import site
from pathlib import Path

site_dir = Path(site.getsitepackages()[0])
(site_dir / "rocm-migraphx.pth").write_text("/opt/rocm/lib\n")
print(site_dir / "rocm-migraphx.pth")
PY
```

Then retry:

```bash
python -c "import migraphx; print('MIGraphX OK')"
```

---

## Data and model preparation

Expected inputs:

```text
pose_model1_fp16_ref1.mxr
coco/
  val2017/
  annotations/
    person_keypoints_val2017.json
cctv_1280x720_24fps_*.mp4
```

The main validated model is:

```text
pose_model1_fp16_ref1.mxr
```

The validation work was performed with the FP16 one-refinement-stage MIGraphX model.

---

## Running accuracy validation

Template:

```bash
python accuracy_validation.py \
  --models pose_model1_fp16_ref1.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017
```

Depending on the branch version, variants may be selected through a variant argument or by running the script for individual configured modes.

Recommended variants for final validation:

```text
standard
optimized_batch_k20
optimized_batch_k20_fast
cpu_k20_fast_two_process
gpu_nms_fullres_two_process
gpu_nms_lowres_two_process
migraphx_nms
migraphx_nms_k20
fast_no_resize
lowres_cpu_group
```

Expected outputs:

```text
outputs/accuracy_validation_<variant>/
  detections_<variant>.json
  accuracy_validation_summary_<variant>.csv
  accuracy_validation_summary_<variant>.json
```

---

## Running speed validation

Template:

```bash
python speed_validation.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100
```

For a specific variant:

```bash
python speed_validation.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --frames 100 \
  --variant gpu_nms_fullres_two_process
```

Recommended speed-validation variants:

```text
standard
optimized_batch_k20
optimized_batch_k20_fast
cpu_k20_fast_two_process
gpu_nms_fullres_two_process
gpu_nms_lowres_two_process
migraphx_nms
migraphx_nms_k20
fast_no_resize
lowres_cpu_group
```

Expected outputs:

```text
speed_validation_<variant>.csv
speed_validation_<variant>.json
```

Merged final outputs:

```text
speed_validation_all_merged.csv
speed_validation_all_merged.json
```

---

## Running MIGraphX NMS experiments

### Export NMS head

```bash
python export_heatmap_nms_head.py
```

### Compile NMS graph with MIGraphX

```bash
python compile_heatmap_nms_migraphx.py
```

### Sanity-check compiled NMS

```bash
python test_migraphx_nms_sanity.py
```

### Benchmark MIGraphX postprocess path

```bash
python benchmark_migraphx_postprocess.py
```

The compiled NMS path is useful for proving that this part of postprocessing can be represented as a MIGraphX graph. Current validation shows that it is accuracy-safe, but not faster than the full-resolution GPU-NMS runtime path.

---

## Running video validation/debug timing

For frame-level timing investigation:

```bash
python video_val.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --model pose_model1_fp16_ref1.mxr \
  --max-frames 100 \
  --no-draw \
  --no-write
```

Typical timing columns include:

```text
Pre
Infer
Decode
HM Resize
PAF Resize
Extract
Group
Post Total
```

These columns are useful for identifying whether the current bottleneck is in inference, resizing, extraction, grouping, or queue/runtime behavior.

---

## Optional: multi-camera simulation

The repository also contains live-feed simulation tooling:

```bash
python simulate_10_camera_stream.py \
  --model pose_model1_fp16_ref1.mxr \
  --variant gpu_nms_fullres_two_process \
  --num-cameras 10 \
  --duration-s 60 \
  --realtime \
  --camera-fps 24
```

This is useful for testing queue behavior, latest-frame buffering, dropped frames, per-camera throughput, and grid monitor output. It is treated as a separate story from the final accuracy/speed report.

---

## Reporting outputs

Recommended reporting files:

```text
final_report/
  final_acc_summary.csv
  final_acc_summary.json
  speed_validation_all_merged.csv
  speed_validation_all_merged.json
  report_cpu_side_postprocessing_optimizations.md
  report_gpu_postprocessing_experiments.md
  report_runtime_architecture_experiments.md
  report_migraphx_compiled_nms_experiments.md
  report_final_accuracy_and_speed_validation.md
  final_optimization_report.md


## Practical decision guide

Use this when selecting a runtime path:

```text
Need maximum accuracy and simple baseline?
  -> standard

Need CPU-only fallback with preserved AP?
  -> optimized_batch_k20_fast
  -> cpu_k20_fast_two_process for pipeline form

Need best practical accuracy-preserving runtime?
  -> gpu_nms_fullres_two_process

Need maximum speed and can accept AP loss?
  -> gpu_nms_lowres_two_process or lowres_cpu_group

Need MIGraphX-only NMS feasibility / compiled graph testing?
  -> migraphx_nms or migraphx_nms_k20
```

Final recommendation:

```text
Use pose_model1_fp16_ref1.mxr with gpu_nms_fullres_two_process
as the main optimized runtime path.

Keep optimized_batch_k20_fast / cpu_k20_fast_two_process
as the stable CPU-side fallback.

Treat low-resolution variants as speed/accuracy tradeoff experiments,
not as final accuracy-preserving deployments.

Treat MIGraphX NMS as a validated compiled-NMS experiment,
not as the fastest final production path in the current implementation.
```

---

## References

This repository is based on the Lightweight OpenPose implementation:

```text
Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
Daniil Osokin, 2018
```

Original upstream project:

```text
osokin/lightweight-human-pose-estimation.pytorch
```
