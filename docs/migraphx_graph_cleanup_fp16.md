# MIGraphX graph cleanup and FP16 output experiment

Branch: `opt/migraphx-graph-cleanup-fp16`

## Goal

This branch isolates graph-level experiments for the pose model:

- inspect ONNX graphs before MIGraphX compilation,
- remove or reduce output-side dequant/cast noise,
- optionally run ONNX simplification,
- compile static MIGraphX batches from arbitrary ONNX files,
- compare kernel counts and speed with the existing baseline.

The main target is to verify whether output-side `DeQuantizeLinear`, `Cast`, `Convert`, `Slice`, `Concat`, or similar small operators create avoidable MIGraphX kernel launch overhead.

## What changed

### 1. Export-only dequant switch

`models/with_mobilenet.py` now has:

```python
self.export_without_dequant = False
```

The default inference path is unchanged.

When `export_without_dequant=True`, `forward()` returns the final heatmap/PAF tensors before the model-level `DeQuantStub`.

### 2. ONNX graph inspector

New script:

```bash
tools/inspect_onnx_graph.py
```

It reports:

- total node count,
- top ONNX operator counts,
- input/output dtypes and shapes,
- counts for `Cast`, `Slice`, `Concat`, `DequantizeLinear`, `QuantizeLinear`, `Identity`, etc.,
- output producer nodes,
- suspicious tail conversion nodes.

### 3. Graph-clean ONNX exporter

New script:

```bash
export_onnx_graph_clean.py
```

It can export:

- normal two-output ONNX,
- dequant-free two-output ONNX,
- optional single concatenated output for profiling only,
- optional `onnxsim` simplified model.

The default output contract remains:

```text
heatmaps, pafs
```

so existing postprocessing can still be used.

### 4. Parameterized static MIGraphX compiler

`compile_migraphx_static_batches.py` now accepts CLI flags instead of requiring manual edits.

It still defaults to:

```text
height=544
width=968
batches=1,2,4,8
fp16=True
```

## Recommended local workflow

### 0. Pull the branch

```bash
cd ~/lightweight-human-pose-estimation.pytorch-AMD
git fetch ognjen
git checkout opt/migraphx-graph-cleanup-fp16
git pull
```

### 1. Export baseline ONNX

```bash
python3 export_dynamic_onnx.py
```

### 2. Inspect baseline ONNX

```bash
python3 tools/inspect_onnx_graph.py \
  pose_model_dynamic.onnx \
  --json-out outputs/graph_inspection/baseline_pose_model_dynamic.json \
  --txt-out outputs/graph_inspection/baseline_pose_model_dynamic.txt
```

### 3. Export dequant-free graph-clean ONNX

```bash
python3 export_onnx_graph_clean.py \
  --checkpoint models/checkpoint_iter_370000.pth \
  --output models/onnx/pose_model_clean_bdyn_no_dequant.onnx \
  --without-dequant \
  --height 544 \
  --width 968 \
  --batch-size 1 \
  --report-json outputs/graph_inspection/export_clean_no_dequant.json
```

### 4. Inspect graph-clean ONNX

```bash
python3 tools/inspect_onnx_graph.py \
  models/onnx/pose_model_clean_bdyn_no_dequant.onnx \
  --json-out outputs/graph_inspection/clean_no_dequant.json \
  --txt-out outputs/graph_inspection/clean_no_dequant.txt
```

### 5. Optional ONNX simplification

Install if needed:

```bash
pip install onnxsim
```

Then export and simplify:

```bash
python3 export_onnx_graph_clean.py \
  --checkpoint models/checkpoint_iter_370000.pth \
  --output models/onnx/pose_model_clean_bdyn_no_dequant.onnx \
  --without-dequant \
  --simplify \
  --simplified-output models/onnx/pose_model_clean_bdyn_no_dequant_sim.onnx \
  --report-json outputs/graph_inspection/export_clean_no_dequant_sim.json
```

Inspect the simplified graph:

```bash
python3 tools/inspect_onnx_graph.py \
  models/onnx/pose_model_clean_bdyn_no_dequant_sim.onnx \
  --json-out outputs/graph_inspection/clean_no_dequant_sim.json \
  --txt-out outputs/graph_inspection/clean_no_dequant_sim.txt
```

### 6. Compile static MIGraphX batches

For the non-simplified graph:

```bash
python3 compile_migraphx_static_batches.py \
  --onnx models/onnx/pose_model_clean_bdyn_no_dequant.onnx \
  --height 544 \
  --width 968 \
  --batches 1 2 4 8 \
  --out-dir models/migraphx_graph_clean \
  --output-prefix pose_model_clean_no_dequant
```

For the simplified graph:

```bash
python3 compile_migraphx_static_batches.py \
  --onnx models/onnx/pose_model_clean_bdyn_no_dequant_sim.onnx \
  --height 544 \
  --width 968 \
  --batches 1 2 4 8 \
  --out-dir models/migraphx_graph_clean \
  --output-prefix pose_model_clean_no_dequant_sim
```

Expected outputs:

```text
models/migraphx_graph_clean/pose_model_clean_no_dequant_b1_fp16.mxr
models/migraphx_graph_clean/pose_model_clean_no_dequant_b2_fp16.mxr
models/migraphx_graph_clean/pose_model_clean_no_dequant_b4_fp16.mxr
models/migraphx_graph_clean/pose_model_clean_no_dequant_b8_fp16.mxr
```

## Validation commands

### Accuracy smoke test

`accuracy_validation.py` accepts `--models`, so old and new MXR files can be compared in one run:

```bash
python3 accuracy_validation.py \
  --models pose_model_b1_fp16.mxr models/migraphx_graph_clean/pose_model_clean_no_dequant_b1_fp16.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017 \
  --variants optimized_batch_k20_fast \
  --max-images 500
```

Then run the target GPU-postprocess variant:

```bash
python3 accuracy_validation.py \
  --models pose_model_b1_fp16.mxr models/migraphx_graph_clean/pose_model_clean_no_dequant_b1_fp16.mxr \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017 \
  --variants gpu_nms_fullres_two_process \
  --max-images 500
```

### Speed validation

`speed_validation.py` accepts a single `--model` argument, so run baseline and graph-clean models separately.

Baseline:

```bash
python3 speed_validation.py \
  --model pose_model_b1_fp16.mxr \
  --video cctv_1280x720_24fps_3.mp4 \
  --variants optimized_batch_k20_fast gpu_nms_fullres_two_process \
  --frames 100 \
  --warmup 5 \
  --csv outputs/graph_clean_speed_baseline.csv \
  --json outputs/graph_clean_speed_baseline.json
```

Graph-clean model:

```bash
python3 speed_validation.py \
  --model models/migraphx_graph_clean/pose_model_clean_no_dequant_b1_fp16.mxr \
  --video cctv_1280x720_24fps_3.mp4 \
  --variants optimized_batch_k20_fast gpu_nms_fullres_two_process \
  --frames 100 \
  --warmup 5 \
  --csv outputs/graph_clean_speed_no_dequant.csv \
  --json outputs/graph_clean_speed_no_dequant.json
```

For the legacy-export model generated during the dynamic-axis fix, use this path instead:

```text
models/migraphx_graph_clean/pose_model_clean_no_dequant_legacy_b1_fp16.mxr
```

## rocprof comparison

Baseline:

```bash
rocprofv3 --kernel-trace --runtime-trace \
  --output-dir outputs/rocprof_graph_clean/baseline \
  python3 speed_validation.py \
    --model pose_model_b1_fp16.mxr \
    --video cctv_1280x720_24fps_3.mp4 \
    --variants optimized_batch_k20_fast \
    --frames 100 \
    --warmup 5
```

Graph-clean model:

```bash
rocprofv3 --kernel-trace --runtime-trace \
  --output-dir outputs/rocprof_graph_clean/clean_no_dequant \
  python3 speed_validation.py \
    --model models/migraphx_graph_clean/pose_model_clean_no_dequant_b1_fp16.mxr \
    --video cctv_1280x720_24fps_3.mp4 \
    --variants optimized_batch_k20_fast \
    --frames 100 \
    --warmup 5
```

## Success criteria

Treat the experiment as successful only if:

- AP/AR are unchanged or negligibly changed,
- inference time improves,
- total end-to-end time does not regress,
- rocprof shows fewer output-side conversion/tail kernels,
- batch 2/4/8 remain compilable and usable for multi-camera testing.

## Notes

Do not optimize ELU topology in this branch. Moving ELU across convolutions changes model mathematics and should be treated as a retraining-level architecture change, not as graph cleanup.

The optional `--concat-outputs` export is profiling-only because it changes the model output contract from two tensors to one tensor.
