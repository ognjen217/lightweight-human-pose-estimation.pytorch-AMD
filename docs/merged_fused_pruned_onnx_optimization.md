# Merged fused-pruned ONNX optimization experiment

Branch: `opt/migraphx-graph-cleanup-fp16`

This workflow targets large generated postprocess graphs, especially the `merged_fused_pruned` / `mx_fused_cubic_topk_fullres_paf_pruned` head.

Unlike the pose model graph, these generated postprocess ONNX graphs contain more non-convolutional structure: resize math, TopK, pair scoring, reshapes, casts, masking, and pruning. They are better candidates for graph-level cleanup.

## 1. Compile/generate the fused-pruned ONNX and MXR

Use `--keep-onnx` so the generated ONNX remains in the cache directory.

```bash
python3 modules/migraphx_fused_postprocess_pruned_compiler.py \
  --video cctv_1280x720_24fps_3.mp4 \
  --target-width 968 \
  --target-height 544 \
  --stride 8 \
  --output-dir models/fused_postprocess_pruned_cache \
  --topk 20 \
  --limb-topm 20 \
  --threshold 0.1 \
  --nms-radius 6 \
  --nms-impl separable \
  --points-per-limb 8 \
  --min-paf-score 0.05 \
  --success-ratio-thr 0.8 \
  --min-pair-score 0.0 \
  --force \
  --keep-onnx
```

The output path is printed by the compiler. The ONNX name usually starts with:

```text
models/fused_postprocess_pruned_cache/fused_cubic_topk_fullres_paf_pruned_68x121_to_720x1280_...
```

## 2. Inspect the original ONNX

Set the path once:

```bash
PRUNED_ONNX="models/fused_postprocess_pruned_cache/<generated-pruned-head>.onnx"
```

Then inspect:

```bash
python3 tools/inspect_onnx_graph.py "$PRUNED_ONNX" \
  --txt-out outputs/graph_inspection/merged_fused_pruned_original.txt \
  --json-out outputs/graph_inspection/merged_fused_pruned_original.json
```

## 3. Optimize the ONNX graph

Conservative cleanup only:

```bash
OPT_ONNX="${PRUNED_ONNX%.onnx}_opt.onnx"

python3 tools/optimize_onnx_for_migraphx.py \
  "$PRUNED_ONNX" \
  "$OPT_ONNX" \
  --report-json outputs/graph_inspection/merged_fused_pruned_opt_report.json
```

More aggressive test with `onnxsim`, if installed:

```bash
SIM_ONNX="${PRUNED_ONNX%.onnx}_sim.onnx"

python3 tools/optimize_onnx_for_migraphx.py \
  "$PRUNED_ONNX" \
  "$SIM_ONNX" \
  --onnxsim \
  --report-json outputs/graph_inspection/merged_fused_pruned_sim_report.json
```

If `onnxsim` is not installed:

```bash
pip install onnxsim
```

## 4. Inspect the optimized ONNX

```bash
python3 tools/inspect_onnx_graph.py "$OPT_ONNX" \
  --txt-out outputs/graph_inspection/merged_fused_pruned_opt.txt \
  --json-out outputs/graph_inspection/merged_fused_pruned_opt.json
```

Compare:

```text
outputs/graph_inspection/merged_fused_pruned_original.json
outputs/graph_inspection/merged_fused_pruned_opt.json
```

Look for reductions in:

```text
Identity
Constant
Shape/Gather/Slice/Unsqueeze
Reshape/Transpose chains
Cast
Concat
Where/Greater chains
Dead-end debug outputs
```

## 5. Compile optimized ONNX to MXR

Do not use `compile_migraphx_static_batches.py` here, because this is not the pose model and its inputs are not named `input`.

```bash
OPT_MXR="${OPT_ONNX%.onnx}.mxr"

python3 tools/compile_onnx_to_migraphx.py \
  --onnx "$OPT_ONNX" \
  --out "$OPT_MXR"
```

Optional FP16 quantization test:

```bash
OPT_FP16_MXR="${OPT_ONNX%.onnx}_fp16.mxr"

python3 tools/compile_onnx_to_migraphx.py \
  --onnx "$OPT_ONNX" \
  --out "$OPT_FP16_MXR" \
  --fp16
```

## 6. Speed validation with explicit optimized postprocess head

Use the regular pose model for inference and override only the fused-pruned postprocess MXR.

```bash
python3 speed_validation.py \
  --model pose_model_b1_fp16.mxr \
  --video cctv_1280x720_24fps_3.mp4 \
  --variants merged_fused_pruned \
  --frames 100 \
  --warmup 5 \
  --fused-pruned-postprocess-mxr "$OPT_MXR" \
  --csv outputs/merged_fused_pruned_opt_speed.csv \
  --json outputs/merged_fused_pruned_opt_speed.json
```

Baseline, using the generated non-optimized head:

```bash
BASE_MXR="${PRUNED_ONNX%.onnx}.mxr"

python3 speed_validation.py \
  --model pose_model_b1_fp16.mxr \
  --video cctv_1280x720_24fps_3.mp4 \
  --variants merged_fused_pruned \
  --frames 100 \
  --warmup 5 \
  --fused-pruned-postprocess-mxr "$BASE_MXR" \
  --csv outputs/merged_fused_pruned_baseline_speed.csv \
  --json outputs/merged_fused_pruned_baseline_speed.json
```

## 7. Success criteria

Treat the optimized graph as useful only if:

- the output contract is unchanged,
- the graph compiles to MXR,
- speed validation succeeds,
- accuracy validation does not regress,
- rocprof shows fewer small kernels or lower runtime overhead.

If the optimized ONNX has fewer nodes but runtime is the same, MIGraphX may already be canonicalizing those patterns internally. In that case, the next target should be graph restructuring, not cleanup.
