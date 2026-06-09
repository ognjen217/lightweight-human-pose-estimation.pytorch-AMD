# Fused postprocess v2: GPU TopM pair pruning + parallel CPU assembly

## Runtime files

```text
modules/migraphx_fused_postprocess_pruned_compiler.py
modules/migraphx_fused_postprocess_pruned.py
modules/mx_pair_assembly_pruned.py
modules/postprocessing.py
simulate_10_camera_stream.py
```

## Sanity check

```bash
python -m py_compile \
  modules/migraphx_fused_postprocess_pruned_compiler.py \
  modules/migraphx_fused_postprocess_pruned.py \
  modules/mx_pair_assembly_pruned.py \
  modules/postprocessing.py \
  simulate_10_camera_stream.py
```

## Compile v2 MXR

```bash
python modules/migraphx_fused_postprocess_pruned_compiler.py \
  --video cctv_1280x720_24fps_2.mp4 \
  --output-dir models/fused_postprocess_pruned_cache \
  --topk 20 \
  --limb-topm 20 \
  --threshold 0.1 \
  --nms-radius 6 \
  --nms-impl separable \
  --heatmap-cubic-a -0.75 \
  --paf-cubic-a -0.75 \
  --points-per-limb 8 \
  --min-paf-score 0.05 \
  --success-ratio-thr 0.8 \
  --min-pair-score 0.0 \
  --force
```

## Stream test with multiple CPU post workers

```bash
python simulate_10_camera_stream.py \
  --model pose_model1_fp16_ref1.mxr \
  --variant mx_fused_cubic_topk_fullres_paf_pruned \
  --num-cameras 10 \
  --duration-s 60 \
  --frames-per-camera 0 \
  --realtime \
  --camera-fps 24 \
  --buffer-mode latest \
  --backpressure-mode soft \
  --infer-workers 1 \
  --post-workers 4 \
  --shared-map-slots 16 \
  --shared-dtype float32 \
  --torch-device cuda \
  --require-gpu \
  --max-keypoints 20 \
  --threshold 0.1 \
  --nms-radius-fullres 6 \
  --nms-impl separable \
  --manual-cubic-a -0.75 \
  --fused-paf-cubic-a -0.75 \
  --fused-points-per-limb 8 \
  --fused-min-paf-score 0.05 \
  --fused-success-ratio-thr 0.8 \
  --limb-topm 20 \
  --min-pair-score 0.0 \
  --migraphx-fused-pruned-postprocess-cache-dir models/fused_postprocess_pruned_cache \
  --compile-fused-pruned-postprocess \
  --warmup-s 10 \
  --pin-cpus \
  --pin-camera-cores 2-11 \
  --pin-inference-cores 16 \
  --pin-post-cores 18-21 \
  --pin-main-cores 0,1,12-15,17,22-31 \
  --pin-all-threads \
  --monitor-system \
  --print-affinity \
  --summary-json outputs/stream_pruned_v2_10cam_p4_summary.json \
  --detailed-csv outputs/stream_pruned_v2_10cam_p4_detailed.csv
```

## Batching pose_model.mxr

Yes, batched pose_model.mxr is possible and likely useful for throughput, but it changes latency. Use micro-batches first:

```text
B=2 or B=4
collect latest frames from different cameras
run pose_model batch
run postprocess per item or batched v2 head
drop stale frames aggressively
```
