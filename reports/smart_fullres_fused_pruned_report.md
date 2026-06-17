# Smart-Full-Res Fused-Pruned Postprocessing Report

## Executive Summary

This report summarizes the implementation and validation of the **smart-full-res fused-pruned postprocessing** path for the lightweight human pose estimation pipeline on AMD ROCm/MIGraphX. The optimization targets the most expensive part of the merged postprocessing graph: full-resolution heatmap upsampling, full-resolution NMS, masking, and TopK extraction.

The key result is that smart-full-res preserves COCO accuracy on the evaluated subset while significantly improving high-resolution live-stream throughput. On the 1000-image COCO2017 dominant-resolution subset, smart-full-res achieved **AP=0.4324**, effectively matching the CPU references at **AP=0.4321**. On the pinned 10-camera 1080p live-stream simulation, the smart B4 merged model increased aggregate throughput from **28.16 FPS to 50.57 FPS** and reduced average end-to-end latency from **156.43 ms to 96.37 ms**.



## Motivation

The previous full-resolution fused-pruned graph preserved accuracy, but profiling showed that the heatmap branch was a major bottleneck. The original path performs dense full-resolution work for all heatmap pixels:

```text
low-res heatmaps
-> full-resolution bicubic resize
-> full-resolution NMS / mask
-> full-resolution TopK
-> PAF pair scoring
-> per-limb TopM pruning
-> CPU pose assembly
```

At 1080p, this creates very large intermediate tensors. For B4, the full-resolution heatmap tensor alone has:

```text
B * C * H * W = 4 * 18 * 1080 * 1920 = 149,299,200 elements
```

Earlier heatmap branch isolation showed that this part of the graph consumed a large fraction of total runtime.

| Component / Head | Shape | Avg Runtime |
|---|---:|---:|
| Full heatmap resize + NMS + mask + TopK branch | B4, 68x121 -> 1080x1920 | ~64 ms |
| Manual cubic resize-only ablation | B4, 68x121 -> 1080x1920 | ~46.6 ms |
| Share of final heatmap branch from resize-only ablation | B4 | ~72% |

This motivated a strategy that keeps full-resolution output coordinates, but avoids applying full-resolution processing everywhere.



## Smart-Full-Res Method

Smart-full-res changes only the heatmap candidate generation strategy. The rest of the fused-pruned pipeline remains compatible with the existing output contract.

The original full-res path asks:

```text
Where are the best keypoint peaks in the entire full-resolution heatmap?
```

Smart-full-res instead asks:

```text
Which low-resolution regions are likely to contain good peaks, and where exactly are those peaks after local full-resolution refinement?
```

The implemented `sp64_lr8_lnms1` configuration works as follows:

| Stage | Description |
|---|---|
| Low-resolution proposal stage | Run proposal selection on low-resolution heatmaps instead of immediately resizing the entire heatmap to full resolution. |
| Proposal count | Keep up to `smart_proposals=64` candidate regions per keypoint type before final TopK pruning. |
| Local full-resolution refinement | Around each low-resolution proposal, evaluate a small full-resolution local search window with `smart_local_radius=8`. |
| Low-res NMS | Use `smart_lowres_nms_radius=1` to suppress nearby duplicate proposals before local refinement. |
| Final output contract | Return full-resolution `top_scores` and `top_indices`, preserving downstream compatibility. |
| PAF / limb scoring | Keep fused PAF scoring and per-limb TopM pruning unchanged. |
| CPU tail | Keep the existing reduced CPU pose assembly from pruned pair outputs. |

This means smart-full-res is not a strict graph-equivalent rewrite. It is an approximation that preserves the same interface and final coordinate space, but avoids dense full-resolution heatmap processing across the whole image.

### Output Contract

The smart head returns the same six tensors as the full-res fused-pruned head:

| Output | Shape | Purpose |
|---|---:|---|
| `top_scores` | `[B, 18, K]` | Top keypoint scores per keypoint type. |
| `top_indices` | `[B, 18, K]` | Full-resolution flattened keypoint indices. |
| `limb_top_pair_a_idx` | `[B, 19, M]` | TopM endpoint-A indices per limb. |
| `limb_top_pair_b_idx` | `[B, 19, M]` | TopM endpoint-B indices per limb. |
| `limb_top_pair_score` | `[B, 19, M]` | TopM limb affinity scores. |
| `limb_top_pair_valid` | `[B, 19, M]` | Validity mask for pruned limb pairs. |

For the tested configuration:

```text
B = 4
K = 20
M = 20
smart_proposals = 64
smart_local_radius = 8
smart_lowres_nms_radius = 1
```



## Implementation Summary

Support was added so that smart-full-res can be used both in fixed-resolution stream models and in COCO accuracy validation.

| Area | Implementation |
|---|---|
| Smart head export | Added `--heatmap-mode smart-full-res` to the batch-aware fused-pruned postprocess exporter. |
| Smart parameters | Added `--smart-proposals`, `--smart-local-radius`, and `--smart-lowres-nms-radius`. |
| Merged stream model | Compiled smart-full-res B4 merged pose+postprocess MXR for 1080p stream simulation. |
| COCO validation | Added `accuracy_validation_smart.py` wrapper to pass smart-specific parameters into the existing validation flow. |
| Shape-aware autocompile | Extended postprocess autocompile so COCO validation compiles only the required shape-specific smart MXR heads. |
| Cache isolation | Smart heads are stored in smart-specific cache directories to avoid collision with full-res fused-pruned heads. |


## Standalone and Merged Runtime Validation

The smart postprocess head was first tested as a postprocess-only MXR and then as a merged pose+postprocess B4 model.

| Test | Baseline | Smart-Full-Res | Change |
|---|---:|---:|---:|
| Postprocess-only MXR runtime | 67.83 ms | 41.75 ms | **1.62x faster** |
| Merged B4 MXR runtime | 89.37 ms | 43.76 ms | **2.04x faster** |

The strict output comparison returned `passed=false`, which is expected. Smart-full-res intentionally changes candidate generation and is therefore not bitwise or index-equivalent to full-res. For this optimization, COCO AP and stream behavior are the decisive validation criteria.



## 10-Camera 1080p Stream Results

Both runs used the same pinned process configuration:

| Parameter | Value |
|---|---:|
| Cameras | 10 |
| Input video FPS target | 24 FPS per camera |
| Runtime | 130 s |
| Batch size | B4 |
| Batch timeout | 4 ms |
| Inference workers | 1 |
| Post workers | 4 |
| Buffer mode | latest |
| Backpressure | soft |
| Shared input slots | 10 |
| CPU pinning | enabled |

### Throughput and Latency

| Metric | Full-Res B4 | Smart-Full-Res B4 | Change |
|---|---:|---:|---:|
| Processed frames | 3,718 | 6,666 | **+79.3%** |
| Aggregate output FPS | 28.16 | 50.57 | **1.80x** |
| Avg FPS / camera | 2.82 | 5.06 | **1.80x** |
| Avg inference / frame | 25.25 ms | 14.55 ms | **-42.4%** |
| Approx. inference / B4 batch | 100.88 ms | 58.21 ms | **-42.3%** |
| Avg postprocess | 2.71 ms | 1.18 ms | **-56.4%** |
| Avg E2E latency | 156.43 ms | 96.37 ms | **-38.4%** |
| P95 E2E latency | 179.45 ms | 116.70 ms | **-35.0%** |

### Batch Collector Health

| Metric | Full-Res B4 | Smart-Full-Res B4 |
|---|---:|---:|
| Batch runs | 1,011 | 1,816 |
| Avg real batch size | 3.998 | 3.999 |
| Replaced before post | 0 | 0 |
| Stale records discarded pre-batch | 0 | 0 |
| Throttle skips | 0 | 0 |

The batch collector was healthy in both runs, so the improvement is attributable to the smart merged MXR being faster, not to different batching behavior.

---

## COCO2017 Accuracy Validation

The COCO validation used the `accuracy_validation_smart.py` wrapper and the existing `accuracy_validation.py` evaluation flow. The test requested 1000 images with `--image-selection dominant-dimensions`, which grouped COCO images by resolution and selected the most common shape first.

Observed selection:

| Selection Mode | Requested | Selected | Selected Shape | Available in Shape Group |
|---|---:|---:|---:|---:|
| dominant-dimensions | 1000 | 1000 | 480x640 | 1061 |

Because the dominant `480x640` group already contained more than 1000 images, only one smart postprocess head was required:

```text
low-res: 68x91
full-res: 480x640
```

### Accuracy Results

| Variant | AP | AP50 | AP75 | AR |
|---|---:|---:|---:|---:|
| Standard CPU reference | 0.4321 | 0.6953 | 0.4470 | 0.4976 |
| Optimized K20 CPU reference | 0.4321 | 0.6953 | 0.4470 | 0.4976 |
| Smart-full-res fused-pruned `sp64_lr8` | **0.4324** | **0.6981** | 0.4437 | 0.4920 |

### Accuracy Delta vs Optimized K20 CPU

| Metric | Delta |
|---|---:|
| AP | **+0.0003** |
| AP50 | **+0.0028** |
| AP75 | -0.0033 |
| AR | -0.0056 |

The result shows that smart-full-res preserved AP on this subset. The small AP75 and AR reductions suggest that the candidate approximation can miss or shift some stricter/localized matches, but the overall AP remained effectively unchanged.

### COCO Runtime Results

| Variant | Postprocess | E2E | FPS |
|---|---:|---:|---:|
| Standard CPU reference | 41.87 ms | 54.06 ms | 18.50 |
| Optimized K20 CPU reference | **13.01 ms** | **25.20 ms** | **39.67** |
| Smart-full-res fused-pruned `sp64_lr8` | 25.99 ms | 37.94 ms | 26.36 |

At `480x640`, CPU K20 remains faster than the smart MIGraphX postprocess head. This is expected because the dense full-resolution branch is much smaller at COCO's dominant resolution than at 1080p. The smart approach is primarily valuable for high-resolution live streams, where full-resolution heatmap processing becomes much more expensive.


## Interpretation

The smart-full-res approach achieved the main goal: it reduced high-resolution stream runtime while preserving COCO accuracy on the evaluated subset.

The most important observation is the difference between the COCO and live-stream results. On COCO `480x640`, the CPU K20 path remains faster because the image is relatively small and CPU full-resolution postprocessing is not yet the main bottleneck. On 1080p live streams, the dense full-resolution heatmap branch becomes large enough that avoiding global full-res processing produces a major runtime benefit.

| Environment | Resolution | Main Finding |
|---|---:|---|
| COCO accuracy subset | 480x640 | Smart preserves AP but is slower than CPU K20. |
| 10-camera stream | 1080p | Smart substantially improves throughput and latency. |

This means smart-full-res should not be evaluated only by low-resolution COCO runtime. Its target use case is fixed or semi-fixed high-resolution live video, where the full-resolution heatmap branch dominates the graph.


## Limitations

| Limitation | Impact |
|---|---|
| Not strict-equivalent | Random-output compare fails by design because smart proposal selection changes candidate generation. |
| Shape-specific compilation | COCO or multi-resolution inputs require one MXR head per selected output resolution. |
| Approximation risk | Very crowded scenes or small peaks could be missed if the low-resolution proposal stage does not include them. |
| COCO subset coverage | The reported AP result covers 1000 images from the dominant `480x640` group, not the full 5000-image val2017 set. |
| CPU K20 still faster at 480x640 | Smart is not intended to replace CPU K20 for small images; it targets high-resolution MIGraphX graph scalability. |

---

## Conclusion

Smart-full-res fused-pruned postprocessing is a valid performance candidate for high-resolution live pose estimation. It keeps the same downstream output contract as the full-res fused-pruned path, but avoids dense full-resolution heatmap processing by combining low-resolution proposal selection with local full-resolution bicubic refinement.

The method achieved **AP=0.4324** on the 1000-image COCO2017 dominant-resolution subset, effectively matching the CPU references at **AP=0.4321**. In the pinned 10-camera 1080p stream simulation, it increased aggregate throughput from **28.16 FPS to 50.57 FPS** and reduced average end-to-end latency from **156.43 ms to 96.37 ms**.

The recommended use of smart-full-res is therefore high-resolution live-stream inference, especially fixed-resolution deployments where the required smart postprocess MXR heads can be compiled ahead of time and reused.

---

## Reproducibility Notes

Smart COCO validation command used the following structure:

```bash
python accuracy_validation_smart.py \
  --models pose_model1_fp16_ref1.mxr \
  --variants merged_fused_pruned \
  --labels coco/annotations/person_keypoints_val2017.json \
  --images-folder coco/val2017/ \
  --num-of-test-img 1000 \
  --image-selection dominant-dimensions \
  --threshold 0.1 \
  --nms-radius-fullres 6 \
  --nms-impl separable \
  --max-keypoints 20 \
  --limb-topm 20 \
  --min-pair-score 0.0 \
  --manual-cubic-a -0.75 \
  --points-per-limb 8 \
  --min-paf-score 0.05 \
  --success-ratio-thr 0.8 \
  --paf-cubic-a -0.75 \
  --compile-missing-postprocess-heads \
  --keep-postprocess-onnx \
  --fused-pruned-heatmap-mode smart-full-res \
  --output-dir outputs/accuracy_coco_smart_sp64_lr8_1000
```

Generated COCO outputs:

```text
outputs/accuracy_coco_smart_sp64_lr8_1000/accuracy_validation_summary.json
outputs/accuracy_coco_smart_sp64_lr8_1000/accuracy_validation_summary.csv
outputs/accuracy_coco_smart_sp64_lr8_1000/selected_images_manifest.json
```
