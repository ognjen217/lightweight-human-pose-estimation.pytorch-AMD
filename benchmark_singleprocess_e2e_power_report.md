# Single-process E2E postprocess benchmark

This benchmark measures speed only, not AP/AR accuracy.

All variants are executed as separate single-process video passes: frame read, preprocess, MIGraphX inference, and one selected postprocess path. Drawing and video writing are intentionally excluded.

GPU power is sampled with `rocm-smi`; energy and FPS/W are estimates based on GPU package power, not whole-system power.

## Command context

- Video: `migraphx_postprocess_deliverable/cctv_1280x720_24fps_original.mp4`
- Model: `migraphx_postprocess_deliverable/pose_model1_fp16_ref1.mxr`
- Frames: `100` measured, `5` warmup
- MIGraphX NMS MXR: `migraphx_postprocess_deliverable/models/heatmap_nms_head.mxr`

## Variant definitions

- `standard_cpu`: Original full-res CPU: extract_keypoints per channel + group_keypoints.
- `optimized_batch_k20_findnonzero_v1_cpu`: Best CPU-only: full-res batch cv2/findNonZero K20 extraction + group_keypoints_fast.
- `gpu-fullres-nms-cpu-group`: Full-res torch GPU NMS/top-K extraction + CPU group_keypoints_fast.
- `full-gpu`: Experimental: torch GPU NMS/top-K + torch GPU PAF affinity scoring; final dynamic pose assembly on CPU.
- `migraphx-nms`: Full-res MIGraphX NMS dense peak mask + CPU extract_from_mask + CPU group_keypoints_fast.

## Summary

| variant | pre ms | infer ms | post ms | e2e ms | p95 ms | FPS | Power W | J/frame | FPS/W | post speedup | e2e speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standard_cpu | 3.72 | 7.72 | 227.76 | 240.06 | 246.57 | 4.17 | 56.31 | 13.5180 | 0.07 | 1.00x | 1.00x |
| optimized_batch_k20_findnonzero_v1_cpu | 3.57 | 8.53 | 60.05 | 72.60 | 75.38 | 13.77 | 63.85 | 4.6358 | 0.22 | 3.79x | 3.31x |
| gpu-fullres-nms-cpu-group | 4.00 | 5.97 | 38.95 | 49.33 | 51.79 | 20.27 | 81.68 | 4.0291 | 0.25 | 5.85x | 4.87x |
| full-gpu | 3.58 | 5.91 | 46.51 | 57.64 | 60.05 | 17.35 | 82.06 | 4.7303 | 0.21 | 4.90x | 4.16x |
| migraphx-nms | 3.72 | 6.89 | 110.83 | 125.11 | 127.47 | 7.99 | 61.14 | 7.6493 | 0.13 | 2.05x | 1.92x |

## Postprocess breakdown

| variant | decode | hm resize | paf resize | mx nms | extract | mask extract | group | post total |
|---|---|---|---|---|---|---|---|---|
| standard_cpu | 1.08 | 2.62 | 4.26 | 0.00 | 119.60 | 0.00 | 100.17 | 227.76 |
| optimized_batch_k20_findnonzero_v1_cpu | 1.09 | 2.28 | 3.84 | 0.00 | 47.29 | 0.00 | 5.53 | 60.05 |
| gpu-fullres-nms-cpu-group | 1.10 | 2.28 | 3.85 | 0.00 | 26.20 | 0.00 | 5.48 | 38.95 |
| full-gpu | 1.09 | 2.30 | 4.12 | 0.00 | 26.02 | 0.00 | 12.95 | 46.51 |
| migraphx-nms | 1.09 | 2.38 | 4.10 | 5.66 | 59.79 | 59.79 | 5.80 | 110.83 |

## Notes

- `standard_cpu` uses the original CPU keypoint extraction and original CPU grouping.
- `optimized_batch_k20_findnonzero_v1_cpu` is the best CPU-only path: batched K20 extraction plus `group_keypoints_fast`.
- `gpu-fullres-nms-cpu-group` only moves heatmap NMS/top-K extraction to torch GPU; grouping stays CPU fast group.
- `full-gpu` is GPU-heavy but not literally 100% GPU: NMS and PAF affinity scoring run on torch GPU, while final dynamic greedy pose assembly remains CPU.
- `migraphx-nms` uses one fixed full-resolution NMS MXR, appropriate for fixed-size 1280x720 video.
