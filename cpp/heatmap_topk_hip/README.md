# HIP heatmap TopK backend skeleton

This directory is the native backend boundary for the split MXR experiment.

Target contract:

```text
input:
  heatmaps   [B,18,68,121] fp32, GPU-resident

output:
  top_scores [B,18,20] fp32, GPU-resident
  top_indices[B,18,20] int64, GPU-resident, flattened full-res indices in 1080x1920 space
```

The intended pipeline is:

```text
MXR1: input image -> heatmaps_dev + pafs_dev
HIP backend: heatmaps_dev -> top_scores_dev + top_indices_dev
MXR2: pafs_dev + top_scores_dev + top_indices_dev -> pruned limb pairs
CPU: final small tensors only
```

The first implementation target should be a correctness backend with the same semantics as the existing manual cubic heatmap branch:

```text
heatmaps [B,18,68,121]
-> manual cubic resize to [B,18,1080,1920]
-> separable NMS radius 6
-> threshold mask > 0.1
-> TopK K=20 per B/channel
```

After that, optimize away the full materialized resize with candidate generation / local refinement.

## Build direction

The native ABI should remain small and C-compatible so it can later be called from Python, a C++ MIGraphX wrapper, or an IPC worker:

```cpp
extern "C" int heatmap_topk_hip_run(
    const float* heatmaps_dev,
    float* top_scores_dev,
    long long* top_indices_dev,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius,
    void* hip_stream);
```

The `void* hip_stream` argument is intentionally opaque at the ABI boundary. Internally it is cast to `hipStream_t`.

## Current status

This directory currently provides the ABI placeholder and build scaffold. The implementation returns `HIP_TOPK_NOT_IMPLEMENTED` until the kernels are added.
