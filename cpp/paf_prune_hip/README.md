# HIP PAF pruning backend skeleton

This directory contains the native HIP2 backend for replacing split MXR2 in the
split pipeline.

Target contract:

```text
input:
  pafs        [B,38,68,121] fp32
  top_scores  [B,18,K] fp32
  top_indices [B,18,K] int64, flattened full-resolution indices

output:
  limb_top_pair_a_idx [B,19,M] int64
  limb_top_pair_b_idx [B,19,M] int64
  limb_top_pair_score [B,19,M] fp32
  limb_top_pair_valid [B,19,M] fp32
```

The intended runtime pipeline is:

```text
MXR1: input image -> heatmaps_dev + pafs_dev
HIP heatmap backend: heatmaps_dev -> top_scores_dev + top_indices_dev
HIP PAF backend: pafs_dev + top_scores_dev + top_indices_dev -> pruned limb pairs
CPU: final small-tensor pose assembly
```

## Current implementation

The first implementation target is a correctness baseline equivalent to split
MXR2:

```text
K20 x K20 candidate pairs per limb
8 line samples per pair
4x4 cubic PAF sampling per sample point
validity checks using min_paf_score and success_ratio_thr
TopM pruning per limb
```

This baseline intentionally scores the full candidate grid before any N64/N96
pre-pruning is introduced.  It is designed to answer one question first:

```text
Can a custom HIP backend reproduce MXR2's PAF scoring/pruning output contract?
```

## Build

```bash
bash tools/build_paf_prune_hip.sh
```

The build output is:

```text
build/paf_prune_hip/libpaf_prune_hip.so
```

You can override the target architecture with:

```bash
HIP_PAF_PRUNE_OFFLOAD_ARCH=gfx1150 bash tools/build_paf_prune_hip.sh
```

## Correctness comparison

After building the library, compare HIP2 against MXR2 with:

```bash
python tools/compare_split_pipeline_hip2_vs_mxr2.py \
  --mxr1 models/split_pose_adapter/pose_adapter_b4_1080x1920.mxr \
  --mxr2 models/split_paf_pruning_from_topk/split_paf_pruning_from_topk_b4_68x121_to_1080x1920_k20_m20_p8_min0p05_sr0p8_pam0p75_mp0p0.mxr \
  --batch-size 4 \
  --heatmap-backend hip_host_smart \
  --paf-backend hip_host \
  --runs 3
```

## Next planned variants

Once this N400 correctness baseline is validated, add pre-pruned variants:

```text
HIP2_N96: score only 96 pre-ranked pairs per limb
HIP2_N64: score only 64 pre-ranked pairs per limb
HIP2_K16: K16/M20 control variant
```
