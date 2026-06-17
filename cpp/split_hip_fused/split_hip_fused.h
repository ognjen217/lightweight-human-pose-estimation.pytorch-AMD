#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum SplitHipFusedStatus {
    HIP_SPLIT_FUSED_SUCCESS = 0,
    HIP_SPLIT_FUSED_INVALID_ARGUMENT = 1,
    HIP_SPLIT_FUSED_HIP_ERROR = 2,
    HIP_SPLIT_FUSED_NOT_IMPLEMENTED = 3,
};

const char* split_hip_fused_status_string(int status);

// Host-mediated fused split postprocess:
//   heatmaps [B,18,H,W] + pafs [B,38,H,W]
//      -> smart heatmap TopK [B,18,K]
//      -> HIP2 PAF pair scoring/pruning [B,19,M]
//
// Compared to the previous Python-level composition, this keeps top_scores and
// top_indices on the GPU between HIP1 and HIP2.  Only the final small tensors are
// copied back to host for CPU pose assembly.
int split_hip_fused_run_host(
    const float* heatmaps_host,
    const float* pafs_host,
    float* top_scores_host,
    long long* top_indices_host,
    long long* limb_top_pair_a_idx_host,
    long long* limb_top_pair_b_idx_host,
    float* limb_top_pair_score_host,
    float* limb_top_pair_valid_host,
    int batch,
    int channels,
    int paf_channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    int limb_topm,
    float threshold,
    int lowres_nms_radius,
    int smart_proposals,
    int smart_local_radius,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a);

#ifdef __cplusplus
}
#endif
