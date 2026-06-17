#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum PafPruneHipStatus {
    HIP_PAF_PRUNE_SUCCESS = 0,
    HIP_PAF_PRUNE_INVALID_ARGUMENT = 1,
    HIP_PAF_PRUNE_HIP_ERROR = 2,
    HIP_PAF_PRUNE_NOT_IMPLEMENTED = 3,
};

typedef struct PafPruneHipProfile {
    float h2d_ms;
    float score_ms;
    float prune_ms;
    float d2h_ms;
    float device_total_ms;
    float total_ms;
} PafPruneHipProfile;

const char* paf_prune_hip_status_string(int status);

int paf_prune_hip_run(
    const float* pafs_dev,
    const float* top_scores_dev,
    const long long* top_indices_dev,
    long long* limb_top_pair_a_idx_dev,
    long long* limb_top_pair_b_idx_dev,
    float* limb_top_pair_score_dev,
    float* limb_top_pair_valid_dev,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a,
    void* hip_stream);

int paf_prune_hip_run_host(
    const float* pafs_host,
    const float* top_scores_host,
    const long long* top_indices_host,
    long long* limb_top_pair_a_idx_host,
    long long* limb_top_pair_b_idx_host,
    float* limb_top_pair_score_host,
    float* limb_top_pair_valid_host,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a);

int paf_prune_hip_run_host_profile(
    const float* pafs_host,
    const float* top_scores_host,
    const long long* top_indices_host,
    long long* limb_top_pair_a_idx_host,
    long long* limb_top_pair_b_idx_host,
    float* limb_top_pair_score_host,
    float* limb_top_pair_valid_host,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a,
    PafPruneHipProfile* profile);

#ifdef __cplusplus
}
#endif
