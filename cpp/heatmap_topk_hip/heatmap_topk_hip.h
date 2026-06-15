#pragma once

#ifdef __cplusplus
extern "C" {
#endif

enum HeatmapTopKHipStatus {
    HIP_TOPK_SUCCESS = 0,
    HIP_TOPK_INVALID_ARGUMENT = 1,
    HIP_TOPK_HIP_ERROR = 2,
    HIP_TOPK_NOT_IMPLEMENTED = 3,
};

typedef struct HeatmapTopKHipProfile {
    float h2d_ms;
    float resize_ms;
    float vertical_ms;
    float horizontal_ms;
    float topk_ms;
    float d2h_scores_ms;
    float d2h_indices_ms;
    float device_total_ms;
    float total_ms;
} HeatmapTopKHipProfile;

const char* heatmap_topk_hip_status_string(int status);

int heatmap_topk_hip_run(
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

// Host-mediated correctness/test entrypoint.  This intentionally performs
// CPU<->GPU copies around heatmap_topk_hip_run so the native kernels can be
// validated before the true zero-copy handoff is implemented.
int heatmap_topk_hip_run_host(
    const float* heatmaps_host,
    float* top_scores_host,
    long long* top_indices_host,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius);

// Profiling entrypoint for the host-mediated baseline path.  It performs the
// same work as heatmap_topk_hip_run_host, but records HIP-event timings.
int heatmap_topk_hip_run_host_profile(
    const float* heatmaps_host,
    float* top_scores_host,
    long long* top_indices_host,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius,
    HeatmapTopKHipProfile* profile);

// Experimental E3 path: keep full-res resize and vertical max, but fuse the
// horizontal max check with TopK candidate selection.  This avoids materializing
// the pooled full-resolution buffer used by the baseline dense path.
int heatmap_topk_hip_run_host_fused(
    const float* heatmaps_host,
    float* top_scores_host,
    long long* top_indices_host,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius);

int heatmap_topk_hip_run_host_fused_profile(
    const float* heatmaps_host,
    float* top_scores_host,
    long long* top_indices_host,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius,
    HeatmapTopKHipProfile* profile);

#ifdef __cplusplus
}
#endif
