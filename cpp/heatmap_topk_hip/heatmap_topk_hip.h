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

// Experimental E4 smart-full-res path.  It avoids dense full-resolution heatmap
// NMS/TopK by selecting low-resolution proposals, locally refining them in
// full-resolution coordinates, and finally applying TopK over the refined list.
// The generic profile fields are reused as follows:
//   resize_ms     -> low-resolution proposal selection
//   vertical_ms   -> local full-resolution refinement
//   topk_ms       -> final TopK over refined proposals
int heatmap_topk_hip_run_host_smart(
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
    int lowres_nms_radius,
    int smart_proposals,
    int smart_local_radius);

int heatmap_topk_hip_run_host_smart_profile(
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
    int lowres_nms_radius,
    int smart_proposals,
    int smart_local_radius,
    HeatmapTopKHipProfile* profile);

#ifdef __cplusplus
}
#endif
