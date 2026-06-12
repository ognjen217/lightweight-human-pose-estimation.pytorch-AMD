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

#ifdef __cplusplus
}
#endif
