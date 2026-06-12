#include "heatmap_topk_hip.h"

#include <hip/hip_runtime.h>

namespace {

bool invalid_shape(
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    int nms_radius) {
    return batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0 || full_h <= 0 || full_w <= 0 ||
           topk <= 0 || nms_radius < 0;
}

}  // namespace

const char* heatmap_topk_hip_status_string(int status) {
    switch (status) {
        case HIP_TOPK_SUCCESS:
            return "HIP_TOPK_SUCCESS";
        case HIP_TOPK_INVALID_ARGUMENT:
            return "HIP_TOPK_INVALID_ARGUMENT";
        case HIP_TOPK_HIP_ERROR:
            return "HIP_TOPK_HIP_ERROR";
        case HIP_TOPK_NOT_IMPLEMENTED:
            return "HIP_TOPK_NOT_IMPLEMENTED";
        default:
            return "HIP_TOPK_UNKNOWN_STATUS";
    }
}

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
    void* hip_stream) {
    (void)threshold;
    (void)hip_stream;

    if (!heatmaps_dev || !top_scores_dev || !top_indices_dev) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, channels, in_h, in_w, full_h, full_w, topk, nms_radius)) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }

    // Placeholder: the ABI and build scaffold are now in place.  The first real
    // kernel implementation should fill top_scores/top_indices with the same
    // contract as the existing manual cubic heatmap branch.
    return HIP_TOPK_NOT_IMPLEMENTED;
}
