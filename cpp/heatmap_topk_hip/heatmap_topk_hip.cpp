#include "heatmap_topk_hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

constexpr float kInvalidScore = -1.0e9f;

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

int hip_status(hipError_t err) {
    return err == hipSuccess ? HIP_TOPK_SUCCESS : HIP_TOPK_HIP_ERROR;
}

__host__ __device__ inline float cubic_weight(float distance, float a) {
    float x = fabsf(distance);
    float x2 = x * x;
    float x3 = x2 * x;
    if (x <= 1.0f) {
        return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
    }
    if (x < 2.0f) {
        return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

__device__ inline int clamp_int(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

__device__ float sample_cubic_heatmap(
    const float* heatmaps,
    int bc,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int y_full,
    int x_full) {
    constexpr float a = -0.75f;
    float src_x = (static_cast<float>(x_full) + 0.5f) * (static_cast<float>(in_w) / static_cast<float>(full_w)) - 0.5f;
    float src_y = (static_cast<float>(y_full) + 0.5f) * (static_cast<float>(in_h) / static_cast<float>(full_h)) - 0.5f;
    int base_x = static_cast<int>(floorf(src_x));
    int base_y = static_cast<int>(floorf(src_y));

    float acc = 0.0f;
    const long long plane_offset = static_cast<long long>(bc) * in_h * in_w;
    for (int oy = -1; oy <= 2; ++oy) {
        int raw_y = base_y + oy;
        int yy = clamp_int(raw_y, 0, in_h - 1);
        float wy = cubic_weight(src_y - static_cast<float>(raw_y), a);
        for (int ox = -1; ox <= 2; ++ox) {
            int raw_x = base_x + ox;
            int xx = clamp_int(raw_x, 0, in_w - 1);
            float wx = cubic_weight(src_x - static_cast<float>(raw_x), a);
            acc += heatmaps[plane_offset + static_cast<long long>(yy) * in_w + xx] * wy * wx;
        }
    }
    return acc;
}

__global__ void heatmap_topk_dense_kernel(
    const float* __restrict__ heatmaps,
    float* __restrict__ top_scores,
    long long* __restrict__ top_indices,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int nms_radius) {
    int bc = blockIdx.x;
    int total_bc = batch * channels;
    if (bc >= total_bc) {
        return;
    }

    // One block owns one (batch, channel) plane.  This first correctness kernel
    // is intentionally simple: each thread scans a strided subset of full-res
    // pixels, performs separable-equivalent square max NMS by direct local scan,
    // keeps a local TopK, then the block reduces through shared memory.
    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    long long* sh_indices = reinterpret_cast<long long*>(sh_scores + blockDim.x * topk);

    const int tid = threadIdx.x;
    const int full_size = full_h * full_w;

    // Limit used by the current graph contract.  If topk changes beyond this,
    // fail gracefully by writing invalid outputs from thread 0.
    if (topk > 64) {
        if (tid == 0) {
            const long long out_base = static_cast<long long>(bc) * topk;
            for (int k = 0; k < topk; ++k) {
                top_scores[out_base + k] = kInvalidScore;
                top_indices[out_base + k] = 0;
            }
        }
        return;
    }

    float local_scores[64];
    long long local_indices[64];
    for (int k = 0; k < topk; ++k) {
        local_scores[k] = kInvalidScore;
        local_indices[k] = 0;
    }

    for (int idx = tid; idx < full_size; idx += blockDim.x) {
        int y = idx / full_w;
        int x = idx - y * full_w;
        float center = sample_cubic_heatmap(heatmaps, bc, in_h, in_w, full_h, full_w, y, x);
        bool peak = center > threshold;
        if (peak && nms_radius > 0) {
            int y0 = max(0, y - nms_radius);
            int y1 = min(full_h - 1, y + nms_radius);
            int x0 = max(0, x - nms_radius);
            int x1 = min(full_w - 1, x + nms_radius);
            for (int yy = y0; yy <= y1 && peak; ++yy) {
                for (int xx = x0; xx <= x1; ++xx) {
                    float v = sample_cubic_heatmap(heatmaps, bc, in_h, in_w, full_h, full_w, yy, xx);
                    if (v > center) {
                        peak = false;
                        break;
                    }
                }
            }
        }
        if (!peak) {
            continue;
        }

        // Insert into local sorted TopK.  Tie-breaking favors the lower flat
        // index, which is deterministic.  Invalid ties are semantically ignored
        // by the Python comparison tool.
        for (int k = 0; k < topk; ++k) {
            if (center > local_scores[k] || (center == local_scores[k] && idx < local_indices[k])) {
                for (int j = topk - 1; j > k; --j) {
                    local_scores[j] = local_scores[j - 1];
                    local_indices[j] = local_indices[j - 1];
                }
                local_scores[k] = center;
                local_indices[k] = static_cast<long long>(idx);
                break;
            }
        }
    }

    const int sh_base = tid * topk;
    for (int k = 0; k < topk; ++k) {
        sh_scores[sh_base + k] = local_scores[k];
        sh_indices[sh_base + k] = local_indices[k];
    }
    __syncthreads();

    if (tid == 0) {
        float best_scores[64];
        long long best_indices[64];
        for (int k = 0; k < topk; ++k) {
            best_scores[k] = kInvalidScore;
            best_indices[k] = 0;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            const int base = t * topk;
            for (int kk = 0; kk < topk; ++kk) {
                float score = sh_scores[base + kk];
                long long index = sh_indices[base + kk];
                for (int k = 0; k < topk; ++k) {
                    if (score > best_scores[k] || (score == best_scores[k] && index < best_indices[k])) {
                        for (int j = topk - 1; j > k; --j) {
                            best_scores[j] = best_scores[j - 1];
                            best_indices[j] = best_indices[j - 1];
                        }
                        best_scores[k] = score;
                        best_indices[k] = index;
                        break;
                    }
                }
            }
        }

        const long long out_base = static_cast<long long>(bc) * topk;
        for (int k = 0; k < topk; ++k) {
            top_scores[out_base + k] = best_scores[k];
            top_indices[out_base + k] = best_indices[k];
        }
    }
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
    if (!heatmaps_dev || !top_scores_dev || !top_indices_dev) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, channels, in_h, in_w, full_h, full_w, topk, nms_radius)) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }
    if (topk > 64) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
    const int total_bc = batch * channels;
    const int threads = 256;
    const std::size_t shared_bytes = static_cast<std::size_t>(threads) * static_cast<std::size_t>(topk) *
                                     (sizeof(float) + sizeof(long long));

    hipLaunchKernelGGL(
        heatmap_topk_dense_kernel,
        dim3(total_bc),
        dim3(threads),
        shared_bytes,
        stream,
        heatmaps_dev,
        top_scores_dev,
        top_indices_dev,
        batch,
        channels,
        in_h,
        in_w,
        full_h,
        full_w,
        topk,
        threshold,
        nms_radius);

    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        return HIP_TOPK_HIP_ERROR;
    }
    if (stream == nullptr) {
        hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            return HIP_TOPK_HIP_ERROR;
        }
    }
    return HIP_TOPK_SUCCESS;
}

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
    int nms_radius) {
    if (!heatmaps_host || !top_scores_host || !top_indices_host) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, channels, in_h, in_w, full_h, full_w, topk, nms_radius)) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }

    const std::size_t heatmap_count = static_cast<std::size_t>(batch) * channels * in_h * in_w;
    const std::size_t topk_count = static_cast<std::size_t>(batch) * channels * topk;

    float* heatmaps_dev = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&heatmaps_dev), heatmap_count * sizeof(float));
    if (err != hipSuccess) return HIP_TOPK_HIP_ERROR;
    err = hipMalloc(reinterpret_cast<void**>(&top_scores_dev), topk_count * sizeof(float));
    if (err != hipSuccess) {
        hipFree(heatmaps_dev);
        return HIP_TOPK_HIP_ERROR;
    }
    err = hipMalloc(reinterpret_cast<void**>(&top_indices_dev), topk_count * sizeof(long long));
    if (err != hipSuccess) {
        hipFree(heatmaps_dev);
        hipFree(top_scores_dev);
        return HIP_TOPK_HIP_ERROR;
    }

    err = hipMemcpy(heatmaps_dev, heatmaps_host, heatmap_count * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        hipFree(heatmaps_dev);
        hipFree(top_scores_dev);
        hipFree(top_indices_dev);
        return HIP_TOPK_HIP_ERROR;
    }

    int status = heatmap_topk_hip_run(
        heatmaps_dev,
        top_scores_dev,
        top_indices_dev,
        batch,
        channels,
        in_h,
        in_w,
        full_h,
        full_w,
        topk,
        threshold,
        nms_radius,
        nullptr);

    if (status == HIP_TOPK_SUCCESS) {
        err = hipMemcpy(top_scores_host, top_scores_dev, topk_count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) status = HIP_TOPK_HIP_ERROR;
        err = hipMemcpy(top_indices_host, top_indices_dev, topk_count * sizeof(long long), hipMemcpyDeviceToHost);
        if (err != hipSuccess) status = HIP_TOPK_HIP_ERROR;
    }

    hipFree(heatmaps_dev);
    hipFree(top_scores_dev);
    hipFree(top_indices_dev);
    return status;
}
