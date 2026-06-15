#include "heatmap_topk_hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr float kInvalidScore = -1.0e9f;
constexpr int kMaxTopK = 64;

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
           topk <= 0 || topk > kMaxTopK || nms_radius < 0;
}

__device__ __host__ inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __host__ inline float cubic_weight(float distance, float a) {
    const float x = fabsf(distance);
    const float x2 = x * x;
    const float x3 = x2 * x;
    if (x <= 1.0f) {
        return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
    }
    if (x < 2.0f) {
        return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

__global__ void resize_cubic_kernel(
    const float* __restrict__ heatmaps,
    float* __restrict__ resized,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w) {
    const std::size_t total = static_cast<std::size_t>(batch) * channels * full_h * full_w;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    constexpr float a = -0.75f;

    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        const int x_full = static_cast<int>(linear % full_w);
        const int y_full = static_cast<int>((linear / full_w) % full_h);
        const int bc = static_cast<int>(linear / (static_cast<std::size_t>(full_h) * full_w));

        const float src_x = (static_cast<float>(x_full) + 0.5f) * (static_cast<float>(in_w) / static_cast<float>(full_w)) - 0.5f;
        const float src_y = (static_cast<float>(y_full) + 0.5f) * (static_cast<float>(in_h) / static_cast<float>(full_h)) - 0.5f;
        const int base_x = static_cast<int>(floorf(src_x));
        const int base_y = static_cast<int>(floorf(src_y));

        float acc = 0.0f;
        const std::size_t in_plane = static_cast<std::size_t>(bc) * in_h * in_w;
        for (int oy = -1; oy <= 2; ++oy) {
            const int raw_y = base_y + oy;
            const int yy = clamp_int(raw_y, 0, in_h - 1);
            const float wy = cubic_weight(src_y - static_cast<float>(raw_y), a);
            for (int ox = -1; ox <= 2; ++ox) {
                const int raw_x = base_x + ox;
                const int xx = clamp_int(raw_x, 0, in_w - 1);
                const float wx = cubic_weight(src_x - static_cast<float>(raw_x), a);
                acc += heatmaps[in_plane + static_cast<std::size_t>(yy) * in_w + xx] * wy * wx;
            }
        }
        resized[linear] = acc;
    }
}

__global__ void vertical_max_kernel(
    const float* __restrict__ resized,
    float* __restrict__ vertical,
    int batch,
    int channels,
    int full_h,
    int full_w,
    int radius) {
    const std::size_t total = static_cast<std::size_t>(batch) * channels * full_h * full_w;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        const int x = static_cast<int>(linear % full_w);
        const int y = static_cast<int>((linear / full_w) % full_h);
        const std::size_t plane = (linear / (static_cast<std::size_t>(full_h) * full_w)) * full_h * full_w;
        const int y0 = clamp_int(y - radius, 0, full_h - 1);
        const int y1 = clamp_int(y + radius, 0, full_h - 1);
        float m = -3.402823466e38F;
        for (int yy = y0; yy <= y1; ++yy) {
            const float v = resized[plane + static_cast<std::size_t>(yy) * full_w + x];
            m = v > m ? v : m;
        }
        vertical[linear] = m;
    }
}

__global__ void horizontal_max_kernel(
    const float* __restrict__ vertical,
    float* __restrict__ pooled,
    int batch,
    int channels,
    int full_h,
    int full_w,
    int radius) {
    const std::size_t total = static_cast<std::size_t>(batch) * channels * full_h * full_w;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        const int x = static_cast<int>(linear % full_w);
        const int y = static_cast<int>((linear / full_w) % full_h);
        const std::size_t plane = (linear / (static_cast<std::size_t>(full_h) * full_w)) * full_h * full_w;
        const int x0 = clamp_int(x - radius, 0, full_w - 1);
        const int x1 = clamp_int(x + radius, 0, full_w - 1);
        float m = -3.402823466e38F;
        for (int xx = x0; xx <= x1; ++xx) {
            const float v = vertical[plane + static_cast<std::size_t>(y) * full_w + xx];
            m = v > m ? v : m;
        }
        pooled[linear] = m;
    }
}

__device__ inline void insert_topk(float score, long long index, float* scores, long long* indices, int topk) {
    for (int k = 0; k < topk; ++k) {
        if (score > scores[k] || (score == scores[k] && index < indices[k])) {
            for (int j = topk - 1; j > k; --j) {
                scores[j] = scores[j - 1];
                indices[j] = indices[j - 1];
            }
            scores[k] = score;
            indices[k] = index;
            break;
        }
    }
}

__global__ void topk_from_pooled_kernel(
    const float* __restrict__ resized,
    const float* __restrict__ pooled,
    float* __restrict__ top_scores,
    long long* __restrict__ top_indices,
    int batch,
    int channels,
    int full_h,
    int full_w,
    int topk,
    float threshold) {
    const int bc = blockIdx.x;
    const int total_bc = batch * channels;
    if (bc >= total_bc) {
        return;
    }

    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    long long* sh_indices = reinterpret_cast<long long*>(sh_scores + blockDim.x * topk);

    const int tid = threadIdx.x;
    const int full_size = full_h * full_w;
    float local_scores[kMaxTopK];
    long long local_indices[kMaxTopK];
    for (int k = 0; k < topk; ++k) {
        local_scores[k] = kInvalidScore;
        local_indices[k] = 9223372036854775807LL;
    }

    const std::size_t plane = static_cast<std::size_t>(bc) * full_h * full_w;
    for (int idx = tid; idx < full_size; idx += blockDim.x) {
        const float hm = resized[plane + idx];
        const float pm = pooled[plane + idx];
        if ((hm == pm) && (hm > threshold)) {
            insert_topk(hm, static_cast<long long>(idx), local_scores, local_indices, topk);
        }
    }

    const int sh_base = tid * topk;
    for (int k = 0; k < topk; ++k) {
        sh_scores[sh_base + k] = local_scores[k];
        sh_indices[sh_base + k] = local_indices[k];
    }
    __syncthreads();

    if (tid == 0) {
        float best_scores[kMaxTopK];
        long long best_indices[kMaxTopK];
        for (int k = 0; k < topk; ++k) {
            best_scores[k] = kInvalidScore;
            best_indices[k] = 9223372036854775807LL;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            const int base = t * topk;
            for (int kk = 0; kk < topk; ++kk) {
                insert_topk(sh_scores[base + kk], sh_indices[base + kk], best_scores, best_indices, topk);
            }
        }

        const std::size_t out_base = static_cast<std::size_t>(bc) * topk;
        for (int k = 0; k < topk; ++k) {
            top_scores[out_base + k] = best_scores[k];
            top_indices[out_base + k] = (best_indices[k] == 9223372036854775807LL) ? 0LL : best_indices[k];
        }
    }
}

int check_last_error() {
    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? HIP_TOPK_SUCCESS : HIP_TOPK_HIP_ERROR;
}

void free_if_needed(float* ptr) {
    if (ptr) {
        (void)hipFree(ptr);
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

    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
    const std::size_t dense_count = static_cast<std::size_t>(batch) * channels * full_h * full_w;
    const int threads = 256;
    const int blocks = static_cast<int>((dense_count + threads - 1) / threads);
    const int total_bc = batch * channels;
    const std::size_t dense_bytes = dense_count * sizeof(float);

    float* resized = nullptr;
    float* vertical = nullptr;
    float* pooled = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&resized), dense_bytes);
    if (err != hipSuccess) return HIP_TOPK_HIP_ERROR;
    err = hipMalloc(reinterpret_cast<void**>(&vertical), dense_bytes);
    if (err != hipSuccess) {
        free_if_needed(resized);
        return HIP_TOPK_HIP_ERROR;
    }
    err = hipMalloc(reinterpret_cast<void**>(&pooled), dense_bytes);
    if (err != hipSuccess) {
        free_if_needed(resized);
        free_if_needed(vertical);
        return HIP_TOPK_HIP_ERROR;
    }

    hipLaunchKernelGGL(
        resize_cubic_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        heatmaps_dev,
        resized,
        batch,
        channels,
        in_h,
        in_w,
        full_h,
        full_w);
    int status = check_last_error();
    if (status != HIP_TOPK_SUCCESS) goto cleanup;

    hipLaunchKernelGGL(
        vertical_max_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        resized,
        vertical,
        batch,
        channels,
        full_h,
        full_w,
        nms_radius);
    status = check_last_error();
    if (status != HIP_TOPK_SUCCESS) goto cleanup;

    hipLaunchKernelGGL(
        horizontal_max_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        stream,
        vertical,
        pooled,
        batch,
        channels,
        full_h,
        full_w,
        nms_radius);
    status = check_last_error();
    if (status != HIP_TOPK_SUCCESS) goto cleanup;

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * static_cast<std::size_t>(topk) *
                                         (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(
            topk_from_pooled_kernel,
            dim3(total_bc),
            dim3(threads),
            shared_bytes,
            stream,
            resized,
            pooled,
            top_scores_dev,
            top_indices_dev,
            batch,
            channels,
            full_h,
            full_w,
            topk,
            threshold);
        status = check_last_error();
        if (status != HIP_TOPK_SUCCESS) goto cleanup;
    }

    err = (stream == nullptr) ? hipDeviceSynchronize() : hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        status = HIP_TOPK_HIP_ERROR;
    }

cleanup:
    free_if_needed(resized);
    free_if_needed(vertical);
    free_if_needed(pooled);
    return status;
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
        free_if_needed(heatmaps_dev);
        return HIP_TOPK_HIP_ERROR;
    }
    err = hipMalloc(reinterpret_cast<void**>(&top_indices_dev), topk_count * sizeof(long long));
    if (err != hipSuccess) {
        free_if_needed(heatmaps_dev);
        free_if_needed(top_scores_dev);
        return HIP_TOPK_HIP_ERROR;
    }

    err = hipMemcpy(heatmaps_dev, heatmaps_host, heatmap_count * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        free_if_needed(heatmaps_dev);
        free_if_needed(top_scores_dev);
        free_if_needed(top_indices_dev);
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

    free_if_needed(heatmaps_dev);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    return status;
}
