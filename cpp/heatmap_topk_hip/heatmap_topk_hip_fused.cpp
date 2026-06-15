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

__global__ void resize_cubic_fused_kernel(
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

__global__ void vertical_max_fused_kernel(
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

// Fuses the baseline horizontal max pass and TopK pass.  For each full-res pixel
// this recomputes the horizontal max directly from the vertical max buffer and
// immediately feeds valid local maxima into the per-plane TopK reducer.  It is
// intended as an E3 experiment: same output contract, one less dense output
// buffer, and no full pooled-buffer materialization.
__global__ void fused_horizontal_topk_kernel(
    const float* __restrict__ resized,
    const float* __restrict__ vertical,
    float* __restrict__ top_scores,
    long long* __restrict__ top_indices,
    int batch,
    int channels,
    int full_h,
    int full_w,
    int topk,
    float threshold,
    int radius) {
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
        const int x = idx % full_w;
        const int y = idx / full_w;
        const int x0 = clamp_int(x - radius, 0, full_w - 1);
        const int x1 = clamp_int(x + radius, 0, full_w - 1);
        float hmax = -3.402823466e38F;
        const std::size_t row = plane + static_cast<std::size_t>(y) * full_w;
        for (int xx = x0; xx <= x1; ++xx) {
            const float v = vertical[row + xx];
            hmax = v > hmax ? v : hmax;
        }
        const float hm = resized[plane + idx];
        if ((hm == hmax) && (hm > threshold)) {
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

void clear_profile(HeatmapTopKHipProfile* profile) {
    if (profile) {
        profile->h2d_ms = 0.0f;
        profile->resize_ms = 0.0f;
        profile->vertical_ms = 0.0f;
        profile->horizontal_ms = 0.0f;
        profile->topk_ms = 0.0f;
        profile->d2h_scores_ms = 0.0f;
        profile->d2h_indices_ms = 0.0f;
        profile->device_total_ms = 0.0f;
        profile->total_ms = 0.0f;
    }
}

void free_if_needed(float* ptr) {
    if (ptr) (void)hipFree(ptr);
}

void free_if_needed(long long* ptr) {
    if (ptr) (void)hipFree(ptr);
}

float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    return ms;
}

int run_fused_impl(
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
    HeatmapTopKHipProfile* profile) {
    clear_profile(profile);
    if (!heatmaps_host || !top_scores_host || !top_indices_host) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, channels, in_h, in_w, full_h, full_w, topk, nms_radius)) {
        return HIP_TOPK_INVALID_ARGUMENT;
    }

    const std::size_t heatmap_count = static_cast<std::size_t>(batch) * channels * in_h * in_w;
    const std::size_t dense_count = static_cast<std::size_t>(batch) * channels * full_h * full_w;
    const std::size_t topk_count = static_cast<std::size_t>(batch) * channels * topk;
    const std::size_t dense_bytes = dense_count * sizeof(float);
    const int threads = 256;
    const int blocks = static_cast<int>((dense_count + threads - 1) / threads);
    const int total_bc = batch * channels;

    float* heatmaps_dev = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;
    float* resized = nullptr;
    float* vertical = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t events[7] = {};
    int status = HIP_TOPK_SUCCESS;

    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) return HIP_TOPK_HIP_ERROR;
    for (int i = 0; i < 7; ++i) {
        err = hipEventCreate(&events[i]);
        if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    }

    err = hipMalloc(reinterpret_cast<void**>(&heatmaps_dev), heatmap_count * sizeof(float));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&top_scores_dev), topk_count * sizeof(float));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&top_indices_dev), topk_count * sizeof(long long));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&resized), dense_bytes);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&vertical), dense_bytes);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }

    (void)hipEventRecord(events[0], stream);
    err = hipMemcpyAsync(heatmaps_dev, heatmaps_host, heatmap_count * sizeof(float), hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[1], stream);

    hipLaunchKernelGGL(
        resize_cubic_fused_kernel,
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
    if (hipGetLastError() != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[2], stream);

    hipLaunchKernelGGL(
        vertical_max_fused_kernel,
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
    if (hipGetLastError() != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[3], stream);

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * static_cast<std::size_t>(topk) *
                                         (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(
            fused_horizontal_topk_kernel,
            dim3(total_bc),
            dim3(threads),
            shared_bytes,
            stream,
            resized,
            vertical,
            top_scores_dev,
            top_indices_dev,
            batch,
            channels,
            full_h,
            full_w,
            topk,
            threshold,
            nms_radius);
        if (hipGetLastError() != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    }
    (void)hipEventRecord(events[4], stream);

    err = hipMemcpyAsync(top_scores_host, top_scores_dev, topk_count * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[5], stream);

    err = hipMemcpyAsync(top_indices_host, top_indices_dev, topk_count * sizeof(long long), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[6], stream);

    err = hipEventSynchronize(events[6]);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }

    if (profile) {
        profile->h2d_ms = elapsed_ms(events[0], events[1]);
        profile->resize_ms = elapsed_ms(events[1], events[2]);
        profile->vertical_ms = elapsed_ms(events[2], events[3]);
        profile->horizontal_ms = 0.0f;
        profile->topk_ms = elapsed_ms(events[3], events[4]);
        profile->d2h_scores_ms = elapsed_ms(events[4], events[5]);
        profile->d2h_indices_ms = elapsed_ms(events[5], events[6]);
        profile->device_total_ms = elapsed_ms(events[1], events[4]);
        profile->total_ms = elapsed_ms(events[0], events[6]);
    }

cleanup:
    free_if_needed(heatmaps_dev);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    free_if_needed(resized);
    free_if_needed(vertical);
    for (int i = 0; i < 7; ++i) {
        if (events[i]) (void)hipEventDestroy(events[i]);
    }
    if (stream) (void)hipStreamDestroy(stream);
    return status;
}

}  // namespace

extern "C" int heatmap_topk_hip_run_host_fused(
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
    return run_fused_impl(
        heatmaps_host,
        top_scores_host,
        top_indices_host,
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
}

extern "C" int heatmap_topk_hip_run_host_fused_profile(
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
    HeatmapTopKHipProfile* profile) {
    if (!profile) return HIP_TOPK_INVALID_ARGUMENT;
    return run_fused_impl(
        heatmaps_host,
        top_scores_host,
        top_indices_host,
        batch,
        channels,
        in_h,
        in_w,
        full_h,
        full_w,
        topk,
        threshold,
        nms_radius,
        profile);
}
