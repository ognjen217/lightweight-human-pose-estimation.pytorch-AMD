#include "heatmap_topk_hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr float kInvalidScore = -1.0e9f;
constexpr int kMaxTopK = 64;
constexpr int kMaxSmartProposals = 256;
constexpr int kLocalTopPerThread = 8;

bool invalid_shape(
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    int lowres_nms_radius,
    int smart_proposals,
    int smart_local_radius) {
    return batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0 || full_h <= 0 || full_w <= 0 ||
           topk <= 0 || topk > kMaxTopK || lowres_nms_radius < 0 || smart_local_radius < 0 ||
           smart_proposals <= 0 || smart_proposals > kMaxSmartProposals;
}

__device__ __host__ inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __host__ inline float cubic_weight(float distance, float a) {
    const float x = fabsf(distance);
    const float x2 = x * x;
    const float x3 = x2 * x;
    if (x <= 1.0f) return (a + 2.0f) * x3 - (a + 3.0f) * x2 + 1.0f;
    if (x < 2.0f) return a * x3 - 5.0f * a * x2 + 8.0f * a * x - 4.0f * a;
    return 0.0f;
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

__device__ inline float cubic_sample_lowres(
    const float* __restrict__ heatmaps,
    int bc,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int y_full,
    int x_full) {
    constexpr float a = -0.75f;
    const float src_x = (static_cast<float>(x_full) + 0.5f) * (static_cast<float>(in_w) / static_cast<float>(full_w)) - 0.5f;
    const float src_y = (static_cast<float>(y_full) + 0.5f) * (static_cast<float>(in_h) / static_cast<float>(full_h)) - 0.5f;
    const int base_x = static_cast<int>(floorf(src_x));
    const int base_y = static_cast<int>(floorf(src_y));
    const std::size_t in_plane = static_cast<std::size_t>(bc) * in_h * in_w;
    float acc = 0.0f;
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
    return acc;
}

__global__ void lowres_proposal_kernel(
    const float* __restrict__ heatmaps,
    float* __restrict__ proposal_scores,
    int* __restrict__ proposal_indices,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int smart_proposals,
    float threshold,
    int lowres_nms_radius) {
    const int bc = blockIdx.x;
    const int total_bc = batch * channels;
    if (bc >= total_bc) return;

    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    int* sh_indices = reinterpret_cast<int*>(sh_scores + blockDim.x * kLocalTopPerThread);

    const int tid = threadIdx.x;
    float local_scores[kLocalTopPerThread];
    long long local_indices[kLocalTopPerThread];
    for (int k = 0; k < kLocalTopPerThread; ++k) {
        local_scores[k] = kInvalidScore;
        local_indices[k] = 2147483647LL;
    }

    const int plane_size = in_h * in_w;
    const std::size_t plane = static_cast<std::size_t>(bc) * plane_size;
    for (int idx = tid; idx < plane_size; idx += blockDim.x) {
        const float score = heatmaps[plane + idx];
        if (score <= threshold) continue;
        const int x = idx % in_w;
        const int y = idx / in_w;
        const int y0 = clamp_int(y - lowres_nms_radius, 0, in_h - 1);
        const int y1 = clamp_int(y + lowres_nms_radius, 0, in_h - 1);
        const int x0 = clamp_int(x - lowres_nms_radius, 0, in_w - 1);
        const int x1 = clamp_int(x + lowres_nms_radius, 0, in_w - 1);
        bool peak = true;
        for (int yy = y0; yy <= y1 && peak; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                const int nidx = yy * in_w + xx;
                const float v = heatmaps[plane + nidx];
                if (v > score || (v == score && nidx < idx)) {
                    peak = false;
                    break;
                }
            }
        }
        if (peak) {
            insert_topk(score, static_cast<long long>(idx), local_scores, local_indices, kLocalTopPerThread);
        }
    }

    const int base = tid * kLocalTopPerThread;
    for (int k = 0; k < kLocalTopPerThread; ++k) {
        sh_scores[base + k] = local_scores[k];
        sh_indices[base + k] = static_cast<int>(local_indices[k]);
    }
    __syncthreads();

    if (tid == 0) {
        float best_scores[kMaxSmartProposals];
        long long best_indices[kMaxSmartProposals];
        for (int k = 0; k < smart_proposals; ++k) {
            best_scores[k] = kInvalidScore;
            best_indices[k] = 2147483647LL;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            const int tbase = t * kLocalTopPerThread;
            for (int kk = 0; kk < kLocalTopPerThread; ++kk) {
                insert_topk(sh_scores[tbase + kk], static_cast<long long>(sh_indices[tbase + kk]), best_scores, best_indices, smart_proposals);
            }
        }
        const std::size_t out_base = static_cast<std::size_t>(bc) * smart_proposals;
        for (int k = 0; k < smart_proposals; ++k) {
            proposal_scores[out_base + k] = best_scores[k];
            proposal_indices[out_base + k] = (best_indices[k] == 2147483647LL) ? 0 : static_cast<int>(best_indices[k]);
        }
    }
}

__global__ void refine_proposals_kernel(
    const float* __restrict__ heatmaps,
    const float* __restrict__ proposal_scores,
    const int* __restrict__ proposal_indices,
    float* __restrict__ refined_scores,
    long long* __restrict__ refined_indices,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int smart_proposals,
    int smart_local_radius) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * channels * smart_proposals;
    if (linear >= total) return;
    const int proposal = linear % smart_proposals;
    const int bc = linear / smart_proposals;
    const std::size_t base = static_cast<std::size_t>(bc) * smart_proposals + proposal;
    if (proposal_scores[base] <= kInvalidScore * 0.5f) {
        refined_scores[base] = kInvalidScore;
        refined_indices[base] = 0LL;
        return;
    }

    const int lr_idx = proposal_indices[base];
    const int lr_x = lr_idx % in_w;
    const int lr_y = lr_idx / in_w;
    const float center_x_f = (static_cast<float>(lr_x) + 0.5f) * (static_cast<float>(full_w) / static_cast<float>(in_w)) - 0.5f;
    const float center_y_f = (static_cast<float>(lr_y) + 0.5f) * (static_cast<float>(full_h) / static_cast<float>(in_h)) - 0.5f;
    const int center_x = clamp_int(static_cast<int>(floorf(center_x_f + 0.5f)), 0, full_w - 1);
    const int center_y = clamp_int(static_cast<int>(floorf(center_y_f + 0.5f)), 0, full_h - 1);
    const int y0 = clamp_int(center_y - smart_local_radius, 0, full_h - 1);
    const int y1 = clamp_int(center_y + smart_local_radius, 0, full_h - 1);
    const int x0 = clamp_int(center_x - smart_local_radius, 0, full_w - 1);
    const int x1 = clamp_int(center_x + smart_local_radius, 0, full_w - 1);

    float best_score = kInvalidScore;
    long long best_index = 0LL;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float score = cubic_sample_lowres(heatmaps, bc, in_h, in_w, full_h, full_w, y, x);
            const long long full_idx = static_cast<long long>(y) * full_w + x;
            if (score > best_score || (score == best_score && full_idx < best_index)) {
                best_score = score;
                best_index = full_idx;
            }
        }
    }
    refined_scores[base] = best_score;
    refined_indices[base] = best_index;
}

__global__ void final_topk_kernel(
    const float* __restrict__ refined_scores,
    const long long* __restrict__ refined_indices,
    float* __restrict__ top_scores,
    long long* __restrict__ top_indices,
    int batch,
    int channels,
    int smart_proposals,
    int topk,
    float threshold) {
    const int bc = blockIdx.x;
    const int total_bc = batch * channels;
    if (bc >= total_bc) return;
    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    long long* sh_indices = reinterpret_cast<long long*>(sh_scores + blockDim.x * topk);
    const int tid = threadIdx.x;
    float local_scores[kMaxTopK];
    long long local_indices[kMaxTopK];
    for (int k = 0; k < topk; ++k) {
        local_scores[k] = kInvalidScore;
        local_indices[k] = 9223372036854775807LL;
    }
    const std::size_t base = static_cast<std::size_t>(bc) * smart_proposals;
    for (int i = tid; i < smart_proposals; i += blockDim.x) {
        const float s = refined_scores[base + i];
        if (s > threshold) {
            insert_topk(s, refined_indices[base + i], local_scores, local_indices, topk);
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
            const int tbase = t * topk;
            for (int kk = 0; kk < topk; ++kk) {
                insert_topk(sh_scores[tbase + kk], sh_indices[tbase + kk], best_scores, best_indices, topk);
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
        profile->h2d_ms = profile->resize_ms = profile->vertical_ms = profile->horizontal_ms = 0.0f;
        profile->topk_ms = profile->d2h_scores_ms = profile->d2h_indices_ms = 0.0f;
        profile->device_total_ms = profile->total_ms = 0.0f;
    }
}

void free_if_needed(float* ptr) { if (ptr) (void)hipFree(ptr); }
void free_if_needed(int* ptr) { if (ptr) (void)hipFree(ptr); }
void free_if_needed(long long* ptr) { if (ptr) (void)hipFree(ptr); }

float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    return ms;
}

int run_smart_impl(
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
    HeatmapTopKHipProfile* profile) {
    clear_profile(profile);
    if (!heatmaps_host || !top_scores_host || !top_indices_host) return HIP_TOPK_INVALID_ARGUMENT;
    if (invalid_shape(batch, channels, in_h, in_w, full_h, full_w, topk, lowres_nms_radius, smart_proposals, smart_local_radius)) return HIP_TOPK_INVALID_ARGUMENT;

    const int threads = 256;
    const int total_bc = batch * channels;
    const std::size_t heatmap_count = static_cast<std::size_t>(batch) * channels * in_h * in_w;
    const std::size_t proposals_count = static_cast<std::size_t>(total_bc) * smart_proposals;
    const std::size_t topk_count = static_cast<std::size_t>(total_bc) * topk;

    float* heatmaps_dev = nullptr;
    float* proposal_scores = nullptr;
    int* proposal_indices = nullptr;
    float* refined_scores = nullptr;
    long long* refined_indices = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;
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
    err = hipMalloc(reinterpret_cast<void**>(&proposal_scores), proposals_count * sizeof(float));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&proposal_indices), proposals_count * sizeof(int));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&refined_scores), proposals_count * sizeof(float));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&refined_indices), proposals_count * sizeof(long long));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&top_scores_dev), topk_count * sizeof(float));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    err = hipMalloc(reinterpret_cast<void**>(&top_indices_dev), topk_count * sizeof(long long));
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }

    (void)hipEventRecord(events[0], stream);
    err = hipMemcpyAsync(heatmaps_dev, heatmaps_host, heatmap_count * sizeof(float), hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    (void)hipEventRecord(events[1], stream);

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * kLocalTopPerThread * (sizeof(float) + sizeof(int));
        hipLaunchKernelGGL(lowres_proposal_kernel, dim3(total_bc), dim3(threads), shared_bytes, stream,
                           heatmaps_dev, proposal_scores, proposal_indices, batch, channels, in_h, in_w,
                           smart_proposals, threshold, lowres_nms_radius);
        if (hipGetLastError() != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    }
    (void)hipEventRecord(events[2], stream);

    {
        const int refine_total = total_bc * smart_proposals;
        const int refine_blocks = (refine_total + threads - 1) / threads;
        hipLaunchKernelGGL(refine_proposals_kernel, dim3(refine_blocks), dim3(threads), 0, stream,
                           heatmaps_dev, proposal_scores, proposal_indices, refined_scores, refined_indices,
                           batch, channels, in_h, in_w, full_h, full_w, smart_proposals, smart_local_radius);
        if (hipGetLastError() != hipSuccess) { status = HIP_TOPK_HIP_ERROR; goto cleanup; }
    }
    (void)hipEventRecord(events[3], stream);

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * topk * (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(final_topk_kernel, dim3(total_bc), dim3(threads), shared_bytes, stream,
                           refined_scores, refined_indices, top_scores_dev, top_indices_dev,
                           batch, channels, smart_proposals, topk, threshold);
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
    free_if_needed(proposal_scores);
    free_if_needed(proposal_indices);
    free_if_needed(refined_scores);
    free_if_needed(refined_indices);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    for (int i = 0; i < 7; ++i) if (events[i]) (void)hipEventDestroy(events[i]);
    if (stream) (void)hipStreamDestroy(stream);
    return status;
}

}  // namespace

extern "C" int heatmap_topk_hip_run_host_smart(
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
    int smart_local_radius) {
    return run_smart_impl(heatmaps_host, top_scores_host, top_indices_host, batch, channels, in_h, in_w, full_h, full_w,
                          topk, threshold, lowres_nms_radius, smart_proposals, smart_local_radius, nullptr);
}

extern "C" int heatmap_topk_hip_run_host_smart_profile(
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
    HeatmapTopKHipProfile* profile) {
    if (!profile) return HIP_TOPK_INVALID_ARGUMENT;
    return run_smart_impl(heatmaps_host, top_scores_host, top_indices_host, batch, channels, in_h, in_w, full_h, full_w,
                          topk, threshold, lowres_nms_radius, smart_proposals, smart_local_radius, profile);
}
