#include "split_hip_fused.h"

#include "paf_prune_hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr float kInvalidScore = -1.0e9f;
constexpr int kMaxTopK = 64;
constexpr int kMaxSmartProposals = 256;
constexpr int kLocalTopPerThread = 8;
constexpr int kExpectedHeatmapChannels = 18;
constexpr int kExpectedPafChannels = 38;
constexpr int kNumLimbs = 19;

bool invalid_shape(
    int batch,
    int channels,
    int paf_channels,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int topk,
    int limb_topm,
    int lowres_nms_radius,
    int smart_proposals,
    int smart_local_radius,
    int points_per_limb) {
    return batch <= 0 || channels != kExpectedHeatmapChannels || paf_channels != kExpectedPafChannels ||
           in_h <= 0 || in_w <= 0 || full_h <= 0 || full_w <= 0 ||
           topk <= 0 || topk > kMaxTopK || limb_topm <= 0 || limb_topm > kMaxTopK ||
           lowres_nms_radius < 0 || smart_local_radius < 0 ||
           smart_proposals <= 0 || smart_proposals > kMaxSmartProposals ||
           points_per_limb <= 0 || points_per_limb > 64;
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

template <typename T>
void free_if_needed(T* ptr) {
    if (ptr) (void)hipFree(ptr);
}

int check_hip() {
    return hipGetLastError() == hipSuccess ? HIP_SPLIT_FUSED_SUCCESS : HIP_SPLIT_FUSED_HIP_ERROR;
}

}  // namespace

const char* split_hip_fused_status_string(int status) {
    switch (status) {
        case HIP_SPLIT_FUSED_SUCCESS: return "HIP_SPLIT_FUSED_SUCCESS";
        case HIP_SPLIT_FUSED_INVALID_ARGUMENT: return "HIP_SPLIT_FUSED_INVALID_ARGUMENT";
        case HIP_SPLIT_FUSED_HIP_ERROR: return "HIP_SPLIT_FUSED_HIP_ERROR";
        case HIP_SPLIT_FUSED_NOT_IMPLEMENTED: return "HIP_SPLIT_FUSED_NOT_IMPLEMENTED";
        default: return "HIP_SPLIT_FUSED_UNKNOWN_STATUS";
    }
}

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
    float paf_cubic_a) {
    if (!heatmaps_host || !pafs_host || !top_scores_host || !top_indices_host || !limb_top_pair_a_idx_host ||
        !limb_top_pair_b_idx_host || !limb_top_pair_score_host || !limb_top_pair_valid_host) {
        return HIP_SPLIT_FUSED_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, channels, paf_channels, in_h, in_w, full_h, full_w, topk, limb_topm,
                      lowres_nms_radius, smart_proposals, smart_local_radius, points_per_limb)) {
        return HIP_SPLIT_FUSED_INVALID_ARGUMENT;
    }

    const int threads = 256;
    const int total_bc = batch * channels;
    const std::size_t heatmap_count = static_cast<std::size_t>(batch) * channels * in_h * in_w;
    const std::size_t paf_count = static_cast<std::size_t>(batch) * paf_channels * in_h * in_w;
    const std::size_t proposals_count = static_cast<std::size_t>(total_bc) * smart_proposals;
    const std::size_t topk_count = static_cast<std::size_t>(total_bc) * topk;
    const std::size_t out_count = static_cast<std::size_t>(batch) * kNumLimbs * limb_topm;

    float* heatmaps_dev = nullptr;
    float* pafs_dev = nullptr;
    float* proposal_scores = nullptr;
    int* proposal_indices = nullptr;
    float* refined_scores = nullptr;
    long long* refined_indices = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;
    long long* a_dev = nullptr;
    long long* b_dev = nullptr;
    float* pair_score_dev = nullptr;
    float* pair_valid_dev = nullptr;
    hipStream_t stream = nullptr;
    int status = HIP_SPLIT_FUSED_SUCCESS;

    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) return HIP_SPLIT_FUSED_HIP_ERROR;

    auto alloc = [&](auto** ptr, std::size_t bytes) -> bool {
        return hipMalloc(reinterpret_cast<void**>(ptr), bytes) == hipSuccess;
    };

    if (!alloc(&heatmaps_dev, heatmap_count * sizeof(float)) ||
        !alloc(&pafs_dev, paf_count * sizeof(float)) ||
        !alloc(&proposal_scores, proposals_count * sizeof(float)) ||
        !alloc(&proposal_indices, proposals_count * sizeof(int)) ||
        !alloc(&refined_scores, proposals_count * sizeof(float)) ||
        !alloc(&refined_indices, proposals_count * sizeof(long long)) ||
        !alloc(&top_scores_dev, topk_count * sizeof(float)) ||
        !alloc(&top_indices_dev, topk_count * sizeof(long long)) ||
        !alloc(&a_dev, out_count * sizeof(long long)) ||
        !alloc(&b_dev, out_count * sizeof(long long)) ||
        !alloc(&pair_score_dev, out_count * sizeof(float)) ||
        !alloc(&pair_valid_dev, out_count * sizeof(float))) {
        status = HIP_SPLIT_FUSED_HIP_ERROR;
        goto cleanup;
    }

    err = hipMemcpyAsync(heatmaps_dev, heatmaps_host, heatmap_count * sizeof(float), hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(pafs_dev, pafs_host, paf_count * sizeof(float), hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * kLocalTopPerThread * (sizeof(float) + sizeof(int));
        hipLaunchKernelGGL(lowres_proposal_kernel, dim3(total_bc), dim3(threads), shared_bytes, stream,
                           heatmaps_dev, proposal_scores, proposal_indices, batch, channels, in_h, in_w,
                           smart_proposals, threshold, lowres_nms_radius);
        status = check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const int refine_total = total_bc * smart_proposals;
        const int refine_blocks = (refine_total + threads - 1) / threads;
        hipLaunchKernelGGL(refine_proposals_kernel, dim3(refine_blocks), dim3(threads), 0, stream,
                           heatmaps_dev, proposal_scores, proposal_indices, refined_scores, refined_indices,
                           batch, channels, in_h, in_w, full_h, full_w, smart_proposals, smart_local_radius);
        status = check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * topk * (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(final_topk_kernel, dim3(total_bc), dim3(threads), shared_bytes, stream,
                           refined_scores, refined_indices, top_scores_dev, top_indices_dev,
                           batch, channels, smart_proposals, topk, threshold);
        status = check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const int paf_status = paf_prune_hip_run(
            pafs_dev,
            top_scores_dev,
            top_indices_dev,
            a_dev,
            b_dev,
            pair_score_dev,
            pair_valid_dev,
            batch,
            topk,
            limb_topm,
            in_h,
            in_w,
            full_h,
            full_w,
            points_per_limb,
            min_paf_score,
            success_ratio_thr,
            min_pair_score,
            paf_cubic_a,
            stream);
        if (paf_status != HIP_PAF_PRUNE_SUCCESS) {
            status = HIP_SPLIT_FUSED_HIP_ERROR;
            goto cleanup;
        }
    }

    err = hipMemcpyAsync(top_scores_host, top_scores_dev, topk_count * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(top_indices_host, top_indices_dev, topk_count * sizeof(long long), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(limb_top_pair_a_idx_host, a_dev, out_count * sizeof(long long), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(limb_top_pair_b_idx_host, b_dev, out_count * sizeof(long long), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(limb_top_pair_score_host, pair_score_dev, out_count * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }
    err = hipMemcpyAsync(limb_top_pair_valid_host, pair_valid_dev, out_count * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) { status = HIP_SPLIT_FUSED_HIP_ERROR; goto cleanup; }

    err = hipStreamSynchronize(stream);
    if (err != hipSuccess) status = HIP_SPLIT_FUSED_HIP_ERROR;

cleanup:
    free_if_needed(heatmaps_dev);
    free_if_needed(pafs_dev);
    free_if_needed(proposal_scores);
    free_if_needed(proposal_indices);
    free_if_needed(refined_scores);
    free_if_needed(refined_indices);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    free_if_needed(a_dev);
    free_if_needed(b_dev);
    free_if_needed(pair_score_dev);
    free_if_needed(pair_valid_dev);
    if (stream) (void)hipStreamDestroy(stream);
    return status;
}
