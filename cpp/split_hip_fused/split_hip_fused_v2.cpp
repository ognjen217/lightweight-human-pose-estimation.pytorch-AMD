#include "split_hip_fused.h"

// Reuse the already validated smart heatmap kernels in the same translation unit.
// We do not call run_smart_impl(), because it copies TopK back to host.  Instead
// we launch lowres_proposal_kernel -> refine_proposals_kernel -> final_topk_kernel
// directly, then launch the inline PAF score/prune kernels below on the same HIP
// stream.
#include "../heatmap_topk_hip/heatmap_topk_hip_smart.cpp"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kFusedHeatmapChannels = 18;
constexpr int kFusedPafChannels = 38;
constexpr int kFusedNumLimbs = 19;
constexpr int kFusedMaxTopK = 64;
constexpr int kFusedMaxTopM = 64;
constexpr float kFusedInvalidScore = -1.0e9f;

__device__ __constant__ int c_fused_body_parts_kpt[kFusedNumLimbs][2] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7},
    {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
    {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17},
};

__device__ __constant__ int c_fused_body_parts_paf[kFusedNumLimbs][2] = {
    {12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1},
    {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31},
    {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27},
};

bool fused_invalid_shape(
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
    return batch <= 0 || channels != kFusedHeatmapChannels || paf_channels != kFusedPafChannels ||
           in_h <= 0 || in_w <= 0 || full_h <= 0 || full_w <= 0 ||
           topk <= 0 || topk > kFusedMaxTopK || limb_topm <= 0 || limb_topm > kFusedMaxTopM ||
           lowres_nms_radius < 0 || smart_local_radius < 0 ||
           smart_proposals <= 0 || smart_proposals > kMaxSmartProposals ||
           points_per_limb <= 0 || points_per_limb > 64;
}

__device__ float fused_sample_paf_cubic(
    const float* __restrict__ pafs,
    int b,
    int channel,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    float x_full,
    float y_full,
    float cubic_a) {
    const float src_x = (x_full + 0.5f) * (static_cast<float>(in_w) / static_cast<float>(full_w)) - 0.5f;
    const float src_y = (y_full + 0.5f) * (static_cast<float>(in_h) / static_cast<float>(full_h)) - 0.5f;
    const int base_x = static_cast<int>(floorf(src_x));
    const int base_y = static_cast<int>(floorf(src_y));
    const std::size_t plane = (static_cast<std::size_t>(b) * kFusedPafChannels + channel) * in_h * in_w;

    float acc = 0.0f;
    for (int oy = -1; oy <= 2; ++oy) {
        const int raw_y = base_y + oy;
        const int yy = clamp_int(raw_y, 0, in_h - 1);
        const float wy = cubic_weight(src_y - static_cast<float>(raw_y), cubic_a);
        for (int ox = -1; ox <= 2; ++ox) {
            const int raw_x = base_x + ox;
            const int xx = clamp_int(raw_x, 0, in_w - 1);
            const float wx = cubic_weight(src_x - static_cast<float>(raw_x), cubic_a);
            acc += pafs[plane + static_cast<std::size_t>(yy) * in_w + xx] * wy * wx;
        }
    }
    return acc;
}

__device__ inline void fused_insert_topm(float score, long long flat_idx, float* scores, long long* indices, int topm) {
    for (int k = 0; k < topm; ++k) {
        if (score > scores[k] || (score == scores[k] && flat_idx < indices[k])) {
            for (int j = topm - 1; j > k; --j) {
                scores[j] = scores[j - 1];
                indices[j] = indices[j - 1];
            }
            scores[k] = score;
            indices[k] = flat_idx;
            break;
        }
    }
}

__global__ void fused_score_pairs_kernel(
    const float* __restrict__ pafs,
    const float* __restrict__ top_scores,
    const long long* __restrict__ top_indices,
    float* __restrict__ pair_scores,
    int batch,
    int topk,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float paf_cubic_a) {
    const std::size_t flat_dim = static_cast<std::size_t>(batch) * kFusedNumLimbs * topk * topk;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < flat_dim;
         linear += stride) {
        const int b_idx = static_cast<int>(linear % topk);
        const int a_idx = static_cast<int>((linear / topk) % topk);
        const int limb = static_cast<int>((linear / (static_cast<std::size_t>(topk) * topk)) % kFusedNumLimbs);
        const int b = static_cast<int>(linear / (static_cast<std::size_t>(kFusedNumLimbs) * topk * topk));

        const int kpt_a = c_fused_body_parts_kpt[limb][0];
        const int kpt_b = c_fused_body_parts_kpt[limb][1];
        const int paf_x = c_fused_body_parts_paf[limb][0];
        const int paf_y = c_fused_body_parts_paf[limb][1];

        const std::size_t score_a_off = (static_cast<std::size_t>(b) * kFusedHeatmapChannels + kpt_a) * topk + a_idx;
        const std::size_t score_b_off = (static_cast<std::size_t>(b) * kFusedHeatmapChannels + kpt_b) * topk + b_idx;
        const float score_a = top_scores[score_a_off];
        const float score_b = top_scores[score_b_off];
        if (score_a <= -1.0e8f || score_b <= -1.0e8f) {
            pair_scores[linear] = kFusedInvalidScore;
            continue;
        }

        const long long flat_a = top_indices[score_a_off];
        const long long flat_b = top_indices[score_b_off];
        const float ax = static_cast<float>(flat_a % full_w);
        const float ay = floorf(static_cast<float>(flat_a) / static_cast<float>(full_w));
        const float bx = static_cast<float>(flat_b % full_w);
        const float by = floorf(static_cast<float>(flat_b) / static_cast<float>(full_w));

        const float dx = bx - ax;
        const float dy = by - ay;
        const float norm = sqrtf(dx * dx + dy * dy);
        if (norm <= 1.0e-6f) {
            pair_scores[linear] = kFusedInvalidScore;
            continue;
        }
        const float vx = dx / (norm + 1.0e-6f);
        const float vy = dy / (norm + 1.0e-6f);

        float score_sum = 0.0f;
        int valid_num = 0;
        const float denom = static_cast<float>(points_per_limb > 1 ? points_per_limb - 1 : 1);
        for (int p = 0; p < points_per_limb; ++p) {
            const float alpha = static_cast<float>(p) / denom;
            const float px = ax + dx * alpha;
            const float py = ay + dy * alpha;
            const float field_x = fused_sample_paf_cubic(pafs, b, paf_x, in_h, in_w, full_h, full_w, px, py, paf_cubic_a);
            const float field_y = fused_sample_paf_cubic(pafs, b, paf_y, in_h, in_w, full_h, full_w, px, py, paf_cubic_a);
            const float dot = field_x * vx + field_y * vy;
            if (dot > min_paf_score) {
                score_sum += dot;
                valid_num += 1;
            }
        }

        const float affinity = score_sum / (static_cast<float>(valid_num) + 1.0e-6f);
        const float success_ratio = static_cast<float>(valid_num) / static_cast<float>(points_per_limb);
        const bool valid = (affinity > 0.0f) && (success_ratio > success_ratio_thr);
        pair_scores[linear] = valid ? affinity : kFusedInvalidScore;
    }
}

__global__ void fused_prune_pairs_kernel(
    const float* __restrict__ pair_scores,
    long long* __restrict__ limb_top_pair_a_idx,
    long long* __restrict__ limb_top_pair_b_idx,
    float* __restrict__ limb_top_pair_score,
    float* __restrict__ limb_top_pair_valid,
    int batch,
    int topk,
    int limb_topm,
    float min_pair_score) {
    const int bl = blockIdx.x;
    const int total = batch * kFusedNumLimbs;
    if (bl >= total) return;
    const int tid = threadIdx.x;
    const int flat_dim = topk * topk;

    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    long long* sh_indices = reinterpret_cast<long long*>(sh_scores + blockDim.x * limb_topm);

    float local_scores[kFusedMaxTopM];
    long long local_indices[kFusedMaxTopM];
    for (int k = 0; k < limb_topm; ++k) {
        local_scores[k] = kFusedInvalidScore;
        local_indices[k] = 9223372036854775807LL;
    }

    const std::size_t base = static_cast<std::size_t>(bl) * flat_dim;
    for (int idx = tid; idx < flat_dim; idx += blockDim.x) {
        const float s = pair_scores[base + idx];
        fused_insert_topm(s, static_cast<long long>(idx), local_scores, local_indices, limb_topm);
    }

    const int sh_base = tid * limb_topm;
    for (int k = 0; k < limb_topm; ++k) {
        sh_scores[sh_base + k] = local_scores[k];
        sh_indices[sh_base + k] = local_indices[k];
    }
    __syncthreads();

    if (tid == 0) {
        float best_scores[kFusedMaxTopM];
        long long best_indices[kFusedMaxTopM];
        for (int k = 0; k < limb_topm; ++k) {
            best_scores[k] = kFusedInvalidScore;
            best_indices[k] = 9223372036854775807LL;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            const int tbase = t * limb_topm;
            for (int kk = 0; kk < limb_topm; ++kk) {
                fused_insert_topm(sh_scores[tbase + kk], sh_indices[tbase + kk], best_scores, best_indices, limb_topm);
            }
        }
        const std::size_t out_base = static_cast<std::size_t>(bl) * limb_topm;
        for (int k = 0; k < limb_topm; ++k) {
            const long long flat = best_indices[k] == 9223372036854775807LL ? 0LL : best_indices[k];
            const float s = best_scores[k];
            limb_top_pair_a_idx[out_base + k] = flat / topk;
            limb_top_pair_b_idx[out_base + k] = flat % topk;
            limb_top_pair_score[out_base + k] = s;
            limb_top_pair_valid[out_base + k] = (s > min_pair_score) ? 1.0f : 0.0f;
        }
    }
}

template <typename T>
void fused_free_if_needed(T* ptr) {
    if (ptr) (void)hipFree(ptr);
}

int fused_check_hip() {
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
    if (fused_invalid_shape(batch, channels, paf_channels, in_h, in_w, full_h, full_w, topk, limb_topm,
                            lowres_nms_radius, smart_proposals, smart_local_radius, points_per_limb)) {
        return HIP_SPLIT_FUSED_INVALID_ARGUMENT;
    }

    const int threads = 256;
    const int total_bc = batch * channels;
    const std::size_t heatmap_count = static_cast<std::size_t>(batch) * channels * in_h * in_w;
    const std::size_t paf_count = static_cast<std::size_t>(batch) * paf_channels * in_h * in_w;
    const std::size_t proposals_count = static_cast<std::size_t>(total_bc) * smart_proposals;
    const std::size_t topk_count = static_cast<std::size_t>(total_bc) * topk;
    const std::size_t pair_count = static_cast<std::size_t>(batch) * kFusedNumLimbs * topk * topk;
    const std::size_t out_count = static_cast<std::size_t>(batch) * kFusedNumLimbs * limb_topm;

    float* heatmaps_dev = nullptr;
    float* pafs_dev = nullptr;
    float* proposal_scores = nullptr;
    int* proposal_indices = nullptr;
    float* refined_scores = nullptr;
    long long* refined_indices = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;
    float* pair_scores_dev = nullptr;
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
        !alloc(&pair_scores_dev, pair_count * sizeof(float)) ||
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
        status = fused_check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const int refine_total = total_bc * smart_proposals;
        const int refine_blocks = (refine_total + threads - 1) / threads;
        hipLaunchKernelGGL(refine_proposals_kernel, dim3(refine_blocks), dim3(threads), 0, stream,
                           heatmaps_dev, proposal_scores, proposal_indices, refined_scores, refined_indices,
                           batch, channels, in_h, in_w, full_h, full_w, smart_proposals, smart_local_radius);
        status = fused_check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * topk * (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(final_topk_kernel, dim3(total_bc), dim3(threads), shared_bytes, stream,
                           refined_scores, refined_indices, top_scores_dev, top_indices_dev,
                           batch, channels, smart_proposals, topk, threshold);
        status = fused_check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const int score_threads = 256;
        const int score_blocks = static_cast<int>((pair_count + score_threads - 1) / score_threads);
        hipLaunchKernelGGL(fused_score_pairs_kernel, dim3(score_blocks), dim3(score_threads), 0, stream,
                           pafs_dev, top_scores_dev, top_indices_dev, pair_scores_dev,
                           batch, topk, in_h, in_w, full_h, full_w, points_per_limb,
                           min_paf_score, success_ratio_thr, paf_cubic_a);
        status = fused_check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
    }

    {
        const int prune_threads = 128;
        const int prune_blocks = batch * kFusedNumLimbs;
        const std::size_t shared_bytes = static_cast<std::size_t>(prune_threads) * static_cast<std::size_t>(limb_topm) *
                                         (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(fused_prune_pairs_kernel, dim3(prune_blocks), dim3(prune_threads), shared_bytes, stream,
                           pair_scores_dev, a_dev, b_dev, pair_score_dev, pair_valid_dev,
                           batch, topk, limb_topm, min_pair_score);
        status = fused_check_hip();
        if (status != HIP_SPLIT_FUSED_SUCCESS) goto cleanup;
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
    fused_free_if_needed(heatmaps_dev);
    fused_free_if_needed(pafs_dev);
    fused_free_if_needed(proposal_scores);
    fused_free_if_needed(proposal_indices);
    fused_free_if_needed(refined_scores);
    fused_free_if_needed(refined_indices);
    fused_free_if_needed(top_scores_dev);
    fused_free_if_needed(top_indices_dev);
    fused_free_if_needed(pair_scores_dev);
    fused_free_if_needed(a_dev);
    fused_free_if_needed(b_dev);
    fused_free_if_needed(pair_score_dev);
    fused_free_if_needed(pair_valid_dev);
    if (stream) (void)hipStreamDestroy(stream);
    return status;
}
