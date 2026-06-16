#include "paf_prune_hip.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr float kInvalidScore = -1.0e9f;
constexpr int kNumLimbs = 19;
constexpr int kNumPafChannels = 38;
constexpr int kMaxTopK = 64;
constexpr int kMaxTopM = 64;

__device__ __constant__ int c_body_parts_kpt[kNumLimbs][2] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7},
    {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
    {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17},
};

__device__ __constant__ int c_body_parts_paf[kNumLimbs][2] = {
    {12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1},
    {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31},
    {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27},
};

bool invalid_shape(
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb) {
    return batch <= 0 || topk <= 0 || topk > kMaxTopK || limb_topm <= 0 || limb_topm > kMaxTopM ||
           in_h <= 0 || in_w <= 0 || full_h <= 0 || full_w <= 0 || points_per_limb <= 0 || points_per_limb > 64;
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

__device__ float sample_paf_cubic(
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
    const std::size_t plane = (static_cast<std::size_t>(b) * kNumPafChannels + channel) * in_h * in_w;

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

__global__ void score_pairs_kernel(
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
    const std::size_t flat_dim = static_cast<std::size_t>(batch) * kNumLimbs * topk * topk;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < flat_dim;
         linear += stride) {
        const int b_idx = static_cast<int>(linear % topk);
        const int a_idx = static_cast<int>((linear / topk) % topk);
        const int limb = static_cast<int>((linear / (static_cast<std::size_t>(topk) * topk)) % kNumLimbs);
        const int b = static_cast<int>(linear / (static_cast<std::size_t>(kNumLimbs) * topk * topk));

        const int kpt_a = c_body_parts_kpt[limb][0];
        const int kpt_b = c_body_parts_kpt[limb][1];
        const int paf_x = c_body_parts_paf[limb][0];
        const int paf_y = c_body_parts_paf[limb][1];

        const std::size_t score_a_off = (static_cast<std::size_t>(b) * 18 + kpt_a) * topk + a_idx;
        const std::size_t score_b_off = (static_cast<std::size_t>(b) * 18 + kpt_b) * topk + b_idx;
        const float score_a = top_scores[score_a_off];
        const float score_b = top_scores[score_b_off];
        if (score_a <= -1.0e8f || score_b <= -1.0e8f) {
            pair_scores[linear] = kInvalidScore;
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
            pair_scores[linear] = kInvalidScore;
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
            const float field_x = sample_paf_cubic(pafs, b, paf_x, in_h, in_w, full_h, full_w, px, py, paf_cubic_a);
            const float field_y = sample_paf_cubic(pafs, b, paf_y, in_h, in_w, full_h, full_w, px, py, paf_cubic_a);
            const float dot = field_x * vx + field_y * vy;
            if (dot > min_paf_score) {
                score_sum += dot;
                valid_num += 1;
            }
        }

        const float affinity = score_sum / (static_cast<float>(valid_num) + 1.0e-6f);
        const float success_ratio = static_cast<float>(valid_num) / static_cast<float>(points_per_limb);
        const bool valid = (affinity > 0.0f) && (success_ratio > success_ratio_thr);
        pair_scores[linear] = valid ? affinity : kInvalidScore;
    }
}

__device__ inline void insert_topm(float score, long long flat_idx, float* scores, long long* indices, int topm) {
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

__global__ void prune_pairs_kernel(
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
    const int total = batch * kNumLimbs;
    if (bl >= total) return;
    const int tid = threadIdx.x;
    const int flat_dim = topk * topk;

    extern __shared__ unsigned char shared_raw[];
    float* sh_scores = reinterpret_cast<float*>(shared_raw);
    long long* sh_indices = reinterpret_cast<long long*>(sh_scores + blockDim.x * limb_topm);

    float local_scores[kMaxTopM];
    long long local_indices[kMaxTopM];
    for (int k = 0; k < limb_topm; ++k) {
        local_scores[k] = kInvalidScore;
        local_indices[k] = 9223372036854775807LL;
    }

    const std::size_t base = static_cast<std::size_t>(bl) * flat_dim;
    for (int idx = tid; idx < flat_dim; idx += blockDim.x) {
        const float s = pair_scores[base + idx];
        insert_topm(s, static_cast<long long>(idx), local_scores, local_indices, limb_topm);
    }

    const int sh_base = tid * limb_topm;
    for (int k = 0; k < limb_topm; ++k) {
        sh_scores[sh_base + k] = local_scores[k];
        sh_indices[sh_base + k] = local_indices[k];
    }
    __syncthreads();

    if (tid == 0) {
        float best_scores[kMaxTopM];
        long long best_indices[kMaxTopM];
        for (int k = 0; k < limb_topm; ++k) {
            best_scores[k] = kInvalidScore;
            best_indices[k] = 9223372036854775807LL;
        }
        for (int t = 0; t < blockDim.x; ++t) {
            const int tbase = t * limb_topm;
            for (int kk = 0; kk < limb_topm; ++kk) {
                insert_topm(sh_scores[tbase + kk], sh_indices[tbase + kk], best_scores, best_indices, limb_topm);
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

int check_last_error() {
    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? HIP_PAF_PRUNE_SUCCESS : HIP_PAF_PRUNE_HIP_ERROR;
}

template <typename T>
void free_if_needed(T* ptr) {
    if (ptr) (void)hipFree(ptr);
}

float elapsed_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

}  // namespace

const char* paf_prune_hip_status_string(int status) {
    switch (status) {
        case HIP_PAF_PRUNE_SUCCESS: return "HIP_PAF_PRUNE_SUCCESS";
        case HIP_PAF_PRUNE_INVALID_ARGUMENT: return "HIP_PAF_PRUNE_INVALID_ARGUMENT";
        case HIP_PAF_PRUNE_HIP_ERROR: return "HIP_PAF_PRUNE_HIP_ERROR";
        case HIP_PAF_PRUNE_NOT_IMPLEMENTED: return "HIP_PAF_PRUNE_NOT_IMPLEMENTED";
        default: return "HIP_PAF_PRUNE_UNKNOWN_STATUS";
    }
}

int paf_prune_hip_run(
    const float* pafs_dev,
    const float* top_scores_dev,
    const long long* top_indices_dev,
    long long* limb_top_pair_a_idx_dev,
    long long* limb_top_pair_b_idx_dev,
    float* limb_top_pair_score_dev,
    float* limb_top_pair_valid_dev,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a,
    void* hip_stream) {
    if (!pafs_dev || !top_scores_dev || !top_indices_dev || !limb_top_pair_a_idx_dev || !limb_top_pair_b_idx_dev ||
        !limb_top_pair_score_dev || !limb_top_pair_valid_dev) {
        return HIP_PAF_PRUNE_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, topk, limb_topm, in_h, in_w, full_h, full_w, points_per_limb)) {
        return HIP_PAF_PRUNE_INVALID_ARGUMENT;
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
    const std::size_t pair_count = static_cast<std::size_t>(batch) * kNumLimbs * topk * topk;
    float* pair_scores_dev = nullptr;
    hipError_t err = hipMalloc(reinterpret_cast<void**>(&pair_scores_dev), pair_count * sizeof(float));
    if (err != hipSuccess) return HIP_PAF_PRUNE_HIP_ERROR;

    const int score_threads = 256;
    const int score_blocks = static_cast<int>((pair_count + score_threads - 1) / score_threads);
    hipLaunchKernelGGL(
        score_pairs_kernel,
        dim3(score_blocks),
        dim3(score_threads),
        0,
        stream,
        pafs_dev,
        top_scores_dev,
        top_indices_dev,
        pair_scores_dev,
        batch,
        topk,
        in_h,
        in_w,
        full_h,
        full_w,
        points_per_limb,
        min_paf_score,
        success_ratio_thr,
        paf_cubic_a);
    int status = check_last_error();
    if (status != HIP_PAF_PRUNE_SUCCESS) goto cleanup;

    {
        const int prune_threads = 128;
        const int prune_blocks = batch * kNumLimbs;
        const std::size_t shared_bytes = static_cast<std::size_t>(prune_threads) * static_cast<std::size_t>(limb_topm) *
                                         (sizeof(float) + sizeof(long long));
        hipLaunchKernelGGL(
            prune_pairs_kernel,
            dim3(prune_blocks),
            dim3(prune_threads),
            shared_bytes,
            stream,
            pair_scores_dev,
            limb_top_pair_a_idx_dev,
            limb_top_pair_b_idx_dev,
            limb_top_pair_score_dev,
            limb_top_pair_valid_dev,
            batch,
            topk,
            limb_topm,
            min_pair_score);
        status = check_last_error();
        if (status != HIP_PAF_PRUNE_SUCCESS) goto cleanup;
    }

    err = (stream == nullptr) ? hipDeviceSynchronize() : hipStreamSynchronize(stream);
    if (err != hipSuccess) status = HIP_PAF_PRUNE_HIP_ERROR;

cleanup:
    free_if_needed(pair_scores_dev);
    return status;
}

int paf_prune_hip_run_host_profile(
    const float* pafs_host,
    const float* top_scores_host,
    const long long* top_indices_host,
    long long* limb_top_pair_a_idx_host,
    long long* limb_top_pair_b_idx_host,
    float* limb_top_pair_score_host,
    float* limb_top_pair_valid_host,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a,
    PafPruneHipProfile* profile) {
    if (!pafs_host || !top_scores_host || !top_indices_host || !limb_top_pair_a_idx_host || !limb_top_pair_b_idx_host ||
        !limb_top_pair_score_host || !limb_top_pair_valid_host) {
        return HIP_PAF_PRUNE_INVALID_ARGUMENT;
    }
    if (invalid_shape(batch, topk, limb_topm, in_h, in_w, full_h, full_w, points_per_limb)) {
        return HIP_PAF_PRUNE_INVALID_ARGUMENT;
    }
    PafPruneHipProfile local_profile{};
    auto t_total0 = std::chrono::steady_clock::now();

    const std::size_t paf_count = static_cast<std::size_t>(batch) * kNumPafChannels * in_h * in_w;
    const std::size_t top_score_count = static_cast<std::size_t>(batch) * 18 * topk;
    const std::size_t out_count = static_cast<std::size_t>(batch) * kNumLimbs * limb_topm;

    float* pafs_dev = nullptr;
    float* top_scores_dev = nullptr;
    long long* top_indices_dev = nullptr;
    long long* a_dev = nullptr;
    long long* b_dev = nullptr;
    float* score_dev = nullptr;
    float* valid_dev = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void**>(&pafs_dev), paf_count * sizeof(float));
    if (err != hipSuccess) return HIP_PAF_PRUNE_HIP_ERROR;
    err = hipMalloc(reinterpret_cast<void**>(&top_scores_dev), top_score_count * sizeof(float));
    if (err != hipSuccess) goto hip_error;
    err = hipMalloc(reinterpret_cast<void**>(&top_indices_dev), top_score_count * sizeof(long long));
    if (err != hipSuccess) goto hip_error;
    err = hipMalloc(reinterpret_cast<void**>(&a_dev), out_count * sizeof(long long));
    if (err != hipSuccess) goto hip_error;
    err = hipMalloc(reinterpret_cast<void**>(&b_dev), out_count * sizeof(long long));
    if (err != hipSuccess) goto hip_error;
    err = hipMalloc(reinterpret_cast<void**>(&score_dev), out_count * sizeof(float));
    if (err != hipSuccess) goto hip_error;
    err = hipMalloc(reinterpret_cast<void**>(&valid_dev), out_count * sizeof(float));
    if (err != hipSuccess) goto hip_error;

    {
        auto t0 = std::chrono::steady_clock::now();
        err = hipMemcpy(pafs_dev, pafs_host, paf_count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) goto hip_error;
        err = hipMemcpy(top_scores_dev, top_scores_host, top_score_count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) goto hip_error;
        err = hipMemcpy(top_indices_dev, top_indices_host, top_score_count * sizeof(long long), hipMemcpyHostToDevice);
        if (err != hipSuccess) goto hip_error;
        auto t1 = std::chrono::steady_clock::now();
        local_profile.h2d_ms = elapsed_ms(t0, t1);
    }

    {
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        hipEventRecord(e0, nullptr);
        int status = paf_prune_hip_run(
            pafs_dev, top_scores_dev, top_indices_dev, a_dev, b_dev, score_dev, valid_dev,
            batch, topk, limb_topm, in_h, in_w, full_h, full_w, points_per_limb,
            min_paf_score, success_ratio_thr, min_pair_score, paf_cubic_a, nullptr);
        hipEventRecord(e1, nullptr);
        hipEventSynchronize(e1);
        float device_ms = 0.0f;
        hipEventElapsedTime(&device_ms, e0, e1);
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        local_profile.device_total_ms = device_ms;
        local_profile.score_ms = device_ms;
        local_profile.prune_ms = 0.0f;
        if (status != HIP_PAF_PRUNE_SUCCESS) goto hip_error;
    }

    {
        auto t0 = std::chrono::steady_clock::now();
        err = hipMemcpy(limb_top_pair_a_idx_host, a_dev, out_count * sizeof(long long), hipMemcpyDeviceToHost);
        if (err != hipSuccess) goto hip_error;
        err = hipMemcpy(limb_top_pair_b_idx_host, b_dev, out_count * sizeof(long long), hipMemcpyDeviceToHost);
        if (err != hipSuccess) goto hip_error;
        err = hipMemcpy(limb_top_pair_score_host, score_dev, out_count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) goto hip_error;
        err = hipMemcpy(limb_top_pair_valid_host, valid_dev, out_count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) goto hip_error;
        auto t1 = std::chrono::steady_clock::now();
        local_profile.d2h_ms = elapsed_ms(t0, t1);
    }

    local_profile.total_ms = elapsed_ms(t_total0, std::chrono::steady_clock::now());
    if (profile) *profile = local_profile;
    free_if_needed(pafs_dev);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    free_if_needed(a_dev);
    free_if_needed(b_dev);
    free_if_needed(score_dev);
    free_if_needed(valid_dev);
    return HIP_PAF_PRUNE_SUCCESS;

hip_error:
    free_if_needed(pafs_dev);
    free_if_needed(top_scores_dev);
    free_if_needed(top_indices_dev);
    free_if_needed(a_dev);
    free_if_needed(b_dev);
    free_if_needed(score_dev);
    free_if_needed(valid_dev);
    return HIP_PAF_PRUNE_HIP_ERROR;
}

int paf_prune_hip_run_host(
    const float* pafs_host,
    const float* top_scores_host,
    const long long* top_indices_host,
    long long* limb_top_pair_a_idx_host,
    long long* limb_top_pair_b_idx_host,
    float* limb_top_pair_score_host,
    float* limb_top_pair_valid_host,
    int batch,
    int topk,
    int limb_topm,
    int in_h,
    int in_w,
    int full_h,
    int full_w,
    int points_per_limb,
    float min_paf_score,
    float success_ratio_thr,
    float min_pair_score,
    float paf_cubic_a) {
    return paf_prune_hip_run_host_profile(
        pafs_host,
        top_scores_host,
        top_indices_host,
        limb_top_pair_a_idx_host,
        limb_top_pair_b_idx_host,
        limb_top_pair_score_host,
        limb_top_pair_valid_host,
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
        nullptr);
}
