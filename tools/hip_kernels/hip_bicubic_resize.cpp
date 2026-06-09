#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(x) do { hipError_t err = (x); if (err != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    return -1; }} while(0)

static __device__ __forceinline__ float cubic_weight(float x) {
    // OpenCV INTER_CUBIC uses Keys cubic kernel with A=-0.75.
    const float A = -0.75f;
    float ax = fabsf(x);
    float ax2 = ax * ax;
    float ax3 = ax2 * ax;
    if (ax <= 1.0f) {
        return (A + 2.0f) * ax3 - (A + 3.0f) * ax2 + 1.0f;
    }
    if (ax < 2.0f) {
        return A * ax3 - 5.0f * A * ax2 + 8.0f * A * ax - 4.0f * A;
    }
    return 0.0f;
}

static __device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void bicubic_resize_hwc_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_h * out_w * channels;
    if (idx >= total) return;

    int c = idx % channels;
    int t = idx / channels;
    int ox = t % out_w;
    int oy = t / out_w;

    float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
    float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);

    // Half-pixel mapping, close to OpenCV resize coordinate convention.
    float fx = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
    float fy = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;

    int sx = static_cast<int>(floorf(fx));
    int sy = static_cast<int>(floorf(fy));

    float sum = 0.0f;
    float wsum = 0.0f;

    #pragma unroll
    for (int ky = -1; ky <= 2; ++ky) {
        int yy = clamp_int(sy + ky, 0, in_h - 1);
        float wy = cubic_weight(fy - static_cast<float>(sy + ky));
        #pragma unroll
        for (int kx = -1; kx <= 2; ++kx) {
            int xx = clamp_int(sx + kx, 0, in_w - 1);
            float wx = cubic_weight(fx - static_cast<float>(sx + kx));
            float w = wx * wy;
            sum += src[(yy * in_w + xx) * channels + c] * w;
            wsum += w;
        }
    }

    // wsum is normally 1.0, but keep the normalization for border robustness.
    dst[idx] = (wsum != 0.0f) ? (sum / wsum) : 0.0f;
}

extern "C" int hip_bicubic_resize_hwc_f32(
    const float* host_src,
    float* host_dst,
    int in_h, int in_w,
    int out_h, int out_w,
    int channels
) {
    if (!host_src || !host_dst) return -2;
    if (in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 || channels <= 0) return -3;

    size_t src_bytes = static_cast<size_t>(in_h) * in_w * channels * sizeof(float);
    size_t dst_bytes = static_cast<size_t>(out_h) * out_w * channels * sizeof(float);

    float* d_src = nullptr;
    float* d_dst = nullptr;

    HIP_CHECK(hipMalloc(&d_src, src_bytes));
    HIP_CHECK(hipMalloc(&d_dst, dst_bytes));
    HIP_CHECK(hipMemcpy(d_src, host_src, src_bytes, hipMemcpyHostToDevice));

    int total = out_h * out_w * channels;
    int block = 256;
    int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(
        bicubic_resize_hwc_kernel,
        dim3(grid), dim3(block), 0, 0,
        d_src, d_dst, in_h, in_w, out_h, out_w, channels
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host_dst, d_dst, dst_bytes, hipMemcpyDeviceToHost));

    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}
