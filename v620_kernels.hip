// Custom RDNA2 (gfx1030, wave32) Fused HIP Kernels for YuE2-3B + Oobleck VAE
// Zero external dependencies beyond HIP runtime. 100% HIP-Graph capturable.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor(val, 16, 32);
    val += __shfl_xor(val, 8, 32);
    val += __shfl_xor(val, 4, 32);
    val += __shfl_xor(val, 2, 32);
    val += __shfl_xor(val, 1, 32);
    return val;
}

static __device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor(val, 16, 32));
    val = fmaxf(val, __shfl_xor(val, 8, 32));
    val = fmaxf(val, __shfl_xor(val, 4, 32));
    val = fmaxf(val, __shfl_xor(val, 2, 32));
    val = fmaxf(val, __shfl_xor(val, 1, 32));
    return val;
}

// ===========================================================================
// 1. Token Embedding Lookup (single token from GPU int64* id)
// ===========================================================================
__global__ void embed_lookup_fp16_kernel(
    const __half* __restrict__ table,
    const int64_t* __restrict__ token_id_ptr,
    __half* __restrict__ out,
    int H
) {
    int64_t tid = *token_id_ptr;
    int idx = threadIdx.x;
    int H8 = H >> 3;
    const float4* src = reinterpret_cast<const float4*>(table + tid * H);
    float4* dst = reinterpret_cast<float4*>(out);
    for (int i = idx; i < H8; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ===========================================================================
// 2. Fused Batched / Single-Token RMSNorm (H must be multiple of 8, e.g. 2048)
// Grid: (M,), Block: (256,) = 8 warps
// ===========================================================================
__global__ void rmsnorm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    int H, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int H8 = H >> 3;
    const float4* x_row = reinterpret_cast<const float4*>(x + (size_t)row * H);
    const float4* w_vec = reinterpret_cast<const float4*>(w);
    float4* o_row = reinterpret_cast<float4*>(out + (size_t)row * H);

    float ss = 0.0f;
    for (int i = tid; i < H8; i += 256) {
        float4 xv = x_row[i];
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            ss = fmaf(xf.x, xf.x, ss);
            ss = fmaf(xf.y, xf.y, ss);
        }
    }
    ss = warp_reduce_sum(ss);
    __shared__ float smem[8];
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)H + eps);
    }
    __syncthreads();
    float inv_rms = smem[0];

    for (int i = tid; i < H8; i += 256) {
        float4 xv = x_row[i];
        float4 wv = w_vec[i];
        float4 ov;
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        __half2* oh = reinterpret_cast<__half2*>(&ov);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 wf = __half22float2(wh[j]);
            oh[j] = __floats2half2_rn(xf.x * inv_rms * wf.x, xf.y * inv_rms * wf.y);
        }
        o_row[i] = ov;
    }
}

// ===========================================================================
// 3. Vectorized GEMV Kernels (FP16, W8A16, W4A16) with optional residual add
// ===========================================================================
template<bool ADD_RESIDUAL>
__global__ void gemv_fp16_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const float4* w_row = reinterpret_cast<const float4*>(W + (size_t)row * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int K8 = K >> 3;

    float acc = 0.0f;
    for (int i = lane; i < K8; i += 32) {
        float4 wv = w_row[i];
        float4 xv = x_vec[i];
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __half22float2(wh[j]);
            float2 xf = __half22float2(xh[j]);
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        if constexpr (ADD_RESIDUAL) {
            y[row] = __float2half(__half2float(y[row]) + acc);
        } else {
            y[row] = __float2half(acc);
        }
    }
}

template<bool ADD_RESIDUAL>
__global__ void gemv_w8a16_kernel(
    const int8_t* __restrict__ W,
    const __half* __restrict__ scales,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    int K16 = K >> 4;

    float acc = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        uint4 wv = w_row[i];
        float4 xv0 = x_vec[i * 2];
        float4 xv1 = x_vec[i * 2 + 1];
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        const __half2* xh0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xh1 = reinterpret_cast<const __half2*>(&xv1);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh0[j]);
            acc = fmaf((float)wb[j * 2],     xf.x, acc);
            acc = fmaf((float)wb[j * 2 + 1], xf.y, acc);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh1[j]);
            acc = fmaf((float)wb[8 + j * 2],     xf.x, acc);
            acc = fmaf((float)wb[8 + j * 2 + 1], xf.y, acc);
        }
    }
    acc = warp_reduce_sum(acc) * __half2float(scales[row]);
    if (lane == 0) {
        if constexpr (ADD_RESIDUAL) {
            y[row] = __float2half(__half2float(y[row]) + acc);
        } else {
            y[row] = __float2half(acc);
        }
    }
}

template<bool ADD_RESIDUAL>
__global__ void gemv_w4a16_kernel(
    const uint32_t* __restrict__ W_q,
    const __half* __restrict__ scales,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    // 128-bit vectorized loads: 1 uint4 = 4 uint32 = 32 int4 weights per thread!
    // 32 threads * 32 weights = 1024 elements per warp iteration.
    int K32 = K >> 5;         // K / 32
    int num_groups = K >> 7;  // K / 128
    const uint4* w_row = reinterpret_cast<const uint4*>(W_q + (size_t)row * (K >> 3));
    const __half* s_row = scales + (size_t)row * num_groups;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc = 0.0f;
    for (int i = lane; i < K32; i += 32) {
        uint4 w4 = w_row[i];
        float sc = __half2float(s_row[i >> 2]);  // 4 uint4s = 128 elements = 1 group!
        uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
        float gacc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t packed = ws[k];
            float4 xv = x_vec[i * 4 + k];
            const __half2* xh = reinterpret_cast<const __half2*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int q0 = (int)(packed & 0xF) - 8;
                int q1 = (int)((packed >> 4) & 0xF) - 8;
                packed >>= 8;
                float2 xf = __half22float2(xh[j]);
                gacc = fmaf((float)q0, xf.x, gacc);
                gacc = fmaf((float)q1, xf.y, gacc);
            }
        }
        acc = fmaf(gacc, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        if constexpr (ADD_RESIDUAL) {
            y[row] = __float2half(__half2float(y[row]) + acc);
        } else {
            y[row] = __float2half(acc);
        }
    }
}

// ===========================================================================
// 4. Fused Gate+Up+SiLU*Up GEMV (FP16, W8A16, W4A16)
// ===========================================================================
__global__ void gemv_fp16_swiglu_kernel(
    const __half* __restrict__ Wg,
    const __half* __restrict__ Wu,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    int K8 = K >> 3;
    const float4* wg_row = reinterpret_cast<const float4*>(Wg + (size_t)row * K);
    const float4* wu_row = reinterpret_cast<const float4*>(Wu + (size_t)row * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int i = lane; i < K8; i += 32) {
        float4 wgv = wg_row[i];
        float4 wuv = wu_row[i];
        float4 xv = x_vec[i];
        const __half2* wgh = reinterpret_cast<const __half2*>(&wgv);
        const __half2* wuh = reinterpret_cast<const __half2*>(&wuv);
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 gf = __half22float2(wgh[j]);
            float2 uf = __half22float2(wuh[j]);
            acc_g = fmaf(gf.x, xf.x, acc_g);
            acc_g = fmaf(gf.y, xf.y, acc_g);
            acc_u = fmaf(uf.x, xf.x, acc_u);
            acc_u = fmaf(uf.y, xf.y, acc_u);
        }
    }
    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        float silu_g = acc_g / (1.0f + __expf(-acc_g));
        y[row] = __float2half(silu_g * acc_u);
    }
}

__global__ void gemv_w8a16_swiglu_kernel(
    const int8_t* __restrict__ Wg, const __half* __restrict__ sg,
    const int8_t* __restrict__ Wu, const __half* __restrict__ su,
    const __half* __restrict__ x, __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    int K16 = K >> 4;
    const uint4* wg_row = reinterpret_cast<const uint4*>(Wg + (size_t)row * K);
    const uint4* wu_row = reinterpret_cast<const uint4*>(Wu + (size_t)row * K);
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int i = lane; i < K16; i += 32) {
        uint4 wgv = wg_row[i];
        uint4 wuv = wu_row[i];
        float4 xv0 = x_vec[i * 2];
        float4 xv1 = x_vec[i * 2 + 1];
        const int8_t* bg = reinterpret_cast<const int8_t*>(&wgv);
        const int8_t* bu = reinterpret_cast<const int8_t*>(&wuv);
        const __half2* xh0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xh1 = reinterpret_cast<const __half2*>(&xv1);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh0[j]);
            acc_g = fmaf((float)bg[j * 2],     xf.x, acc_g);
            acc_g = fmaf((float)bg[j * 2 + 1], xf.y, acc_g);
            acc_u = fmaf((float)bu[j * 2],     xf.x, acc_u);
            acc_u = fmaf((float)bu[j * 2 + 1], xf.y, acc_u);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh1[j]);
            acc_g = fmaf((float)bg[8 + j * 2],     xf.x, acc_g);
            acc_g = fmaf((float)bg[8 + j * 2 + 1], xf.y, acc_g);
            acc_u = fmaf((float)bu[8 + j * 2],     xf.x, acc_u);
            acc_u = fmaf((float)bu[8 + j * 2 + 1], xf.y, acc_u);
        }
    }
    acc_g = warp_reduce_sum(acc_g) * __half2float(sg[row]);
    acc_u = warp_reduce_sum(acc_u) * __half2float(su[row]);
    if (lane == 0) {
        float silu_g = acc_g / (1.0f + __expf(-acc_g));
        y[row] = __float2half(silu_g * acc_u);
    }
}

__global__ void gemv_w4a16_swiglu_kernel(
    const uint32_t* __restrict__ Wg_q, const __half* __restrict__ sg,
    const uint32_t* __restrict__ Wu_q, const __half* __restrict__ su,
    const __half* __restrict__ x, __half* __restrict__ y,
    int N, int K
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp_id;
    if (row >= N) return;

    int K32 = K >> 5;
    int num_groups = K >> 7;
    const uint4* wg_row = reinterpret_cast<const uint4*>(Wg_q + (size_t)row * (K >> 3));
    const uint4* wu_row = reinterpret_cast<const uint4*>(Wu_q + (size_t)row * (K >> 3));
    const __half* sg_row = sg + (size_t)row * num_groups;
    const __half* su_row = su + (size_t)row * num_groups;
    const float4* x_vec = reinterpret_cast<const float4*>(x);

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int i = lane; i < K32; i += 32) {
        uint4 wg4 = wg_row[i];
        uint4 wu4 = wu_row[i];
        float scg = __half2float(sg_row[i >> 2]);
        float scu = __half2float(su_row[i >> 2]);
        uint32_t wgs[4] = {wg4.x, wg4.y, wg4.z, wg4.w};
        uint32_t wus[4] = {wu4.x, wu4.y, wu4.z, wu4.w};
        float gg = 0.0f, gu = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t pg = wgs[k];
            uint32_t pu = wus[k];
            float4 xv = x_vec[i * 4 + k];
            const __half2* xh = reinterpret_cast<const __half2*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __half22float2(xh[j]);
                int g0 = (int)(pg & 0xF) - 8;
                int g1 = (int)((pg >> 4) & 0xF) - 8;
                pg >>= 8;
                gg = fmaf((float)g0, xf.x, gg);
                gg = fmaf((float)g1, xf.y, gg);

                int u0 = (int)(pu & 0xF) - 8;
                int u1 = (int)((pu >> 4) & 0xF) - 8;
                pu >>= 8;
                gu = fmaf((float)u0, xf.x, gu);
                gu = fmaf((float)u1, xf.y, gu);
            }
        }
        acc_g = fmaf(gg, scg, acc_g);
        acc_u = fmaf(gu, scu, acc_u);
    }
    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        float silu_g = acc_g / (1.0f + __expf(-acc_g));
        y[row] = __float2half(silu_g * acc_u);
    }
}

// Fused RMSNorm + W4A16 GEMV (256 threads = 8 warps/block -> 1 float4/thread for RMSNorm, 8 rows/block!)
__global__ void gemv_w4a16_rmsnorm_kernel(
    const uint32_t* __restrict__ W_q,
    const __half* __restrict__ scales,
    const __half* __restrict__ x,
    const __half* __restrict__ ln_w,
    __half* __restrict__ y,
    int N, int K, float eps
) {
    extern __shared__ __half s_x[];
    int tid = threadIdx.x;       // 0..255 (8 warps)
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int K8 = K >> 3;             // 2048 / 8 = 256 -> EXACTLY 1 iteration per thread!
    const float4* x_g = reinterpret_cast<const float4*>(x);
    const float4* lw_g = reinterpret_cast<const float4*>(ln_w);
    float4* sx_v = reinterpret_cast<float4*>(s_x);

    float ss = 0.0f;
    for (int i = tid; i < K8; i += 256) {
        float4 xv = x_g[i];
        sx_v[i] = xv;
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            ss = fmaf(xf.x, xf.x, ss);
            ss = fmaf(xf.y, xf.y, ss);
        }
    }
    ss = warp_reduce_sum(ss);
    __shared__ float smem[8];
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)K + eps);
    }
    __syncthreads();
    float inv_rms = smem[0];

    for (int i = tid; i < K8; i += 256) {
        float4 xv = sx_v[i];
        float4 wv = lw_g[i];
        float4 ov;
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        __half2* oh = reinterpret_cast<__half2*>(&ov);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 wf = __half22float2(wh[j]);
            oh[j] = __floats2half2_rn(xf.x * inv_rms * wf.x, xf.y * inv_rms * wf.y);
        }
        sx_v[i] = ov;
    }
    __syncthreads();

    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    int K32 = K >> 5;
    int num_groups = K >> 7;
    const uint4* w_row = reinterpret_cast<const uint4*>(W_q + (size_t)row * (K >> 3));
    const __half* s_row = scales + (size_t)row * num_groups;

    float acc = 0.0f;
    for (int i = lane; i < K32; i += 32) {
        uint4 w4 = w_row[i];
        float sc = __half2float(s_row[i >> 2]);
        uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
        float gacc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t packed = ws[k];
            float4 xv = sx_v[i * 4 + k];
            const __half2* xh = reinterpret_cast<const __half2*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int q0 = (int)(packed & 0xF) - 8;
                int q1 = (int)((packed >> 4) & 0xF) - 8;
                packed >>= 8;
                float2 xf = __half22float2(xh[j]);
                gacc = fmaf((float)q0, xf.x, gacc);
                gacc = fmaf((float)q1, xf.y, gacc);
            }
        }
        acc = fmaf(gacc, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

// Fused RMSNorm + W4A16 Gate+Up+SiLU*Up GEMV (256 threads = 8 warps/block -> 8 rows/block!)
__global__ void gemv_w4a16_rmsnorm_swiglu_kernel(
    const uint32_t* __restrict__ Wg_q, const __half* __restrict__ sg,
    const uint32_t* __restrict__ Wu_q, const __half* __restrict__ su,
    const __half* __restrict__ x,
    const __half* __restrict__ ln_w,
    __half* __restrict__ y,
    int N, int K, float eps
) {
    extern __shared__ __half s_x[];
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int K8 = K >> 3;
    const float4* x_g = reinterpret_cast<const float4*>(x);
    const float4* lw_g = reinterpret_cast<const float4*>(ln_w);
    float4* sx_v = reinterpret_cast<float4*>(s_x);

    float ss = 0.0f;
    for (int i = tid; i < K8; i += 256) {
        float4 xv = x_g[i];
        sx_v[i] = xv;
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            ss = fmaf(xf.x, xf.x, ss);
            ss = fmaf(xf.y, xf.y, ss);
        }
    }
    ss = warp_reduce_sum(ss);
    __shared__ float smem[8];
    if (lane == 0) smem[warp_id] = ss;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < 8) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = rsqrtf(v / (float)K + eps);
    }
    __syncthreads();
    float inv_rms = smem[0];

    for (int i = tid; i < K8; i += 256) {
        float4 xv = sx_v[i];
        float4 wv = lw_g[i];
        float4 ov;
        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* wh = reinterpret_cast<const __half2*>(&wv);
        __half2* oh = reinterpret_cast<__half2*>(&ov);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 xf = __half22float2(xh[j]);
            float2 wf = __half22float2(wh[j]);
            oh[j] = __floats2half2_rn(xf.x * inv_rms * wf.x, xf.y * inv_rms * wf.y);
        }
        sx_v[i] = ov;
    }
    __syncthreads();

    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    int K32 = K >> 5;
    int num_groups = K >> 7;
    const uint4* wg_row = reinterpret_cast<const uint4*>(Wg_q + (size_t)row * (K >> 3));
    const uint4* wu_row = reinterpret_cast<const uint4*>(Wu_q + (size_t)row * (K >> 3));
    const __half* sg_row = sg + (size_t)row * num_groups;
    const __half* su_row = su + (size_t)row * num_groups;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int i = lane; i < K32; i += 32) {
        uint4 wg4 = wg_row[i];
        uint4 wu4 = wu_row[i];
        float scg = __half2float(sg_row[i >> 2]);
        float scu = __half2float(su_row[i >> 2]);
        uint32_t wgs[4] = {wg4.x, wg4.y, wg4.z, wg4.w};
        uint32_t wus[4] = {wu4.x, wu4.y, wu4.z, wu4.w};
        float gg = 0.0f, gu = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t pg = wgs[k];
            uint32_t pu = wus[k];
            float4 xv = sx_v[i * 4 + k];
            const __half2* xh = reinterpret_cast<const __half2*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __half22float2(xh[j]);
                int g0 = (int)(pg & 0xF) - 8;
                int g1 = (int)((pg >> 4) & 0xF) - 8;
                pg >>= 8;
                gg = fmaf((float)g0, xf.x, gg);
                gg = fmaf((float)g1, xf.y, gg);

                int u0 = (int)(pu & 0xF) - 8;
                int u1 = (int)((pu >> 4) & 0xF) - 8;
                pu >>= 8;
                gu = fmaf((float)u0, xf.x, gu);
                gu = fmaf((float)u1, xf.y, gu);
            }
        }
        acc_g = fmaf(gg, scg, acc_g);
        acc_u = fmaf(gu, scu, acc_u);
    }
    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);
    if (lane == 0) {
        float silu_g = acc_g / (1.0f + __expf(-acc_g));
        y[row] = __float2half(silu_g * acc_u);
    }
}

// ===========================================================================
// 5. Single-Token Fused QK-RMSNorm + RoPE + KV Cache Write (100% HIP-Graph!)
// qkv: [Hq*128 + Hkv*128 + Hkv*128] = [4096]
// k_cache, v_cache: [Hkv, max_seq, 128]
// Grid: (Hq + Hkv + Hkv,) = (32,) blocks of 64 threads (2 warps = 64 pairs for D=128)
// ===========================================================================
__global__ void qkv_norm_rope_kvcache_kernel(
    const __half* __restrict__ qkv,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const float* __restrict__ inv_freq,  // [64]
    const int32_t* __restrict__ pos_ptr,
    __half* __restrict__ q_out,          // [Hq, 128]
    __half* __restrict__ k_cache,        // [Hkv, max_seq, 128]
    __half* __restrict__ v_cache,        // [Hkv, max_seq, 128]
    int Hq, int Hkv, int max_seq, float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;  // 0..63 (handles pair (tid, tid+64))
    int pos = *pos_ptr;

    if (bid >= Hq + Hkv) {
        // V copy to v_cache[kv_h, pos, :]
        int kv_h = bid - (Hq + Hkv);
        const __half* v_src = qkv + (Hq + Hkv + kv_h) * 128;
        __half* v_dst = v_cache + ((size_t)kv_h * max_seq + pos) * 128;
        v_dst[tid] = v_src[tid];
        v_dst[tid + 64] = v_src[tid + 64];
        return;
    }

    bool is_q = (bid < Hq);
    const __half* src = qkv + bid * 128;
    const __half* nw = is_q ? q_norm_w : k_norm_w;

    float x1 = __half2float(src[tid]);
    float x2 = __half2float(src[tid + 64]);
    float ss = x1 * x1 + x2 * x2;

    // Reduce across 64 threads (2 warps)
    ss = warp_reduce_sum(ss);
    __shared__ float smem[2];
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    float inv_rms = rsqrtf((smem[0] + smem[1]) * (1.0f / 128.0f) + eps);

    float n1 = x1 * inv_rms * __half2float(nw[tid]);
    float n2 = x2 * inv_rms * __half2float(nw[tid + 64]);

    float angle = (float)pos * inv_freq[tid];
    float c = cosf(angle);
    float s = sinf(angle);
    float r1 = n1 * c - n2 * s;
    float r2 = n2 * c + n1 * s;

    __half* dst = is_q ? (q_out + bid * 128)
                       : (k_cache + ((size_t)(bid - Hq) * max_seq + pos) * 128);
    dst[tid] = __float2half(r1);
    dst[tid + 64] = __float2half(r2);
}

// ===========================================================================
// 6. GQA-Paired Split-KV FlashDecoding (reads K/V ONCE for BOTH Q heads in GQA group!)
// Stage 1: Grid (DECODE_SPLITS, Hkv), Block (128) = 4 warps
//   Cuts KV cache HBM bandwidth in HALF for GQA (Hq=16, Hkv=8) with 100% exact math!
// ===========================================================================
#define DECODE_SPLITS 16

__global__ void flash_decode_gqa_stage1_kernel(
    const __half* __restrict__ Q,        // [Hq, 128]
    const __half* __restrict__ K_cache,  // [Hkv, max_seq, 128]
    const __half* __restrict__ V_cache,  // [Hkv, max_seq, 128]
    const int32_t* __restrict__ pos_ptr,
    float* __restrict__ mid_O,           // [Hq, DECODE_SPLITS, 128]
    float* __restrict__ mid_M,           // [Hq, DECODE_SPLITS]
    float* __restrict__ mid_L,           // [Hq, DECODE_SPLITS]
    int Hq, int Hkv, int max_seq, float sm_scale
) {
    int split_id = blockIdx.x;
    int kvh = blockIdx.y;
    int qh0 = kvh * 2;
    int qh1 = qh0 + 1;
    int seq_len = (*pos_ptr) + 1;

    int chunk = (seq_len + DECODE_SPLITS - 1) / DECODE_SPLITS;
    int start = split_id * chunk;
    int end = min(start + chunk, seq_len);

    int tid = threadIdx.x;  // 0..127 -> 4 warps of 32
    int wid = tid >> 5;
    int lane = tid & 31;

    if (start >= end) {
        if (tid == 0) {
            mid_M[qh0 * DECODE_SPLITS + split_id] = -1e30f;
            mid_L[qh0 * DECODE_SPLITS + split_id] = 0.0f;
            mid_M[qh1 * DECODE_SPLITS + split_id] = -1e30f;
            mid_L[qh1 * DECODE_SPLITS + split_id] = 0.0f;
        }
        mid_O[(qh0 * DECODE_SPLITS + split_id) * 128 + tid] = 0.0f;
        mid_O[(qh1 * DECODE_SPLITS + split_id) * 128 + tid] = 0.0f;
        return;
    }

    // Load BOTH Q heads for this KV head into registers (64-bit coalesced load!)
    const uint2* q0_u2 = reinterpret_cast<const uint2*>(Q + qh0 * 128);
    const uint2* q1_u2 = reinterpret_cast<const uint2*>(Q + qh1 * 128);
    uint2 q0_raw = q0_u2[lane];
    uint2 q1_raw = q1_u2[lane];
    const __half2* q0_h2 = reinterpret_cast<const __half2*>(&q0_raw);
    const __half2* q1_h2 = reinterpret_cast<const __half2*>(&q1_raw);

    float2 qa0 = __half22float2(q0_h2[0]); float2 qa1 = __half22float2(q0_h2[1]);
    qa0.x *= sm_scale; qa0.y *= sm_scale; qa1.x *= sm_scale; qa1.y *= sm_scale;
    float2 qb0 = __half22float2(q1_h2[0]); float2 qb1 = __half22float2(q1_h2[1]);
    qb0.x *= sm_scale; qb0.y *= sm_scale; qb1.x *= sm_scale; qb1.y *= sm_scale;

    const __half* k_head = K_cache + (size_t)kvh * max_seq * 128;
    const __half* v_head = V_cache + (size_t)kvh * max_seq * 128;

    float m0 = -1e30f, l0 = 0.0f;
    float o0_0x = 0.0f, o0_0y = 0.0f, o0_1x = 0.0f, o0_1y = 0.0f;
    float m1 = -1e30f, l1 = 0.0f;
    float o1_0x = 0.0f, o1_0y = 0.0f, o1_1x = 0.0f, o1_1y = 0.0f;

    for (int t = start + wid; t < end; t += 4) {
        // Single 64-bit load of K (4 fp16) shared by BOTH Q heads!
        uint2 k_raw = reinterpret_cast<const uint2*>(k_head + (size_t)t * 128)[lane];
        const __half2* k_h2 = reinterpret_cast<const __half2*>(&k_raw);
        float2 k0 = __half22float2(k_h2[0]);
        float2 k1 = __half22float2(k_h2[1]);

        float dot0 = warp_reduce_sum(qa0.x * k0.x + qa0.y * k0.y + qa1.x * k1.x + qa1.y * k1.y);
        float dot1 = warp_reduce_sum(qb0.x * k0.x + qb0.y * k0.y + qb1.x * k1.x + qb1.y * k1.y);

        float m0_new = fmaxf(m0, dot0);
        float a0 = __expf(m0 - m0_new);
        float p0 = __expf(dot0 - m0_new);

        float m1_new = fmaxf(m1, dot1);
        float a1 = __expf(m1 - m1_new);
        float p1 = __expf(dot1 - m1_new);

        // Single 64-bit load of V (4 fp16) shared by BOTH Q heads!
        uint2 v_raw = reinterpret_cast<const uint2*>(v_head + (size_t)t * 128)[lane];
        const __half2* v_h2 = reinterpret_cast<const __half2*>(&v_raw);
        float2 v0 = __half22float2(v_h2[0]);
        float2 v1 = __half22float2(v_h2[1]);

        o0_0x = fmaf(o0_0x, a0, p0 * v0.x); o0_0y = fmaf(o0_0y, a0, p0 * v0.y);
        o0_1x = fmaf(o0_1x, a0, p0 * v1.x); o0_1y = fmaf(o0_1y, a0, p0 * v1.y);
        l0 = fmaf(l0, a0, p0); m0 = m0_new;

        o1_0x = fmaf(o1_0x, a1, p1 * v0.x); o1_0y = fmaf(o1_0y, a1, p1 * v0.y);
        o1_1x = fmaf(o1_1x, a1, p1 * v1.x); o1_1y = fmaf(o1_1y, a1, p1 * v1.y);
        l1 = fmaf(l1, a1, p1); m1 = m1_new;
    }

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_o[4][128];

    // Merge Q head 0
    if (lane == 0) { s_m[wid] = m0; s_l[wid] = l0; }
    s_o[wid][lane * 4 + 0] = o0_0x; s_o[wid][lane * 4 + 1] = o0_0y;
    s_o[wid][lane * 4 + 2] = o0_1x; s_o[wid][lane * 4 + 3] = o0_1y;
    __syncthreads();
    {
        float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float w0 = (s_l[0] > 0.0f) ? __expf(s_m[0] - gm) : 0.0f;
        float w1 = (s_l[1] > 0.0f) ? __expf(s_m[1] - gm) : 0.0f;
        float w2 = (s_l[2] > 0.0f) ? __expf(s_m[2] - gm) : 0.0f;
        float w3 = (s_l[3] > 0.0f) ? __expf(s_m[3] - gm) : 0.0f;
        mid_O[(qh0 * DECODE_SPLITS + split_id) * 128 + tid] =
            w0 * s_o[0][tid] + w1 * s_o[1][tid] + w2 * s_o[2][tid] + w3 * s_o[3][tid];
        if (tid == 0) {
            mid_M[qh0 * DECODE_SPLITS + split_id] = gm;
            mid_L[qh0 * DECODE_SPLITS + split_id] = w0 * s_l[0] + w1 * s_l[1] + w2 * s_l[2] + w3 * s_l[3];
        }
    }
    __syncthreads();

    // Merge Q head 1
    if (lane == 0) { s_m[wid] = m1; s_l[wid] = l1; }
    s_o[wid][lane * 4 + 0] = o1_0x; s_o[wid][lane * 4 + 1] = o1_0y;
    s_o[wid][lane * 4 + 2] = o1_1x; s_o[wid][lane * 4 + 3] = o1_1y;
    __syncthreads();
    {
        float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
        float w0 = (s_l[0] > 0.0f) ? __expf(s_m[0] - gm) : 0.0f;
        float w1 = (s_l[1] > 0.0f) ? __expf(s_m[1] - gm) : 0.0f;
        float w2 = (s_l[2] > 0.0f) ? __expf(s_m[2] - gm) : 0.0f;
        float w3 = (s_l[3] > 0.0f) ? __expf(s_m[3] - gm) : 0.0f;
        mid_O[(qh1 * DECODE_SPLITS + split_id) * 128 + tid] =
            w0 * s_o[0][tid] + w1 * s_o[1][tid] + w2 * s_o[2][tid] + w3 * s_o[3][tid];
        if (tid == 0) {
            mid_M[qh1 * DECODE_SPLITS + split_id] = gm;
            mid_L[qh1 * DECODE_SPLITS + split_id] = w0 * s_l[0] + w1 * s_l[1] + w2 * s_l[2] + w3 * s_l[3];
        }
    }
}

__global__ void flash_decode_gqa_stage2_kernel(
    const float* __restrict__ mid_O,
    const float* __restrict__ mid_M,
    const float* __restrict__ mid_L,
    __half* __restrict__ out  // [Hq, 128]
) {
    int qh = blockIdx.x;
    int d = threadIdx.x;  // 0..127

    float gm = -1e30f;
    #pragma unroll
    for (int s = 0; s < DECODE_SPLITS; ++s) {
        gm = fmaxf(gm, mid_M[qh * DECODE_SPLITS + s]);
    }
    float gl = 0.0f;
    float go = 0.0f;
    #pragma unroll
    for (int s = 0; s < DECODE_SPLITS; ++s) {
        float l = mid_L[qh * DECODE_SPLITS + s];
        if (l > 0.0f) {
            float w = __expf(mid_M[qh * DECODE_SPLITS + s] - gm);
            gl += w * l;
            go += w * mid_O[(qh * DECODE_SPLITS + s) * 128 + d];
        }
    }
    out[qh * 128 + d] = __float2half(go / gl);
}

// ===========================================================================
// 6b. Fused FlashDecode Stage2 + W4A16 O-Proj + Residual Add!
// Eliminates 28 separate stage2 kernel launches + 28 global d_attn round-trips!
// Grid: ((H + 7) / 8), Block: (256) = 8 warps
// ===========================================================================
__global__ void gemv_w4a16_o_from_splits_kernel(
    const uint32_t* __restrict__ W_q,
    const __half* __restrict__ scales,
    const float* __restrict__ mid_O,     // [Hq, DECODE_SPLITS, 128]
    const float* __restrict__ mid_M,     // [Hq, DECODE_SPLITS]
    const float* __restrict__ mid_L,     // [Hq, DECODE_SPLITS]
    __half* __restrict__ y,              // [H] (adds residual in-place!)
    int N, int K, int Hq
) {
    // Merge all Hq=16 heads x 128 dims = 2048 elements into LDS (256 threads x 8 elems each!)
    extern __shared__ __half s_attn[];
    int tid = threadIdx.x;  // 0..255
    int warp_id = tid >> 5;
    int lane = tid & 31;

    // 256 threads cover 2048 dims in 8 iterations (each thread handles 8 (qh, d) pairs)
    for (int idx = tid; idx < K; idx += 256) {
        int qh = idx >> 7;
        int d = idx & 127;
        float gm = -1e30f;
        #pragma unroll
        for (int s = 0; s < DECODE_SPLITS; ++s) {
            gm = fmaxf(gm, mid_M[qh * DECODE_SPLITS + s]);
        }
        float gl = 0.0f;
        float go = 0.0f;
        #pragma unroll
        for (int s = 0; s < DECODE_SPLITS; ++s) {
            float l = mid_L[qh * DECODE_SPLITS + s];
            if (l > 0.0f) {
                float w = __expf(mid_M[qh * DECODE_SPLITS + s] - gm);
                gl += w * l;
                go += w * mid_O[(qh * DECODE_SPLITS + s) * 128 + d];
            }
        }
        s_attn[idx] = __float2half(go / gl);
    }
    __syncthreads();

    int row = blockIdx.x * 8 + warp_id;
    if (row >= N) return;

    int K32 = K >> 5;
    int num_groups = K >> 7;
    const uint4* w_row = reinterpret_cast<const uint4*>(W_q + (size_t)row * (K >> 3));
    const __half* s_row = scales + (size_t)row * num_groups;
    const float4* sx_v = reinterpret_cast<const float4*>(s_attn);

    float acc = 0.0f;
    for (int i = lane; i < K32; i += 32) {
        uint4 w4 = w_row[i];
        float sc = __half2float(s_row[i >> 2]);
        uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
        float gacc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t packed = ws[k];
            float4 xv = sx_v[i * 4 + k];
            const __half2* xh = reinterpret_cast<const __half2*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int q0 = (int)(packed & 0xF) - 8;
                int q1 = (int)((packed >> 4) & 0xF) - 8;
                packed >>= 8;
                float2 xf = __half22float2(xh[j]);
                gacc = fmaf((float)q0, xf.x, gacc);
                gacc = fmaf((float)q1, xf.y, gacc);
            }
        }
        acc = fmaf(gacc, sc, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        y[row] = __float2half(__half2float(y[row]) + acc);
    }
}

__global__ void inc_pos_kernel(int32_t* pos_ptr) {
    if (threadIdx.x == 0) (*pos_ptr) += 1;
}

// ===========================================================================
// 7. NAR Batched Fused QK-RMSNorm + RoPE + Transpose to [H, M, 128]
// Input: qkv [M, (Hq + 2*Hkv)*128]
// Output: Q_out [Hq, M, 128], K_out [Hkv, M, 128], V_out [Hkv, M, 128]
// Grid: (Hq + 2*Hkv, M), Block: (64,)
// ===========================================================================
__global__ void nar_qkv_norm_rope_transpose_kernel(
    const __half* __restrict__ qkv,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const float* __restrict__ cos_tab,   // [M, 64]
    const float* __restrict__ sin_tab,   // [M, 64]
    __half* __restrict__ Q_out,          // [Hq, M, 128]
    __half* __restrict__ K_out,          // [Hkv, M, 128]
    __half* __restrict__ V_out,          // [Hkv, M, 128]
    int M, int Hq, int Hkv, float eps
) {
    int head_idx = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;  // 0..63
    int total_dim = (Hq + 2 * Hkv) * 128;
    const __half* src = qkv + (size_t)m * total_dim + head_idx * 128;

    if (head_idx >= Hq + Hkv) {
        int kv_h = head_idx - (Hq + Hkv);
        __half* dst = V_out + ((size_t)kv_h * M + m) * 128;
        dst[tid] = src[tid];
        dst[tid + 64] = src[tid + 64];
        return;
    }

    bool is_q = (head_idx < Hq);
    const __half* nw = is_q ? q_norm_w : k_norm_w;

    float x1 = __half2float(src[tid]);
    float x2 = __half2float(src[tid + 64]);
    float ss = x1 * x1 + x2 * x2;
    ss = warp_reduce_sum(ss);
    __shared__ float smem[2];
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) smem[wid] = ss;
    __syncthreads();
    float inv_rms = rsqrtf((smem[0] + smem[1]) * (1.0f / 128.0f) + eps);

    float n1 = x1 * inv_rms * __half2float(nw[tid]);
    float n2 = x2 * inv_rms * __half2float(nw[tid + 64]);

    float c = cos_tab[(size_t)m * 64 + tid];
    float s = sin_tab[(size_t)m * 64 + tid];
    float r1 = n1 * c - n2 * s;
    float r2 = n2 * c + n1 * s;

    __half* dst = is_q ? (Q_out + ((size_t)head_idx * M + m) * 128)
                       : (K_out + ((size_t)(head_idx - Hq) * M + m) * 128);
    dst[tid] = __float2half(r1);
    dst[tid + 64] = __float2half(r2);
}

// ===========================================================================
// 8. NAR Batched Fused SwiGLU: input [M, 2*I] -> output [M, I]
// ===========================================================================
__global__ void swiglu_batched_fp16_kernel(
    const __half* __restrict__ gate_up,  // [M, 2*I]
    __half* __restrict__ out,            // [M, I]
    int M, int I
) {
    int m = blockIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (col >= I) return;

    const __half2* g_ptr = reinterpret_cast<const __half2*>(gate_up + (size_t)m * (2 * I) + col);
    const __half2* u_ptr = reinterpret_cast<const __half2*>(gate_up + (size_t)m * (2 * I) + I + col);
    __half2* o_ptr = reinterpret_cast<__half2*>(out + (size_t)m * I + col);

    float2 g = __half22float2(*g_ptr);
    float2 u = __half22float2(*u_ptr);
    float r0 = (g.x / (1.0f + __expf(-g.x))) * u.x;
    float r1 = (g.y / (1.0f + __expf(-g.y))) * u.y;
    *o_ptr = __floats2half2_rn(r0, r1);
}

// ===========================================================================
// 9. Oobleck VAE Fused SnakeBeta (FP32):
// y[c, t] = x[c, t] + inv_beta[c] * sin(x[c, t] * exp_alpha[c])^2
// ===========================================================================
__global__ void snake_beta_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ exp_alpha,
    const float* __restrict__ inv_beta,
    float* __restrict__ y,
    int C, int T
) {
    int c = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float ea = exp_alpha[c];
    float ib = inv_beta[c];
    size_t idx = (size_t)c * T + t;
    float xv = x[idx];
    float s = __sinf(xv * ea);
    y[idx] = fmaf(s * s, ib, xv);
}

// ===========================================================================
// C API
// ===========================================================================
extern "C" {

void launch_embed_lookup_fp16(const void* table, const void* token_id_ptr, void* out, int H, hipStream_t stream) {
    hipLaunchKernelGGL(embed_lookup_fp16_kernel, dim3(1), dim3(256), 0, stream,
        (const __half*)table, (const int64_t*)token_id_ptr, (__half*)out, H);
}

void launch_rmsnorm_fp16(const void* x, const void* w, void* out, int M, int H, float eps, hipStream_t stream) {
    hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(M), dim3(256), 0, stream,
        (const __half*)x, (const __half*)w, (__half*)out, H, eps);
}

void launch_gemv_fp16(const void* W, const void* x, void* y, int N, int K, int add_res, hipStream_t stream) {
    dim3 block(128), grid((N + 3) / 4);
    if (add_res)
        hipLaunchKernelGGL(gemv_fp16_kernel<true>, grid, block, 0, stream, (const __half*)W, (const __half*)x, (__half*)y, N, K);
    else
        hipLaunchKernelGGL(gemv_fp16_kernel<false>, grid, block, 0, stream, (const __half*)W, (const __half*)x, (__half*)y, N, K);
}

void launch_gemv_w8a16(const void* W, const void* scales, const void* x, void* y, int N, int K, int add_res, hipStream_t stream) {
    dim3 block(128), grid((N + 3) / 4);
    if (add_res)
        hipLaunchKernelGGL(gemv_w8a16_kernel<true>, grid, block, 0, stream, (const int8_t*)W, (const __half*)scales, (const __half*)x, (__half*)y, N, K);
    else
        hipLaunchKernelGGL(gemv_w8a16_kernel<false>, grid, block, 0, stream, (const int8_t*)W, (const __half*)scales, (const __half*)x, (__half*)y, N, K);
}

void launch_gemv_w4a16(const void* W_q, const void* scales, const void* x, void* y, int N, int K, int add_res, hipStream_t stream) {
    dim3 block(128), grid((N + 3) / 4);
    if (add_res)
        hipLaunchKernelGGL(gemv_w4a16_kernel<true>, grid, block, 0, stream, (const uint32_t*)W_q, (const __half*)scales, (const __half*)x, (__half*)y, N, K);
    else
        hipLaunchKernelGGL(gemv_w4a16_kernel<false>, grid, block, 0, stream, (const uint32_t*)W_q, (const __half*)scales, (const __half*)x, (__half*)y, N, K);
}

void launch_gemv_fp16_swiglu(const void* Wg, const void* Wu, const void* x, void* y, int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(gemv_fp16_swiglu_kernel, dim3((N + 3) / 4), dim3(128), 0, stream,
        (const __half*)Wg, (const __half*)Wu, (const __half*)x, (__half*)y, N, K);
}

void launch_gemv_w8a16_swiglu(const void* Wg, const void* sg, const void* Wu, const void* su, const void* x, void* y, int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(gemv_w8a16_swiglu_kernel, dim3((N + 3) / 4), dim3(128), 0, stream,
        (const int8_t*)Wg, (const __half*)sg, (const int8_t*)Wu, (const __half*)su, (const __half*)x, (__half*)y, N, K);
}

void launch_gemv_w4a16_swiglu(const void* Wg_q, const void* sg, const void* Wu_q, const void* su, const void* x, void* y, int N, int K, hipStream_t stream) {
    hipLaunchKernelGGL(gemv_w4a16_swiglu_kernel, dim3((N + 3) / 4), dim3(128), 0, stream,
        (const uint32_t*)Wg_q, (const __half*)sg, (const uint32_t*)Wu_q, (const __half*)su, (const __half*)x, (__half*)y, N, K);
}

void launch_qkv_norm_rope_kvcache(
    const void* qkv, const void* q_norm_w, const void* k_norm_w,
    const void* inv_freq, const void* pos_ptr,
    void* q_out, void* k_cache, void* v_cache,
    int Hq, int Hkv, int max_seq, float eps, hipStream_t stream
) {
    hipLaunchKernelGGL(qkv_norm_rope_kvcache_kernel, dim3(Hq + 2 * Hkv), dim3(64), 0, stream,
        (const __half*)qkv, (const __half*)q_norm_w, (const __half*)k_norm_w,
        (const float*)inv_freq, (const int32_t*)pos_ptr,
        (__half*)q_out, (__half*)k_cache, (__half*)v_cache, Hq, Hkv, max_seq, eps);
}

void launch_flash_decode_gqa(
    const void* Q, const void* K_cache, const void* V_cache,
    const void* pos_ptr, void* mid_O, void* mid_M, void* mid_L, void* out,
    int Hq, int Hkv, int max_seq, float sm_scale, hipStream_t stream
) {
    hipLaunchKernelGGL(flash_decode_gqa_stage1_kernel, dim3(DECODE_SPLITS, Hkv), dim3(128), 0, stream,
        (const __half*)Q, (const __half*)K_cache, (const __half*)V_cache,
        (const int32_t*)pos_ptr, (float*)mid_O, (float*)mid_M, (float*)mid_L,
        Hq, Hkv, max_seq, sm_scale);
    hipLaunchKernelGGL(flash_decode_gqa_stage2_kernel, dim3(Hq), dim3(128), 0, stream,
        (const float*)mid_O, (const float*)mid_M, (const float*)mid_L, (__half*)out);
}

void launch_inc_pos(void* pos_ptr, hipStream_t stream) {
    hipLaunchKernelGGL(inc_pos_kernel, dim3(1), dim3(1), 0, stream, (int32_t*)pos_ptr);
}

void launch_nar_qkv_norm_rope_transpose(
    const void* qkv, const void* q_norm_w, const void* k_norm_w,
    const void* cos_tab, const void* sin_tab,
    void* Q_out, void* K_out, void* V_out,
    int M, int Hq, int Hkv, float eps, hipStream_t stream
) {
    hipLaunchKernelGGL(nar_qkv_norm_rope_transpose_kernel, dim3(Hq + 2 * Hkv, M), dim3(64), 0, stream,
        (const __half*)qkv, (const __half*)q_norm_w, (const __half*)k_norm_w,
        (const float*)cos_tab, (const float*)sin_tab,
        (__half*)Q_out, (__half*)K_out, (__half*)V_out, M, Hq, Hkv, eps);
}

void launch_swiglu_batched_fp16(const void* gate_up, void* out, int M, int I, hipStream_t stream) {
    int half_I = I / 2;
    dim3 block(256);
    dim3 grid((half_I + 255) / 256, M);
    hipLaunchKernelGGL(swiglu_batched_fp16_kernel, grid, block, 0, stream,
        (const __half*)gate_up, (__half*)out, M, I);
}

void launch_snake_beta_fp32(const void* x, const void* exp_alpha, const void* inv_beta, void* y, int C, int T, hipStream_t stream) {
    dim3 block(256);
    dim3 grid((T + 255) / 256, C);
    hipLaunchKernelGGL(snake_beta_fp32_kernel, grid, block, 0, stream,
        (const float*)x, (const float*)exp_alpha, (const float*)inv_beta, (float*)y, C, T);
}

// ===========================================================================
// 10. Single-Call Full 28-Layer AR Decode Step (Zero Python Overhead!)
// quant_mode: 0 = fp16, 1 = w8a16, 2 = w4a16
// ===========================================================================
struct ARLayerPtrs {
    const void* in_ln;
    const void* post_ln;
    const void* q_nw;
    const void* k_nw;
    const void* qkv_w;
    const void* qkv_s;
    const void* o_w;
    const void* o_s;
    const void* g_w;
    const void* g_s;
    const void* u_w;
    const void* u_s;
    const void* d_w;
    const void* d_s;
    void* k_cache;
    void* v_cache;
};

void launch_ar_decode_chunk(
    const ARLayerPtrs* layers,
    int start_layer,
    int end_layer,
    int quant_mode,
    int do_embed,
    int do_lm_head,
    const void* embed_tokens,
    const void* final_norm_w,
    const void* lm_w,
    const void* lm_s,
    int lm_vocab,
    int lm_is_w4,
    const void* inv_freq,
    const void* token_id_ptr,
    void* pos_ptr,
    void* d_x,
    void* d_norm,
    void* d_qkv,
    void* d_q,
    void* d_attn,
    void* d_mlp,
    void* d_mid_O,
    void* d_mid_M,
    void* d_mid_L,
    void* d_logits,
    int H, int I, int Hq, int Hkv, int max_seq, float eps, float sm_scale,
    hipStream_t stream
) {
    int qkv_dim = (Hq + 2 * Hkv) * 128;
    dim3 b128(128), b256(256), b64(64);
    dim3 g_qkv((qkv_dim + 3) / 4);
    dim3 g_H((H + 3) / 4);
    dim3 g_I((I + 3) / 4);
    dim3 g_lm((lm_vocab + 3) / 4);

    if (do_embed) {
        hipLaunchKernelGGL(embed_lookup_fp16_kernel, dim3(1), b256, 0, stream,
            (const __half*)embed_tokens, (const int64_t*)token_id_ptr, (__half*)d_x, H);
    }

    for (int i = start_layer; i < end_layer; ++i) {
        const ARLayerPtrs& L = layers[i];
        hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
            (const __half*)d_x, (const __half*)L.in_ln, (__half*)d_norm, H, eps);

        if (quant_mode == 2) {
            hipLaunchKernelGGL(gemv_w4a16_kernel<false>, g_qkv, b128, 0, stream,
                (const uint32_t*)L.qkv_w, (const __half*)L.qkv_s, (const __half*)d_norm, (__half*)d_qkv, qkv_dim, H);
        } else if (quant_mode == 1) {
            hipLaunchKernelGGL(gemv_w8a16_kernel<false>, g_qkv, b128, 0, stream,
                (const int8_t*)L.qkv_w, (const __half*)L.qkv_s, (const __half*)d_norm, (__half*)d_qkv, qkv_dim, H);
        } else {
            hipLaunchKernelGGL(gemv_fp16_kernel<false>, g_qkv, b128, 0, stream,
                (const __half*)L.qkv_w, (const __half*)d_norm, (__half*)d_qkv, qkv_dim, H);
        }

        hipLaunchKernelGGL(qkv_norm_rope_kvcache_kernel, dim3(Hq + 2 * Hkv), b64, 0, stream,
            (const __half*)d_qkv, (const __half*)L.q_nw, (const __half*)L.k_nw,
            (const float*)inv_freq, (const int32_t*)pos_ptr,
            (__half*)d_q, (__half*)L.k_cache, (__half*)L.v_cache, Hq, Hkv, max_seq, eps);

        hipLaunchKernelGGL(flash_decode_gqa_stage1_kernel, dim3(DECODE_SPLITS, Hkv), b128, 0, stream,
            (const __half*)d_q, (const __half*)L.k_cache, (const __half*)L.v_cache,
            (const int32_t*)pos_ptr, (float*)d_mid_O, (float*)d_mid_M, (float*)d_mid_L,
            Hq, Hkv, max_seq, sm_scale);
        hipLaunchKernelGGL(flash_decode_gqa_stage2_kernel, dim3(Hq), b128, 0, stream,
            (const float*)d_mid_O, (const float*)d_mid_M, (const float*)d_mid_L, (__half*)d_attn);

        if (quant_mode == 2) {
            hipLaunchKernelGGL(gemv_w4a16_kernel<true>, g_H, b128, 0, stream,
                (const uint32_t*)L.o_w, (const __half*)L.o_s, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_w4a16_swiglu_kernel, g_I, b128, 0, stream,
                (const uint32_t*)L.g_w, (const __half*)L.g_s, (const uint32_t*)L.u_w, (const __half*)L.u_s,
                (const __half*)d_norm, (__half*)d_mlp, I, H);
            hipLaunchKernelGGL(gemv_w4a16_kernel<true>, g_H, b128, 0, stream,
                (const uint32_t*)L.d_w, (const __half*)L.d_s, (const __half*)d_mlp, (__half*)d_x, H, I);
        } else if (quant_mode == 1) {
            hipLaunchKernelGGL(gemv_w8a16_kernel<true>, g_H, b128, 0, stream,
                (const int8_t*)L.o_w, (const __half*)L.o_s, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_w8a16_swiglu_kernel, g_I, b128, 0, stream,
                (const int8_t*)L.g_w, (const __half*)L.g_s, (const int8_t*)L.u_w, (const __half*)L.u_s,
                (const __half*)d_norm, (__half*)d_mlp, I, H);
            hipLaunchKernelGGL(gemv_w8a16_kernel<true>, g_H, b128, 0, stream,
                (const int8_t*)L.d_w, (const __half*)L.d_s, (const __half*)d_mlp, (__half*)d_x, H, I);
        } else {
            hipLaunchKernelGGL(gemv_fp16_kernel<true>, g_H, b128, 0, stream,
                (const __half*)L.o_w, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_fp16_swiglu_kernel, g_I, b128, 0, stream,
                (const __half*)L.g_w, (const __half*)L.u_w, (const __half*)d_norm, (__half*)d_mlp, I, H);
            hipLaunchKernelGGL(gemv_fp16_kernel<true>, g_H, b128, 0, stream,
                (const __half*)L.d_w, (const __half*)d_mlp, (__half*)d_x, H, I);
        }
    }

    if (do_lm_head) {
        hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
            (const __half*)d_x, (const __half*)final_norm_w, (__half*)d_norm, H, eps);
        if (lm_is_w4) {
            hipLaunchKernelGGL(gemv_w4a16_kernel<false>, g_lm, b128, 0, stream,
                (const uint32_t*)lm_w, (const __half*)lm_s, (const __half*)d_norm, (__half*)d_logits, lm_vocab, H);
        } else {
            hipLaunchKernelGGL(gemv_w8a16_kernel<false>, g_lm, b128, 0, stream,
                (const int8_t*)lm_w, (const __half*)lm_s, (const __half*)d_norm, (__half*)d_logits, lm_vocab, H);
        }
        hipLaunchKernelGGL(inc_pos_kernel, dim3(1), dim3(1), 0, stream, (int32_t*)pos_ptr);
    }
}

void* create_ar_chunk_graph(
    const ARLayerPtrs* layers, int start_layer, int end_layer, int quant_mode,
    int do_embed, int do_lm_head,
    const void* embed_tokens, const void* final_norm_w,
    const void* lm_w, const void* lm_s, int lm_vocab, int lm_is_w4,
    const void* inv_freq, const void* token_id_ptr, void* pos_ptr,
    void* d_x, void* d_norm, void* d_qkv, void* d_q, void* d_attn, void* d_mlp,
    void* d_mid_O, void* d_mid_M, void* d_mid_L, void* d_logits,
    int H, int I, int Hq, int Hkv, int max_seq, float eps, float sm_scale
) {
    hipStream_t cap_stream;
    if (hipStreamCreateWithFlags(&cap_stream, hipStreamNonBlocking) != hipSuccess) return nullptr;
    if (hipStreamBeginCapture(cap_stream, hipStreamCaptureModeThreadLocal) != hipSuccess) {
        (void)hipStreamDestroy(cap_stream);
        return nullptr;
    }
    launch_ar_decode_chunk(
        layers, start_layer, end_layer, quant_mode, do_embed, do_lm_head,
        embed_tokens, final_norm_w, lm_w, lm_s, lm_vocab, lm_is_w4,
        inv_freq, token_id_ptr, pos_ptr,
        d_x, d_norm, d_qkv, d_q, d_attn, d_mlp,
        d_mid_O, d_mid_M, d_mid_L, d_logits,
        H, I, Hq, Hkv, max_seq, eps, sm_scale, cap_stream
    );
    hipGraph_t graph = nullptr;
    if (hipStreamEndCapture(cap_stream, &graph) != hipSuccess) {
        (void)hipStreamDestroy(cap_stream);
        return nullptr;
    }
    (void)hipStreamDestroy(cap_stream);
    hipGraphExec_t exec = nullptr;
    if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess) {
        (void)hipGraphDestroy(graph);
        return nullptr;
    }
    (void)hipGraphDestroy(graph);
    return (void*)exec;
}

static hipStream_t g_decode_stream = nullptr;

void replay_ar_chunk_graphs_dedicated(void** execs, int num_chunks, hipStream_t caller_stream) {
    if (!g_decode_stream) {
        (void)hipStreamCreateWithFlags(&g_decode_stream, hipStreamNonBlocking);
    }
    // Wait for caller_stream (e.g. d_token.fill_) before launching graph on dedicated stream
    (void)hipStreamSynchronize(caller_stream);
    for (int i = 0; i < num_chunks; ++i) {
        (void)hipGraphLaunch((hipGraphExec_t)execs[i], g_decode_stream);
    }
    (void)hipStreamSynchronize(g_decode_stream);
}

void replay_ar_chunk_graphs(void** execs, int num_chunks, hipStream_t stream) {
    for (int i = 0; i < num_chunks; ++i) {
        (void)hipGraphLaunch((hipGraphExec_t)execs[i], stream);
    }
}

void launch_ar_decode_step(
    const ARLayerPtrs* layers,
    int num_layers,
    int quant_mode,
    const void* embed_tokens,
    const void* final_norm_w,
    const void* lm_w,
    const void* lm_s,
    int lm_vocab,
    int lm_is_w4,
    const void* inv_freq,
    const void* token_id_ptr,
    void* pos_ptr,
    void* d_x,
    void* d_norm,
    void* d_qkv,
    void* d_q,
    void* d_attn,
    void* d_mlp,
    void* d_mid_O,
    void* d_mid_M,
    void* d_mid_L,
    void* d_logits,
    int H, int I, int Hq, int Hkv, int max_seq, float eps, float sm_scale,
    hipStream_t stream
) {
    int qkv_dim = (Hq + 2 * Hkv) * 128;
    dim3 b128(128), b256(256), b64(64);
    dim3 g_qkv((qkv_dim + 3) / 4);
    dim3 g_qkv8((qkv_dim + 7) / 8);
    dim3 g_H((H + 3) / 4);
    dim3 g_I((I + 3) / 4);
    dim3 g_I8((I + 7) / 8);
    dim3 g_lm((lm_vocab + 3) / 4);
    dim3 g_lm8((lm_vocab + 7) / 8);

    hipLaunchKernelGGL(embed_lookup_fp16_kernel, dim3(1), b256, 0, stream,
        (const __half*)embed_tokens, (const int64_t*)token_id_ptr, (__half*)d_x, H);

    for (int i = 0; i < num_layers; ++i) {
        const ARLayerPtrs& L = layers[i];
        if (quant_mode == 2) {
            hipLaunchKernelGGL(gemv_w4a16_rmsnorm_kernel, g_qkv8, b256, H * sizeof(__half), stream,
                (const uint32_t*)L.qkv_w, (const __half*)L.qkv_s, (const __half*)d_x, (const __half*)L.in_ln,
                (__half*)d_qkv, qkv_dim, H, eps);
        } else if (quant_mode == 1) {
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.in_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_w8a16_kernel<false>, g_qkv, b128, 0, stream,
                (const int8_t*)L.qkv_w, (const __half*)L.qkv_s, (const __half*)d_norm, (__half*)d_qkv, qkv_dim, H);
        } else {
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.in_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_fp16_kernel<false>, g_qkv, b128, 0, stream,
                (const __half*)L.qkv_w, (const __half*)d_norm, (__half*)d_qkv, qkv_dim, H);
        }

        hipLaunchKernelGGL(qkv_norm_rope_kvcache_kernel, dim3(Hq + 2 * Hkv), b64, 0, stream,
            (const __half*)d_qkv, (const __half*)L.q_nw, (const __half*)L.k_nw,
            (const float*)inv_freq, (const int32_t*)pos_ptr,
            (__half*)d_q, (__half*)L.k_cache, (__half*)L.v_cache, Hq, Hkv, max_seq, eps);

        hipLaunchKernelGGL(flash_decode_gqa_stage1_kernel, dim3(DECODE_SPLITS, Hkv), b128, 0, stream,
            (const __half*)d_q, (const __half*)L.k_cache, (const __half*)L.v_cache,
            (const int32_t*)pos_ptr, (float*)d_mid_O, (float*)d_mid_M, (float*)d_mid_L,
            Hq, Hkv, max_seq, sm_scale);
        hipLaunchKernelGGL(flash_decode_gqa_stage2_kernel, dim3(Hq), b128, 0, stream,
            (const float*)d_mid_O, (const float*)d_mid_M, (const float*)d_mid_L, (__half*)d_attn);

        if (quant_mode == 2) {
            hipLaunchKernelGGL(gemv_w4a16_kernel<true>, g_H, b128, 0, stream,
                (const uint32_t*)L.o_w, (const __half*)L.o_s, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(gemv_w4a16_rmsnorm_swiglu_kernel, g_I8, b256, H * sizeof(__half), stream,
                (const uint32_t*)L.g_w, (const __half*)L.g_s, (const uint32_t*)L.u_w, (const __half*)L.u_s,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_mlp, I, H, eps);
            hipLaunchKernelGGL(gemv_w4a16_kernel<true>, g_H, b128, 0, stream,
                (const uint32_t*)L.d_w, (const __half*)L.d_s, (const __half*)d_mlp, (__half*)d_x, H, I);
        } else if (quant_mode == 1) {
            hipLaunchKernelGGL(flash_decode_gqa_stage2_kernel, dim3(Hq), b128, 0, stream,
                (const float*)d_mid_O, (const float*)d_mid_M, (const float*)d_mid_L, (__half*)d_attn);
            hipLaunchKernelGGL(gemv_w8a16_kernel<true>, g_H, b128, 0, stream,
                (const int8_t*)L.o_w, (const __half*)L.o_s, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_w8a16_swiglu_kernel, g_I, b128, 0, stream,
                (const int8_t*)L.g_w, (const __half*)L.g_s, (const int8_t*)L.u_w, (const __half*)L.u_s,
                (const __half*)d_norm, (__half*)d_mlp, I, H);
            hipLaunchKernelGGL(gemv_w8a16_kernel<true>, g_H, b128, 0, stream,
                (const int8_t*)L.d_w, (const __half*)L.d_s, (const __half*)d_mlp, (__half*)d_x, H, I);
        } else {
            hipLaunchKernelGGL(flash_decode_gqa_stage2_kernel, dim3(Hq), b128, 0, stream,
                (const float*)d_mid_O, (const float*)d_mid_M, (const float*)d_mid_L, (__half*)d_attn);
            hipLaunchKernelGGL(gemv_fp16_kernel<true>, g_H, b128, 0, stream,
                (const __half*)L.o_w, (const __half*)d_attn, (__half*)d_x, H, H);
            hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
                (const __half*)d_x, (const __half*)L.post_ln, (__half*)d_norm, H, eps);
            hipLaunchKernelGGL(gemv_fp16_swiglu_kernel, g_I, b128, 0, stream,
                (const __half*)L.g_w, (const __half*)L.u_w, (const __half*)d_norm, (__half*)d_mlp, I, H);
            hipLaunchKernelGGL(gemv_fp16_kernel<true>, g_H, b128, 0, stream,
                (const __half*)L.d_w, (const __half*)d_mlp, (__half*)d_x, H, I);
        }
    }

    if (lm_is_w4) {
        hipLaunchKernelGGL(gemv_w4a16_rmsnorm_kernel, g_lm8, b256, H * sizeof(__half), stream,
            (const uint32_t*)lm_w, (const __half*)lm_s, (const __half*)d_x, (const __half*)final_norm_w,
            (__half*)d_logits, lm_vocab, H, eps);
    } else {
        hipLaunchKernelGGL(rmsnorm_fp16_kernel, dim3(1), b256, 0, stream,
            (const __half*)d_x, (const __half*)final_norm_w, (__half*)d_norm, H, eps);
        hipLaunchKernelGGL(gemv_w8a16_kernel<false>, g_lm, b128, 0, stream,
            (const int8_t*)lm_w, (const __half*)lm_s, (const __half*)d_norm, (__half*)d_logits, lm_vocab, H);
    }
    hipLaunchKernelGGL(inc_pos_kernel, dim3(1), dim3(1), 0, stream, (int32_t*)pos_ptr);
}

// ===========================================================================
// 11. Native C++ HIP Graph Capture & Launch (100% crash-free, 1 submission!)
// ===========================================================================
void* create_ar_decode_graph(
    const ARLayerPtrs* layers, int num_layers, int quant_mode,
    const void* embed_tokens, const void* final_norm_w,
    const void* lm_w, const void* lm_s, int lm_vocab,
    const void* inv_freq, const void* token_id_ptr, void* pos_ptr,
    void* d_x, void* d_norm, void* d_qkv, void* d_q, void* d_attn, void* d_mlp,
    void* d_mid_O, void* d_mid_M, void* d_mid_L, void* d_logits,
    int H, int I, int Hq, int Hkv, int max_seq, float eps, float sm_scale
) {
    hipStream_t cap_stream;
    if (hipStreamCreateWithFlags(&cap_stream, hipStreamNonBlocking) != hipSuccess) return nullptr;
    if (hipStreamBeginCapture(cap_stream, hipStreamCaptureModeGlobal) != hipSuccess) {
        hipStreamDestroy(cap_stream);
        return nullptr;
    }
    launch_ar_decode_step(
        layers, num_layers, quant_mode, embed_tokens, final_norm_w,
        lm_w, lm_s, lm_vocab, 0, inv_freq, token_id_ptr, pos_ptr,
        d_x, d_norm, d_qkv, d_q, d_attn, d_mlp,
        d_mid_O, d_mid_M, d_mid_L, d_logits,
        H, I, Hq, Hkv, max_seq, eps, sm_scale, cap_stream
    );
    hipGraph_t graph = nullptr;
    if (hipStreamEndCapture(cap_stream, &graph) != hipSuccess) {
        hipStreamDestroy(cap_stream);
        return nullptr;
    }
    hipStreamDestroy(cap_stream);
    hipGraphExec_t exec = nullptr;
    if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess) {
        hipGraphDestroy(graph);
        return nullptr;
    }
    hipGraphDestroy(graph);
    return (void*)exec;
}

void replay_ar_decode_graph(void* exec, hipStream_t stream) {
    hipGraphLaunch((hipGraphExec_t)exec, stream);
}

void destroy_ar_decode_graph(void* exec) {
    if (exec) hipGraphExecDestroy((hipGraphExec_t)exec);
}

// ===========================================================================
// 12. Fused Windowed Repetition Penalty Kernel (zero .unique() sync!)
// Applies penalty in-place on scores_fp32[V] using history_gpu[hist_len] (max 100)
// ===========================================================================
__global__ void apply_rep_penalty_kernel(
    float* __restrict__ scores,
    const int32_t* __restrict__ hist,
    int hist_len,
    float penalty
) {
    int i = threadIdx.x;  // 0..hist_len-1 (max 128)
    if (i >= hist_len) return;
    int tok = hist[i];
    if (tok < 0) return;
    int count = 0;
    int earlier = 0;
    for (int j = 0; j < hist_len; ++j) {
        if (hist[j] == tok) {
            count++;
            if (j < i) earlier++;
        }
    }
    if (earlier == 0) {
        float alpha = powf(penalty, (float)count);
        float v = scores[tok];
        scores[tok] = (v < 0.0f) ? (v * alpha) : (v / alpha);
    }
}

__global__ void apply_rep_penalty_fp16_kernel(
    __half* __restrict__ scores,
    const int32_t* __restrict__ hist,
    int hist_len,
    float penalty,
    int mask_end_idx
) {
    int i = threadIdx.x;
    if (i == 0 && mask_end_idx >= 0) {
        scores[mask_end_idx] = __float2half(-65504.0f);
    }
    if (i >= hist_len || penalty == 1.0f) return;
    int tok = hist[i];
    if (tok < 0) return;
    int count = 0;
    int earlier = 0;
    for (int j = 0; j < hist_len; ++j) {
        if (hist[j] == tok) {
            count++;
            if (j < i) earlier++;
        }
    }
    if (earlier == 0) {
        float alpha = powf(penalty, (float)count);
        float v = __half2float(scores[tok]);
        scores[tok] = __float2half((v < 0.0f) ? (v * alpha) : (v / alpha));
    }
}

void launch_apply_rep_penalty_fp16(void* scores_fp16, const void* hist_i32, int hist_len, float penalty, int mask_end_idx, hipStream_t stream) {
    if (hist_len <= 0 && mask_end_idx < 0) return;
    hipLaunchKernelGGL(apply_rep_penalty_fp16_kernel, dim3(1), dim3(128), 0, stream,
        (__half*)scores_fp16, (const int32_t*)hist_i32, hist_len, penalty, mask_end_idx);
}

void launch_apply_rep_penalty(void* scores_fp32, const void* hist_i32, int hist_len, float penalty, hipStream_t stream) {
    if (hist_len <= 0 || penalty == 1.0f) return;
    hipLaunchKernelGGL(apply_rep_penalty_kernel, dim3(1), dim3(128), 0, stream,
        (float*)scores_fp32, (const int32_t*)hist_i32, hist_len, penalty);
}

// ===========================================================================
// 13. Native Wave32 HIP Online-Softmax + Acc Kernels for rocBLAS Tiled Attention
// (Replaces Triton -> ZERO Triton JIT overhead AND allows 100% crash-free HIP Graphs!)
// ===========================================================================

// Single-pass online softmax for N_COLS <= 512 (1 block = 256 threads = 8 warps, 2 elems/thread)
__global__ void online_softmax_1pass_hip_kernel(
    __half* __restrict__ S,          // [M_tot, stride_sm] in-place S -> P
    float* __restrict__ M_buf,
    float* __restrict__ L_buf,
    float* __restrict__ Alpha_buf,
    int stride_sm, int N_COLS, float sm_scale_log2e
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;  // 0..255
    int wid = tid >> 5;
    int lane = tid & 31;

    __half* s_row = S + (size_t)row * stride_sm;
    float v0 = (tid < N_COLS) ? (__half2float(s_row[tid]) * sm_scale_log2e) : -1e30f;
    float v1 = (tid + 256 < N_COLS) ? (__half2float(s_row[tid + 256]) * sm_scale_log2e) : -1e30f;

    float mx = warp_reduce_max(fmaxf(v0, v1));
    __shared__ float smem[8];
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < 8) ? smem[lane] : -1e30f;
        t = warp_reduce_max(t);
        if (lane == 0) smem[0] = fmaxf(M_buf[row], t);
    }
    __syncthreads();
    float row_max = smem[0];

    float p0 = (tid < N_COLS) ? exp2f(v0 - row_max) : 0.0f;
    float p1 = (tid + 256 < N_COLS) ? exp2f(v1 - row_max) : 0.0f;
    if (tid < N_COLS) s_row[tid] = __float2half(p0);
    if (tid + 256 < N_COLS) s_row[tid + 256] = __float2half(p1);

    float sm = warp_reduce_sum(p0 + p1);
    if (lane == 0) smem[wid] = sm;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < 8) ? smem[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) {
            float m_old = M_buf[row];
            float alpha = (m_old > -1e29f) ? exp2f(m_old - row_max) : 0.0f;
            M_buf[row] = row_max;
            L_buf[row] = L_buf[row] * alpha + t;
            Alpha_buf[row] = alpha;
        }
    }
}

__global__ void online_softmax_causal_hip_kernel(
    __half* __restrict__ S,
    float* __restrict__ M_buf,
    float* __restrict__ L_buf,
    float* __restrict__ Alpha_buf,
    int stride_sm, int N_Q, int N_COLS, int kv_start, float sm_scale_log2e
) {
    int row = blockIdx.x;
    int q_idx = row % N_Q;
    int tid = threadIdx.x;  // 0..255
    int wid = tid >> 5;
    int lane = tid & 31;

    __half* s_row = S + (size_t)row * stride_sm;
    bool m0 = (tid < N_COLS) && (kv_start + tid <= q_idx);
    bool m1 = (tid + 256 < N_COLS) && (kv_start + tid + 256 <= q_idx);
    float v0 = m0 ? (__half2float(s_row[tid]) * sm_scale_log2e) : -1e30f;
    float v1 = m1 ? (__half2float(s_row[tid + 256]) * sm_scale_log2e) : -1e30f;

    float mx = warp_reduce_max(fmaxf(v0, v1));
    __shared__ float smem[8];
    if (lane == 0) smem[wid] = mx;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < 8) ? smem[lane] : -1e30f;
        t = warp_reduce_max(t);
        if (lane == 0) smem[0] = fmaxf(M_buf[row], t);
    }
    __syncthreads();
    float row_max = smem[0];
    bool valid_row = row_max > -1e29f;
    float safe_max = valid_row ? row_max : 0.0f;

    float p0 = m0 ? exp2f(v0 - safe_max) : 0.0f;
    float p1 = m1 ? exp2f(v1 - safe_max) : 0.0f;
    if (tid < N_COLS) s_row[tid] = __float2half(p0);
    if (tid + 256 < N_COLS) s_row[tid + 256] = __float2half(p1);

    float sm = warp_reduce_sum(p0 + p1);
    if (lane == 0) smem[wid] = sm;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < 8) ? smem[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) {
            float m_old = M_buf[row];
            float alpha = (valid_row && (m_old > -1e29f)) ? exp2f(m_old - safe_max) : 1.0f;
            M_buf[row] = row_max;
            L_buf[row] = L_buf[row] * alpha + t;
            Alpha_buf[row] = alpha;
        }
    }
}

__global__ void acc_update_hip_kernel(
    float* __restrict__ Acc,
    const __half* __restrict__ PV,
    const float* __restrict__ Alpha,
    int D
) {
    int row = blockIdx.x;
    int d = threadIdx.x;  // 0..127
    float alpha = Alpha[row];
    size_t idx = (size_t)row * D + d;
    Acc[idx] = fmaf(Acc[idx], alpha, __half2float(PV[idx]));
}

__global__ void acc_finalize_transpose_hip_kernel(
    const float* __restrict__ Acc,
    const float* __restrict__ L,
    __half* __restrict__ Out,
    int Hq, int N_Q, int D
) {
    int row = blockIdx.x;
    int h = row / N_Q;
    int m = row % N_Q;
    int d = threadIdx.x;  // 0..127
    float inv_l = 1.0f / L[row];
    Out[((size_t)m * Hq + h) * D + d] = __float2half(Acc[(size_t)row * D + d] * inv_l);
}

void launch_online_softmax_hip(
    void* S, void* M_buf, void* L_buf, void* Alpha_buf,
    int M_tot, int stride_sm, int N_Q, int N_COLS, int kv_start, float sm_scale, int causal, hipStream_t stream
) {
    float sm_scale_log2e = sm_scale * 1.4426950408889634f;
    if (causal) {
        hipLaunchKernelGGL(online_softmax_causal_hip_kernel, dim3(M_tot), dim3(256), 0, stream,
            (__half*)S, (float*)M_buf, (float*)L_buf, (float*)Alpha_buf,
            stride_sm, N_Q, N_COLS, kv_start, sm_scale_log2e);
    } else {
        hipLaunchKernelGGL(online_softmax_1pass_hip_kernel, dim3(M_tot), dim3(256), 0, stream,
            (__half*)S, (float*)M_buf, (float*)L_buf, (float*)Alpha_buf,
            stride_sm, N_COLS, sm_scale_log2e);
    }
}

void launch_acc_update_hip(void* Acc, const void* PV, const void* Alpha, int M_tot, int D, hipStream_t stream) {
    hipLaunchKernelGGL(acc_update_hip_kernel, dim3(M_tot), dim3(D), 0, stream,
        (float*)Acc, (const __half*)PV, (const float*)Alpha, D);
}

void launch_acc_finalize_transpose_hip(const void* Acc, const void* L, void* Out, int Hq, int N_Q, int D, hipStream_t stream) {
    hipLaunchKernelGGL(acc_finalize_transpose_hip_kernel, dim3(Hq * N_Q), dim3(D), 0, stream,
        (const float*)Acc, (const float*)L, (__half*)Out, Hq, N_Q, D);
}

} // extern "C"
