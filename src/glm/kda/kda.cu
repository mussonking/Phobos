// src/glm/kda/kda.cu - GLM-5.3-Flash KDA attention branch (see include/strata/glm/kda.hpp for the math and the
// state contract).
//
// Launch sequence for one batch of T tokens:
//   quantize x -> 6 projections (q, k, v, f_a, g_a, beta) -> f_b(f_a), g_b(g_a)
//   kda_prep   conv4 + SiLU for q/k/v, per-head L2 of q/k, decay gate g, beta, new conv history
//   recurrence sequential warp-per-value-row scan (decode/verify) or the chunked form (prefill)
//   kda_post   per-head RMSNorm(o) * ssm_norm * sigmoid(g2)
//   out projection
// Projections go through the shared glm_matmul (Q8 GEMV for T <= 32, F16 tensor-core GEMM above); q/k/v/f_a/g_a/beta
// share one prepared input.
#include "kda_internal.hpp"

#include "strata/glm/matmul.hpp"
#include "strata/glm/q8_block.cuh"

#include <cuda_runtime.h>

#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::glm {

using namespace kda;

namespace {

constexpr int kGgmlQ8_0 = 8;
constexpr int kTypeBf16 = 30;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("kda: ") + what + ": " + cudaGetErrorString(e));
}

size_t al(size_t n) { return (n + 255) & ~size_t(255); }

Views g_views;
int   g_prefill_mode = 1;
bool  g_fused_decode = true;
bool  g_skip_proj = false;   // bench only: time the non-projection part alone
cudaEvent_t * g_stage_ev = nullptr;   // 6 events: start, in-proj, prep, scan, post, out-proj (bench only)
inline void stage(int i, cudaStream_t st) { if (g_stage_ev) cudaEventRecord(g_stage_ev[i], st); }

// ---------------------------------------------------------------------------------------------------------
// small device helpers

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------------------------------------
// kda_prep: grid (64 heads, ceil(T / kPrepT)), block 128 (one thread per channel of the head).

constexpr int kPrepT = 8;

__global__ void __launch_bounds__(128) kda_prep_kernel(
        int T,
        const float * __restrict__ raw_q, const float * __restrict__ raw_k, const float * __restrict__ raw_v,
        const float * __restrict__ hist_in, float * __restrict__ hist_out,
        const float * __restrict__ wq, const float * __restrict__ wk, const float * __restrict__ wv,
        const float * __restrict__ glin, const float * __restrict__ dt_bias, const float * __restrict__ A,
        const float * __restrict__ braw,
        float * __restrict__ q, float * __restrict__ k, float * __restrict__ v, float * __restrict__ g,
        float * __restrict__ beta) {
    const int h  = blockIdx.x;
    const int c  = threadIdx.x;
    const int ch = h * kDim + c;
    const int warp = c >> 5, lane = c & 31;
    __shared__ float red[2][4];

    const float * raw[3]  = {raw_q, raw_k, raw_v};
    const float * wts[3]  = {wq, wk, wv};
    float w[3][4], hist[3][3];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) w[m][i] = wts[m][ch * 4 + i];
#pragma unroll
        for (int r = 0; r < kHist; ++r) hist[m][r] = hist_in[(m * kHist + r) * kInner + ch];
    }
    const float a_h = A[h];
    const float dt  = dt_bias[ch];

    const int t0 = blockIdx.y * kPrepT;
    const int t1 = min(T, t0 + kPrepT);
    for (int t = t0; t < t1; ++t) {
        float y[3];
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = t + i;   // index into concat(hist[3], raw[T])
                const float xv = idx < kHist ? hist[m][idx] : raw[m][(size_t) (idx - kHist) * kInner + ch];
                acc += xv * w[m][i];
            }
            y[m] = siluf_(acc);
        }
        // per-head L2 norm of q and k: x / sqrt(sum x^2 + 1e-6)
        float sq = warp_sum(y[0] * y[0]);
        float sk = warp_sum(y[1] * y[1]);
        if (lane == 0) { red[0][warp] = sq; red[1][warp] = sk; }
        __syncthreads();
        sq = red[0][0] + red[0][1] + red[0][2] + red[0][3];
        sk = red[1][0] + red[1][1] + red[1][2] + red[1][3];
        __syncthreads();
        const size_t o = (size_t) t * kInner + ch;
        q[o] = y[0] * rsqrtf(sq + 1e-6f);
        k[o] = y[1] * rsqrtf(sk + 1e-6f);
        v[o] = y[2];
        // g = -5 * sigmoid(-(z * A)), z = W_fb W_fa x + dt_bias
        const float z = glin[o] + dt;
        g[o] = -5.0f * sigmoidf_(-(z * a_h));
        if (c == 0) beta[t * kHeads + h] = sigmoidf_(braw[t * kHeads + h]);
    }

    // new history = rows T, T+1, T+2 of concat(hist, raw)
    if (blockIdx.y == 0) {
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
            for (int r = 0; r < kHist; ++r) {
                const int idx = T + r;
                const float xv = idx < kHist ? hist[m][idx] : raw[m][(size_t) (idx - kHist) * kInner + ch];
                hist_out[(m * kHist + r) * kInner + ch] = xv;
            }
    }
}

// kda_prep_rows (long batches): one warp per (token, head) row, lane = float4 channel group, no block barriers;
// grid (T * 64 / 4), block 128. Memory-bound: reads the raw q/k/v (+3 earlier rows, cached) and glin, writes
// q/k/v/g. Block 0 also writes the new conv history (T > kHist here, so the rows all come from raw).
__global__ void __launch_bounds__(128) kda_prep_rows_kernel(
        int T,
        const float * __restrict__ raw_q, const float * __restrict__ raw_k, const float * __restrict__ raw_v,
        const float * __restrict__ hist_in, float * __restrict__ hist_out,
        const float * __restrict__ wq, const float * __restrict__ wk, const float * __restrict__ wv,
        const float * __restrict__ glin, const float * __restrict__ dt_bias, const float * __restrict__ A,
        const float * __restrict__ braw,
        float * __restrict__ q, float * __restrict__ k, float * __restrict__ v, float * __restrict__ g,
        float * __restrict__ beta) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);   // t * 64 + h
    const int lane = threadIdx.x & 31;
    if (blockIdx.x == 0) {
        const float * raw[3] = {raw_q, raw_k, raw_v};
        for (int i = threadIdx.x; i < 3 * kHist * (kInner / 4); i += 128) {
            const int c4 = i % (kInner / 4), r = (i / (kInner / 4)) % kHist, m = i / (kHist * (kInner / 4));
            reinterpret_cast<float4 *>(hist_out + (size_t) (m * kHist + r) * kInner)[c4] =
                reinterpret_cast<const float4 *>(raw[m] + (size_t) (T - kHist + r) * kInner)[c4];
        }
    }
    if (row >= T * kHeads) return;
    const int t = row / kHeads, h = row % kHeads;
    const int c0 = h * kDim + 4 * lane;   // first channel of this lane
    const float * raw[3] = {raw_q, raw_k, raw_v};
    const float * wts[3] = {wq, wk, wv};
    float4 y[3];
#pragma unroll
    for (int m = 0; m < 3; ++m) {
        float wv16[16];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float4 w4 = reinterpret_cast<const float4 *>(wts[m])[c0 + e];
            wv16[e * 4 + 0] = w4.x; wv16[e * 4 + 1] = w4.y; wv16[e * 4 + 2] = w4.z; wv16[e * 4 + 3] = w4.w;
        }
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = t - kHist + i;
            const float4 x = r >= 0 ? *reinterpret_cast<const float4 *>(raw[m] + (size_t) r * kInner + c0)
                                    : *reinterpret_cast<const float4 *>(hist_in + (size_t) (m * kHist + kHist + r) * kInner + c0);
            acc[0] += x.x * wv16[0 * 4 + i]; acc[1] += x.y * wv16[1 * 4 + i];
            acc[2] += x.z * wv16[2 * 4 + i]; acc[3] += x.w * wv16[3 * 4 + i];
        }
        y[m] = make_float4(siluf_(acc[0]), siluf_(acc[1]), siluf_(acc[2]), siluf_(acc[3]));
    }
    const float rq = rsqrtf(warp_sum(y[0].x * y[0].x + y[0].y * y[0].y + y[0].z * y[0].z + y[0].w * y[0].w) + 1e-6f);
    const float rk = rsqrtf(warp_sum(y[1].x * y[1].x + y[1].y * y[1].y + y[1].z * y[1].z + y[1].w * y[1].w) + 1e-6f);
    const size_t o = (size_t) t * kInner + c0;
    *reinterpret_cast<float4 *>(q + o) = make_float4(y[0].x * rq, y[0].y * rq, y[0].z * rq, y[0].w * rq);
    *reinterpret_cast<float4 *>(k + o) = make_float4(y[1].x * rk, y[1].y * rk, y[1].z * rk, y[1].w * rk);
    *reinterpret_cast<float4 *>(v + o) = y[2];
    const float4 gl = *reinterpret_cast<const float4 *>(glin + o);
    const float4 dt = *reinterpret_cast<const float4 *>(dt_bias + c0);
    const float a_h = A[h];
    *reinterpret_cast<float4 *>(g + o) = make_float4(-5.0f * sigmoidf_(-((gl.x + dt.x) * a_h)),
                                                     -5.0f * sigmoidf_(-((gl.y + dt.y) * a_h)),
                                                     -5.0f * sigmoidf_(-((gl.z + dt.z) * a_h)),
                                                     -5.0f * sigmoidf_(-((gl.w + dt.w) * a_h)));
    if (lane == 0) beta[t * kHeads + h] = sigmoidf_(braw[t * kHeads + h]);
}

// ---------------------------------------------------------------------------------------------------------
// Sequential recurrence. Each warp owns kRowsW value rows j of one head; lane l holds key channels 4l..4l+3 of
// those rows in registers for the whole batch, so the state is read and written exactly once. Token inputs are
// staged into shared memory kTile tokens at a time (one round of loads), so the per-token chain is arithmetic
// and shuffles only.

constexpr int kRowsW = 4;
constexpr int kWarpsScan = 4;
constexpr int kRowsB = kRowsW * kWarpsScan;   // 16 rows per block -> grid (8, 64)
constexpr int kTile = 16;

// One token of the delta rule for a warp's kRowsW rows; returns o for row `lane` (lane < kRowsW) when WRITE_O.
template <bool WRITE_O>
__device__ __forceinline__ float delta_step(float4 (&s)[kRowsW], float4 kk, float4 dc, float4 qq, float b,
                                            const float * vrow, int lane) {
    float p[kRowsW];
#pragma unroll
    for (int r = 0; r < kRowsW; ++r) {
        s[r].x *= dc.x; s[r].y *= dc.y; s[r].z *= dc.z; s[r].w *= dc.w;
        p[r] = s[r].x * kk.x + s[r].y * kk.y + s[r].z * kk.z + s[r].w * kk.w;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int r = 0; r < kRowsW; ++r) p[r] += __shfl_xor_sync(0xffffffffu, p[r], off);
#pragma unroll
    for (int r = 0; r < kRowsW; ++r) {
        const float d = b * (vrow[r] - p[r]);
        s[r].x += kk.x * d; s[r].y += kk.y * d; s[r].z += kk.z * d; s[r].w += kk.w * d;
    }
    if (!WRITE_O) return 0.0f;
#pragma unroll
    for (int r = 0; r < kRowsW; ++r) p[r] = s[r].x * qq.x + s[r].y * qq.y + s[r].z * qq.z + s[r].w * qq.w;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int r = 0; r < kRowsW; ++r) p[r] += __shfl_xor_sync(0xffffffffu, p[r], off);
    float pv = p[0];
#pragma unroll
    for (int r = 1; r < kRowsW; ++r) if (lane == r) pv = p[r];
    return pv;
}

template <bool WRITE_O>
__global__ void __launch_bounds__(kWarpsScan * 32) kda_scan_seq_kernel(
        int T, const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * S_in, float * S_out, float * __restrict__ o) {
    const int h = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int jb = blockIdx.x * kRowsB, j0 = jb + warp * kRowsW;
    const float scale = rsqrtf((float) kDim);
    __shared__ float4 sk[kTile][kDim / 4], sq[kTile][kDim / 4], sdc[kTile][kDim / 4];
    __shared__ float sv[kTile][kRowsB];
    __shared__ float sb[kTile];

    float4 s[kRowsW];
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
        s[r] = reinterpret_cast<const float4 *>(S_in + ((size_t) h * kDim + j0 + r) * kDim)[lane];

    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int nt = min(kTile, T - t0);
        // stage: 128 threads, 32 float4 per token row
        for (int i = tid; i < nt * (kDim / 4); i += kWarpsScan * 32) {
            const int t = i / (kDim / 4), c4 = i % (kDim / 4);
            const size_t base = (size_t) (t0 + t) * kInner + (size_t) h * kDim;
            sk[t][c4] = reinterpret_cast<const float4 *>(k + base)[c4];
            if (WRITE_O) sq[t][c4] = reinterpret_cast<const float4 *>(q + base)[c4];
            const float4 gg = reinterpret_cast<const float4 *>(g + base)[c4];
            sdc[t][c4] = make_float4(expf(gg.x), expf(gg.y), expf(gg.z), expf(gg.w));
        }
        for (int i = tid; i < nt * kRowsB; i += kWarpsScan * 32) {
            const int t = i / kRowsB, r = i % kRowsB;
            sv[t][r] = v[(size_t) (t0 + t) * kInner + (size_t) h * kDim + jb + r];
        }
        if (tid < nt) sb[tid] = beta[(t0 + tid) * kHeads + h];
        __syncthreads();
        for (int t = 0; t < nt; ++t) {
            const float pv = delta_step<WRITE_O>(s, sk[t][lane], sdc[t][lane], WRITE_O ? sq[t][lane] : sk[t][lane], sb[t],
                                                 &sv[t][warp * kRowsW], lane);
            if (WRITE_O && lane < kRowsW) o[(size_t) (t0 + t) * kInner + (size_t) h * kDim + j0 + lane] = pv * scale;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
        reinterpret_cast<float4 *>(S_out + ((size_t) h * kDim + j0 + r) * kDim)[lane] = s[r];
}

// ---------------------------------------------------------------------------------------------------------
// Sequential recurrence for long batches (prefill). Four threads share R value rows j, each thread holding the
// same 32 key channels of all R rows (float4 chunks qq, qq+4, ..., interleaved so the four threads read 64
// contiguous bytes of staged k/q/decay). Staged operands are read once per R rows: shared-memory bandwidth, not
// the FP32 pipe, limited the one-row-per-thread layout. Block = RB rows of one head (RB/R*4 threads).

constexpr int kRqTile = 16;

template <int R, int RB, bool WRITE_O>
__global__ void __launch_bounds__(RB / R * 4) kda_scan_rq_kernel(
        int T, const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * S_in, float * S_out, float * __restrict__ o) {
    constexpr int NT = RB / R * 4;
    const int h = blockIdx.y;
    const int tid = threadIdx.x, pr = tid >> 2, qq = tid & 3;
    const int jb = blockIdx.x * RB, jr = pr * R;     // this thread's rows: jb + jr + rr
    const float scale = rsqrtf((float) kDim);
    __shared__ float4 sk[kRqTile][kDim / 4], sq[kRqTile][kDim / 4], sdc[kRqTile][kDim / 4];
    __shared__ float sv[kRqTile][RB];
    __shared__ float sb[kRqTile];

    float4 s[R][8];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const float4 * srow = reinterpret_cast<const float4 *>(S_in + ((size_t) h * kDim + jb + jr + rr) * kDim);
#pragma unroll
        for (int m = 0; m < 8; ++m) s[rr][m] = srow[qq + 4 * m];
    }

    for (int t0 = 0; t0 < T; t0 += kRqTile) {
        const int nt = min(kRqTile, T - t0);
        for (int i = tid; i < nt * (kDim / 4); i += NT) {
            const int t = i / (kDim / 4), c4 = i % (kDim / 4);
            const size_t base = (size_t) (t0 + t) * kInner + (size_t) h * kDim;
            sk[t][c4] = reinterpret_cast<const float4 *>(k + base)[c4];
            if (WRITE_O) sq[t][c4] = reinterpret_cast<const float4 *>(q + base)[c4];
            const float4 gg = reinterpret_cast<const float4 *>(g + base)[c4];
            sdc[t][c4] = make_float4(expf(gg.x), expf(gg.y), expf(gg.z), expf(gg.w));
        }
        for (int i = tid; i < nt * RB; i += NT) {
            const int t = i / RB, rr = i % RB;
            sv[t][rr] = v[(size_t) (t0 + t) * kInner + (size_t) h * kDim + jb + rr];
        }
        if (tid < nt) sb[tid] = beta[(t0 + tid) * kHeads + h];
        __syncthreads();
        for (int t = 0; t < nt; ++t) {
            float4 kk[8];
            float p[R];
#pragma unroll
            for (int rr = 0; rr < R; ++rr) p[rr] = 0.0f;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                kk[m] = sk[t][qq + 4 * m];
                const float4 dc = sdc[t][qq + 4 * m];
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    float4 & x = s[rr][m];
                    x.x *= dc.x; x.y *= dc.y; x.z *= dc.z; x.w *= dc.w;
                    p[rr] += x.x * kk[m].x + x.y * kk[m].y + x.z * kk[m].z + x.w * kk[m].w;
                }
            }
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                p[rr] += __shfl_xor_sync(0xffffffffu, p[rr], 1);
                p[rr] += __shfl_xor_sync(0xffffffffu, p[rr], 2);
            }
            const float b = sb[t];
            float d[R], ov[R];
#pragma unroll
            for (int rr = 0; rr < R; ++rr) { d[rr] = b * (sv[t][jr + rr] - p[rr]); ov[rr] = 0.0f; }
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                float4 qv;
                if (WRITE_O) qv = sq[t][qq + 4 * m];
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    float4 & x = s[rr][m];
                    x.x += kk[m].x * d[rr]; x.y += kk[m].y * d[rr]; x.z += kk[m].z * d[rr]; x.w += kk[m].w * d[rr];
                    if (WRITE_O) ov[rr] += x.x * qv.x + x.y * qv.y + x.z * qv.z + x.w * qv.w;
                }
            }
            if (WRITE_O) {
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    ov[rr] += __shfl_xor_sync(0xffffffffu, ov[rr], 1);
                    ov[rr] += __shfl_xor_sync(0xffffffffu, ov[rr], 2);
                }
                if (qq < R) {   // thread qq writes row qq (R <= 4)
                    float w = ov[0];
#pragma unroll
                    for (int rr = 1; rr < R; ++rr) if (qq == rr) w = ov[rr];
                    o[(size_t) (t0 + t) * kInner + (size_t) h * kDim + jb + jr + qq] = w * scale;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        float4 * drow = reinterpret_cast<float4 *>(S_out + ((size_t) h * kDim + jb + jr + rr) * kDim);
#pragma unroll
        for (int m = 0; m < 8; ++m) drow[qq + 4 * m] = s[rr][m];
    }
}

template <int R, int RB>
void launch_rq(int T, const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * S_in, float * S_out, float * o, cudaStream_t st) {
    const dim3 grid(kDim / RB, kHeads);
    if (o) kda_scan_rq_kernel<R, RB, true><<<grid, RB / R * 4, 0, st>>>(T, q, k, v, g, beta, S_in, S_out, o);
    else   kda_scan_rq_kernel<R, RB, false><<<grid, RB / R * 4, 0, st>>>(T, q, k, v, g, beta, S_in, S_out, nullptr);
}

int g_rq_variant = 0;

// ---------------------------------------------------------------------------------------------------------
// kda_post: grid (64, T), block 128. gated = RMSNorm_head(o) * norm * sigmoid(g2)

__global__ void __launch_bounds__(128) kda_post_kernel(
        const float * __restrict__ o, const float * __restrict__ g2, const float * __restrict__ norm_w,
        float eps, float * __restrict__ gated, int8_t * __restrict__ gq, float * __restrict__ gd) {
    const int h = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    const int warp = j >> 5, lane = j & 31;
    __shared__ float red[4];
    const size_t idx = (size_t) t * kInner + (size_t) h * kDim + j;
    const float x = o[idx];
    const float s = warp_sum(x * x);
    if (lane == 0) red[warp] = s;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / (float) kDim;
    const float gv = x * rsqrtf(mean + eps) * norm_w[j] * sigmoidf_(g2[idx]);
    gated[idx] = gv;
    if (gq) {   // Q8 prepared input of the output projection, one warp per 32-block
        const float amax = q8_group_amax<32>(fabsf(gv));
        gq[idx] = q8_quant(gv, q8_inv_scale(amax));
        if (lane == 0) gd[idx / 32] = q8_scale(amax);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Decode / verify (T <= kKdaLogT): prep + recurrence + post in one launch, T a template parameter. Grid (8, 64),
// block 128. The state loads are issued first; every block then redoes its head's conv / L2 / gates for all T
// tokens at once (all loads in one round, one barrier for the 2T norms) into shared memory - 8x redundant, a
// few KB of L2 reads. Block x == 0 also writes q, the prepared k/v/g/beta log and the new conv history. The last
// block of a head to finish (counter + fences) applies RMSNorm * sigmoid(g2) to the head's output.

template <int T>
__global__ void __launch_bounds__(kWarpsScan * 32) kda_decode_fused_kernel(
        const float * __restrict__ raw_q, const float * __restrict__ raw_k, const float * __restrict__ raw_v,
        const float * __restrict__ hist_in, float * __restrict__ hist_out,
        const float * __restrict__ wq, const float * __restrict__ wk, const float * __restrict__ wv,
        const float * __restrict__ glin, const float * __restrict__ dt_bias, const float * __restrict__ A,
        const float * __restrict__ braw,
        float * __restrict__ q_out, float * __restrict__ k_log, float * __restrict__ v_log, float * __restrict__ g_log,
        float * __restrict__ beta_log,
        const float * __restrict__ S_in, float * __restrict__ S_out, float * o,
        const float * __restrict__ g2, const float * __restrict__ norm_w, float eps, float * __restrict__ gated,
        unsigned * head_done, int8_t * __restrict__ gq, float * __restrict__ gd) {
    const int h = blockIdx.y;
    const int c = threadIdx.x;
    const int ch = h * kDim + c;
    const int warp = c >> 5, lane = c & 31;
    const int jb = blockIdx.x * kRowsB, j0 = jb + warp * kRowsW;
    const bool writer = blockIdx.x == 0;
    __shared__ float4 sq[T][kDim / 4], sk[T][kDim / 4], sdc[T][kDim / 4];
    __shared__ float sv[T][kDim];
    __shared__ float sbeta[T];
    __shared__ float red[2][T][4];
    __shared__ bool last;

    float4 s[kRowsW];
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
        s[r] = reinterpret_cast<const float4 *>(S_in + ((size_t) h * kDim + j0 + r) * kDim)[lane];

    // ---- prep, all T tokens at once
    {
        const float * raw[3] = {raw_q, raw_k, raw_v};
        const float * wts[3] = {wq, wk, wv};
        float xs[3][kHist + T], w[3][4];
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const float4 w4 = reinterpret_cast<const float4 *>(wts[m])[ch];
            w[m][0] = w4.x; w[m][1] = w4.y; w[m][2] = w4.z; w[m][3] = w4.w;
#pragma unroll
            for (int r = 0; r < kHist; ++r) xs[m][r] = hist_in[(m * kHist + r) * kInner + ch];
#pragma unroll
            for (int t = 0; t < T; ++t) xs[m][kHist + t] = raw[m][(size_t) t * kInner + ch];
        }
        float gl[T];
#pragma unroll
        for (int t = 0; t < T; ++t) gl[t] = glin[(size_t) t * kInner + ch];
        const float braw_c = c < T ? braw[c * kHeads + h] : 0.0f;
        const float a_h = A[h];
        const float dt = dt_bias[ch];

        float y[3][T], ss[2][T];
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
            for (int t = 0; t < T; ++t)
                y[m][t] = siluf_(xs[m][t] * w[m][0] + xs[m][t + 1] * w[m][1] + xs[m][t + 2] * w[m][2] +
                                 xs[m][t + 3] * w[m][3]);
#pragma unroll
        for (int t = 0; t < T; ++t) { ss[0][t] = y[0][t] * y[0][t]; ss[1][t] = y[1][t] * y[1][t]; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int t = 0; t < T; ++t) {
                ss[0][t] += __shfl_xor_sync(0xffffffffu, ss[0][t], off);
                ss[1][t] += __shfl_xor_sync(0xffffffffu, ss[1][t], off);
            }
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < T; ++t) { red[0][t][warp] = ss[0][t]; red[1][t][warp] = ss[1][t]; }
        if (c < T) sbeta[c] = sigmoidf_(braw_c);
        __syncthreads();
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float qn = y[0][t] * rsqrtf(red[0][t][0] + red[0][t][1] + red[0][t][2] + red[0][t][3] + 1e-6f);
            const float kn = y[1][t] * rsqrtf(red[1][t][0] + red[1][t][1] + red[1][t][2] + red[1][t][3] + 1e-6f);
            const float gv = -5.0f * sigmoidf_(-((gl[t] + dt) * a_h));
            reinterpret_cast<float *>(sq[t])[c] = qn;
            reinterpret_cast<float *>(sk[t])[c] = kn;
            reinterpret_cast<float *>(sdc[t])[c] = expf(gv);
            sv[t][c] = y[2][t];
            if (writer) {
                const size_t o_idx = (size_t) t * kInner + ch;
                q_out[o_idx] = qn;
                k_log[o_idx] = kn;
                v_log[o_idx] = y[2][t];
                g_log[o_idx] = gv;
            }
        }
        if (writer) {
#pragma unroll
            for (int m = 0; m < 3; ++m)
#pragma unroll
                for (int r = 0; r < kHist; ++r) hist_out[(m * kHist + r) * kInner + ch] = xs[m][T + r];
            if (c < T) beta_log[c * kHeads + h] = sbeta[c];
        }
        __syncthreads();
    }

    // ---- recurrence
    const float scale = rsqrtf((float) kDim);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float pv = delta_step<true>(s, sk[t][lane], sdc[t][lane], sq[t][lane], sbeta[t], &sv[t][j0], lane);
        if (lane < kRowsW) o[(size_t) t * kInner + (size_t) h * kDim + j0 + lane] = pv * scale;
    }

    // ---- hand the head's output to its last block, then store the state
    __threadfence();
    __syncthreads();
    if (c == 0) {
        const unsigned prev = atomicAdd(&head_done[h], 1u);
        last = prev == gridDim.x - 1;
        if (last) head_done[h] = 0;   // ready for the next launch / graph replay
    }
#pragma unroll
    for (int r = 0; r < kRowsW; ++r)
        reinterpret_cast<float4 *>(S_out + ((size_t) h * kDim + j0 + r) * kDim)[lane] = s[r];
    __syncthreads();
    if (!last) return;
    __threadfence();
    float x[T], ss[T];
#pragma unroll
    for (int t = 0; t < T; ++t) { x[t] = __ldcg(o + (size_t) t * kInner + ch); ss[t] = x[t] * x[t]; }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int t = 0; t < T; ++t) ss[t] += __shfl_xor_sync(0xffffffffu, ss[t], off);
    if (lane == 0)
#pragma unroll
        for (int t = 0; t < T; ++t) red[0][t][warp] = ss[t];
    __syncthreads();
    const float nw = norm_w[c];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const size_t idx = (size_t) t * kInner + ch;
        const float mean = (red[0][t][0] + red[0][t][1] + red[0][t][2] + red[0][t][3]) / (float) kDim;
        const float gv = x[t] * rsqrtf(mean + eps) * nw * sigmoidf_(g2[idx]);
        gated[idx] = gv;
        if (gq) {   // as kda_post_kernel
            const float amax = q8_group_amax<32>(fabsf(gv));
            gq[idx] = q8_quant(gv, q8_inv_scale(amax));
            if (lane == 0) gd[idx / 32] = q8_scale(amax);
        }
    }
}

#define KDA_FUSED_ARGS const float * a0, const float * a1, const float * a2, const float * a3, float * a4, \
    const float * a5, const float * a6, const float * a7, const float * a8, const float * a9, const float * a10, \
    const float * a11, float * a12, float * a13, float * a14, float * a15, float * a16, const float * a17, \
    float * a18, float * a19, const float * a20, const float * a21, float a22, float * a23, unsigned * a24, \
    int8_t * a25, float * a26
#define KDA_FUSED_PASS a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, \
    a20, a21, a22, a23, a24, a25, a26

void launch_fused(int T, KDA_FUSED_ARGS, cudaStream_t st) {
    const dim3 grid(kDim / kRowsB, kHeads), block(kWarpsScan * 32);
    switch (T) {
        case 1: kda_decode_fused_kernel<1><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 2: kda_decode_fused_kernel<2><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 3: kda_decode_fused_kernel<3><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 4: kda_decode_fused_kernel<4><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 5: kda_decode_fused_kernel<5><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 6: kda_decode_fused_kernel<6><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 7: kda_decode_fused_kernel<7><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        case 8: kda_decode_fused_kernel<8><<<grid, block, 0, st>>>(KDA_FUSED_PASS); break;
        default: throw std::runtime_error("kda: fused decode needs 1 <= T <= 8");
    }
}
#undef KDA_FUSED_ARGS
#undef KDA_FUSED_PASS

// conv history rollback: out row r = row (n_keep + r) of concat(hist_old[3], raw[T])
__global__ void kda_hist_rollback_kernel(int n_keep, int T, const float * __restrict__ hist_old,
                                         const float * __restrict__ raw, float * __restrict__ hist_out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * kHist * kInner) return;
    const int ch = i % kInner, r = (i / kInner) % kHist, m = i / (kHist * kInner);
    const int idx = n_keep + r;
    hist_out[i] = idx < kHist ? hist_old[(m * kHist + idx) * kInner + ch]
                              : raw[((size_t) m * T + (idx - kHist)) * kInner + ch];
}

// ---------------------------------------------------------------------------------------------------------
// projections

void check_mat(const WeightRef & w, int n_in, int n_out, const char * name) {
    if (!w.data || w.ne[0] != n_in || w.ne[1] != n_out)
        throw std::runtime_error(std::string("kda: weight ") + name + " is not [" + std::to_string(n_in) + "," +
                                 std::to_string(n_out) + "]");
}

// scratch carve-up
struct Carve {
    char * p;
    size_t used = 0;
    explicit Carve(void * base) : p(static_cast<char *>(base)) {}
    template <class T> T * take(size_t n) {
        T * r = p ? reinterpret_cast<T *>(p + used) : nullptr;
        used += al(n * sizeof(T));
        return r;
    }
};

// scratch sizing without the weights: the largest need over the types the engine may pass (Q8_0, Q8_0R, BF16)
bool is_q8_type(int type) { return type == kGgmlQ8_0 || type == kTypeQ8_0R; }

WeightRef shape_ref(int type, int n_in, int n_out) {
    WeightRef w;
    w.type = type;
    w.ne[0] = n_in;
    w.ne[1] = n_out;
    w.nbytes = type == kTypeBf16 ? (uint64_t) n_in * n_out * 2 : (uint64_t) n_in / 32 * 34 * n_out;
    return w;
}

size_t mm_bytes(int T) {
    size_t m = 0;
    for (int type : {kGgmlQ8_0, kTypeQ8_0R, kTypeBf16})
        m = std::max({m, matmul_run_bytes(shape_ref(type, kEmbd, kInner), T),
                      matmul_scratch_bytes(shape_ref(type, kInner, kEmbd), T),
                      matmul_scratch_bytes(shape_ref(type, kRank, kInner), T)});
    return m;
}

// Several projections of the same prepared input: one launch when the types agree (GEMV path), else a loop.
void run_group(const WeightRef * ws, float * const * ys, int n, const MatmulInput & in, void * mm, size_t mm_n,
               cudaStream_t st) {
    bool same = true;
    for (int i = 1; i < n; ++i) same &= ws[i].type == ws[0].type;
    if (same) matmul_run_multi(ws, ys, nullptr, n, in, mm, mm_n, st);
    else for (int i = 0; i < n; ++i) matmul_run(ws[i], in, ys[i], mm, mm_n, st);
}

struct Bufs {
    void * prep, * mm;             // shared prepared x; weight / per-call matmul scratch
    size_t mm_bytes;
    float * raw, * kvg, * beta;    // only when T > kKdaLogT (else in the state log)
    float * q, * fa, * ga, * glin, * braw, * g2, * o, * gated;
    void * chunk_work;
};

Bufs carve(void * scratch, int T, size_t * total) {
    Carve c(scratch);
    Bufs b{};
    b.prep  = c.take<char>(matmul_prepare_bytes(kEmbd, T));
    b.mm_bytes = mm_bytes(T);
    b.mm    = c.take<char>(b.mm_bytes);
    if (T > kKdaLogT) {
        b.raw  = c.take<float>((size_t) 3 * T * kInner);
        b.kvg  = c.take<float>((size_t) 3 * T * kInner);
        b.beta = c.take<float>((size_t) T * kHeads);
    }
    b.q     = c.take<float>((size_t) T * kInner);
    b.fa    = c.take<float>((size_t) T * kRank);
    b.ga    = c.take<float>((size_t) T * kRank);
    b.glin  = c.take<float>((size_t) T * kInner);
    b.braw  = c.take<float>((size_t) T * kHeads);
    b.g2    = c.take<float>((size_t) T * kInner);
    b.o     = c.take<float>((size_t) T * kInner);
    b.gated = c.take<float>((size_t) T * kInner);
    if (T > kKdaLogT) b.chunk_work = c.take<char>(scan_chunked_work_bytes(T));
    if (total) *total = c.used;
    return b;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// state

struct KdaState {
    float * S[2]    = {nullptr, nullptr};   // [64][128][128]
    float * conv[2] = {nullptr, nullptr};   // [3][3][8192]
    float * snap_S = nullptr, * snap_conv = nullptr;
    int cur = 0;
    // log of the last batch when T <= kKdaLogT: raw q,k,v [3][T][8192]; prepared k,v,g [3][T][8192]; beta [T][64]
    float * log_raw = nullptr, * log_kvg = nullptr, * log_beta = nullptr;
    int log_T = -1;
    unsigned * head_done = nullptr;   // per-head block counters of the fused decode kernel (self-resetting)
};

namespace {
constexpr size_t kSFloats    = (size_t) kHeads * kDim * kDim;
constexpr size_t kConvFloats = (size_t) 3 * kHist * kInner;
}

KdaState * kda_state_create(const Geometry & g) {
    if (g.kda_heads != kHeads || g.kda_head_dim != kDim || g.kda_conv != kHist + 1 || g.n_embd != kEmbd)
        throw std::runtime_error("kda: geometry differs from the compiled KDA shape (64 x 128, conv 4, 4096)");
    auto * s = new KdaState();
    const size_t n = 2 * kSFloats + 2 * kConvFloats + kSFloats + kConvFloats +
                     2 * (size_t) 3 * kKdaLogT * kInner + (size_t) kKdaLogT * kHeads;
    float * p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(float)), "cudaMalloc state");
    s->S[0] = p;              p += kSFloats;
    s->S[1] = p;              p += kSFloats;
    s->snap_S = p;            p += kSFloats;
    s->conv[0] = p;           p += kConvFloats;
    s->conv[1] = p;           p += kConvFloats;
    s->snap_conv = p;         p += kConvFloats;
    s->log_raw = p;           p += (size_t) 3 * kKdaLogT * kInner;
    s->log_kvg = p;           p += (size_t) 3 * kKdaLogT * kInner;
    s->log_beta = p;
    ck(cudaMemset(s->S[0], 0, n * sizeof(float)), "cudaMemset state");
    ck(cudaMalloc(&s->head_done, kHeads * sizeof(unsigned)), "cudaMalloc counters");
    ck(cudaMemset(s->head_done, 0, kHeads * sizeof(unsigned)), "cudaMemset counters");
    return s;
}

void kda_state_destroy(KdaState * s) {
    if (!s) return;
    cudaFree(s->S[0]);
    cudaFree(s->head_done);
    delete s;
}

void kda_state_reset(KdaState * s, cudaStream_t st) {
    s->cur = 0;
    s->log_T = -1;
    ck(cudaMemsetAsync(s->S[0], 0, kSFloats * sizeof(float), st), "reset S");
    ck(cudaMemsetAsync(s->conv[0], 0, kConvFloats * sizeof(float), st), "reset conv");
}

void kda_state_snapshot(KdaState * s, cudaStream_t st) {
    ck(cudaMemcpyAsync(s->snap_S, s->S[s->cur], kSFloats * sizeof(float), cudaMemcpyDeviceToDevice, st), "snap S");
    ck(cudaMemcpyAsync(s->snap_conv, s->conv[s->cur], kConvFloats * sizeof(float), cudaMemcpyDeviceToDevice, st),
       "snap conv");
}

void kda_state_restore(KdaState * s, cudaStream_t st) {
    ck(cudaMemcpyAsync(s->S[s->cur], s->snap_S, kSFloats * sizeof(float), cudaMemcpyDeviceToDevice, st), "restore S");
    ck(cudaMemcpyAsync(s->conv[s->cur], s->snap_conv, kConvFloats * sizeof(float), cudaMemcpyDeviceToDevice, st),
       "restore conv");
    s->log_T = -1;
}

void kda_state_rollback(const Geometry &, KdaState * s, int n_keep, cudaStream_t st) {
    const int T = s->log_T;
    if (T < 0) throw std::runtime_error("kda: rollback without a logged batch (T > kKdaLogT or already rolled back)");
    if (n_keep < 0 || n_keep > T) throw std::runtime_error("kda: rollback n_keep out of range");
    s->log_T = -1;
    if (n_keep == T) return;
    const int pre = s->cur ^ 1;
    if (n_keep == 0) { s->cur = pre; return; }
    kda_hist_rollback_kernel<<<(unsigned) ((kConvFloats + 255) / 256), 256, 0, st>>>(n_keep, T, s->conv[pre], s->log_raw,
                                                                                    s->conv[s->cur]);
    const float * lk = s->log_kvg;
    const float * lv = lk + (size_t) T * kInner;
    const float * lg = lv + (size_t) T * kInner;
    kda_scan_seq_kernel<false><<<dim3(kDim / kRowsB, kHeads), kWarpsScan * 32, 0, st>>>(
        n_keep, lk, lk, lv, lg, s->log_beta, s->S[pre], s->S[s->cur], nullptr);
    ck(cudaGetLastError(), "rollback");
}

void kda_state_save(const KdaState * s, void * host, cudaStream_t st) {
    auto * h = static_cast<float *>(host);
    ck(cudaMemcpyAsync(h, s->S[s->cur], kSFloats * sizeof(float), cudaMemcpyDeviceToHost, st), "save S");
    ck(cudaMemcpyAsync(h + kSFloats, s->conv[s->cur], kConvFloats * sizeof(float), cudaMemcpyDeviceToHost, st),
       "save conv");
}

void kda_state_load(KdaState * s, const void * host, cudaStream_t st) {
    const auto * h = static_cast<const float *>(host);
    ck(cudaMemcpyAsync(s->S[s->cur], h, kSFloats * sizeof(float), cudaMemcpyHostToDevice, st), "load S");
    ck(cudaMemcpyAsync(s->conv[s->cur], h + kSFloats, kConvFloats * sizeof(float), cudaMemcpyHostToDevice, st),
       "load conv");
    s->log_T = -1;
}

const float * kda_state_S(const KdaState * s) { return s->S[s->cur]; }
const float * kda_state_conv(const KdaState * s) { return s->conv[s->cur]; }
size_t kda_state_bytes(const Geometry &) { return (kSFloats + kConvFloats) * sizeof(float); }
int kda_state_cur(const KdaState * s) { return s->cur; }

void kda_state_advance(KdaState * s, int T) {
    if (T <= 0) return;
    s->cur ^= 1;
    s->log_T = T <= kKdaLogT ? T : -1;
}

// ---------------------------------------------------------------------------------------------------------

size_t kda_scratch_bytes(const Geometry &, int max_T) {
    size_t total = 0;
    carve(nullptr, max_T, &total);
    return total + 256;
}

namespace kda {

const Views & last_views() { return g_views; }
void set_prefill_mode(int mode) { g_prefill_mode = mode; }
void set_stage_events(cudaEvent_t * ev) { g_stage_ev = ev; }
void set_fused_decode(bool on) { g_fused_decode = on; }
void set_skip_projections(bool on) { g_skip_proj = on; }
void set_rq_variant(int v) { g_rq_variant = v; }

void scan_sequential(int T, const float * q, const float * k, const float * v, const float * g, const float * beta,
                     const float * S_in, float * S_out, float * o, cudaStream_t st) {
    if (T <= kKdaLogT) {   // short batches (rollback replay): more blocks, latency-bound
        const dim3 grid(kDim / kRowsB, kHeads);
        if (o) kda_scan_seq_kernel<true><<<grid, kWarpsScan * 32, 0, st>>>(T, q, k, v, g, beta, S_in, S_out, o);
        else   kda_scan_seq_kernel<false><<<grid, kWarpsScan * 32, 0, st>>>(T, q, k, v, g, beta, S_in, S_out, nullptr);
    } else {
        switch (g_rq_variant) {
            case 1: launch_rq<1, 64>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            case 2: launch_rq<2, 64>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            case 3: launch_rq<2, 32>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            case 4: launch_rq<4, 64>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            case 5: launch_rq<4, 32>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            case 6: launch_rq<4, 128>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
            default: launch_rq<2, 64>(T, q, k, v, g, beta, S_in, S_out, o, st); break;
        }
    }
    ck(cudaGetLastError(), "scan_sequential");
}

}  // namespace kda

void kda_forward(const Geometry & geo, const KdaWeights & w, KdaState * s, const Batch & batch, const float * x,
                 float * out, void * scratch, cudaStream_t st, const MatmulInput * x_in) {
    (void) geo;
    const int T = batch.T;
    if (T <= 0) return;
    check_mat(w.q, kEmbd, kInner, "attn_q");
    check_mat(w.k, kEmbd, kInner, "attn_k");
    check_mat(w.v, kEmbd, kInner, "attn_v");
    check_mat(w.out, kInner, kEmbd, "attn_output");
    check_mat(w.f_a, kEmbd, kRank, "ssm_f_a");
    check_mat(w.f_b, kRank, kInner, "ssm_f_b");
    check_mat(w.g_a, kEmbd, kRank, "ssm_g_a");
    check_mat(w.g_b, kRank, kInner, "ssm_g_b");
    check_mat(w.beta, kEmbd, kHeads, "ssm_beta");

    void * base = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(scratch) + 255) & ~uintptr_t(255));
    Bufs b = carve(base, T, nullptr);
    const bool logged = T <= kKdaLogT;
    float * raw  = logged ? s->log_raw : b.raw;
    float * kvg  = logged ? s->log_kvg : b.kvg;
    float * bet  = logged ? s->log_beta : b.beta;
    float * raw_q = raw, * raw_k = raw + (size_t) T * kInner, * raw_v = raw + (size_t) 2 * T * kInner;
    float * kk = kvg, * vv = kvg + (size_t) T * kInner, * gg = kvg + (size_t) 2 * T * kInner;
    // the output projection's Q8 input is written by the post epilogue into the matmul scratch (the GEMV path
    // needs no weight scratch)
    const bool q8_out = !g_skip_proj && is_q8_type(w.out.type) && matmul_input_is_q8(kInner, T);
    int8_t * gq = q8_out ? static_cast<int8_t *>(b.mm) : nullptr;
    float * gd = q8_out ? reinterpret_cast<float *>(static_cast<char *>(b.mm) + q8_scales_offset(T, kInner)) : nullptr;
    auto out_proj = [&]() {
        if (g_skip_proj) return;
        if (q8_out) matmul_run(w.out, matmul_input_q8(b.gated, kInner, T, b.mm), out, nullptr, 0, st);
        else glm_matmul(w.out, b.gated, out, T, b.mm, b.mm_bytes, st);
    };

    // projections from x
    stage(0, st);
    if (!g_skip_proj) {
    if (x_in && (x_in->n_in != kEmbd || x_in->T != T || x_in->batch != 1))
        throw std::runtime_error("kda: prepared input does not match x [T][4096]");
    const MatmulInput xin = x_in ? *x_in : matmul_prepare(x, kEmbd, T, 1, 0, 0, b.prep, st);
    {
        const WeightRef wq[3] = {w.q, w.k, w.v};
        float * const yq[3] = {raw_q, raw_k, raw_v};
        run_group(wq, yq, 3, xin, b.mm, b.mm_bytes, st);
        const WeightRef wg[3] = {w.f_a, w.g_a, w.beta};
        float * const yg[3] = {b.fa, b.ga, b.braw};
        run_group(wg, yg, 3, xin, b.mm, b.mm_bytes, st);
    }
    glm_matmul(w.f_b, b.fa, b.glin, T, b.mm, b.mm_bytes, st);
    glm_matmul(w.g_b, b.ga, b.g2, T, b.mm, b.mm_bytes, st);
    }
    stage(1, st);
    const int pre = s->cur, nxt = s->cur ^ 1;
    if (logged && g_fused_decode) {
        launch_fused(T, raw_q, raw_k, raw_v, s->conv[pre], s->conv[nxt],
            static_cast<const float *>(w.conv_q.data), static_cast<const float *>(w.conv_k.data),
            static_cast<const float *>(w.conv_v.data), b.glin, static_cast<const float *>(w.dt_bias.data),
            static_cast<const float *>(w.a.data), b.braw, b.q, kk, vv, gg, bet, s->S[pre], s->S[nxt], b.o, b.g2,
            static_cast<const float *>(w.norm.data), geo.rms_eps, b.gated, s->head_done, gq, gd, st);
        ck(cudaGetLastError(), "kda_decode_fused");
        s->cur = nxt;
        s->log_T = T;
        stage(2, st);
        stage(3, st);
        stage(4, st);
        out_proj();
        stage(5, st);
        g_views = Views{T, raw_q, raw_k, raw_v, b.fa, b.ga, b.glin, b.braw, b.g2, b.q, kk, vv, gg, bet, b.o, b.gated};
        return;
    }
    if (T > kKdaLogT)
        kda_prep_rows_kernel<<<(unsigned) ((T * kHeads + 3) / 4), 128, 0, st>>>(
            T, raw_q, raw_k, raw_v, s->conv[pre], s->conv[nxt],
            static_cast<const float *>(w.conv_q.data), static_cast<const float *>(w.conv_k.data),
            static_cast<const float *>(w.conv_v.data), b.glin, static_cast<const float *>(w.dt_bias.data),
            static_cast<const float *>(w.a.data), b.braw, b.q, kk, vv, gg, bet);
    else
    kda_prep_kernel<<<dim3(kHeads, (T + kPrepT - 1) / kPrepT), 128, 0, st>>>(
        T, raw_q, raw_k, raw_v, s->conv[pre], s->conv[nxt],
        static_cast<const float *>(w.conv_q.data), static_cast<const float *>(w.conv_k.data),
        static_cast<const float *>(w.conv_v.data), b.glin, static_cast<const float *>(w.dt_bias.data),
        static_cast<const float *>(w.a.data), b.braw, b.q, kk, vv, gg, bet);
    ck(cudaGetLastError(), "kda_prep");
    stage(2, st);

    if (T > kKdaLogT && g_prefill_mode == 1)
        scan_chunked(T, b.q, kk, vv, gg, bet, s->S[pre], s->S[nxt], b.o, b.chunk_work, st);
    else
        scan_sequential(T, b.q, kk, vv, gg, bet, s->S[pre], s->S[nxt], b.o, st);
    s->cur = nxt;
    s->log_T = logged ? T : -1;
    stage(3, st);

    kda_post_kernel<<<dim3(kHeads, T), 128, 0, st>>>(b.o, b.g2, static_cast<const float *>(w.norm.data), geo.rms_eps,
                                                     b.gated, gq, gd);
    ck(cudaGetLastError(), "kda_post");
    stage(4, st);

    out_proj();
    stage(5, st);

    g_views = Views{T, raw_q, raw_k, raw_v, b.fa, b.ga, b.glin, b.braw, b.g2, b.q, kk, vv, gg, bet, b.o, b.gated};
}

}  // namespace strata::glm
