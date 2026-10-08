// IQ4 cache hits with F32 activations; no allocation or host routing reads during capture.
#include "strata/glm/experts.hpp"

#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>

namespace strata::glm {
namespace {
constexpr int warp = 32, warps = 4, batch = 4;
constexpr int iq4_xs = 23, iq4_nl = 20;
__device__ __constant__ int codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ float weight_xs(const uint8_t* row, int k, int table_value) {
    const uint8_t* b = row + (k / 256) * 136;
    const int sub = (k % 256) / 32, lane = k % 32;
    const int high = int(b[2]) | (int(b[3]) << 8);
    const int scale = ((b[4 + sub / 2] >> (4 * (sub % 2))) & 15) |
                      (((high >> (2 * sub)) & 3) << 4);
    const int q = b[8 + sub * 16 + lane % 16];
    const int value = __shfl_sync(0xffffffff, table_value, (q >> (lane / 16 * 4)) & 15);
    return (half_at(b) * float(scale - 32)) * float(value);
}
__device__ float weight_nl(const uint8_t* row, int k, int table_value) {
    const uint8_t* b = row + (k / 32) * 18;
    const int q = b[2 + k % 16];
    const int value = __shfl_sync(0xffffffff, table_value, (q >> ((k % 32) / 16 * 4)) & 15);
    return half_at(b) * float(value);
}
__device__ float sum_warp(float v) {
    for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}
__device__ bool live(const ExpertRef& ref, int id, int ne) {
    return id >= 0 && id < ne && ref.gate_up && ref.down &&
           ref.type_gate_up == iq4_xs && ref.type_down == iq4_nl &&
           (ref.type_up < 0 || ref.type_up == iq4_xs);
}

__global__ void gate_up_decode(int hidden, int ff, int top, int ne, float clamp,
                               const float* x, const int* ids, const ExpertRef* slots, float* h) {
    const int job = blockIdx.y, row = blockIdx.x * warps + threadIdx.x / warp;
    const int lane = threadIdx.x % warp;
    if (row >= ff) return;
    // Keep the codebook distributed in registers; shuffles avoid per-weight constant serialization.
    const int table_value = codebook[lane & 15];
    const ExpertRef ref = slots[job];
    float g = 0, u = 0;
    if (live(ref, ids[job], ne)) {
        const size_t stride = size_t(hidden / 256) * 136;
        const auto* base = static_cast<const uint8_t*>(ref.gate_up);
        const auto* gate = base + row * stride;
        const auto* up = (ref.up ? static_cast<const uint8_t*>(ref.up) : base + ff * stride) + row * stride;
        const float* input = x + size_t(job / top) * hidden;
        for (int k = lane; k < hidden; k += warp) {
            g = fmaf(weight_xs(gate, k, table_value), input[k], g);
            u = fmaf(weight_xs(up, k, table_value), input[k], u);
        }
    }
    g = sum_warp(g); u = sum_warp(u);
    if (lane == 0) {
        g = fminf(g, clamp); u = fmaxf(-clamp, fminf(u, clamp));
        h[size_t(job) * ff + row] = (g / (1.0f + expf(-g))) * u;
    }
}

// One bounded block per expert, not a host launch per token. Group order is stable.
__global__ void group_jobs(int jobs, int ne, const int* ids, const ExpertRef* slots,
                           int* counts, int* grouped) {
    const int e = blockIdx.x;
    if (threadIdx.x != 0 || e >= ne) return;
    int n = 0;
    for (int j = 0; j < jobs; ++j)
        if (ids[j] == e && live(slots[j], e, ne)) grouped[size_t(e) * jobs + n++] = j;
    counts[e] = n;
}

// Bounded grouped F32 GEMM baseline: four activations share a decoded weight.
// This is not tensor-core MMQ; repeated small groups and low occupancy can limit prefill throughput.
template<bool Down>
__global__ void grouped_product(int hidden, int ff, int top, int jobs, float clamp,
                                const float* x, const ExpertRef* slots, const int* counts,
                                const int* grouped, float* dst) {
    const int e = blockIdx.y, row = blockIdx.x * warps + threadIdx.x / warp;
    const int lane = threadIdx.x % warp, rows = Down ? hidden : ff, cols = Down ? ff : hidden;
    if (row >= rows) return;
    const int table_value = codebook[lane & 15];
    const int n = counts[e];
    for (int first = 0; first < n; first += batch) {
        int js[batch]; const uint8_t* a[batch]; const uint8_t* up[batch]; const float* inputs[batch];
        float sums[batch] = {}, ups[batch] = {};
        const int nb = min(batch, n - first);
        for (int b = 0; b < nb; ++b) {
            js[b] = grouped[size_t(e) * jobs + first + b];
            const ExpertRef ref = slots[js[b]];
            const size_t stride = Down ? size_t(cols / 32) * 18 : size_t(cols / 256) * 136;
            a[b] = static_cast<const uint8_t*>(Down ? ref.down : ref.gate_up) + row * stride;
            if (!Down) {
                const auto* base = ref.up ? static_cast<const uint8_t*>(ref.up) :
                                           static_cast<const uint8_t*>(ref.gate_up) + size_t(ff) * stride;
                up[b] = base + row * stride;
            }
            inputs[b] = x + size_t(Down ? js[b] : js[b] / top) * cols;
        }
        for (int k = lane; k < cols; k += warp) {
            const float v0 = Down ? weight_nl(a[0], k, table_value) : weight_xs(a[0], k, table_value);
            const float u0 = Down ? 0 : weight_xs(up[0], k, table_value);
            for (int b = 0; b < nb; ++b) {
                const float v = a[b] == a[0] ? v0 : (Down ? weight_nl(a[b], k, table_value) : weight_xs(a[b], k, table_value));
                sums[b] = fmaf(v, inputs[b][k], sums[b]);
                if (!Down) {
                    const float u = up[b] == up[0] ? u0 : weight_xs(up[b], k, table_value);
                    ups[b] = fmaf(u, inputs[b][k], ups[b]);
                }
            }
        }
        for (int b = 0; b < nb; ++b) {
            float v = sum_warp(sums[b]);
            const float u = sum_warp(ups[b]);
            if (!Down) {
                v = fminf(v, clamp);
                v = (v / (1.0f + expf(-v))) * fmaxf(-clamp, fminf(u, clamp));
            }
            if (lane == 0) dst[size_t(js[b]) * rows + row] = v;
        }
    }
}

__global__ void down_decode(int hidden, int ff, int ne, const int* ids,
                            const ExpertRef* slots, const float* h, float* result) {
    const int job = blockIdx.y, row = blockIdx.x * warps + threadIdx.x / warp;
    const int lane = threadIdx.x % warp;
    if (row >= hidden) return;
    const int table_value = codebook[lane & 15];
    float v = 0;
    const ExpertRef ref = slots[job];
    if (live(ref, ids[job], ne)) {
        const auto* a = static_cast<const uint8_t*>(ref.down) + size_t(row) * (ff / 32) * 18;
        for (int k = lane; k < ff; k += warp) v = fmaf(weight_nl(a, k, table_value), h[size_t(job) * ff + k], v);
    }
    v = sum_warp(v);
    if (lane == 0) result[size_t(job) * hidden + row] = v;
}
__global__ void combine(int T, int hidden, int top, int ne, const int* ids,
                        const float* weights, const ExpertRef* slots, const float* result, float* out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * hidden) return;
    const int row = i % hidden, t = i / hidden;
    float v = out[i];
    for (int s = 0; s < top; ++s) {
        const int j = t * top + s;
        if (live(slots[j], ids[j], ne)) v += result[size_t(j) * hidden + row] * weights[j];
    }
    out[i] = v;
}
void validate(const Geometry& g, int T) {
    if (T < 0 || T > 4096 || g.n_embd <= 0 || g.n_embd % 256 ||
        g.ffn_expert <= 0 || g.ffn_expert % 32 || g.n_expert_used <= 0 ||
        g.n_expert_used > 32 || g.n_expert <= 0 || g.n_expert > 1024 ||
        !(g.swiglu_clamp > 0)) throw std::invalid_argument("invalid IQ4 expert geometry/token count");
}
} // namespace

// Exact formula: J*(FF+H)*4 bytes plus (NE + NE*J)*4 bytes for T>8, J=T*K.
// cudaMalloc alignment suffices. Scratch/output/input/slots must be disjoint and device accessible.
size_t experts_gpu_scratch_bytes(const Geometry& g, int T) {
    validate(g, T);
    const size_t jobs = size_t(T) * g.n_expert_used;
    return jobs * (size_t(g.ffn_expert) + g.n_embd) * sizeof(float) +
           (T > 8 ? (size_t(g.n_expert) * (jobs + 1)) * sizeof(int) : 0);
}
void experts_gpu(const Geometry& g, int T, const float* x, const int* ids, const float* w,
                 const ExpertRef* slots, float* out, void* scratch, cudaStream_t stream) {
    validate(g, T);
    if (!T) return;
    if (!x || !ids || !w || !slots || !out || !scratch)
        throw std::invalid_argument("null IQ4 expert device buffer");
    const int jobs = T * g.n_expert_used;
    float* h = static_cast<float*>(scratch);
    float* result = h + size_t(jobs) * g.ffn_expert;
    if (T <= 8) {
        gate_up_decode<<<dim3((g.ffn_expert + warps - 1) / warps, jobs), warp * warps, 0, stream>>>(
            g.n_embd, g.ffn_expert, g.n_expert_used, g.n_expert, g.swiglu_clamp, x, ids, slots, h);
        down_decode<<<dim3((g.n_embd + warps - 1) / warps, jobs), warp * warps, 0, stream>>>(
            g.n_embd, g.ffn_expert, g.n_expert, ids, slots, h, result);
    } else {
        int* counts = reinterpret_cast<int*>(result + size_t(jobs) * g.n_embd);
        int* grouped = counts + g.n_expert;
        group_jobs<<<g.n_expert, 1, 0, stream>>>(jobs, g.n_expert, ids, slots, counts, grouped);
        grouped_product<false><<<dim3((g.ffn_expert + warps - 1) / warps, g.n_expert), warp * warps, 0, stream>>>(
            g.n_embd, g.ffn_expert, g.n_expert_used, jobs, g.swiglu_clamp, x, slots, counts, grouped, h);
        grouped_product<true><<<dim3((g.n_embd + warps - 1) / warps, g.n_expert), warp * warps, 0, stream>>>(
            g.n_embd, g.ffn_expert, g.n_expert_used, jobs, g.swiglu_clamp, h, slots, counts, grouped, result);
    }
    combine<<<(size_t(T) * g.n_embd + 255) / 256, 256, 0, stream>>>(
        T, g.n_embd, g.n_expert_used, g.n_expert, ids, w, slots, result, out);
}
void experts_gpu_prefill(const Geometry& g, int T, const float* x, const int* ids, const float* w,
                         const ExpertRef* slots, float* out, void* scratch, cudaStream_t stream) {
    experts_gpu(g, T, x, ids, w, slots, out, scratch, stream);
}
} // namespace strata::glm
