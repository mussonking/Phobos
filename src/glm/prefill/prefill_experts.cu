// src/glm/prefill/prefill_experts.cu - see prefill_experts.hpp.
#include "prefill_experts.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/device/device_radix_sort.cuh>

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_experts: ") + what + ": " + cudaGetErrorString(e));
}
void cb(cublasStatus_t s, const char * what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("prefill_experts: cuBLAS ") + what + " " +
                                                             std::to_string((int) s));
}
size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

constexpr int kXsBlock = 136;   // IQ4_XS: d f16, scales_h u16, scales_l[4], qs[128] per 256 values
constexpr int kNlBlock = 18;    // IQ4_NL: d f16, qs[16] per 32 values
__device__ __constant__ int8_t kCodebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__global__ void count_kernel(const int32_t * __restrict__ ids, int jobs, int n_expert, int * counts) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < jobs) {
        const int e = ids[j];
        if (e >= 0 && e < n_expert) atomicAdd(&counts[e], 1);
    }
}

__global__ void iota_kernel(int * v, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) v[j] = j;
}

// Sorted position of every job (inverse permutation) and the gathered F16 activation rows in sorted order.
__global__ void gather_kernel(const float * __restrict__ x, const int * __restrict__ sorted_job, int jobs, int K,
                              int D, __half * __restrict__ xs, int * __restrict__ pos) {
    const int i = blockIdx.x;
    if (i >= jobs) return;
    const int job = sorted_job[i];
    if (threadIdx.x == 0) pos[job] = i;
    const float4 * src = reinterpret_cast<const float4 *>(x + (size_t) (job / K) * D);
    __half2 * dst = reinterpret_cast<__half2 *>(xs + (size_t) i * D);
    for (int c = threadIdx.x; c < D / 4; c += blockDim.x) {
        const float4 v = src[c];
        dst[2 * c] = __floats2half2_rn(v.x, v.y);
        dst[2 * c + 1] = __floats2half2_rn(v.z, v.w);
    }
}

// One thread per 32-value sub-block (codebook in shared memory: a divergent __constant__ read serializes the warp;
// 16-byte stores). gate | up: 2 * ff rows of n_embd IQ4_XS values -> wgu [2 ff][n_embd]; down: n_embd rows of ff
// IQ4_NL values -> wd [n_embd][ff]. blockIdx.y = expert slot in the group.
__global__ void dequant_kernel(const ExpertRef * __restrict__ table, const int * __restrict__ experts, int D, int F,
                               __half * __restrict__ wgu, __half * __restrict__ wd) {
    __shared__ float cb[16];
    if (threadIdx.x < 16) cb[threadIdx.x] = (float) kCodebook[threadIdx.x];
    __syncthreads();
    const ExpertRef r = table[experts[blockIdx.y]];
    const int64_t n_gu = (int64_t) 2 * F * (D / 32), n_d = (int64_t) D * (F / 32);
    const int64_t s = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    __half * out;
    float dl;
    const uint8_t * qs;
    if (s < n_gu) {
        const int row = (int) (s / (D / 32)), sub = (int) (s % (D / 32));
        const size_t row_bytes = (size_t) (D / 256) * kXsBlock;
        const uint8_t * base = static_cast<const uint8_t *>(row < F || !r.up ? r.gate_up : r.up);
        const int rr = (row < F || !r.up) ? row : row - F;
        const uint8_t * b = base + (size_t) rr * row_bytes + (size_t) (sub / 8) * kXsBlock;
        const int ib = sub % 8;
        const int high = int(b[2]) | (int(b[3]) << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((high >> (2 * ib)) & 3) << 4);
        dl = __half2float(*reinterpret_cast<const __half *>(b)) * float(ls - 32);
        qs = b + 8 + ib * 16;
        out = wgu + (size_t) blockIdx.y * 2 * F * D + (size_t) row * D + (size_t) sub * 32;
    } else if (s < n_gu + n_d) {
        const int64_t t = s - n_gu;
        const int row = (int) (t / (F / 32)), sub = (int) (t % (F / 32));
        const uint8_t * b = static_cast<const uint8_t *>(r.down) + (size_t) row * (F / 32) * kNlBlock +
                            (size_t) sub * kNlBlock;
        dl = __half2float(*reinterpret_cast<const __half *>(b));
        qs = b + 2;
        out = wd + (size_t) blockIdx.y * D * F + (size_t) row * F + (size_t) sub * 32;
    } else return;
    uint8_t q[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) q[j] = qs[j];
    __half2 v[16];
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        v[j / 2] = __floats2half2_rn(dl * cb[q[j] & 15], dl * cb[q[j + 1] & 15]);
        v[8 + j / 2] = __floats2half2_rn(dl * cb[q[j] >> 4], dl * cb[q[j + 1] >> 4]);
    }
    uint4 * o4 = reinterpret_cast<uint4 *>(out);
    const uint4 * v4 = reinterpret_cast<const uint4 *>(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) o4[j] = v4[j];
}

// a[i][j] = silu(min(g, c)) * clamp(u, -c, c) from h[i] = gate | up (F32), as F16 for the down GEMM.
__global__ void swiglu_kernel(const float * __restrict__ h, int rows, int F, float c, __half * __restrict__ a) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) rows * F) return;
    const int64_t r = i / F, j = i % F;
    const float g = fminf(h[r * 2 * F + j], c);
    const float u = fmaxf(-c, fminf(h[r * 2 * F + F + j], c));
    a[i] = __float2half_rn((g / (1.0f + expf(-g))) * u);
}

// out[t] = sum_k w[t][k] * o[pos[t K + k]] in slot order; experts without weights in the table contribute nothing.
__global__ void combine_kernel(const float * __restrict__ o, const int * __restrict__ pos,
                               const int32_t * __restrict__ ids, const float * __restrict__ w,
                               const ExpertRef * __restrict__ table, int n_expert, int K, int D, float * out) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int e = ids[t * K + k];
            if (e < 0 || e >= n_expert || !table[e].gate_up) continue;
            v += w[t * K + k] * o[(size_t) pos[t * K + k] * D + d];
        }
        out[(size_t) t * D + d] = v;
    }
}

}  // namespace

struct PrefillExperts {
    Geometry g;
    int max_T = 0, group = 0, group_max = 0, row_cap = 0;
    cublasHandle_t blas = nullptr;
    void * arena = nullptr;
    size_t bytes = 0;
    int * counts = nullptr;      // [n_expert]
    int * iota = nullptr;        // [jobs]
    int * keys = nullptr;        // sorted expert ids [jobs]
    int * sorted = nullptr;      // sorted job index [jobs]
    int * pos = nullptr;         // job -> sorted position [jobs]
    int * order = nullptr;       // active experts in processing order [n_expert]
    void * sort_tmp = nullptr;
    size_t sort_tmp_bytes = 0;
    __half * xs = nullptr;       // [jobs][D]
    float * o = nullptr;         // [jobs][D]
    float * h = nullptr;         // [row_cap][2F]
    __half * a = nullptr;        // [row_cap][F]
    __half * wgu = nullptr;      // [group][2F][D]
    __half * wd = nullptr;       // [group][D][F]
    int * h_counts = nullptr;    // pinned
    int * h_order = nullptr;     // pinned
    bool timing = false;
    cudaEvent_t ev[8] = {};
    PrefillExpertsTiming last;
};

namespace {

// Device buffers for max_T tokens: sizes, and with p the pointers to assign (row_cap and sort_tmp_bytes set in p).
struct PeBuf { void ** ptr; size_t bytes; };
std::vector<PeBuf> experts_layout(const Geometry & g, int max_T, int group, PrefillExperts * p) {
    const int D = g.n_embd, F = g.ffn_expert, NE = g.n_expert;
    const size_t jobs = (size_t) max_T * g.n_expert_used;
    // a group never holds more rows than this (the host splits groups by rows); one expert alone is <= max_T
    p->row_cap = std::max(max_T, (int) std::min<size_t>(jobs, (size_t) 16384));
    cub::DeviceRadixSort::SortPairs(nullptr, p->sort_tmp_bytes, (const int *) nullptr, (int *) nullptr,
                                    (const int *) nullptr, (int *) nullptr, (int) jobs, 0, 16);
    return {
        {(void **) &p->counts, (size_t) NE * 4},
        {(void **) &p->iota, jobs * 4},
        {(void **) &p->keys, jobs * 4},
        {(void **) &p->sorted, jobs * 4},
        {(void **) &p->pos, jobs * 4},
        {(void **) &p->order, (size_t) NE * 4},
        {&p->sort_tmp, p->sort_tmp_bytes},
        {(void **) &p->xs, jobs * D * 2},
        {(void **) &p->o, jobs * D * 4},
        {(void **) &p->h, (size_t) p->row_cap * 2 * F * 4},
        {(void **) &p->a, (size_t) p->row_cap * F * 2},
        {(void **) &p->wgu, (size_t) group * 2 * F * D * 2},
        {(void **) &p->wd, (size_t) group * D * F * 2},
    };
}

size_t layout_bytes(const std::vector<PeBuf> & bufs) {
    size_t total = 0;
    for (const auto & b : bufs) total += align256(b.bytes);
    return total;
}

}  // namespace

size_t prefill_experts_bytes(const Geometry & g, int max_T, int group) {
    PrefillExperts p;
    return layout_bytes(experts_layout(g, max_T, group, &p));
}

PrefillExperts * prefill_experts_create_unbound(const Geometry & g, int group) {
    if (group < 1) throw std::runtime_error("prefill_experts: bad group");
    auto * p = new PrefillExperts();
    p->g = g;
    p->group = p->group_max = group;
    ck(cudaMallocHost(&p->h_counts, g.n_expert * 4), "pinned counts");
    ck(cudaMallocHost(&p->h_order, g.n_expert * 4), "pinned order");
    cb(cublasCreate(&p->blas), "create");
    cb(cublasSetMathMode(p->blas, CUBLAS_TENSOR_OP_MATH), "math mode");
    for (auto & e : p->ev) ck(cudaEventCreate(&e), "event");
    return p;
}

void prefill_experts_bind(PrefillExperts * p, int max_T, void * mem, cudaStream_t st, int group) {
    if (p->arena) {
        ck(cudaDeviceSynchronize(), "rebind");
        cudaFree(p->arena);
        p->arena = nullptr;
    }
    p->max_T = 0;
    p->bytes = 0;
    if (!mem) return;
    if (max_T < 1) throw std::runtime_error("prefill_experts: bad max_T");
    if (group > p->group_max) throw std::runtime_error("prefill_experts: group above the created one");
    p->group = group > 0 ? group : p->group_max;
    const auto bufs = experts_layout(p->g, max_T, p->group, p);
    char * c = static_cast<char *>(mem);
    for (const auto & b : bufs) {
        *b.ptr = c;
        c += align256(b.bytes);
    }
    p->max_T = max_T;
    p->bytes = layout_bytes(bufs);
    const size_t jobs = (size_t) max_T * p->g.n_expert_used;
    iota_kernel<<<(unsigned) ((jobs + 255) / 256), 256, 0, st>>>(p->iota, (int) jobs);
    ck(cudaGetLastError(), "iota");
}

PrefillExperts * prefill_experts_create(const Geometry & g, int max_T, int group) {
    if (max_T < 1) throw std::runtime_error("prefill_experts: bad sizes");
    PrefillExperts * p = prefill_experts_create_unbound(g, group);
    void * mem = nullptr;
    ck(cudaMalloc(&mem, prefill_experts_bytes(g, max_T, group)), "arena");
    prefill_experts_bind(p, max_T, mem, 0);
    p->arena = mem;
    ck(cudaDeviceSynchronize(), "create");
    return p;
}

void prefill_experts_destroy(PrefillExperts * p) {
    if (!p) return;
    cublasDestroy(p->blas);
    for (auto & e : p->ev) cudaEventDestroy(e);
    if (p->arena) cudaFree(p->arena);
    cudaFreeHost(p->h_counts);
    cudaFreeHost(p->h_order);
    delete p;
}

size_t prefill_experts_device_bytes(const PrefillExperts * p) { return p->bytes; }
void prefill_experts_set_timing(PrefillExperts * p, bool on) { p->timing = on; }
PrefillExpertsTiming prefill_experts_timing(PrefillExperts * p) { return p->last; }

namespace {

// Sort + gather, then the planned groups and the combine. h_counts / h_order: host; order entries index h_order.
void run_planned(PrefillExperts * p, int T, const float * x, const int32_t * ids, const float * w,
                 const ExpertRef * table, const int * h_counts, const int * h_order, int n_act,
                 const std::function<void(int)> & before, const std::function<void(int)> & after, float * out,
                 cudaStream_t st, bool sorted_already) {
    const Geometry & g = p->g;
    const int D = g.n_embd, F = g.ffn_expert, NE = g.n_expert, K = g.n_expert_used;
    const int jobs = T * K;
    int end_bit = 1;
    while ((1 << end_bit) < NE) ++end_bit;
    auto mark = [&](int i) { if (p->timing) ck(cudaEventRecord(p->ev[i], st), "event"); };
    float dq = 0, gu = 0, sw = 0, dn = 0;
    cudaEvent_t e0 = p->ev[6], e1 = p->ev[7];
    auto lap = [&](float & acc, auto && f) {
        if (!p->timing) { f(); return; }
        ck(cudaEventRecord(e0, st), "event");
        f();
        ck(cudaEventRecord(e1, st), "event");
        ck(cudaEventSynchronize(e1), "event");
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        acc += ms;
    };
    if (!sorted_already) {
        mark(0);
        size_t tmp = p->sort_tmp_bytes;
        ck(cub::DeviceRadixSort::SortPairs(p->sort_tmp, tmp, ids, p->keys, p->iota, p->sorted, jobs, 0, end_bit, st),
           "sort");
        mark(1);
        gather_kernel<<<jobs, 256, 0, st>>>(x, p->sorted, jobs, K, D, p->xs, p->pos);
        ck(cudaGetLastError(), "gather");
        mark(2);
    }

    // sorted rows are in expert id order: expert e owns rows [off[e], off[e] + counts[e])
    std::vector<int> off(NE + 1, 0);
    for (int e = 0; e < NE; ++e) off[e + 1] = off[e] + h_counts[e];
    if (h_order != p->h_order) {
        ck(cudaEventSynchronize(p->ev[5]), "order reuse");   // the previous run's order upload has read h_order
        std::memcpy(p->h_order, h_order, (size_t) n_act * 4);
    }
    ck(cudaMemcpyAsync(p->order, p->h_order, (size_t) n_act * 4, cudaMemcpyHostToDevice, st), "order upload");
    ck(cudaEventRecord(p->ev[5], st), "order event");
    cb(cublasSetStream(p->blas, st), "stream");
    const float one = 1.0f, zero = 0.0f;
    for (int i0 = 0; i0 < n_act;) {
        // a group: consecutive experts of the order, rows contiguous per expert but not across experts
        int i1 = i0, rows = 0;
        while (i1 < n_act && i1 - i0 < p->group && rows + h_counts[p->h_order[i1]] <= p->row_cap)
            rows += h_counts[p->h_order[i1++]];
        if (before)
            for (int i = i0; i < i1; ++i) before(i);
        const int64_t per = (int64_t) 2 * F * (D / 32) + (int64_t) D * (F / 32);
        lap(dq, [&] {
            dequant_kernel<<<dim3((unsigned) ((per + 255) / 256), i1 - i0), 256, 0, st>>>(table, p->order + i0, D, F,
                                                                                       p->wgu, p->wd);
            ck(cudaGetLastError(), "dequant");
        });
        if (after)
            for (int i = i0; i < i1; ++i) after(i);
        // gate | up: h[r][2F] = xs[r][D] . wgu^T (column-major: C^T = W^T' x^T); h / a rows packed per group
        lap(gu, [&] {
            int r = 0;
            for (int i = i0; i < i1; ++i) {
                const int e = p->h_order[i], n = h_counts[e];
                cb(cublasGemmEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * F, n, D, &one,
                                p->wgu + (size_t) (i - i0) * 2 * F * D, CUDA_R_16F, D, p->xs + (size_t) off[e] * D,
                                CUDA_R_16F, D, &zero, p->h + (size_t) r * 2 * F, CUDA_R_32F, 2 * F,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "gate_up");
                r += n;
            }
        });
        lap(sw, [&] {
            swiglu_kernel<<<(unsigned) (((int64_t) rows * F + 255) / 256), 256, 0, st>>>(p->h, rows, F, g.swiglu_clamp,
                                                                                        p->a);
            ck(cudaGetLastError(), "swiglu");
        });
        lap(dn, [&] {
            int r = 0;
            for (int i = i0; i < i1; ++i) {
                const int e = p->h_order[i], n = h_counts[e];
                cb(cublasGemmEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, D, n, F, &one, p->wd + (size_t) (i - i0) * D * F,
                                CUDA_R_16F, F, p->a + (size_t) r * F, CUDA_R_16F, F, &zero,
                                p->o + (size_t) off[e] * D, CUDA_R_32F, D, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "down");
                r += n;
            }
        });
        i0 = i1;
    }
    mark(3);
    combine_kernel<<<T, 256, 0, st>>>(p->o, p->pos, ids, w, table, NE, K, D, out);
    ck(cudaGetLastError(), "combine");
    mark(4);
    if (p->timing) {
        ck(cudaEventSynchronize(p->ev[4]), "timing");
        PrefillExpertsTiming t;
        if (!sorted_already) {
            cudaEventElapsedTime(&t.sort, p->ev[0], p->ev[1]);
            cudaEventElapsedTime(&t.gather, p->ev[1], p->ev[2]);
        }
        cudaEventElapsedTime(&t.combine, p->ev[3], p->ev[4]);
        t.dequant = dq;
        t.gemm_gu = gu;
        t.swiglu = sw;
        t.gemm_down = dn;
        t.active_experts = n_act;
        p->last = t;
    }
}

}  // namespace

void prefill_experts_run(PrefillExperts * p, int T, const float * x, const int32_t * ids, const float * w,
                         const ExpertRef * table, float * out, cudaStream_t st) {
    const Geometry & g = p->g;
    const int D = g.n_embd, NE = g.n_expert, K = g.n_expert_used;
    if (T < 1 || T > p->max_T) throw std::runtime_error("prefill_experts_run: T out of range");
    const int jobs = T * K;
    int end_bit = 1;
    while ((1 << end_bit) < NE) ++end_bit;
    // routing sort and counts (the host reads the counts: every active expert runs, in id order)
    if (p->timing) ck(cudaEventRecord(p->ev[0], st), "event");
    ck(cudaMemsetAsync(p->counts, 0, NE * 4, st), "counts");
    count_kernel<<<(jobs + 255) / 256, 256, 0, st>>>(ids, jobs, NE, p->counts);
    size_t tmp = p->sort_tmp_bytes;
    ck(cub::DeviceRadixSort::SortPairs(p->sort_tmp, tmp, ids, p->keys, p->iota, p->sorted, jobs, 0, end_bit, st),
       "sort");
    ck(cudaMemcpyAsync(p->h_counts, p->counts, NE * 4, cudaMemcpyDeviceToHost, st), "counts download");
    if (p->timing) ck(cudaEventRecord(p->ev[1], st), "event");
    gather_kernel<<<jobs, 256, 0, st>>>(x, p->sorted, jobs, K, D, p->xs, p->pos);
    ck(cudaGetLastError(), "gather");
    if (p->timing) ck(cudaEventRecord(p->ev[2], st), "event");
    ck(cudaStreamSynchronize(st), "counts");
    int n_act = 0;
    for (int e = 0; e < NE; ++e)
        if (p->h_counts[e] > 0) p->h_order[n_act++] = e;
    run_planned(p, T, x, ids, w, table, p->h_counts, p->h_order, n_act, {}, {}, out, st, true);
    if (p->timing) {
        cudaEventElapsedTime(&p->last.sort, p->ev[0], p->ev[1]);
        cudaEventElapsedTime(&p->last.gather, p->ev[1], p->ev[2]);
    }
}

void prefill_experts_run(PrefillExperts * p, int T, const float * x, const int32_t * ids, const float * w,
                         const ExpertRef * table, const PrefillExpertsPlan & plan, float * out, cudaStream_t st) {
    if (T < 1 || T > p->max_T) throw std::runtime_error("prefill_experts_run: T out of range");
    if (plan.n_order < 0 || plan.n_order > p->g.n_expert) throw std::runtime_error("prefill_experts_run: bad plan");
    for (int i = 0; i < plan.n_order; ++i)
        if (plan.counts[plan.order[i]] > p->max_T) throw std::runtime_error("prefill_experts_run: bad counts");
    run_planned(p, T, x, ids, w, table, plan.counts, plan.order, plan.n_order, plan.before, plan.after, out, st,
                false);
}

}  // namespace strata::glm
