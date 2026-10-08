// prefill_bench: the large-T routed experts of one layer (prefill_experts) on synthetic IQ4 experts with random
// routing - accuracy against pack's F32 experts_gpu (T <= 4096) and time per layer.
//
//   glm_prefill_bench experts [--T 8192] [--group 16] [--ref 1] [--reps 3]
//   glm_prefill_bench cpu [--U 64] [--counts 1,2,4,8,16,32,64,128] [--threads 48] [--check 1]
//     CPU share (prefill_cpu, pack's VNNI rows) on U synthetic experts in host memory laid out as the arena (32-row
//     chunks alternating NUMA nodes), T = U * count / 8 tokens routed to 8 of the U experts each: ms per layer, per
//     job and per expert; --check compares with pack's F32 experts_gpu_prefill on the same blobs.
//   glm_prefill_bench cpu-exact
//     prefill_cpu's kernel (quantize, unpack, VNNI tile) on 32 IQ4_XS / IQ4_NL rows against an independent reference
//     on the SAME int8 activations: GGUF decoded as ggml does, integer dot products exact (int64), scales in double.
//     Stress cases: sub-scales 0 / 63, codes 0 / 15, negative and extreme d, zero groups, single spikes, saturated
//     and tiny activations. Error = |kernel - ref| / sum of |group terms| (float rounding is ~1e-7). Cases where a
//     weight scale x activation scale product falls below FLT_MIN (f16 subnormal d x 1e-36 activations) are reported
//     apart: float32 cannot hold it, in this kernel as in any F32 path. Every case also runs on R8 row groups (the
//     reference repack) through the R8 unpack: outputs bit-identical to the GGUF rows.
//   glm_prefill_bench swiglu-exact [--threads 16]
//     prefill_cpu's vector SwiGLU bitwise against expert_swiglu_clamp on every float gate.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "iq4_cpu.hpp"
#include "pool.hpp"
#include "prefill_cpu.hpp"
#include "prefill_experts.hpp"
#include "strata/glm/experts.hpp"
#include "strata/glm/geometry.hpp"

using namespace strata::glm;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::string arg(int argc, char ** argv, const char * key, const std::string & def) {
    for (int i = 2; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

__host__ __device__ inline uint32_t hash32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return (uint32_t) x;
}

// Random IQ4 bytes with sane block scales: XS blocks (136 B) in [0, xs_bytes), NL blocks (18 B) after.
// Byte i of a blob (host or device: the same bytes).
__host__ __device__ inline uint8_t blob_byte(size_t i, size_t xs_bytes, uint64_t seed) {
    const uint32_t r = hash32(seed * 0x9e3779b97f4a7c15ull + i);
    uint8_t v = (uint8_t) r;
    if (i < xs_bytes) {
        const size_t o = i % 136;
        if (o < 2) {   // d: f16 in [0.002, 0.004)
            const __half d = __float2half(0.002f + 0.002f * ((hash32(seed + (i - o)) & 1023) / 1024.0f));
            v = reinterpret_cast<const uint8_t *>(&d)[o];
        }
    } else {
        const size_t o = (i - xs_bytes) % 18;
        if (o < 2) {   // d: f16 in [0.02, 0.04)
            const __half d = __float2half(0.02f + 0.02f * ((hash32(seed + (i - xs_bytes - o)) & 1023) / 1024.0f));
            v = reinterpret_cast<const uint8_t *>(&d)[o];
        }
    }
    return v;
}
__global__ void fill_blob(uint8_t * blob, size_t xs_bytes, size_t nl_bytes, uint64_t seed) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < xs_bytes + nl_bytes) blob[i] = blob_byte(i, xs_bytes, seed);
}

__global__ void expand_slots(const int32_t * ids, const ExpertRef * table, int jobs, ExpertRef * slots) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < jobs) slots[j] = table[ids[j]];
}

int run_experts(int argc, char ** argv) {
    const int T = std::atoi(arg(argc, argv, "--T", "8192").c_str());
    const int group = std::atoi(arg(argc, argv, "--group", "16").c_str());
    const bool ref = arg(argc, argv, "--ref", T <= 4096 ? "1" : "0") != "0";
    const int reps = std::atoi(arg(argc, argv, "--reps", "3").c_str());
    const Geometry g;
    const int D = g.n_embd, F = g.ffn_expert, NE = g.n_expert, K = g.n_expert_used;
    const size_t xs_row = (size_t) (D / 256) * 136, nl_row = (size_t) (F / 32) * 18;
    const size_t gu_bytes = 2 * (size_t) F * xs_row, d_bytes = (size_t) D * nl_row, blob = gu_bytes + d_bytes;
    std::printf("experts of one layer: T %d, %d experts x %.2f MB IQ4, group %d\n", T, NE, blob / 1e6, group);

    uint8_t * blobs;
    ck(cudaMalloc(&blobs, blob * NE), "blobs");
    std::vector<ExpertRef> table(NE);
    for (int e = 0; e < NE; ++e) {
        uint8_t * b = blobs + blob * e;
        fill_blob<<<(unsigned) ((blob + 255) / 256), 256>>>(b, gu_bytes, d_bytes, 1000 + e);
        table[e].gate_up = b;
        table[e].down = b + gu_bytes;
    }
    ExpertRef * d_table;
    ck(cudaMalloc(&d_table, NE * sizeof(ExpertRef)), "table");
    ck(cudaMemcpy(d_table, table.data(), NE * sizeof(ExpertRef), cudaMemcpyHostToDevice), "table");

    // routing: K distinct experts per token, skewed (some experts far more popular), softmax-like weights
    std::mt19937 rng(7);
    std::vector<int32_t> ids((size_t) T * K);
    std::vector<float> w((size_t) T * K), x((size_t) T * D);
    std::vector<double> pop(NE);
    for (int e = 0; e < NE; ++e) pop[e] = 1.0 / (1.0 + 0.02 * e);
    std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    for (int t = 0; t < T; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            do e = pick(rng);
            while (std::find(ids.begin() + (size_t) t * K, ids.begin() + (size_t) t * K + k, e) !=
                   ids.begin() + (size_t) t * K + k);
            ids[(size_t) t * K + k] = e;
            w[(size_t) t * K + k] = 0.2f + U(rng);
            s += w[(size_t) t * K + k];
        }
        for (int k = 0; k < K; ++k) w[(size_t) t * K + k] *= 2.5f / s;
    }
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (auto & v : x) v = N(rng);
    int32_t * d_ids;
    float *d_w, *d_x, *d_out, *d_ref;
    ck(cudaMalloc(&d_ids, ids.size() * 4), "ids");
    ck(cudaMalloc(&d_w, w.size() * 4), "w");
    ck(cudaMalloc(&d_x, x.size() * 4), "x");
    ck(cudaMalloc(&d_out, x.size() * 4), "out");
    ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");

    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    PrefillExperts * p = prefill_experts_create(g, T, group);
    std::printf("prefill_experts device bytes: %.2f GB\n", prefill_experts_device_bytes(p) / 1e9);
    prefill_experts_run(p, T, d_x, d_ids, d_w, d_table, d_out, st);   // warm-up
    ck(cudaStreamSynchronize(st), "warm-up");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float tot = 0;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, st);
        prefill_experts_run(p, T, d_x, d_ids, d_w, d_table, d_out, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        tot += ms;
    }
    const double ms = tot / reps;
    const double flops = 2.0 * T * K * (3.0 * F * D);
    std::printf("prefill_experts: %.2f ms per layer (%.1f TFLOP/s effective, %.0f tok/s per layer)\n", ms,
                flops / ms / 1e9, T / ms * 1e3);
    prefill_experts_set_timing(p, true);
    prefill_experts_run(p, T, d_x, d_ids, d_w, d_table, d_out, st);
    const PrefillExpertsTiming tm = prefill_experts_timing(p);
    std::printf("  phases (ms, synchronized): sort %.2f gather %.2f dequant %.2f gemm_gu %.2f swiglu %.2f "
                "gemm_down %.2f combine %.2f, %d active experts\n",
                tm.sort, tm.gather, tm.dequant, tm.gemm_gu, tm.swiglu, tm.gemm_down, tm.combine, tm.active_experts);

    if (ref) {
        // pack's F32 path (exact IQ4 x F32) as the reference
        ExpertRef * d_slots;
        ck(cudaMalloc(&d_slots, (size_t) T * K * sizeof(ExpertRef)), "slots");
        expand_slots<<<(T * K + 255) / 256, 256, 0, st>>>(d_ids, d_table, T * K, d_slots);
        void * scratch;
        ck(cudaMalloc(&scratch, experts_gpu_scratch_bytes(g, T)), "scratch");
        ck(cudaMalloc(&d_ref, x.size() * 4), "ref");
        ck(cudaMemsetAsync(d_ref, 0, x.size() * 4, st), "ref zero");
        cudaEventRecord(e0, st);
        experts_gpu_prefill(g, T, d_x, d_ids, d_w, d_slots, d_ref, scratch, st);
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float rms;
        cudaEventElapsedTime(&rms, e0, e1);
        std::vector<float> a(x.size()), b(x.size());
        ck(cudaMemcpy(a.data(), d_out, a.size() * 4, cudaMemcpyDeviceToHost), "out");
        ck(cudaMemcpy(b.data(), d_ref, b.size() * 4, cudaMemcpyDeviceToHost), "ref");
        double worst = 1, dot = 0, na = 0, nb = 0, max_rel = 0;
        for (int t = 0; t < T; ++t) {
            double d = 0, x2 = 0, y2 = 0;
            for (int i = 0; i < D; ++i) {
                const double u = a[(size_t) t * D + i], v = b[(size_t) t * D + i];
                d += u * v;
                x2 += u * u;
                y2 += v * v;
            }
            worst = std::min(worst, d / std::sqrt(x2 * y2 + 1e-30));
            dot += d;
            na += x2;
            nb += y2;
            max_rel = std::max(max_rel, std::sqrt(std::max(0.0, x2 + y2 - 2 * d)) / std::sqrt(y2 + 1e-30));
        }
        std::printf("vs pack experts_gpu (F32, %.1f ms): cos %.7f, worst row cos %.7f, worst row rel err %.2e\n", rms,
                    dot / std::sqrt(na * nb), worst, max_rel);
    }
    prefill_experts_destroy(p);
    return 0;
}

std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Compares out [T][D] with ref: cos, worst row cos, worst row relative error.
void compare(const std::vector<float> & a, const std::vector<float> & b, int T, int D, const char * what) {
    double worst = 1, dot = 0, na = 0, nb = 0, max_rel = 0;
    for (int t = 0; t < T; ++t) {
        double d = 0, x2 = 0, y2 = 0;
        for (int i = 0; i < D; ++i) {
            const double u = a[(size_t) t * D + i], v = b[(size_t) t * D + i];
            d += u * v;
            x2 += u * u;
            y2 += v * v;
        }
        worst = std::min(worst, d / std::sqrt(x2 * y2 + 1e-30));
        dot += d;
        na += x2;
        nb += y2;
        max_rel = std::max(max_rel, std::sqrt(std::max(0.0, x2 + y2 - 2 * d)) / std::sqrt(y2 + 1e-30));
    }
    std::printf("  %s: cos %.7f, worst row cos %.7f, worst row rel err %.2e\n", what, dot / std::sqrt(na * nb), worst,
                max_rel);
}

// R8 repack of one row group (8 GGUF rows), the reference implementation (r8.inc, LAYOUT.md v1) copied as is: the authority prefill_cpu's R8 unpack is checked against.
namespace r8ref {
constexpr int kXsBlock = 136, kNlBlock = 18, kXsTile = 8 * kXsBlock, kNlTile = 8 * kNlBlock;
inline int gguf_w(const uint8_t * q, int k) { return k < 16 ? q[k] & 15 : q[k - 16] >> 4; }
inline void put_halves(const uint8_t * const q[8], uint8_t * out /* 128 B */) {
    for (int h = 0; h < 2; ++h)
        for (int r = 0; r < 8; ++r)
            for (int s = 0; s < 2; ++s)
                for (int i = 0; i < 4; ++i) {
                    const int k = 16 * h + 4 * s + i;
                    out[64 * h + 4 * (2 * r + s) + i] = uint8_t(gguf_w(q[r], k) | gguf_w(q[r], k + 8) << 4);
                }
}
inline void repack_xs(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 256;
    const size_t row = size_t(nb) * kXsBlock;
    for (int b = 0; b < nb; ++b) {
        uint8_t * t = dst + size_t(b) * kXsTile;
        const uint8_t * blk[8];
        for (int r = 0; r < 8; ++r) blk[r] = src + r * row + size_t(b) * kXsBlock;
        uint8_t S[64];
        for (int r = 0; r < 8; ++r) {
            memcpy(t + 2 * r, blk[r], 2);
            const int sh = blk[r][2] | blk[r][3] << 8;
            for (int g = 0; g < 8; ++g)
                S[8 * g + r] = uint8_t(((blk[r][4 + g / 2] >> (4 * (g % 2))) & 15) | (((sh >> (2 * g)) & 3) << 4));
        }
        for (int j = 0; j < 32; ++j) t[16 + j] = uint8_t((S[j] & 15) | (S[j + 32] & 15) << 4);
        for (int j = 0; j < 16; ++j)
            t[48 + j] = uint8_t((S[j] >> 4) | (S[j + 16] >> 4) << 2 | (S[j + 32] >> 4) << 4 | (S[j + 48] >> 4) << 6);
        for (int g = 0; g < 8; ++g) {
            const uint8_t * q[8];
            for (int r = 0; r < 8; ++r) q[r] = blk[r] + 8 + 16 * g;
            put_halves(q, t + 64 + 128 * g);
        }
    }
}
inline void repack_nl(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 32;
    const size_t row = size_t(nb) * kNlBlock;
    for (int b = 0; b < nb; ++b) {
        uint8_t * t = dst + size_t(b) * kNlTile;
        const uint8_t * q[8];
        for (int r = 0; r < 8; ++r) {
            const uint8_t * blk = src + r * row + size_t(b) * kNlBlock;
            memcpy(t + 2 * r, blk, 2);
            q[r] = blk + 2;
        }
        put_halves(q, t + 16);
    }
}
}  // namespace r8ref

int run_cpu(int argc, char ** argv) {
    const int U = std::atoi(arg(argc, argv, "--U", "64").c_str());
    const std::vector<int> counts = parse_ints(arg(argc, argv, "--counts", "1,2,4,8,16,32,64,128"));
    const int threads = std::atoi(arg(argc, argv, "--threads", "48").c_str());
    const bool check = arg(argc, argv, "--check", "1") != "0";
    const int reps = std::atoi(arg(argc, argv, "--reps", "3").c_str());
    // --r8 1: the same experts also as R8 host copies (layout 1, r8ref repack): output must be bit-identical;
    // --r8 2: the R8 copies only (prod's host layout) for every run, A/B included
    const int r8mode = std::atoi(arg(argc, argv, "--r8", "0").c_str());
    const bool r8 = r8mode != 0;
    // --panels 1: panel tiles from 8 tokens per expert (default), 0: GGUF / R8 tiles only, 2: both, alternated (A/B)
    const int panels = std::atoi(arg(argc, argv, "--panels", "1").c_str());
    const Geometry g;
    const int D = g.n_embd, F = g.ffn_expert, K = g.n_expert_used;
    if (U < K) throw std::runtime_error("--U must be >= 8");
    const size_t xs_row = (size_t) (D / 256) * 136, nl_row = (size_t) (F / 32) * 18;
    const size_t gu_bytes = 2 * (size_t) F * xs_row, d_bytes = (size_t) D * nl_row, blob = gu_bytes + d_bytes;
    Pool pool(threads - 1);
    std::printf("cpu share: %d experts x %.2f MB, pool %d threads on %d NUMA nodes\n", U, blob / 1e6, pool.size(),
                pool.nodes());

    // device blobs (same generator as the experts mode), copied to host memory placed as the arena: item c of the
    // pool first-touches 32-row chunk c of every plane of every blob
    // (--check 0: no GPU at all, the same bytes generated on the host)
    uint8_t * d_blobs = nullptr;
    std::vector<ExpertRef> table(U), href(U);
    std::vector<uint8_t> staging;
    if (check) {
        ck(cudaMalloc(&d_blobs, blob * U), "blobs");
        for (int e = 0; e < U; ++e) {
            uint8_t * b = d_blobs + blob * e;
            fill_blob<<<(unsigned) ((blob + 255) / 256), 256>>>(b, gu_bytes, d_bytes, 1000 + e);
            table[e].gate_up = b;
            table[e].down = b + gu_bytes;
        }
        staging.resize(blob * U);
        ck(cudaMemcpy(staging.data(), d_blobs, staging.size(), cudaMemcpyDeviceToHost), "blobs d2h");
    }
    const auto put = [&](uint8_t * dst, size_t o, size_t n) {   // blob bytes [o, o + n) of all experts' blob e
        if (check) std::memcpy(dst + o, staging.data() + o, n);
        else
            for (size_t i = 0; i < n; ++i) dst[o + i] = blob_byte((o + i) % blob, gu_bytes, 1000 + (o + i) / blob);
    };
    uint8_t * host = static_cast<uint8_t *>(std::aligned_alloc(4096, blob * U));
    const size_t cg = 32 * xs_row, cd = 32 * nl_row;
    const int n_g = 2 * F / 32, n_d = D / 32;   // gate | up chunks, down chunks
    pool.run(std::max(n_g, n_d), [&](int c) {
        for (int e = 0; e < U; ++e) {
            const size_t o = blob * e;
            if (c < n_g) put(host, o + c * cg, cg);
            if (c < n_d) put(host, o + gu_bytes + c * cd, cd);
        }
    });
    for (int e = 0; e < U; ++e) {
        href[e].gate_up = host + blob * e;
        href[e].down = host + blob * e + gu_bytes;
    }
    // R8 copies, placed as the GGUF ones (a 32-row chunk is 4 row groups)
    uint8_t * host8 = r8 ? static_cast<uint8_t *>(std::aligned_alloc(4096, blob * U)) : nullptr;
    std::vector<ExpertRef> href8(U);
    if (r8) {
        pool.run(std::max(n_g, n_d), [&](int c) {
            for (int e = 0; e < U; ++e) {
                const size_t og = blob * e + c * cg, od = blob * e + gu_bytes + c * cd;
                for (int q = 0; q < 4; ++q) {
                    if (c < n_g) r8ref::repack_xs(host + og + q * 8 * xs_row, host8 + og + q * 8 * xs_row, D);
                    if (c < n_d) r8ref::repack_nl(host + od + q * 8 * nl_row, host8 + od + q * 8 * nl_row, F);
                }
            }
        });
        for (int e = 0; e < U; ++e) {
            href8[e].gate_up = host8 + blob * e;
            href8[e].down = host8 + blob * e + gu_bytes;
            href8[e].layout = 1;
        }
    }
    ExpertRef * d_table = nullptr;
    if (check) {
        ck(cudaMalloc(&d_table, U * sizeof(ExpertRef)), "table");
        ck(cudaMemcpy(d_table, table.data(), U * sizeof(ExpertRef), cudaMemcpyHostToDevice), "table");
    }

    std::printf("%8s %6s %7s %10s %10s %12s\n", "count", "T", "jobs", "ms/layer", "us/job", "ms/expert");
    std::mt19937 rng(11);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::uniform_real_distribution<float> Uw(0.2f, 1.2f);
    for (int c : counts) {
        const int T = std::max(1, U * c / K);
        std::vector<int32_t> ids((size_t) T * K);
        std::vector<float> w((size_t) T * K), x((size_t) T * D);
        for (auto & v : x) v = N(rng);
        // balanced: token t takes experts (t * K + k) % U, shuffled per round so the pairs vary
        std::vector<int> perm(U);
        for (int t = 0; t < T; ++t) {
            if ((t * K) % U == 0 || t == 0) {
                for (int e = 0; e < U; ++e) perm[e] = e;
                std::shuffle(perm.begin(), perm.end(), rng);
            }
            for (int k = 0; k < K; ++k) {
                ids[(size_t) t * K + k] = perm[(t * K + k) % U];
                w[(size_t) t * K + k] = Uw(rng);
            }
        }
        // a token must not take an expert twice (U % 8 == 0 keeps a token's 8 slots inside one permutation)
        CpuExpertJobs jobs;
        std::vector<ExpertRef> refs8;
        std::vector<std::vector<std::pair<int, float>>> per(U);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) per[ids[(size_t) t * K + k]].emplace_back(t, w[(size_t) t * K + k]);
        for (int e = 0; e < U; ++e) {
            if (per[e].empty()) continue;
            jobs.refs.push_back(href[e]);
            refs8.push_back(href8[e]);
            for (const auto & [t, wt] : per[e]) {
                jobs.row.push_back(t);
                jobs.wt.push_back(wt);
            }
            jobs.start.push_back(jobs.jobs());
        }
        if (r8mode == 2) jobs.refs = refs8;
        PrefillCpu * pc = prefill_cpu_create(g, T, jobs.jobs());
        std::vector<float> acc((size_t) T * D);
        prefill_cpu_set_panels(pc, panels != 0);
        prefill_cpu_quantize(pc, pool, x.data(), T);
        prefill_cpu_run(pc, pool, jobs, T, acc.data());   // warm-up
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_ms();
            prefill_cpu_quantize(pc, pool, x.data(), T);
            prefill_cpu_run(pc, pool, jobs, T, acc.data());
            best = std::min(best, now_ms() - t0);
        }
        const PrefillCpuTiming tm = prefill_cpu_timing(pc);
        std::printf("%8d %6d %7d %10.2f %10.2f %12.3f   (gate_up %.2f quant %.2f down %.2f)\n", c, T, jobs.jobs(),
                    best, 1e3 * best / jobs.jobs(), best / jobs.experts(), tm.gate_up, tm.quant, tm.down);
        if (panels == 2) {   // the same jobs without the panels, the two alternated; panels vs tiles max |diff|
            std::vector<float> acc0((size_t) T * D);
            double b[2] = {1e30, 1e30};
            PrefillCpuTiming t[2];
            for (int r = 0; r <= reps; ++r)   // r == 0: warm-up
                for (int on = 0; on < 2; ++on) {
                    prefill_cpu_set_panels(pc, on);
                    const double t0 = now_ms();
                    prefill_cpu_quantize(pc, pool, x.data(), T);
                    prefill_cpu_run(pc, pool, jobs, T, on ? acc.data() : acc0.data());
                    if (r && now_ms() - t0 < b[on]) {
                        b[on] = now_ms() - t0;
                        t[on] = prefill_cpu_timing(pc);
                    }
                }
            double md = 0, ma = 0;
            for (size_t i = 0; i < acc.size(); ++i) {
                md = std::max(md, (double) std::fabs(acc[i] - acc0[i]));
                ma = std::max(ma, (double) std::fabs(acc0[i]));
            }
            std::printf("%8s A/B tiles %.2f ms (gate_up %.2f down %.2f) panels %.2f ms (gate_up %.2f down %.2f) = %.2fx; "
                        "max |diff| %.1e of max |out|\n", "", b[0], t[0].gate_up, t[0].down, b[1], t[1].gate_up,
                        t[1].down, b[0] / b[1], ma > 0 ? md / ma : md);
        }
        if (r8mode == 1) {
            CpuExpertJobs j8 = jobs;
            j8.refs = refs8;
            std::vector<float> acc8((size_t) T * D);
            double best8 = 1e30;
            for (int r = 0; r <= reps; ++r) {   // r == 0: warm-up
                const double t0 = now_ms();
                prefill_cpu_quantize(pc, pool, x.data(), T);
                prefill_cpu_run(pc, pool, j8, T, acc8.data());
                if (r) best8 = std::min(best8, now_ms() - t0);
            }
            // R8 tiles vs the GGUF unpack tiles: the same integers, float sums in another order
            double md = 0, ma = 0;
            for (size_t i = 0; i < acc.size(); ++i) {
                md = std::max(md, (double) std::fabs(acc[i] - acc8[i]));
                ma = std::max(ma, (double) std::fabs(acc[i]));
            }
            const PrefillCpuTiming t8 = prefill_cpu_timing(pc);
            std::printf("%8s %6s %7s %10.2f %10.2f %12.3f   (gate_up %.2f quant %.2f down %.2f) R8 vs GGUF: max |diff| "
                        "%.1e of max |out|\n", "", "", "", best8, 1e3 * best8 / jobs.jobs(), best8 / jobs.experts(),
                        t8.gate_up, t8.quant, t8.down, ma > 0 ? md / ma : md);
        }
        if (check && T <= 4096) {
            int32_t * d_ids;
            float *d_w, *d_x, *d_ref;
            ExpertRef * d_slots;
            void * scratch;
            ck(cudaMalloc(&d_ids, ids.size() * 4), "ids");
            ck(cudaMalloc(&d_w, w.size() * 4), "w");
            ck(cudaMalloc(&d_x, x.size() * 4), "x");
            ck(cudaMalloc(&d_ref, x.size() * 4), "ref");
            ck(cudaMalloc(&d_slots, ids.size() * sizeof(ExpertRef)), "slots");
            ck(cudaMalloc(&scratch, experts_gpu_scratch_bytes(g, T)), "scratch");
            ck(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
            ck(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
            ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            ck(cudaMemset(d_ref, 0, x.size() * 4), "ref zero");
            expand_slots<<<(T * K + 255) / 256, 256>>>(d_ids, d_table, T * K, d_slots);
            experts_gpu_prefill(g, T, d_x, d_ids, d_w, d_slots, d_ref, scratch, 0);
            std::vector<float> ref(x.size());
            ck(cudaMemcpy(ref.data(), d_ref, ref.size() * 4, cudaMemcpyDeviceToHost), "ref d2h");
            compare(acc, ref, T, D, "vs pack F32 experts_gpu_prefill");
            for (void * q : {(void *) d_ids, (void *) d_w, (void *) d_x, (void *) d_ref, (void *) d_slots, scratch})
                cudaFree(q);
        }
        prefill_cpu_destroy(pc);
    }
    std::free(host);
    std::free(host8);
    if (check) {   // no CUDA call at all without --check (no context on the GPU)
        cudaFree(d_blobs);
        cudaFree(d_table);
    }
    return 0;
}

// ---- cpu-exact --------------------------------------------------------------------------------------------------
constexpr int kKv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};   // kvalues_iq4nl

// Reference group scales and codes of row bytes, as ggml's dequantize_row_iq4_xs / _iq4_nl.
void ref_decode(const uint8_t * row, bool xs, int width, std::vector<double> & sc, std::vector<int> & code) {
    sc.assign(width / 32, 0.0);
    code.assign(width, 0);
    if (xs) {
        for (int b = 0; b < width / 256; ++b) {
            const uint8_t * blk = row + (size_t) b * 136;
            __half_raw hr;
            std::memcpy(&hr.x, blk, 2);
            const double d = __half2float(__half(hr));
            const int sh = blk[2] | (blk[3] << 8);
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                sc[b * 8 + ib] = d * (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    code[b * 256 + ib * 32 + j] = blk[8 + ib * 16 + j] & 15;
                    code[b * 256 + ib * 32 + j + 16] = blk[8 + ib * 16 + j] >> 4;
                }
            }
        }
    } else {
        for (int b = 0; b < width / 32; ++b) {
            const uint8_t * blk = row + (size_t) b * 18;
            __half_raw hr;
            std::memcpy(&hr.x, blk, 2);
            sc[b] = __half2float(__half(hr));
            for (int j = 0; j < 16; ++j) {
                code[b * 32 + j] = blk[2 + j] & 15;
                code[b * 32 + j + 16] = blk[2 + j] >> 4;
            }
        }
    }
}

void put_half(uint8_t * p, float v) {
    const __half_raw hr = __half_raw(__float2half(v));
    std::memcpy(p, &hr.x, 2);
}

int run_cpu_exact(int, char **) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(0.001f, 0.1f);
    std::normal_distribution<float> nd(0.f, 1.f);
    const char * wcases[] = {"random", "subscale 0/63", "codes 0/15", "d negative", "d 65504 / subnormal"};
    const char * nl_case1 = "random (no sub-scale)";   // IQ4_NL has no sub-scales
    const char * xcases[] = {"normal", "zero groups", "single spikes", "saturated", "tiny 1e-36"};
    double worst = 0;
    int r8_diff = 0, r8_runs = 0;   // R8 row groups (reference repack) through prefill_cpu's R8 unpack: same bits
    for (int xs = 1; xs >= 0; --xs) {
        const int width = xs ? 4096 : 2048, bsz = xs ? 136 : 18, vals = xs ? 256 : 32;
        const size_t rb = (size_t) (width / vals) * bsz;
        std::vector<uint8_t> rows(32 * rb);
        for (int wc = 0; wc < 5; ++wc) {
            for (auto & b : rows) b = (uint8_t) rng();
            for (int r = 0; r < 32; ++r)
                for (int b = 0; b < width / vals; ++b) {
                    uint8_t * blk = rows.data() + r * rb + (size_t) b * bsz;
                    float d = ud(rng);
                    if (wc == 3) d = -d;
                    if (wc == 4) d = r % 2 ? 65504.f : 5.96e-8f;   // f16 max / smallest subnormal
                    put_half(blk, d);
                    if (xs && wc == 1) {   // every sub-scale 0 (scale -32) or 63 (+31)
                        const int ls = r % 2 ? 63 : 0;
                        blk[2] = blk[3] = (uint8_t) (ls >> 4 ? 0xFF : 0);
                        for (int k = 0; k < 4; ++k) blk[4 + k] = (uint8_t) ((ls & 15) | (ls & 15) << 4);
                    }
                    if (wc == 2) std::memset(blk + (xs ? 8 : 2), r % 2 ? 0xFF : 0x00, xs ? 128 : 16);
                }
            for (int xc = 0; xc < 5; ++xc)
                for (int n = 1; n <= 16; ++n) {   // GGUF tiles take 1..5 tokens, R8 tiles 1..12, panels 1..16
                    const bool gguf = n <= 5, r8t = n <= 12;
                    std::vector<float> x((size_t) n * width), out(32 * n), o8(32 * n), d((size_t) n * (width / 32));
                    std::vector<int8_t> q((size_t) n * width);
                    for (size_t i = 0; i < x.size(); ++i) {
                        const int grp = (int) (i % width) / 32, k = (int) (i % 32);
                        float v = nd(rng);
                        if (xc == 1 && grp % 2) v = 0;
                        if (xc == 2) v = k == grp % 32 ? (grp % 2 ? 1000.f : -1000.f) : 0.f;
                        if (xc == 3) v = (k % 2 ? 3.f : -3.f);
                        if (xc == 4) v *= 1e-36f;
                        x[i] = v;
                    }
                    if (!r8t) {   // the activations as the panels quantize them (checked against R8 / GGUF up to 12)
                        std::vector<float> op(32 * n);
                        prefill_cpu_dot_panels(rows.data(), xs, false, width, x.data(), n, op.data(), q.data(), d.data());
                    } else {   // R8 tiles on the row groups: the same quantized activations (checked), own F64 bound below
                        std::vector<uint8_t> r8(rows.size());
                        for (int g = 0; g < 4; ++g)
                            (xs ? r8ref::repack_xs : r8ref::repack_nl)(rows.data() + 8 * g * rb, r8.data() + 8 * g * rb, width);
                        prefill_cpu_dot(r8.data(), xs, true, width, x.data(), n, o8.data(), q.data(), d.data());
                    }
                    if (gguf) {
                        std::vector<int8_t> qg(q.size());
                        std::vector<float> dg(d.size());
                        prefill_cpu_dot(rows.data(), xs, false, width, x.data(), n, out.data(), qg.data(), dg.data());
                        if (qg != q || dg != d) throw std::runtime_error("R8 and GGUF tiles quantized different activations");
                        ++r8_runs;
                        if (std::memcmp(o8.data(), out.data(), out.size() * 4)) ++r8_diff;
                    }
                    double err = 0, err8 = 0;
                    bool under = false;   // a weight x activation scale product below FLT_MIN: float32 underflows it
                    std::vector<double> sc;
                    std::vector<int> code;
                    for (int r = 0; r < 32; ++r) {
                        ref_decode(rows.data() + r * rb, xs, width, sc, code);
                        for (int t = 0; t < n; ++t) {
                            double ref = 0, mag = 0;
                            for (int g = 0; g < width / 32; ++g) {
                                int64_t is = 0;
                                for (int k = 0; k < 32; ++k)
                                    is += (int64_t) kKv[code[g * 32 + k]] * q[(size_t) t * width + g * 32 + k];
                                const double sd = sc[g] * (double) d[(size_t) t * (width / 32) + g];
                                if (sd != 0 && std::fabs(sd) < 1.17549435e-38) under = true;
                                const double term = sd * (double) is;
                                ref += term;
                                mag += std::fabs(term);
                            }
                            const auto rel = [&](float v) { return mag > 0 ? std::fabs(v - ref) / mag : std::fabs(v); };
                            if (gguf) err = std::max(err, rel(out[r * n + t]));
                            if (r8t) err8 = std::max(err8, rel(o8[r * n + t]));
                        }
                    }
                    // panel tiles (n <= 16), GGUF rows and R8 row groups: the same quantized activations, own F64 bound
                    double errp = 0;
                    std::vector<float> op0;
                    for (int lay = 0; lay < 2; ++lay) {
                        std::vector<uint8_t> src = rows;
                        if (lay)
                            for (int g = 0; g < 4; ++g)
                                (xs ? r8ref::repack_xs : r8ref::repack_nl)(rows.data() + 8 * g * rb, src.data() + 8 * g * rb, width);
                        std::vector<float> op(32 * n), dp(d.size());
                        std::vector<int8_t> qp(q.size());
                        prefill_cpu_dot_panels(src.data(), xs, lay, width, x.data(), n, op.data(), qp.data(), dp.data());
                        if (qp != q || dp != d) throw std::runtime_error("panel tiles quantized different activations");
                        // R8 straight to panels: the same panel bytes and scales as the GGUF unpack + transpose
                        if (!lay) op0 = op;
                        else if (std::memcmp(op.data(), op0.data(), op.size() * 4))
                            throw std::runtime_error("R8 panels not bit-identical to the GGUF panels");
                        for (int r = 0; r < 32; ++r) {
                            ref_decode(rows.data() + r * rb, xs, width, sc, code);
                            for (int t = 0; t < n; ++t) {
                                double ref = 0, mag = 0;
                                for (int g = 0; g < width / 32; ++g) {
                                    int64_t is = 0;
                                    for (int k = 0; k < 32; ++k)
                                        is += (int64_t) kKv[code[g * 32 + k]] * q[(size_t) t * width + g * 32 + k];
                                    const double term = sc[g] * (double) d[(size_t) t * (width / 32) + g] * (double) is;
                                    ref += term;
                                    mag += std::fabs(term);
                                }
                                const float v = op[r * n + t];
                                errp = std::max(errp, mag > 0 ? std::fabs(v - ref) / mag : std::fabs(v));
                            }
                        }
                    }
                    if (!under) worst = std::max(worst, std::max(errp, std::max(err, err8)));
                    if (n == 1 || n == 5 || n == 12 || (errp > 1e-5 && !under))
                        std::printf("  %s %-20s x %-14s n=%2d: max rel error panels %.2e%s\n", xs ? "XS" : "NL",
                                    !xs && wc == 1 ? nl_case1 : wcases[wc], xcases[xc], n, errp,
                                    under ? "  (float32 underflow domain, not counted)" : errp > 1e-5 ? "  <-- OVER 1e-5" : "");
                    if (n == 1 || n == 5 || n == 12 || (std::max(err, err8) > 1e-5 && !under))
                        std::printf("  %s %-20s x %-14s n=%2d: max rel error GGUF %.2e R8 %.2e%s\n", xs ? "XS" : "NL",
                                    !xs && wc == 1 ? nl_case1 : wcases[wc], xcases[xc], n, err, err8,
                                    under ? "  (scale product below FLT_MIN: float32 underflow domain, not counted)"
                                          : std::max(err, err8) > 1e-5 ? "  <-- OVER 1e-5" : "");
                }
        }
    }
    std::printf("cpu-exact R8 tiles: %d / %d runs bit-identical to the GGUF tiles (float order differs: the F64 bound "
                "above is the check)\n", r8_runs - r8_diff, r8_runs);
    {   // one thread, rows in cache: quantize + unpack + tile of 32 rows x 1 token, GGUF vs R8 (the unpack's compute)
        for (int xs = 1; xs >= 0; --xs) {
            const int width = xs ? 4096 : 2048;
            const size_t rb = (size_t) (width / (xs ? 256 : 32)) * (xs ? 136 : 18);
            std::vector<uint8_t> rows(32 * rb), r8(32 * rb);
            for (auto & b : rows) b = (uint8_t) (rng() & 0x7F);   // keeps the f16 d finite
            for (int g = 0; g < 4; ++g)
                (xs ? r8ref::repack_xs : r8ref::repack_nl)(rows.data() + 8 * g * rb, r8.data() + 8 * g * rb, width);
            std::vector<float> x(width), out(32), d(width / 32);
            std::vector<int8_t> q(width);
            for (auto & v : x) v = nd(rng);
            // best of 400 trials of 25 calls, the two layouts alternating (a loaded host: the minimum is the compute)
            double best[2] = {1e30, 1e30};
            for (int trial = 0; trial < 400; ++trial)
                for (int lay = 0; lay < 2; ++lay) {
                    const int reps = 25;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i)
                        prefill_cpu_dot(lay ? r8.data() : rows.data(), xs, lay, width, x.data(), 1, out.data(), q.data(),
                                        d.data());
                    best[lay] = std::min(
                        best[lay], std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
                }
            for (int lay = 0; lay < 2; ++lay)
                std::printf("  %s %s: %.2f us per 32 rows (quantize + unpack + 1-token tile, in cache, best of 400)\n",
                            xs ? "XS" : "NL", lay ? "R8  " : "GGUF", best[lay]);
        }
    }
    const bool ok = worst <= 1e-5;   // R8 tiles reassociate the float sums: the F64 bound is the contract, not bit identity
    std::printf("cpu-exact: worst max rel error %.2e -> %s\n", worst, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---- swiglu-exact --------------------------------------------------------------------------------------------------
// prefill_cpu's vector SwiGLU against expert_swiglu_clamp (this host's libm expf) on every float gate (2^32 bit
// patterns, quiet NaN and infinities included; signaling NaN excepted), up cycling through values that hit both
// clamps, zeros and NaN: bitwise.
int run_swiglu_exact(int argc, char ** argv) {
    const int threads = std::atoi(arg(argc, argv, "--threads", "16").c_str());
    const float ups[16] = {1.0f, -1.0f, 0.5f, -3.75f, 9.99f, 10.0f, 12.5f, -10.0f, -1e6f, 0.0f, -0.0f, 1e-30f,
                           -2.3e-41f, 7.123456f, std::numeric_limits<float>::quiet_NaN(), INFINITY};
    std::atomic<uint64_t> bad{0}, skipped{0};
    std::atomic<uint32_t> first_bad{0};
    std::vector<std::thread> pool;
    constexpr uint64_t kChunk = 1u << 16;
    const uint64_t chunks = (1ull << 32) / kChunk;
    std::atomic<uint64_t> next{0};
    for (int th = 0; th < threads; ++th)
        pool.emplace_back([&] {
            std::vector<float> g(kChunk), u(kChunk), o(kChunk);
            uint64_t snan = 0;
            for (uint64_t c; (c = next.fetch_add(1)) < chunks;) {
                for (uint64_t i = 0; i < kChunk; ++i) {
                    const uint32_t bits = (uint32_t) (c * kChunk + i);
                    std::memcpy(&g[i], &bits, 4);
                    u[i] = ups[(i + c) % 16];
                }
                prefill_cpu_swiglu(g.data(), u.data(), o.data(), (int) kChunk);
                for (uint64_t i = 0; i < kChunk; ++i) {
                    const float r = expert_swiglu_clamp(g[i], u[i]);
                    uint32_t a, b;
                    std::memcpy(&a, &o[i], 4);
                    std::memcpy(&b, &r, 4);
                    // signaling NaN gates excepted: glibc's fminf quiets them (NaN out), vminps returns the clamp;
                    // float arithmetic never produces one
                    const uint32_t gb = (uint32_t) (c * kChunk + i);
                    if ((gb & 0x7fc00000u) == 0x7f800000u && (gb & 0x003fffffu)) {
                        ++snan;
                        continue;
                    }
                    if (a != b && !(std::isnan(o[i]) && std::isnan(r))) {
                        if (bad.fetch_add(1) == 0) first_bad = (uint32_t) (c * kChunk + i);
                    }
                }
            }
            skipped += snan;
        });
    for (auto & t : pool) t.join();
    if (bad) {
        float g;
        const uint32_t fb = first_bad;
        std::memcpy(&g, &fb, 4);
        std::printf("swiglu-exact: %llu mismatches (first gate %a = 0x%08x) -> FAIL\n", (unsigned long long) bad.load(),
                    g, fb);
        return 1;
    }
    std::printf("swiglu-exact: 2^32 gates bit-identical to expert_swiglu_clamp (%llu signaling NaN gates skipped) -> "
                "PASS\n", (unsigned long long) skipped.load());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: glm_prefill_bench experts|cpu|cpu-exact [options]\n");
        return 2;
    }
    try {
        if (!std::strcmp(argv[1], "experts")) return run_experts(argc, argv);
        if (!std::strcmp(argv[1], "cpu")) return run_cpu(argc, argv);
        if (!std::strcmp(argv[1], "cpu-exact")) return run_cpu_exact(argc, argv);
        if (!std::strcmp(argv[1], "swiglu-exact")) return run_swiglu_exact(argc, argv);
        if (!std::strcmp(argv[1], "cpu-kernel")) {   // one thread, in cache: GMAC/s of the GGUF tile vs the panel tile
            const int width = 4096, reps = std::atoi(arg(argc, argv, "--reps", "2000").c_str());
            const auto gmacs = [&](int n, double ns) { return 32.0 * width * n / ns; };
            for (int n : {1, 5}) {
                const double ns = prefill_cpu_kernel_bench(0, width, n, reps);
                std::printf("GGUF tile   n=%2d: %8.0f ns  %6.1f GMAC/s\n", n, ns, gmacs(n, ns));
            }
            for (int n : {1, 4, 8, 12, 16}) {
                const double ns = prefill_cpu_kernel_bench(1, width, n, reps);
                std::printf("panel tile  n=%2d: %8.0f ns  %6.1f GMAC/s\n", n, ns, gmacs(n, ns));
            }
            std::printf("unpack 32 rows: %.0f ns; unpack + panels: %.0f ns; R8 to panels: %.0f ns\n",
                        prefill_cpu_kernel_bench(2, width, 1, reps), prefill_cpu_kernel_bench(3, width, 1, reps),
                        prefill_cpu_kernel_bench(6, width, 1, reps));
            std::printf("SwiGLU 32 rows x 16 tokens: scalar %.0f ns, vector %.0f ns\n",
                        prefill_cpu_kernel_bench(4, width, 16, reps), prefill_cpu_kernel_bench(5, width, 16, reps));
            return 0;
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
