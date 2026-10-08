// glm_mla_bench: V100 timings of one MLA+DSA layer.
//
//   glm_mla_bench --gguf <shard1> [--layer 3] [--ctx 4096,32768,131072] [--T 1,4] [--iters 50]
//                 [--prefill N --chunk C] [--graph [--no-node-prio]] [--latent f16|q8]
//
// Decode/verify: the cache is filled with pseudo-random rows (mla_cache_fill_random) up to ctx, then each
// iteration runs mla_forward on T new tokens at position ctx and truncates back. Prefill: runs N tokens from an
// empty cache in batches of C and reports tokens/s. Inputs are pseudo-random activations.
// --graph: decode also through mla_forward_dev captured in a CUDA graph (position in device memory). The graph is
// captured once per (ctx, T), replayed at positions ctx-3 .. ctx and compared bitwise with mla_forward, then timed.
// The graph is instantiated with cudaGraphInstantiateFlagUseNodePriority (the side stream's priority carries over
// to its captured kernels) unless --no-node-prio.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "mla_test_util.hpp"

using namespace strata::glm;
using namespace mla_test;

namespace {

std::vector<int64_t> parse_list(const std::string & s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(std::atoll(x.c_str()));
    return v;
}

void random_rows(std::vector<float> & h, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : h) v = nd(rng);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf;
    int layer = 3, iters = 50;
    std::vector<int64_t> ctxs = {4096, 32768, 131072}, Ts = {1, 4};
    int64_t prefill = 0, chunk = 4096;
    bool graph = false, node_prio = true;
    MlaLatent latent = MlaLatent::F16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) ctxs = parse_list(argv[++i]);
        else if (a == "--T" && i + 1 < argc) Ts = parse_list(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill = std::atoll(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) chunk = std::atoll(argv[++i]);
        else if (a == "--graph") graph = true;
        else if (a == "--latent" && i + 1 < argc) latent = std::string(argv[++i]) == "q8" ? MlaLatent::Q8 : MlaLatent::F16;
        else if (a == "--no-node-prio") node_prio = false;
        else {
            std::fprintf(stderr, "bad argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty()) {
        std::fprintf(stderr, "usage: %s --gguf <shard1> [--layer L] [--ctx a,b] [--T 1,4] [--prefill N --chunk C]\n", argv[0]);
        return 2;
    }
    const Geometry g;
    const int D = g.n_embd;
    Model m(gguf);
    const MlaWeights w = load_layer(m, layer);
    std::printf("weight types: q_a %d q_b %d kv_a %d k_b %d v_b %d out %d idx_q_b %d idx_k %d idx_gate %d idx_proj %d\n",
                w.q_a.type, w.q_b.type, w.kv_a.type, w.k_b.type, w.v_b.type, w.out.type, w.idx_q_b.type, w.idx_k.type,
                w.idx_gate.type, w.idx_proj.type);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    if (prefill > 0) {
        const int T = (int)std::min<int64_t>(chunk, prefill);
        MlaCache * c = mla_cache_create(g, prefill, latent);
        void * scratch = nullptr;
        CK(cudaMalloc(&scratch, mla_scratch_bytes(g, T, prefill)));
        float *x = nullptr, *out = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * T * D));
        CK(cudaMalloc(&out, sizeof(float) * T * D));
        std::vector<float> h((size_t)T * D);
        random_rows(h, 7);
        CK(cudaMemcpy(x, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        // warm-up (cuBLAS init, first-touch)
        Batch b;
        b.T = T;
        b.pos0 = 0;
        mla_forward(g, w, c, b, x, out, scratch, s);
        mla_cache_truncate(c, 0);
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0, s));
        for (int64_t p = 0; p < prefill; p += T) {
            b.T = (int)std::min<int64_t>(T, prefill - p);
            b.pos0 = p;
            mla_forward(g, w, c, b, x, out, scratch, s);
        }
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("prefill layer %d: %lld tokens in batches of %d: %.1f ms, %.0f tok/s per layer\n", layer,
                    (long long)prefill, T, ms, prefill / (ms / 1e3));
        mla_cache_destroy(c);
        cudaFree(scratch);
        cudaFree(x);
        cudaFree(out);
    }

    for (int64_t ctx : ctxs) {
        int max_T = 1;
        for (auto T : Ts) max_T = std::max<int>(max_T, (int)T);
        MlaCache * c = mla_cache_create(g, ctx + max_T, latent);
        void * scratch = nullptr;
        CK(cudaMalloc(&scratch, mla_scratch_bytes(g, max_T, ctx + max_T)));
        float *x = nullptr, *out = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * max_T * D));
        CK(cudaMalloc(&out, sizeof(float) * max_T * D));
        std::vector<float> h((size_t)max_T * D);
        random_rows(h, 11);
        CK(cudaMemcpy(x, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        mla_cache_fill_random(c, w, ctx, 1234, s);
        for (int64_t T : Ts) {
            Batch b;
            b.T = (int)T;
            b.pos0 = ctx;
            for (int i = 0; i < 3; ++i) {
                mla_forward(g, w, c, b, x, out, scratch, s);
                mla_cache_truncate(c, ctx);
            }
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) {
                mla_forward(g, w, c, b, x, out, scratch, s);
                mla_cache_truncate(c, ctx);
            }
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("decode layer %d ctx %7lld T=%lld: %.3f ms per call (%.3f ms per token)\n", layer,
                        (long long)ctx, (long long)T, ms / iters, ms / iters / T);
            if (!graph) continue;

            // capture once at position ctx, replay at ctx-3 .. ctx against mla_forward (bitwise)
            int64_t * d_pos = nullptr;
            int64_t * h_pos = nullptr;
            CK(cudaMalloc(&d_pos, sizeof(int64_t)));
            CK(cudaMallocHost(&h_pos, sizeof(int64_t)));
            float * out_g = nullptr;
            CK(cudaMalloc(&out_g, sizeof(float) * T * D));
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            mla_forward_dev(g, w, c, (int)T, d_pos, x, out_g, scratch, s);
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&ge, gr, node_prio ? cudaGraphInstantiateFlagUseNodePriority : 0));
            size_t n_nodes = 0;
            CK(cudaGraphGetNodes(gr, nullptr, &n_nodes));
            std::vector<float> ref((size_t)T * D), got((size_t)T * D);
            int bad = 0;
            for (int64_t p = ctx - 3; p <= ctx; ++p) {
                mla_cache_truncate(c, p);
                Batch bp;
                bp.T = (int)T;
                bp.pos0 = p;
                mla_forward(g, w, c, bp, x, out, scratch, s);
                CK(cudaMemcpyAsync(ref.data(), out, ref.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
                mla_cache_truncate(c, p);
                *h_pos = p;
                CK(cudaMemcpyAsync(d_pos, h_pos, sizeof(int64_t), cudaMemcpyHostToDevice, s));
                CK(cudaGraphLaunch(ge, s));
                CK(cudaMemcpyAsync(got.data(), out_g, got.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                mla_cache_set_size(c, p + T);
                if (std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) != 0) {
                    ++bad;
                    std::printf("  graph replay at pos %lld differs from mla_forward\n", (long long)p);
                }
            }
            mla_cache_truncate(c, ctx);
            *h_pos = ctx;
            CK(cudaMemcpyAsync(d_pos, h_pos, sizeof(int64_t), cudaMemcpyHostToDevice, s));
            for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e0, s));
            for (int i = 0; i < iters; ++i) CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float msg = 0;
            CK(cudaEventElapsedTime(&msg, e0, e1));
            std::printf("  graph (%zu nodes) T=%lld: %.3f ms per replay, replays at ctx-3..ctx %s\n", n_nodes,
                        (long long)T, msg / iters, bad ? "DIFFER" : "bitwise identical");
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(gr));
            cudaFree(out_g);
            cudaFree(d_pos);
            cudaFreeHost(h_pos);
            if (bad) return 1;
        }
        mla_cache_destroy(c);
        cudaFree(scratch);
        cudaFree(x);
        cudaFree(out);
    }
    return 0;
}
