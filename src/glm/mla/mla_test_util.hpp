#pragma once
// Shared helpers for the MLA test programs: GGUF shard loading to the GPU, F16 decode, device copies.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/mla.hpp"

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "cuda: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

namespace mla_test {

using namespace strata::glm;

inline float half_to_float(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

// ---- GGUF shards ------------------------------------------------------------------------------------------
struct Model {
    std::vector<std::unique_ptr<strata::GgufFile>> shards;
    std::vector<void *> dev;   // uploaded tensors
    explicit Model(const std::string & first) {
        const auto at = first.find("-00001-of-");
        if (at == std::string::npos) {
            shards.emplace_back(new strata::GgufFile(first));
            return;
        }
        const int n = std::atoi(first.c_str() + at + 10);
        for (int i = 1; i <= n; ++i) {
            char num[16];
            std::snprintf(num, sizeof num, "%05d", i);
            std::string p = first;
            p.replace(at + 1, 5, num);
            shards.emplace_back(new strata::GgufFile(p));
        }
    }
    ~Model() {
        for (void * p : dev) cudaFree(p);
    }
    WeightRef upload(const std::string & name) {
        for (auto & f : shards) {
            const auto * t = f->find(name);
            if (!t) continue;
            int elems = 0, bytes = 0;
            if (!strata::block_geometry(t->type, elems, bytes)) {
                std::fprintf(stderr, "unsupported type for %s\n", name.c_str());
                std::exit(1);
            }
            WeightRef w;
            w.type = (int)t->type;
            for (size_t d = 0; d < t->shape.size(); ++d) w.ne[d] = (int64_t)t->shape[d];
            w.nbytes = t->elements() / (uint64_t)elems * (uint64_t)bytes;
            void * p = nullptr;
            CK(cudaMalloc(&p, w.nbytes));
            CK(cudaMemcpy(p, f->tensor_data(*t), w.nbytes, cudaMemcpyHostToDevice));
            dev.push_back(p);
            w.data = p;
            return w;
        }
        std::fprintf(stderr, "tensor %s not found\n", name.c_str());
        std::exit(1);
    }
    // A Q8_0 tensor dequantized to BF16 on the host (round to nearest even), as the final pack ships some
    // projections; other types are uploaded unchanged.
    WeightRef upload_bf16(const std::string & name) {
        for (auto & f : shards) {
            const auto * t = f->find(name);
            if (!t) continue;
            if ((int)t->type != 8) return upload(name);
            const uint64_t ne = t->elements();
            const auto * src = static_cast<const uint8_t *>(f->tensor_data(*t));
            std::vector<uint16_t> h(ne);
            for (uint64_t b = 0; b < ne / 32; ++b) {
                const uint8_t * blk = src + b * 34;
                const float d = half_to_float((uint16_t)(blk[0] | blk[1] << 8));
                for (int i = 0; i < 32; ++i) {
                    const float v = d * (float)(int8_t)blk[2 + i];
                    uint32_t u;
                    std::memcpy(&u, &v, 4);
                    h[b * 32 + i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
                }
            }
            WeightRef w;
            w.type = 30;
            for (size_t d = 0; d < t->shape.size(); ++d) w.ne[d] = (int64_t)t->shape[d];
            w.nbytes = ne * 2;
            void * p = nullptr;
            CK(cudaMalloc(&p, w.nbytes));
            CK(cudaMemcpy(p, h.data(), w.nbytes, cudaMemcpyHostToDevice));
            dev.push_back(p);
            w.data = p;
            return w;
        }
        std::fprintf(stderr, "tensor %s not found\n", name.c_str());
        std::exit(1);
    }
};

template <typename T>
std::vector<T> d2h(const T * p, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDefault));   // device or mapped host (mla_cache_kg)
    return v;
}

// Q8_0 -> Q8_0R as the engine's loader does (the contract: the engine passes repacked refs)
inline WeightRef repack(Model & m, const WeightRef & w) {
    if (w.type != 8) return w;
    void * p = nullptr;
    CK(cudaMalloc(&p, matmul_repack_bytes(w)));
    m.dev.push_back(p);
    const WeightRef r = matmul_repack(w, p, nullptr);
    CK(cudaDeviceSynchronize());
    return r;
}

// GLM_MLA_BF16=1: the tensors the final pack keeps in BF16 (k_b, v_b, kv_a, indexer k / q_b / gate) are loaded
// as BF16 dequantized from the shard's Q8_0.
inline MlaWeights load_layer(Model & m, int L) {
    auto n = [&](const char * s) { return "blk." + std::to_string(L) + "." + s; };
    static const bool bf16 = std::getenv("GLM_MLA_BF16") && std::atoi(std::getenv("GLM_MLA_BF16")) != 0;
    auto load = [&](const char * s, bool pack_bf16) {
        return bf16 && pack_bf16 ? m.upload_bf16(n(s)) : repack(m, m.upload(n(s)));
    };
    MlaWeights w;
    w.q_a = load("attn_q_a.weight", false);
    w.q_a_norm = m.upload(n("attn_q_a_norm.weight"));
    w.q_b = load("attn_q_b.weight", false);
    w.kv_a = load("attn_kv_a_mqa.weight", true);
    w.kv_a_norm = m.upload(n("attn_kv_a_norm.weight"));
    w.k_b = load("attn_k_b.weight", true);
    w.v_b = load("attn_v_b.weight", true);
    w.out = load("attn_output.weight", false);
    w.idx_q_b = load("indexer.attn_q_b.weight", true);
    w.idx_k = load("indexer.attn_k.weight", true);
    w.idx_k_norm_w = m.upload(n("indexer.k_norm.weight"));
    w.idx_k_norm_b = m.upload(n("indexer.k_norm.bias"));
    w.idx_proj = m.upload(n("indexer.proj.weight"));
    w.idx_gate = load("indexer_compressor_gate.weight", true);
    w.idx_ape = m.upload(n("indexer_compressor_ape.weight"));
    return w;
}

}  // namespace mla_test
