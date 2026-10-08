#pragma once
// The GLM-5.3-Flash model as the engine sees it: geometry validated against the GGUF, every dense weight in one
// VRAM arena (Q8_0 matrices repacked to Q8_0R for the GEMV), routed experts left in the mmapped GGUF for the
// expert cache / CPU pool, token embedding left on the host (one row per token is dequantized there).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/glm/geometry.hpp"
#include "strata/glm/gguf.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/types.hpp"

namespace strata::glm {

// One MoE layer's routed experts, straight from the mmapped GGUF: tensor [.., .., n_expert], expert e at
// base + e * bytes. gate/up are [n_embd -> ffn_expert], down [ffn_expert -> n_embd].
struct HostExperts {
    const uint8_t * gate = nullptr;
    const uint8_t * up = nullptr;
    const uint8_t * down = nullptr;
    int type_gate = -1, type_up = -1, type_down = -1;
    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;   // per expert
};

struct LayerWeights {
    int il = 0;
    bool mla = false, moe = false, mtp = false;
    WeightRef attn_norm, ffn_norm;
    WeightRef hc_attn_fn, hc_attn_base, hc_attn_scale;   // absent on the MTP block
    WeightRef hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    KdaWeights kda;                                      // KDA blocks
    MlaWeights mla_w;                                    // MLA + DSA blocks (and the MTP block)
    WeightRef ffn_gate, ffn_up, ffn_down;                // dense blocks 0..n_dense_lead-1
    WeightRef gate_inp, exp_probs_b;                     // MoE blocks
    WeightRef gate_shexp, up_shexp, down_shexp;
    HostExperts experts;
    WeightRef eh_proj, enorm, hnorm, shared_head_norm;   // MTP block only
};

struct LoadOptions {
    bool repack_q8 = true;   // Q8_0 matrices -> Q8_0R (same values, faster GEMV)
    bool load_mtp = true;    // dense weights of the MTP block (its experts are always mapped)
    bool verbose = true;
    // Test emulation of another quant recipe: Q8_0 tensors whose name ends with one of these (e.g.
    // "ssm_f_a.weight") are dequantized and uploaded as BF16.
    std::vector<std::string> as_bf16;
};

// The tensors pack's final GGUF (glm53-strata-q4) keeps in BF16 (source BF16, excluded from Q8_0 by mainline
// llama-quantize): as_bf16 with this list makes the Q4_K_XL oracle GGUF exercise the same trunk types.
inline std::vector<std::string> pack_recipe_bf16() {
    return {"attn_k_b.weight", "attn_v_b.weight", "attn_kv_a_mqa.weight", "indexer.attn_k.weight",
            "indexer.attn_q_b.weight", "indexer_compressor_gate.weight", "hc_attn_fn.weight", "hc_ffn_fn.weight",
            "ssm_beta.weight", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight"};
}

struct Model {
    Geometry g;
    std::unique_ptr<GgufSet> gguf;
    std::vector<LayerWeights> layers;   // g.n_layer trunk blocks, then the MTP block(s)
    WeightRef output_norm, output;
    HostTensor token_embd;              // host, [n_embd, n_vocab] (points at token_embd_copy)
    std::vector<uint8_t> token_embd_copy;
    void * arena = nullptr;             // device, every dense weight
    size_t arena_bytes = 0;
    ~Model();
};

// Throws std::runtime_error with every mismatch found (metadata, tensor set, shapes, types).
std::unique_ptr<Model> load_model(const std::string & gguf_shard1, const LoadOptions & opt, cudaStream_t stream);

// Token embedding rows on the host: out[t * n_embd + i] for ids[t] (F32). Supports Q8_0, BF16, F16, F32 tables.
void embed_rows_host(const Model & m, const int32_t * ids, int T, float * out);

}  // namespace strata::glm
