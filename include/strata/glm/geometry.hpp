#pragma once
// GLM-5.3-Flash (llama.cpp arch "glm5-next") geometry, shared by every GLM component.
// Values are the real GGUF ones (its glm5-next.* metadata keys); the engine
// validates them against the GGUF at load time instead of trusting these defaults blindly.

#include <cstdint>

namespace strata::glm {

struct Geometry {
    int n_layer        = 45;      // trunk blocks; block 45 is the MTP (NextN) block (load_model: from the GGUF)
    int n_layer_mtp    = 1;
    int n_embd         = 4096;
    int n_vocab        = 154880;
    int hc             = 4;       // mHC streams
    int hc_mix         = 24;      // (2 + hc) * hc: 4 pre, 4 post, 16 combine logits
    int hc_sinkhorn    = 20;
    float hc_eps       = 1e-6f;
    float rms_eps      = 1e-5f;   // all RMSNorms except the indexer key LayerNorm
    float ln_eps       = 1e-6f;   // indexer key LayerNorm

    // KDA (linear attention) layers
    int kda_heads      = 64;
    int kda_head_dim   = 128;     // d_inner = 8192
    int kda_conv       = 4;       // causal conv kernel, 3 history rows per q/k/v
    float kda_gate_lb  = -5.0f;   // g = lb * sigmoid(-(z * A))

    // MLA + DSA layers (every 4th block: 3, 7, ..., 43)
    int mla_heads      = 64;
    int mla_q_rank     = 1536;
    int mla_kv_rank    = 512;     // cached latent per token
    int mla_head_dim   = 256;     // qk_nope = v head dim; attention scale 1/sqrt(256)
    int idx_heads      = 32;
    int idx_head_dim   = 128;
    int idx_top_k      = 2048;    // cells; pools of idx_kpool cells -> up to 512 pools
    int idx_kpool      = 4;

    // FFN
    int n_dense_lead   = 3;       // blocks 0-2: dense SwiGLU (load_model: from the GGUF)
    int ffn_dense      = 12288;
    int n_expert       = 288;
    int n_expert_used  = 8;
    int ffn_expert     = 2048;
    int ffn_shared     = 2048;
    float expert_scale = 2.5f;    // after normalizing the 8 selected sigmoid weights
    float weight_norm_floor = 6.103515625e-5f;
    float swiglu_clamp = 10.0f;   // silu(min(gate, c)) * clamp(up, -c, c)

    bool is_mla(int il) const { return il % 4 == 3; }      // checked against head_count_kv[] at load
    bool is_kda(int il) const { return !is_mla(il); }
    bool is_moe(int il) const { return il >= n_dense_lead; }
    int  d_inner() const { return kda_heads * kda_head_dim; }
};

}  // namespace strata::glm
