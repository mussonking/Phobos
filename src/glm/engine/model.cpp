// src/glm/engine/model.cpp - see model.hpp.
#include "model.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>

#include "strata/glm/matmul.hpp"

namespace strata::glm {
namespace {

constexpr int kF32 = 0, kF16 = 1, kQ8_0 = 8, kBF16 = 30;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm loader: ") + what + ": " + cudaGetErrorString(e));
}

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

// ---- metadata -------------------------------------------------------------------------------------------

struct MetaCheck {
    const GgufSet & f;
    std::vector<std::string> errors;

    const MetaValue * get(const std::string & key) {
        const MetaValue * v = f.meta(key);
        if (!v) errors.push_back("missing metadata " + key);
        return v;
    }
    void num(const std::string & key, double want, double tol = 0) {
        const MetaValue * v = get("glm5-next." + key);
        if (v && std::fabs(v->num() - want) > tol + 1e-12 * std::fabs(want)) {
            std::ostringstream o;
            o << "glm5-next." << key << " = " << v->num() << ", engine expects " << want;
            errors.push_back(o.str());
        }
    }
};

// Depth from the GGUF, every other dimension stays the real one: trunk blocks = block_count - nextn_predict_layers,
// and leading_dense_block_count. A truncated test model (tools/glm/mini_gguf.py: same formats and expert count, a
// few blocks) loads through the same code; the layer kinds still follow is_mla, checked against head_count_kv.
// Values that do not make sense are left to validate_metadata to report.
void read_depth(const GgufSet & f, Geometry & g) {
    const MetaValue * bc = f.meta("glm5-next.block_count");
    const MetaValue * nn = f.meta("glm5-next.nextn_predict_layers");
    const MetaValue * dl = f.meta("glm5-next.leading_dense_block_count");
    if (!bc || !nn || !dl || (int) nn->num() != g.n_layer_mtp) return;
    const int n = (int) bc->num() - g.n_layer_mtp, lead = (int) dl->num();
    if (n < 1 || lead < 0 || lead > n) return;
    g.n_layer = n;
    g.n_dense_lead = lead;
}

void validate_metadata(const GgufSet & f, const Geometry & g) {
    MetaCheck c{f, {}};
    if (const MetaValue * a = c.get("general.architecture"); a && a->s != "glm5-next")
        c.errors.push_back("general.architecture = " + a->s + ", engine expects glm5-next");
    c.num("block_count", g.n_layer + g.n_layer_mtp);
    c.num("nextn_predict_layers", g.n_layer_mtp);
    c.num("embedding_length", g.n_embd);
    c.num("vocab_size", g.n_vocab);
    c.num("attention.head_count", g.kda_heads);
    c.num("attention.head_count", g.mla_heads);
    c.num("attention.layer_norm_rms_epsilon", g.rms_eps, 1e-9);
    c.num("attention.layer_norm_epsilon", g.ln_eps, 1e-10);
    c.num("attention.q_lora_rank", g.mla_q_rank);
    c.num("attention.kv_lora_rank", g.mla_kv_rank);
    c.num("attention.key_length_mla", g.mla_head_dim);
    c.num("attention.value_length_mla", g.mla_head_dim);
    c.num("rope.dimension_count", 0);
    c.num("attention.indexer.head_count", g.idx_heads);
    c.num("attention.indexer.key_length", g.idx_head_dim);
    c.num("attention.indexer.top_k", g.idx_top_k);
    c.num("attention.indexer.kpool", g.idx_kpool);
    c.num("hyper_connection.count", g.hc);
    c.num("hyper_connection.sinkhorn_iterations", g.hc_sinkhorn);
    c.num("hyper_connection.epsilon", g.hc_eps, 1e-10);
    c.num("kda.head_dim", g.kda_head_dim);
    c.num("ssm.conv_kernel", g.kda_conv);
    c.num("kda.gate_lower_bound", g.kda_gate_lb, 1e-6);
    c.num("feed_forward_length", g.ffn_dense);
    c.num("expert_feed_forward_length", g.ffn_expert);
    c.num("expert_shared_feed_forward_length", g.ffn_shared);
    c.num("expert_shared_count", 1);
    c.num("expert_count", g.n_expert);
    c.num("expert_used_count", g.n_expert_used);
    c.num("expert_group_count", 1);
    c.num("expert_group_used_count", 1);
    c.num("expert_gating_func", 2);   // sigmoid
    c.num("expert_weights_norm", 1);
    c.num("expert_weights_scale", g.expert_scale, 1e-6);
    c.num("leading_dense_block_count", g.n_dense_lead);
    const int n_all = g.n_layer + g.n_layer_mtp;
    if (const MetaValue * a = c.get("glm5-next.attention.head_count_kv")) {
        if ((int) a->count != n_all) c.errors.push_back("head_count_kv has " + std::to_string(a->count) + " entries");
        for (int il = 0; il < (int) a->items.size() && il < n_all; ++il) {
            const bool mla = a->items[il].num() != 0;
            const bool want = il >= g.n_layer || g.is_mla(il);
            if (mla != want)
                c.errors.push_back("head_count_kv[" + std::to_string(il) + "] says " + (mla ? "MLA" : "KDA") +
                                   ", geometry says " + (want ? "MLA" : "KDA"));
        }
    }
    for (const char * k : {"glm5-next.swiglu_clamp_exp", "glm5-next.swiglu_clamp_shexp"})
        if (const MetaValue * a = c.get(k))
            for (const auto & it : a->items)
                if (std::fabs(it.num() - g.swiglu_clamp) > 1e-6) {
                    c.errors.push_back(std::string(k) + " entry " + std::to_string(it.num()) + " != " +
                                       std::to_string(g.swiglu_clamp));
                    break;
                }
    if (const MetaValue * a = f.meta("glm5-next.attention.indexer.types"))
        c.errors.push_back("indexer.types present (" + std::to_string(a->count) +
                           " entries): shared indexers are not implemented");
    if (!c.errors.empty()) {
        std::string msg = "GGUF does not match the GLM-5.3-Flash geometry:";
        for (const auto & e : c.errors) msg += "\n  " + e;
        throw std::runtime_error(msg);
    }
}

// ---- tensor plan ----------------------------------------------------------------------------------------

enum Kind { kVec, kMat };   // kVec: F32 only (norms, biases, conv taps...); kMat: anything glm_matmul takes

struct Want {
    std::string name;
    WeightRef * dst;
    int64_t ne[3];
    Kind kind;
};

struct Planner {
    const GgufSet & f;
    std::vector<Want> wants;
    std::vector<std::string> errors;

    void add(const std::string & name, WeightRef & dst, int64_t ne0, int64_t ne1, int64_t ne2, Kind k) {
        wants.push_back({name, &dst, {ne0, ne1, ne2}, k});
    }
    void blk(int il, const std::string & suffix, WeightRef & dst, int64_t ne0, int64_t ne1, int64_t ne2, Kind k) {
        add("blk." + std::to_string(il) + "." + suffix, dst, ne0, ne1, ne2, k);
    }
};

void plan_layer(Planner & p, const Geometry & g, LayerWeights & L, bool dense_mtp) {
    const int il = L.il, D = g.n_embd;
    if (L.mtp && !dense_mtp) return;
    p.blk(il, "attn_norm.weight", L.attn_norm, D, 1, 1, kVec);
    p.blk(il, "ffn_norm.weight", L.ffn_norm, D, 1, 1, kVec);
    if (!L.mtp) {
        const int hd = g.hc * D, mix = g.hc_mix;
        p.blk(il, "hc_attn_fn.weight", L.hc_attn_fn, hd, mix, 1, kMat);
        p.blk(il, "hc_attn_base.weight", L.hc_attn_base, mix, 1, 1, kVec);
        p.blk(il, "hc_attn_scale.weight", L.hc_attn_scale, 3, 1, 1, kVec);
        p.blk(il, "hc_ffn_fn.weight", L.hc_ffn_fn, hd, mix, 1, kMat);
        p.blk(il, "hc_ffn_base.weight", L.hc_ffn_base, mix, 1, 1, kVec);
        p.blk(il, "hc_ffn_scale.weight", L.hc_ffn_scale, 3, 1, 1, kVec);
    } else {
        p.blk(il, "nextn.eh_proj.weight", L.eh_proj, 2 * D, D, 1, kMat);
        p.blk(il, "nextn.enorm.weight", L.enorm, D, 1, 1, kVec);
        p.blk(il, "nextn.hnorm.weight", L.hnorm, D, 1, 1, kVec);
        p.blk(il, "nextn.shared_head_norm.weight", L.shared_head_norm, D, 1, 1, kVec);
    }
    if (!L.mla) {
        KdaWeights & k = L.kda;
        const int di = g.d_inner();
        p.blk(il, "attn_q.weight", k.q, D, di, 1, kMat);
        p.blk(il, "attn_k.weight", k.k, D, di, 1, kMat);
        p.blk(il, "attn_v.weight", k.v, D, di, 1, kMat);
        p.blk(il, "attn_output.weight", k.out, di, D, 1, kMat);
        p.blk(il, "ssm_conv1d_q.weight", k.conv_q, g.kda_conv, 1, di, kVec);
        p.blk(il, "ssm_conv1d_k.weight", k.conv_k, g.kda_conv, 1, di, kVec);
        p.blk(il, "ssm_conv1d_v.weight", k.conv_v, g.kda_conv, 1, di, kVec);
        p.blk(il, "ssm_f_a.weight", k.f_a, D, g.kda_head_dim, 1, kMat);
        p.blk(il, "ssm_f_b.weight", k.f_b, g.kda_head_dim, di, 1, kMat);
        p.blk(il, "ssm_dt.bias", k.dt_bias, di, 1, 1, kVec);
        p.blk(il, "ssm_a", k.a, g.kda_heads, 1, 1, kVec);
        p.blk(il, "ssm_beta.weight", k.beta, D, g.kda_heads, 1, kMat);
        p.blk(il, "ssm_g_a.weight", k.g_a, D, g.kda_head_dim, 1, kMat);
        p.blk(il, "ssm_g_b.weight", k.g_b, g.kda_head_dim, di, 1, kMat);
        p.blk(il, "ssm_norm.weight", k.norm, g.kda_head_dim, 1, 1, kVec);
    } else {
        MlaWeights & m = L.mla_w;
        const int qd = g.mla_heads * g.mla_head_dim, ih = g.idx_heads * g.idx_head_dim;
        p.blk(il, "attn_q_a.weight", m.q_a, D, g.mla_q_rank, 1, kMat);
        p.blk(il, "attn_q_a_norm.weight", m.q_a_norm, g.mla_q_rank, 1, 1, kVec);
        p.blk(il, "attn_q_b.weight", m.q_b, g.mla_q_rank, qd, 1, kMat);
        p.blk(il, "attn_kv_a_mqa.weight", m.kv_a, D, g.mla_kv_rank, 1, kMat);
        p.blk(il, "attn_kv_a_norm.weight", m.kv_a_norm, g.mla_kv_rank, 1, 1, kVec);
        p.blk(il, "attn_k_b.weight", m.k_b, g.mla_head_dim, g.mla_kv_rank, g.mla_heads, kMat);
        p.blk(il, "attn_v_b.weight", m.v_b, g.mla_kv_rank, g.mla_head_dim, g.mla_heads, kMat);
        p.blk(il, "attn_output.weight", m.out, qd, D, 1, kMat);
        p.blk(il, "indexer.attn_q_b.weight", m.idx_q_b, g.mla_q_rank, ih, 1, kMat);
        p.blk(il, "indexer.attn_k.weight", m.idx_k, D, g.idx_head_dim, 1, kMat);
        p.blk(il, "indexer.k_norm.weight", m.idx_k_norm_w, g.idx_head_dim, 1, 1, kVec);
        p.blk(il, "indexer.k_norm.bias", m.idx_k_norm_b, g.idx_head_dim, 1, 1, kVec);
        p.blk(il, "indexer.proj.weight", m.idx_proj, D, g.idx_heads, 1, kMat);
        p.blk(il, "indexer_compressor_gate.weight", m.idx_gate, D, g.idx_head_dim, 1, kMat);
        p.blk(il, "indexer_compressor_ape.weight", m.idx_ape, g.idx_head_dim, g.idx_kpool, 1, kVec);
    }
    if (!L.moe) {
        p.blk(il, "ffn_gate.weight", L.ffn_gate, D, g.ffn_dense, 1, kMat);
        p.blk(il, "ffn_up.weight", L.ffn_up, D, g.ffn_dense, 1, kMat);
        p.blk(il, "ffn_down.weight", L.ffn_down, g.ffn_dense, D, 1, kMat);
    } else {
        p.blk(il, "ffn_gate_inp.weight", L.gate_inp, D, g.n_expert, 1, kMat);
        p.blk(il, "exp_probs_b.bias", L.exp_probs_b, g.n_expert, 1, 1, kVec);
        p.blk(il, "ffn_gate_shexp.weight", L.gate_shexp, D, g.ffn_shared, 1, kMat);
        p.blk(il, "ffn_up_shexp.weight", L.up_shexp, D, g.ffn_shared, 1, kMat);
        p.blk(il, "ffn_down_shexp.weight", L.down_shexp, g.ffn_shared, D, 1, kMat);
    }
}

bool mat_type_ok(uint32_t t) { return t == kQ8_0 || t == kBF16 || t == kF16 || t == kF32; }

}  // namespace

Model::~Model() {
    if (arena) cudaFree(arena);
}

std::unique_ptr<Model> load_model(const std::string & shard1, const LoadOptions & opt, cudaStream_t stream) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    auto m = std::make_unique<Model>();
    m->gguf = std::make_unique<GgufSet>(shard1);
    const GgufSet & f = *m->gguf;
    read_depth(f, m->g);
    const Geometry & g = m->g;
    validate_metadata(f, g);

    const int n_all = g.n_layer + g.n_layer_mtp;
    m->layers.resize(n_all);
    Planner p{f, {}, {}};
    for (int il = 0; il < n_all; ++il) {
        LayerWeights & L = m->layers[il];
        L.il = il;
        L.mtp = il >= g.n_layer;
        L.mla = L.mtp || g.is_mla(il);
        L.moe = L.mtp || g.is_moe(il);
        plan_layer(p, g, L, opt.load_mtp);
    }
    p.add("output_norm.weight", m->output_norm, g.n_embd, 1, 1, kVec);
    p.add("output.weight", m->output, g.n_embd, g.n_vocab, 1, kMat);

    // Shapes and types of every planned dense tensor, and the arena size.
    std::vector<std::string> & errors = p.errors;
    std::vector<HostTensor> host(p.wants.size());
    std::vector<size_t> dev_bytes(p.wants.size());
    std::vector<char> to_bf16(p.wants.size(), 0);
    size_t arena = 0, staging = 0;
    for (size_t i = 0; i < p.wants.size(); ++i) {
        const Want & w = p.wants[i];
        if (!f.has(w.name)) {
            errors.push_back("missing tensor " + w.name);
            continue;
        }
        host[i] = f.tensor(w.name);
        const HostTensor & h = host[i];
        int64_t ne[3] = {1, 1, 1};
        for (size_t d = 0; d < h.shape.size() && d < 3; ++d) ne[d] = (int64_t) h.shape[d];
        if (h.shape.size() > 3 || ne[0] != w.ne[0] || ne[1] != w.ne[1] || ne[2] != w.ne[2]) {
            std::ostringstream o;
            o << w.name << " shape [" << ne[0] << "," << ne[1] << "," << ne[2] << "], expected [" << w.ne[0] << ","
              << w.ne[1] << "," << w.ne[2] << "]";
            errors.push_back(o.str());
            continue;
        }
        if (w.kind == kVec ? h.type != kF32 : !mat_type_ok(h.type)) {
            errors.push_back(w.name + " has ggml type " + std::to_string(h.type) +
                             (w.kind == kVec ? ", expected F32" : ", glm_matmul takes Q8_0/BF16/F16/F32"));
            continue;
        }
        if (h.type == kQ8_0)
            for (const auto & suf : opt.as_bf16)
                if (w.name.size() > suf.size() && w.name.compare(w.name.size() - suf.size(), suf.size(), suf) == 0 &&
                    w.name[w.name.size() - suf.size() - 1] == '.')
                    to_bf16[i] = true;
        const bool repack = opt.repack_q8 && h.type == kQ8_0 && !to_bf16[i];
        dev_bytes[i] = to_bf16[i] ? h.nbytes / 34 * 32 * 2 : repack ? matmul_repack_bytes(h.ref(nullptr)) : h.nbytes;
        if (repack) staging = std::max<size_t>(staging, h.nbytes);
        arena += align256(dev_bytes[i]);
    }

    // Token embedding (host) and routed experts (mapped).
    if (!f.has("token_embd.weight")) errors.push_back("missing tensor token_embd.weight");
    else {
        m->token_embd = f.tensor("token_embd.weight");
        const auto & s = m->token_embd.shape;
        if (s.size() != 2 || (int) s[0] != g.n_embd || (int) s[1] != g.n_vocab)
            errors.push_back("token_embd.weight shape mismatch");
        if (!mat_type_ok(m->token_embd.type)) errors.push_back("token_embd.weight type unsupported");
        // resident copy: a decode-time page fault on the mapping can hit the HDD (hundreds of ms)
        m->token_embd_copy.assign(m->token_embd.data, m->token_embd.data + m->token_embd.nbytes);
        m->token_embd.data = m->token_embd_copy.data();
    }
    for (LayerWeights & L : m->layers) {
        if (!L.moe) continue;
        const std::string b = "blk." + std::to_string(L.il) + ".";
        const struct { const char * n; int64_t a, c; const uint8_t ** p; int * t; size_t * bytes; } ex[3] = {
            {"ffn_gate_exps.weight", g.n_embd, g.ffn_expert, &L.experts.gate, &L.experts.type_gate,
             &L.experts.gate_bytes},
            {"ffn_up_exps.weight", g.n_embd, g.ffn_expert, &L.experts.up, &L.experts.type_up, &L.experts.up_bytes},
            {"ffn_down_exps.weight", g.ffn_expert, g.n_embd, &L.experts.down, &L.experts.type_down,
             &L.experts.down_bytes},
        };
        for (const auto & e : ex) {
            if (!f.has(b + e.n)) {
                errors.push_back("missing tensor " + b + e.n);
                continue;
            }
            const HostTensor h = f.tensor(b + e.n);
            if (h.shape.size() != 3 || (int64_t) h.shape[0] != e.a || (int64_t) h.shape[1] != e.c ||
                (int) h.shape[2] != g.n_expert) {
                errors.push_back(b + e.n + " shape mismatch");
                continue;
            }
            *e.p = h.data;
            *e.t = (int) h.type;
            *e.bytes = h.nbytes / g.n_expert;
        }
    }

    // Every GGUF tensor must be accounted for: an unknown name means a model the engine does not understand.
    {
        std::set<std::string> known;
        for (const Want & w : p.wants) known.insert(w.name);
        known.insert("token_embd.weight");
        for (const LayerWeights & L : m->layers)
            if (L.moe)
                for (const char * n : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
                    known.insert("blk." + std::to_string(L.il) + "." + n);
        int unknown = 0;
        for (size_t s = 0; s < f.shards(); ++s)
            for (const auto & t : f.shard(s).tensors())
                if (!known.count(t.name)) {
                    const bool skipped_mtp = !opt.load_mtp && t.name.rfind("blk." + std::to_string(g.n_layer) + ".", 0) == 0;
                    if (!skipped_mtp && unknown++ < 20) errors.push_back("unexpected tensor " + t.name);
                }
    }
    if (!errors.empty()) {
        std::string msg = "GGUF tensors do not match the engine:";
        for (const auto & e : errors) msg += "\n  " + e;
        throw std::runtime_error(msg);
    }

    // Upload.
    ck(cudaMalloc(&m->arena, arena), "dense arena");
    m->arena_bytes = arena;
    void * stage = nullptr;
    if (staging) ck(cudaMalloc(&stage, staging), "repack staging");
    char * cur = static_cast<char *>(m->arena);
    size_t done = 0, total = 0;
    for (const HostTensor & h : host) total += h.nbytes;
    double next_report = 0.1;
    for (size_t i = 0; i < p.wants.size(); ++i) {
        const HostTensor & h = host[i];
        const bool repack = opt.repack_q8 && h.type == kQ8_0 && !to_bf16[i];
        if (to_bf16[i]) {
            const size_t nb = h.nbytes / 34;
            std::vector<uint16_t> bf(nb * 32);
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t * q = h.data + b * 34;
                __half_raw hr;
                std::memcpy(&hr.x, q, 2);
                const float d = __half2float(__half(hr));
                for (int k = 0; k < 32; ++k) {
                    const float v = d * (float) (int8_t) q[2 + k];
                    uint32_t u;
                    std::memcpy(&u, &v, 4);
                    bf[b * 32 + k] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
                }
            }
            ck(cudaMemcpy(cur, bf.data(), bf.size() * 2, cudaMemcpyHostToDevice), "upload bf16");
            WeightRef r = h.ref(cur);
            r.type = kBF16;
            r.nbytes = bf.size() * 2;
            *p.wants[i].dst = r;
        } else if (repack) {
            ck(cudaMemcpyAsync(stage, h.data, h.nbytes, cudaMemcpyHostToDevice, stream), "upload");
            *p.wants[i].dst = matmul_repack(h.ref(stage), cur, stream);
            ck(cudaStreamSynchronize(stream), "repack");   // the staging buffer is reused by the next tensor
        } else {
            ck(cudaMemcpyAsync(cur, h.data, h.nbytes, cudaMemcpyHostToDevice, stream), "upload");
            *p.wants[i].dst = h.ref(cur);
        }
        cur += align256(dev_bytes[i]);
        done += h.nbytes;
        if (opt.verbose && (double) done / total >= next_report) {
            const double s = std::chrono::duration<double>(clock::now() - t0).count();
            std::fprintf(stderr, "glm loader: %3.0f%% of %.2f GB dense, %.1f s\n", 100.0 * done / total, total / 1e9, s);
            next_report += 0.1;
        }
    }
    ck(cudaStreamSynchronize(stream), "upload");
    if (stage) cudaFree(stage);
    if (opt.verbose)
        std::fprintf(stderr, "glm loader: %zu dense tensors, %.2f GB in VRAM, %.1f s\n", p.wants.size(), arena / 1e9,
                     std::chrono::duration<double>(clock::now() - t0).count());
    return m;
}

void embed_rows_host(const Model & m, const int32_t * ids, int T, float * out) {
    const HostTensor & e = m.token_embd;
    const int D = m.g.n_embd;
    const size_t row = ggml_nbytes(e.type, {(uint64_t) D});
    for (int t = 0; t < T; ++t) {
        if (ids[t] < 0 || ids[t] >= m.g.n_vocab) throw std::runtime_error("token id out of range");
        const uint8_t * r = e.data + (size_t) ids[t] * row;
        float * o = out + (size_t) t * D;
        switch (e.type) {
            case kQ8_0:
                for (int b = 0; b < D / 32; ++b) {
                    __half_raw hr;
                    std::memcpy(&hr.x, r + b * 34, 2);
                    const float d = __half2float(__half(hr));
                    for (int j = 0; j < 32; ++j) o[b * 32 + j] = d * (float) (int8_t) r[b * 34 + 2 + j];
                }
                break;
            case kF32: std::memcpy(o, r, (size_t) D * 4); break;
            case kF16:
                for (int i = 0; i < D; ++i) {
                    __half_raw hr;
                    std::memcpy(&hr.x, r + 2 * i, 2);
                    o[i] = __half2float(__half(hr));
                }
                break;
            default:   // BF16
                for (int i = 0; i < D; ++i) {
                    uint32_t u = 0;
                    std::memcpy(reinterpret_cast<char *>(&u) + 2, r + 2 * i, 2);
                    std::memcpy(&o[i], &u, 4);
                }
        }
    }
}

}  // namespace strata::glm
