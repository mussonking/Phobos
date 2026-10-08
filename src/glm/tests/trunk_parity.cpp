// glm_trunk_parity: the engine trunk (embedding, mHC pre/post + Sinkhorn, norms, dense FFN, router, shared expert,
// stream mean + output head) against the llama.cpp oracle on the 30-token prompt (phase 0). The attention outputs
// and the routed-expert sums are fed from the oracle by default, so every shared op is checked layer by layer
// without the error of the other components; --real-attn runs the merged kda/mla layers instead, --cpu-experts the
// ggml-cpu reference experts (any GGUF expert type) for the decode token only (--cpu-experts-prefill: also for
// the prompt, which reads nearly every expert of the cold GGUF). --decode runs the next token (phase 1) at T.
//
//   glm_trunk_parity [--layers N] [--real-attn] [--cpu-experts] [--decode] [--verbose] [--gguf SHARD1] [--oracle DIR]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "forward.hpp"
#include "ref_experts.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/oracle.hpp"

using namespace strata::glm;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Worst {
    double cos = 1, rel = 0;
    int il = -1;
};

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf = "/mnt/models-cold/glm53-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";
    std::string odir = "/mnt/models/glm-oracle/q4kxl-p32";
    int n_layers = 45;
    bool real_attn = false, verbose = false, cpu_experts = false, cpu_prefill = false, decode = false;
    bool pack_types = false;   // --pack-types: pack GGUF trunk types (BF16 list) on the oracle GGUF
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) odir = argv[++i];
        else if (a == "--layers" && i + 1 < argc) n_layers = std::atoi(argv[++i]);
        else if (a == "--real-attn") real_attn = true;
        else if (a == "--verbose") verbose = true;
        else if (a == "--pack-types") pack_types = true;
        else if (a == "--cpu-experts") cpu_experts = true;
        else if (a == "--cpu-experts-prefill") cpu_experts = cpu_prefill = true;
        else if (a == "--decode") decode = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    LoadOptions opt;
    if (pack_types) opt.as_bf16 = pack_recipe_bf16();
    opt.load_mtp = false;
    std::unique_ptr<Model> m;
    try {
        m = load_model(gguf, opt, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "LOAD FAILED: %s\n", e.what());
        return 1;
    }
    const Geometry & g = m->g;
    test::Oracle oracle(odir);

    std::vector<int32_t> toks;
    {
        FILE * f = std::fopen((odir + "/tokens.txt").c_str(), "r");
        int v;
        while (f && std::fscanf(f, "%d", &v) == 1) toks.push_back(v);
        if (f) std::fclose(f);
    }
    const int T = (int) oracle.node(0, "inp_embd").ne[1];
    if ((int) toks.size() < T + 2) {
        std::fprintf(stderr, "tokens.txt has %zu ids, need %d\n", toks.size(), T + 2);
        return 1;
    }
    auto ws = workspace_create(*m, T);

    // ---- hooks -------------------------------------------------------------------------------------------
    auto upload = [&](float * dst, const std::vector<float> & v) {
        ck(cudaMemcpyAsync(dst, v.data(), v.size() * 4, cudaMemcpyHostToDevice, st), "upload");
        ck(cudaStreamSynchronize(st), "upload");
    };

    std::vector<KdaState *> kda(g.n_layer, nullptr);
    std::vector<MlaCache *> mla(g.n_layer, nullptr);
    void * attn_scratch = nullptr;
    if (real_attn) {
        size_t sb = std::max(kda_scratch_bytes(g, T), mla_scratch_bytes(g, T, 64));
        ck(cudaMalloc(&attn_scratch, sb), "attn scratch");
        for (int il = 0; il < g.n_layer; ++il) {
            if (m->layers[il].mla) mla[il] = mla_cache_create(g, 64);
            else {
                kda[il] = kda_state_create(g);
                kda_state_reset(kda[il], st);
            }
        }
    }

    int phase = 0;
    ForwardHooks hk;
    hk.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * out, const Batch & b, cudaStream_t s) {
        const LayerWeights & L = m->layers[il];
        if (real_attn) {
            if (L.mla) mla_forward(g, L.mla_w, mla[il], b, y, out, attn_scratch, s, nullptr, y_in);
            else kda_forward(g, L.kda, kda[il], b, y, out, attn_scratch, s, y_in);
            return;
        }
        upload(out, oracle.load(phase, (L.mla ? "attn_out-" : "kda_out-") + std::to_string(il)));
    };
    hk.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T_,
                     cudaStream_t s) {
        if (!cpu_experts || (phase == 0 && !cpu_prefill)) {
            upload(out, oracle.load(phase, "ffn_moe_out-" + std::to_string(il)));
            return;
        }
        const int K = g.n_expert_used;
        std::vector<float> hx((size_t) T_ * g.n_embd), hw((size_t) T_ * K), ho(hx.size());
        std::vector<int32_t> hid((size_t) T_ * K);
        ck(cudaStreamSynchronize(s), "experts sync");
        ck(cudaMemcpy(hx.data(), y, hx.size() * 4, cudaMemcpyDeviceToHost), "experts x");
        ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "experts ids");
        ck(cudaMemcpy(hw.data(), w, hw.size() * 4, cudaMemcpyDeviceToHost), "experts w");
        test::ref_experts_cpu(g, m->layers[il].experts, hx.data(), hid.data(), hw.data(), T_, ho.data());
        upload(out, ho);
    };

    // ---- taps vs oracle ----------------------------------------------------------------------------------
    std::map<std::string, Worst> worst;
    std::vector<std::string> order;
    int route_mismatch = 0, route_total = 0;
    auto record = [&](const std::string & key, int il, const test::Stats & s) {
        if (!worst.count(key)) order.push_back(key);
        Worst & w = worst[key];
        if (s.cosine < w.cos || std::isnan(s.cosine)) { w.cos = s.cosine; w.il = il; }
        w.rel = std::max(w.rel, s.max_abs / std::max(s.ref_rms, 1e-30));
        if (verbose) {
            char lbl[64];
            std::snprintf(lbl, sizeof lbl, "%s-%d", key.c_str(), il);
            s.print(lbl);
        }
    };
    auto fetch = [&](const float * dev, int64_t n) {
        std::vector<float> h((size_t) n);
        ck(cudaStreamSynchronize(st), "tap sync");
        ck(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost), "tap copy");
        return h;
    };
    // our [T][24] -> oracle pre [T][4], post [T][4], comb [T][16]
    auto check_hcp = [&](int il, const std::vector<float> & h, int occ) {
        const char * names[3] = {"hc_pre", "hc_post", "hc_comb"};
        const int off[3] = {0, 4, 8}, len[3] = {4, 4, 16};
        for (int k = 0; k < 3; ++k) {
            const std::vector<float> ref = oracle.load(phase, std::string(names[k]) + "-" + std::to_string(il), occ);
            const int nt = (int) (h.size() / g.hc_mix);
            std::vector<float> a((size_t) nt * len[k]);
            for (int t = 0; t < nt; ++t)
                for (int j = 0; j < len[k]; ++j) a[(size_t) t * len[k] + j] = h[(size_t) t * g.hc_mix + off[k] + j];
            record(std::string(names[k]) + (occ ? "(ffn)" : "(attn)"), il,
                   test::compare(a.data(), ref.data(), std::min(a.size(), ref.size())));
        }
    };
    hk.tap = [&](int il, const char * node, const float * dev, int64_t n) {
        const std::string nd = node;
        std::vector<float> h = fetch(dev, n);
        if (nd == "hcp_attn" || nd == "hcp_ffn") {
            check_hcp(il, h, nd == "hcp_ffn" ? 1 : 0);
            return;
        }
        if ((nd == "kda_out" || nd == "attn_out") && !real_attn) return;   // fed from the oracle
        if (nd == "ffn_moe_out" && (!cpu_experts || (phase == 0 && !cpu_prefill))) return;
        const std::string oname = il >= 0 ? nd + "-" + std::to_string(il) : nd;
        const test::OracleNode & on = oracle.node(phase, oname);
        std::vector<float> ref = oracle.load(on);
        if (nd == "ffn_moe_weights_scaled") {
            // also the selected ids (ffn_moe_topk; the dumper wrote the i32 ids as float values)
            const int nt = (int) (h.size() / g.n_expert_used);
            std::vector<int32_t> ids((size_t) nt * g.n_expert_used);
            ck(cudaMemcpy(ids.data(), ws->ids, ids.size() * 4, cudaMemcpyDeviceToHost), "ids");
            std::vector<float> rid = oracle.load(phase, "ffn_moe_topk-" + std::to_string(il));
            for (int t = 0; t < nt; ++t) {
                std::set<int> a, b;
                for (int k = 0; k < g.n_expert_used; ++k) {
                    a.insert(ids[(size_t) t * g.n_expert_used + k]);
                    b.insert((int) rid[(size_t) t * g.n_expert_used + k]);
                }
                route_total++;
                if (a != b) {
                    route_mismatch++;
                    std::printf("  route mismatch phase %d il=%d t=%d:", phase, il, t);
                    for (int e : a) if (!b.count(e)) std::printf(" +%d", e);
                    for (int e : b) if (!a.count(e)) std::printf(" -%d", e);
                    std::printf("\n");
                }
            }
        }
        size_t cmp = std::min(h.size(), ref.size());
        const float * a = h.data();
        if (h.size() != ref.size()) {   // the oracle keeps only the last token at the end (l_out-44)
            a = h.data() + (h.size() - ref.size());
            cmp = ref.size();
        }
        record(nd, il, test::compare(a, ref.data(), cmp));
    };

    // ---- run ---------------------------------------------------------------------------------------------
    auto report = [&](int n_tok, int expect) {
        std::printf("trunk parity phase %d, %d layers, T=%d, attention %s, routed experts %s\n", phase, n_layers, n_tok,
                    real_attn ? "real (kda/mla)" : "from the oracle",
                    cpu_experts && (phase == 1 || cpu_prefill) ? "ggml-cpu reference" : "from the oracle");
        for (const auto & k : order) {
            const Worst & w = worst[k];
            std::printf("  %-26s worst cos=%.9f (layer %d)  max_abs/rms=%.3e\n", k.c_str(), w.cos, w.il, w.rel);
        }
        std::printf("  router top-8 sets: %d / %d tokens differ\n", route_mismatch, route_total);
        if (n_layers == g.n_layer) {
            std::vector<float> lg = fetch(ws->logits, g.n_vocab);
            const int am = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            std::printf("  greedy next token %d (oracle %d) %s\n", am, expect, am == expect ? "MATCH" : "DIFFERENT");
        }
        worst.clear();
        order.clear();
        route_mismatch = route_total = 0;
    };
    forward_trunk(*m, *ws, toks.data(), T, 0, n_layers, hk, st);
    ck(cudaStreamSynchronize(st), "forward");
    report(T, toks[T]);
    if (decode) {
        if (!real_attn) std::printf("(decode with oracle-fed attention: the states are not exercised)\n");
        phase = 1;
        forward_trunk(*m, *ws, toks.data() + T, 1, T, n_layers, hk, st);
        ck(cudaStreamSynchronize(st), "decode");
        report(1, toks[T + 1]);
    }
    for (auto * k : kda) if (k) kda_state_destroy(k);
    for (auto * c : mla) if (c) mla_cache_destroy(c);
    if (attn_scratch) cudaFree(attn_scratch);
    return 0;
}
