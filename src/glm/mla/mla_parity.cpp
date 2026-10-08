// glm_mla_parity: one GLM-5.3-Flash MLA+DSA layer on the GPU against the llama.cpp oracle dump.
//
//   glm_mla_parity --gguf <shard-00001.gguf> --oracle <dir> --layer 3 [--layer 7 ...]
//
// For each layer: feeds the oracle's attn_norm-<L> input of phase 0 (prefill, 30 tokens) then phases 1 and 2
// (one decode token each) through mla_forward on one cache, and compares every oracle node of the branch:
// q_resid, kv_cmpr (+ the F16 latent cache rows), q_absorbed, indexer q/k/gate/weights, pooled keys, pool
// scores, the ordered top-k pools, the per-head V (fattn_mla) and the branch output attn_out.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mla_test_util.hpp"

using namespace strata::glm;

namespace {

using namespace mla_test;

// ---- oracle -----------------------------------------------------------------------------------------------
struct Node {
    std::string file;
    int64_t ne[4];
};
struct Oracle {
    std::string dir;
    std::map<std::pair<int, std::string>, Node> nodes;   // first occurrence of (phase, name)
    explicit Oracle(const std::string & d) : dir(d) {
        std::ifstream in(d + "/index.tsv");
        if (!in) {
            std::fprintf(stderr, "cannot read %s/index.tsv\n", d.c_str());
            std::exit(1);
        }
        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            std::stringstream ss(line);
            std::string x;
            while (std::getline(ss, x, '\t')) f.push_back(x);
            if (f.size() < 10) continue;
            Node n;
            n.file = f[9];
            for (int i = 0; i < 4; ++i) n.ne[i] = std::atoll(f[5 + i].c_str());
            nodes.emplace(std::make_pair(std::atoi(f[1].c_str()), f[2]), n);
        }
    }
    std::vector<float> get(int phase, const std::string & name, int64_t expect = -1) const {
        auto it = nodes.find({phase, name});
        if (it == nodes.end()) {
            std::fprintf(stderr, "oracle: no node '%s' in phase %d\n", name.c_str(), phase);
            std::exit(1);
        }
        const Node & n = it->second;
        const int64_t cnt = n.ne[0] * n.ne[1] * n.ne[2] * n.ne[3];
        if (expect >= 0 && cnt != expect) {
            std::fprintf(stderr, "oracle: %s has %lld values, expected %lld\n", name.c_str(), (long long)cnt,
                         (long long)expect);
            std::exit(1);
        }
        std::vector<float> v(cnt);
        FILE * fp = std::fopen((dir + "/" + n.file).c_str(), "rb");
        if (!fp || std::fread(v.data(), sizeof(float), cnt, fp) != (size_t)cnt) {
            std::fprintf(stderr, "oracle: cannot read %s\n", n.file.c_str());
            std::exit(1);
        }
        std::fclose(fp);
        return v;
    }
};

// ---- comparison -------------------------------------------------------------------------------------------
struct Stat {
    double max_abs = 0, ref_max = 0, dot = 0, na = 0, nb = 0;
    int64_t n = 0;
    void add(double a, double r) {
        max_abs = std::max(max_abs, std::fabs(a - r));
        ref_max = std::max(ref_max, std::fabs(r));
        dot += a * r;
        na += a * a;
        nb += r * r;
        ++n;
    }
    double rel() const { return ref_max > 0 ? max_abs / ref_max : max_abs; }
    double cos() const { return na > 0 && nb > 0 ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0); }
};

int g_fail = 0;
double g_min_cos = 0.9999;

void report(const char * what, const Stat & s, double min_cos) {
    const bool ok = s.cos() >= min_cos;
    if (!ok) ++g_fail;
    std::printf("  %-26s n=%-8lld max_abs=%.3e rel=%.3e cos=%.7f %s\n", what, (long long)s.n, s.max_abs, s.rel(),
                s.cos(), ok ? "ok" : "FAIL");
}

Stat compare(const std::vector<float> & a, const std::vector<float> & r) {
    Stat s;
    for (size_t i = 0; i < r.size(); ++i) s.add(a[i], r[i]);
    return s;
}

int64_t g_cap = 256;   // --cap: cache capacity (the engine's 131072 exercises the two-stage decode selection)

void run_layer(Model & m, const Oracle & orc, int L) {
    const Geometry g;
    const int D = g.n_embd;
    const std::string sfx = "-" + std::to_string(L);
    std::printf("layer %d\n", L);
    const MlaWeights w = load_layer(m, L);
    const int64_t max_ctx = g_cap;
    MlaCache * cache = mla_cache_create(g, max_ctx);
    const int max_T = 64;
    void * scratch = nullptr;
    CK(cudaMalloc(&scratch, mla_scratch_bytes(g, max_T, max_ctx)));
    float *x = nullptr, *out = nullptr;
    CK(cudaMalloc(&x, sizeof(float) * max_T * D));
    CK(cudaMalloc(&out, sizeof(float) * max_T * D));

    int64_t pos0 = 0;
    for (int phase = 0; phase <= 2; ++phase) {
        const auto xin = orc.get(phase, "attn_norm" + sfx);
        const int T = (int)(xin.size() / D);
        std::printf(" phase %d: T=%d pos0=%lld\n", phase, T, (long long)pos0);
        CK(cudaMemcpy(x, xin.data(), xin.size() * sizeof(float), cudaMemcpyHostToDevice));
        Batch b;
        b.T = T;
        b.pos0 = pos0;
        mla_forward(g, w, cache, b, x, out, scratch, nullptr);
        CK(cudaDeviceSynchronize());
        const MlaTaps tp = mla_taps(g, cache, scratch);
        if (tp.T != T || tp.pos0 != pos0) {
            std::fprintf(stderr, "unexpected sub-chunking\n");
            std::exit(1);
        }

        report("q_resid", compare(d2h(tp.qr, (size_t)T * 1536), orc.get(phase, "q_resid" + sfx, T * 1536)), g_min_cos);
        const auto kv_ref = orc.get(phase, "kv_cmpr" + sfx, T * 512);
        report("kv_cmpr", compare(d2h(tp.kv, (size_t)T * 512), kv_ref), g_min_cos);
        {
            const auto raw = d2h(static_cast<const uint16_t *>(mla_cache_latent(cache)) + pos0 * 512, (size_t)T * 512);
            std::vector<float> lat(raw.size());
            for (size_t i = 0; i < raw.size(); ++i) lat[i] = half_to_float(raw[i]);
            report("latent cache (f16)", compare(lat, kv_ref), g_min_cos);
        }
        report("q_absorbed", compare(d2h(tp.q_abs, (size_t)T * 64 * 512), orc.get(phase, "q_absorbed" + sfx, T * 64 * 512)), g_min_cos);
        report("indexer_q", compare(d2h(tp.iq, (size_t)T * 4096), orc.get(phase, "indexer_q" + sfx, T * 4096)), g_min_cos);
        report("indexer_k", compare(d2h(tp.ik, (size_t)T * 128), orc.get(phase, "indexer_k" + sfx, T * 128)), g_min_cos);
        report("indexer_gate", compare(d2h(tp.ig, (size_t)T * 128), orc.get(phase, "indexer_gate" + sfx, T * 128)), g_min_cos);
        report("indexer_weights", compare(d2h(tp.iw, (size_t)T * 32), orc.get(phase, "indexer_weights" + sfx, T * 32)), g_min_cos);

        const int64_t n_pool = (pos0 + T) / 4;
        {
            const auto ref = orc.get(phase, "indexer_pool_k" + sfx);
            const auto raw = d2h(static_cast<const uint16_t *>(mla_cache_pooled(cache)), (size_t)n_pool * 128);
            Stat s;
            for (int64_t i = 0; i < n_pool * 128; ++i) s.add(half_to_float(raw[i]), ref[i]);
            report("indexer_pool_k (cache)", s, g_min_cos);
        }
        {
            const auto ref = orc.get(phase, "indexer_score" + sfx);
            const int64_t ld_ref = (int64_t)ref.size() / T;
            const auto mine = d2h(tp.score, (size_t)T * tp.n_pool_ld);
            const auto top_ref = orc.get(phase, "indexer_top_k" + sfx);
            const int64_t ld_top = (int64_t)top_ref.size() / T;
            const auto top = d2h(tp.top, (size_t)T * 512);
            const auto nsel = d2h(tp.nsel, (size_t)T);
            Stat s;
            int64_t order_mismatch = 0, sel_bad = 0, checked = 0;
            for (int t = 0; t < T; ++t) {
                const int64_t nv = (pos0 + t + 1) / 4;
                for (int64_t p = 0; p < nv; ++p) s.add(mine[t * tp.n_pool_ld + p], ref[t * ld_ref + p]);
                const int64_t nt = std::min<int64_t>(nv, 512);
                for (int64_t i = 0; i < nt; ++i) {
                    ++checked;
                    if (top[t * 512 + i] != (int32_t)top_ref[t * ld_top + i]) ++order_mismatch;
                }
                if (nsel[t] != nt * 4 + (pos0 + t + 1) % 4) ++sel_bad;
            }
            report("indexer_score (visible)", s, g_min_cos);
            std::printf("  %-26s %lld/%lld ranks differ, %lld bad cell counts %s\n", "indexer_top_k order",
                        (long long)order_mismatch, (long long)checked, (long long)sel_bad,
                        sel_bad == 0 ? "ok" : "FAIL");
            if (sel_bad) ++g_fail;
        }
        report("fattn_mla (per-head V)", compare(d2h(tp.v, (size_t)T * 16384),
                                                 orc.get(phase, "fattn_mla" + sfx + " (permuted) (cont)", T * 16384)), g_min_cos);
        report("attn_out", compare(d2h(out, (size_t)T * D), orc.get(phase, "attn_out" + sfx, T * D)), g_min_cos);
        pos0 += T;
    }
    mla_cache_destroy(cache);
    cudaFree(scratch);
    cudaFree(x);
    cudaFree(out);
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, oracle;
    std::vector<int> layers;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layers.push_back(std::atoi(argv[++i]));
        else if (a == "--min-cos" && i + 1 < argc) g_min_cos = std::atof(argv[++i]);
        else if (a == "--cap" && i + 1 < argc) g_cap = std::max<int64_t>(256, std::atoll(argv[++i]));
        else {
            std::fprintf(stderr, "usage: %s --gguf <shard1> --oracle <dir> --layer L [--layer L2] [--min-cos c] [--cap N]\n", argv[0]);
            return 2;
        }
    }
    if (gguf.empty() || oracle.empty() || layers.empty()) {
        std::fprintf(stderr, "usage: %s --gguf <shard1> --oracle <dir> --layer L\n", argv[0]);
        return 2;
    }
    Model m(gguf);
    Oracle orc(oracle);
    for (int L : layers) run_layer(m, orc, L);
    std::printf("%s (%d failing checks, min cos %.6f)\n", g_fail ? "FAIL" : "PASS", g_fail, g_min_cos);
    return g_fail ? 1 : 0;
}
