// glm_mla_select_test: long-context checks of the MLA+DSA layer that the 30-token oracle cannot reach.
//
//   glm_mla_select_test --gguf <shard1> --oracle <dir> [--layer 3] [--tokens 6000] [--long 140000] [--latent q8]
//
// --latent q8: every cache is a Q8 one (int8 latent); the references use the dequantized rows the attention sees,
// and the cached rows are checked against the F32 kv_cmpr within the quantization bound.
// 1. Pruning: N tokens (> 2052, so tokens see more than 512 pools) built from mixtures of the oracle's real
//    attn_norm rows go through mla_forward (prefill in uneven batches, a T=4 verify batch, T=1 decodes). For the
//    checked tokens of every batch a float64 CPU reference implementing llama.cpp's semantics recomputes from the
//    GPU's own inputs (indexer q/weights taps, F16 caches): the pooled keys, every visible pool score, the top-512
//    pool set, the cell list (pools + incomplete tail), and the latent attention output.
// 2. --long: a pseudo-random cache of that many tokens (past 32768 pools: the selector's fallback path) plus a
//    decode step, checked the same way.
// 3. Truncate: two caches see the same prefix; one appends a token that completes a pool, is cut back before it
//    and continues with different tokens; the other goes straight there. Outputs and caches must be identical.

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mla_test_util.hpp"
#include "strata/glm/matmul.hpp"

using namespace strata::glm;
using namespace mla_test;

namespace {

int g_fail = 0;

void check(bool ok, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
void check(bool ok, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::printf(" %s\n", ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// oracle rows of attn_norm-<L>, phases 0-2 (32 tokens), used as a basis for realistic inputs
std::vector<float> oracle_rows(const std::string & dir, int L, int D) {
    std::ifstream in(dir + "/index.tsv");
    std::string line, want = "attn_norm-" + std::to_string(L);
    std::vector<float> rows;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t')) f.push_back(x);
        if (f.size() < 10 || f[2] != want) continue;
        const int64_t n = std::atoll(f[5].c_str()) * std::atoll(f[6].c_str());
        std::vector<float> v(n);
        FILE * fp = std::fopen((dir + "/" + f[9]).c_str(), "rb");
        if (!fp || std::fread(v.data(), 4, n, fp) != (size_t)n) {
            std::fprintf(stderr, "cannot read %s\n", f[9].c_str());
            std::exit(1);
        }
        std::fclose(fp);
        rows.insert(rows.end(), v.begin(), v.end());
    }
    if (rows.size() < (size_t)D) {
        std::fprintf(stderr, "no attn_norm-%d rows in the oracle\n", L);
        std::exit(1);
    }
    return rows;
}

// token rows: random signed mixtures of 3 oracle rows plus 10% noise, rescaled to the first row's RMS
std::vector<float> make_inputs(const std::vector<float> & basis, int D, int64_t n, uint32_t seed) {
    const int64_t nb = (int64_t)basis.size() / D;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> pick(0, nb - 1);
    std::normal_distribution<float> nd(0.f, 1.f);
    double rms0 = 0;
    for (int i = 0; i < D; ++i) rms0 += basis[i] * basis[i];
    rms0 = std::sqrt(rms0 / D);
    std::vector<float> x((size_t)n * D);
    for (int64_t t = 0; t < n; ++t) {
        float * r = &x[(size_t)t * D];
        for (int k = 0; k < 3; ++k) {
            const float * b = &basis[(size_t)pick(rng) * D];
            const float wgt = nd(rng);
            for (int i = 0; i < D; ++i) r[i] += wgt * b[i];
        }
        double rms = 0;
        for (int i = 0; i < D; ++i) rms += r[i] * r[i];
        rms = std::sqrt(rms / D);
        for (int i = 0; i < D; ++i) r[i] = (float)(r[i] / rms * rms0) + 0.1f * (float)rms0 * nd(rng);
    }
    return x;
}

std::vector<float> f16_vec(const void * dev, size_t n) {
    const auto raw = d2h(static_cast<const uint16_t *>(dev), n);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = half_to_float(raw[i]);
    return v;
}

// latent rows [0, n) as the attention sees them (dequantized for a Q8 cache)
std::vector<float> latent_vec(const MlaCache * c, int64_t n) {
    void * tmp = nullptr;
    CK(cudaMalloc(&tmp, (size_t)std::max<int64_t>(n, 1) * 512 * 2));
    mla_cache_latent_f16(c, 0, n, tmp, nullptr);
    CK(cudaDeviceSynchronize());
    auto v = f16_vec(tmp, (size_t)n * 512);
    cudaFree(tmp);
    return v;
}

MlaLatent g_latent = MlaLatent::F16;   // --latent q8: every cache of the test is a Q8 one

struct Checker {
    const Geometry & g;
    const MlaWeights & w;
    std::vector<float> ape;   // [4][128]
    explicit Checker(const Geometry & g_, const MlaWeights & w_) : g(g_), w(w_) {
        ape = d2h(static_cast<const float *>(w.idx_ape.data), 4 * 128);
    }

    // checks `tokens` (indices into the last sub-chunk) of the last mla_forward on cache c
    void run(const char * label, MlaCache * c, void * scratch, const std::vector<int> & tokens) {
        const MlaTaps tp = mla_taps(g, c, scratch);
        const int64_t n = mla_cache_size(c);
        const int64_t n_pool = n / 4;
        const auto kg = f16_vec(mla_cache_kg(c), (size_t)n * 256);
        const auto pooled = f16_vec(mla_cache_pooled(c), (size_t)n_pool * 128);
        const auto latent = latent_vec(c, n);
        const int T = tp.T;
        const auto iq = d2h(tp.iq, (size_t)T * 4096);
        const auto iw = d2h(tp.iw, (size_t)T * 32);
        const auto qabs = d2h(tp.q_abs, (size_t)T * 64 * 512);
        const auto o = d2h(tp.o, (size_t)T * 64 * 512);
        const auto top = d2h(tp.top, (size_t)T * 512);
        const auto sel = d2h(tp.sel, (size_t)T * tp.sel_ld);
        const auto nsel = d2h(tp.nsel, (size_t)T);
        const auto score = d2h(tp.score, (size_t)T * tp.n_pool_ld);
        const auto kv = d2h(tp.kv, (size_t)T * 512);

        // cached latent rows of the sub-chunk vs the F32 kv_cmpr they were written from: F16 rounding, or (Q8) half a
        // step of the group's scale max |x| / 127 plus the F16 rounding of scale and product
        double lat_err = 0;
        for (int t = 0; t < T; ++t)
            for (int g0 = 0; g0 < 512; g0 += 64) {
                double amax = 0;
                for (int k = g0; k < g0 + 64; ++k) amax = std::max(amax, (double)std::fabs(kv[(size_t)t * 512 + k]));
                for (int k = g0; k < g0 + 64; ++k) {
                    const double x = kv[(size_t)t * 512 + k], e = std::fabs(latent[(tp.pos0 + t) * 512 + k] - x);
                    const double bound = mla_cache_latent_type(c) == MlaLatent::Q8 ? 0.5 * amax / 127 * 1.01 + 2e-3 * amax
                                                                                     : 1e-3 * std::fabs(x) + 1e-7;
                    lat_err = std::max(lat_err, e / bound);
                }
            }
        check(lat_err <= 1.0, "  [%s] cached latent rows vs F32 kv: max err %.2f of the bound", label, lat_err);

        // pooled keys of every complete pool, from the F16 key|gate rows (pooling is in F32 on the GPU)
        double pool_err = 0;
        for (int64_t p = 0; p < n_pool; ++p) {
            for (int d = 0; d < 128; ++d) {
                double lg[4], m = -1e300;
                for (int j = 0; j < 4; ++j) {
                    lg[j] = (double)kg[(p * 4 + j) * 256 + 128 + d] + ape[j * 128 + d];
                    m = std::max(m, lg[j]);
                }
                double se = 0, acc = 0;
                for (int j = 0; j < 4; ++j) {
                    const double e = std::exp(lg[j] - m);
                    se += e;
                    acc += e * kg[(p * 4 + j) * 256 + d];
                }
                const double ref = acc / se;
                pool_err = std::max(pool_err, std::fabs(ref - pooled[p * 128 + d]) / std::max(1.0, std::fabs(ref)));
            }
        }
        check(pool_err < 2e-3, "  [%s] pooled keys (%lld pools) max rel err %.2e", label, (long long)n_pool, pool_err);

        double score_err = 0, attn_err = 0, attn_cos_min = 1;
        int64_t set_diff = 0, hard_diff = 0, cells_bad = 0, scored = 0;
        for (int t : tokens) {
            const int64_t pos = tp.pos0 + t;
            const int64_t nv = (pos + 1) / 4;
            // scores of every visible pool, float64 from the same F32 q / weights and F16 pooled keys
            std::vector<double> sref(nv);
            double smax = 0;
            for (int64_t p = 0; p < nv; ++p) {
                double s = 0;
                for (int h = 0; h < 32; ++h) {
                    double dot = 0;
                    for (int d = 0; d < 128; ++d) dot += (double)iq[(size_t)t * 4096 + h * 128 + d] * pooled[p * 128 + d];
                    s += std::max(dot, 0.0) * iw[t * 32 + h];
                }
                sref[p] = s;
                smax = std::max(smax, std::fabs(s));
            }
            for (int64_t p = 0; p < nv; ++p)
                score_err = std::max(score_err, std::fabs(sref[p] - score[(size_t)t * tp.n_pool_ld + p]) / std::max(smax, 1e-30));
            scored += nv;
            // top-512 set: GPU vs reference; a difference is tolerated only between near-tied scores
            const int64_t ntop = std::min<int64_t>(nv, 512);
            std::vector<int64_t> order(nv);
            for (int64_t p = 0; p < nv; ++p) order[p] = p;
            std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return sref[a] > sref[b] || (sref[a] == sref[b] && a < b); });
            std::set<int64_t> ref_set(order.begin(), order.begin() + ntop), gpu_set;
            for (int64_t i = 0; i < ntop; ++i) gpu_set.insert(top[(size_t)t * 512 + i]);
            const double thr = sref[order[ntop - 1]];
            for (int64_t p : gpu_set) {
                if (ref_set.count(p)) continue;
                ++set_diff;
                if (p < 0 || p >= nv || std::fabs(sref[p] - thr) > 2e-3 * smax) ++hard_diff;
            }
            // cell list: 4 cells per selected pool + the incomplete tail, all causal and distinct
            const int n_tail = (int)((pos + 1) % 4);
            std::set<int64_t> cells;
            for (int i = 0; i < nsel[t]; ++i) cells.insert(sel[(size_t)t * tp.sel_ld + i]);
            bool ok = nsel[t] == ntop * 4 + n_tail && (int64_t)cells.size() == nsel[t];
            for (int64_t p : gpu_set)
                for (int j = 0; j < 4; ++j) ok = ok && cells.count(p * 4 + j);
            for (int k = 0; k < n_tail; ++k) ok = ok && cells.count(pos - k);
            for (int64_t cpos : cells) ok = ok && cpos >= 0 && cpos <= pos;
            if (!ok) ++cells_bad;
            // latent attention over the GPU's cell list, float64
            for (int h = 0; h < 64; ++h) {
                const float * q = &qabs[((size_t)t * 64 + h) * 512];
                std::vector<double> s;
                double m = -1e300;
                for (int64_t cpos : cells) {
                    double d = 0;
                    for (int k = 0; k < 512; ++k) d += (double)q[k] * latent[cpos * 512 + k];
                    s.push_back(d / 16.0);
                    m = std::max(m, d / 16.0);
                }
                double se = 0;
                for (double & v : s) se += (v = std::exp(v - m));
                std::vector<double> ref(512, 0.0);
                size_t i = 0;
                for (int64_t cpos : cells) {
                    for (int k = 0; k < 512; ++k) ref[k] += s[i] / se * latent[cpos * 512 + k];
                    ++i;
                }
                double dot = 0, na = 0, nb = 0, emax = 0, rmax = 0;
                for (int k = 0; k < 512; ++k) {
                    const double a = o[((size_t)t * 64 + h) * 512 + k];
                    dot += a * ref[k];
                    na += a * a;
                    nb += ref[k] * ref[k];
                    emax = std::max(emax, std::fabs(a - ref[k]));
                    rmax = std::max(rmax, std::fabs(ref[k]));
                }
                attn_err = std::max(attn_err, emax / std::max(rmax, 1e-30));
                attn_cos_min = std::min(attn_cos_min, dot / std::sqrt(na * nb));
            }
        }
        check(score_err < 2e-3, "  [%s] pool scores: %lld checked, max err %.2e of the row's max |score|", label,
              (long long)scored, score_err);
        check(hard_diff == 0, "  [%s] top-512 sets: %lld pools differ from the float64 reference (%lld not near-ties)",
              label, (long long)set_diff, (long long)hard_diff);
        check(cells_bad == 0, "  [%s] cell lists (pools + tail, causal, distinct): %lld bad of %zu", label,
              (long long)cells_bad, tokens.size());
        check(attn_cos_min > 0.99995, "  [%s] latent attention vs float64: max rel err %.2e, min cos %.7f", label,
              attn_err, attn_cos_min);
    }
};

std::vector<int> sample(int T, int k) {
    std::vector<int> v;
    if (T <= k) {
        for (int i = 0; i < T; ++i) v.push_back(i);
        return v;
    }
    for (int i = 0; i < k; ++i) v.push_back((int)((int64_t)i * (T - 1) / (k - 1)));
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, oracle;
    int layer = 3;
    int64_t n_tok = 6000, n_long = 0, cap_min = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) oracle = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) n_tok = std::atoll(argv[++i]);
        else if (a == "--long" && i + 1 < argc) n_long = std::atoll(argv[++i]);
        else if (a == "--cap" && i + 1 < argc) cap_min = std::atoll(argv[++i]);
        else if (a == "--latent" && i + 1 < argc) g_latent = std::string(argv[++i]) == "q8" ? MlaLatent::Q8 : MlaLatent::F16;
        else {
            std::fprintf(stderr, "usage: %s --gguf <shard1> --oracle <dir> [--layer L] [--tokens N] [--long N] [--cap N] [--latent f16|q8]\n", argv[0]);
            return 2;
        }
    }
    if (gguf.empty() || oracle.empty()) {
        std::fprintf(stderr, "usage: %s --gguf <shard1> --oracle <dir> [--layer L] [--tokens N] [--long N] [--cap N] [--latent f16|q8]\n", argv[0]);
        return 2;
    }
    const Geometry g;
    const int D = g.n_embd;
    Model m(gguf);
    const MlaWeights w = load_layer(m, layer);
    Checker chk(g, w);
    const auto basis = oracle_rows(oracle, layer, D);
    const int max_T = 1024;

    // ---- 1. pruning over a long prompt ------------------------------------------------------------------
    {
        std::printf("pruning: layer %d, %lld tokens\n", layer, (long long)n_tok);
        const int64_t cap = n_tok + 16;
        const auto X = make_inputs(basis, D, cap, 1);
        // --cap: the engine's larger cache capacity switches decode batches to the two-stage selection
        MlaCache * c = mla_cache_create(g, std::max(cap, cap_min), g_latent);
        void * scratch = nullptr;
        CK(cudaMalloc(&scratch, mla_scratch_bytes(g, max_T, std::max(cap, cap_min))));
        float *x = nullptr, *out = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * cap * D));
        CK(cudaMalloc(&out, sizeof(float) * max_T * D));
        CK(cudaMemcpy(x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        // uneven prefill batches (internal sub-chunks of 512), then verify T=4 and two decode steps
        std::vector<int> batches;
        int64_t left = n_tok - 6;
        for (int b : {1000, 777, 1333}) {
            while (left > 0) {
                const int T = (int)std::min<int64_t>(b, left);
                batches.push_back(T);
                left -= T;
                if (batches.size() % 3 == 0) break;
            }
        }
        while (left > 0) {
            const int T = (int)std::min<int64_t>(1000, left);
            batches.push_back(T);
            left -= T;
        }
        batches.push_back(4);
        batches.push_back(1);
        batches.push_back(1);
        int64_t pos = 0;
        for (int T : batches) {
            Batch b;
            b.T = T;
            b.pos0 = pos;
            mla_forward(g, w, c, b, x + pos * D, out, scratch, nullptr);
            CK(cudaDeviceSynchronize());
            pos += T;
            const MlaTaps tp = mla_taps(g, c, scratch);
            if (T > 8 && pos < n_tok - 1500) continue;   // check the long-context batches only
            char label[64];
            std::snprintf(label, sizeof label, "pos %lld..%lld T=%d", (long long)tp.pos0, (long long)(tp.pos0 + tp.T - 1), T);
            chk.run(label, c, scratch, sample(tp.T, 12));
        }
        mla_cache_destroy(c);
        cudaFree(scratch);
        cudaFree(x);
        cudaFree(out);
    }

    // ---- 2. very long context: the selector's fallback (more than 32768 pools) ----------------------------
    if (n_long > 0) {
        std::printf("long context: %lld random cached tokens + decode\n", (long long)n_long);
        MlaCache * c = mla_cache_create(g, n_long + 8, g_latent);
        void * scratch = nullptr;
        CK(cudaMalloc(&scratch, mla_scratch_bytes(g, 4, n_long + 8)));
        float *x = nullptr, *out = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * 4 * D));
        CK(cudaMalloc(&out, sizeof(float) * 4 * D));
        const auto X = make_inputs(basis, D, 4, 2);
        CK(cudaMemcpy(x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        mla_cache_fill_random(c, w, n_long, 99, nullptr);
        Batch b;
        b.T = 3;
        b.pos0 = n_long;
        mla_forward(g, w, c, b, x, out, scratch, nullptr);
        CK(cudaDeviceSynchronize());
        chk.run("long", c, scratch, {0, 1, 2});
        mla_cache_destroy(c);
        cudaFree(scratch);
        cudaFree(x);
        cudaFree(out);
    }

    // ---- 3. truncate --------------------------------------------------------------------------------------
    {
        const int64_t n0 = 4 * 700 + 2;   // the next two tokens complete pool 700
        std::printf("truncate: prefix %lld tokens\n", (long long)n0);
        const auto X = make_inputs(basis, D, n0 + 16, 3);
        float * x = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * X.size()));
        CK(cudaMemcpy(x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        const float * xa1 = x + (n0 + 0) * D;   // shared next token
        const float * xa2 = x + (n0 + 1) * D;   // completes pool 700, then cut away
        const float * xb = x + (n0 + 2) * D;    // 4 replacement tokens
        const int64_t cap = n0 + 16;
        void * s1 = nullptr;
        void * s2 = nullptr;
        CK(cudaMalloc(&s1, mla_scratch_bytes(g, max_T, cap)));
        CK(cudaMalloc(&s2, mla_scratch_bytes(g, max_T, cap)));
        float *o1 = nullptr, *o2 = nullptr;
        CK(cudaMalloc(&o1, sizeof(float) * cap * D));   // the prefix is one batch of n0 tokens
        CK(cudaMalloc(&o2, sizeof(float) * cap * D));
        MlaCache * A = mla_cache_create(g, cap, g_latent);
        MlaCache * B = mla_cache_create(g, cap, g_latent);
        auto fwd = [&](MlaCache * c, const float * xin, int T, int64_t p0, void * sc, float * o) {
            Batch b;
            b.T = T;
            b.pos0 = p0;
            mla_forward(g, w, c, b, xin, o, sc, nullptr);
        };
        fwd(A, x, (int)n0, 0, s1, o1);
        fwd(B, x, (int)n0, 0, s2, o2);
        fwd(A, xa1, 1, n0, s1, o1);
        fwd(B, xa1, 1, n0, s2, o2);
        // A: append the pool-completing token and two more, then cut back to n0 + 1
        fwd(A, xa2, 1, n0 + 1, s1, o1);
        fwd(A, xa2, 1, n0 + 2, s1, o1);
        fwd(A, xa2, 1, n0 + 3, s1, o1);
        mla_cache_truncate(A, n0 + 1);
        // both continue with the replacement tokens: a T=4 verify batch then a decode step
        fwd(A, xb, 4, n0 + 1, s1, o1);
        fwd(B, xb, 4, n0 + 1, s2, o2);
        CK(cudaDeviceSynchronize());
        const auto ra = d2h(o1, (size_t)4 * D), rb = d2h(o2, (size_t)4 * D);
        bool same = std::memcmp(ra.data(), rb.data(), ra.size() * 4) == 0;
        check(same, "  verify batch after truncate: outputs bitwise identical");
        // and a cut deeper than a pool boundary, re-prefilled as one batch on A vs token by token on B
        mla_cache_truncate(A, n0 - 9);
        mla_cache_truncate(B, n0 - 9);
        fwd(A, x + (n0 - 9) * D, 12, n0 - 9, s1, o1);
        for (int i = 0; i < 12; ++i) fwd(B, x + (n0 - 9 + i) * D, 1, n0 - 9 + i, s2, o2);
        CK(cudaDeviceSynchronize());
        const int64_t n = mla_cache_size(A);
        check(n == mla_cache_size(B) && n == n0 + 3, "  cache sizes after deep cut + refill: %lld / %lld",
              (long long)n, (long long)mla_cache_size(B));
        const auto la = latent_vec(A, n), lb = latent_vec(B, n);
        const auto pa = f16_vec(mla_cache_pooled(A), (size_t)(n / 4) * 128), pb = f16_vec(mla_cache_pooled(B), (size_t)(n / 4) * 128);
        double el = 0, ep = 0;
        for (size_t i = 0; i < la.size(); ++i) el = std::max(el, (double)std::fabs(la[i] - lb[i]));
        for (size_t i = 0; i < pa.size(); ++i) ep = std::max(ep, (double)std::fabs(pa[i] - pb[i]));
        // batch (T=12 GEMV) vs token-by-token (T=1 GEMV) projections round differently: compare at F16 precision
        check(el < 2e-2 && ep < 2e-2, "  latent / pooled caches, batch vs token refill: max abs diff %.2e / %.2e", el, ep);
        const auto fa = d2h(o1 + 11 * D, (size_t)D), fb = d2h(o2, (size_t)D);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < D; ++i) {
            dot += (double)fa[i] * fb[i];
            na += (double)fa[i] * fa[i];
            nb += (double)fb[i] * fb[i];
        }
        check(dot / std::sqrt(na * nb) > 0.9999, "  last-token output, batch vs token refill: cos %.7f", dot / std::sqrt(na * nb));

        // ---- 4. shared selection (MTP index_share_mtp): step 0's cells fed back as MlaSelection ----------------
        {
            const float * xs = x + (n0 + 6) * D;
            int32_t * dsel = nullptr;
            int32_t * dnsel = nullptr;
            CK(cudaMalloc(&dsel, sizeof(int32_t) * 2052));
            CK(cudaMalloc(&dnsel, sizeof(int32_t)));
            fwd(A, xs, 1, n, s1, o1);   // step 0: own indexer
            const MlaTaps tp = mla_taps(g, A, s1);
            CK(cudaMemcpy(dsel, tp.sel, sizeof(int32_t) * tp.sel_ld, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dnsel, tp.nsel, sizeof(int32_t), cudaMemcpyDeviceToDevice));
            const auto r0 = d2h(o1, (size_t)D);
            MlaSelection sh;
            sh.sel = dsel;
            sh.nsel = dnsel;
            sh.sel_ld = tp.sel_ld;
            mla_cache_truncate(A, n);
            Batch b;
            b.T = 1;
            b.pos0 = n;
            mla_forward(g, w, A, b, xs, o1, s1, nullptr, &sh);
            CK(cudaDeviceSynchronize());
            const auto r1 = d2h(o1, (size_t)D);
            check(std::memcmp(r0.data(), r1.data(), r0.size() * 4) == 0,
                  "  shared selection = own selection at the same position: output bitwise identical");
            // x prepared by the caller (the trunk's fused Q8) = x prepared inside
            void * xbuf = nullptr;
            CK(cudaMalloc(&xbuf, matmul_prepare_bytes(D, 1)));
            const MatmulInput xin = matmul_prepare(xs, D, 1, 1, D, 0, xbuf, nullptr);
            mla_cache_truncate(A, n);
            mla_forward(g, w, A, b, xs, o1, s1, nullptr, nullptr, &xin);
            CK(cudaDeviceSynchronize());
            const auto r2 = d2h(o1, (size_t)D);
            check(std::memcmp(r0.data(), r2.data(), r0.size() * 4) == 0,
                  "  caller-prepared x input: output bitwise identical");
            cudaFree(xbuf);
            // next step attends over step 0's cells only, but still writes and pools its own cache row
            fwd(B, xs, 1, n, s2, o2);
            fwd(B, xs + D, 1, n + 1, s2, o2);
            b.pos0 = n + 1;
            mla_forward(g, w, A, b, xs + D, o1, s1, nullptr, &sh);
            CK(cudaDeviceSynchronize());
            const auto lA = latent_vec(A, n + 2), lB = latent_vec(B, n + 2);
            const auto pA = f16_vec(mla_cache_pooled(A), (size_t)((n + 2) / 4) * 128),
                       pB = f16_vec(mla_cache_pooled(B), (size_t)((n + 2) / 4) * 128);
            check(std::memcmp(lA.data() + (size_t)n * 512, lB.data() + (size_t)n * 512, 2 * 512 * sizeof(float)) == 0 &&
                      std::memcmp(pA.data(), pB.data(), pA.size() * sizeof(float)) == 0,
                  "  shared-selection step writes the same latent rows and pools as a normal step");
            cudaFree(dsel);
            cudaFree(dnsel);
        }
        mla_cache_destroy(A);
        mla_cache_destroy(B);
        cudaFree(s1);
        cudaFree(s2);
        cudaFree(o1);
        cudaFree(o2);
        cudaFree(x);
    }

    // ---- 5. prefix blobs: export, import into a cache that held another sequence / kept its own prefix ------
    {
        const int64_t n1 = 4 * 500 + 3, n2 = 4 * 700 + 2, cap = n2 + 16;   // both cuts inside a pool
        std::printf("export / import: ranges [0, %lld) [%lld, %lld) + tail\n", (long long)n1, (long long)n1,
                    (long long)n2);
        const auto X = make_inputs(basis, D, cap, 5), Y = make_inputs(basis, D, cap, 6);
        float *x = nullptr, *y = nullptr, *o1 = nullptr, *o2 = nullptr;
        CK(cudaMalloc(&x, sizeof(float) * X.size()));
        CK(cudaMalloc(&y, sizeof(float) * Y.size()));
        CK(cudaMemcpy(x, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(y, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&o1, sizeof(float) * cap * D));
        CK(cudaMalloc(&o2, sizeof(float) * cap * D));
        void * s1 = nullptr;
        void * s2 = nullptr;
        CK(cudaMalloc(&s1, mla_scratch_bytes(g, max_T, cap)));
        CK(cudaMalloc(&s2, mla_scratch_bytes(g, max_T, cap)));
        auto fwd = [&](MlaCache * c, const float * xin, int T, int64_t p0, void * sc, float * o) {
            Batch b;
            b.T = T;
            b.pos0 = p0;
            mla_forward(g, w, c, b, xin, o, sc, nullptr);
        };
        MlaCache * A = mla_cache_create(g, cap, g_latent);
        fwd(A, x, (int)n1, 0, s1, o1);
        fwd(A, x + n1 * D, (int)(n2 - n1), n1, s1, o1);
        const size_t b0 = mla_cache_rows_bytes(A, 0, n1), b1 = mla_cache_rows_bytes(A, n1, n2);
        const size_t bt = mla_cache_tail_bytes(A);
        char * h = nullptr;
        CK(cudaMallocHost(&h, b0 + b1 + bt));
        mla_cache_export_rows(A, 0, n1, h, nullptr);
        mla_cache_export_rows(A, n1, n2, h + b0, nullptr);
        mla_cache_export_tail(A, n2, h + b0 + b1, nullptr);
        CK(cudaDeviceSynchronize());
        // the same 5 tokens after the prefix: a T=4 batch then a decode step
        auto tail_out = [&](MlaCache * c, void * sc, float * o) {
            fwd(c, x + n2 * D, 4, n2, sc, o);
            fwd(c, x + (n2 + 4) * D, 1, n2 + 4, sc, o + 4 * D);
            CK(cudaDeviceSynchronize());
            return d2h(o, (size_t)5 * D);
        };
        auto same_cache = [&](MlaCache * a, MlaCache * b, const char * what) {
            const int64_t n = mla_cache_size(a);
            const auto la = latent_vec(a, n), lb = latent_vec(b, n);
            const auto pa = f16_vec(mla_cache_pooled(a), (size_t)(n / 4) * 128),
                       pb = f16_vec(mla_cache_pooled(b), (size_t)(n / 4) * 128);
            check(n == mla_cache_size(b) && la == lb && pa == pb, "  %s: latent rows and pools bitwise identical", what);
        };
        const auto ra = tail_out(A, s1, o1);

        // C held another sequence past n2 (stale key | gate ring rows at the tail positions): cut to 0, import
        MlaCache * C = mla_cache_create(g, cap, g_latent);
        fwd(C, y, (int)(n2 + 5), 0, s2, o2);
        mla_cache_truncate(C, 0);
        check(!mla_cache_import_rows(C, 0, n1 + 1, h, nullptr), "  import with a range other than the blob's refused");
        check(mla_cache_import_rows(C, 0, n1, h, nullptr) && mla_cache_import_rows(C, n1, n2, h + b0, nullptr) &&
                  mla_cache_import_tail(C, n2, h + b0 + b1, nullptr),
              "  import of the two ranges + tail");
        check(!mla_cache_can_truncate(C, n1) && mla_cache_can_truncate(C, n1 / 4 * 4) &&
                  mla_cache_can_truncate(C, n2 - 1) && mla_cache_can_truncate(C, n2),
              "  imported cache: cuts needing key | gate rows of the imported ranges refused");
        const auto rc = tail_out(C, s2, o2);
        check(std::memcmp(ra.data(), rc.data(), ra.size() * 4) == 0,
              "  imported into a cache of another sequence: next 5 outputs bitwise identical");
        same_cache(A, C, "imported cache after 5 more tokens");

        // D computed [0, n1) itself, imports [n1, n2) + tail
        MlaCache * Dc = mla_cache_create(g, cap, g_latent);
        fwd(Dc, y, (int)(n2 + 5), 0, s2, o2);
        mla_cache_truncate(Dc, 0);
        fwd(Dc, x, (int)n1, 0, s2, o2);
        check(mla_cache_import_rows(Dc, n1, n2, h + b0, nullptr) && mla_cache_import_tail(Dc, n2, h + b0 + b1, nullptr),
              "  import of the second range on its own prefix");
        check(mla_cache_can_truncate(Dc, n1) && !mla_cache_can_truncate(Dc, n1 + 2),
              "  own prefix: cut at n1 allowed, inside the imported range refused");
        const auto rd = tail_out(Dc, s2, o2);
        check(std::memcmp(ra.data(), rd.data(), ra.size() * 4) == 0,
              "  imported on its own prefix: next 5 outputs bitwise identical");
        same_cache(A, Dc, "own prefix + import after 5 more tokens");
        // a cut that needs only held rows, then the same tokens again
        mla_cache_truncate(Dc, n1);
        mla_cache_truncate(A, n1);
        fwd(Dc, x + n1 * D, 5, n1, s2, o2);
        fwd(A, x + n1 * D, 5, n1, s1, o1);
        CK(cudaDeviceSynchronize());
        same_cache(A, Dc, "cut back to n1 and refilled");

        mla_cache_destroy(A);
        mla_cache_destroy(C);
        mla_cache_destroy(Dc);
        cudaFreeHost(h);
        for (void * p : {(void *)x, (void *)y, (void *)o1, (void *)o2, s1, s2}) cudaFree(p);
    }
    std::printf("%s (%d failing checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
