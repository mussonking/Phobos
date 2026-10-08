// src/glm/kda/kda_test.cpp - KDA parity against the llama.cpp oracle dump, chunked-vs-sequential check,
// rollback/snapshot check and V100 timings.
//
//   kda_test parity  --gguf <shard-00001> --oracle <dir> [--layers 0,4]
//   kda_test chunk   [--T 2048]                       synthetic inputs, sequential vs chunked recurrence
//   kda_test rollback --gguf <shard-00001> [--layer 4] speculative rollback + snapshot/restore
//   kda_test bench   --gguf <shard-00001> [--layer 4]  decode T=1/4/8 and prefill timings
//
// Oracle: index.tsv (seq, phase, name, op, type, ne0..ne3, file), <seq>.f32 raw F32 in ggml order.
#include "kda_internal.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/matmul.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::glm;
using namespace strata::glm::kda;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::string arg(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

// ---- GGUF weights ------------------------------------------------------------------------------------------

struct Model {
    std::vector<std::unique_ptr<strata::GgufFile>> shards;
    explicit Model(const std::string & shard1) {
        const auto p = shard1.find("-00001-of-");
        if (p == std::string::npos) { shards.emplace_back(new strata::GgufFile(shard1)); return; }
        const int n = std::atoi(shard1.c_str() + p + 10);
        for (int i = 1; i <= n; ++i) {
            char num[16];
            std::snprintf(num, sizeof num, "%05d", i);
            std::string f = shard1;
            f.replace(p + 1, 5, num);
            shards.emplace_back(new strata::GgufFile(f));
        }
    }
    WeightRef upload(const std::string & name, std::vector<void *> & owned) const {
        for (const auto & s : shards) {
            const auto * t = s->find(name);
            if (!t) continue;
            int be = 0, bb = 0;
            if (!strata::block_geometry(t->type, be, bb)) { std::fprintf(stderr, "type? %s\n", name.c_str()); std::exit(1); }
            WeightRef w;
            w.type = (int) t->type;
            for (size_t d = 0; d < t->shape.size() && d < 4; ++d) w.ne[d] = (int64_t) t->shape[d];
            w.nbytes = t->elements() / be * bb;
            void * dp = nullptr;
            ck(cudaMalloc(&dp, w.nbytes), "cudaMalloc weight");
            ck(cudaMemcpy(dp, s->tensor_data(*t), w.nbytes, cudaMemcpyHostToDevice), "upload weight");
            owned.push_back(dp);
            w.data = dp;
            return w;
        }
        std::fprintf(stderr, "tensor %s not found\n", name.c_str());
        std::exit(1);
    }
};

struct LayerWeights {
    KdaWeights w;
    std::vector<void *> owned;
    ~LayerWeights() { for (void * p : owned) cudaFree(p); }
};

bool g_repack = true;   // pass Q8_0R refs like the engine's loader (--raw 1: raw Q8_0)

// Q8_0 -> Q8_0R in a fresh buffer, as src/glm/engine/model.cpp does at load time
void repack(WeightRef & w, std::vector<void *> & owned) {
    if (!g_repack || w.type != 8) return;
    void * dst = nullptr;
    ck(cudaMalloc(&dst, matmul_repack_bytes(w)), "cudaMalloc repack");
    owned.push_back(dst);
    w = matmul_repack(w, dst, 0);
    ck(cudaDeviceSynchronize(), "repack");
}

std::unique_ptr<LayerWeights> load_layer(const Model & m, int il) {
    auto lw = std::make_unique<LayerWeights>();
    auto n = [&](const char * s) { return "blk." + std::to_string(il) + "." + s; };
    auto & w = lw->w;
    auto & o = lw->owned;
    w.q = m.upload(n("attn_q.weight"), o);
    w.k = m.upload(n("attn_k.weight"), o);
    w.v = m.upload(n("attn_v.weight"), o);
    w.out = m.upload(n("attn_output.weight"), o);
    w.conv_q = m.upload(n("ssm_conv1d_q.weight"), o);
    w.conv_k = m.upload(n("ssm_conv1d_k.weight"), o);
    w.conv_v = m.upload(n("ssm_conv1d_v.weight"), o);
    w.f_a = m.upload(n("ssm_f_a.weight"), o);
    w.f_b = m.upload(n("ssm_f_b.weight"), o);
    w.dt_bias = m.upload(n("ssm_dt.bias"), o);
    w.a = m.upload(n("ssm_a"), o);
    w.beta = m.upload(n("ssm_beta.weight"), o);
    w.g_a = m.upload(n("ssm_g_a.weight"), o);
    w.g_b = m.upload(n("ssm_g_b.weight"), o);
    w.norm = m.upload(n("ssm_norm.weight"), o);
    for (WeightRef * r : {&w.q, &w.k, &w.v, &w.out, &w.f_a, &w.f_b, &w.g_a, &w.g_b, &w.beta}) repack(*r, o);
    return lw;
}

// ---- oracle --------------------------------------------------------------------------------------------------

struct Oracle {
    std::string dir;
    struct Node { int seq; int64_t ne[4]; };
    std::map<std::pair<int, std::string>, std::vector<Node>> nodes;
    explicit Oracle(const std::string & d) : dir(d) {
        std::ifstream f(d + "/index.tsv");
        if (!f) { std::fprintf(stderr, "cannot open %s/index.tsv\n", d.c_str()); std::exit(1); }
        std::string line;
        while (std::getline(f, line)) {
            std::vector<std::string> c;
            std::stringstream ss(line);
            std::string x;
            while (std::getline(ss, x, '\t')) c.push_back(x);
            if (c.size() < 10) continue;
            Node n{std::atoi(c[0].c_str()), {std::atoll(c[5].c_str()), std::atoll(c[6].c_str()), std::atoll(c[7].c_str()),
                                             std::atoll(c[8].c_str())}};
            nodes[{std::atoi(c[1].c_str()), c[2]}].push_back(n);
        }
    }
    std::vector<float> get(int phase, const std::string & name, int which = 0, int64_t * ne = nullptr) const {
        auto it = nodes.find({phase, name});
        if (it == nodes.end() || (int) it->second.size() <= which) {
            std::fprintf(stderr, "oracle node phase %d '%s' #%d missing\n", phase, name.c_str(), which);
            std::exit(1);
        }
        const Node & n = it->second[which];
        const size_t cnt = (size_t) (n.ne[0] * n.ne[1] * n.ne[2] * n.ne[3]);
        std::vector<float> v(cnt);
        std::FILE * fp = std::fopen((dir + "/" + std::to_string(n.seq) + ".f32").c_str(), "rb");
        if (!fp || std::fread(v.data(), 4, cnt, fp) != cnt) { std::fprintf(stderr, "read %d.f32 failed\n", n.seq); std::exit(1); }
        std::fclose(fp);
        if (ne) std::memcpy(ne, n.ne, sizeof n.ne);
        return v;
    }
};

// ---- comparison --------------------------------------------------------------------------------------------

struct Cmp { double max_abs = 0, rel_max = 0, max_rel = 0, cos = 0; };

Cmp compare(const float * a, const float * ref, size_t n) {
    double dot = 0, na = 0, nr = 0, mref = 0;
    Cmp c;
    for (size_t i = 0; i < n; ++i) mref = std::max(mref, (double) std::fabs(ref[i]));
    const double floor = 1e-3 * mref;   // elementwise rel error ignores values below 0.1% of the tensor's max
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - ref[i]);
        c.max_abs = std::max(c.max_abs, d);
        c.max_rel = std::max(c.max_rel, d / std::max((double) std::fabs(ref[i]), floor));
        dot += (double) a[i] * ref[i];
        na += (double) a[i] * a[i];
        nr += (double) ref[i] * ref[i];
    }
    c.rel_max = mref > 0 ? c.max_abs / mref : c.max_abs;
    c.cos = (na > 0 && nr > 0) ? dot / std::sqrt(na * nr) : (na == nr ? 1.0 : 0.0);
    return c;
}

std::vector<float> d2h(const float * d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "d2h");
    return h;
}

bool report(const char * what, const Cmp & c, double cos_min, double relmax_max) {
    const bool ok = c.cos >= cos_min && c.rel_max <= relmax_max;
    std::printf("  %-14s max_abs %.3e  max_abs/max|ref| %.3e  max_rel %.3e  cos %.9f  %s\n", what, c.max_abs, c.rel_max,
                c.max_rel, c.cos, ok ? "ok" : "FAIL");
    return ok;
}

void l2_heads(std::vector<float> & x) {
    for (size_t h = 0; h < x.size() / kDim; ++h) {
        double s = 0;
        for (int i = 0; i < kDim; ++i) s += (double) x[h * kDim + i] * x[h * kDim + i];
        const float r = (float) (1.0 / std::sqrt(s + 1e-6));
        for (int i = 0; i < kDim; ++i) x[h * kDim + i] *= r;
    }
}

// ---- parity -----------------------------------------------------------------------------------------------

int run_parity(int argc, char ** argv) {
    const Model m(arg(argc, argv, "--gguf", ""));
    const Oracle orc(arg(argc, argv, "--oracle", "/mnt/models/glm-oracle/q4kxl-p32"));
    std::vector<int> layers;
    {
        std::stringstream ss(arg(argc, argv, "--layers", "0,4"));
        std::string x;
        while (std::getline(ss, x, ',')) layers.push_back(std::atoi(x.c_str()));
    }
    Geometry geo;
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const int maxT = 64;
    void * scratch = nullptr;
    ck(cudaMalloc(&scratch, kda_scratch_bytes(geo, maxT)), "scratch");
    float * dx = nullptr, * dout = nullptr;
    ck(cudaMalloc(&dx, (size_t) maxT * kEmbd * 4), "x");
    ck(cudaMalloc(&dout, (size_t) maxT * kEmbd * 4), "out");

    bool all_ok = true;
    for (int fused = 1; fused >= 0; --fused)
    for (int il : layers) {
        set_fused_decode(fused != 0);
        std::printf("== path: %s\n", fused ? "fused decode kernel" : "separate kernels");
        auto lw = load_layer(m, il);
        KdaState * s = kda_state_create(geo);
        kda_state_reset(s, st);
        const std::string L = "-" + std::to_string(il);
        for (int phase = 0; phase < 3; ++phase) {
            int64_t ne[4];
            const auto x = orc.get(phase, "attn_norm" + L, 0, ne);
            const int T = (int) ne[1];
            ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x h2d");
            Batch b;
            b.T = T;
            kda_forward(geo, lw->w, s, b, dx, dout, scratch, st);
            ck(cudaStreamSynchronize(st), "forward");
            const Views & v = last_views();
            const size_t n8 = (size_t) T * kInner;
            std::printf("layer %d phase %d (T=%d)\n", il, phase, T);
            bool ok = true;
            if (v.k) {   // the fused prefill path does not materialize the prepared q/k/v/g/beta
                auto rq = orc.get(phase, "kda_q_conv" + L);
                auto rk = orc.get(phase, "kda_k_conv" + L);
                l2_heads(rq);
                l2_heads(rk);
                ok &= report("q (l2)", compare(d2h(v.q, n8).data(), rq.data(), n8), 0.9999, 2e-2);
                ok &= report("k (l2)", compare(d2h(v.k, n8).data(), rk.data(), n8), 0.9999, 2e-2);
                ok &= report("v_conv", compare(d2h(v.v, n8).data(), orc.get(phase, "kda_v_conv" + L).data(), n8), 0.9999, 2e-2);
                ok &= report("g1", compare(d2h(v.g, n8).data(), orc.get(phase, "kda_g1" + L).data(), n8), 0.9999, 2e-2);
                ok &= report("beta", compare(d2h(v.beta, (size_t) T * kHeads).data(), orc.get(phase, "kda_beta" + L).data(),
                                             (size_t) T * kHeads), 0.9999, 2e-2);
            }
            ok &= report("g2", compare(d2h(v.g2, n8).data(), orc.get(phase, "kda_g2" + L).data(), n8), 0.9999, 2e-2);
            ok &= report("scan_out", compare(d2h(v.o, n8).data(), orc.get(phase, "kda_scan_out" + L).data(), n8), 0.9999, 2e-2);
            ok &= report("kda_out", compare(d2h(dout, (size_t) T * kEmbd).data(), orc.get(phase, "kda_out" + L).data(),
                                            (size_t) T * kEmbd), 0.9999, 2e-2);
            const size_t nS = (size_t) kHeads * kDim * kDim;
            ok &= report("state S", compare(d2h(kda_state_S(s), nS).data(), orc.get(phase, "new_state" + L).data(), nS),
                         0.9999, 2e-2);
            // conv history: oracle has 3 CPY nodes per phase (q, k, v), each [3 time][8192] with time fastest
            const auto conv = d2h(kda_state_conv(s), (size_t) 3 * kHist * kInner);
            const std::string cname = "cache_r_l" + std::to_string(il) + " (view) (copy of  (view))";
            for (int mm = 0; mm < 3; ++mm) {
                const auto r = orc.get(phase, cname, mm);
                std::vector<float> mine((size_t) kHist * kInner);
                for (int c = 0; c < kInner; ++c)
                    for (int t = 0; t < kHist; ++t) mine[(size_t) c * kHist + t] = conv[((size_t) mm * kHist + t) * kInner + c];
                const char * nm[3] = {"conv hist q", "conv hist k", "conv hist v"};
                ok &= report(nm[mm], compare(mine.data(), r.data(), mine.size()), 0.9999, 2e-2);
            }
            all_ok &= ok;
        }
        kda_state_destroy(s);
    }
    std::printf("%s\n", all_ok ? "PARITY OK" : "PARITY FAIL");
    return all_ok ? 0 : 1;
}

// ---- synthetic inputs for recurrence tests -------------------------------------------------------------------

struct Synth {
    float * q, * k, * v, * g, * beta;
    std::vector<void *> owned;
    ~Synth() { for (void * p : owned) cudaFree(p); }
};

// q, k unit-norm per head, v ~ N(0,1), g in [-5, 0] skewed toward 0 like the real gate, beta in (0, 1).
std::unique_ptr<Synth> make_synth(int T, unsigned seed) {
    auto s = std::make_unique<Synth>();
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const size_t n = (size_t) T * kInner;
    std::vector<float> q(n), k(n), v(n), g(n), b((size_t) T * kHeads);
    for (size_t i = 0; i < n; ++i) { q[i] = nd(rng); k[i] = nd(rng); v[i] = nd(rng); }
    l2_heads(q);
    l2_heads(k);
    for (size_t i = 0; i < n; ++i) { const float u = ud(rng); g[i] = -5.0f * u * u * u; }
    for (auto & x : b) x = 0.05f + 0.9f * ud(rng);
    auto up = [&](const std::vector<float> & h) {
        float * d = nullptr;
        ck(cudaMalloc(&d, h.size() * 4), "synth");
        ck(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "synth h2d");
        s->owned.push_back(d);
        return d;
    };
    s->q = up(q); s->k = up(k); s->v = up(v); s->g = up(g); s->beta = up(b);
    return s;
}

int run_chunk(int argc, char ** argv) {
    const int T = std::atoi(arg(argc, argv, "--T", "2048").c_str());
    set_rq_variant(std::atoi(arg(argc, argv, "--rq", "0").c_str()));
    auto in = make_synth(T, 1234);
    const size_t nS = (size_t) kHeads * kDim * kDim, n8 = (size_t) T * kInner;
    float * S0, * Sa, * Sb, * oa, * ob;
    void * work = nullptr;
    ck(cudaMalloc(&S0, nS * 4), "S0"); ck(cudaMalloc(&Sa, nS * 4), "Sa"); ck(cudaMalloc(&Sb, nS * 4), "Sb");
    ck(cudaMalloc(&oa, n8 * 4), "oa"); ck(cudaMalloc(&ob, n8 * 4), "ob");
    if (scan_chunked_work_bytes(T)) ck(cudaMalloc(&work, scan_chunked_work_bytes(T)), "work");
    {   // a nonzero starting state, as after an earlier chunk
        auto pre = make_synth(64, 99);
        ck(cudaMemset(S0, 0, nS * 4), "S0 zero");
        scan_sequential(64, pre->q, pre->k, pre->v, pre->g, pre->beta, S0, S0, nullptr, 0);
    }
    cudaStream_t st = 0;
    scan_sequential(T, in->q, in->k, in->v, in->g, in->beta, S0, Sa, oa, st);
    scan_chunked(T, in->q, in->k, in->v, in->g, in->beta, S0, Sb, ob, work, st);
    ck(cudaDeviceSynchronize(), "scan");
    {   // the long-batch kernel against the decode-layout kernel run 8 tokens at a time (independent code path)
        float * Sc, * oc;
        ck(cudaMalloc(&Sc, nS * 4), "Sc");
        ck(cudaMalloc(&oc, n8 * 4), "oc");
        ck(cudaMemcpy(Sc, S0, nS * 4, cudaMemcpyDeviceToDevice), "Sc");
        for (int t0 = 0; t0 < T; t0 += kKdaLogT) {
            const int nt = std::min(kKdaLogT, T - t0);
            const size_t off = (size_t) t0 * kInner;
            scan_sequential(nt, in->q + off, in->k + off, in->v + off, in->g + off, in->beta + (size_t) t0 * kHeads,
                            Sc, Sc, oc + off, st);
        }
        ck(cudaDeviceSynchronize(), "steps");
        std::printf("long-batch sequential vs 8-token steps, T=%d\n", T);
        report("o", compare(d2h(oa, n8).data(), d2h(oc, n8).data(), n8), 0.99999, 1e-4);
        report("final S", compare(d2h(Sa, nS).data(), d2h(Sc, nS).data(), nS), 0.99999, 1e-4);
        cudaFree(Sc);
        cudaFree(oc);
    }
    std::printf("chunked vs sequential, T=%d\n", T);
    bool ok = report("o", compare(d2h(ob, n8).data(), d2h(oa, n8).data(), n8), 0.99999, 1e-3);
    ok &= report("final S", compare(d2h(Sb, nS).data(), d2h(Sa, nS).data(), nS), 0.99999, 1e-3);

    // timings (recurrence only)
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int mode = 0; mode < 2; ++mode) {
        const int reps = 5;
        cudaEventRecord(e0, st);
        for (int r = 0; r < reps; ++r) {
            if (mode == 0) scan_sequential(T, in->q, in->k, in->v, in->g, in->beta, S0, Sa, oa, st);
            else           scan_chunked(T, in->q, in->k, in->v, in->g, in->beta, S0, Sb, ob, work, st);
        }
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= reps;
        std::printf("  %-10s recurrence %.3f ms per layer for %d tokens (%.0f tok/s per layer)\n",
                    mode == 0 ? "sequential" : "chunked", ms, T, T / (ms * 1e-3));
    }
    std::printf("%s\n", ok ? "CHUNK OK" : "CHUNK FAIL");
    return ok ? 0 : 1;
}


// ---- speculative rollback and snapshot/restore ----------------------------------------------------------------

int run_rollback(int argc, char ** argv) {
    const Model m(arg(argc, argv, "--gguf", ""));
    const int il = std::atoi(arg(argc, argv, "--layer", "4").c_str());
    auto lw = load_layer(m, il);
    Geometry geo;
    const int TP = 30, TX = kKdaLogT, TY = 1;
    void * scratch = nullptr;
    ck(cudaMalloc(&scratch, kda_scratch_bytes(geo, TP)), "scratch");
    float * dP, * dX, * dY, * dout;
    ck(cudaMalloc(&dP, (size_t) TP * kEmbd * 4), "P");
    ck(cudaMalloc(&dX, (size_t) TX * kEmbd * 4), "X");
    ck(cudaMalloc(&dY, (size_t) TY * kEmbd * 4), "Y");
    ck(cudaMalloc(&dout, (size_t) TP * kEmbd * 4), "out");
    {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto fill = [&](float * d, int T) {
            std::vector<float> h((size_t) T * kEmbd);
            for (auto & e : h) e = nd(rng);
            ck(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "h2d");
        };
        fill(dP, TP); fill(dX, TX); fill(dY, TY);
    }
    cudaStream_t st = 0;
    auto fwd = [&](KdaState * s, const float * x, int T) {
        Batch b;
        b.T = T;
        kda_forward(geo, lw->w, s, b, x, dout, scratch, st);
    };
    const size_t nS = (size_t) kHeads * kDim * kDim, nC = (size_t) 3 * kHist * kInner;
    struct Snap { std::vector<float> out, S, C; };
    auto grab = [&](KdaState * s) {
        ck(cudaDeviceSynchronize(), "sync");
        return Snap{d2h(dout, (size_t) TY * kEmbd), d2h(kda_state_S(s), nS), d2h(kda_state_conv(s), nC)};
    };
    auto same = [&](const char * what, const Snap & a, const Snap & r) {
        const Cmp co = compare(a.out.data(), r.out.data(), a.out.size());
        const Cmp cs = compare(a.S.data(), r.S.data(), a.S.size());
        const Cmp cc = compare(a.C.data(), r.C.data(), a.C.size());
        const double worst = std::max({co.max_abs, cs.max_abs, cc.max_abs});
        const bool ok = co.cos > 0.9999999 && cs.cos > 0.9999999 && cc.cos > 0.9999999 && co.rel_max < 1e-5 &&
                        cs.rel_max < 1e-5 && cc.rel_max < 1e-5;
        std::printf("  %-34s next-token out max_abs %.2e, state S %.2e, conv %.2e  %s\n", what, co.max_abs, cs.max_abs,
                    cc.max_abs, ok ? (worst == 0 ? "ok (bitwise)" : "ok") : "FAIL");
        return ok;
    };
    bool ok = true;
    KdaState * s = kda_state_create(geo);
    KdaState * r = kda_state_create(geo);
    std::printf("rollback after a T=%d verify batch, layer %d (reference: the accepted prefix run as its own batch)\n", TX, il);
    for (int n = 0; n <= TX; ++n) {
        kda_state_reset(s, st);
        fwd(s, dP, TP);
        fwd(s, dX, TX);
        kda_state_rollback(geo, s, n, st);
        fwd(s, dY, TY);
        const Snap a = grab(s);
        kda_state_reset(r, st);
        fwd(r, dP, TP);
        if (n > 0) fwd(r, dX, n);
        fwd(r, dY, TY);
        const Snap ref = grab(r);
        char what[64];
        std::snprintf(what, sizeof what, "keep %d of %d", n, TX);
        ok &= same(what, a, ref);
    }
    std::printf("snapshot / restore\n");
    kda_state_reset(s, st);
    fwd(s, dP, TP);
    kda_state_snapshot(s, st);
    fwd(s, dX, TX);
    fwd(s, dY, TY);
    kda_state_restore(s, st);
    fwd(s, dY, TY);
    const Snap a = grab(s);
    kda_state_reset(r, st);
    fwd(r, dP, TP);
    fwd(r, dY, TY);
    ok &= same("restore after 9 tokens", a, grab(r));

    // CUDA graphs: one capture per parity, replays + kda_state_advance, then a partial rollback of the last replay
    std::printf("CUDA-graph replay (T=%d, one graph per parity)\n", TX);
    {
        cudaStream_t cs;
        ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
        kda_state_reset(s, st);
        fwd(s, dP, TP);
        ck(cudaDeviceSynchronize(), "sync");
        cudaGraphExec_t ge[2];
        for (int k = 0; k < 2; ++k) {
            const int p = kda_state_cur(s);
            cudaGraph_t gr;
            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture");
            Batch b;
            b.T = TX;
            kda_forward(geo, lw->w, s, b, dX, dout, scratch, cs);
            ck(cudaStreamEndCapture(cs, &gr), "end capture");
            ck(cudaGraphInstantiate(&ge[p], gr, 0), "instantiate");
            cudaGraphDestroy(gr);
        }
        const int keep = 3;
        for (int i = 0; i < 3; ++i) {
            ck(cudaGraphLaunch(ge[kda_state_cur(s)], cs), "launch");
            kda_state_advance(s, TX);
        }
        ck(cudaStreamSynchronize(cs), "sync");
        kda_state_rollback(geo, s, keep, st);
        fwd(s, dY, TY);
        const Snap g = grab(s);
        kda_state_reset(r, st);
        fwd(r, dP, TP);
        fwd(r, dX, TX);
        fwd(r, dX, TX);
        fwd(r, dX, keep);
        fwd(r, dY, TY);
        ok &= same("3 replays, keep 3 of the last", g, grab(r));
        cudaGraphExecDestroy(ge[0]);
        cudaGraphExecDestroy(ge[1]);
        cudaStreamDestroy(cs);
    }

    // cost of a partial rollback (replay of n_keep tokens)
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int n : {1, 4, 7}) {
        float tot = 0;
        const int reps = 50;
        for (int i = 0; i < reps; ++i) {
            fwd(s, dX, TX);
            cudaEventRecord(e0, st);
            kda_state_rollback(geo, s, n, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            tot += ms;
        }
        std::printf("  rollback keep %d of %d: %.4f ms per layer\n", n, TX, tot / reps);
    }
    kda_state_destroy(s);
    kda_state_destroy(r);
    std::printf("%s\n", ok ? "ROLLBACK OK" : "ROLLBACK FAIL");
    return ok ? 0 : 1;
}

// ---- timings --------------------------------------------------------------------------------------------------

int run_bench(int argc, char ** argv) {
    const Model m(arg(argc, argv, "--gguf", ""));
    const int il = std::atoi(arg(argc, argv, "--layer", "4").c_str());
    const int reps = std::atoi(arg(argc, argv, "--reps", "200").c_str());
    std::vector<int> Ts;
    {
        std::stringstream ss(arg(argc, argv, "--T", "1,4,8,512,2048"));
        std::string x;
        while (std::getline(ss, x, ',')) Ts.push_back(std::atoi(x.c_str()));
    }
    auto lw = load_layer(m, il);
    uint64_t wbytes = 0;
    for (const WeightRef * w : {&lw->w.q, &lw->w.k, &lw->w.v, &lw->w.out, &lw->w.f_a, &lw->w.f_b, &lw->w.g_a, &lw->w.g_b,
                                &lw->w.beta})
        wbytes += w->nbytes;
    Geometry geo;
    const int maxT = *std::max_element(Ts.begin(), Ts.end());
    void * scratch = nullptr;
    float * dx = nullptr, * dout = nullptr;
    ck(cudaMalloc(&scratch, kda_scratch_bytes(geo, maxT)), "scratch");
    ck(cudaMalloc(&dx, (size_t) maxT * kEmbd * 4), "x");
    ck(cudaMalloc(&dout, (size_t) maxT * kEmbd * 4), "out");
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> x((size_t) maxT * kEmbd);
        for (auto & e : x) e = nd(rng);
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x h2d");
    }
    KdaState * s = kda_state_create(geo);
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t ev[6], g0, g1;
    for (auto & e : ev) cudaEventCreate(&e);
    cudaEventCreate(&g0);
    cudaEventCreate(&g1);
    std::printf("layer %d, weights %.1f MB (HBM floor at 900 GB/s: %.3f ms)\n", il, wbytes / 1e6, wbytes / 900e9 * 1e3);
    const char * names[5] = {"in-proj", "prep", "recurrence", "post", "out-proj"};
    for (int T : Ts) {
        Batch b;
        b.T = T;
        kda_state_reset(s, st);
        const int r = T >= 512 ? std::max(3, reps / 40) : reps;
        set_fused_decode(arg(argc, argv, "--fused", "1") == "1");
        for (int i = 0; i < 3; ++i) kda_forward(geo, lw->w, s, b, dx, dout, scratch, st);
        ck(cudaStreamSynchronize(st), "warm");
        double acc[5] = {0, 0, 0, 0, 0};
        set_stage_events(ev);
        for (int i = 0; i < r; ++i) {
            kda_forward(geo, lw->w, s, b, dx, dout, scratch, st);
            ck(cudaStreamSynchronize(st), "fwd");
            for (int k = 0; k < 5; ++k) {
                float ms = 0;
                cudaEventElapsedTime(&ms, ev[k], ev[k + 1]);
                acc[k] += ms;
            }
        }
        set_stage_events(nullptr);
        double tot = 0;
        for (double a : acc) tot += a;
        std::printf("T=%-5d eager %.4f ms/layer:", T, tot / r);
        for (int k = 0; k < 5; ++k) std::printf("  %s %.4f", names[k], acc[k] / r);
        std::printf("  | non-proj %.4f ms\n", (acc[1] + acc[2] + acc[3]) / r);
        if (T <= 32) {   // the engine replays captured graphs: time that too
            float msg[2];
            for (int skip = 0; skip < 2; ++skip) {
                set_skip_projections(skip == 1);
                cudaGraph_t graph;
                cudaGraphExec_t exec;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                kda_forward(geo, lw->w, s, b, dx, dout, scratch, st);
                ck(cudaStreamEndCapture(st, &graph), "end capture");
                ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
                for (int i = 0; i < 10; ++i) cudaGraphLaunch(exec, st);
                cudaEventRecord(g0, st);
                for (int i = 0; i < r; ++i) cudaGraphLaunch(exec, st);
                cudaEventRecord(g1, st);
                ck(cudaEventSynchronize(g1), "graph");
                cudaEventElapsedTime(&msg[skip], g0, g1);
                msg[skip] /= r;
                cudaGraphExecDestroy(exec);
                cudaGraphDestroy(graph);
            }
            set_skip_projections(false);
            std::printf("        graph %.4f ms/layer, non-proj alone %.4f ms (34 layers: %.2f ms per step)\n",
                        msg[0], msg[1], msg[0] * 34);
        } else {
            std::printf("        prefill %.0f tok/s per layer\n", T / (tot / r * 1e-3));
        }
    }
    kda_state_destroy(s);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: kda_test parity|chunk|rollback|bench [options]\n");
        return 2;
    }
    const std::string mode = argv[1];
    g_repack = arg(argc, argv, "--raw", "0") != "1";
    try {
        if (mode == "parity") return run_parity(argc, argv);
        if (mode == "chunk") return run_chunk(argc, argv);
        if (mode == "bench") return run_bench(argc, argv);
        if (mode == "rollback") return run_rollback(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
