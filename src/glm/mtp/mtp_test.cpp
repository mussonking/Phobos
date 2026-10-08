// mtp_test: the MTP drafter (NextN block 45) against the llama.cpp PR #27917 graph, dumped by examples/mtp-oracle
// (fed with the trunk hidden states of the q4kxl-p32 trunk oracle), and its V100 cost.
//
//   mtp_test parity [--oracle DIR] [--gguf SHARD1] [--trunk-oracle DIR] [--chain 1] [--fed-attn 1] [--verbose 1]
//       phase 0   catch-up over the 30 prompt tokens (T = 30), head on the last row
//       phase 1-3 drafts from token 30 at position 30 (h of position 29): teacher-forced (each step gets the
//                 oracle's previous draft and h_nextn) or, with --chain, one 3-token greedy chain
//       phase 4   2-token catch-up at position 30 (implicit rollback of the drafts)
//     The routed experts of block 45 are fed from the oracle (mtp_ffn_moe_out-45); --fed-attn also feeds the
//     attention output (isolates the non-MLA part); everything else runs here.
//   mtp_test accept --dir DIR [--gguf SHARD1] [--k 4] [--share 0|1] [--vocab IDS] [--pr 1]
//       draft acceptance on a real generation dumped by examples/mtp-accept (tokens.txt, meta.txt, h_trunk.f32):
//       catch-up over the prompt, then at every generated position p one round from token p (h of p - 1) drafting
//       k tokens, compared with the trunk's greedy tokens p + 1 .. p + k (the rounds' draft rows are overwritten by
//       the next round, as in the PR loop). Routed experts of block 45 on the CPU (ref_experts, any GGUF type).
//       --vocab: draft head restricted to the ids listed in the file; --pr: agreement with the PR's drafts
//   mtp_test bench [--gguf SHARD1] [--ctx N] [--share 1]
//       draft cost per token without the routed experts (they are the experts component's), k = 1..4
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "forward.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/mtp.hpp"
#include "strata/glm/oracle.hpp"
#include "mtp_internal.hpp"
#include "ref_experts.hpp"

using namespace strata::glm;

namespace {

const char * kGguf = "/mnt/models-cold/glm53-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

std::string arg(int argc, char ** argv, const char * key, const std::string & def) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

std::vector<int> read_ints(const std::string & path, int col) {
    std::vector<int> v;
    FILE * f = std::fopen(path.c_str(), "r");
    if (!f) return v;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        char * p = line;
        for (int c = 0; c < col && p; ++c) {
            p = std::strchr(p, '\t');
            if (p) ++p;
        }
        if (p) v.push_back(std::atoi(p));
    }
    std::fclose(f);
    return v;
}

std::unique_ptr<Model> load(const std::string & gguf, cudaStream_t st) {
    LoadOptions opt;
    opt.load_mtp = true;
    opt.verbose = false;
    return load_model(gguf, opt, st);
}

// ---- parity -----------------------------------------------------------------------------------------------------

int run_parity(int argc, char ** argv) {
    const std::string odir = arg(argc, argv, "--oracle", "/mnt/models/glm-port/scratch/kda/mtp-oracle-noshare");
    const std::string tdir = arg(argc, argv, "--trunk-oracle", "/mnt/models/glm-oracle/q4kxl-p32");
    const bool verbose = arg(argc, argv, "--verbose", "0") != "0";
    const bool chain = arg(argc, argv, "--chain", "0") != "0";
    const bool fed_attn = arg(argc, argv, "--fed-attn", "0") != "0";
    const bool share = arg(argc, argv, "--share", "0") != "0";
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    auto m = load(arg(argc, argv, "--gguf", kGguf), st);
    const Geometry & g = m->g;
    const int D = g.n_embd, il = g.n_layer;
    test::Oracle oracle(odir);

    const std::vector<int> toks = read_ints(tdir + "/tokens.txt", 0);
    const std::vector<int> ref_drafts = read_ints(odir + "/drafts.txt", 2);
    const int N = 30;
    if ((int) toks.size() < N + 2 || ref_drafts.size() < 5) {
        std::fprintf(stderr, "need 32 tokens and 5 oracle drafts\n");
        return 1;
    }
    std::vector<float> h_trunk((size_t) (N + 2) * D);
    {
        FILE * f = std::fopen((odir + "/h_trunk.f32").c_str(), "rb");
        if (!f || std::fread(h_trunk.data(), 4, h_trunk.size(), f) != h_trunk.size()) {
            std::fprintf(stderr, "cannot read h_trunk.f32\n");
            return 1;
        }
        std::fclose(f);
    }

    auto ws = workspace_create(*m, N);
    MtpState * s = mtp_create(*m, 64, N);
    mtp_set_share_selection(s, share);
    float * dh;
    ck(cudaMalloc(&dh, (size_t) N * D * 4), "h");

    int phase = -1;
    struct Worst { double cos = 1, rel = 0; int phase = -1; };
    std::map<std::string, Worst> worst;
    std::vector<std::string> order;
    int route_bad = 0, route_total = 0;
    auto fetch = [&](const float * dev, int64_t n) {
        std::vector<float> h((size_t) n);
        ck(cudaStreamSynchronize(st), "tap sync");
        ck(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost), "tap copy");
        return h;
    };
    // last occurrence of a name in the phase (the norm nodes are named twice: rms_norm, then the weight mul)
    auto last = [&](int ph, const std::string & name) -> const test::OracleNode & {
        const test::OracleNode * r = nullptr;
        for (const auto & n : oracle.nodes())
            if (n.phase == ph && n.name == name) r = &n;
        if (!r) throw std::runtime_error("oracle: no node " + name + " in phase " + std::to_string(ph));
        return *r;
    };

    ForwardHooks hk;
    hk.attn = nullptr;
    hk.experts = [&](int, const float *, const int32_t *, const float *, float * out, int T, cudaStream_t) {
        const std::vector<float> v = oracle.load(last(phase, "mtp_ffn_moe_out-" + std::to_string(il)));
        if ((int64_t) v.size() != (int64_t) T * D) throw std::runtime_error("moe_out size");
        ck(cudaMemcpyAsync(out, v.data(), v.size() * 4, cudaMemcpyHostToDevice, st), "moe upload");
        ck(cudaStreamSynchronize(st), "moe upload");
    };
    hk.tap = [&](int til, const char * node, const float * dev, int64_t n) {
        std::string nd = node;
        if (nd == "mtp_eh_proj") ++phase;   // first node of every block forward
        if (nd == "ffn_moe_out") return;    // fed
        if (nd == "ffn_shexp") nd = "mtp_ffn_shexp";
        const std::string oname = til >= 0 ? nd + "-" + std::to_string(til) : nd;
        const std::vector<float> a = fetch(dev, n);
        const test::OracleNode & on = last(phase, oname);
        const std::vector<float> ref = oracle.load(on);
        if (nd == "ffn_moe_weights_scaled") {
            const int T = (int) (n / g.n_expert_used);
            std::vector<int32_t> ids((size_t) n);
            ck(cudaMemcpy(ids.data(), ws->ids, ids.size() * 4, cudaMemcpyDeviceToHost), "ids");
            const std::vector<float> rid = oracle.load(last(phase, "ffn_moe_topk-" + std::to_string(il)));
            for (int t = 0; t < T; ++t) {
                std::set<int> x, y;
                for (int k = 0; k < g.n_expert_used; ++k) {
                    x.insert(ids[(size_t) t * g.n_expert_used + k]);
                    y.insert((int) rid[(size_t) t * g.n_expert_used + k]);
                }
                route_total++;
                if (x != y) route_bad++;
            }
        }
        std::vector<float> a_sorted;
        if (nd == "ffn_moe_weights_scaled") {
            // compare per (row, expert id): the top-k order is not part of the math
            const int T = (int) (n / g.n_expert_used), K = g.n_expert_used;
            std::vector<int32_t> ids((size_t) n);
            ck(cudaMemcpy(ids.data(), ws->ids, ids.size() * 4, cudaMemcpyDeviceToHost), "ids");
            const std::vector<float> rid = oracle.load(last(phase, "ffn_moe_topk-" + std::to_string(il)));
            a_sorted.assign((size_t) n, 0.0f);
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < K; ++j)
                        if (ids[(size_t) t * K + j] == (int) rid[(size_t) t * K + k]) a_sorted[(size_t) t * K + k] = a[(size_t) t * K + j];
        }
        const std::vector<float> & av = a_sorted.empty() ? a : a_sorted;
        if (av.size() != ref.size()) {
            std::printf("  size mismatch %s phase %d: %zu vs %zu\n", oname.c_str(), phase, a.size(), ref.size());
            worst[oname].cos = 0;
            return;
        }
        const test::Stats sres = test::compare(av.data(), ref.data(), av.size());
        if (verbose) {
            char lbl[80];
            std::snprintf(lbl, sizeof lbl, "p%d %s", phase, oname.c_str());
            sres.print(lbl);
        }
        if (!worst.count(nd)) order.push_back(nd);
        Worst & w = worst[nd];
        if (sres.cosine < w.cos || std::isnan(sres.cosine)) { w.cos = sres.cosine; w.phase = phase; }
        w.rel = std::max(w.rel, sres.max_abs / std::max(sres.ref_rms, 1e-30));
    };

    auto upload_h = [&](const std::vector<int> & pos) {   // rows = trunk h of these positions (-1: zeros)
        std::vector<float> h(pos.size() * D, 0.0f);
        for (size_t r = 0; r < pos.size(); ++r)
            if (pos[r] >= 0) std::memcpy(&h[r * D], &h_trunk[(size_t) pos[r] * D], D * 4);
        ck(cudaMemcpyAsync(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice, st), "h upload");
        ck(cudaStreamSynchronize(st), "h upload");
    };

    if (fed_attn)
        mtp::set_attn_override([&](const float *, float * out, int T, cudaStream_t) {
            const std::vector<float> v = oracle.load(last(phase, "mtp_attn_out-" + std::to_string(il)));
            if ((int64_t) v.size() != (int64_t) T * D) throw std::runtime_error("attn_out size");
            ck(cudaMemcpyAsync(out, v.data(), v.size() * 4, cudaMemcpyHostToDevice, st), "attn upload");
            ck(cudaStreamSynchronize(st), "attn upload");
        });

    std::vector<int32_t> got;
    int32_t d[4];
    // phase 0: prompt catch-up, head on the last row
    {
        std::vector<int> pos(N);
        for (int p = 0; p < N; ++p) pos[p] = p - 1;
        upload_h(pos);
        mtp_draft(s, *m, *ws, hk, toks.data(), dh, N, 0, 1, d, st);
        got.push_back(d[0]);
    }
    // phases 1-3: drafts from token N at position N (h of position N - 1)
    upload_h({N - 1});
    if (!chain)   // teacher forcing: each step gets the oracle's previous draft and h_nextn
        mtp::set_chain_override([&](int i, int32_t & tok, float * h, cudaStream_t cs) {
            tok = ref_drafts[i];
            const std::vector<float> v = oracle.load(last(i, "h_nextn"));
            ck(cudaMemcpyAsync(h, v.data(), v.size() * 4, cudaMemcpyHostToDevice, cs), "h forced");
            ck(cudaStreamSynchronize(cs), "h forced");
        });
    mtp_draft(s, *m, *ws, hk, &toks[N], dh, 1, N, 3, d, st);
    got.insert(got.end(), d, d + 3);
    mtp::set_chain_override(nullptr);
    // phase 4: the accepted pair as one batch at position N
    upload_h({N - 1, N});
    mtp_draft(s, *m, *ws, hk, &toks[N], dh, 2, N, 1, d, st);
    got.push_back(d[0]);
    mtp::set_attn_override(nullptr);

    const double thr = chain ? 0.999 : fed_attn ? 0.99999 : 0.9998;
    std::printf("MTP block parity vs %s\n  routed experts fed from the oracle, attention %s%s, %s, pass: cos > %g\n",
                odir.c_str(), fed_attn ? "fed from the oracle" : "mla_forward (block 45 cache)",
                share ? ", shared selection" : "", chain ? "3-step greedy chain" : "teacher-forced steps", thr);
    bool ok = true;
    for (const auto & k : order) {
        const Worst & w = worst[k];
        const bool good = w.cos > thr;
        ok &= good;
        std::printf("  %-28s worst cos=%.9f (phase %d)  max_abs/rms=%.3e%s\n", k.c_str(), w.cos, w.phase, w.rel,
                    good ? "" : "  FAIL");
    }
    std::printf("  router top-8 sets: %d / %d rows differ\n", route_bad, route_total);
    std::printf("  greedy drafts:");
    for (size_t i = 0; i < got.size(); ++i) {
        std::printf(" %d(%d)", got[i], ref_drafts[i]);
        ok &= got[i] == ref_drafts[i];
    }
    std::printf("   [ours(oracle)]\n%s\n", ok ? "MTP PARITY OK" : "MTP PARITY FAIL");
    mtp_destroy(s);
    cudaFree(dh);
    return ok ? 0 : 1;
}

// ---- acceptance ---------------------------------------------------------------------------------------------------

int run_accept(int argc, char ** argv) {
    const std::string dir = arg(argc, argv, "--dir", "/mnt/models/glm-port/scratch/kda/accept/run");
    const int K = std::atoi(arg(argc, argv, "--k", "4").c_str());
    const bool share = arg(argc, argv, "--share", "1") != "0";
    const std::string vocab = arg(argc, argv, "--vocab", "");
    const bool cmp_pr = arg(argc, argv, "--pr", "1") != "0";
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    auto m = load(arg(argc, argv, "--gguf", kGguf), st);
    const Geometry & g = m->g;
    const int D = g.n_embd;

    int P = 0, n_tok = 0, n_embd = 0;
    {
        FILE * f = std::fopen((dir + "/meta.txt").c_str(), "r");
        if (!f || std::fscanf(f, "%d %d %d", &P, &n_tok, &n_embd) != 3) throw std::runtime_error("no meta.txt in " + dir);
        std::fclose(f);
    }
    if (n_embd != D) throw std::runtime_error("n_embd mismatch");
    const std::vector<int> tok = read_ints(dir + "/tokens.txt", 0);
    if ((int) tok.size() != n_tok) throw std::runtime_error("tokens.txt length mismatch");
    std::vector<float> h((size_t) n_tok * D);
    {
        FILE * f = std::fopen((dir + "/h_trunk.f32").c_str(), "rb");
        if (!f || std::fread(h.data(), 4, h.size(), f) != h.size()) throw std::runtime_error("bad h_trunk.f32");
        std::fclose(f);
    }
    // h_prev row p = h[p - 1], zeros for position 0
    float * dh;
    ck(cudaMalloc(&dh, ((size_t) n_tok + 1) * D * 4), "h");
    ck(cudaMemset(dh, 0, (size_t) D * 4), "h0");
    ck(cudaMemcpy(dh + D, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "h");

    const int chunk = 128;
    auto ws = workspace_create(*m, chunk);
    MtpState * s = mtp_create(*m, n_tok + K + 8, chunk);
    mtp_set_share_selection(s, share);
    std::vector<int32_t> sub;
    if (!vocab.empty()) {
        for (int v : read_ints(vocab, 0)) sub.push_back(v);
        std::sort(sub.begin(), sub.end());
        sub.erase(std::unique(sub.begin(), sub.end()), sub.end());
        mtp_set_draft_vocab(s, *m, sub.data(), (int) sub.size(), st);
    }
    ForwardHooks hk;
    double expert_ms = 0;
    hk.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                     cudaStream_t cs) {
        const auto t0 = std::chrono::steady_clock::now();
        const int KE = g.n_expert_used;
        std::vector<float> hx((size_t) T * D), hw((size_t) T * KE), ho(hx.size());
        std::vector<int32_t> hid((size_t) T * KE);
        ck(cudaStreamSynchronize(cs), "experts sync");
        ck(cudaMemcpy(hx.data(), y, hx.size() * 4, cudaMemcpyDeviceToHost), "x");
        ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ids");
        ck(cudaMemcpy(hw.data(), w, hw.size() * 4, cudaMemcpyDeviceToHost), "w");
        test::ref_experts_cpu(g, m->layers[il].experts, hx.data(), hid.data(), hw.data(), T, ho.data());
        ck(cudaMemcpyAsync(out, ho.data(), ho.size() * 4, cudaMemcpyHostToDevice, cs), "out");
        ck(cudaStreamSynchronize(cs), "out");
        expert_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // catch-up over the prompt
    std::vector<int32_t> ids(tok.begin(), tok.end());
    int32_t d[16];
    for (int p0 = 0; p0 < P; p0 += chunk) {
        const int T = std::min(chunk, P - p0);
        mtp_draft(s, *m, *ws, hk, ids.data() + p0, dh + (size_t) p0 * D, T, p0, 0, d, st);
    }
    // one round per generated position
    std::vector<int> m_at(n_tok, 0), hist(K + 1, 0);
    std::vector<std::vector<int>> drafts;
    int n_sub_miss = 0;
    for (int p = P; p <= n_tok - 2; ++p) {
        mtp_draft(s, *m, *ws, hk, ids.data() + p, dh + (size_t) p * D, 1, p, K, d, st);
        int a = 0;
        while (a < K && p + 1 + a < n_tok && d[a] == ids[p + 1 + a]) ++a;
        m_at[p] = a;
        hist[a]++;
        drafts.emplace_back(d, d + K);
        if (!sub.empty() && !std::binary_search(sub.begin(), sub.end(), ids[p + 1])) ++n_sub_miss;
    }
    const int n_rounds = n_tok - 1 - P;
    const std::string mode = std::string(share ? "share" : "noshare") +
                             (sub.empty() ? "" : " vocab " + std::to_string(sub.size()));
    std::printf("MTP acceptance on %s: prompt %d, %d positions, k = %d, %s\n", dir.c_str(), P, n_rounds, K,
                mode.c_str());
    if (!sub.empty())
        std::printf("  next tokens outside the draft vocabulary: %d / %d (%.2f%%)\n", n_sub_miss, n_rounds,
                    100.0 * n_sub_miss / n_rounds);
    std::printf("  matched prefix histogram (0..%d):", K);
    for (int a = 0; a <= K; ++a) std::printf(" %d", hist[a]);
    std::printf("\n");
    for (int k = 1; k <= K; ++k) {
        int ge = 0;
        for (int a = k; a <= K; ++a) ge += hist[a];
        long rounds = 0, acc = 0, emitted = 0;
        for (int p = P; p <= n_tok - 2;) {
            const int a = std::min(m_at[p], std::min(k, n_tok - 1 - p));
            ++rounds;
            acc += a;
            emitted += a + 1;
            p += a + 1;
        }
        std::printf("  k=%d  depth-%d rate %.4f | greedy loop: %ld rounds, draft acceptance %.4f, %.3f tokens/round\n",
                    k, k, (double) ge / n_rounds, rounds, (double) acc / ((double) rounds * k),
                    (double) emitted / rounds);
    }
    std::printf("  routed experts (CPU reference): %.1f s total\n", expert_ms / 1e3);

    // agreement with the PR's own drafts on the same positions (index share must match)
    const std::string pr = dir + "/drafts_" + (share ? "share" : "noshare") + ".tsv";
    if (cmp_pr && sub.empty()) {
        std::vector<std::vector<int>> prd;
        for (int c = 0; c <= K; ++c) {
            const std::vector<int> col = read_ints(pr, c);
            if (col.empty()) break;
            if (prd.empty()) prd.assign(col.size(), {});
            for (size_t i = 0; i < col.size() && i < prd.size(); ++i) prd[i].push_back(col[i]);
        }
        if (prd.size() == drafts.size()) {
            std::vector<int> same(K, 0), chain_same(K, 0);
            for (size_t i = 0; i < drafts.size(); ++i) {
                bool all = true;
                for (int j = 0; j < K && j + 1 < (int) prd[i].size(); ++j) {
                    const bool eq = drafts[i][j] == prd[i][j + 1];
                    same[j] += eq;
                    all = all && eq;
                    chain_same[j] += all;
                }
            }
            std::printf("  vs PR drafts (%s):", pr.c_str());
            for (int j = 0; j < K; ++j)
                std::printf(" d%d %.4f (prefix %.4f)", j + 1, (double) same[j] / drafts.size(),
                            (double) chain_same[j] / drafts.size());
            std::printf("\n");
        } else {
            std::printf("  vs PR drafts: %s missing or %zu rows (expected %zu)\n", pr.c_str(), prd.size(),
                        drafts.size());
        }
    }
    mtp_destroy(s);
    cudaFree(dh);
    return 0;
}

// ---- cost ---------------------------------------------------------------------------------------------------------

int run_bench(int argc, char ** argv) {
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    auto m = load(arg(argc, argv, "--gguf", kGguf), st);
    const Geometry & g = m->g;
    const int64_t ctx = std::atoll(arg(argc, argv, "--ctx", "4096").c_str());
    auto ws = workspace_create(*m, 8);
    MtpState * s = mtp_create(*m, ctx + 64, 8);
    const bool share = arg(argc, argv, "--share", "0") != "0";
    mtp_set_share_selection(s, share);
    ForwardHooks hk;
    hk.experts = [&](int, const float *, const int32_t *, const float *, float * out, int T, cudaStream_t cs) {
        ck(cudaMemsetAsync(out, 0, (size_t) T * g.n_embd * 4, cs), "experts stub");
    };
    // the attention / indexer cost depends on the context: random cache rows up to ctx
    mla_cache_fill_random(mtp_cache(s), m->layers[g.n_layer].mla_w, ctx, 7, st);
    std::vector<float> h((size_t) 8 * g.n_embd);
    for (size_t i = 0; i < h.size(); ++i) h[i] = std::sin(0.37f * (float) i);
    float * dh;
    ck(cudaMalloc(&dh, h.size() * 4), "h");
    ck(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "h");
    const int32_t ids[8] = {264, 1140, 315, 11, 13, 279, 264, 1140};
    int32_t d[8];
    std::printf("MTP draft cost on the V100, routed experts stubbed (zero), ctx %lld, index share %d\n",
                (long long) ctx, (int) share);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int reps = 20;
    for (int T : {1, 4}) {
        for (int k = 0; k <= 4; ++k) {
            mtp_draft(s, *m, *ws, hk, ids, dh, T, ctx, k, d, st);   // warm-up
            float tot = 0;
            for (int r = 0; r < reps; ++r) {
                cudaEventRecord(e0, st);
                mtp_draft(s, *m, *ws, hk, ids, dh, T, ctx, k, d, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                tot += ms;
            }
            const float per = tot / reps;
            std::printf("  catch-up T=%d + %d drafts: %.3f ms%s", T, k, per, k ? "" : "\n");
            if (k) std::printf("  (%.3f ms per draft incl. catch-up)\n", per / k);
        }
    }

    // breakdown of one T = 1, k = 4 call: an event after every tapped node (stream order), deltas summed by the
    // node that ends the interval; the interval ending at a step's mtp_eh_proj also holds the previous step's
    // argmax, its host sync and the embedding upload
    {
        std::vector<std::pair<std::string, cudaEvent_t>> evs;
        std::vector<cudaEvent_t> pool(256);
        for (auto & e : pool) cudaEventCreate(&e);
        ForwardHooks th = hk;
        th.tap = [&](int, const char * node, const float *, int64_t) {
            if (evs.size() >= pool.size()) return;
            cudaEvent_t e = pool[evs.size()];
            cudaEventRecord(e, st);
            evs.emplace_back(node, e);
        };
        std::map<std::string, double> by;
        std::vector<std::string> order;
        double total = 0;
        for (int r = 0; r < reps; ++r) {
            evs.clear();
            cudaEventRecord(e0, st);
            mtp_draft(s, *m, *ws, th, ids, dh, 1, ctx, 4, d, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            total += ms;
            cudaEvent_t prev = e0;
            for (auto & [name, e] : evs) {
                cudaEventElapsedTime(&ms, prev, e);
                if (!by.count(name)) order.push_back(name);
                by[name] += ms;
                prev = e;
            }
            cudaEventElapsedTime(&ms, prev, e1);
            by["(end)"] += ms;
        }
        order.push_back("(end)");
        std::printf("  breakdown, catch-up T=1 + 4 drafts (4 block forwards, 4 heads), %.3f ms per call with events:\n",
                    total / reps);
        for (const auto & n : order) std::printf("    %-24s %.3f ms\n", n.c_str(), by[n] / reps);
        for (auto & e : pool) cudaEventDestroy(e);
    }
    // the head alone, full and trimmed rows
    {
        const size_t sb = std::max(matmul_scratch_bytes(m->output, 1),
                                   matmul_scratch_bytes(m->layers[g.n_layer].eh_proj, 1));
        void * sc;
        ck(cudaMalloc(&sc, sb), "scratch");
        auto time_mm = [&](const WeightRef & w) {
            for (int r = 0; r < 3; ++r) glm_matmul(w, dh, ws->logits, 1, sc, sb, st);
            cudaEventRecord(e0, st);
            for (int r = 0; r < reps; ++r) glm_matmul(w, dh, ws->logits, 1, sc, sb, st);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            return ms / reps;
        };
        std::printf("  output head alone (type %d, %lld rows): %.3f ms\n", m->output.type, (long long) m->output.ne[1],
                    time_mm(m->output));
        std::printf("  eh_proj alone: %.3f ms\n", time_mm(m->layers[g.n_layer].eh_proj));
        cudaFree(sc);
        for (int n : {16384, 32768, 65536}) {
            std::vector<int32_t> sub(n);
            for (int i = 0; i < n; ++i) sub[i] = i;
            mtp_set_draft_vocab(s, *m, sub.data(), n, st);
            mtp_draft(s, *m, *ws, hk, ids, dh, 1, ctx, 4, d, st);
            float tot = 0;
            for (int r = 0; r < reps; ++r) {
                cudaEventRecord(e0, st);
                mtp_draft(s, *m, *ws, hk, ids, dh, 1, ctx, 4, d, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                tot += ms;
            }
            std::printf("  draft head trimmed to %d rows: catch-up T=1 + 4 drafts %.3f ms (%.3f ms per draft)\n", n,
                        tot / reps, tot / reps / 4);
        }
        mtp_set_draft_vocab(s, *m, nullptr, 0, st);
    }
    mtp_destroy(s);
    cudaFree(dh);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: mtp_test parity|bench [options]\n");
        return 2;
    }
    try {
        const std::string mode = argv[1];
        if (mode == "parity") return run_parity(argc, argv);
        if (mode == "bench") return run_bench(argc, argv);
        if (mode == "accept") return run_accept(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
