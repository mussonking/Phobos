// glm_prefill_test: chunked prefill with the hybrid routed experts (prefill_moe) on the real IQ4 GGUF, timed, and
// checked against the T = 1 decode path (the engine's captured decode graphs, routed hits + VNNI misses).
//
//   glm_prefill_test --gguf GGUF --profile STRP --tokens FILE[,FILE..] [--ctx N0] [--new N] [--chunk C] [--ref 0|1]
//                    [--gen G] [--modes hybrid,gpu,cpu] [--ring 48] [--group 16] [--threads 48] [--reserve GB] [--slots N]
//                    [--ref-kernels vnni[,f32]] [--plan CTX:NEW[,CTX:NEW..]] [--mtp 0|1]
//                    [--async-reclaim 0|1] [--decode-after G] [--calibrate 0|1] [--cpu-cost E,J] [--upload-cost U] [--r8 0|1]
//
// Tokens: the files' ids concatenated (whitespace separated), repeated to reach N0 + N. The first N0 tokens are an
// existing context (prefilled too, timed apart), then N new tokens are timed. --ref 1 runs the same N0 + N tokens
// through the T = 1 graphs first: last-token logits (cos, max |diff|, top-1, top-5 overlap) and G greedy tokens
// after the prompt are compared (the prefilled side continues with the same T = 1 graphs). --ref-kernels vnni,f32
// runs the reference with both miss kernels (VNNI int8 and exact F32) to show the noise floor between them. --modes
// repeats the prefill per expert placement: hybrid (upload + CPU split), gpu (all uploaded), cpu (no upload),
// hybrid-prev (hybrid with the CPU share before 26ec8cc: R8 experts unpacked to the GGUF tiles, cost model 0.086 /
// 0.007 / upload 1.108; same process, same reference: the R8 tiles' quality A/B). The comparisons print KL(T=1 ||
// prefill) of the next-token distributions too.
// --plan times several cases in one load (no reference): NEW tokens on top of a CTX-token context (prefilled once per
// distinct CTX, then restored per case: KDA snapshot, MLA truncate).
// --mtp 1 also refreshes the MTP (block 45) cache: mtp_catch_up after every prefill chunk, against the T = 1
// reference's per-token mtp_draft refresh (k = 0); then both draft 4 tokens after the prompt (last h_nextn cos,
// drafts compared).
// --decode-after G: after each case, G timed T = 1 decode steps (routed_step between them, as a server would) from
// the prefilled state, to see the reclaim's effect on the first tokens (discarded: the next case restores / resets).
// --calibrate 0: the CPU cost model as given (no online calibration); each case's "fit:" line has the raw CPU share
// totals (experts, routed tokens, layers, measured ms) to refit PrefillMoeOptions::cpu_expert_ms / cpu_job_ms;
// --cpu-cost E,J overrides those two (ms per expert, ms per routed token), --upload-cost U the upload side's
// PrefillMoeOptions::upload_cost_ms.
// --r8 1 (default, as glm_serve): the R8 host arena (8-row interleaved; the CPU share's R8 paths); 0 = the GGUF arena.
// --modes decode1 / decode8: the NEW tokens go through the decode graphs instead (T = 1 steps, or T = 8 steps with
// the all-rows head and T = 1 for the tail; routed hits + CPU misses, MTP refreshed per step, routed_step between
// steps, as session_prompt below prefill_min), timed with the miss traffic. The context is prefilled as hybrid and
// kept from the previous mode when it has the same size (a decode mode after a prefill mode reuses it). decode1f64 =
// decode1 with the F64 miss kernels; decode1f64pN / decode1pN = F64 / int8 misses plus seeded draw N of a 1e-6 nudge
// (GLM_PERTURB), summed per arm on "draws" lines. With two decode modes or more, every case's last-token logits and G
// greedy tokens are compared with the first decode mode's (put decode1f64 first: the F64 anchor of the float-order
// rule).
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode.hpp"
#include "forward.hpp"
#include "model.hpp"
#include "prefill_moe.hpp"
#include "prefill_prompt.hpp"
#include "routed.hpp"
#include "state_file.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/mtp.hpp"

using namespace strata::glm;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

// decode1f64p[N] / decode1p[N] (F64 / int8 misses plus one perturbed miss sum, draw N, default 0): N; else -1
int perturb_site(const std::string & mode) {
    for (const char * p : {"decode1f64p", "decode1p"}) {
        const size_t n = std::strlen(p);
        if (mode.compare(0, n, p) != 0) continue;
        if (mode.size() == n) return 0;
        if (mode.find_first_not_of("0123456789", n) != std::string::npos) return -1;
        return std::atoi(mode.c_str() + n);
    }
    return -1;
}

// the arm of a draw: its mode without the trailing draw number (decode1p3 -> decode1p)
std::string draw_arm(const std::string & mode) {
    return perturb_site(mode) < 0 ? mode : mode.substr(0, mode.find_last_not_of("0123456789") + 1);
}

std::vector<int> top_k(const std::vector<float> & v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, profile, tokens, ref_kernels = "vnni", modes = "hybrid", plan;
    int64_t n_ctx = 0, n_new = 2048;
    int chunk = 4096, n_gen = 16, decode_after = 0, ring = 48, group = 16, threads = 48, profile_span = 0, spec = -1, slots = 0;
    bool chunk_set = false;   // --chunk given: no automatic sizing
    bool ref = false, use_mtp = false, async_reclaim = true, calibrate = true, r8 = true;
    std::vector<int> spec_alt;
    double reserve_gb = 7.0, spec_window = -1, cpu_expert_ms = -1, cpu_job_ms = -1, upload_cost_ms = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--gguf") gguf = next();
        else if (a == "--profile") profile = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--ctx") n_ctx = std::atoll(next().c_str());
        else if (a == "--new") n_new = std::atoll(next().c_str());
        else if (a == "--chunk") chunk = std::atoi(next().c_str()), chunk_set = true;
        else if (a == "--gen") n_gen = std::atoi(next().c_str());
        else if (a == "--ref") ref = next() != "0";
        else if (a == "--modes") modes = next();
        else if (a == "--plan") plan = next();
        else if (a == "--ring") ring = std::atoi(next().c_str());
        else if (a == "--group") group = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--reserve") reserve_gb = std::atof(next().c_str());
        else if (a == "--slots") slots = std::atoi(next().c_str());
        else if (a == "--ref-kernels") ref_kernels = next();
        else if (a == "--mtp") use_mtp = next() != "0";
        else if (a == "--async-reclaim") async_reclaim = next() != "0";
        else if (a == "--calibrate") calibrate = next() != "0";
        else if (a == "--r8") r8 = next() != "0";
        else if (a == "--upload-cost") upload_cost_ms = std::atof(next().c_str());
        else if (a == "--cpu-cost") {
            const auto v = split(next());
            if (v.size() != 2) throw std::runtime_error("--cpu-cost E,J");
            cpu_expert_ms = std::atof(v[0].c_str());
            cpu_job_ms = std::atof(v[1].c_str());
        }
        else if (a == "--decode-after") decode_after = std::atoi(next().c_str());
        else if (a == "--spec") {   // N, or A,B,..: the spans alternate (A/B on a noisy host)
            for (const auto & v : split(next())) spec_alt.push_back(std::atoi(v.c_str()));
            spec = *std::max_element(spec_alt.begin(), spec_alt.end());
        }
        else if (a == "--spec-window") spec_window = std::atof(next().c_str());
        else if (a == "--profile-span") profile_span = std::atoi(next().c_str());   // nsys --capture-range=cudaProfilerApi
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: glm_prefill_test --gguf GGUF --profile STRP --tokens FILE[,FILE..] [options]\n");
        return 2;
    }
    std::vector<int32_t> src;
    for (const auto & f : split(tokens)) {
        FILE * fp = std::fopen(f.c_str(), "r");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", f.c_str()); return 2; }
        int v;
        while (std::fscanf(fp, "%d", &v) == 1) src.push_back(v);
        std::fclose(fp);
    }
    if (src.empty()) { std::fprintf(stderr, "no tokens\n"); return 2; }
    struct Case { int64_t ctx, n; };
    std::vector<Case> cases;
    if (plan.empty()) cases.push_back({n_ctx, n_new});
    for (const auto & c : split(plan)) {
        const size_t k = c.find(':');
        if (k == std::string::npos) { std::fprintf(stderr, "--plan entries are CTX:NEW\n"); return 2; }
        cases.push_back({std::atoll(c.substr(0, k).c_str()), std::atoll(c.substr(k + 1).c_str())});
    }
    std::stable_sort(cases.begin(), cases.end(), [](const Case & a, const Case & b) { return a.ctx < b.ctx; });
    if (cases.size() > 1) ref = false;   // the reference covers one case
    int64_t n_all = 0;
    for (const Case & c : cases) n_all = std::max(n_all, c.ctx + c.n);
    std::vector<int32_t> toks(n_all);
    for (int64_t i = 0; i < n_all; ++i) toks[i] = src[i % src.size()];
    const int64_t max_ctx = n_all + std::max(n_gen, decode_after) + 8;
    std::printf("tokens: %zu source ids, context %lld + new %lld, chunk %d, ref %d, gen %d\n", src.size(),
                (long long) n_ctx, (long long) n_new, chunk, (int) ref, n_gen);

    try {
        cudaStream_t st;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        LoadOptions lo;
        lo.load_mtp = use_mtp;
        auto m = load_model(gguf, lo, st);
        const Geometry & g = m->g;
        matmul_init();

        // attention state shared by both paths (reset between)
        std::vector<KdaState *> kda(g.n_layer, nullptr);
        std::vector<MlaCache *> mla(g.n_layer, nullptr);
        for (int il = 0; il < g.n_layer; ++il) {
            if (m->layers[il].mla) mla[il] = mla_cache_create(g, max_ctx);
            else kda[il] = kda_state_create(g);
        }
        KdaState * k0 = nullptr;
        for (auto * k : kda) if (k && !k0) k0 = k;
        // MTP: one state per path (the reference's T = 1 refresh, the prefill's catch-up); h_carry = trunk h of the
        // last prefilled position (row 0 of the next chunk's hrows), h_ref / h_new the reference's
        const int D = g.n_embd;
        MtpState * mtp_ref = use_mtp ? mtp_create(*m, max_ctx, 8) : nullptr;
        MtpState * mtp_pre = use_mtp ? mtp_create(*m, max_ctx, 8) : nullptr;
        float * h_carry = nullptr, * h_ctx = nullptr, * h_ref = nullptr, * h_new = nullptr;
        for (float ** q : {&h_carry, &h_ctx, &h_ref, &h_new}) ck(cudaMalloc(q, D * 4), "h row");
        auto reset = [&] {
            for (auto * k : kda) if (k) kda_state_reset(k, st);
            for (auto * c : mla) if (c) mla_cache_set_size(c, 0);
            for (auto * t : {mtp_ref, mtp_pre}) if (t) mtp_truncate(t, 0);
            for (float * q : {h_carry, h_ref}) ck(cudaMemsetAsync(q, 0, D * 4, st), "h zero");
        };

        // the prefill buffers come from a routed_lend loan per prefill (prefill_moe_begin): only T = 1 here, plus
        // T = 8 for --modes decode8
        bool decode8 = false;
        for (const auto & md : split(modes)) decode8 |= md == "decode8";
        const int max_dec = decode8 ? kMaxHeadRows : 1;
        auto ws1 = workspace_create(*m, 1);
        std::unique_ptr<Workspace> ws8 = decode8 ? workspace_create(*m, max_dec) : nullptr;
        void * scratch1 = nullptr;
        ck(cudaMalloc(&scratch1, std::max(kda_scratch_bytes(g, max_dec), mla_scratch_bytes(g, max_dec, max_ctx))),
           "scratch");
        float * hcat = nullptr;   // decode modes: MTP h_prev rows [max_dec + 1][D]
        ck(cudaMalloc(&hcat, (size_t) (max_dec + 1) * D * 4), "hcat");
        int64_t * d_pos0 = nullptr, * h_pos0 = nullptr;
        ck(cudaMalloc(&d_pos0, sizeof(int64_t)), "d_pos0");
        ck(cudaMallocHost(&h_pos0, sizeof(int64_t)), "h_pos0");

        RoutedOptions ro;
        ro.cpu_threads = threads;
        ro.arena = true;
        ro.vnni = true;
        ro.pin_arena = true;
        ro.r8 = r8;
        ro.max_T = max_dec;
        ro.reserve_bytes = (size_t) (reserve_gb * 1e9);
        ro.slots = slots;   // --slots N: exactly N expert slots (as glm_eval / glm_serve), else free VRAM minus the reserve
        auto rt = routed_create(*m, ro);
        routed_load(*rt, profile.empty() ? routed_rank_counts(*rt) : routed_read_profile(profile), st);
        if (!routed_can_lend(*rt)) {   // as glm_serve, which then sends prompts through the decode graph
            std::printf("prefill SKIPPED: the host arena is neither pinned nor served by a bounce ring (routed_lend "
                        "needs one), nothing to test\n");
            return 0;
        }

        // GLM_SEL_TRACE=1: the T = 1 graph also copies each MLA layer's selected pools and pool scores out of the
        // shared scratch (decode1 modes compare them per position: overlap, rank of the flipped pools)
        const bool sel_trace = std::getenv("GLM_SEL_TRACE") && std::atoi(std::getenv("GLM_SEL_TRACE"));
        std::vector<int> mla_slot(g.n_layer, -1);
        int n_mla = 0;
        for (int il = 0; il < g.n_layer; ++il) if (mla[il]) mla_slot[il] = n_mla++;
        const int64_t n_pools = max_ctx / 4;
        int32_t * d_sel = nullptr;
        float * d_score = nullptr;
        if (sel_trace) {
            ck(cudaMalloc(&d_sel, (size_t) n_mla * 512 * 4), "sel trace");
            ck(cudaMalloc(&d_score, (size_t) n_mla * n_pools * 4), "score trace");
        }

        // T = 1 decode graphs (as glm_eval)
        ForwardHooks gh;
        gh.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * o, const Batch & b,
                         cudaStream_t s) {
            const LayerWeights & L = m->layers[il];
            if (L.mla) {
                mla_forward_dev(g, L.mla_w, mla[il], b.T, d_pos0, y, o, scratch1, s, nullptr, y_in);
                if (sel_trace && b.T == 1) {
                    const MlaTaps tp = mla_taps(g, mla[il], scratch1);
                    ck(cudaMemcpyAsync(d_sel + (size_t) mla_slot[il] * 512, tp.top, 512 * 4, cudaMemcpyDeviceToDevice,
                                       s), "sel copy");
                    ck(cudaMemcpyAsync(d_score + (size_t) mla_slot[il] * n_pools, tp.score,
                                       (size_t) std::min<int64_t>(n_pools, tp.n_pool_ld) * 4, cudaMemcpyDeviceToDevice,
                                       s), "score copy");
                }
            } else {
                kda_forward(g, L.kda, kda[il], b, y, o, scratch1, s, y_in);
            }
        };
        gh.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * o, int T,
                         cudaStream_t s) { routed_start(*rt, il, y, ids, w, o, T, s); };
        gh.experts_finish = [&](int il, float * o, int T, cudaStream_t s) { routed_finish(*rt, il, o, T, s); };
        std::unique_ptr<SegmentedGraph> sg[2];
        for (int p = 0; p < 2; ++p) {
            const int c = kda_state_cur(k0);
            sg[c] = capture_decode(*m, *ws1, 1, gh, st);
        }
        std::unique_ptr<SegmentedGraph> sg8[2];   // decode8: all-rows head (ws8->hn = the MTP h rows)
        if (decode8)
            for (int p = 0; p < 2; ++p) {
                const int c = kda_state_cur(k0);
                sg8[c] = capture_decode(*m, *ws8, max_dec, gh, st, true);
            }
        auto step1 = [&](int32_t tok, int64_t pos) -> int32_t {
            embed_upload(*m, *ws1, &tok, 1, st);
            *h_pos0 = pos;
            ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
            replay_decode(*sg[kda_state_cur(k0)], *ws1, gh, st);
            for (auto * k : kda) if (k) kda_state_advance(k, 1);
            for (auto * c : mla) if (c) mla_cache_set_size(c, pos + 1);
            int32_t o = -1;
            ck(cudaMemcpyAsync(&o, ws1->tok, 4, cudaMemcpyDeviceToHost, st), "tok");
            ck(cudaStreamSynchronize(st), "step");
            return o;
        };
        auto generate = [&](int32_t first, int64_t pos) {
            std::vector<int32_t> out;
            int32_t t = first;
            for (int i = 0; i < n_gen; ++i) {
                out.push_back(t);
                t = step1(t, pos + i);
            }
            return out;
        };
        std::vector<float> lpre(g.n_vocab);
        std::vector<int32_t> gpre;
        const std::vector<std::string> kernels = split(ref_kernels);
        std::vector<std::vector<float>> lref;
        std::vector<std::vector<int32_t>> gref;
        std::vector<float> mref_h(g.n_embd), mpre_h(g.n_embd);
        std::vector<int32_t> mref_d(4, -1), mpre_d(4, -1);
        if (ref)
            for (const auto & kn : kernels) {
                routed_set_vnni(*rt, kn == "vnni");
                reset();
                const double t0 = now_ms();
                int32_t next = -1;
                for (int64_t i = 0; i < n_all; ++i) {
                    next = step1(toks[i], i);
                    if (!use_mtp) continue;
                    // ws1->y = output_norm row of position i; MTP row i = (token i, h[i - 1])
                    ck(cudaMemcpyAsync(h_new, ws1->y, D * 4, cudaMemcpyDeviceToDevice, st), "h new");
                    mtp_draft(mtp_ref, *m, *ws1, gh, &toks[i], h_ref, 1, i, 0, nullptr, st);
                    ck(cudaMemcpyAsync(h_ref, h_new, D * 4, cudaMemcpyDeviceToDevice, st), "h ref");
                }
                const double ms = now_ms() - t0;
                lref.emplace_back(g.n_vocab);
                ck(cudaMemcpy(lref.back().data(), ws1->logits, g.n_vocab * 4, cudaMemcpyDeviceToHost), "ref logits");
                std::printf("reference T=1 (%s misses): %lld tokens in %.1f s (%.1f tok/s)\n", kn.c_str(),
                            (long long) n_all, ms / 1e3, n_all / ms * 1e3);
                if (use_mtp) {
                    std::vector<float> h(D);
                    std::vector<int32_t> d(4, -1);
                    ck(cudaMemcpy(h.data(), mtp_last_hidden(mtp_ref), D * 4, cudaMemcpyDeviceToHost), "h_nextn");
                    mtp_draft(mtp_ref, *m, *ws1, gh, &next, h_ref, 1, n_all, 4, d.data(), st);
                    mtp_truncate(mtp_ref, n_all);
                    if (kn == kernels[0]) {
                        mref_h = h;
                        mref_d = d;
                    } else {
                        double dot = 0, na = 0, nb = 0;
                        for (int i = 0; i < D; ++i) {
                            dot += (double) mref_h[i] * h[i];
                            na += (double) mref_h[i] * mref_h[i];
                            nb += (double) h[i] * h[i];
                        }
                        int same = 0;
                        while (same < 4 && mref_d[same] == d[same]) ++same;
                        std::printf("mtp T=1 %s vs T=1 %s: last h_nextn cos %.6f, drafts %d/4 identical\n",
                                    kernels[0].c_str(), kn.c_str(), dot / std::sqrt(na * nb), same);
                    }
                }
                gref.push_back(generate(next, n_all));
            }
        routed_set_vnni(*rt, true);

        auto cmp = [&](const std::vector<float> & a, const std::vector<int32_t> & ga, const std::vector<float> & b,
                       const std::vector<int32_t> & gb, const std::string & what) {
            double dot = 0, na = 0, nb = 0, mx = 0, ma = -1e30, mb = -1e30, za = 0, zb = 0, kl = 0;
            for (int i = 0; i < g.n_vocab; ++i) {
                dot += (double) a[i] * b[i];
                na += (double) a[i] * a[i];
                nb += (double) b[i] * b[i];
                mx = std::max(mx, (double) std::fabs(a[i] - b[i]));
                ma = std::max(ma, (double) a[i]);
                mb = std::max(mb, (double) b[i]);
            }
            // KL(a || b) of the next-token distributions (a = the reference), in nats
            for (int i = 0; i < g.n_vocab; ++i) {
                za += std::exp(a[i] - ma);
                zb += std::exp(b[i] - mb);
            }
            for (int i = 0; i < g.n_vocab; ++i) {
                const double pa = std::exp(a[i] - ma) / za;
                kl += pa * ((a[i] - ma - std::log(za)) - (b[i] - mb - std::log(zb)));
            }
            const auto ta = top_k(a, 5), tb = top_k(b, 5);
            int ov = 0;
            for (int x : ta) ov += std::count(tb.begin(), tb.end(), x) ? 1 : 0;
            int same = 0;
            while (same < n_gen && ga[same] == gb[same]) ++same;
            // the compared top-1's rank in the reference's top 5 (1 = same top-1, 0 = outside): a near-tie flip stays
            // in the top 5, a broken path does not
            const int rank = (int) (std::find(ta.begin(), ta.end(), tb[0]) - ta.begin()) + 1;
            std::printf("%s: logits cos %.6f, max |diff| %.4f, KL %.6g, top-1 %s, top-5 overlap %d/5, greedy %d/%d "
                        "identical, top-1 at reference rank %d\n", what.c_str(), dot / std::sqrt(na * nb), mx, kl,
                        ta[0] == tb[0] ? "same" : "DIFFERENT", ov, same, n_gen, rank > 5 ? 0 : rank);
        };
        auto show = [&](const std::string & name, const std::vector<int32_t> & v) {
            std::printf("  %s:", name.c_str());
            for (int32_t t : v) std::printf(" %d", t);
            std::printf("\n");
        };
        if (ref) {
            for (size_t i = 1; i < kernels.size(); ++i)
                cmp(lref[0], gref[0], lref[i], gref[i], "T=1 " + kernels[0] + " vs T=1 " + kernels[i]);
            for (size_t i = 0; i < kernels.size(); ++i) show("T=1 " + kernels[i], gref[i]);
        }

        // prefill, once per mode: hybrid (upload + CPU), gpu (every non-resident expert uploaded), cpu (none)
        PrefillMoe * pm = nullptr;
        // one prefill of tokens [a, b) (prefill_prompt: loan, chunks, MTP catch-up, reclaim)
        int32_t next = -1;
        bool shown = false;
        int spans = 0, shown_cap = -1;
        auto run_span = [&](int64_t a, int64_t b, const std::string & what) {
            const bool prof = ++spans == profile_span;
            if (spec_alt.size() > 1) {
                PrefillMoeOptions o = prefill_moe_options(pm);
                o.spec = spec_alt[(spans - 1) % spec_alt.size()];
                prefill_moe_set_options(pm, o);
                std::printf("(spec %d) ", o.spec);
            }
            prefill_moe_stats(pm) = PrefillMoeStats{};
            const int cap = (int) std::min<int64_t>(chunk, b - a);
            if (cap != shown_cap) {
                // the loan prefill_prompt takes
                shown_cap = cap;
                const PrefillLoanBytes lb = prefill_prompt_loan_bytes(pm, *m, b - a, max_ctx, use_mtp);
                std::printf("loan for cap %d (ring %d slots, up to %d speculative, group %d): ring %.2f + experts %.2f + workspace "
                            "%.2f + scratch %.2f = %.2f GB (%lld routed slots)\n",
                            lb.cap, lb.ring_slots, lb.spec_slots, lb.group, lb.ring / 1e9, lb.experts / 1e9, lb.workspace / 1e9, lb.scratch / 1e9,
                            lb.total / 1e9, (long long) ((lb.total + routed_blob_bytes(g) - 1) / routed_blob_bytes(g)));
            }
            PrefillPrompt pp;
            pp.kda = &kda;
            pp.mla = &mla;
            pp.max_ctx = max_ctx;
            pp.mtp = mtp_pre;
            pp.h_prev = h_carry;
            pp.logits = lpre.data();
            double c0 = now_ms();
            int64_t last = a;
            pp.on_chunk = [&](int done, int n) {
                if (!shown && last == a) std::printf("routed slots: %d during the loan\n", routed_slots(*rt));
                if (n > chunk)
                    std::printf("  chunk pos %lld T %lld: %.2f s (%.0f tok/s)\n", (long long) last,
                                (long long) (a + done - last), (now_ms() - c0) / 1e3,
                                (a + done - last) / (now_ms() - c0) * 1e3);
                last = a + done;
                c0 = now_ms();
            };
            const double t0 = now_ms();
            if (prof) ck(cudaProfilerStart(), "profiler start");
            next = prefill_prompt(pm, *m, pp, toks.data() + a, (int) (b - a), a, st);
            if (prof) ck(cudaProfilerStop(), "profiler stop");
            if (!shown) std::printf("routed slots: %d after the reclaim\n", routed_slots(*rt));
            shown = true;
            const double ms = now_ms() - t0;
            const PrefillMoeStats & s = prefill_moe_stats(pm);
            std::printf("%s: %lld tokens in %.2f s = %.0f tok/s\n", what.c_str(), (long long) (b - a), ms / 1e3,
                        (b - a) / ms * 1e3);
            const double J = (double) std::max<int64_t>(1, s.jobs), C = (double) std::max<int64_t>(1, s.calls);
            std::printf("  experts per MoE layer: resident %.1f, uploaded %.1f, cpu %.1f; jobs resident %.1f%% "
                        "uploaded %.1f%% cpu %.1f%%\n",
                        s.res_experts / C, s.up_experts / C, s.cpu_experts / C, 100 * s.res_jobs / J,
                        100 * s.up_jobs / J, 100 * s.cpu_jobs / J);
            if (s.spec_experts + s.spec_wasted)
                std::printf("  speculative uploads per MoE layer: %.1f used (jobs %.1f%%), %.1f unrouted\n",
                            s.spec_experts / C, 100 * s.spec_jobs / J, s.spec_wasted / C);
            std::printf("  trunk windows (experts join -> next routing) %.2f s\n", s.window_ms / 1e3);
            std::printf("  host: routing wait %.2f s, activation wait %.2f s, cpu %.2f s; model: pcie %.2f s cpu %.2f "
                        "s\n",
                        s.plan_wait_ms / 1e3, s.cpu_wait_ms / 1e3, s.cpu_ms / 1e3, s.model_up_ms / 1e3,
                        s.model_cpu_ms / 1e3);
            std::printf("  loan: lend %.3f s, reclaim %.3f s; mtp catch-up %.2f s (in the total)\n", s.lend_ms / 1e3,
                        s.reclaim_ms / 1e3, s.mtp_ms / 1e3);
            std::printf("  fit: layers %lld cpu_experts %lld cpu_jobs %lld cpu_ms %.1f model_cpu_ms %.1f\n",
                        (long long) s.calls, (long long) s.cpu_experts, (long long) s.cpu_jobs, s.cpu_ms, s.model_cpu_ms);
        };
        // decode modes: tokens [a, b) through the decode graphs, rows tokens per step (T = 1 for the tail)
        std::vector<float> dec_rows_h;   // the last run_decode's output_norm rows [b - a][D]
        // GLM_SEL_TRACE: per position and MLA layer the selected pools and their scores [b - a][n_mla][512] (-1 past
        // the visible pools)
        std::vector<int32_t> dec_sel_h, sel_h((size_t) n_mla * 512);
        std::vector<float> dec_sc_h, score_h(sel_trace ? n_pools : 0);
        // per-position next-token distributions (T = 1 decode modes): the first run of a CTX:NEW keeps each
        // position's top kTopKl log-probs; later runs of it get KL(first || run) per position, the rest of the
        // vocabulary lumped into one bin, and top-1 agreement
        constexpr int kTopKl = 64;
        struct PosRef {
            std::vector<int32_t> idx;   // [n][kTopKl], descending
            std::vector<float> lp;
        };
        std::map<std::pair<int64_t, int64_t>, PosRef> pos_ref;
        std::vector<float> dec_kl, logit_h(g.n_vocab);
        std::vector<uint8_t> dec_agree;
        std::vector<int32_t> vord(g.n_vocab);
        auto run_decode =[&](int64_t a, int64_t b, int rows, const std::string & what) {
            routed_stats(*rt) = RoutedStats{};
            const auto key = std::make_pair(a, b - a);
            const bool kl_ref = rows == 1 && !pos_ref.count(key);
            if (kl_ref) pos_ref[key] = PosRef{std::vector<int32_t>((size_t) (b - a) * kTopKl),
                                              std::vector<float>((size_t) (b - a) * kTopKl)};
            dec_kl.assign(rows == 1 && !kl_ref ? (size_t) (b - a) : 0, 0.0f);
            dec_agree.assign(dec_kl.size(), 0);
            dec_rows_h.assign((size_t) (b - a) * D, 0.0f);
            dec_sel_h.assign(sel_trace ? (size_t) (b - a) * n_mla * 512 : 0, -1);
            dec_sc_h.assign(dec_sel_h.size(), 0.0f);
            const double t0 = now_ms();
            ck(cudaMemcpyAsync(hcat, h_carry, D * 4, cudaMemcpyDeviceToDevice, st), "hcat row 0");
            int64_t steps = 0;
            Workspace * lw = ws1.get();
            int lT = 1;
            for (int64_t p = a; p < b; ++steps) {
                const int T = rows > 1 && b - p >= rows ? rows : 1;
                Workspace & w = T > 1 ? *ws8 : *ws1;
                lw = &w;
                lT = T;
                embed_upload(*m, w, toks.data() + p, T, st);
                *h_pos0 = p;
                ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
                replay_decode(T > 1 ? *sg8[kda_state_cur(k0)] : *sg[kda_state_cur(k0)], w, gh, st);
                for (auto * k : kda) if (k) kda_state_advance(k, T);
                for (auto * c : mla) if (c) mla_cache_set_size(c, p + T);
                ck(cudaMemcpyAsync(&next, w.tok + (T - 1), 4, cudaMemcpyDeviceToHost, st), "tok");
                if (mtp_pre) {
                    // MTP rows p + t = (token p + t, h[p + t - 1]) = hcat row t: the trunk's rows go to rows 1..T
                    // (T = 1 graph: ws1->y is the output_norm row, as the reference's)
                    ck(cudaMemcpyAsync(hcat + D, T > 1 ? w.hn : w.y, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, st),
                       "h rows");
                    mtp_draft(mtp_pre, *m, w, gh, toks.data() + p, hcat, T, p, 0, nullptr, st);
                    ck(cudaMemcpyAsync(hcat, hcat + (size_t) T * D, D * 4, cudaMemcpyDeviceToDevice, st), "h row 0");
                }
                ck(cudaStreamSynchronize(st), "decode step");
                // output_norm rows of positions p .. p + T - 1 (compared per position between decode modes)
                ck(cudaMemcpy(dec_rows_h.data() + (size_t) (p - a) * D, T > 1 ? w.hn : w.y, (size_t) T * D * 4,
                              cudaMemcpyDeviceToHost), "decode rows");
                if (rows == 1) {
                    ck(cudaMemcpy(logit_h.data(), w.logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost), "logits");
                    double mx = -1e300, z = 0;
                    for (float v : logit_h) mx = std::max(mx, (double) v);
                    for (float v : logit_h) z += std::exp((double) v - mx);
                    const double lse = mx + std::log(z);
                    PosRef & pr = pos_ref[key];
                    int32_t * ri = pr.idx.data() + (size_t) (p - a) * kTopKl;
                    float * rl = pr.lp.data() + (size_t) (p - a) * kTopKl;
                    if (kl_ref) {
                        std::iota(vord.begin(), vord.end(), 0);
                        std::partial_sort(vord.begin(), vord.begin() + kTopKl, vord.end(),
                                          [&](int32_t u, int32_t v) { return logit_h[u] > logit_h[v]; });
                        for (int k = 0; k < kTopKl; ++k) {
                            ri[k] = vord[k];
                            rl[k] = (float) (logit_h[vord[k]] - lse);
                        }
                    } else {
                        double kl = 0, mr = 0, mx2 = 0;
                        for (int k = 0; k < kTopKl; ++k) {
                            const double pr_k = std::exp((double) rl[k]), lx = logit_h[ri[k]] - lse;
                            kl += pr_k * (rl[k] - lx);
                            mr += pr_k;
                            mx2 += std::exp(lx);
                        }
                        const double tr = 1 - mr, tx = 1 - mx2;   // the lumped rest
                        if (tr > 1e-12 && tx > 1e-12) kl += tr * std::log(tr / tx);
                        dec_kl[p - a] = (float) std::max(0.0, kl);
                        dec_agree[p - a] = std::max_element(logit_h.begin(), logit_h.end()) - logit_h.begin() == ri[0];
                    }
                }
                if (sel_trace && T == 1) {
                    const int64_t nv = std::min<int64_t>((p + 1) / 4, n_pools);
                    const int ns = (int) std::min<int64_t>(nv, 512);
                    ck(cudaMemcpy(sel_h.data(), d_sel, sel_h.size() * 4, cudaMemcpyDeviceToHost), "sel trace");
                    for (int l = 0; l < n_mla; ++l) {
                        ck(cudaMemcpy(score_h.data(), d_score + (size_t) l * n_pools, (size_t) nv * 4,
                                      cudaMemcpyDeviceToHost), "score trace");
                        const size_t o = ((size_t) (p - a) * n_mla + l) * 512;
                        for (int k = 0; k < ns; ++k) {
                            const int32_t q = sel_h[(size_t) l * 512 + k];
                            dec_sel_h[o + k] = q;
                            dec_sc_h[o + k] = q >= 0 && q < nv ? score_h[q] : 0.0f;
                        }
                    }
                }
                routed_step(*rt, st);
                p += T;
            }
            ck(cudaMemcpyAsync(h_carry, hcat, D * 4, cudaMemcpyDeviceToDevice, st), "carry");
            // the last token's logits: row T - 1 of the last step's all-rows head
            ck(cudaMemcpyAsync(lpre.data(), lw->logits + (size_t) (lT - 1) * g.n_vocab, g.n_vocab * 4,
                               cudaMemcpyDeviceToHost, st), "decode logits");
            ck(cudaStreamSynchronize(st), "decode feed");
            const double ms = now_ms() - t0;
            const RoutedStats & rs = routed_stats(*rt);
            const double n = (double) (b - a), C = (double) std::max<int64_t>(1, rs.calls);
            std::printf("%s: %lld tokens in %.2f s = %.0f tok/s, next token %d\n", what.c_str(), (long long) (b - a),
                        ms / 1e3, n / ms * 1e3, next);
            std::printf("  decode: %lld steps, %lld routed calls, hit %.1f%%, distinct experts %.1f / misses %.1f per call, "
                        "miss reads %.2f GB/token (%.1f GB); misses %.2f s, routing wait %.2f s, swaps %lld\n",
                        (long long) steps, (long long) rs.calls, rs.jobs ? 100.0 * rs.hits / rs.jobs : 0.0,
                        rs.distinct / C, rs.distinct_misses / C, rs.distinct_misses * (double) routed_blob_bytes(g) / n / 1e9,
                        rs.distinct_misses * (double) routed_blob_bytes(g) / 1e9, rs.miss_ms / 1e3, rs.wait_ms / 1e3,
                        (long long) rs.swaps);
        };
        // GLM_STATE_HASH=1: every layer's attention state hashed at each case start (KDA S + conv; MLA latent rows,
        // pooled indexer keys and the key | gate tail at the context), a restored case against the first case of
        // its context (right after the context prefill)
        const bool state_hash = std::getenv("GLM_STATE_HASH") && std::atoi(std::getenv("GLM_STATE_HASH"));
        std::vector<uint64_t> hash_ref;
        int64_t hash_ctx = -1;
        std::vector<uint8_t> hbuf;
        auto layer_hashes = [&](int64_t ctx) {
            auto fnv = [](const uint8_t * p, size_t n) {
                uint64_t x = 1469598103934665603ull;
                for (size_t i = 0; i + 8 <= n; i += 8) {
                    uint64_t v;
                    std::memcpy(&v, p + i, 8);
                    x = (x ^ v) * 1099511628211ull;
                }
                for (size_t i = n / 8 * 8; i < n; ++i) x = (x ^ p[i]) * 1099511628211ull;
                return x;
            };
            std::vector<uint64_t> h(g.n_layer, 0);
            for (int il = 0; il < g.n_layer; ++il) {
                if (kda[il]) {
                    hbuf.resize(kda_state_bytes(g));
                    kda_state_save(kda[il], hbuf.data(), st);
                    ck(cudaStreamSynchronize(st), "hash kda");
                    h[il] = fnv(hbuf.data(), hbuf.size());
                } else if (mla[il]) {
                    const size_t rb = mla_cache_rows_bytes(mla[il], 0, ctx), tb = mla_cache_tail_bytes(mla[il]);
                    hbuf.assign(rb + tb, 0);
                    mla_cache_export_rows(mla[il], 0, ctx, hbuf.data(), st);
                    mla_cache_export_tail(mla[il], ctx, hbuf.data() + rb, st);
                    ck(cudaStreamSynchronize(st), "hash mla");
                    h[il] = fnv(hbuf.data(), hbuf.size());
                }
            }
            return h;
        };
        int64_t have_ctx = -1;   // the context in the caches (a decode mode keeps the previous mode's)
        struct DecOut {
            std::string mode;
            size_t ci;
            std::vector<float> logits;
            std::vector<int32_t> gen;
            std::vector<float> rows;   // output_norm row per new position
            std::vector<int32_t> sel;  // GLM_SEL_TRACE: selected pools [n][n_mla][512]
            std::vector<float> sc;     // and their scores
            std::vector<float> kl;     // per position KL(first run of the case || this run), empty for the first
            std::vector<uint8_t> agree;   // per position top-1 agreement with the first run
        };
        // GLM_VERIFY_SLOTS=1: every resident expert's VRAM blob checked against its host copy after each decode case
        const bool verify_slots = std::getenv("GLM_VERIFY_SLOTS") && std::atoi(std::getenv("GLM_VERIFY_SLOTS"));
        std::vector<DecOut> dec_out;   // per decode mode and case: last-token logits, greedy tokens after
        for (const auto & mode : split(modes)) {
            std::printf("--- prefill mode %s\n", mode.c_str());
            const int dec_rows = mode == "decode1" || mode.rfind("decode1f64", 0) == 0 || perturb_site(mode) >= 0 ? 1
                                 : mode == "decode8"                                                          ? max_dec
                                                                                                              : 0;
            PrefillMoeOptions po;
            po.max_T = chunk;
            po.group = group;
            po.ring = ring;
            po.cpu = mode != "gpu";
            po.upload = mode != "cpu";
            po.async_reclaim = async_reclaim;
            po.calibrate = calibrate;
            if (cpu_expert_ms >= 0) {
                po.cpu_expert_ms = cpu_expert_ms;
                po.cpu_job_ms = cpu_job_ms;
            }
            if (upload_cost_ms > 0) po.upload_cost_ms = upload_cost_ms;
            if (mode == "hybrid-prev") {   // the CPU share before 26ec8cc: R8 unpack + GGUF tiles, the old cost model
                po.r8_tiles = false;
                po.cpu_expert_ms = 0.086;
                po.cpu_job_ms = 0.007;
                po.upload_cost_ms = po.upload_ms;
            }
            if (spec >= 0) po.spec = spec;
            if (spec_window >= 0) po.spec_window = spec_window;
            pm = prefill_moe_create(*m, *rt, po, st);
            if (!chunk_set) {   // as glm_serve: the loan leaves a quarter of the slots (small VRAM)
                const int cap = prefill_prompt_fit_cap(pm, *m, max_ctx, use_mtp, routed_slots(*rt));
                if (cap == 0) {
                    std::fprintf(stderr, "the prefill's loan does not fit in 3/4 of %d expert slots even for 64-token chunks\n",
                                 routed_slots(*rt));
                    return 1;
                }
                if (cap != chunk) {
                    std::printf("prefill chunk %d -> %d (its loan must leave 1/4 of %d expert slots)\n", chunk, cap,
                                routed_slots(*rt));
                    prefill_moe_destroy(pm, st);
                    po.max_T = chunk = cap;
                    pm = prefill_moe_create(*m, *rt, po, st);
                }
            }
            shown = false;
            std::printf("prefill_moe: ring %d slots, %d routed slots between prefills\n", ring, routed_slots(*rt));
            // cases by context: one context prefill per distinct size, snapshot (KDA) / truncate (MLA) between
            if (!dec_rows) have_ctx = -1;
            for (const Case & c : cases) {
                if (c.ctx != have_ctx || c.ctx == 0) {
                    reset();
                    // GLM_STATE_LOAD=PREFIX: the context's attention state and pending token from PREFIX.CTX instead
                    // of a prefill (state_file.hpp; one shared state for runs that must start equal, e.g. two
                    // GLM_CPU_ISA processes). Not with --mtp 1 (the file holds no MTP cache).
                    const char * sl = std::getenv("GLM_STATE_LOAD");
                    if (c.ctx > 0 && sl) {
                        if (use_mtp) throw std::runtime_error("GLM_STATE_LOAD needs --mtp 0");
                        const std::string path = std::string(sl) + "." + std::to_string(c.ctx);
                        const auto pt = state_file_read(path, g, kda, mla, st);
                        if (pt.first != c.ctx) throw std::runtime_error(path + ": context mismatch");
                        next = pt.second;
                        std::printf("state load: %s (pending token %d)\n", path.c_str(), next);
                        for (auto * k : kda) if (k) kda_state_snapshot(k, st);
                    } else if (c.ctx > 0) {
                        run_span(0, c.ctx, "context " + std::to_string(c.ctx));
                        // GLM_STATE_DUMP=PREFIX: the context's attention state and pending token to PREFIX.CTX
                        // (state_file.hpp, read by GLM_STATE_LOAD and glm_two_gpu)
                        if (const char * sd = std::getenv("GLM_STATE_DUMP")) {
                            const std::string path = std::string(sd) + "." + std::to_string(c.ctx);
                            state_file_write(path, g, c.ctx, next, kda, mla, st);
                            std::printf("state dump: %s (pending token %d)\n", path.c_str(), next);
                        }
                        for (auto * k : kda) if (k) kda_state_snapshot(k, st);
                        ck(cudaMemcpyAsync(h_ctx, h_carry, D * 4, cudaMemcpyDeviceToDevice, st), "carry snapshot");
                    }
                    have_ctx = c.ctx;
                } else {
                    for (auto * k : kda) if (k) kda_state_restore(k, st);
                    for (auto * mc : mla) if (mc) mla_cache_truncate(mc, c.ctx);
                    if (mtp_pre) mtp_truncate(mtp_pre, c.ctx);
                    ck(cudaMemcpyAsync(h_carry, h_ctx, D * 4, cudaMemcpyDeviceToDevice, st), "carry restore");
                }
                ck(cudaStreamSynchronize(st), "case setup");
                if (state_hash) {
                    const std::vector<uint64_t> h = layer_hashes(c.ctx);
                    if (c.ctx != hash_ctx) {
                        hash_ref = h;
                        hash_ctx = c.ctx;
                        std::printf("state hash [%s] case %zu, context %lld: reference (after the context prefill)\n",
                                    mode.c_str(), (size_t) (&c - cases.data()), (long long) c.ctx);
                    } else {
                        int nd = 0, first = -1;
                        for (int il = 0; il < g.n_layer; ++il)
                            if (h[il] != hash_ref[il]) { ++nd; if (first < 0) first = il; }
                        std::printf("state hash [%s] case %zu, context %lld: %d layers differ from the reference%s",
                                    mode.c_str(), (size_t) (&c - cases.data()), (long long) c.ctx, nd,
                                    nd ? ", first " : "\n");
                        if (nd) std::printf("%d (%s)\n", first, mla[first] ? "mla" : "kda");
                    }
                }
                const std::string what = "[" + mode + "] new " + std::to_string(c.n) + " on context " + std::to_string(c.ctx);
                if (dec_rows) {
                    const bool f64 = mode.rfind("decode1f64", 0) == 0;
                    routed_set_vnni(*rt, !f64);
                    routed_set_f64(*rt, f64);
                    if (verify_slots) routed_verify_slots(*rt, st);
                    // decode1f64p[N]: F64 misses plus a (1 +- GLM_PERTURB) scale, default 1e-6, on one miss sum of
                    // new position 0 (draw N = the 1st / 8th / 24th / 40th / ... call with misses, signs + - + -): the
                    // model's own amplification of a rounding-size change, the noise floor of the decode comparisons.
                    // decode1p[N]: the same draw on the int8 miss path (GLM_CPU_ISA picks it): 5+ draws per path are
                    // the distributional gate of docs/glm-serve.md (D)
                    if (const int site = perturb_site(mode); site >= 0) {
                        const char * pe = std::getenv("GLM_PERTURB");
                        const float rel = pe ? (float) std::atof(pe) : 1e-6f;
                        routed_perturb_once(*rt, site % 2 ? -rel : rel, site == 0 ? 0 : site == 1 ? 7 : 23 + 16 * (site - 2));
                    }
                    run_decode(c.ctx, c.ctx + c.n, dec_rows, what);
                    if (verify_slots) routed_verify_slots(*rt, st);
                    // GLM_DUMP_LOGITS=prefix: the case's last-token logits to prefix.MODE.NEW (f32), across processes
                    // (prefix.MODE.NEW.cI for plan entry I > 0)
                    if (const char * dl = std::getenv("GLM_DUMP_LOGITS")) {
                        const size_t ci = (size_t) (&c - cases.data());
                        const std::string path = std::string(dl) + "." + mode + "." + std::to_string(c.n) +
                                                 (ci ? ".c" + std::to_string(ci) : std::string());
                        if (FILE * f = std::fopen(path.c_str(), "wb")) {
                            std::fwrite(lpre.data(), 4, lpre.size(), f);
                            std::fclose(f);
                        }
                    }
                    dec_out.push_back({mode, (size_t) (&c - cases.data()), lpre, generate(next, c.ctx + c.n), dec_rows_h,
                                       dec_sel_h, dec_sc_h, dec_kl, dec_agree});
                    routed_set_f64(*rt, false);
                    routed_set_vnni(*rt, true);
                } else {
                    run_span(c.ctx, c.ctx + c.n, what);
                }
                if (decode_after > 0) {
                    const double d0 = now_ms();
                    std::vector<double> ms;
                    int32_t t = next;
                    for (int i = 0; i < decode_after; ++i) {
                        const double s0 = now_ms();
                        t = step1(t, c.ctx + c.n + i);
                        routed_step(*rt, st);
                        ms.push_back(now_ms() - s0);
                    }
                    const double tot = now_ms() - d0;
                    std::printf("  decode after: %d tokens in %.3f s; first steps ms:", decode_after, tot / 1e3);
                    for (int i = 0; i < std::min(decode_after, 8); ++i) std::printf(" %.1f", ms[i]);
                    std::printf("; last %.1f\n", ms.back());
                    routed_quiesce(*rt, st);
                }
            }
            prefill_moe_destroy(pm, st);
            pm = nullptr;
            routed_quiesce(*rt, st);   // the reclaimed experts resident again, as in the reference
            if (use_mtp && ref && !dec_rows) {   // decode modes: no reference comparison
                ck(cudaMemcpy(mpre_h.data(), mtp_last_hidden(mtp_pre), D * 4, cudaMemcpyDeviceToHost), "h_nextn");
                mtp_draft(mtp_pre, *m, *ws1, gh, &next, h_carry, 1, n_all, 4, mpre_d.data(), st);
                mtp_truncate(mtp_pre, n_all);
                std::printf("mtp drafts after the prompt [%s]: %d %d %d %d\n", mode.c_str(), mpre_d[0], mpre_d[1],
                            mpre_d[2], mpre_d[3]);
                {
                    double dot = 0, na = 0, nb = 0;
                    for (int i = 0; i < D; ++i) {
                        dot += (double) mref_h[i] * mpre_h[i];
                        na += (double) mref_h[i] * mref_h[i];
                        nb += (double) mpre_h[i] * mpre_h[i];
                    }
                    int same = 0;
                    while (same < 4 && mref_d[same] == mpre_d[same]) ++same;
                    std::printf("mtp prefill %s vs T=1: last h_nextn cos %.6f, drafts %d/4 identical (ref %d %d %d %d)\n",
                                mode.c_str(), dot / std::sqrt(na * nb), same, mref_d[0], mref_d[1], mref_d[2],
                                mref_d[3]);
                }
            }
            if (ref && !dec_rows) {
                gpre = generate(next, n_all);
                for (size_t i = 0; i < kernels.size(); ++i)
                    cmp(lref[i], gref[i], lpre, gpre, "prefill " + mode + " vs T=1 " + kernels[i]);
                show("prefill " + mode, gpre);
            }
        }
        // per-position KL of a decode run against the first T = 1 run of its CTX:NEW: mean, p95, max, top-1 agreement
        struct KlStats {
            double mean = 0, p95 = 0, max = 0, agree = 0;
        };
        auto kl_stats = [](const DecOut & o) {
            KlStats k;
            std::vector<float> v = o.kl;
            std::sort(v.begin(), v.end());
            for (float x : v) k.mean += x / v.size();
            k.p95 = v[std::min(v.size() - 1, (size_t) (0.95 * v.size()))];
            k.max = v.back();
            for (uint8_t x : o.agree) k.agree += 100.0 * x / o.agree.size();
            return k;
        };
        // each decode case against the first decode case of the same CTX:NEW (another mode, or the same mode on a
        // repeated plan entry: the first run follows the context prefill, the later ones a snapshot restore)
        for (size_t oi = 0; oi < dec_out.size(); ++oi) {
            const DecOut & o = dec_out[oi];
            const Case & c = cases[o.ci];
            size_t ri = 0;
            while (cases[dec_out[ri].ci].ctx != c.ctx || cases[dec_out[ri].ci].n != c.n) ++ri;
            if (ri == oi) continue;
            const DecOut & r = dec_out[ri];
                {
                    cmp(r.logits, r.gen, o.logits, o.gen,
                        "[" + o.mode + "] case " + std::to_string(o.ci) + " vs [" + r.mode + "] case " +
                            std::to_string(r.ci) + ": new " + std::to_string(c.n) + " on context " + std::to_string(c.ctx));
                    show(r.mode, r.gen);
                    show(o.mode, o.gen);
                    // where the teacher-forced states part: output_norm row cos per new position
                    int first = -1, worst = 0;
                    double wcos = 2;
                    std::string head;   // the first positions' cos
                    for (int64_t i = 0; i < c.n; ++i) {
                        const float * x = r.rows.data() + (size_t) i * D, * y = o.rows.data() + (size_t) i * D;
                        double dot = 0, nx = 0, ny = 0;
                        for (int k = 0; k < D; ++k) {
                            dot += (double) x[k] * y[k];
                            nx += (double) x[k] * x[k];
                            ny += (double) y[k] * y[k];
                        }
                        const double cs = dot / std::sqrt(nx * ny + 1e-30);
                        if (cs < 0.999 && first < 0) first = (int) i;
                        if (cs < wcos) { wcos = cs; worst = (int) i; }
                        if (i < 4) {
                            char b[32];
                            std::snprintf(b, sizeof b, " %.6f", cs);
                            head += b;
                        }
                    }
                    std::printf("  output_norm rows: first cos < 0.999 at new position %d, min cos %.6f at %d; positions "
                                "0..3:%s\n",
                                first, wcos, worst, head.c_str());
                    // GLM_SEL_TRACE: the MLA layers' selected pools per position, overlap (Jaccard) and the rank, by
                    // its own run's score (0 = best, 511 = last), of each pool only one run selected
                    if (!r.sel.empty() && !o.sel.empty()) {
                        // the pools of X not in Y: count, ranks appended; returns |X|
                        auto only = [](const int32_t * X, const float * SX, const int32_t * Y, int & n_only,
                                       std::vector<int> & rk) {
                            std::vector<int> ord;
                            std::vector<int32_t> ys;
                            for (int k = 0; k < 512; ++k) {
                                if (X[k] >= 0) ord.push_back(k);
                                if (Y[k] >= 0) ys.push_back(Y[k]);
                            }
                            std::sort(ord.begin(), ord.end(), [&](int u, int v) { return SX[u] > SX[v]; });
                            std::sort(ys.begin(), ys.end());
                            for (size_t q = 0; q < ord.size(); ++q)
                                if (!std::binary_search(ys.begin(), ys.end(), X[ord[q]])) {
                                    ++n_only;
                                    rk.push_back((int) q);
                                }
                            return (int) ord.size();
                        };
                        double jall = 0;
                        int64_t flips = 0, first_flip = -1;
                        std::vector<int> ranks;
                        for (int64_t i = 0; i < c.n; ++i) {
                            double jp = 0, jmin = 1;
                            int fp = 0, lmin = 0;
                            for (int l = 0; l < n_mla; ++l) {
                                const size_t at = ((size_t) i * n_mla + l) * 512;
                                int oa = 0, ob = 0;
                                const int na = only(r.sel.data() + at, r.sc.data() + at, o.sel.data() + at, oa, ranks);
                                only(o.sel.data() + at, o.sc.data() + at, r.sel.data() + at, ob, ranks);
                                const double jl = na + ob ? (double) (na - oa) / (na + ob) : 1.0;
                                jp += jl / n_mla;
                                if (jl < jmin) { jmin = jl; lmin = l; }
                                fp += oa + ob;
                            }
                            jall += jp / (double) c.n;
                            flips += fp;
                            if (fp && first_flip < 0) first_flip = i;
                            if (i < 4 || i % 50 == 49)
                                std::printf("  selection at new position %lld: Jaccard mean %.4f, min %.4f (MLA layer %d), "
                                            "%d pools selected by one run only\n",
                                            (long long) i, jp, jmin, lmin, fp);
                        }
                        std::sort(ranks.begin(), ranks.end());
                        const int64_t deep =
                            std::lower_bound(ranks.begin(), ranks.end(), 448) - ranks.begin();
                        std::printf("  selection: first flip at new position %lld, Jaccard mean %.4f over %lld positions x "
                                    "%d MLA layers, %lld one-run pools, their rank median %d, %.1f%% above rank 448\n",
                                    (long long) first_flip, jall, (long long) c.n, n_mla, (long long) flips,
                                    ranks.empty() ? -1 : ranks[ranks.size() / 2],
                                    ranks.empty() ? 0.0 : 100.0 * deep / ranks.size());
                    }
                    if (!o.kl.empty()) {
                        const KlStats k = kl_stats(o);
                        std::printf("  per-position KL vs the first T = 1 run: mean %.4g, p95 %.4g, max %.4g; top-1 "
                                    "agreement %.1f%% of %zu positions\n",
                                    k.mean, k.p95, k.max, k.agree, o.kl.size());
                    }
                }
        }
        // the distributional gate (long context; the per-token teacher-forced comparison is not one, diag6): per
        // case, each decode run's per-position KL against the noise floor of the decode1f64p* runs (F64 misses plus
        // one rounding-size perturbation, averaged over the sites); PASS = mean and p95 both within 2x the floor's
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            KlStats f;
            int nf = 0;
            for (const DecOut & o : dec_out)
                if (o.ci == ci && !o.kl.empty() && o.mode.rfind("decode1f64p", 0) == 0) {
                    const KlStats k = kl_stats(o);
                    f.mean += k.mean;
                    f.p95 += k.p95;
                    f.agree += k.agree;
                    ++nf;
                }
            if (!nf) continue;
            f.mean /= nf;
            f.p95 /= nf;
            f.agree /= nf;
            std::printf("gate floor, new %lld on context %lld (%d eps runs): KL mean %.4g, p95 %.4g, top-1 %.1f%%\n",
                        (long long) cases[ci].n, (long long) cases[ci].ctx, nf, f.mean, f.p95, f.agree);
            for (const DecOut & o : dec_out)
                if (o.ci == ci && !o.kl.empty() && o.mode.rfind("decode1f64p", 0) != 0) {
                    const KlStats k = kl_stats(o);
                    const bool pass = k.mean <= 2 * f.mean && k.p95 <= 2 * f.p95;
                    std::printf("gate [%s] new %lld on context %lld: KL mean %.4g (%.2fx floor), p95 %.4g (%.2fx), "
                                "top-1 %.1f%%: %s\n",
                                o.mode.c_str(), (long long) cases[ci].n, (long long) cases[ci].ctx, k.mean,
                                k.mean / f.mean, k.p95, k.p95 / f.p95, k.agree, pass ? "PASS" : "FAIL");
                }
        }
        // per case and arm with 2+ seeded draws (decode1f64pN, decode1pN): the spread of the draws, the distribution
        // the gate compares between processes (one arm per miss path), never one run against another
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            std::vector<std::string> arms;
            for (const DecOut & o : dec_out)
                if (o.ci == ci && !o.kl.empty() && perturb_site(o.mode) >= 0 &&
                    std::find(arms.begin(), arms.end(), draw_arm(o.mode)) == arms.end())
                    arms.push_back(draw_arm(o.mode));
            for (const auto & arm : arms) {
                std::vector<double> mean, p95, agree;
                for (const DecOut & o : dec_out)
                    if (o.ci == ci && !o.kl.empty() && draw_arm(o.mode) == arm && perturb_site(o.mode) >= 0) {
                        const KlStats k = kl_stats(o);
                        mean.push_back(k.mean);
                        p95.push_back(k.p95);
                        agree.push_back(k.agree);
                    }
                if (mean.size() < 2) continue;
                auto mmm = [](std::vector<double> v) {
                    std::sort(v.begin(), v.end());
                    char b[96];
                    std::snprintf(b, sizeof b, "%.4g / %.4g / %.4g", v.front(), v[v.size() / 2], v.back());
                    return std::string(b);
                };
                std::printf("draws [%s] new %lld on context %lld, %zu draws (min / median / max): KL mean %s, p95 %s, "
                            "top-1 %s\n",
                            arm.c_str(), (long long) cases[ci].n, (long long) cases[ci].ctx, mean.size(),
                            mmm(mean).c_str(), mmm(p95).c_str(), mmm(agree).c_str());
            }
        }
        for (auto * t : {mtp_ref, mtp_pre}) mtp_destroy(t);
        for (float * q : {h_carry, h_ctx, h_ref, h_new}) cudaFree(q);
        for (auto * k : kda) if (k) kda_state_destroy(k);
        for (auto * c : mla) if (c) mla_cache_destroy(c);
        cudaFree(scratch1);
        cudaFree(hcat);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
