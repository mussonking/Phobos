// glm_eval: greedy generation on a real IQ4 GGUF for a set of tokenized prompts, one model + arena load for all.
// Per prompt, per miss kernel (--kernels vnni,f32[,ggml][,f64]; ggml = every expert through ggml-cpu's own vec_dot on
// the CPU, the llama.cpp reference, slow: HDD mmap; f64 = the misses in double, order-neutral reference, VRAM hits as
// every other kernel): states reset, the prompt teacher-forced through the T = 1 decode
// graph (routed experts: VRAM hits + host arena misses), then greedy until --tokens new tokens or a stop id.
// Writes OUT/<id>/<kernel>/{tokens.txt, meta.txt ("P n_tok 4096"), h_trunk.f32 ([n_tok][4096] post-output_norm row of
// every position: the head input, MTP h_prev), routing.bin (rows int32 pos, int32 layer, int32 ids[8], float w[8]
// in call order; pos < P = prompt), steps.tsv (pos, input token, output token, top1-top2 logit margin,
// ms)} and OUT/summary.tsv: per prompt, the kernels' tok/s and, against the first kernel, the first divergent
// generated token, identical tokens over the first --gate N, the min cos of the h rows before the divergence and
// the logit margins at it (a near tie diverging is benign).
//
//   glm_eval --gguf GGUF --out DIR [--tokens N] [--gate N] [--kernels vnni,f32] [--arena GB] [--profile STRP]
//            [--threads N] [--slots N] [--stop ID,ID] [--adapt N[,N..] [--adapt-swaps N] [--adapt-decay D]] PROMPT_DIR...
//            [--force-from DIR [--force-kernel K] [--save-logits] [--ref-logits DIR [--ref-kernel K]]] [--test-lend GB] [--down-q8-k] [--r8 [--r8-gguf-down]] [--dump-router-in]   (each PROMPT_DIR/prompt_tokens.txt, id = its basename)
// --force-from: reference-aligned comparison. Every kernel is teacher-forced through the same whole sequence
// DIR/<id>/<K>/tokens.txt (K default vnni; its meta.txt gives the prompt length), no generation, so all kernels
// see the same inputs at every position. Against the first kernel (e.g. ggml, the llama.cpp CPU reference):
// top-1 agreement and KL(first || kernel) of the full-vocabulary distributions, per position in
// OUT/<id>/<kernel>/aligned.tsv and per prompt in OUT/aligned.tsv (generated region and whole sequence).
// --save-logits writes each kernel's distributions (OUT/<id>/<kernel>/logits.f32, [N - 1][n_vocab], N = forced length);
// --ref-logits DIR [--ref-kernel K[,K..]] compares every kernel against DIR/<id>/K/logits.f32 instead of the first
// kernel (several K: one OUT/aligned.tsv line per ref, per-position OUT/<id>/<kernel>/aligned-<K>.tsv)
// (another process: --r8 is process-wide, so R8 vs GGUF arena needs two runs).
// --dump-router-in also writes router_in.bf16 next to routing.bin: each row's MoE router input y ([D] bf16, same
// row order), to replay other routers on it offline (expert cache lookahead study).
// OUT/policy.txt records the expert policy the run used (VNNI down activation format, adapt, profile, GGUF, VRAM
// slots). --slots N: exactly N expert slots (default: free VRAM minus the reserve, which varies by a few slots between
// processes; the experts at the profile's edge then flip between GPU hit and CPU miss, not bitwise equal kernels).
// --save-logits also writes slots.txt next to logits.f32; --ref-logits fails when a reference's slot count differs
// (ggml references excepted: every expert on the host, independent of the residency).
// --adapt: adaptive VRAM tier (routed.hpp) every N steps, 0 = static. Several values: one pass per value in the
// same process (one arena load), the cache reset to the seed ranking before each, outputs under OUT/adapt<N>/ and
// one line per pass in OUT/adapt.tsv (hit rate, swaps, upload GB per token). The tier is kept across prompts and
// kernels within a pass, like a served conversation.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "decode.hpp"
#include "forward.hpp"
#include "ref_experts.hpp"
#include "routed.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"

using namespace strata::glm;

namespace {
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> v;
    size_t a = 0;
    for (size_t b; (b = s.find(c, a)) != std::string::npos; a = b + 1) v.push_back(s.substr(a, b - a));
    v.push_back(s.substr(a));
    return v;
}
void mkdirs(const std::string & p) {
    for (size_t i = 1; i <= p.size(); ++i)
        if (i == p.size() || p[i] == '/') mkdir(p.substr(0, i).c_str(), 0775);
}
std::string prompt_id(std::string p) {   // the prompt directory's basename
    while (!p.empty() && p.back() == '/') p.pop_back();
    return p.substr(p.find_last_of('/') + 1);
}
struct Run {
    std::vector<int32_t> toks;      // prompt + generated
    std::vector<float> h;           // [n_tok][D]
    std::vector<float> margin;      // per position
    std::vector<float> logits;      // [position][n_vocab] (--force-from only)
    std::vector<double> gen_ms;     // generated steps
};
}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, out, profile, force_from, force_kernel = "vnni", ref_logits, ref_kernel = "vnni";
    bool save_logits = false;   // --force-from: OUT/<id>/<kernel>/logits.f32 ([decisions][n_vocab])
    int n_gen = 64, gate = 64, threads = 48, adapt_swaps = 96;
    float adapt_decay = 0.7f;
    int slots = 0;            // --slots N: fixed VRAM expert slots (RoutedOptions::slots), 0 = free VRAM
    bool down_q8_0 = true;    // --down-q8-k: VNNI down activations as Q8_K multi (default Q8_0 per token)
    bool r8 = false;          // --r8: R8 arena (routed.hpp RoutedOptions::r8)
    bool dump_rin = false;    // --dump-router-in: router_in.bf16, every MoE block's router input (lookahead study)
    bool r8_gguf_down = false;   // --r8-gguf-down: with --r8, GGUF down kernels on unpacked rows (diagnostic)
    double test_lend_gb = 0;   // --test-lend GB: lend + scribble + reclaim before each prompt's later kernels
    std::vector<int> adapts = {0};
    double arena_gb = 0;
    std::vector<std::string> kernels = {"vnni", "f32"}, prompts;
    std::vector<int32_t> stops = {154820, 154827};   // eos, eot (<|user|>) of the GLM-5.3 vocabulary
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--gate" && i + 1 < argc) gate = std::atoi(argv[++i]);
        else if (a == "--kernels" && i + 1 < argc) kernels = split(argv[++i], ',');
        else if (a == "--arena" && i + 1 < argc) arena_gb = std::atof(argv[++i]);
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--adapt" && i + 1 < argc) {
            adapts.clear();
            for (const auto & s : split(argv[++i], ',')) adapts.push_back(std::atoi(s.c_str()));
        }
        else if (a == "--down-q8-k") down_q8_0 = false;
        else if (a == "--r8") r8 = true;
        else if (a == "--r8-gguf-down") r8_gguf_down = true;
        else if (a == "--test-lend" && i + 1 < argc) test_lend_gb = std::atof(argv[++i]);
        else if (a == "--force-from" && i + 1 < argc) force_from = argv[++i];
        else if (a == "--force-kernel" && i + 1 < argc) force_kernel = argv[++i];
        else if (a == "--save-logits") save_logits = true;
        else if (a == "--ref-logits" && i + 1 < argc) ref_logits = argv[++i];
        else if (a == "--ref-kernel" && i + 1 < argc) ref_kernel = argv[++i];
        else if (a == "--adapt-swaps" && i + 1 < argc) adapt_swaps = std::atoi(argv[++i]);
        else if (a == "--adapt-decay" && i + 1 < argc) adapt_decay = (float) std::atof(argv[++i]);
        else if (a == "--slots" && i + 1 < argc) slots = std::atoi(argv[++i]);
        else if (a == "--dump-router-in") dump_rin = true;
        else if (a == "--stop" && i + 1 < argc) {
            stops.clear();
            for (const auto & s : split(argv[++i], ',')) stops.push_back(std::atoi(s.c_str()));
        } else if (a.rfind("--", 0) == 0) { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
        else prompts.push_back(a);
    }
    for (const auto & k : kernels)
        if (k != "vnni" && k != "f32" && k != "ggml" && k != "f64") {
            std::fprintf(stderr, "kernel %s: vnni, f32, ggml or f64\n", k.c_str());
            return 2;
        }
    if (gguf.empty() || out.empty() || prompts.empty() || kernels.empty()) {
        std::fprintf(stderr, "usage: glm_eval --gguf GGUF --out DIR [options] PROMPT_DIR...\n");
        return 2;
    }
    std::vector<std::vector<int32_t>> ptoks;
    std::vector<int> p_orig;   // --force-from: prompt length of the forced sequence
    int64_t max_ctx = 0;
    if (!force_from.empty()) n_gen = 0;
    for (const auto & d : prompts) {
        std::vector<int32_t> t;
        std::string id = d;
        while (!id.empty() && id.back() == '/') id.pop_back();
        id = id.substr(id.find_last_of('/') + 1);
        const std::string src = force_from.empty() ? d + "/prompt_tokens.txt"
                                                   : force_from + "/" + id + "/" + force_kernel + "/tokens.txt";
        if (!force_from.empty()) {
            FILE * mf = std::fopen((force_from + "/" + id + "/" + force_kernel + "/meta.txt").c_str(), "r");
            int P0 = 0;
            if (!mf || std::fscanf(mf, "%d", &P0) != 1) { std::fprintf(stderr, "%s: no meta.txt\n", src.c_str()); return 2; }
            std::fclose(mf);
            p_orig.push_back(P0);
        }
        FILE * f = std::fopen(src.c_str(), "r");
        int v;
        while (f && std::fscanf(f, "%d", &v) == 1) t.push_back(v);
        if (f) std::fclose(f);
        if (t.empty()) { std::fprintf(stderr, "%s: no tokens\n", src.c_str()); return 2; }
        max_ctx = std::max<int64_t>(max_ctx, (int64_t) t.size() + n_gen + 1);
        ptoks.push_back(std::move(t));
    }

    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    std::unique_ptr<Model> m;
    try {
        m = load_model(gguf, LoadOptions{}, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "LOAD FAILED: %s\n", e.what());
        return 1;
    }
    const Geometry & g = m->g;
    const int D = g.n_embd;
    auto ws = workspace_create(*m, 1);
    std::vector<KdaState *> kda(g.n_layer, nullptr);
    std::vector<MlaCache *> mla(g.n_layer, nullptr);
    for (int il = 0; il < g.n_layer; ++il) {
        if (m->layers[il].mla) mla[il] = mla_cache_create(g, max_ctx);
        else kda[il] = kda_state_create(g);
    }
    void * attn_scratch = nullptr;
    ck(cudaMalloc(&attn_scratch, std::max(kda_scratch_bytes(g, 1), mla_scratch_bytes(g, 1, max_ctx))), "scratch");
    int64_t * d_pos0 = nullptr;
    int64_t * h_pos0 = nullptr;
    ck(cudaMalloc(&d_pos0, sizeof(int64_t)), "d_pos0");
    ck(cudaMallocHost(&h_pos0, sizeof(int64_t)), "h_pos0");

    RoutedOptions ro;
    ro.cpu_threads = threads;
    ro.arena = true;
    ro.arena_max_bytes = (size_t) (arena_gb * 1e9);
    ro.vnni = true;
    ro.reserve_bytes = (size_t) 2 << 30;
    ro.pin_arena = true;
    ro.down_q8_0 = down_q8_0;
    ro.r8 = r8;
    ro.r8_gguf_down = r8_gguf_down;
    ro.adapt_every = *std::max_element(adapts.begin(), adapts.end());   // > 0: the arena keeps every expert
    ro.adapt_swaps = adapt_swaps;
    ro.adapt_decay = adapt_decay;
    ro.slots = slots;
    // routing trace of the current run: rows {pos, layer, ids[8], w[8]} (72 B)
    struct RouteRow { int32_t pos, layer, ids[8]; float w[8]; };
    std::vector<RouteRow> trace;
    // --dump-router-in: the router input y of each MoE block, copied in the graph (pinned [n_layer][D]) ahead of
    // routed_start, so it has landed when on_route runs; rows in routing.bin order, bf16 (round to nearest even)
    float * h_rin = nullptr;
    std::vector<uint16_t> rin;
    if (dump_rin) ck(cudaMallocHost(&h_rin, (size_t) g.n_layer * D * 4), "router input");
    int64_t cur_pos = 0;
    ro.on_route = [&](int il, const int32_t * ids, const float * w, int) {
        RouteRow row{(int32_t) cur_pos, il, {}, {}};
        for (int k = 0; k < 8; ++k) { row.ids[k] = ids[k]; row.w[k] = w[k]; }
        trace.push_back(row);
        if (h_rin)
            for (int i = 0; i < D; ++i) {
                uint32_t b;
                std::memcpy(&b, h_rin + (size_t) il * D + i, 4);
                rin.push_back((uint16_t) ((b + 0x7fffu + ((b >> 16) & 1u)) >> 16));
            }
    };
    RoutedPtr rt;
    std::vector<std::pair<int, int>> seed;
    try {
        rt = routed_create(*m, ro);
        seed = profile.empty() ? routed_rank_counts(*rt) : routed_read_profile(profile);
        routed_load(*rt, seed, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ROUTED FAILED: %s\n", e.what());
        return 1;
    }

    ForwardHooks gh;
    gh.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * o, const Batch & b, cudaStream_t s) {
        const LayerWeights & L = m->layers[il];
        if (L.mla) mla_forward_dev(g, L.mla_w, mla[il], b.T, d_pos0, y, o, attn_scratch, s, nullptr, y_in);
        else kda_forward(g, L.kda, kda[il], b, y, o, attn_scratch, s, y_in);
    };
    gh.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * o, int T_, cudaStream_t s) {
        if (h_rin) ck(cudaMemcpyAsync(h_rin + (size_t) il * D, y, (size_t) D * 4, cudaMemcpyDeviceToHost, s), "rin");
        routed_start(*rt, il, y, ids, w, o, T_, s);
    };
    gh.experts_finish = [&](int il, float * o, int T_, cudaStream_t s) { routed_finish(*rt, il, o, T_, s); };
    KdaState * k0 = nullptr;
    for (auto * k : kda) if (k && !k0) k0 = k;
    std::unique_ptr<SegmentedGraph> sg[2];
    for (int p = 0; p < 2; ++p) {
        const int c = kda_state_cur(k0);
        sg[c] = capture_decode(*m, *ws, 1, gh, st);
    }
    // ggml reference: graphs split at the experts call, experts on the host between segments
    ForwardHooks rh = gh;
    rh.experts_finish = nullptr;
    rh.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * o, int T_, cudaStream_t s) {
        const int K = g.n_expert_used;
        std::vector<float> hx((size_t) T_ * D), hw((size_t) T_ * K), ho(hx.size());
        std::vector<int32_t> hid((size_t) T_ * K);
        ck(cudaStreamSynchronize(s), "ref sync");
        ck(cudaMemcpy(hx.data(), y, hx.size() * 4, cudaMemcpyDeviceToHost), "ref x");
        ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ref ids");
        ck(cudaMemcpy(hw.data(), w, hw.size() * 4, cudaMemcpyDeviceToHost), "ref w");
        // the planes from the host arena (RAM) rather than the GGUF mapping; an R8 arena copy is not GGUF rows, so
        // those experts read the GGUF mapping itself (the source bytes, in the page cache after the first touch)
        // instead of a per-job unpack of the three planes (scalar repack, ~12 s per position)
        std::vector<HostExperts> pj((size_t) T_ * K);
        const HostExperts & hx0 = m->layers[il].experts;
        for (size_t j = 0; j < pj.size(); ++j) {
            const ExpertRef e = routed_host_expert(*rt, il, hid[j]);
            if (e.layout == 1) {
                pj[j].gate = hx0.gate + (size_t) hid[j] * hx0.gate_bytes;
                pj[j].up = hx0.up + (size_t) hid[j] * hx0.up_bytes;
                pj[j].down = hx0.down + (size_t) hid[j] * hx0.down_bytes;
                continue;
            }
            pj[j].gate = static_cast<const uint8_t *>(e.gate_up);
            pj[j].up = e.up ? static_cast<const uint8_t *>(e.up) : pj[j].gate + hx0.gate_bytes;
            pj[j].down = static_cast<const uint8_t *>(e.down);
        }
        test::ref_experts_cpu(g, m->layers[il].experts, hx.data(), hid.data(), hw.data(), T_, ho.data(), pj.data());
        ck(cudaMemcpy(o, ho.data(), ho.size() * 4, cudaMemcpyHostToDevice), "ref out");
    };
    std::unique_ptr<SegmentedGraph> sr[2];
    if (std::find(kernels.begin(), kernels.end(), "ggml") != kernels.end())
        for (int p = 0; p < 2; ++p) {
            const int c = kda_state_cur(k0);
            sr[c] = capture_decode(*m, *ws, 1, rh, st);
        }
    bool ref = false;

    std::vector<float> logits(g.n_vocab);
    auto run = [&](const std::vector<int32_t> & prompt) {
        Run r;
        for (auto * k : kda) if (k) kda_state_reset(k, st);
        for (auto * c : mla) if (c) mla_cache_set_size(c, 0);
        const int P = (int) prompt.size();
        r.toks = prompt;
        int32_t tok = prompt[0];
        for (int64_t pos = 0;; ++pos) {
            const double s0 = now_ms();
            embed_upload(*m, *ws, &tok, 1, st);
            *h_pos0 = pos;
            cur_pos = pos;
            ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
            if (ref) replay_decode(*sr[kda_state_cur(k0)], *ws, rh, st);
            else replay_decode(*sg[kda_state_cur(k0)], *ws, gh, st);
            for (auto * k : kda) if (k) kda_state_advance(k, 1);
            int32_t o = -1;
            ck(cudaMemcpyAsync(&o, ws->tok, 4, cudaMemcpyDeviceToHost, st), "tok");
            ck(cudaStreamSynchronize(st), "step");
            const double ms = now_ms() - s0;
            for (auto * c : mla) if (c) mla_cache_set_size(c, pos + 1);
            if (!ref) routed_step(*rt, st);
            r.h.resize((size_t) (pos + 1) * D);
            ck(cudaMemcpy(r.h.data() + (size_t) pos * D, ws->y, (size_t) D * 4, cudaMemcpyDeviceToHost), "h");
            ck(cudaMemcpy(logits.data(), ws->logits, logits.size() * 4, cudaMemcpyDeviceToHost), "logits");
            float t1 = -INFINITY, t2 = -INFINITY;
            for (float x : logits) {
                if (x > t1) { t2 = t1; t1 = x; }
                else if (x > t2) t2 = x;
            }
            r.margin.push_back(t1 - t2);
            if (!force_from.empty()) r.logits.insert(r.logits.end(), logits.begin(), logits.end());
            if (pos + 1 < P) { tok = prompt[pos + 1]; continue; }
            r.gen_ms.push_back(ms);
            r.toks.push_back(o);
            tok = o;
            const int n = (int) r.toks.size() - P;
            if (n >= n_gen || std::find(stops.begin(), stops.end(), o) != stops.end()) break;
        }
        return r;
    };

    mkdirs(out);
    FILE * ad = std::fopen((out + "/adapt.tsv").c_str(), "w");
    std::fprintf(ad, "adapt_every\tadapt_swaps\tsteps\tgenerated\thit_rate\tswaps\tupload_gb\tupload_gb_per_step\t"
                     "misses_per_step\n");
    const std::string out0 = out;
    // the policy every result in OUT was produced with
    FILE * pol = std::fopen((out + "/policy.txt").c_str(), "w");
    std::fprintf(pol, "vnni_down\t%s\narena\t%s\nadapt\t%s\nadapt_swaps\t%d\nprofile\t%s\ngguf\t%s\nslots\t%d\n",
                 down_q8_0 ? "q8_0_per_token" : "q8_k_multi", r8 ? (r8_gguf_down ? "r8_gate_up+gguf_down" : "r8") : "gguf", [&] {
                     std::string s;
                     for (int x : adapts) s += (s.empty() ? "" : ",") + std::to_string(x);
                     return s;
                 }().c_str(), adapt_swaps, profile.c_str(), gguf.c_str(), routed_slots(*rt));
    std::fclose(pol);
    // a reference from another process must have run with the same VRAM slots (same hits and misses), checked for
    // every prompt before the first run
    if (!ref_logits.empty())
        for (const auto & pr : prompts)
            for (const std::string & rk : split(ref_kernel, ',')) {
                if (rk == "ggml") continue;
                const std::string sp = ref_logits + "/" + prompt_id(pr) + "/" + rk + "/slots.txt";
                FILE * sf = std::fopen(sp.c_str(), "r");
                int rs = -1;
                if (sf) {
                    if (std::fscanf(sf, "%d", &rs) != 1) rs = -1;
                    std::fclose(sf);
                }
                if (rs != routed_slots(*rt)) {
                    std::fprintf(stderr, "FAIL: %s: reference slots %s, this run %d (rerun with --slots %d%s)\n",
                                 sp.c_str(), rs < 0 ? "unknown" : std::to_string(rs).c_str(), routed_slots(*rt),
                                 rs < 0 ? routed_slots(*rt) : rs,
                                 rs < 0 ? ", or write the reference's slot count from its log into slots.txt" : "");
                    return 2;
                }
            }
    FILE * al = std::fopen((out + "/aligned.tsv").c_str(), "w");
    std::fprintf(al, "prompt\tP\tpositions\tref\tkernel\tgen_top1\tgen_kl_mean\tgen_kl_p99\tgen_kl_max\tall_top1\t"
                     "all_kl_mean\n");
    for (size_t ci = 0; ci < adapts.size(); ++ci) {
    if (adapts.size() > 1) {
        out = out0 + "/adapt" + std::to_string(adapts[ci]);
        mkdirs(out);
        if (ci) routed_fill(*rt, seed, st);   // back to the seed placement
    }
    routed_set_adapt(*rt, adapts[ci], adapt_swaps);
    int64_t c_steps = 0, c_gen = 0, c_hits = 0, c_jobs = 0, c_swaps = 0;
    FILE * sum = std::fopen((out + "/summary.tsv").c_str(), "w");
    std::fprintf(sum, "prompt\tP\tkernel\tgenerated\ttok_s\tvs_kernel\tfirst_diff\tsame_over_gate\tmin_cos_h\t"
                      "margin_ref\tmargin_this\n");
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string id = prompt_id(prompts[pi]);
        const auto & prompt = ptoks[pi];
        const int P = (int) prompt.size();
        std::vector<Run> runs;
        for (const auto & k : kernels) {
            if (test_lend_gb > 0 && !runs.empty()) {
                // the loaned range is overwritten; routed_reclaim must restore the evicted experts exactly
                const size_t lb = (size_t) (test_lend_gb * 1e9);
                void * p = routed_lend(*rt, lb, st);
                ck(cudaMemsetAsync(p, 0x5a, lb, st), "scribble");   // on st: ordered before the reclaim uploads
                routed_reclaim(*rt, st);
                std::printf("test-lend: %.1f GB lent, scribbled and reclaimed\n", test_lend_gb);
            }
            ref = k == "ggml";
            if (!ref) {
                routed_set_vnni(*rt, k == "vnni");
                routed_set_f64(*rt, k == "f64");
            }
            routed_stats(*rt) = RoutedStats{};
            trace.clear();
            rin.clear();
            const double t0 = now_ms();
            runs.push_back(run(prompt));
            const Run & r = runs.back();
            // a kernel listed again (determinism check) gets <kernel>-<n>
            const int dup = (int) std::count(kernels.begin(), kernels.begin() + (runs.size() - 1), k);
            const std::string kn = dup ? k + "-" + std::to_string(dup + 1) : k;
            const std::string d = out + "/" + id + "/" + kn;
            mkdirs(d);
            const int n_tok = (int) r.toks.size();
            FILE * f = std::fopen((d + "/tokens.txt").c_str(), "w");
            for (int32_t t : r.toks) std::fprintf(f, "%d\n", t);
            std::fclose(f);
            f = std::fopen((d + "/meta.txt").c_str(), "w");
            std::fprintf(f, "%d %d %d\n", P, n_tok, D);
            std::fclose(f);
            // h rows of positions 0 .. n_tok - 2 (the last generated token was never fed back)
            f = std::fopen((d + "/h_trunk.f32").c_str(), "wb");
            std::fwrite(r.h.data(), 4, r.h.size(), f);
            std::fclose(f);
            if (!ref) {
                // routing.bin: positions < P are prompt tokens (teacher-forced, T = 1), the rest generated
                f = std::fopen((d + "/routing.bin").c_str(), "wb");
                std::fwrite(trace.data(), sizeof(RouteRow), trace.size(), f);
                std::fclose(f);
                if (h_rin) {
                    f = std::fopen((d + "/router_in.bf16").c_str(), "wb");
                    std::fwrite(rin.data(), 2, rin.size(), f);
                    std::fclose(f);
                }
            }
            f = std::fopen((d + "/steps.tsv").c_str(), "w");
            for (size_t p = 0; p < r.margin.size(); ++p)
                std::fprintf(f, "%zu\t%d\t%d\t%.4f\t%.2f\n", p, r.toks[p], r.toks[p + 1], r.margin[p],
                             p + 1 >= (size_t) P ? r.gen_ms[p + 1 - P] : 0.0);
            std::fclose(f);
            std::vector<double> v(r.gen_ms.begin() + std::min<size_t>(1, r.gen_ms.size()), r.gen_ms.end());
            std::sort(v.begin(), v.end());
            const double med = v.empty() ? 0 : v[v.size() / 2];
            const RoutedStats & rs = routed_stats(*rt);
            if (!ref) {
                c_steps += (int64_t) r.margin.size();
                c_gen += n_tok - P;
                c_hits += rs.hits;
                c_jobs += rs.jobs;
                c_swaps += rs.swaps;
            }
            std::printf("%s [%s] P=%d generated %d in %.1f s, median %.2f ms/token (%.1f tok/s), hits %.1f%%, "
                        "misses %.2f ms/step, spilled %lld, swaps %lld:", id.c_str(), kn.c_str(), P, n_tok - P,
                        (now_ms() - t0) / 1000.0, med, med > 0 ? 1000.0 / med : 0.0,
                        rs.jobs ? 100.0 * rs.hits / rs.jobs : 0.0, rs.miss_ms / std::max<size_t>(1, r.margin.size()),
                        (long long) rs.spilled, (long long) rs.swaps);
            for (int i = P; i < n_tok && i < P + 24; ++i) std::printf(" %d", r.toks[i]);
            std::printf("%s\n", n_tok - P > 24 ? " ..." : "");
            std::fflush(stdout);
            if (!force_from.empty()) {
                // positions 0 .. N - 2 predict the forced sequence's next token; the last one (N - 1) predicts past
                // its end, not a decision of the forced continuation: left out of logits.f32 and of the statistics
                const int V = g.n_vocab, P0 = p_orig[pi];
                const size_t rows = r.margin.size() - 1;
                if (save_logits) {
                    FILE * lf = std::fopen((d + "/logits.f32").c_str(), "wb");
                    std::fwrite(r.logits.data(), 4, rows * V, lf);
                    std::fclose(lf);
                    lf = std::fopen((d + "/slots.txt").c_str(), "w");
                    std::fprintf(lf, "%d\n", routed_slots(*rt));
                    std::fclose(lf);
                }
                if (ref_logits.empty() && runs.size() == 1) continue;
                // same inputs at every position: top-1 and KL(ref || this) of the next-token distributions, ref =
                // the first kernel, or --ref-logits DIR/<id>/<ref-kernel>/logits.f32 from another process (e.g. the
                // other arena layout) for every kernel
                const std::vector<std::string> refs = ref_logits.empty() ? std::vector<std::string>{""}
                                                                         : split(ref_kernel, ',');
                for (const std::string & rk : refs) {
                    Run ext;
                    if (!ref_logits.empty()) {
                        const std::string lp = ref_logits + "/" + id + "/" + rk + "/logits.f32";
                        FILE * lf = std::fopen(lp.c_str(), "rb");
                        if (!lf) { std::fprintf(stderr, "%s: cannot open\n", lp.c_str()); return 2; }
                        std::fseek(lf, 0, SEEK_END);
                        const size_t ext_rows = (size_t) std::ftell(lf) / 4 / V;
                        std::fseek(lf, 0, SEEK_SET);
                        ext.logits.resize(ext_rows * V);
                        if (std::fread(ext.logits.data(), 4, ext.logits.size(), lf) != ext.logits.size()) {
                            std::fprintf(stderr, "%s: short read\n", lp.c_str());
                            return 2;
                        }
                        std::fclose(lf);
                        ext.margin.resize(ext_rows + 1);   // rows = margin.size() - 1, like an in-process run
                    }
                    const Run & a = ref_logits.empty() ? runs[0] : ext;
                    const std::string an = ref_logits.empty() ? kernels[0] : "ref-" + rk;
                    const size_t n = std::min(a.margin.size() - 1, rows);
                    FILE * pf = std::fopen((d + (refs.size() > 1 ? "/aligned-" + rk : std::string("/aligned")) + ".tsv").c_str(), "w");
                    std::fprintf(pf, "pos\tnext\ttop1_%s\ttop1_%s\tkl\n", an.c_str(), kn.c_str());
                    double kl_sum[2] = {0, 0}, kl_max[2] = {0, 0};
                    int agree[2] = {0, 0}, cnt[2] = {0, 0};
                    std::vector<double> kls;
                    for (size_t q = 0; q < n; ++q) {
                        const float * la = a.logits.data() + q * V, * lb = r.logits.data() + q * V;
                        float ma = -INFINITY, mb = -INFINITY;
                        int ia = 0, ib = 0;
                        for (int v = 0; v < V; ++v) {
                            if (la[v] > ma) { ma = la[v]; ia = v; }
                            if (lb[v] > mb) { mb = lb[v]; ib = v; }
                        }
                        double za = 0, zb = 0;
                        for (int v = 0; v < V; ++v) { za += std::exp((double) la[v] - ma); zb += std::exp((double) lb[v] - mb); }
                        const double lza = std::log(za), lzb = std::log(zb);
                        double kl = 0;
                        for (int v = 0; v < V; ++v) {
                            const double lpa = la[v] - ma - lza, lpb = lb[v] - mb - lzb;
                            kl += std::exp(lpa) * (lpa - lpb);
                        }
                        kl = std::max(0.0, kl);
                        std::fprintf(pf, "%zu\t%d\t%d\t%d\t%.6g\n", q, r.toks[q + 1], ia, ib, kl);
                        // [0] generated region (outputs at positions P0 - 1 ..), [1] whole sequence
                        for (int w = 0; w < 2; ++w) {
                            if (w == 0 && (int) q < P0 - 1) continue;
                            ++cnt[w];
                            agree[w] += ia == ib;
                            kl_sum[w] += kl;
                            kl_max[w] = std::max(kl_max[w], kl);
                        }
                        if ((int) q >= P0 - 1) kls.push_back(kl);
                    }
                    std::fclose(pf);
                    std::sort(kls.begin(), kls.end());
                    const double p99 = kls.empty() ? 0 : kls[std::min(kls.size() - 1, (size_t) (kls.size() * 0.99))];
                    std::fprintf(al, "%s\t%d\t%zu\t%s\t%s\t%d/%d\t%.6g\t%.6g\t%.6g\t%d/%d\t%.6g\n", id.c_str(), P0, n,
                                 an.c_str(), kn.c_str(), agree[0], cnt[0], kl_sum[0] / std::max(1, cnt[0]), p99,
                                 kl_max[0], agree[1], cnt[1], kl_sum[1] / std::max(1, cnt[1]));
                    std::fflush(al);
                    std::printf("  aligned vs %s: generated region top-1 %d/%d, KL mean %.3g p99 %.3g max %.3g; whole "
                                "sequence top-1 %d/%d, KL mean %.3g\n", an.c_str(), agree[0], cnt[0],
                                kl_sum[0] / std::max(1, cnt[0]), p99, kl_max[0], agree[1], cnt[1],
                                kl_sum[1] / std::max(1, cnt[1]));
                    std::fflush(stdout);
                }
                continue;
            }
            if (runs.size() == 1) {
                std::fprintf(sum, "%s\t%d\t%s\t%d\t%.2f\t-\t-\t-\t-\t-\t-\n", id.c_str(), P, kn.c_str(), n_tok - P,
                             med > 0 ? 1000.0 / med : 0.0);
                continue;
            }
            // against the first kernel
            const Run & a = runs[0];
            int first = -1, same = 0;
            const int ng = std::min((int) a.toks.size(), n_tok) - P;
            for (int i = 0; i < ng; ++i) {
                if (a.toks[P + i] != r.toks[P + i]) { if (first < 0) first = i; }
                else if (first < 0 && i < gate) ++same;
            }
            // h rows agree up to the first divergent generated token's input position
            const size_t rows = std::min(a.margin.size(), r.margin.size());
            const size_t lim = first < 0 ? rows : std::min(rows, (size_t) (P + first));
            double min_cos = 1.0;
            for (size_t p = 0; p < lim; ++p) {
                double xy = 0, xx = 0, yy = 0;
                for (int c = 0; c < D; ++c) {
                    const double x = a.h[p * D + c], y = r.h[p * D + c];
                    xy += x * y; xx += x * x; yy += y * y;
                }
                min_cos = std::min(min_cos, xy / std::sqrt(xx * yy + 1e-30));
            }
            const size_t dp = first < 0 ? 0 : (size_t) (P - 1 + first);
            std::fprintf(sum, "%s\t%d\t%s\t%d\t%.2f\t%s\t%d\t%d/%d\t%.7f\t%.4f\t%.4f\n", id.c_str(), P, kn.c_str(),
                         n_tok - P, med > 0 ? 1000.0 / med : 0.0, kernels[0].c_str(), first,
                         first < 0 ? std::min(gate, ng) : same, std::min(gate, ng), min_cos,
                         first < 0 ? 0.0 : a.margin[dp], first < 0 ? 0.0 : r.margin[dp]);
            std::printf("  vs %s: first divergent generated token %d, identical %d of first %d, min cos h %.7f%s\n",
                        kernels[0].c_str(), first, first < 0 ? std::min(gate, ng) : same, std::min(gate, ng), min_cos,
                        first < 0 ? "" : (" (margins at divergence " + std::to_string(a.margin[dp]) + " / " +
                                          std::to_string(r.margin[dp]) + ")").c_str());
            std::fflush(stdout);
        }
        std::fflush(sum);
    }
    std::fclose(sum);
    const double up_gb = (double) c_swaps * routed_blob_bytes(g) / 1e9;
    std::fprintf(ad, "%d\t%d\t%lld\t%lld\t%.4f\t%lld\t%.2f\t%.4f\t%.2f\n", adapts[ci], adapt_swaps, (long long) c_steps,
                 (long long) c_gen, c_jobs ? (double) c_hits / c_jobs : 0.0, (long long) c_swaps, up_gb,
                 up_gb / std::max<int64_t>(1, c_steps), (double) (c_jobs - c_hits) / std::max<int64_t>(1, c_steps));
    std::fflush(ad);
    std::printf("== adapt %d: hit rate %.1f%% over %lld steps, %lld swaps = %.1f GB uploaded (%.3f GB/step), %.1f "
                "misses/step\n", adapts[ci], c_jobs ? 100.0 * c_hits / c_jobs : 0.0, (long long) c_steps,
                (long long) c_swaps, up_gb, up_gb / std::max<int64_t>(1, c_steps),
                (double) (c_jobs - c_hits) / std::max<int64_t>(1, c_steps));
    std::fflush(stdout);
    }
    std::fclose(ad);
    std::fclose(al);
    for (auto * k : kda) if (k) kda_state_destroy(k);
    for (auto * c : mla) if (c) mla_cache_destroy(c);
    return 0;
}
