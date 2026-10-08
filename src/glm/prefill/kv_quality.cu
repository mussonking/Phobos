// glm_kv_quality: quality of a compressed MLA cache (long context), simulated on the F16 cache. Prefills N tokens of a
// file (exact prefill), then per mode: restores the attention state, round-trips the MLA caches' rows [0, N) through
// the mode's quantization (in place, F16 in and out), and decodes the next M tokens of the file teacher-forced through
// the T = 1 graphs. Each step's logits are compared with the first mode's (logits cos, top-1 agreement) and scored on
// the true next token (mean NLL, top-1 accuracy). The first mode is the baseline (f16); repeat it for the noise floor.
//
//   glm_kv_quality --gguf GGUF --profile STRP --tokens FILE --ctx N [--steps M] [--modes f16,f16,lat8g512,..]
//                  [--chunk 8192] [--threads 48] [--reserve GB] [--latent f16|q8] [--logits-out F] [--logits-ref F]
//   glm_kv_quality --gguf GGUF --profile STRP --needles LIST [--gen 24] [--modes ..]
//     needle-in-a-haystack prompts (make_needles.py): per prompt and mode, greedy tokens after the prompt on the
//     mode's cache, written next to the prompt as <mode>.gen.txt (check_needles.py looks for the code word).
// Modes: f16, or '+'-joined parts lat<bits>g<group> (latent rows [N][512]) / pool<bits>g<group> (pooled indexer keys
// [N/4][128]): symmetric int<bits> with one scale (absmax) per group of <group> values. f32x: the F16 cache with the
// exact F32 routed-miss kernel instead of VNNI int8 (the engine's own numeric noise, as a reference).
// --latent q8: the MLA caches are real Q8 ones (int8 latent written by the attention itself; lat modes are refused).
// --logits-out FILE / --logits-ref FILE (teacher-forced): the first mode's logits to a file / compared with a file
// written by another run (e.g. q8 against f16).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode.hpp"
#include "forward.hpp"
#include "model.hpp"
#include "prefill_moe.hpp"
#include "prefill_prompt.hpp"
#include "routed.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/mla.hpp"

using namespace strata::glm;

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(sep, i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

// x [rows][W] F16 in place: per group of G values, q = rint(x / s) in [-qmax, qmax] with s = absmax / qmax, x = q * s.
// One block per row, W threads (W <= 1024); thread 0 of each group finds its absmax.
__global__ void roundtrip_rows(__half * x, int W, int G, int qmax) {
    extern __shared__ float sh[];   // [W] values, then [W / G] scales
    float * v = sh, * sc = sh + W;
    __half * row = x + (size_t) blockIdx.x * W;
    const int j = threadIdx.x;
    v[j] = __half2float(row[j]);
    __syncthreads();
    if (j % G == 0) {
        float m = 0;
        for (int k = 0; k < G; ++k) m = fmaxf(m, fabsf(v[j + k]));
        sc[j / G] = m / qmax;
    }
    __syncthreads();
    const float s = sc[j / G];
    const float q = s > 0 ? fminf(fmaxf(rintf(v[j] / s), (float) -qmax), (float) qmax) : 0.0f;
    row[j] = __float2half(q * s);
}

struct Part { bool pooled; int bits, group; };
std::vector<Part> parse_mode(const std::string & mode) {
    std::vector<Part> out;
    if (mode == "f16" || mode == "f32x") return out;   // f32x: F16 cache, exact F32 miss kernel (noise reference)
    for (const auto & p : split(mode, '+')) {
        Part q{};
        size_t k;
        if (p.rfind("lat", 0) == 0) { q.pooled = false; k = 3; }
        else if (p.rfind("pool", 0) == 0) { q.pooled = true; k = 4; }
        else throw std::runtime_error("bad mode part " + p);
        const size_t gpos = p.find('g', k);
        if (gpos == std::string::npos) throw std::runtime_error("bad mode part " + p + " (need g<group>)");
        q.bits = std::atoi(p.substr(k, gpos - k).c_str());
        q.group = std::atoi(p.substr(gpos + 1).c_str());
        const int W = q.pooled ? 128 : 512;
        if (q.bits < 2 || q.bits > 8 || q.group < 1 || W % q.group) throw std::runtime_error("bad mode part " + p);
        out.push_back(q);
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, profile, tokens, needles, modes = "f16,f16,f32x,lat8g512,lat8g64,lat8g64+pool8g128,lat4g32";
    std::string logits_out, logits_ref;
    MlaLatent latent = MlaLatent::F16;
    int64_t n_ctx = 16384;
    int steps = 128, chunk = 8192, threads = 48, gen = 24;
    double reserve_gb = 7.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--gguf") gguf = next();
        else if (a == "--profile") profile = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--needles") needles = next();
        else if (a == "--ctx") n_ctx = std::atoll(next().c_str());
        else if (a == "--steps") steps = std::atoi(next().c_str());
        else if (a == "--gen") gen = std::atoi(next().c_str());
        else if (a == "--modes") modes = next();
        else if (a == "--chunk") chunk = std::atoi(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--reserve") reserve_gb = std::atof(next().c_str());
        else if (a == "--latent") latent = next() == "q8" ? MlaLatent::Q8 : MlaLatent::F16;
        else if (a == "--logits-out") logits_out = next();
        else if (a == "--logits-ref") logits_ref = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty() || (tokens.empty() == needles.empty())) {
        std::fprintf(stderr, "usage: glm_kv_quality --gguf GGUF --profile STRP (--tokens FILE --ctx N | --needles LIST)"
                             " [options]\n");
        return 2;
    }
    auto read_ids = [](const std::string & path) {
        std::vector<int32_t> v;
        std::ifstream f(path);
        for (long long x; f >> x;) v.push_back((int32_t) x);
        return v;
    };
    // needles: lines "id n_tokens dir" (make_needles.py); the prompt is dir/prompt_tokens.txt
    struct Needle { std::string id, dir; int64_t n; };
    std::vector<Needle> nl;
    std::vector<int32_t> toks;
    int64_t max_ctx = 0;
    if (!needles.empty()) {
        std::ifstream f(needles);
        for (Needle x; f >> x.id >> x.n >> x.dir;) nl.push_back(x);
        if (nl.empty()) { std::fprintf(stderr, "%s: no needles\n", needles.c_str()); return 2; }
        for (const auto & x : nl) max_ctx = std::max(max_ctx, x.n + gen + 8);
    } else {
        steps += steps & 1;
        toks = read_ids(tokens);
        if ((int64_t) toks.size() < n_ctx + steps + 1) {
            std::fprintf(stderr, "%s: %zu tokens, need ctx + steps + 1 = %lld\n", tokens.c_str(), toks.size(),
                         (long long) (n_ctx + steps + 1));
            return 2;
        }
        max_ctx = n_ctx + steps + 8;
    }
    const std::vector<std::string> mode_list = split(modes, ',');
    std::vector<std::vector<Part>> parts;
    for (const auto & md : mode_list) parts.push_back(parse_mode(md));
    for (const auto & pv : parts)
        for (const Part & q : pv)
            if (!q.pooled && latent == MlaLatent::Q8) {
                std::fprintf(stderr, "lat<b>g<G> modes round-trip F16 latent rows: not with --latent q8\n");
                return 2;
            }
    std::vector<std::string> mode_name = mode_list;   // labels and needle file names: q8-<mode> on Q8 caches
    if (latent == MlaLatent::Q8)
        for (auto & x : mode_name) x = "q8-" + x;

    try {
        cudaStream_t st;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        auto m = load_model(gguf, LoadOptions{}, st);
        const Geometry & g = m->g;
        matmul_init();
        std::vector<KdaState *> kda(g.n_layer, nullptr);
        std::vector<MlaCache *> mla(g.n_layer, nullptr);
        for (int il = 0; il < g.n_layer; ++il) {
            if (m->layers[il].mla) mla[il] = mla_cache_create(g, max_ctx, latent);
            else kda[il] = kda_state_create(g);
        }
        KdaState * k0 = nullptr;
        for (auto * k : kda) if (k && !k0) k0 = k;

        auto ws1 = workspace_create(*m, 1);
        void * scratch1 = nullptr;
        ck(cudaMalloc(&scratch1, kda_scratch_bytes(g, 1)), "scratch");
        // one MLA scratch per layer: mla_taps reads the layer's selection after the step
        std::vector<void *> mscr(g.n_layer, nullptr);
        for (int il = 0; il < g.n_layer; ++il)
            if (mla[il]) ck(cudaMalloc(&mscr[il], mla_scratch_bytes(g, 1, max_ctx)), "mla scratch");
        int64_t * d_pos0 = nullptr, * h_pos0 = nullptr;
        ck(cudaMalloc(&d_pos0, sizeof(int64_t)), "d_pos0");
        ck(cudaMallocHost(&h_pos0, sizeof(int64_t)), "h_pos0");

        RoutedOptions ro;
        ro.cpu_threads = threads;
        ro.arena = true;
        ro.vnni = true;
        ro.pin_arena = true;
        ro.reserve_bytes = (size_t) (reserve_gb * 1e9);
        auto rt = routed_create(*m, ro);
        routed_load(*rt, profile.empty() ? routed_rank_counts(*rt) : routed_read_profile(profile), st);

        ForwardHooks gh;
        gh.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * o, const Batch & b,
                         cudaStream_t s) {
            const LayerWeights & L = m->layers[il];
            if (L.mla) mla_forward_dev(g, L.mla_w, mla[il], b.T, d_pos0, y, o, mscr[il], s, nullptr, y_in);
            else kda_forward(g, L.kda, kda[il], b, y, o, scratch1, s, y_in);
        };
        gh.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * o, int T,
                         cudaStream_t s) { routed_start(*rt, il, y, ids, w, o, T, s); };
        gh.experts_finish = [&](int il, float * o, int T, cudaStream_t s) { routed_finish(*rt, il, o, T, s); };

        // exact prefill (frozen cost model, sync reclaim, no speculative uploads)
        PrefillMoeOptions po;
        po.max_T = chunk;
        po.async_reclaim = false;
        po.calibrate = false;
        po.spec = 0;
        PrefillMoe * pm = prefill_moe_create(*m, *rt, po, st);
        PrefillPrompt pp;
        pp.kda = &kda;
        pp.mla = &mla;
        pp.max_ctx = max_ctx;
        // T = 1 graphs, captured after the first prefill: mla_taps follows the last host-side call on a cache
        std::unique_ptr<SegmentedGraph> sg[2];
        auto prefill = [&](const int32_t * ids, int64_t n) {
            for (auto * k : kda) if (k) kda_state_reset(k, st);
            for (auto * c : mla) if (c) mla_cache_set_size(c, 0);
            const double t0 = now_ms();
            prefill_prompt(pm, *m, pp, ids, (int) n, 0, st);
            std::printf("prefill %lld tokens: %.1f s\n", (long long) n, (now_ms() - t0) / 1e3);
            if (!sg[0])
                for (int p = 0; p < 2; ++p) {
                    const int c = kda_state_cur(k0);
                    sg[c] = capture_decode(*m, *ws1, 1, gh, st);
                }
        };

        // saved state after a prefill of n tokens: KDA S + conv, MLA latent / pooled rows [0, n) (host)
        const size_t kb = kda_state_bytes(g);
        // latent row bytes (int8 or F16) and Q8 scale bytes per row
        const size_t lat_row = latent == MlaLatent::Q8 ? 512 : 1024, sc_row = latent == MlaLatent::Q8 ? 16 : 0;
        const size_t lat_cap = (size_t) max_ctx * lat_row, pool_cap = (size_t) (max_ctx / 4) * 128 * 2;
        std::vector<void *> kda_save(g.n_layer, nullptr), lat_save(g.n_layer, nullptr), pool_save(g.n_layer, nullptr);
        std::vector<void *> sc_save(g.n_layer, nullptr);
        for (int il = 0; il < g.n_layer; ++il) {
            if (kda[il]) ck(cudaMallocHost(&kda_save[il], kb), "kda save");
            else {
                ck(cudaMallocHost(&lat_save[il], lat_cap), "latent save");
                ck(cudaMallocHost(&pool_save[il], pool_cap), "pooled save");
                if (sc_row) ck(cudaMallocHost(&sc_save[il], (size_t) max_ctx * sc_row), "scale save");
            }
        }
        auto save = [&](int64_t n) {
            for (int il = 0; il < g.n_layer; ++il) {
                if (kda[il]) kda_state_save(kda[il], kda_save[il], st);
                else {
                    ck(cudaMemcpyAsync(lat_save[il], mla_cache_latent(mla[il]), (size_t) n * lat_row,
                                       cudaMemcpyDeviceToHost, st), "lat");
                    if (sc_row)
                        ck(cudaMemcpyAsync(sc_save[il], mla_cache_latent_scale(mla[il]), (size_t) n * sc_row,
                                           cudaMemcpyDeviceToHost, st), "scale");
                    ck(cudaMemcpyAsync(pool_save[il], mla_cache_pooled(mla[il]), (size_t) (n / 4) * 256,
                                       cudaMemcpyDeviceToHost, st), "pool");
                }
            }
            ck(cudaStreamSynchronize(st), "save");
        };
        // restore, then round-trip the context rows through the mode's quantization
        auto restore = [&](int64_t n, size_t mi) {
            for (int il = 0; il < g.n_layer; ++il) {
                if (kda[il]) {
                    kda_state_load(kda[il], kda_save[il], st);
                    continue;
                }
                mla_cache_set_size(mla[il], n);
                __half * lat = (__half *) const_cast<void *>(mla_cache_latent(mla[il]));
                __half * pool = (__half *) const_cast<void *>(mla_cache_pooled(mla[il]));
                ck(cudaMemcpyAsync(lat, lat_save[il], (size_t) n * lat_row, cudaMemcpyHostToDevice, st), "lat restore");
                if (sc_row)
                    ck(cudaMemcpyAsync(const_cast<void *>(mla_cache_latent_scale(mla[il])), sc_save[il],
                                       (size_t) n * sc_row, cudaMemcpyHostToDevice, st), "scale restore");
                ck(cudaMemcpyAsync(pool, pool_save[il], (size_t) (n / 4) * 256, cudaMemcpyHostToDevice, st),
                   "pool restore");
                for (const Part & q : parts[mi]) {
                    const int W = q.pooled ? 128 : 512;
                    const int64_t rows = q.pooled ? n / 4 : n;
                    if (rows <= 0) continue;
                    roundtrip_rows<<<(unsigned) rows, W, (W + W / q.group) * sizeof(float), st>>>(
                        q.pooled ? pool : lat, W, q.group, (1 << (q.bits - 1)) - 1);
                    ck(cudaGetLastError(), "roundtrip");
                }
            }
            routed_set_vnni(*rt, mode_list[mi] != "f32x");
        };
        auto step1 = [&](int32_t tok, int64_t pos) {
            embed_upload(*m, *ws1, &tok, 1, st);
            *h_pos0 = pos;
            ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
            replay_decode(*sg[kda_state_cur(k0)], *ws1, gh, st);
            for (auto * k : kda) if (k) kda_state_advance(k, 1);
            for (auto * c : mla) if (c) mla_cache_set_size(c, pos + 1);
        };

        if (!nl.empty()) {
            // needles: prefill all but the last prompt token, then per mode the last token and gen greedy tokens
            // through the T = 1 graphs on the mode's cache; ids to dir/<mode>.gen.txt (check_needles.py)
            for (const Needle & x : nl) {
                const std::vector<int32_t> ids = read_ids(x.dir + "/prompt_tokens.txt");
                if ((int64_t) ids.size() != x.n) throw std::runtime_error(x.id + ": token count mismatch");
                prefill(ids.data(), x.n - 1);
                save(x.n - 1);
                for (size_t mi = 0; mi < mode_list.size(); ++mi) {
                    restore(x.n - 1, mi);
                    std::vector<int32_t> out;
                    int32_t t = ids[x.n - 1];
                    const double t0 = now_ms();
                    for (int i = 0; i < gen; ++i) {
                        step1(t, x.n - 1 + i);
                        ck(cudaMemcpyAsync(&t, ws1->tok, 4, cudaMemcpyDeviceToHost, st), "tok");
                        ck(cudaStreamSynchronize(st), "step");
                        out.push_back(t);
                    }
                    std::ofstream f(x.dir + "/" + mode_name[mi] + ".gen.txt");
                    for (int32_t v : out) f << v << "\n";
                    std::printf("%s %s (%.1f ms per token at ctx %lld):", x.id.c_str(), mode_name[mi].c_str(),
                                (now_ms() - t0) / gen, (long long) x.n);
                    for (int32_t v : out) std::printf(" %d", v);
                    std::printf("\n");
                    std::fflush(stdout);
                }
            }
        } else {
            prefill(toks.data(), n_ctx);
            save(n_ctx);
            const int V = g.n_vocab;
            // DSA selection overlap between adjacent positions (baseline mode): per MLA layer, the selected pools of
            // step i against step i - 1 (|A & B| / |B|) and the union a T = 2 verify would gather (|A | B| / |B|)
            std::vector<std::vector<int32_t>> prev_top(g.n_layer);
            std::vector<double> ov_sum(g.n_layer, 0), un_sum(g.n_layer, 0);
            int ov_n = 0;
            std::vector<int32_t> top(512);
            std::vector<float> base((size_t) steps * V), lg(V);
            for (size_t mi = 0; mi < mode_list.size(); ++mi) {
                restore(n_ctx, mi);
                double cos_sum = 0, nll = 0;
                int agree = 0, acc = 0;
                const double t0 = now_ms();
                for (int i = 0; i < steps; ++i) {
                    const int64_t pos = n_ctx + i;
                    step1(toks[pos], pos);
                    ck(cudaMemcpyAsync(lg.data(), ws1->logits, (size_t) V * 4, cudaMemcpyDeviceToHost, st), "logits");
                    ck(cudaStreamSynchronize(st), "step");
                    const int32_t truth = toks[pos + 1];
                    double mx = -1e30;
                    int arg = 0;
                    for (int v = 0; v < V; ++v)
                        if (lg[v] > mx) { mx = lg[v]; arg = v; }
                    double se = 0;
                    for (int v = 0; v < V; ++v) se += std::exp((double) lg[v] - mx);
                    nll += -((double) lg[truth] - mx - std::log(se));
                    acc += arg == truth;
                    float * b = base.data() + (size_t) i * V;
                    if (mi == 0) {
                        std::copy(lg.begin(), lg.end(), b);
                        bool counted = false;
                        for (int il = 0; il < g.n_layer; ++il) {
                            if (!mla[il]) continue;
                            const MlaTaps tp = mla_taps(g, mla[il], mscr[il]);
                            const int nv = (int) std::min<int64_t>(512, (pos + 1) / 4);
                            ck(cudaMemcpy(top.data(), tp.top, nv * 4, cudaMemcpyDeviceToHost), "top");
                            std::vector<int32_t> cur(top.begin(), top.begin() + nv);
                            std::sort(cur.begin(), cur.end());
                            if (!prev_top[il].empty()) {
                                std::vector<int32_t> xs;
                                std::set_intersection(cur.begin(), cur.end(), prev_top[il].begin(),
                                                      prev_top[il].end(), std::back_inserter(xs));
                                ov_sum[il] += (double) xs.size() / prev_top[il].size();
                                un_sum[il] +=
                                    (double) (cur.size() + prev_top[il].size() - xs.size()) / prev_top[il].size();
                                counted = true;
                            }
                            prev_top[il] = std::move(cur);
                        }
                        ov_n += counted;
                        continue;
                    }
                    double dot = 0, na = 0, nb = 0, bm = -1e30;
                    int barg = 0;
                    for (int v = 0; v < V; ++v) {
                        dot += (double) lg[v] * b[v];
                        na += (double) lg[v] * lg[v];
                        nb += (double) b[v] * b[v];
                        if (b[v] > bm) { bm = b[v]; barg = v; }
                    }
                    cos_sum += dot / std::sqrt(na * nb);
                    agree += arg == barg;
                }
                if (mi == 0)
                    std::printf("%-22s NLL %.4f  top-1 acc %.3f  (%d steps at ctx %lld, %.1f s)\n",
                                mode_name[mi].c_str(), nll / steps, (double) acc / steps, steps, (long long) n_ctx,
                                (now_ms() - t0) / 1e3);
                else
                    std::printf("%-22s NLL %.4f  top-1 acc %.3f  vs %s: logits cos %.6f, top-1 agree %.3f\n",
                                mode_name[mi].c_str(), nll / steps, (double) acc / steps, mode_name[0].c_str(),
                                cos_sum / steps, (double) agree / steps);
                std::fflush(stdout);
            }
            if (!logits_out.empty()) {
                std::ofstream f(logits_out, std::ios::binary);
                f.write(reinterpret_cast<const char *>(base.data()), (std::streamsize) (base.size() * 4));
                if (!f) throw std::runtime_error("cannot write " + logits_out);
            }
            if (!logits_ref.empty()) {
                std::vector<float> ref(base.size());
                std::ifstream f(logits_ref, std::ios::binary);
                f.read(reinterpret_cast<char *>(ref.data()), (std::streamsize) (ref.size() * 4));
                if (!f) throw std::runtime_error(logits_ref + ": not " + std::to_string(steps) + " steps of logits");
                double cos_sum = 0;
                int agree = 0;
                for (int i = 0; i < steps; ++i) {
                    const float * a = base.data() + (size_t) i * V, * b = ref.data() + (size_t) i * V;
                    double dot = 0, na = 0, nb = 0;
                    int aa = 0, ba = 0;
                    for (int v = 0; v < V; ++v) {
                        dot += (double) a[v] * b[v];
                        na += (double) a[v] * a[v];
                        nb += (double) b[v] * b[v];
                        if (a[v] > a[aa]) aa = v;
                        if (b[v] > b[ba]) ba = v;
                    }
                    cos_sum += dot / std::sqrt(na * nb);
                    agree += aa == ba;
                }
                std::printf("%-22s vs %s: logits cos %.6f, top-1 agree %.3f\n", mode_name[0].c_str(), logits_ref.c_str(),
                            cos_sum / steps, (double) agree / steps);
            }
            if (ov_n) {
                std::printf("DSA selection overlap, adjacent positions (%d pairs): layer overlap / T=2 union\n", ov_n);
                for (int il = 0; il < g.n_layer; ++il)
                    if (mla[il]) std::printf("  layer %2d: %.3f / %.3f\n", il, ov_sum[il] / ov_n, un_sum[il] / ov_n);
            }
            std::printf("cache bytes per token per MLA layer: f16 latent 1024 + pooled 64; int<b>g<G> = 512*b/8 + "
                        "2*512/G (F16 scales)\n");
        }
        prefill_moe_destroy(pm, st);
        for (auto * p : kda_save) if (p) cudaFreeHost(p);
        for (auto * p : lat_save) if (p) cudaFreeHost(p);
        for (auto * p : pool_save) if (p) cudaFreeHost(p);
        for (auto * p : sc_save) if (p) cudaFreeHost(p);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
