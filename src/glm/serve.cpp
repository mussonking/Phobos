// glm_serve: the GLM-5.3-Flash engine behind Strata's OpenAI-compatible server (serve/server.py, --engine strata).
// Same line protocol as `strata --serve` (docs/glm-serve.md): INFO / READY <ctx> stop, then per request
// `GEN <max_new>[ k=v ...] <id,id,...>` -> `PP <done> <total> <ms> <tok/s>` while reading the prompt, `T <id>` per
// token, `DONE <gen> <prompt> <prompt_ms> <decode_ms> <finish> <acc> <off> <reused> <hits> <lookups>` (or `ERR`).
// `STOP` (read on its own thread) ends the current request; `QUIT` / EOF exits. Greedy only: sampling keys are
// accepted and ignored.
//
// One model load: trunk on the GPU, routed experts in the VRAM slot cache (seeded from --profile, adaptive tier every
// --adapt-every steps) over the pinned host arena (all experts), CPU misses through the VNNI kernels on the R8 arena
// (8-row interleaved, default since 2026-10-03: +13 % tok/s at 2K, +23 % at 32K, quality within noise of GGUF against
// an F64 reference; --no-r8 = the GGUF arena, the rollback). INFO reports what is in effect after routed_create's
// fallbacks: arena=r8|gguf slots=N kernels=vnni|f32 down=q8_0|q8_k|f32 (server.py puts them in /health), and Strata's
// own keys for the web app (serve_info.hpp: engine=<version> kv expert_slots expert_cache_mib spec mtp_max
// sampling=greedy). One generation
// session (session.hpp, --spec K MTP drafts per round) holds the conversation. Prefix reuse: a request that extends
// the session's tokens reads only the new suffix; otherwise the longest restorable snapshot inside the common prefix
// is restored (snapshots: end of the first system block, kept, and the end of recent prompts), else it starts over.
// --prefix-cache-gb N (0 = off, the default until its gate passes): kda's host prefix cache (engine/prefix_cache.hpp,
// docs/prefix-cache.md), shared by the sessions: every request's end state and the end of its first system block are
// saved (session_save, copies on the cache's stream); a request whose longest cached prefix beats every session's
// as-is / snapshot reuse is loaded into the least recently used session (session_load). N is a ceiling, pinned
// lazily and clamped at MemAvailable - 64 GB. Its counters go out as `INFO pc_*` lines after each request (/health).
//
//   glm_serve --serve --gguf GGUF [--profile STRP] [--ctx N] [--spec K] [--snapshots N] [--threads N] [--arena GB]
//             [--adapt-every N] [--adapt-swaps N] [--adapt-decay D] [--eos-ids ID,ID,..] [--chunk N] [--prefill-min N] [--prefill-chunk N]
//             [--prefill-exact] [--sessions N] [--kv-latent f16|q8|auto] [--no-r8] [--slots N] [--prefix-cache-gb N]
// --kv-latent: the MLA caches' latent rows (trunk and MTP), F16 or int8 (MlaLatent::Q8: 528 instead of 1024 B per
// token per cache, within engine noise in kda's needle tests). auto (default): Q8 once sessions x ctx passes 2 x 32K
// tokens (the F16 serving configuration validated first); at 2 x 128K, F16 would take ~1.5 GB more VRAM, about 110
// expert slots (2 x 128K with Q8: 1353 slots, against 1398 at 2 x 32K F16).
// Prompt reading: kda's chunked prefill (prefill_moe: hybrid routed experts on a loan of the VRAM cache) for
// suffixes of >= --prefill-min tokens (0 = always the T = 1 decode graph), in session_prompt calls of up to --chunk
// tokens (STOP is checked between them). Default: the fast prefill (asynchronous reclaim of the lent slots, online
// CPU calibration of the GPU / CPU split): equal to a T = 1 prompt up to near-tie flips. --prefill-exact: synchronous
// reclaim, a fixed split and no speculative next-layer uploads, bit-reproducible (+0.2 / 0.35 / 0.9 s at 512 / 2K / 8K tokens).
// The first `T` goes out as soon as the prompt's last chunk has decided it (before the MTP catch-up, the snapshot and
// the first draft round; same token, printed once). stderr gets a `glm_serve: ttft` line per request (TTFT breakdown;
// its last field, `common N`, is the longest prefix the request shared with any session or cache entry at arrival,
// usable or not: N far above `reused` means the reuse was lost to where the state can be cut, not to the client).
// --prefill-chunk N: the prefill's largest chunk (its loan of the slot cache grows with it). Default 8192, halved
// until the loan leaves >= 1/4 of the expert slots (a 24 GB card: smaller chunks instead of a cache emptied).
// CPU: GLM_CPU_ISA=auto|vnni|avx2|scalar (strata/glm/cpu_isa.hpp); without AVX512 VNNI the misses run AVX2 int8 on
// the GGUF arena and the prefill has no CPU share (every non-resident expert uploaded).
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pool.hpp"
#include "routed.hpp"
#include "prefill_moe.hpp"
#include "prefill_prompt.hpp"
#include "pinned_budget.hpp"
#include "prefix_cache.hpp"
#include "serve_info.hpp"
#include "session.hpp"
#include "strata/glm/cpu_isa.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"

using namespace strata::glm;

namespace {
constexpr int32_t kUserTok = 154827;   // <|user|>: the GLM template's first one ends the shared system block

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// wall clock (ms since the epoch): the ttft line's stamps line up with the HTTP client's
double epoch_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();
}
std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> v;
    size_t a = 0;
    for (size_t b; (b = s.find(c, a)) != std::string::npos; a = b + 1) v.push_back(s.substr(a, b - a));
    v.push_back(s.substr(a));
    return v;
}

// stdin on its own thread: STOP sets the flag (also mid-request), other lines queue; EOF queues QUIT
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::pair<std::string, double>> lines;   // with the epoch ms its last byte was read
    std::atomic<bool> stop{false};
    double recv = 0;   // next()'s line's read stamp
    void run() {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.emplace_back(l, epoch_ms());
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        lines.emplace_back("QUIT", epoch_ms());
        cv.notify_one();
    }
    std::string next() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty(); });
        std::string l = std::move(lines.front().first);
        recv = lines.front().second;
        lines.pop_front();
        return l;
    }
};

// GEN <max_new>[ k=v ...] <id,id,...>: false on a malformed line
bool parse_gen(const std::string & line, int64_t & max_new, std::vector<int32_t> & ids, std::string & err) {
    const auto f = split(line, ' ');
    if (f.size() < 3 || f[0] != "GEN") { err = "expected: GEN <max_new> <id,id,...>"; return false; }
    max_new = std::atoll(f[1].c_str());
    if (max_new < 1) { err = "bad request: max_new"; return false; }
    size_t i = 2;
    while (i < f.size() && f[i].find('=') != std::string::npos) ++i;   // sampling keys: greedy engine, ignored
    if (i + 1 != f.size()) { err = "bad request: ids"; return false; }
    ids.clear();
    for (const auto & s : split(f[i], ',')) {
        char * e = nullptr;
        const long v = std::strtol(s.c_str(), &e, 10);
        if (s.empty() || *e != '\0') { err = "bad request: ids"; return false; }
        ids.push_back((int32_t) v);
    }
    return true;
}
}  // namespace

FILE * route_dump = nullptr;   // GLM_ROUTE_DUMP

int main(int argc, char ** argv) {
    std::string gguf, profile;
    int64_t ctx = 32768;
    // --threads 0 (the default): one per physical core, all nodes
    int threads = 0, adapt_every = 16, adapt_swaps = 96, chunk = 8192, spec_k = 1, n_snaps = 4;
    float adapt_decay = 0.7f;
    int prefill_min = 16, prefill_chunk = 8192;
    bool prefill_chunk_set = false;   // --prefill-chunk given: no automatic sizing
    bool prefill_exact = false;
    int n_sess = 2;
    int slots = 0;   // --slots N: fixed VRAM expert slots (0 = free VRAM minus the reserve)
    std::string kv_latent = "auto";
    bool r8 = true;   // --no-r8: GGUF arena (rollback)
    double arena_gb = 0;
    double pc_gb = 0;   // --prefix-cache-gb N: host prefix cache ceiling (0 = off)
    std::vector<int32_t> eos = {154820, 154827, 154829};   // eos, <|user|>, <|observation|> (GLM-5.3 vocabulary)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") {}
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoll(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--arena" && i + 1 < argc) arena_gb = std::atof(argv[++i]);
        else if (a == "--adapt-every" && i + 1 < argc) adapt_every = std::atoi(argv[++i]);
        else if (a == "--adapt-swaps" && i + 1 < argc) adapt_swaps = std::atoi(argv[++i]);
        else if (a == "--adapt-decay" && i + 1 < argc) adapt_decay = (float) std::atof(argv[++i]);
        else if (a == "--chunk" && i + 1 < argc) chunk = std::max(1, std::atoi(argv[++i]));
        else if (a == "--prefill-min" && i + 1 < argc) prefill_min = std::atoi(argv[++i]);
        else if (a == "--prefill-exact") prefill_exact = true;
        else if (a == "--sessions" && i + 1 < argc) n_sess = std::max(1, std::atoi(argv[++i]));
        else if (a == "--prefill-chunk" && i + 1 < argc) {
            prefill_chunk = std::max(64, std::atoi(argv[++i]));
            prefill_chunk_set = true;
        }
        else if (a == "--spec" && i + 1 < argc) spec_k = std::atoi(argv[++i]);
        else if (a == "--snapshots" && i + 1 < argc) n_snaps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--r8") r8 = true;
        else if (a == "--no-r8") r8 = false;
        else if (a == "--slots" && i + 1 < argc) slots = std::atoi(argv[++i]);
        else if (a == "--prefix-cache-gb" && i + 1 < argc) pc_gb = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--kv-latent" && i + 1 < argc) {
            kv_latent = argv[++i];
            if (kv_latent != "f16" && kv_latent != "q8" && kv_latent != "auto") {
                std::fprintf(stderr, "glm_serve: --kv-latent f16|q8|auto\n");
                return 2;
            }
        } else if (a == "--eos-ids" && i + 1 < argc) {
            eos.clear();
            for (const auto & s : split(argv[++i], ',')) eos.push_back(std::atoi(s.c_str()));
        } else { std::fprintf(stderr, "glm_serve: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (gguf.empty()) { std::fprintf(stderr, "usage: glm_serve --serve --gguf GGUF [options]\n"); return 2; }
    if (threads <= 0) {
        threads = physical_cores();
        std::fprintf(stderr, "glm_serve: %d CPU threads (one per physical core)\n", threads);
    } else
        std::fprintf(stderr, "glm_serve: %d CPU threads (--threads)\n", threads);
    const MlaLatent latent = kv_latent == "q8" || (kv_latent == "auto" && (int64_t) n_sess * ctx > 2 * 32768)
                                 ? MlaLatent::Q8 : MlaLatent::F16;

    cudaStream_t st;
    if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) { std::fprintf(stderr, "stream\n"); return 1; }
    std::unique_ptr<Model> m;
    RoutedPtr rt;
    PrefillMoe * pm = nullptr;
    try {
        LoadOptions lo;
        lo.load_mtp = spec_k > 0;
        m = load_model(gguf, lo, st);
        const Geometry & g = m->g;
        // VRAM the slot cache leaves free: per session its MLA caches at full context, workspace, graphs and states
        size_t per = (size_t) 512 << 20;
        for (int il = 0; il < g.n_layer; ++il)
            if (m->layers[il].mla) per += mla_cache_bytes(g, ctx + 16, latent);
        if (spec_k > 0) per += mla_cache_bytes(g, ctx + 16 + spec_k, latent);   // the MTP block's cache
        per += mla_scratch_bytes(g, 8, ctx + 16);
        const size_t reserve = ((size_t) 5 << 29) + (size_t) n_sess * per;
        RoutedOptions ro;
        ro.cpu_threads = threads;
        ro.arena = true;
        ro.arena_max_bytes = (size_t) (arena_gb * 1e9);
        ro.vnni = true;
        ro.r8 = r8;
        ro.pin_arena = true;
        ro.adapt_every = adapt_every;
        ro.adapt_swaps = adapt_swaps;
        ro.adapt_decay = adapt_decay;
        // GLM_ROUTE_DUMP=path: every routed_finish routing appended to path (expert cache studies; off by default).
        // Records: int32 layer, int32 T, int32 ids[T * 8], float w[T * 8]; a request start is layer -1 with T = the
        // request's prompt length, no payload. The prefill's own routing (prefill_moe) is not seen here.
        if (const char * dp = std::getenv("GLM_ROUTE_DUMP")) {
            route_dump = std::fopen(dp, "ab");
            if (!route_dump) throw std::runtime_error(std::string("GLM_ROUTE_DUMP: cannot open ") + dp);
            ro.on_route = [](int il, const int32_t * ids, const float * w, int T) {
                const int32_t h[2] = {il, T};
                std::fwrite(h, 4, 2, route_dump);
                std::fwrite(ids, 4, (size_t) T * 8, route_dump);
                std::fwrite(w, 4, (size_t) T * 8, route_dump);
            };
        }
        ro.max_T = 8;
        ro.reserve_bytes = reserve;
        ro.slots = slots;   // --slots N: fixed residency (reproducible runs), else free VRAM minus the reserve
        rt = routed_create(*m, ro);
        routed_load(*rt, profile.empty() ? routed_rank_counts(*rt) : routed_read_profile(profile), st);
        // arena not pinned (WSL2: 1 GiB of page-locked memory per process in total): the other pinned buffers must fit
        // GLM_PINNED_BUDGET_MB (default 900 under WSL2): fewer snapshots, then a smaller prefill chunk, then fewer
        // sessions (engine/pinned_budget.hpp); no prefix cache
        if (const size_t budget = routed_arena_pinned(*rt) ? 0 : pinned_budget_bytes()) {
            const size_t D = (size_t) g.n_embd, K = (size_t) g.n_expert_used, NE = (size_t) g.n_expert;
            PinnedNeeds nd;
            for (int il = 0; il < g.n_layer; ++il)
                if (m->layers[il].mla) nd.per_session += mla_cache_host_bytes(ctx + 16);
                else nd.per_snapshot += kda_state_bytes(g);
            if (spec_k > 0) nd.per_session += mla_cache_host_bytes(ctx + 16 + spec_k);
            nd.per_session += 2 * 8 * D * 4 + 64;   // workspace + MTP embedding staging (T = 8), positions
            nd.per_snapshot += D * 4;
            nd.per_chunk_token = 2 * K * 4 + 3 * D * 4 + 4;   // prefill_moe: ids, w, y, e, acc, rows
            nd.fixed = routed_pinned_bytes(*rt) + NE * (sizeof(ExpertRef) + 8);   // + prefill table, counts, order
            const PinnedPlan pp = pinned_budget_fit(nd, budget, n_sess, n_snaps, prefill_min > 0 ? prefill_chunk : 0);
            std::fprintf(stderr, "glm_serve: page-locked budget %zu MB (arena not pinned): sessions %d -> %d, snapshots %d -> "
                                 "%d, prefill chunk %d -> %d, %.0f MB%s\n",
                         budget >> 20, n_sess, pp.sessions, n_snaps, pp.snapshots, prefill_min > 0 ? prefill_chunk : 0,
                         pp.chunk, pp.bytes / 1048576.0, pc_gb > 0 ? "; prefix cache off" : "");
            if (!pp.fits)
                throw std::runtime_error("page-locked budget: one session of context " + std::to_string(ctx) + " with one "
                                         "snapshot needs " + std::to_string(pp.bytes >> 20) + " MB, over the " +
                                         std::to_string(budget >> 20) + " MB budget (GLM_PINNED_BUDGET_MB): lower --ctx");
            n_sess = pp.sessions;
            n_snaps = pp.snapshots;
            if (prefill_min > 0) prefill_chunk = pp.chunk;
            pc_gb = 0;
        }
        if (prefill_min > 0 && !routed_can_lend(*rt)) {
            // the prefill's loan (routed_lend) re-uploads the experts it evicts from the arena: pinned, or through the
            // bounce ring
            std::fprintf(stderr, "glm_serve: prefill off: the host arena is neither pinned nor served by a pinned bounce "
                                 "ring, prompts go through the decode graph\n");
            prefill_min = 0;
        }
        if (prefill_min > 0) {
            PrefillMoeOptions po;
            po.max_T = prefill_chunk;
            po.async_reclaim = po.calibrate = !prefill_exact;
            if (prefill_exact) po.spec = 0;   // exact mode: no speculative next-layer uploads either
            pm = prefill_moe_create(*m, *rt, po, st);
            if (!prefill_chunk_set) {
                const int cap = prefill_prompt_fit_cap(pm, *m, ctx + 16, spec_k > 0, routed_slots(*rt));
                if (cap == 0) {
                    std::fprintf(stderr, "glm_serve: prefill off: its loan does not fit in 3/4 of %d expert slots even "
                                         "for 64-token chunks (context %lld), prompts go through the decode graph\n",
                                 routed_slots(*rt), (long long) ctx);
                    prefill_moe_destroy(pm, st);
                    pm = nullptr;
                    prefill_min = 0;
                } else if (cap != prefill_chunk) {
                    std::fprintf(stderr, "glm_serve: prefill chunk %d -> %d (its loan must leave 1/4 of %d expert slots)\n",
                                 prefill_chunk, cap, routed_slots(*rt));
                    prefill_moe_destroy(pm, st);
                    po.max_T = prefill_chunk = cap;
                    pm = prefill_moe_create(*m, *rt, po, st);
                }
            }
            if (pm && !prefill_moe_options(pm).cpu)
                std::fprintf(stderr, "glm_serve: prefill CPU share off (CPU isa %s: it needs AVX512 VNNI), every "
                                     "non-resident expert uploaded\n", cpu_isa_name(cpu_isa()));
        }
        std::fprintf(stderr, "glm_serve: %d expert slots, %.2f GB reserved for %d sessions of %lld tokens (%s MLA latent)\n",
                     routed_slots(*rt), reserve / 1e9, n_sess, (long long) ctx, latent == MlaLatent::Q8 ? "int8" : "F16");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "glm_serve: LOAD FAILED: %s\n", e.what());
        return 1;
    }

    SessionExperts ex;
    ex.eager = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T, cudaStream_t s) {
        routed_run(*rt, il, y, ids, w, out, T, s);
    };
    ex.start = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T, cudaStream_t s) {
        routed_start(*rt, il, y, ids, w, out, T, s);
    };
    ex.finish = [&](int il, float * out, int T, cudaStream_t s) { routed_finish(*rt, il, out, T, s); };
    ex.step = [&](cudaStream_t s) { routed_step(*rt, s); };   // adaptive swaps, and admits landed uploads
    SessionOptions so;
    so.max_ctx = ctx + 16;
    so.spec_k = spec_k;
    so.mla_latent = latent;
    if (pm) {
        ex.prefill = pm;
        so.prefill_min = prefill_min;
    }
    // Conversations: one session each (its own attention caches and snapshots), --sessions of them. A request goes
    // to the one it can reuse the most of; with nothing to reuse, to the least recently used one (started over).
    // Claude Code interleaves its main conversation with other prompts (e.g. a transcript classifier with its own
    // ~30K-token system prompt): with a single session each of them would evict the other.
    // Snapshots (session.hpp): one at the end of the first system block (the shared system prompt + tools, kept),
    // the others at the end of each request's prompt (least recently used replaced). A snapshot is only restorable
    // while its tokens prefix its session's, so the rest are dropped.
    struct Snap {
        SessionSnapshotPtr p;
        bool base = false;
        int64_t used = 0;
    };
    struct Conv {
        SessionPtr ss;
        std::vector<Snap> snaps;
        int64_t used = 0;
    };
    std::vector<Conv> convs(n_sess);
    try {
        for (auto & c : convs) c.ss = session_create(*m, so, ex, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "glm_serve: SESSION FAILED: %s\n", e.what());
        return 1;
    }
    // one host prefix cache for all sessions (created after them: its pinning thread starts at once)
    PrefixCachePtr pc(nullptr, prefix_cache_destroy);
    if (pc_gb > 0) {
        try {
            PrefixCacheOptions po;
            po.cap_bytes = (size_t) (pc_gb * (1ull << 30));
            pc = prefix_cache_create(po);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "glm_serve: PREFIX CACHE FAILED: %s\n", e.what());
            return 1;
        }
    }
    auto pc_info = [&] {
        const PrefixCacheStats s = prefix_cache_stats(*pc);
        std::printf("INFO pc_cap_mb=%zu pc_pinned_mb=%zu pc_used_mb=%zu pc_entries=%lld pc_segments=%lld pc_hits=%lld "
                    "pc_misses=%lld pc_saves=%lld pc_save_skips=%lld pc_evictions=%lld pc_tokens_restored=%lld "
                    "pc_restored_mb=%zu pc_saved_mb=%zu pc_clamped=%d\n", s.cap >> 20, s.pinned >> 20, s.used >> 20,
                    (long long) s.entries, (long long) s.segments, (long long) s.hits, (long long) s.misses,
                    (long long) s.saves, (long long) s.save_skips, (long long) s.evictions,
                    (long long) s.tokens_restored, s.bytes_restored >> 20, s.bytes_saved >> 20, (int) s.clamped);
    };
    int64_t clock = 0;
    const size_t snap_mb = session_snapshot_bytes(*convs[0].ss) >> 20;
    auto is_prefix = [](const std::vector<int32_t> & a, const int32_t * b, int64_t nb) {
        return (int64_t) a.size() <= nb && std::equal(a.begin(), a.end(), b);
    };
    auto prune = [&](Conv & c) {
        const auto & cur = session_tokens(*c.ss);
        for (auto & x : c.snaps)
            if (x.p && !is_prefix(session_snapshot_tokens(*x.p), cur.data(), (int64_t) cur.size())) x.p.reset();
        c.snaps.erase(std::remove_if(c.snaps.begin(), c.snaps.end(), [](const Snap & x) { return !x.p; }),
                      c.snaps.end());
    };
    auto take_snapshot = [&](Conv & c, bool base) {
        const int64_t pos = session_pos(*c.ss);
        for (auto & x : c.snaps)
            if (session_snapshot_pos(*x.p) == pos) { x.used = ++clock; x.base |= base; return; }   // (a prefix: same tokens)
        Snap * slot = nullptr;
        if ((int) c.snaps.size() < n_snaps) slot = &c.snaps.emplace_back();
        else {
            for (auto & x : c.snaps)
                if (!x.base && (!slot || x.used < slot->used)) slot = &x;
            if (!slot) return;
        }
        session_snapshot(*c.ss, slot->p);   // reuses the replaced one's pinned buffer
        slot->base = base;
        slot->used = ++clock;
    };
    // how much of ids conversation c can reuse: the session as it is (-> true in *as_is), or its longest restorable
    // snapshot inside the common prefix (-> *snap)
    auto reusable = [&](Conv & c, const std::vector<int32_t> & ids, bool * as_is, Snap ** snap,
                        int64_t * common) -> int64_t {
        const auto & cur = session_tokens(*c.ss);
        const int64_t n = (int64_t) ids.size();
        int64_t L = 0;
        while (L < (int64_t) cur.size() && L < n && cur[L] == ids[L]) ++L;
        *common = std::max(*common, L);
        *as_is = false;
        *snap = nullptr;
        if (L == (int64_t) cur.size() && L > 0 && (L < n || session_pending(*c.ss) >= 0)) {
            *as_is = true;
            return L;
        }
        for (auto & x : c.snaps) {
            const int64_t sp = session_snapshot_pos(*x.p);
            if (sp > 0 && sp <= L && (!*snap || sp > session_snapshot_pos(*(*snap)->p)) && session_restorable(*c.ss, *x.p))
                *snap = &x;
        }
        return *snap ? session_snapshot_pos(*(*snap)->p) : 0;
    };

    Input in;
    std::thread reader([&] { in.run(); });
    reader.detach();
    const RoutedOptions & ro = routed_options(*rt);
    ServeInfo si;
    si.spec_k = spec_k;
    si.adapt_every = adapt_every;
    si.sessions = n_sess;
    si.snapshots = n_snaps;
    si.snapshot_mb = snap_mb;
    si.prefill_min = pm ? prefill_min : 0;
    si.q8 = latent == MlaLatent::Q8;
    si.r8 = ro.r8;
    si.vnni = ro.vnni;
    si.down_q8_0 = ro.down_q8_0;
    si.slots = routed_slots(*rt);
    si.slot_bytes = routed_blob_bytes(m->g);
    si.cpu_isa = cpu_isa_name(cpu_isa());
    std::printf("%s\n", serve_info_line(si).c_str());
    if (pc) pc_info();
    std::printf("READY %lld stop\n", (long long) ctx);
    std::fflush(stdout);

    GenerateControl ctl;
    ctl.stop_ids = eos;
    ctl.cancel = &in.stop;
    for (;;) {
        const std::string line = in.next();
        const double e_deq = epoch_ms(), e_recv = in.recv;
        if (line == "QUIT") break;
        if (line.empty()) continue;
        int64_t max_new = 0;
        std::vector<int32_t> ids;
        std::string err;
        if (!parse_gen(line, max_new, ids, err)) { std::printf("ERR %s\n", err.c_str()); std::fflush(stdout); continue; }
        const int64_t n = (int64_t) ids.size();
        if (n + max_new + 8 > ctx) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                        (long long) max_new, (long long) ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); continue; }
        in.stop.store(false);   // a STOP between requests is ignored

        if (route_dump) {
            const int32_t h[2] = {-1, (int32_t) ids.size()};
            std::fwrite(h, 4, 2, route_dump);
        }
        const double e_sel = epoch_ms();
        const double t0 = now_ms();
        const char * finish = "length";
        int64_t gen = 0, reused = 0;
        double prompt_ms = 0, decode_ms = 0;
        // the conversation to run in: the most reusable, else the least recently used (started over)
        Conv * cv = nullptr;
        bool as_is = false;
        Snap * snap = nullptr;
        int64_t common = 0;   // (snapshots prefix their session's tokens: the sessions cover them)
        for (auto & c : convs) {
            bool a = false;
            Snap * sn = nullptr;
            const int64_t r = reusable(c, ids, &a, &sn, &common);
            if (r > reused || (!cv && r == reused)) { cv = &c; as_is = a; snap = sn; reused = r; }
        }
        // the prefix cache, when its longest entry beats every session's reuse: loaded into the least recently used
        // session (whatever that one holds is already in the cache, saved at the end of its last request)
        PrefixEntry * pe = nullptr;
        if (pc) {
            common = std::max(common, prefix_cache_common(*pc, ids.data(), n));
            pe = prefix_cache_lookup(*pc, ids.data(), n);
            const int64_t P = pe ? (int64_t) pe->toks.size() : 0;
            if (P <= reused || (P == n && pe->next < 0)) pe = nullptr;
        }
        if (reused == 0 || pe)
            for (auto & c : convs)
                if (c.used < cv->used) cv = &c;
        cv->used = ++clock;
        Session & ss = *cv->ss;
        auto & snaps = cv->snaps;
        RoutedStats & rs = routed_stats(*rt);
        SessionStats & sst = session_stats(ss);
        const int64_t hits0 = rs.hits, look0 = rs.hits + rs.misses, acc0 = sst.accepted, off0 = sst.drafted;
        // TTFT breakdown (stderr "glm_serve: ttft" line, after the first token)
        const double e_parsed = epoch_ms();
        const PrefillMoeStats pms0 = pm ? prefill_moe_stats(pm) : PrefillMoeStats{};
        const SessionStats sst0 = sst;
        const char * path = pe ? "cache" : snap ? "snap" : as_is ? "asis" : "reset";
        double m_match = 0, m_prompt = 0, m_snap = 0, e_first = 0, m_gen0 = 0;
        bool early = false, round0 = false;   // first token streamed by session_prompt / round 0 done
        SessionStats sst1, sstf;
        try {
            const double mt0 = now_ms();
            if (pe) {
                try {
                    session_load(ss, *pc, *pe);
                    reused = (int64_t) pe->toks.size();
                    prune(*cv);
                } catch (const std::exception & e) {   // the session was reset: read the whole prompt
                    std::fprintf(stderr, "glm_serve: prefix cache load failed (%s), reading the prompt\n", e.what());
                    snaps.clear();
                    reused = 0;
                    path = "reset";
                }
            } else if (snap) {
                session_restore(ss, *snap->p);
                snap->used = ++clock;
            } else if (!as_is) {
                session_reset(ss);
                snaps.clear();
                reused = 0;
            }
            m_match = now_ms() - mt0;
            // the first system block ends where the first <|user|> starts (GLM template)
            int64_t base_end = -1;
            for (int64_t i = 0; i < n; ++i)
                if (ids[i] == kUserTok) { base_end = i; break; }
            int64_t p = reused;
            double last_pp = 0;
            while (p < n && !in.stop.load()) {
                int64_t c = std::min<int64_t>(chunk, n - p);
                if (base_end > p && base_end < p + c) c = base_end - p;   // stop at the system block's end
                const int64_t p0 = p;
                const double pt0 = now_ms();
                session_prompt(ss, ids.data() + p, (int) c, [&](int done, int) {
                    const double ms = now_ms() - t0;
                    if (ms - last_pp < 500 && p0 + done < n) return;
                    last_pp = ms;
                    std::printf("PP %lld %lld %.0f %.1f\n", (long long) (p0 + done), (long long) n, ms,
                                ms > 0 ? 1000.0 * (p0 + done - reused) / ms : 0.0);
                    std::fflush(stdout);
                }, [&](int32_t t) {
                    // early first token: the trunk's greedy next, streamed before the MTP catch-up, the snapshot and
                    // round 0 (session_generate emits it again first: not printed twice)
                    if (p0 + c != n) return;
                    std::printf("T %d\n", t);
                    std::fflush(stdout);
                    early = true;
                    e_first = epoch_ms();
                });
                m_prompt += now_ms() - pt0;
                p += c;
                const double st0 = now_ms();
                if (p == base_end) {
                    take_snapshot(*cv, true);
                    if (pc) session_save(ss, *pc);   // the shared system block, for the other sessions too
                }
                m_snap += now_ms() - st0;
            }
            sst1 = sst;
            prune(*cv);
            if (p < n) {
                finish = "cancel";
                prompt_ms = now_ms() - t0;
            } else {
                const double st0 = now_ms();
                if (p > reused) take_snapshot(*cv, false);
                m_snap += now_ms() - st0;
                prompt_ms = now_ms() - t0;
                const double d0 = now_ms();
                std::vector<int32_t> out;
                const FinishReason fr = session_generate(ss, (int) max_new, out,
                    [&](const int32_t * tk, int k, double ms) {
                        const bool first = !round0;
                        for (int i = first && early ? 1 : 0; i < k; ++i) std::printf("T %d\n", tk[i]);
                        std::fflush(stdout);
                        if (first) {
                            round0 = true;
                            if (e_first == 0) e_first = epoch_ms();
                            m_gen0 = ms;
                            sstf = sst;
                        }
                    }, ctl);
                // the early token counts as generated even if a STOP ended the request before round 0
                gen = std::max<int64_t>((int64_t) out.size(), early ? 1 : 0);
                finish = fr == FinishReason::ctx ? "length" : finish_reason_name(fr);
                decode_ms = now_ms() - d0;
            }
        } catch (const std::exception & e) {
            // unknown state after a failure: the next request starts over
            try { session_reset(ss); } catch (...) {}
            snaps.clear();
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        const int64_t hits = rs.hits - hits0, look = rs.hits + rs.misses - look0;
        const int64_t acc = sst.accepted - acc0, off = sst.drafted - off0;
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld\n", (long long) gen, (long long) n, prompt_ms,
                    decode_ms, finish, (long long) acc, (long long) off, (long long) reused, (long long) hits,
                    (long long) look);
        std::fflush(stdout);
        if (route_dump) std::fflush(route_dump);   // whole requests on disk (std::_Exit below skips stdio flushing)
        if (pc) {
            // the request's end state (prompt + emitted tokens), after DONE: copies on the cache's stream, the next
            // request's first call waits for them on the session's stream
            try { session_save(ss, *pc); } catch (const std::exception & e) {
                std::fprintf(stderr, "glm_serve: prefix cache save failed: %s\n", e.what());
            }
            pc_info();
            std::fflush(stdout);
        }
        std::fprintf(stderr, "glm_serve: prompt %lld = %lld reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in "
                             "%.0f ms (%.2f tok/s), %s, drafts %lld/%lld, hits %.1f%%, swaps %lld, session %d, %zu snapshots\n",
                     (long long) n, (long long) reused, (long long) (n - reused), prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (n - reused) / prompt_ms : 0.0, (long long) gen, decode_ms,
                     decode_ms > 0 ? 1000.0 * gen / decode_ms : 0.0, finish, (long long) acc, (long long) off,
                     look ? 100.0 * hits / look : 0.0, (long long) rs.swaps, (int) (cv - convs.data()), snaps.size());
        if (e_first > 0) {
            const PrefillMoeStats pms = pm ? prefill_moe_stats(pm) : PrefillMoeStats{};
            const double C = (double) std::max<int64_t>(1, pms.calls - pms0.calls);
            std::fprintf(stderr,
                "glm_serve: ttft %.1f ms new %lld reused %lld path %s | queue %.1f parse %.1f select %.1f match %.1f "
                "prompt %.1f (draft %.1f; prefill lend %.1f window %.1f plan_wait %.1f cpu_wait %.1f cpu %.1f reclaim "
                "%.1f mtp %.1f; experts/layer res %.1f up %.1f cpu %.1f spec %.1f) snapshot %.1f round0 %.1f (verify "
                "%.1f draft %.1f) early %d | recv %.3f first %.3f common %lld\n",
                e_first - e_recv, (long long) (n - reused), (long long) reused, path, e_deq - e_recv, e_sel - e_deq,
                e_parsed - e_sel, m_match, m_prompt, sst1.draft_ms - sst0.draft_ms, pms.lend_ms - pms0.lend_ms,
                pms.window_ms - pms0.window_ms, pms.plan_wait_ms - pms0.plan_wait_ms,
                pms.cpu_wait_ms - pms0.cpu_wait_ms, pms.cpu_ms - pms0.cpu_ms, pms.reclaim_ms - pms0.reclaim_ms,
                pms.mtp_ms - pms0.mtp_ms, (pms.res_experts - pms0.res_experts) / C,
                (pms.up_experts - pms0.up_experts) / C, (pms.cpu_experts - pms0.cpu_experts) / C,
                (pms.spec_experts - pms0.spec_experts) / C, m_snap, m_gen0, sstf.verify_ms - sst1.verify_ms,
                sstf.draft_ms - sst1.draft_ms, (int) early, e_recv, e_first, (long long) common);
        }
    }
    convs.clear();
    pc.reset();   // unpins its blocks
    if (pm) prefill_moe_destroy(pm, st);
    rt.reset();
    std::fflush(stdout);
    if (route_dump) std::fclose(route_dump);
    std::_Exit(0);   // the reader thread may still block on stdin
}
