// glm_decode: prefill the oracle prompt (real kda/mla, routed experts from the oracle or the ggml-cpu reference),
// then greedy decode with the segmented CUDA graphs (decode.hpp). Checks the first decoded token against the
// oracle and reports per-token times. --zero-experts replaces the routed experts by zeros: the trunk-only floor.
// --routed: decode experts through routed.hpp (VRAM cache hits on the GPU, misses on the host pool); needs an IQ4
// GGUF, or --synthetic GB (random IQ4 blobs: timing only, tokens are meaningless). The cache is filled from
// --profile (STRP) or, with --learn N, from the routing counts of N all-miss decode steps. Prefill experts are
// zeros in routed mode (the oracle prefill belongs to the Q4_K_XL model); --teacher-force instead runs the prompt
// through the decode graph token by token (real routed experts, real routing; without --routed, the ggml-cpu
// reference experts), then compares the greedy tokens with the oracle's. --arena GB copies the non-resident experts into a NUMA-placed host arena at load (0 = all;
// past a cap the misses read the GGUF mapping; --arena-alias aliases them instead: timing only), --save-profile writes the routing counts as an STRP profile.
//
//   glm_decode [--tokens N] [--zero-experts] [--cpu-experts-prefill] [--ctx N] [--gguf SHARD1] [--oracle DIR]
//              [--routed [--synthetic GB | --arena GB [--arena-alias]] [--profile FILE | --learn N] [--threads N] [--vram-gb G | --slots N]
//               [--serial-shexp] [--teacher-force [--prompt TOKENS_FILE]] [--save-profile FILE] [--f32] [--down-q8-k] [--r8]]
//              [--spec K[,K2..] [--no-mtp-share]]   (session.hpp: K MTP drafts per round, 0 = plain greedy; needs
//                                             --routed or --zero-experts; a list runs one session per K in the same
//                                             process, same loaded experts, and checks every K emits the tokens of
//                                             the first K on its prompt path; --test-fault spec|serve: the exit
//                                             code's own test, the last token of every later K (spec@I: of session
//                                             I only) is changed for the K check / for serve-test)
//              [--prefill N [--prefill-chunk C] [--prefill-exact]] [--kv-latent f16|q8]   (with --routed --arena and --spec K,K..: the sessions of odd index
//                                             feed prompts of >= N tokens through the chunked prefill
//                                             (prefill_prompt); --spec 1,1 --prefill 64 compares a prompt fed via
//                                             T = 1 (session 0) and via the prefill (session 1), reported only (the
//                                             exit code counts same-path K comparisons, prefill ones with
//                                             --prefill-exact, and --serve-test; --spec 1,1,1,1 checks both paths);
//                                             --serve-test implies --prefill-exact; a single K prefills its prompt)
//              [--prefix-test]   (with --spec: the host prefix cache between two sessions, exact checks; implies
//                                             --prefill-exact with --prefill; needs --ctx >= prompt + --tokens + 56)
//              [--launch-probe]   (without --spec: after the decode, host cost of each segment launch into an idle
//                                             stream and of the misses' acc upload, uninstrumented)
//              [--profile-span N]   (with --spec: cudaProfilerStart/Stop around N generation rounds of the last K,
//                                             after 4 warm-up rounds; nsys --capture-range=cudaProfilerApi)
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "decode.hpp"
#include "forward.hpp"
#include "prefill_moe.hpp"
#include "prefill_prompt.hpp"
#include "ref_experts.hpp"
#include "routed.hpp"
#include "session.hpp"
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
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --serve-test: the serving API (session.hpp) on a session that fed the prompt P (snapshot s0 taken right after)
// and then generated gen. "exact" checks replay the same batch shapes; "contract" checks cut the rounds elsewhere
// (greedy-equal up to near-tie flips). Deterministic only with --zero-experts or a static routed tier.
bool serve_test(Session & s, const std::vector<int32_t> & P, const std::vector<int32_t> & gen,
                const SessionSnapshot & s0) {
    bool ok = true;
    const auto check = [&](bool c, const char * what) {
        std::printf("  serve: %-66s %s\n", what, c ? "ok" : "FAIL");
        ok = ok && c;
    };
    const auto same_as_gen = [&](const std::vector<int32_t> & o, const char * what) {
        size_t d = 0;
        while (d < o.size() && d < gen.size() && o[d] == gen[d]) ++d;
        if (d == gen.size() && o.size() == gen.size()) return check(true, what);
        std::printf("  serve: %s: first difference at generated token %zu\n", what, d);
        check(false, what);
    };
    const int T = (int) P.size(), N = (int) gen.size();
    if (N < 16) {
        std::printf("  serve: needs --tokens >= 16\n");
        return false;
    }
    std::vector<int32_t> out, rest;
    FinishReason r;
    // 1. back to the prompt; generate with no new prompt token
    session_restore(s, s0);
    check(session_prompt(s, nullptr, 0) == gen[0] && session_pos(s) == T, "restore: pending token = first generated");
    r = session_generate(s, N, out);
    check(r == FinishReason::length, "restore + generate: finish length");
    same_as_gen(out, "restore + generate (empty suffix) = generate [exact]");
    std::vector<int32_t> want(P);
    want.insert(want.end(), gen.begin(), gen.end());
    check(session_tokens(s) == want && session_pos(s) == T + N, "session_tokens = prompt + emitted");
    // 2. max_new cuts rounds: 3 + 3 + the rest
    session_restore(s, s0);
    out.clear();
    for (int n : {3, 3, N - 6}) session_generate(s, n, out);
    same_as_gen(out, "max_new 3 + 3 + rest = generate [contract]");
    // 3. stop id (the first occurrence of gen[N / 2]): emitted and fed, the next token pending
    const int32_t sid = gen[N / 2];
    const int j = (int) (std::find(gen.begin(), gen.end(), sid) - gen.begin());
    session_restore(s, s0);
    out.clear();
    GenerateControl ctl;
    ctl.stop_ids = {sid};
    r = session_generate(s, N, out, {}, ctl);
    check(r == FinishReason::stop && out == std::vector<int32_t>(gen.begin(), gen.begin() + j + 1),
          "stop id: finish stop, emitted up to and including it [contract]");
    check(session_pos(s) == T + j + 1 && session_pending(s) == gen[j + 1], "stop id: pending = the token after it");
    // 4. snapshot mid-conversation; a follow-up turn; restore; the same turn again
    SessionSnapshotPtr s1;
    session_snapshot(s, s1);
    check(session_snapshot_pos(*s1) == T + j + 1 && session_snapshot_tokens(*s1) == session_tokens(s),
          "snapshot accessors");
    const int U = std::min(T, 16);   // the follow-up turn: the prompt's first U tokens
    std::vector<int32_t> a1, a2;
    session_prompt(s, P.data(), U);
    session_generate(s, 12, a1);
    session_restore(s, *s1);
    session_prompt(s, P.data(), U);
    session_generate(s, 12, a2);
    check(a1 == a2, "snapshot, turn, restore, same turn = same tokens [exact]");
    session_restore(s, *s1);
    session_generate(s, N - j - 1, rest);
    out.insert(out.end(), rest.begin(), rest.end());
    same_as_gen(out, "stop + continue after a restore = generate [contract]");
    // 5. cancel before the first round
    std::atomic<bool> cancel{true};
    GenerateControl cc;
    cc.cancel = &cancel;
    rest.clear();
    r = session_generate(s, 8, rest, {}, cc);
    check(r == FinishReason::cancel && rest.empty(), "cancel: finish cancel, nothing emitted");
    // 6. reset; another conversation; restoring s1 must refuse
    session_reset(s);
    check(session_pos(s) == 0 && session_pending(s) == -1 && session_tokens(s).empty(), "reset: empty state");
    session_prompt(s, P.data() + 1, U);
    bool threw = false;
    try {
        session_restore(s, *s1);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "restore of a non-prefix snapshot throws");
    // 7. reset; the original prompt again
    session_reset(s);
    check(session_prompt(s, P.data(), T) == gen[0], "reset + prompt: same first token");
    out.clear();
    session_generate(s, N, out);
    same_as_gen(out, "reset + prompt + generate = generate [exact]");
    // 8. run into the context capacity (glm_decode sizes it at prompt + --tokens + 16)
    rest.clear();
    r = session_generate(s, 1 << 20, rest);
    check(r == FinishReason::ctx && !rest.empty(), "generate to the context capacity: finish ctx");
    return ok;
}

// --prefix-test: the host prefix cache (prefix_cache.hpp) between session a (the one that generated gen from P) and
// a second session b of the same options. "exact" checks compare with a session continuing as is.
bool prefix_test(Session & a, Session & b, const std::vector<int32_t> & P, const std::vector<int32_t> & gen) {
    bool ok = true;
    const auto check = [&](bool c, const char * what) {
        std::printf("  prefix: %-74s %s\n", what, c ? "ok" : "FAIL");
        ok = ok && c;
    };
    const int T = (int) P.size(), N = (int) gen.size();
    if (N < 16 || T < 32) {
        std::printf("  prefix: needs --tokens >= 16 and a prompt of >= 32 tokens\n");
        return false;
    }
    const auto stats = [](const PrefixCache & pc, const char * when) {
        const PrefixCacheStats st = prefix_cache_stats(pc);
        std::printf("  prefix: [%s] entries %lld segments %lld, used %.1f MB of %.1f MB pinned, saves %lld skips %lld "
                    "evictions %lld, hits %lld misses %lld, restored %lld tokens %.1f MB\n", when,
                    (long long) st.entries, (long long) st.segments, st.used / 1e6, st.pinned / 1e6,
                    (long long) st.saves, (long long) st.save_skips, (long long) st.evictions, (long long) st.hits,
                    (long long) st.misses, (long long) st.tokens_restored, st.bytes_restored / 1e6);
        return st;
    };
    const auto timed_load = [](Session & s, PrefixCache & pc, PrefixEntry & e, const char * what) {
        const double t0 = now_ms();
        const int64_t n = session_load(s, pc, e);
        std::printf("  prefix: load %s: %lld rows imported in %.1f ms\n", what, (long long) n, now_ms() - t0);
        return n;
    };
    const auto timed_save = [](Session & s, PrefixCache & pc, const char * what) {
        const double t0 = now_ms();
        session_save(s, pc);
        const double t1 = now_ms();
        cudaStreamSynchronize(pc.st);
        std::printf("  prefix: save %s at %lld: queued in %.1f ms, copies done after %.1f ms\n", what,
                    (long long) session_pos(s), t1 - t0, now_ms() - t0);
    };
    // the gate contract (docs/prefix-cache.md section 3): a loaded state is bitwise the saved one, part by part
    using Digest = std::vector<std::pair<std::string, uint64_t>>;
    const auto same_state = [&](const Digest & want, const Digest & got, const char * what) {
        size_t bad = 0;
        for (size_t i = 0; i < want.size() && i < got.size(); ++i)
            if (want[i] != got[i] && bad++ == 0) std::printf("  prefix: %s: first different part %s\n", what, got[i].first.c_str());
        char line[128];
        std::snprintf(line, sizeof line, "%s: %zu state parts bitwise equal to the saved ones", what, want.size());
        check(bad == 0 && want.size() == got.size(), line);
    };
    PrefixCacheOptions po;
    po.cap_bytes = 8ull << 30;
    po.reserve_bytes = 16ull << 30;
    po.block_bytes = 16ull << 20;   // small blocks and segments: several segments per prompt, eviction granularity
    po.segment_max = 1024;
    PrefixCachePtr pc = prefix_cache_create(po);
    const int U = 16;   // the follow-up turn: the prompt's first U tokens
    std::vector<int32_t> out, x1, x2;

    // 1. a: P, save E0, generate, save E1 (shares E0's segments), a follow-up as the reference
    session_reset(a);
    check(session_prompt(a, P.data(), T) == gen[0], "a: prompt = first generated token");
    timed_save(a, *pc, "E0");
    const Digest dE0 = session_state_digest(a);
    const int64_t seg0 = prefix_cache_stats(*pc).segments;
    check(seg0 == (T + po.segment_max - 1) / po.segment_max, "E0: one segment per segment_max rows");
    session_generate(a, N, out);
    check(out == gen, "a: generate after a save = generate [exact]");
    timed_save(a, *pc, "E1");
    const Digest dE1 = session_state_digest(a);
    check(prefix_cache_stats(*pc).segments == seg0 + 1, "E1: shares E0's segments, one new segment");
    session_prompt(a, P.data(), U);
    session_generate(a, 12, x1);
    timed_save(a, *pc, "E2");
    stats(*pc, "after E0 E1 E2");
    std::vector<int32_t> te1(P);
    te1.insert(te1.end(), gen.begin(), gen.end());

    // 1b. prefix_cache_common (glm_serve's `common N`): longest prefix shared with any entry, no stats, no LRU
    {
        const std::vector<int32_t> & te2 = session_tokens(a);   // E2's tokens
        const PrefixCacheStats st0 = prefix_cache_stats(*pc);
        std::vector<int32_t> q(P);
        q[T - 1] = P[T - 1] == 0 ? 1 : P[T - 1] - 1;
        const bool c1 = prefix_cache_common(*pc, q.data(), T) == T - 1;
        q[0] = P[0] == 0 ? 1 : P[0] - 1;
        const bool c2 = prefix_cache_common(*pc, q.data(), T) == 0;
        std::vector<int32_t> longer(te2);
        longer.push_back(gen[0]);
        const bool c3 = prefix_cache_common(*pc, longer.data(), (int64_t) longer.size()) == (int64_t) te2.size();
        const bool c4 = prefix_cache_common(*pc, te2.data(), U) == U;
        const PrefixCacheStats st1 = prefix_cache_stats(*pc);
        check(c1 && c2 && c3 && c4 && st1.hits == st0.hits && st1.misses == st0.misses,
              "common: last token differs T-1, first 0, past E2 |E2|, short U; no stats");
    }

    // 2. b holds another conversation; lookup + load E1; the same follow-up
    session_reset(b);
    session_prompt(b, P.data() + 1, std::min(T - 1, 64));
    out.clear();
    session_generate(b, 4, out);
    std::vector<int32_t> req(te1);
    req.insert(req.end(), P.begin(), P.begin() + U);
    PrefixEntry * e1 = prefix_cache_lookup(*pc, req.data(), (int64_t) req.size());
    check(e1 && e1->toks == te1, "lookup of E1 + follow-up: E1 (longest prefix)");
    if (!e1) return false;
    check(timed_load(b, *pc, *e1, "E1 into b") == T + N, "load into another conversation imports every row");
    same_state(dE1, session_state_digest(b), "E1 loaded into b");
    check(session_pos(b) == T + N && session_tokens(b) == te1 && session_pending(b) == e1->next,
          "b: position, tokens and pending token of E1");
    session_prompt(b, P.data(), U);
    session_generate(b, 12, x2);
    check(x1 == x2, "b after load + follow-up = a's follow-up as is [exact]");
    check(session_tokens(b) == session_tokens(a), "b and a: same tokens");

    // 3. a: back to E0 (its own chain: nothing imported), generate = gen
    PrefixEntry * e0 = prefix_cache_lookup(*pc, P.data(), T);
    check(e0 && (int) e0->toks.size() == T, "lookup of P: E0");
    if (!e0) return false;
    check(timed_load(a, *pc, *e0, "E0 into a") == 0, "load of an entry on its own chain imports nothing");
    same_state(dE0, session_state_digest(a), "E0 loaded into a (rows kept)");
    out.clear();
    session_generate(a, N, out);
    check(out == gen, "a after load E0 + generate = generate [exact]");

    // 4. snapshots inside loaded rows: b's own snapshot at T, then a load from a's chain (nothing shared)
    session_reset(b);
    session_prompt(b, P.data(), T);
    SessionSnapshotPtr sb;
    session_snapshot(b, sb);
    session_load(b, *pc, *e1);
    check(session_restorable(b, *sb) == (T % 4 == 0), "b's snapshot at T after a load: restorable iff T % 4 == 0");
    bool threw = false;
    try {
        session_restore(b, *sb);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw == (T % 4 != 0), "session_restore refuses exactly the non-restorable snapshot");

    // 5. eviction with shared segments: no more blocks once F1 is in; F2 (same conversation) evicts F1, keeps its
    //    segments; b loads F2 and continues like a
    PrefixCacheOptions po2 = po;
    po2.spare_bytes = 0;   // no blocks pinned ahead: the cap below is exact
    PrefixCachePtr pc2 = prefix_cache_create(po2);
    session_reset(a);
    session_prompt(a, P.data(), T);
    out.clear();
    session_generate(a, N, out);
    timed_save(a, *pc2, "F1");
    pc2->opt.cap_bytes = prefix_cache_stats(*pc2).pinned;
    const int64_t segs1 = prefix_cache_stats(*pc2).segments;
    session_prompt(a, P.data(), U);
    out.clear();
    session_generate(a, 12, out);
    timed_save(a, *pc2, "F2");
    const Digest dF2 = session_state_digest(a);
    const PrefixCacheStats s2 = stats(*pc2, "after F1 F2, cap = F1's blocks");
    check(s2.evictions == 1 && s2.entries == 1 && s2.saves == 2 && s2.segments == segs1 + 1,
          "F2 evicted F1, F1's segments kept by F2's chain");
    PrefixEntry * f2 = prefix_cache_lookup(*pc2, session_tokens(a).data(), session_pos(a));
    check(f2 != nullptr, "lookup of F2");
    if (!f2) return false;
    session_reset(b);
    timed_load(b, *pc2, *f2, "F2 into b");
    same_state(dF2, session_state_digest(b), "F2 (evicted sibling's segments) loaded into b");
    x1.clear();
    x2.clear();
    session_generate(a, 12, x1);
    session_generate(b, 12, x2);
    check(x1 == x2, "b after load F2 + generate = a as is [exact]");
    stats(*pc, "end, cache 1");
    return ok;
}
}  // namespace

int main(int argc, char ** argv) {
    std::string gguf = "/mnt/models-cold/glm53-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";
    std::string odir = "/mnt/models/glm-oracle/q4kxl-p32";
    int n_gen = 8;
    int64_t max_ctx = 4096;
    bool zero_experts = false, cpu_prefill = false, routed = false;
    double synth_gb = 0, vram_gb = 0;
    int slots = 0;   // --slots N: fixed VRAM expert slots (RoutedOptions::slots)
    std::string profile;
    int learn = 0, threads = 32;
    int adapt_every = 0, adapt_swaps = 96;   // --adapt-every N [--adapt-swaps S]: adaptive VRAM tier (routed_step)
    float adapt_decay = 0.7f;                // --adapt-decay D: usage decay per adapt step
    int spec = -1;              // --spec K: generation through session.hpp (K MTP drafts per round, 0 = plain)
    std::vector<int> specs;     // --spec K1,K2,..: one session per K, tokens compared with the first K's
    bool mtp_share = true, test_reject = false, serve = false, prefix = false;
    std::string test_fault;   // --test-fault spec|serve: a wrong last token for the K check / serve-test (later K only)
    int prefill_min = 0, prefill_chunk = 8192;   // --prefill N: odd sessions prefill prompts of >= N tokens
    bool prefill_chunk_set = false;              // --prefill-chunk given: no automatic sizing
    bool prefill_exact = false;   // --prefill-exact: sync reclaim + frozen cost model (bit-reproducible prompts)
    bool kv_q8 = false;           // --kv-latent q8: int8 MLA latent caches (trunk, session, MTP)
    int profile_span = 0;         // --profile-span N: profiler range over N rounds of the last K
    bool launch_probe = false;    // --launch-probe: segment launch / acc upload host costs after the decode
    bool pack_types = false;   // --pack-types: pack GGUF trunk types (BF16 list) on the oracle GGUF
    bool serial_shexp = false;   // --serial-shexp: routed decode without the shared expert / misses overlap
    bool arena = false, arena_alias = false, teacher_force = false, vnni = true, down_q8_0 = true, r8 = false;
    double arena_gb = 0;
    std::string save_profile, prompt_file;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) odir = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) max_ctx = std::atoll(argv[++i]);
        else if (a == "--zero-experts") zero_experts = true;
        else if (a == "--cpu-experts-prefill") cpu_prefill = true;
        else if (a == "--routed") routed = true;
        else if (a == "--pack-types") pack_types = true;
        else if (a == "--serial-shexp") serial_shexp = true;
        else if (a == "--teacher-force") teacher_force = true;
        else if (a == "--vnni") vnni = true;
        else if (a == "--f32") vnni = false;   // exact F32 miss kernel (default: VNNI)
        else if (a == "--down-q8-k") down_q8_0 = false;   // VNNI down activations as Q8_K multi (default Q8_0)
        else if (a == "--r8") r8 = true;   // R8 arena (RoutedOptions::r8)
        else if (a == "--arena" && i + 1 < argc) { arena = true; arena_gb = std::atof(argv[++i]); }
        else if (a == "--arena-alias") arena_alias = true;   // past the cap: alias arena blobs (timing only)
        else if (a == "--save-profile" && i + 1 < argc) save_profile = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt_file = argv[++i];
        else if (a == "--spec" && i + 1 < argc) {
            const std::string v = argv[++i];
            for (size_t b = 0; b < v.size();) {
                size_t e = v.find(',', b);
                if (e == std::string::npos) e = v.size();
                specs.push_back(std::atoi(v.substr(b, e - b).c_str()));
                b = e + 1;
            }
            if (!specs.empty()) spec = *std::max_element(specs.begin(), specs.end());
        }
        else if (a == "--no-mtp-share") mtp_share = false;
        else if (a == "--test-reject") test_reject = true;
        else if (a == "--test-fault" && i + 1 < argc) test_fault = argv[++i];
        else if (a == "--serve-test") serve = true;   // with --spec: serve_test after each K
        else if (a == "--prefix-test") prefix = true;   // with --spec: prefix_test after each K (a second session)
        else if (a == "--prefill" && i + 1 < argc) prefill_min = std::atoi(argv[++i]);
        else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]), prefill_chunk_set = true;
        else if (a == "--prefill-exact") prefill_exact = true;
        else if (a == "--kv-latent" && i + 1 < argc) kv_q8 = std::string(argv[++i]) == "q8";   // f16 | q8 MLA latent
        else if (a == "--launch-probe") launch_probe = true;
        else if (a == "--profile-span" && i + 1 < argc) profile_span = std::atoi(argv[++i]);
        else if (a == "--synthetic" && i + 1 < argc) synth_gb = std::atof(argv[++i]);
        else if (a == "--vram-gb" && i + 1 < argc) vram_gb = std::atof(argv[++i]);
        else if (a == "--slots" && i + 1 < argc) slots = std::atoi(argv[++i]);
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--learn" && i + 1 < argc) learn = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--adapt-every" && i + 1 < argc) adapt_every = std::atoi(argv[++i]);
        else if (a == "--adapt-swaps" && i + 1 < argc) adapt_swaps = std::atoi(argv[++i]);
        else if (a == "--adapt-decay" && i + 1 < argc) adapt_decay = (float) std::atof(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    LoadOptions opt;
    if (pack_types) opt.as_bf16 = pack_recipe_bf16();
    opt.load_mtp = spec > 0;
    if (spec >= 0) teacher_force = true;   // the session feeds the prompt through the decode graph
    std::unique_ptr<Model> m;
    try {
        m = load_model(gguf, opt, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "LOAD FAILED: %s\n", e.what());
        return 1;
    }
    const Geometry & g = m->g;
    std::unique_ptr<test::Oracle> oracle_;   // opened on first use: a --prompt run needs no oracle dump
    auto oracle = [&]() -> test::Oracle & {
        if (!oracle_) oracle_ = std::make_unique<test::Oracle>(odir);
        return *oracle_;
    };
    std::vector<int32_t> toks;
    {
        FILE * f = std::fopen((prompt_file.empty() ? odir + "/tokens.txt" : prompt_file).c_str(), "r");
        int v;
        while (f && std::fscanf(f, "%d", &v) == 1) toks.push_back(v);
        if (f) std::fclose(f);
    }
    if (!prompt_file.empty() && (!teacher_force || toks.empty())) {
        std::fprintf(stderr, "--prompt needs --teacher-force and a non-empty token file\n");
        return 2;
    }
    const int T = prompt_file.empty() ? (int) oracle().node(0, "inp_embd").ne[1] : (int) toks.size();
    if (T + n_gen > max_ctx) max_ctx = T + n_gen;
    // teacher-forced runs only use the T = 1 decode graph (no eager prefill)
    const int ws_T = teacher_force ? 1 : T;
    auto ws = workspace_create(*m, ws_T);

    // attention state
    std::vector<KdaState *> kda(g.n_layer, nullptr);
    std::vector<MlaCache *> mla(g.n_layer, nullptr);
    for (int il = 0; il < g.n_layer && spec < 0; ++il) {
        if (m->layers[il].mla) mla[il] = mla_cache_create(g, max_ctx, kv_q8 ? MlaLatent::Q8 : MlaLatent::F16);
        else {
            kda[il] = kda_state_create(g);
            kda_state_reset(kda[il], st);
        }
    }
    void * attn_scratch = nullptr;
    if (spec < 0)
        ck(cudaMalloc(&attn_scratch, std::max(kda_scratch_bytes(g, ws_T), mla_scratch_bytes(g, ws_T, max_ctx))),
           "scratch");
    RoutedPtr rt;
    if (routed) {
        RoutedOptions ro;
        ro.cpu_threads = threads;
        ro.synthetic_pool_bytes = (size_t) (synth_gb * 1e9);
        ro.arena = arena;
        ro.vnni = vnni;
        ro.down_q8_0 = down_q8_0;
        ro.r8 = r8;
        ro.arena_alias = arena_alias;
        ro.arena_max_bytes = (size_t) (arena_gb * 1e9);
        ro.adapt_every = adapt_every;
        ro.adapt_swaps = adapt_swaps;
        ro.adapt_decay = adapt_decay;
        ro.vram_bytes = (size_t) (vram_gb * 1e9);
        ro.slots = slots;   // fixed residency: runs comparable across processes (free VRAM varies by a few slots)
        ro.pin_arena = prefill_min > 0;   // routed_lend (the prefill's loan) needs every expert in the pinned arena
        ro.reserve_bytes = (size_t) 2 << 30;   // graphs, cuBLAS, MLA growth
        if (spec >= 0) {
            ro.max_T = 8;   // verify rows and MTP catch-up chunks
            ro.reserve_bytes = (size_t) 3 << 30;   // + the session's workspace, states and graphs
        }
        try {
            rt = routed_create(*m, ro);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "ROUTED FAILED: %s\n", e.what());
            return 1;
        }
        if (!profile.empty()) {
            if (arena) routed_load(*rt, routed_read_profile(profile), st);
            else routed_fill(*rt, routed_read_profile(profile), st);
        }
    }
    if (spec >= 0) {
        // ---- generation session (session.hpp) --------------------------------------------------------------
        SessionExperts ex;
        if (prefill_min > 0 && (!routed || !arena || spec < 0)) {
            std::fprintf(stderr, "--prefill needs --routed --arena and --spec\n");
            return 2;
        }
        if (prefill_min > 0 && !routed_can_lend(*rt)) {   // as glm_serve: no loan without a pinned arena or bounce ring
            std::printf("prefill off: the host arena is neither pinned nor served by a bounce ring, prompts go through "
                        "the decode graph (the serve test covers that path only)\n");
            prefill_min = 0;
        }
        PrefillMoe * pm = nullptr;
        if (prefill_min > 0) {
            PrefillMoeOptions po;
            po.max_T = prefill_chunk;
            // the serve test's [exact] checks need bit-reproducible prompts: exact mode (the serving default is the
            // fast mode, equal up to near-tie flips)
            if ((serve || prefix) && !prefill_exact) {
                prefill_exact = true;
                std::printf("--serve-test / --prefix-test with --prefill: exact mode (sync reclaim, frozen cost model, no speculative uploads)\n");
            }
            po.async_reclaim = !prefill_exact;
            po.calibrate = !prefill_exact;
            if (prefill_exact) po.spec = 0;
            pm = prefill_moe_create(*m, *rt, po, st);
            if (!prefill_chunk_set) {   // as glm_serve: the loan leaves a quarter of the slots (small VRAM)
                const int cap = prefill_prompt_fit_cap(pm, *m, max_ctx + 16, spec > 0, routed_slots(*rt));
                if (cap == 0) {
                    std::fprintf(stderr, "the prefill's loan does not fit in 3/4 of %d expert slots even for 64-token chunks\n",
                                 routed_slots(*rt));
                    return 1;
                }
                if (cap != prefill_chunk) {
                    std::printf("prefill chunk %d -> %d (its loan must leave 1/4 of %d expert slots)\n", prefill_chunk,
                                cap, routed_slots(*rt));
                    prefill_moe_destroy(pm, st);
                    po.max_T = prefill_chunk = cap;
                    pm = prefill_moe_create(*m, *rt, po, st);
                }
            }
        }
        if (routed) {
            ex.eager = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T_,
                           cudaStream_t s) { routed_run(*rt, il, y, ids, w, out, T_, s); };
            // routed_step: the adaptive tier, and the admission of the prefill's asynchronously reclaimed experts
            if (adapt_every > 0 || pm) ex.step = [&](cudaStream_t s) { routed_step(*rt, s); };
            if (serial_shexp) {
                ex.start = ex.eager;
            } else {
                ex.start = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T_,
                               cudaStream_t s) { routed_start(*rt, il, y, ids, w, out, T_, s); };
                ex.finish = [&](int il, float * out, int T_, cudaStream_t s) { routed_finish(*rt, il, out, T_, s); };
            }
        } else if (zero_experts) {
            ex.eager = [&](int, const float *, const int32_t *, const float *, float * out, int T_, cudaStream_t s) {
                ck(cudaMemsetAsync(out, 0, (size_t) T_ * g.n_embd * 4, s), "zero experts");
            };
            ex.start = ex.eager;
        } else {
            std::fprintf(stderr, "--spec needs --routed or --zero-experts\n");
            return 2;
        }
        // one reference per prompt path (0: decode graph, 1: chunked prefill), its first K: a later K is checked against
        // its own path's reference. The chunked prefill's sums are not T = 1's (KL vs T = 1 is its quality gate,
        // glm_prefill_test), so the two references are compared once, reported, not counted; prefill vs prefill counts
        // only with --prefill-exact (bit-reproducible prompts; --serve-test implies it)
        struct Ref { int si = -1, spec = 0; std::vector<int32_t> gen; std::vector<float> margin, top1; } refs[2];
        bool all_same = true, serve_ok = true, any_same_path = false, any_cross = false, cross_same = true;
        for (size_t si = 0; si < specs.size(); ++si) {
            spec = specs[si];
            if (si > 0 && routed && adapt_every > 0 && !profile.empty()) {
                // every K starts from the seed placement (the adaptive tier moved it during the previous K)
                routed_fill(*rt, routed_read_profile(profile), st);
                routed_set_adapt(*rt, adapt_every, adapt_swaps);
            }
            SessionOptions so;
            so.max_ctx = max_ctx + 16;
            so.spec_k = spec;
            so.mtp_share = mtp_share;
            so.test_reject = test_reject;
            so.record_margins = specs.size() > 1;
            so.mla_latent = kv_q8 ? MlaLatent::Q8 : MlaLatent::F16;
            SessionExperts exs = ex;
            if (pm && (si % 2 == 1 || specs.size() == 1)) {
                exs.prefill = pm;
                so.prefill_min = prefill_min;
            }
            double t0 = now_ms();
            SessionPtr ss;
            try {
                ss = session_create(*m, so, exs, st);
            } catch (const std::exception & e) {
                std::fprintf(stderr, "SESSION FAILED: %s\n", e.what());
                return 1;
            }
            std::printf("session: spec_k %d%s, created in %.1f ms\n", spec, spec > 0 && mtp_share ? " (MTP index share)" : "",
                        now_ms() - t0);
            // arena without a profile: an initial load (default ranking) so the prompt never reads the HDD mmap
            if (si == 0 && routed && arena && profile.empty()) routed_load(*rt, routed_rank_counts(*rt), st);
            t0 = now_ms();
            const int32_t first = session_prompt(*ss, toks.data(), T);
            SessionSnapshotPtr s0;
            if (serve) session_snapshot(*ss, s0);
            const bool prefilled = exs.prefill && T >= prefill_min;
            std::printf("prompt T=%d through the %s: %.1f ms, first token %d%s\n", T,
                        prefilled ? "chunked prefill" : "decode graph", now_ms() - t0, first,
                        (int) toks.size() > T ? (first == toks[T] ? " (= oracle)" : " (oracle differs)") : "");
            if (routed) {
                // --learn N (N > 0): re-place the cache (and arena) from the first prompt's routing counts; later K
                // keep that placement
                if (si == 0 && profile.empty() && learn > 0) {
                    if (arena) routed_load(*rt, routed_rank_counts(*rt), st);
                    else routed_fill(*rt, routed_rank_counts(*rt), st);
                }
                routed_stats(*rt) = RoutedStats{};
            }
            SessionStats & ssn = session_stats(*ss);
            ssn = SessionStats{};
            std::vector<int32_t> gen;
            int round = 0;
            constexpr int kProfileWarm = 4;
            const bool prof = profile_span > 0 && si + 1 == specs.size();
            if (prof && n_gen < kProfileWarm + profile_span)
                std::printf("--profile-span %d: needs --tokens >= %d (rounds may emit several)\n", profile_span,
                            kProfileWarm + profile_span);
            session_generate(*ss, n_gen, gen, [&](const int32_t * tk, int n, double ms) {
                if (prof && round == kProfileWarm - 1) ck(cudaProfilerStart(), "profiler start");
                if (prof && round == kProfileWarm - 1 + profile_span) ck(cudaProfilerStop(), "profiler stop");
                std::printf("  round %d: %d token(s) %.2f ms:", round++, n, ms);
                for (int i = 0; i < n; ++i) std::printf(" %d", tk[i]);
                std::printf("\n");
            });
            int cmp = 0, same = 0;
            for (size_t i = 0; i < gen.size() && T + i < toks.size(); ++i, ++cmp) {
                if (gen[i] != toks[T + i]) break;
                ++same;
            }
            std::printf("generated %lld tokens in %lld rounds, %.2f ms/token = %.1f tok/s", (long long) ssn.generated,
                        (long long) ssn.rounds, ssn.gen_ms / std::max<int64_t>(1, ssn.generated),
                        1000.0 * ssn.generated / std::max(1e-9, ssn.gen_ms));
            if (spec > 0)
                std::printf("; drafts accepted %lld / %lld (%.1f%%), %.2f tokens/round, verify %.2f ms/round, draft %.2f "
                            "ms/round",
                            (long long) ssn.accepted, (long long) ssn.drafted,
                            100.0 * ssn.accepted / std::max<int64_t>(1, ssn.drafted),
                            (double) ssn.generated / std::max<int64_t>(1, ssn.rounds),
                            ssn.verify_ms / std::max<int64_t>(1, ssn.rounds), ssn.draft_ms / std::max<int64_t>(1, ssn.rounds));
            std::printf("\n");
            if (cmp) std::printf("greedy vs oracle tokens: first %d / %d identical\n", same, cmp);
            if (routed) {
                const RoutedStats & rs = routed_stats(*rt);
                std::printf("routed: hit rate %.1f%% (%lld / %lld), misses %.2f ms/token, routing wait %.2f ms/token, "
                            "spilled %lld, aliased misses %lld%s\n",
                            rs.jobs ? 100.0 * rs.hits / rs.jobs : 0.0, (long long) rs.hits, (long long) rs.jobs,
                            rs.miss_ms / std::max<int64_t>(1, ssn.generated), rs.wait_ms / std::max<int64_t>(1, ssn.generated),
                            (long long) rs.spilled, (long long) rs.aliased,
                            rs.aliased ? " (PARTIAL: aliased experts, outputs not real)" : "");
                std::printf("routed: per call %.2f jobs, %.2f distinct experts, %.2f distinct misses; per token %.1f "
                            "distinct misses\n",
                            (double) rs.jobs / std::max<int64_t>(1, rs.calls), (double) rs.distinct / std::max<int64_t>(1, rs.calls),
                            (double) rs.distinct_misses / std::max<int64_t>(1, rs.calls),
                            (double) rs.distinct_misses / std::max<int64_t>(1, ssn.generated));
                if (!save_profile.empty()) {
                    routed_write_profile(save_profile, routed_rank_counts(*rt), (int) m->layers.size(), g.n_expert);
                    std::printf("routed: profile written to %s\n", save_profile.c_str());
                }
            }
            // ---- same tokens as the path's first K (greedy speculative decoding is exact up to batch-size rounding) --
            const std::vector<float> mg = session_margins(*ss);
            const std::vector<float> & t1 = session_top1(*ss);
            const char * path = prefilled ? "chunked prefill" : "decode graph";
            const auto compare = [&](const Ref & r, const std::vector<int32_t> & kg, const char * what) {
                size_t d = 0;
                while (d < kg.size() && d < r.gen.size() && kg[d] == r.gen[d]) ++d;
                const bool same = d == kg.size() && kg.size() == r.gen.size();
                {   // top-1 logit drift on the common prefix (same context tokens, different batch shapes)
                    std::vector<float> dl;
                    for (size_t i = 0; i + 1 < d && i < t1.size() && i < r.top1.size(); ++i)
                        dl.push_back(std::fabs(t1[i] - r.top1[i]));
                    if (!dl.empty()) {
                        const size_t imax = std::max_element(dl.begin(), dl.end()) - dl.begin();
                        std::vector<float> sd(dl);
                        std::sort(sd.begin(), sd.end());
                        std::printf("spec %d vs spec %d (%s): |top-1 logit difference| over %zu common rows: median "
                                    "%.4f, 90%% %.4f, max %.4f (generated token %zu)\n", spec, r.spec, what, dl.size(),
                                    sd[sd.size() / 2], sd[sd.size() * 9 / 10], dl[imax], imax + 1);
                    }
                }
                if (same) std::printf("spec %d vs spec %d (%s): %zu / %zu tokens identical\n", spec, r.spec, what, d,
                                      kg.size());
                else std::printf("spec %d vs spec %d (%s): FIRST DIFFERENCE at generated token %zu (%d vs %d), top-1 - "
                                 "top-2 logit gap there %.4f (spec %d) / %.4f (spec %d)\n", spec, r.spec, what, d,
                                 d < kg.size() ? kg[d] : -1, d < r.gen.size() ? r.gen[d] : -1,
                                 d >= 1 && d - 1 < mg.size() ? mg[d - 1] : -1.f, spec,
                                 d >= 1 && d - 1 < r.margin.size() ? r.margin[d - 1] : -1.f, r.spec);
                return same;
            };
            Ref & own = refs[prefilled ? 1 : 0];
            const Ref & other = refs[prefilled ? 0 : 1];
            if (own.si < 0) {
                own.si = (int) si;
                own.spec = spec;
                own.gen = gen;
                own.margin = mg;
                own.top1 = t1;
                std::vector<float> sorted(mg);
                std::sort(sorted.begin(), sorted.end());
                if (!sorted.empty())
                    std::printf("spec %d (%s reference): top-1 - top-2 logit gap: min %.4f, 1%% %.4f, median %.4f\n",
                                spec, path, sorted[0], sorted[sorted.size() / 100], sorted[sorted.size() / 2]);
                if (other.si >= 0) {
                    any_cross = true;
                    cross_same = compare(other, gen, prefilled ? "prefill vs decode graph, reported only"
                                                               : "decode graph vs prefill, reported only") &&
                                 cross_same;
                }
            } else {
                std::vector<int32_t> kg(gen);
                if ((test_fault == "spec" || test_fault == "spec@" + std::to_string(si)) && !kg.empty()) {
                    ++kg.back();
                    std::printf("--test-fault spec: spec %d's last token changed for the K check\n", spec);
                }
                const bool counted = !prefilled || prefill_exact;
                const bool same = compare(own, kg, !prefilled ? "decode graph"
                                                   : counted   ? "chunked prefill, exact"
                                                               : "chunked prefill without --prefill-exact, reported only");
                if (counted) {
                    any_same_path = true;
                    all_same = all_same && same;
                }
            }
            if (serve) {
                bool sok = false;
                std::vector<int32_t> sg(gen);
                if (test_fault == "serve" && si > 0 && !sg.empty()) {
                    ++sg.back();
                    std::printf("--test-fault serve: spec %d's last token changed for serve-test\n", spec);
                }
                try {
                    sok = serve_test(*ss, std::vector<int32_t>(toks.begin(), toks.begin() + T), sg, *s0);
                } catch (const std::exception & e) {
                    std::printf("  serve: exception: %s\n", e.what());
                }
                std::printf("spec %d: SERVE %s\n", spec, sok ? "OK" : "FAIL");
                serve_ok = serve_ok && sok;
            }
            if (prefix) {
                bool pok = false;
                try {
                    SessionPtr sb = session_create(*m, so, exs, st);
                    pok = prefix_test(*ss, *sb, std::vector<int32_t>(toks.begin(), toks.begin() + T), gen);
                } catch (const std::exception & e) {
                    std::printf("  prefix: exception: %s\n", e.what());
                }
                std::printf("spec %d: PREFIX %s\n", spec, pok ? "OK" : "FAIL");
                serve_ok = serve_ok && pok;
            }
        }
        if (any_same_path) std::printf("%s\n", all_same ? "SPEC MATCH" : "SPEC MISMATCH");
        if (any_cross)
            std::printf("%s\n", cross_same ? "PREFILL VS DECODE GRAPH: same tokens"
                                           : "PREFILL VS DECODE GRAPH: tokens differ (reported only)");
        prefill_moe_destroy(pm, st);
        return all_same && serve_ok ? 0 : 1;
    }

    int64_t * d_pos0 = nullptr;
    int64_t * h_pos0 = nullptr;
    ck(cudaMalloc(&d_pos0, sizeof(int64_t)), "d_pos0");
    ck(cudaMallocHost(&h_pos0, sizeof(int64_t)), "h_pos0");

    int phase = 0;
    double expert_ms = 0;
    ForwardHooks hk;
    hk.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * out, const Batch & b, cudaStream_t s) {
        const LayerWeights & L = m->layers[il];
        if (L.mla) mla_forward(g, L.mla_w, mla[il], b, y, out, attn_scratch, s, nullptr, y_in);
        else kda_forward(g, L.kda, kda[il], b, y, out, attn_scratch, s, y_in);
    };
    hk.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T_,
                     cudaStream_t s) {
        if (zero_experts || (routed && phase == 0)) {
            ck(cudaMemsetAsync(out, 0, (size_t) T_ * g.n_embd * 4, s), "zero experts");
            return;
        }
        if (routed) {
            routed_run(*rt, il, y, ids, w, out, T_, s);
            return;
        }
        if (phase == 0 && !cpu_prefill) {
            const std::vector<float> v = oracle().load(0, "ffn_moe_out-" + std::to_string(il));
            ck(cudaMemcpyAsync(out, v.data(), v.size() * 4, cudaMemcpyHostToDevice, s), "oracle experts");
            ck(cudaStreamSynchronize(s), "oracle experts");
            return;
        }
        const double t0 = now_ms();
        const int K = g.n_expert_used;
        std::vector<float> hx((size_t) T_ * g.n_embd), hw((size_t) T_ * K), ho(hx.size());
        std::vector<int32_t> hid((size_t) T_ * K);
        ck(cudaStreamSynchronize(s), "experts sync");
        ck(cudaMemcpy(hx.data(), y, hx.size() * 4, cudaMemcpyDeviceToHost), "x");
        ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ids");
        ck(cudaMemcpy(hw.data(), w, hw.size() * 4, cudaMemcpyDeviceToHost), "w");
        test::ref_experts_cpu(g, m->layers[il].experts, hx.data(), hid.data(), hw.data(), T_, ho.data());
        ck(cudaMemcpyAsync(out, ho.data(), ho.size() * 4, cudaMemcpyHostToDevice, s), "out");
        ck(cudaStreamSynchronize(s), "out");
        expert_ms += now_ms() - t0;
    };

    // ---- prefill (eager) ---------------------------------------------------------------------------------
    double t0 = now_ms();
    int32_t tok = -1;
    if (!teacher_force) {
        forward_trunk(*m, *ws, toks.data(), T, 0, g.n_layer, hk, st);
        ck(cudaMemcpyAsync(&tok, ws->tok, 4, cudaMemcpyDeviceToHost, st), "tok");
        ck(cudaStreamSynchronize(st), "prefill");
        std::printf("prefill T=%d: %.1f ms (eager, experts %s), next token %d (oracle %d) %s\n", T, now_ms() - t0,
                    zero_experts || routed ? "zero" : (cpu_prefill ? "cpu ref" : "oracle"), tok, toks[T],
                    tok == toks[T] ? "MATCH" : "DIFFERENT");
    }

    // ---- capture: two graph sets (KDA double buffer parity) ----------------------------------------------
    ForwardHooks gh;
    gh.attn_in = [&](int il, const float * y, const MatmulInput * y_in, float * out, const Batch & b, cudaStream_t s) {
        const LayerWeights & L = m->layers[il];
        if (L.mla) mla_forward_dev(g, L.mla_w, mla[il], b.T, d_pos0, y, out, attn_scratch, s, nullptr, y_in);
        else kda_forward(g, L.kda, kda[il], b, y, out, attn_scratch, s, y_in);
    };
    gh.experts = hk.experts;
    if (routed && !serial_shexp) {
        // the hits are captured with the segment, the misses run between segments while the GPU does the shexp
        gh.experts = [&](int il, const float * y, const int32_t * ids, const float * w, float * out, int T_,
                         cudaStream_t s) { routed_start(*rt, il, y, ids, w, out, T_, s); };
        gh.experts_finish = [&](int il, float * out, int T_, cudaStream_t s) { routed_finish(*rt, il, out, T_, s); };
    }
    t0 = now_ms();
    // sg[c] reads the KDA buffer c; each capture flips the host cur, so back to back covers both parities
    KdaState * k0 = nullptr;
    for (auto * k : kda) if (k && !k0) k0 = k;
    std::unique_ptr<SegmentedGraph> sg[2];
    for (int p = 0; p < 2; ++p) {
        const int c = kda_state_cur(k0);
        sg[c] = capture_decode(*m, *ws, 1, gh, st);
    }
    std::printf("captured 2 x %zu segments, %d nodes per step, %.1f ms\n", sg[0]->segs.size(), sg[0]->nodes,
                now_ms() - t0);

    // ---- greedy decode -----------------------------------------------------------------------------------
    phase = 1;
    int64_t pos = T;
    const bool tf = teacher_force;
    if (tf) {
        // restart from position 0: the prompt goes through the decode graph with the routed experts
        for (auto * k : kda)
            if (k) kda_state_reset(k, st);
        for (auto * c : mla)
            if (c) mla_cache_set_size(c, 0);
        pos = 0;
        tok = toks[0];
    }
    std::vector<int32_t> out_toks;
    std::vector<double> step_ms;
    const int n_learn = routed && profile.empty() ? learn : 0;
    const int n_steps = tf ? std::max(T - 1 + n_gen, n_learn + 1) : n_learn + n_gen;
    int greedy = 0, greedy_match = 0;
    for (int i = 0; i < n_steps; ++i) {
        if (routed && i == n_learn) {
            if (n_learn) {
                if (arena) routed_load(*rt, routed_rank_counts(*rt), st);
                else routed_fill(*rt, routed_rank_counts(*rt), st);
            } else if (arena && profile.empty()) {
                routed_load(*rt, routed_rank_counts(*rt), st);
            }
            routed_stats(*rt) = RoutedStats{};
            out_toks.clear();
            step_ms.clear();
        }
        const double s0 = now_ms();
        expert_ms = 0;
        embed_upload(*m, *ws, &tok, 1, st);
        *h_pos0 = pos;
        ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
        replay_decode(*sg[kda_state_cur(k0)], *ws, gh, st);
        for (auto * k : kda)
            if (k) kda_state_advance(k, 1);
        int32_t out = -1;
        ck(cudaMemcpyAsync(&out, ws->tok, 4, cudaMemcpyDeviceToHost, st), "tok");
        ck(cudaStreamSynchronize(st), "step");
        for (auto * c : mla)
            if (c) mla_cache_set_size(c, pos + 1);
        const bool forced = tf && pos + 1 < T;
        const char * tag = forced ? " [prompt]" : "";
        if (tf && !forced && pos + 1 < (int64_t) toks.size()) {
            ++greedy;
            greedy_match += out == toks[pos + 1];
            tag = out == toks[pos + 1] ? " [= oracle]" : " [oracle differs]";
        }
        tok = forced ? toks[pos + 1] : out;
        ++pos;
        out_toks.push_back(out);
        step_ms.push_back(now_ms() - s0);
        if (routed) {
            static RoutedStats prev;
            const RoutedStats & rs = routed_stats(*rt);
            if (i == n_learn) prev = RoutedStats{};
            std::printf("  step %d: token %d  %.2f ms (hits %lld / %lld, host %.2f ms, misses %.2f ms%s)%s%s\n", i,
                        out, step_ms.back(), (long long) (rs.hits - prev.hits), (long long) (rs.jobs - prev.jobs),
                        rs.host_ms - prev.host_ms, rs.miss_ms - prev.miss_ms,
                        (rs.aliased > prev.aliased ? ", " + std::to_string(rs.aliased - prev.aliased) + " aliased"
                         : rs.spilled > prev.spilled ? ", " + std::to_string(rs.spilled - prev.spilled) + " spilled"
                                                     : std::string()).c_str(),
                        i < n_learn ? " [learn]" : "", tag);
            prev = rs;
        } else {
            std::printf("  step %d: token %d  %.2f ms (experts %.2f ms)\n", i, out, step_ms.back(), expert_ms);
        }
    }
    if (routed) {
        const RoutedStats & rs = routed_stats(*rt);
        const int timed = std::max(1, n_steps - n_learn);
        std::printf("routed: hit rate %.1f%% (%lld / %lld), host %.2f ms/step, misses %.2f ms/step, routing wait "
                    "%.2f ms/step, spilled %lld, aliased misses %lld%s\n",
                    rs.jobs ? 100.0 * rs.hits / rs.jobs : 0.0, (long long) rs.hits, (long long) rs.jobs,
                    rs.host_ms / timed, rs.miss_ms / timed, rs.wait_ms / timed, (long long) rs.spilled,
                    (long long) rs.aliased,
                    rs.aliased ? " (PARTIAL: aliased experts, outputs not real)" : "");
        if (!save_profile.empty()) {
            routed_write_profile(save_profile, routed_rank_counts(*rt), (int) m->layers.size(), g.n_expert);
            std::printf("routed: profile written to %s\n", save_profile.c_str());
        }
    }
    if (tf) std::printf("greedy vs oracle (Q4_K_XL llama.cpp): %d / %d MATCH\n", greedy_match, greedy);
    if (!out_toks.empty() && !routed && !tf)
        std::printf("first decoded token %d (oracle %d) %s\n", out_toks[0], toks[T + 1],
                    out_toks[0] == toks[T + 1] ? "MATCH" : "DIFFERENT");
    if (step_ms.size() > 2) {
        std::vector<double> v(step_ms.begin() + 1, step_ms.end());
        std::sort(v.begin(), v.end());
        std::printf("decode step median %.2f ms (%.1f tok/s), min %.2f ms\n", v[v.size() / 2], 1000.0 / v[v.size() / 2],
                    v[0]);
    }
    if (launch_probe) {
        // The decode's critical path between layers: the host finishes the misses, uploads acc, launches the add and
        // the next segment into an idle stream. Replays the segments (the caches are not used afterwards).
        const SegmentedGraph & sgp = *sg[kda_state_cur(k0)];
        const int S = (int) sgp.segs.size(), reps = 20;
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "probe event");
        ck(cudaEventCreate(&e1), "probe event");
        void * h16 = nullptr;
        void * d16 = nullptr;
        ck(cudaMallocHost(&h16, (size_t) g.n_embd * 4), "probe host");
        ck(cudaMalloc(&d16, (size_t) g.n_embd * 4), "probe dev");
        void * dbusy = nullptr;
        const size_t busy_bytes = (size_t) 256 << 20;   // ~0.3 ms of memset: the stream is busy when the step is queued
        ck(cudaMalloc(&dbusy, busy_bytes), "probe busy");
        std::vector<double> api, cp, idle, queued;
        for (int r = 0; r < reps; ++r)
            for (int i = 0; i < S; ++i) {
                // the host's sequence after a layer's misses: acc H2D, the add (a 16 KB memset stands in), the next
                // segment; into an idle stream (as in the decode) and queued behind busy work (pure GPU time)
                for (int q = 0; q < 2; ++q) {
                    ck(cudaStreamSynchronize(st), "probe");
                    if (q) ck(cudaMemsetAsync(dbusy, 0, busy_bytes, st), "probe busy");
                    ck(cudaEventRecord(e0, st), "probe e0");
                    const double a0 = now_ms();
                    ck(cudaMemcpyAsync(d16, h16, (size_t) g.n_embd * 4, cudaMemcpyHostToDevice, st), "probe h2d");
                    const double a1 = now_ms();
                    ck(cudaMemsetAsync(d16, 0, (size_t) g.n_embd * 4, st), "probe add");
                    const double t0 = now_ms();
                    ck(cudaGraphLaunch(sgp.segs[i], st), "probe launch");
                    const double t1 = now_ms();
                    ck(cudaEventRecord(e1, st), "probe e1");
                    ck(cudaEventSynchronize(e1), "probe sync");
                    float gms = 0;
                    ck(cudaEventElapsedTime(&gms, e0, e1), "probe elapsed");
                    if (q) {
                        queued.push_back(gms);
                    } else {
                        idle.push_back(gms);
                        cp.push_back(a1 - a0);
                        api.push_back(t1 - t0);
                    }
                }
            }
        std::vector<double> over(idle.size());
        for (size_t i = 0; i < idle.size(); ++i) over[i] = idle[i] - queued[i];
        cudaFree(dbusy);
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        auto mean = [](const std::vector<double> & v) { double x = 0; for (double y : v) x += y; return x / v.size(); };
        std::printf("launch probe (%d segments x %d): cudaGraphLaunch API median %.1f us, acc H2D API median %.1f us; "
                    "idle-stream step minus queued step (host critical path per layer) median %.1f us mean %.1f us; "
                    "per step (%d layers): %.2f ms\n", S, reps, 1e3 * med(api), 1e3 * med(cp), 1e3 * med(over),
                    1e3 * mean(over), S, S * mean(over));
        cudaFreeHost(h16);
        cudaFree(d16);
    }
    for (auto * k : kda) if (k) kda_state_destroy(k);
    for (auto * c : mla) if (c) mla_cache_destroy(c);
    return 0;
}
