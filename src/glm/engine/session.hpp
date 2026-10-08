#pragma once
// Generation session: one conversation on one GPU. Owns the attention state (KDA states, MLA caches), the captured
// decode graphs and, with spec_k > 0, the MTP drafter (block 45) for greedy speculative decoding: each round drafts
// k tokens, verifies [last, d1..dk] in one T = k + 1 trunk step (all-rows head), keeps the longest agreeing prefix
// plus the trunk's own next token, rolls the KDA states back (kda_state_rollback) and trims the MLA caches.
// Greedy speculative decoding emits the same tokens as plain greedy decoding up to near-tie flips (the contract of
// llama.cpp's and Strata Qwen's spec decoding; no batch-invariant kernels). The verify rows (T = k + 1) take other
// kernel paths than T = 1 (GEMM vs GEMV, MLA split sizes, KDA chunked vs recurrent): measured on the pack GGUF with
// zero experts, the top-1 logit of a verify row differs from the T = 1 one by median 0.02, max 0.2 (128 tokens, no
// flip); with real experts a routing near-tie can amplify that. The CPU misses are per-row exact (each token sums
// its experts in slot order). Measured (agentic prompt, 128 tokens): plain vs plain identical; k = 1 and k = 3 first
// differ at the token whose top-1 - top-2 gap is the smallest of the run (0.03, median 3.8).
// glm_decode --spec 0,0,1,3 checks this and reports the gaps and the drift.
//
// Prompt tokens go through the chunked prefill (SessionExperts::prefill, prompts of >= prefill_min tokens) or the
// T = 1 decode graph one by one, with the MTP cache caught up per prefill chunk or in chunks of up to 8 rows.
// Positions follow PR #27917: MTP row p = (token p, trunk h[p - 1]).
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "strata/glm/mla.hpp"

#include <cuda_runtime.h>

#include "forward.hpp"
#include "model.hpp"
#include "prefix_cache.hpp"

namespace strata::glm {

struct PrefillMoe;

struct SessionOptions {
    int64_t max_ctx = 4096;
    int     spec_k = 0;        // MTP drafts per round (0 = plain greedy); spec_k + 1 <= kKdaLogT
    bool    mtp_share = true;  // MTP index share (the GGUF's index_share_mtp, PR default)
    bool    test_reject = false;   // tests: round r corrupts draft r % (k + 1) (if < k), forcing partial accepts
    bool    record_margins = false;   // tests: keep the top-1 - top-2 logit gap of every token the trunk decides
    int     prefill_min = 64;  // with SessionExperts::prefill: shortest prompt sent through it
    MlaLatent mla_latent = MlaLatent::F16;   // MLA caches (trunk and MTP): Q8 halves the latent (long contexts)
};

using ExpertsFn = std::function<void(int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                                     cudaStream_t)>;
using ExpertsFinishFn = std::function<void(int il, float * out, int T, cudaStream_t)>;

// Routed experts: start (+ optional finish) are used inside the captured decode graphs exactly like
// ForwardHooks::experts / experts_finish; eager runs the whole thing outside graphs (the MTP block).
struct SessionExperts {
    ExpertsFn start;
    ExpertsFinishFn finish;   // optional
    ExpertsFn eager;
    std::function<void(cudaStream_t)> step;   // optional, between decode steps with the stream idle (routed_step)
    // optional: prompts of >= SessionOptions::prefill_min tokens go through the chunked large-T prefill
    // (prefill_prompt: hybrid routed experts on a routed_lend loan, MTP catch-up per chunk). The prefill reclaims its
    // loan asynchronously: step must then be set (routed_step admits the reclaimed experts back).
    PrefillMoe * prefill = nullptr;
};

struct SessionStats {
    int64_t rounds = 0, drafted = 0, accepted = 0, generated = 0, prompt = 0;
    double  prompt_ms = 0, gen_ms = 0, draft_ms = 0, verify_ms = 0;
};

struct Session;
void session_destroy(Session *);
struct SessionDeleter { void operator()(Session * s) const { session_destroy(s); } };
using SessionPtr = std::unique_ptr<Session, SessionDeleter>;

// spec_k > 0 needs the model loaded with its MTP block (LoadOptions::load_mtp). Captures the graphs (2 per T).
SessionPtr session_create(const Model & m, const SessionOptions & opt, const SessionExperts & ex, cudaStream_t st);

// Conversation state: the caches hold session_tokens() (every prompt token and every emitted token, all fed);
// the pending token (the trunk's greedy next, returned by session_prompt) is neither emitted nor fed yet.
// session_generate emits it first. Tokens a round verified past a stop id or max_new are not kept (the KDA states,
// MLA and MTP caches are cut back to the emitted ones), so the next session_prompt continues right after them.

// Feeds prompt tokens (appends to the conversation; replaces the pending token). Returns the new pending token.
// n = 0 is a no-op returning the current pending token (-1 after create / reset). Throws before feeding anything
// if pos + n > max_ctx. on_progress (optional) sees (tokens fed, n) after each token (after each chunk on the
// prefill path). on_next (optional) gets the new pending token as soon as the trunk has decided it: on the prefill
// path before the MTP catch-up of the last chunk and the first drafts (a server streams it early; session_generate
// still emits it first).
int32_t session_prompt(Session &, const int32_t * ids, int n,
                       const std::function<void(int done, int n)> & on_progress = {},
                       const std::function<void(int32_t next)> & on_next = {});

enum class FinishReason { stop, length, cancel, ctx };
const char * finish_reason_name(FinishReason);

struct GenerateControl {
    std::vector<int32_t> stop_ids;              // emitting one ends the generation (it is emitted and fed)
    const std::atomic<bool> * cancel = nullptr; // polled between rounds
};

// Generates up to max_new tokens greedily (one round may emit up to spec_k + 1), appending them to out; on_round
// (optional) sees each round's emitted tokens (already cut at a stop id / max_new). Needs a pending token (a
// session_prompt or a session_restore before). Returns why it stopped: stop id, max_new, cancel, or ctx (the
// next round does not fit in max_ctx).
FinishReason session_generate(Session &, int max_new, std::vector<int32_t> & out,
                              const std::function<void(const int32_t * toks, int n, double ms)> & on_round = {},
                              const GenerateControl & ctl = {});

// Prefix reuse. A snapshot (pinned host, session_snapshot_bytes) holds the position, the tokens, the pending token
// and every KDA layer's state; the MLA and MTP caches are not copied, so session_restore only goes back to a
// snapshot whose tokens are a prefix of the session's current tokens (it throws otherwise): it cuts the caches
// back, reloads the KDA states and makes the snapshot's pending token pending again. Then session_prompt feeds
// the new suffix, or session_generate continues directly. session_reset empties the conversation (no graph
// recapture).
struct SessionSnapshot;
void session_snapshot_destroy(SessionSnapshot *);
struct SessionSnapshotDeleter { void operator()(SessionSnapshot * p) const { session_snapshot_destroy(p); } };
using SessionSnapshotPtr = std::unique_ptr<SessionSnapshot, SessionSnapshotDeleter>;

size_t session_snapshot_bytes(const Session &);
// Captures the current state into snap (allocated when null, else reused). Synchronizes the stream.
void session_snapshot(Session &, SessionSnapshotPtr & snap);
int64_t                      session_snapshot_pos(const SessionSnapshot &);
const std::vector<int32_t> & session_snapshot_tokens(const SessionSnapshot &);
void session_restore(Session &, const SessionSnapshot &);
// whether session_restore would accept the snapshot: its tokens are a prefix of the session's, and the MLA caches can
// cut there (after a session_load, a cut inside the loaded rows at a position that is not a multiple of 4 cannot:
// mla_cache_can_truncate)
bool session_restorable(const Session &, const SessionSnapshot &);
void session_reset(Session &);

// Host prefix cache (prefix_cache.hpp). session_save queues a copy of the current state (position, tokens, pending
// token) on the cache's stream and returns at once: the MLA rows past the session's own chain (the rows it loaded or
// saved last, shared), the tails, the KDA states, hcat row 0. The session's next call waits for those copies on its
// stream (no host wait). Nothing is saved at position 0, when an entry with the same tokens exists (it is marked
// used), or when the cache cannot make room (stats save_skips).
void session_save(Session &, PrefixCache &);
// Makes the entry's state the session's (any session of the same model and options; the entry stays in the cache).
// Keeps the session's rows below the deepest segment boundary it shares with the entry's chain (same tokens, same
// blobs: bitwise the entry's rows) and imports the rest. Synchronizes the stream. Returns the rows imported.
// Throws if the entry does not fit max_ctx or was saved from other cache types (the session is then reset).
int64_t session_load(Session &, PrefixCache &, PrefixEntry &);
// Tests / gates: 64-bit digests of the state a session_save stores, by part (synchronizes; reads ~7 KB per position
// plus the KDA states): per MLA cache (trunk layers, then "mtp") the latent rows [0, pos) (+ Q8 scales), the pooled
// rows of the complete pools and the key | gate tail; per KDA layer the state; hcat row 0; position, tokens and
// pending token. A load reproduces the saved digests exactly.
std::vector<std::pair<std::string, uint64_t>> session_state_digest(Session &);

int64_t        session_pos(const Session &);
const std::vector<int32_t> & session_tokens(const Session &);
int32_t        session_pending(const Session &);
// record_margins: one entry per generated token after the first (the trunk row that decided it), in order.
const std::vector<float> & session_margins(const Session &);
const std::vector<float> & session_top1(const Session &);   // the top-1 logit of the same rows
SessionStats & session_stats(Session &);

}  // namespace strata::glm
