#pragma once
// GLM-5.3-Flash MTP drafter: the NextN block (blk.45) run as a draft head for speculative decoding.
// Math follows llama.cpp PR #27917 (glm5-next graph_mtp), docs/glm5next-spec.md section 5:
//
//   e = token_embd(token), h = hidden state of the previous position (trunk h_nextn = output_norm(stream mean),
//       or this block's own h_nextn while drafting)
//   u      = eh_proj(concat(RMSNorm(e) * enorm, RMSNorm(h) * hnorm))          [8192 -> 4096]
//   f_in   = u + mla_forward(RMSNorm(u) * attn_norm)                          own MLA + DSA cache (block 45)
//   f      = RMSNorm(f_in) * ffn_norm
//   h_next = RMSNorm(routed(f) + shared(f) + f_in) * shared_head_norm         (this block's h_nextn)
//   logits = output * h_next                                                  (shared head)
//
// Positions: the MTP cache row of position p is computed from token p and the hidden state of position p - 1, so
// the drafter predicts token p + 1 from (token p, h[p - 1]). The cache is one sequence, like the trunk's.
//
// Speculative loop (the engine owns it):
//   prefill    mtp_draft(ids = prompt, h_prev = [0, h[0], ..., h[N-2]], pos0 = 0, k = 0)
//   each round mtp_draft(ids = accepted tokens + newest token, h_prev = their trunk rows, pos0, k)
//              -> k greedy drafts; rows before the last refresh the cache with the trunk's h, the last row starts
//                 the chain. Draft steps write cache rows pos0 + T .. pos0 + T + k - 2, which the next round
//                 overwrites (mla_forward truncates implicitly at its pos0).
// h_prev rows are the trunk's post-output_norm hidden states (one per verified token).
#include <cstddef>
#include <cstdint>

#include "strata/glm/mla.hpp"
#include "strata/glm/types.hpp"

namespace strata::glm {

struct Model;
struct Workspace;
struct ForwardHooks;
struct MtpState;

// max_T: largest catch-up batch (prompt chunk or accepted window + 1). max_ctx: MTP cache capacity in tokens.
// latent: the MTP block's MLA cache type (as the trunk's).
MtpState * mtp_create(const Model & m, int64_t max_ctx, int max_T, MlaLatent latent = MlaLatent::F16);
void       mtp_destroy(MtpState * s);
void       mtp_truncate(MtpState * s, int64_t n_tokens);   // keep MTP cache positions [0, n_tokens)
int64_t    mtp_size(const MtpState * s);

// Index-share mode (GGUF index_share_mtp, PR #27917): draft steps after the first reuse the first step's DSA
// selection instead of recomputing it. Off: every step runs the block-45 indexer (exact causal model).
void mtp_set_share_selection(MtpState * s, bool on);

// Draft head restricted to a vocabulary subset (ids: n distinct token ids): the output rows of ids are gathered
// once into a smaller head, and every draft is the argmax over the subset. A cheaper head when the drafts that
// matter (the ones the trunk accepts) almost always fall in the subset. n = 0 restores the full head.
void mtp_set_draft_vocab(MtpState * s, const Model & m, const int32_t * ids, int n, cudaStream_t st);

// Runs the T rows ids[0..T) (host) at positions [pos0, pos0 + T) with h_prev (device [T][4096], row t = hidden
// state of position pos0 + t - 1, zeros for position 0), then drafts k tokens greedily from the last row into
// drafts[0..k) (host). k = 0: cache refresh only (no head). Uses ws (the trunk workspace, sized >= max(T, 1)) and
// hk.experts for the routed experts of block 45; hk.tap sees the nodes (il = n_layer, PR #27917 names).
// Synchronizes the stream once per drafted token (host embedding of the draft).
void mtp_draft(MtpState * s, const Model & m, Workspace & ws, const ForwardHooks & hk, const int32_t * ids,
               const float * h_prev, int T, int64_t pos0, int k, int32_t * drafts, cudaStream_t st);

// Cache refresh of a large prefill chunk (k = 0, any T <= ws.max_T, not bound by mtp_create's max_T), right after
// forward_trunk / forward_dev ran the chunk's T tokens at [pos0, pos0 + T) through every trunk block on ws: the
// trunk's post-output_norm rows are taken from ws.x and the token embeddings from ws.e (no host staging). hrows:
// device [T + 1][4096]; row 0 = hidden state of position pos0 - 1 (zeros for pos0 = 0), set by the caller for the
// first chunk; rows 1..T are written, and on return row 0 holds the chunk's last row, ready for the next chunk (and
// for the first draft round after the prompt). ws.x / ws.x2 are reused as block-45 buffers. scratch: MLA scratch of
// >= mla_scratch_bytes(g, T, max_ctx) bytes (e.g. the prefill attention scratch). hk.experts: the prefill hooks.
void mtp_catch_up(MtpState * s, const Model & m, Workspace & ws, const ForwardHooks & hk, float * hrows, int T,
                  int64_t pos0, void * scratch, size_t scratch_bytes, cudaStream_t st);

// Device [4096] h_nextn of the last row of the last mtp_draft / mtp_catch_up forward (the chain input of the next step).
const float * mtp_last_hidden(const MtpState * s);

struct MlaCache;
MlaCache * mtp_cache(MtpState * s);   // the block-45 MLA + DSA cache (tests, benchmarks)

}  // namespace strata::glm
