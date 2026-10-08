#pragma once
// A whole prompt through the large-T path: one prefill_moe loan (begin / end), chunks of at most the loan's cap
// tokens through forward_trunk with the hybrid routed experts and the large-T attention (kda_forward / mla_forward),
// and with an MTP state the block-45 cache refreshed per chunk (mtp_catch_up). The engine's session calls this for
// long prompts; afterwards the T = 1 graphs continue from the attention state as after a token-by-token prompt.
#include <cstdint>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "model.hpp"
#include "prefill_moe.hpp"

namespace strata::glm {

struct KdaState;
struct MlaCache;
struct MtpState;

struct PrefillPrompt {
    const std::vector<KdaState *> * kda = nullptr;   // per trunk layer (null for MLA layers)
    const std::vector<MlaCache *> * mla = nullptr;   // per trunk layer (null for KDA layers)
    int64_t max_ctx = 0;                             // MLA cache capacity (sizes the attention scratch)
    MtpState * mtp = nullptr;                        // optional: refresh its cache over the prompt
    // with mtp: device [4096], in = trunk h of position pos0 - 1 (zeros for pos0 = 0), out = trunk h of the prompt's
    // last position (the h_prev of the first draft round)
    float * h_prev = nullptr;
    float * logits = nullptr;                        // optional host [n_vocab]: the last token's logits
    std::function<void(int done, int n)> on_chunk;   // optional, after each chunk (synchronized)
    // optional: the trunk's greedy next token, as soon as the last chunk's trunk has run (before its MTP catch-up and
    // the loan's end): a server can stream it while the rest finishes
    std::function<void(int32_t next)> on_next;
};

// Feeds ids[0..n) at positions [pos0, pos0 + n) (the attention caches must hold exactly pos0 tokens; with an MTP
// state its cache too). Returns the trunk's greedy next token. Synchronous.
int32_t prefill_prompt(PrefillMoe * p, const Model & m, const PrefillPrompt & pp, const int32_t * ids, int n,
                       int64_t pos0, cudaStream_t st);

// The loan prefill_prompt takes for n tokens (cap = min(n, max_T)): prefill_moe_loan_bytes with the attention scratch
// for the cap and, with an MTP state, its h rows.
PrefillLoanBytes prefill_prompt_loan_bytes(const PrefillMoe * p, const Model & m, int64_t n, int64_t max_ctx, bool mtp);
// The chunk to create the PrefillMoe with when none is given: max_T halved until that loan fits in 3/4 of the routed
// slots (a loan evicts the experts it covers: a quarter of the slots stays resident). max_T if it fits; 0 if not even
// 64 tokens do (too few slots for this context: no large-T prefill, prompts through the decode graph).
int prefill_prompt_fit_cap(const PrefillMoe * p, const Model & m, int64_t max_ctx, bool mtp, int slots);

}  // namespace strata::glm
