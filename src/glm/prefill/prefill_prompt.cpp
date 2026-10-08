// src/glm/prefill/prefill_prompt.cpp - see prefill_prompt.hpp.
#include "prefill_prompt.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "forward.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/mtp.hpp"

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_prompt: ") + what + ": " + cudaGetErrorString(e));
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

// the loan's caller scratch: the attention scratch for the chunk cap, then the MTP h rows
size_t attn_scratch(const Geometry & g, int cap, int64_t max_ctx) {
    return align256(std::max(kda_scratch_bytes(g, cap), mla_scratch_bytes(g, cap, max_ctx)));
}
size_t mtp_rows(const Geometry & g, int cap, bool mtp) { return mtp ? (size_t) (cap + 1) * g.n_embd * 4 : 0; }

}  // namespace

PrefillLoanBytes prefill_prompt_loan_bytes(const PrefillMoe * p, const Model & m, int64_t n, int64_t max_ctx, bool mtp) {
    const int cap = (int) std::min<int64_t>(std::max<int64_t>(n, 1), prefill_moe_options(p).max_T);
    return prefill_moe_loan_bytes(p, n, attn_scratch(m.g, cap, max_ctx) + mtp_rows(m.g, cap, mtp));
}

int prefill_prompt_fit_cap(const PrefillMoe * p, const Model & m, int64_t max_ctx, bool mtp, int slots) {
    const size_t budget = (size_t) slots * routed_blob_bytes(m.g) / 4 * 3;
    for (int cap = prefill_moe_options(p).max_T; cap >= 64; cap /= 2)
        if (prefill_prompt_loan_bytes(p, m, cap, max_ctx, mtp).total <= budget) return cap;
    return 0;
}

int32_t prefill_prompt(PrefillMoe * p, const Model & m, const PrefillPrompt & pp, const int32_t * ids, int n,
                       int64_t pos0, cudaStream_t st) {
    const Geometry & g = m.g;
    const int D = g.n_embd;
    if (n < 1) throw std::runtime_error("prefill_prompt: empty prompt");
    if (!pp.kda || !pp.mla) throw std::runtime_error("prefill_prompt: attention states missing");
    if (pp.mtp && !pp.h_prev) throw std::runtime_error("prefill_prompt: mtp needs h_prev");
    if (pos0 + n > pp.max_ctx) throw std::runtime_error("prefill_prompt: context full");

    // loan: attention scratch for the chunk cap (+ the MTP h rows)
    const int cap = (int) std::min<int64_t>(n, prefill_moe_options(p).max_T);
    const size_t sa = attn_scratch(g, cap, pp.max_ctx);
    const size_t sb = sa + mtp_rows(g, cap, pp.mtp != nullptr);
    prefill_moe_begin(p, n, sb, st);
    Workspace & ws = prefill_moe_workspace(p);
    void * scratch = prefill_moe_scratch(p);
    float * hrows = pp.mtp ? reinterpret_cast<float *>(static_cast<char *>(scratch) + sa) : nullptr;
    if (hrows) ck(cudaMemcpyAsync(hrows, pp.h_prev, D * 4, cudaMemcpyDeviceToDevice, st), "h_prev in");

    ForwardHooks hk;
    hk.attn = [&](int il, const float * y, float * o, const Batch & b, cudaStream_t s) {
        const LayerWeights & L = m.layers[il];
        if (L.mla) mla_forward(g, L.mla_w, (*pp.mla)[il], b, y, o, scratch, s);
        else kda_forward(g, L.kda, (*pp.kda)[il], b, y, o, scratch, s);
    };
    hk.experts = [p](int il, const float * y, const int32_t * eids, const float * w, float * o, int T,
                     cudaStream_t s) { prefill_moe_start(p, il, y, eids, w, o, T, s); };
    hk.experts_finish = [p](int il, float * o, int T, cudaStream_t s) { prefill_moe_finish(p, il, o, T, s); };

    PrefillMoeStats & stats = prefill_moe_stats(p);
    int32_t next = -1;
    try {
        for (int done = 0; done < n;) {
            const int T = std::min(cap, n - done);
            forward_trunk(m, ws, ids + done, T, pos0 + done, g.n_layer, hk, st);
            if (pp.on_next && done + T == n) {
                ck(cudaMemcpyAsync(&next, ws.tok, 4, cudaMemcpyDeviceToHost, st), "next");
                ck(cudaStreamSynchronize(st), "next");
                pp.on_next(next);
            }
            if (pp.mtp) {
                ck(cudaStreamSynchronize(st), "chunk");
                const double t0 = now_ms();
                mtp_catch_up(pp.mtp, m, ws, hk, hrows, T, pos0 + done, scratch, sa, st);
                ck(cudaStreamSynchronize(st), "mtp catch-up");
                stats.mtp_ms += now_ms() - t0;
            }
            done += T;
            if (pp.on_chunk) {
                ck(cudaStreamSynchronize(st), "chunk");
                pp.on_chunk(done, n);
            }
        }
        if (pp.logits) ck(cudaMemcpyAsync(pp.logits, ws.logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, st),
                          "logits");
        ck(cudaMemcpyAsync(&next, ws.tok, 4, cudaMemcpyDeviceToHost, st), "next");
        if (hrows) ck(cudaMemcpyAsync(pp.h_prev, hrows, D * 4, cudaMemcpyDeviceToDevice, st), "h_prev out");
        ck(cudaStreamSynchronize(st), "prompt");
    } catch (...) {
        prefill_moe_end(p, st);
        throw;
    }
    prefill_moe_end(p, st);
    return next;
}

}  // namespace strata::glm
