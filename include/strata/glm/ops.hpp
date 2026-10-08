#pragma once
// GLM-5.3-Flash shared trunk ops (docs/glm5next-spec.md sections 3-4): mHC pre/post with Sinkhorn, router,
// clamped SwiGLU, stream-mean head norm. All device F32, token-major. Every op is a single launch.
//
// mHC streams: x[t][s][i] = x[(t * hc + s) * n_embd + i] (ggml [D, hc, N]).
// Mix partials: the 24 mHC logits of a branch need RMSNorm(flat x) and hc_fn [hc*D -> 24]. They are computed in
// the same pass that writes x (hc_init / hc_post), as per-feature-block partial dot products of the RAW streams
// plus the partial sum of squares; hc_pre finishes them (mixes = rinv * W x, the same value as W * RMSNorm(x)).
// hcp[t][24] = pre[4], post[4], comb[16] with comb[dst + 4 * src].

#include <cstddef>
#include <cstdint>

#include "strata/glm/geometry.hpp"
#include "strata/glm/types.hpp"

namespace strata::glm {

struct HcBranch {
    WeightRef fn;      // [hc * n_embd -> 24]; Q8_0, Q8_0R, BF16, F16 or F32
    WeightRef base;    // F32 [24]
    WeightRef scale;   // F32 [3]
};

constexpr int kHcSplit = 128;   // blocks of 128 flattened features in the mix reduction (partials per token)
inline size_t hc_part_floats(int T) { return (size_t) T * kHcSplit * 25; }

// x[t][s] = e[t] for every stream (embedding -> streams), then the mix partials of `next` into part.
void hc_init(const Geometry & g, const float * e, float * x, int T, const HcBranch & next, float * part,
             cudaStream_t st);

// x_out[t][dst] = (branch[t] + branch2[t]) * post[t][dst] + sum_src x_in[t][src] * comb[t][dst + 4 src]
// (x_out must not alias x_in: the engine ping-pongs two stream buffers). branch2 may be null (the MoE shared expert
// is added here). If next is non-null, also writes its mix partials.
void hc_post(const Geometry & g, const float * branch, const float * branch2, const float * hcp, const float * x_in,
             float * x_out, int T, const HcBranch * next, float * part, cudaStream_t st);

// Finishes the mixes of `br`, writes hcp (pre, post, and the RAW combine logits), and
// y[t] = RMSNorm(sum_s pre[t][s] * x[t][s]) * norm_w.
// y_q8 (optional): also writes y as glm_matmul's Q8 prepared input (q8_block.cuh; wrap with matmul_input_q8).
void hc_pre(const Geometry & g, const HcBranch & br, const float * x, const float * part, int T,
            const WeightRef & norm_w, float * hcp, float * y, cudaStream_t st, void * y_q8 = nullptr);

// Sinkhorn on hcp's combine logits, in place (row softmax, eps, 1 column norm, then (iters - 1) row + column
// norms). Needed by hc_post only, so the engine runs it on a side stream in parallel with the branch.
void hc_comb(const Geometry & g, float * hcp, int T, cudaStream_t st);

// y[t] = RMSNorm(mean_s x[t][s]) * norm_w (the pre-head stream mean + output_norm). y_q8 as in hc_pre.
void hc_head(const Geometry & g, const float * x, int T, const WeightRef & norm_w, float * y, cudaStream_t st,
             void * y_q8 = nullptr);

// y[t] = RMSNorm(x[t] + add[t]) * norm_w over n features (add may be null; if sum_out is non-null the sum
// x + add is also written there). Rows at stride ld (0 = n) for x, add, y and sum_out.
void rms_norm(const float * x, const float * add, const WeightRef & norm_w, float * y, float * sum_out, int T, int n,
              int64_t ld, float eps, cudaStream_t st);

// Router: p = sigmoid(logits[t]), top-n_expert_used of p + bias (selection only, descending), weights from the
// unbiased p normalized by max(sum, floor) times expert_scale. ids/w are [T][n_expert_used].
void moe_route(const Geometry & g, const float * logits, const WeightRef & bias, int T, int32_t * ids, float * w,
               cudaStream_t st);

// out[t] = argmax_j x[t][j] over n (first index on ties), rows contiguous.
void argmax_rows(const float * x, int n, int T, int32_t * out, cudaStream_t st);

// h[t][j] = silu(min(gate, c)) * clamp(up, -c, c) for j < F; gate/up rows at stride ld_in, h rows at ldh.
// h_q8 (optional, F % 32 == 0): also writes h as the Q8 prepared input of [T][F] (as hc_pre's y_q8).
void swiglu_clamp(const float * gate, const float * up, float * h, int T, int F, int64_t ld_in, int64_t ldh,
                  float c, cudaStream_t st, void * h_q8 = nullptr);

}  // namespace strata::glm
