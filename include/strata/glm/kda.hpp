#pragma once
// GLM-5.3-Flash KDA (Kimi Delta Attention) layer: the attention branch of the 34 linear-attention trunk blocks,
// from the attention-normed input to the attn_output projection (docs/glm5next-spec.md section 3, "KDA block").
//
//   q,k,v  = conv4_causal(W_{q,k,v} x) -> SiLU             (3 history rows per q/k/v kept in the state)
//   g      = -5 * sigmoid(-(A_h * (W_fb W_fa x + dt_bias)))  per channel, A_h = ssm_a[h] = -exp(A_log)
//   beta   = sigmoid(W_beta x)                              per head
//   q,k    = q / sqrt(|q|^2 + 1e-6) per head (same for k)
//   S_t    = diag(exp g_t) S_{t-1} + k_t delta_t^T,  delta_t = beta_t (v_t - (diag(exp g_t) S_{t-1})^T k_t)
//   o_t    = S_t^T q_t / sqrt(128)
//   out    = W_o ( RMSNorm_head(o) * ssm_norm * sigmoid(W_gb W_ga x) )
//
// State layout (owned here, per layer per sequence, all F32 on the device):
//   S     [64 heads][128 value j][128 key i]  (ggml's transposed layout: row j of a head is contiguous over i,
//                                             so it is bit-compatible with llama.cpp's cache_s / new_state)
//   conv  [3 = q,k,v][3 rows, oldest first][8192]   (time-major; llama.cpp stores it channel-major [q,k,v][8192][3])
// Both are double-buffered: kda_forward reads buffer `cur` and writes `cur ^ 1`, then flips. The raw inputs of the
// last batch (projections before conv / gates) are logged inside the state when T <= kKdaLogT.
//
// Speculative rollback (cheapest scheme): the pre-batch state is never overwritten by a forward, so rejecting
// drafts costs nothing up front. kda_state_rollback(n_keep) flips back to the pre-batch buffers and, if
// n_keep > 0, replays the conv + recurrence for the first n_keep logged tokens (no projections, no output),
// ~one state read + write (8 MiB, ~15 us on a V100) per layer, paid only when a draft is partially accepted.
// Full accept: nothing to do. Generic snapshot/restore (a D2D copy of the current buffers) is also provided.
#include <cstddef>

#include "strata/glm/types.hpp"

namespace strata::glm {

struct MatmulInput;   // matmul.hpp

struct KdaWeights {
    // Q8_0 [4096 -> 8192] / [8192 -> 4096] projections
    WeightRef q, k, v, out;
    // F32 [4][8192] depthwise conv taps (ggml ne = {4, 1, 8192})
    WeightRef conv_q, conv_k, conv_v;
    // decay gate: Q8_0 f_a [4096 -> 128], f_b [128 -> 8192]; F32 dt_bias [8192], a [64] (= -exp(A_log))
    WeightRef f_a, f_b, dt_bias, a;
    // Q8_0 beta [4096 -> 64]
    WeightRef beta;
    // output gate: Q8_0 g_a [4096 -> 128], g_b [128 -> 8192]; F32 norm [128] (shared by all heads)
    WeightRef g_a, g_b, norm;
};

constexpr int kKdaLogT = 8;   // largest batch that can be rolled back (the speculative window)

struct KdaState;

KdaState * kda_state_create(const Geometry & g);
void       kda_state_destroy(KdaState * s);
void       kda_state_reset(KdaState * s, cudaStream_t st);            // zero S and conv history
void       kda_state_snapshot(KdaState * s, cudaStream_t st);         // copy current S + conv to the snapshot slot
void       kda_state_restore(KdaState * s, cudaStream_t st);          // copy the snapshot slot back to current
// Keep only the first n_keep tokens (0 <= n_keep <= T) of the last kda_forward. Requires that forward's T <=
// kKdaLogT and no other forward since. n_keep == T is a no-op.
void       kda_state_rollback(const Geometry & g, KdaState * s, int n_keep, cudaStream_t st);
// Host checkpoint of the current S + conv (kda_state_bytes(g) bytes, [S][conv]; pinned memory keeps the copies
// asynchronous). Load writes the current buffer in place (the parity, hence the captured graph, is unchanged) and
// drops the rollback log. Both are stream-ordered: the host buffer must stay untouched until the stream syncs.
void       kda_state_save(const KdaState * s, void * host, cudaStream_t st);
void       kda_state_load(KdaState * s, const void * host, cudaStream_t st);

// Device views of the current state (for tests and the engine's own checkpointing).
const float * kda_state_S(const KdaState * s);       // [64][128][128]
const float * kda_state_conv(const KdaState * s);    // [3][3][8192]
size_t        kda_state_bytes(const Geometry & g);   // S + conv of one buffer

// CUDA graphs: a captured kda_forward always reads the buffer that was `cur` at capture and writes the other one,
// so capture one graph per parity and pick it with kda_state_cur(). After replaying a graph captured with T tokens,
// call kda_state_advance(s, T): the host bookkeeping of that kda_forward (flip, rollback log), no GPU work. Every
// other per-call input (scratch layout, log pointers, launch configs) depends only on T.
int  kda_state_cur(const KdaState * s);              // 0 or 1
void kda_state_advance(KdaState * s, int T);

// x_norm, out: device [T][4096] F32. scratch: device, kda_scratch_bytes(g, T) bytes, 256-byte aligned.
// x_in (optional): x_norm already prepared for glm_matmul (e.g. matmul_input_q8 of a Q8 written by hc_pre); the
// q/k/v/f_a/g_a/beta projections then skip their matmul_prepare. Must stay valid until the call returns.
void   kda_forward(const Geometry & g, const KdaWeights & w, KdaState * s, const Batch & b,
                   const float * x_norm, float * out, void * scratch, cudaStream_t st,
                   const MatmulInput * x_in = nullptr);
size_t kda_scratch_bytes(const Geometry & g, int max_T);

}  // namespace strata::glm
