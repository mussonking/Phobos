#pragma once
// KDA internals shared by kda.cu, the chunked prefill and the parity/bench drivers. Not an engine interface.
#include <cstddef>

#include "strata/glm/kda.hpp"

typedef struct CUevent_st * cudaEvent_t;

namespace strata::glm::kda {

constexpr int kHeads = 64;
constexpr int kDim   = 128;
constexpr int kInner = kHeads * kDim;   // 8192
constexpr int kEmbd  = 4096;
constexpr int kRank  = 128;             // f_a / g_a bottleneck
constexpr int kHist  = 3;               // conv kernel 4 -> 3 history rows

// Device pointers to the per-batch intermediates of the last kda_forward (valid until the next call that
// reuses the same scratch / state). All token-major [T][...].
struct Views {
    int T = 0;
    float * raw_q = nullptr, * raw_k = nullptr, * raw_v = nullptr;   // projections before conv  [T][8192]
    float * fa = nullptr, * ga = nullptr;                             // [T][128]
    float * glin = nullptr;                                           // W_fb fa (no dt bias)     [T][8192]
    float * braw = nullptr;                                           // W_beta x                 [T][64]
    float * g2 = nullptr;                                             // W_gb ga                  [T][8192]
    float * q = nullptr, * k = nullptr, * v = nullptr;                // conv+SiLU (+L2 for q,k)  [T][8192]
    float * g = nullptr;                                              // log decay in [-5, 0]     [T][8192]
    float * beta = nullptr;                                           // [T][64]
    float * o = nullptr;                                              // recurrence output        [T][8192]
    float * gated = nullptr;                                          // normed * sigmoid(g2)     [T][8192]
};

// Pointers the last forward used (filled by kda_forward; for tests).
const Views & last_views();

// Prefill recurrence selection: 0 = sequential scan for every T, 1 = chunked when T > kKdaLogT (default).
void set_prefill_mode(int mode);

// Decode / verify (T <= kKdaLogT): one fused prep + recurrence + post launch (default) or the three kernels.
void set_fused_decode(bool on);
void set_skip_projections(bool on);   // bench only
void set_rq_variant(int v);            // bench only: long-batch scan layout

// Bench only: record ev[0..5] at the stage boundaries of kda_forward (start, in-proj, prep, recurrence, post,
// out-proj). nullptr disables.
void set_stage_events(cudaEvent_t * ev);

// The recurrence alone on prepared inputs (for the chunked-vs-sequential test and timings).
// S_in may equal S_out only for the sequential path. o may be null (state only).
void scan_sequential(int T, const float * q, const float * k, const float * v, const float * g, const float * beta,
                     const float * S_in, float * S_out, float * o, cudaStream_t st);
void scan_chunked(int T, const float * q, const float * k, const float * v, const float * g, const float * beta,
                  const float * S_in, float * S_out, float * o, void * work, cudaStream_t st);
size_t scan_chunked_work_bytes(int T);

}  // namespace strata::glm::kda
