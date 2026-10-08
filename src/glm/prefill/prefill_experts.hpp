#pragma once
// Routed experts of one MoE layer for a large prompt chunk (prefill), on the V100 tensor cores.
//
// The T x K (token, expert) jobs are sorted by expert on the device (stable: token order inside an expert), the
// activations gathered as F16 rows in that order, and the experts processed in groups: the group's IQ4 weights
// (gate | up IQ4_XS, down IQ4_NL, pack's blob layout) are dequantized to F16, then one grouped cuBLAS GEMM per
// projection (F16 inputs, F32 accumulation) with the clamped SwiGLU in between. The per-job outputs are combined
// per token in slot order (out[t] = sum_k w[t][k] * expert_k(x[t]), deterministic).
//
// The weights come from a per-layer device table of ExpertRef (one per expert, device pointers): streamed ring
// slots or resident cache slots. A null entry contributes nothing (that expert is handled elsewhere).
//
// The host reads the per-expert counts once per call (one stream synchronization after the routing sort) to size
// the GEMMs; everything else is asynchronous.
#include <cstddef>
#include <cstdint>
#include <functional>

#include <cuda_runtime.h>

#include "strata/glm/experts.hpp"
#include "strata/glm/geometry.hpp"

namespace strata::glm {

struct PrefillExperts;

// max_T: largest chunk; group: experts dequantized together (F16 weights of `group` experts live at once).
PrefillExperts * prefill_experts_create(const Geometry & g, int max_T, int group);
void prefill_experts_destroy(PrefillExperts * p);
size_t prefill_experts_device_bytes(const PrefillExperts * p);   // device buffers (owned or bound)

// External device memory (e.g. a routed_lend loan only during the prefill): create without device buffers, then
// bind prefill_experts_bytes(g, max_T, group) bytes (256-aligned, not owned) before the runs. bind(nullptr) unbinds;
// rebinding frees an owned arena first. group (0: the created one) may be smaller than the created one (a small
// prompt's loan).
size_t prefill_experts_bytes(const Geometry & g, int max_T, int group);
PrefillExperts * prefill_experts_create_unbound(const Geometry & g, int group);
void prefill_experts_bind(PrefillExperts * p, int max_T, void * mem, cudaStream_t st, int group = 0);

// x [T][n_embd] F32, ids / w [T][n_expert_used], table [n_expert] ExpertRef (device), out [T][n_embd] (overwritten).
void prefill_experts_run(PrefillExperts * p, int T, const float * x, const int32_t * ids, const float * w,
                         const ExpertRef * table, float * out, cudaStream_t st);

// Host-planned run (no host synchronization): counts = routed tokens per expert (host [n_expert], from ids), order =
// the experts to run here in processing order (each with a non-null table entry; experts outside it must have a null
// one: they contribute nothing). before(i) is called before the dequantization of order[i] is enqueued (e.g. to
// enqueue its upload and make st wait for it), after(i) once it is enqueued (its blob may then be overwritten).
struct PrefillExpertsPlan {
    const int * counts = nullptr;
    const int * order = nullptr;
    int n_order = 0;
    std::function<void(int i)> before, after;
};
void prefill_experts_run(PrefillExperts * p, int T, const float * x, const int32_t * ids, const float * w,
                         const ExpertRef * table, const PrefillExpertsPlan & plan, float * out, cudaStream_t st);

// Phase timings of the last run (ms, events), for the bench.
struct PrefillExpertsTiming {
    float sort = 0, gather = 0, dequant = 0, gemm_gu = 0, swiglu = 0, gemm_down = 0, combine = 0;
    int active_experts = 0;
};
PrefillExpertsTiming prefill_experts_timing(PrefillExperts * p);
void prefill_experts_set_timing(PrefillExperts * p, bool on);

}  // namespace strata::glm
