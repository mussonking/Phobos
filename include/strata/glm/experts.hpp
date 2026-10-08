#pragma once

#include <cstddef>
#include "strata/glm/types.hpp"

namespace strata::glm {

// Gate/up are IQ4_XS rows, down is IQ4_NL. GPU cache blobs use gate | up | down.
// Native mapped GGUF CPU refs may provide a separate up plane to avoid a copy.
// Pointers reference host or device memory according to the entry point.
struct ExpertRef {
    const void * gate_up = nullptr;
    const void * down = nullptr;
    int type_gate_up = 23; // GGML_TYPE_IQ4_XS
    int type_down = 20;    // GGML_TYPE_IQ4_NL
    const void * up = nullptr; // optional separate up plane; null means contiguous gate | up
    int type_up = -1; // -1 inherits type_gate_up; mixed gate/up types are unsupported
    int layout = 0; // 0: GGUF rows; 1: R8 host arena only (device refs remain 0)
};
static_assert(sizeof(void *) != 8 || sizeof(ExpertRef) == 40, "ExpertRef must preserve its 64-bit ABI");
static_assert(sizeof(void *) != 8 || offsetof(ExpertRef, layout) == 36, "R8 layout must use existing tail padding");

// All arrays are device memory. x/out: [T][n_embd]; ids/w/slots: [T][top_k].
// A negative/out-of-range id or null gate_up/down skips the CPU miss.
// Output is additive; the caller zeroes it once before the GPU + CPU sum.
// Geometry and T are capture-time constants. No allocation/synchronization.
// Scratch is device memory, cudaMalloc aligned, disjoint from all arrays.
std::size_t experts_gpu_scratch_bytes(const Geometry &, int T);
void experts_gpu(const Geometry &, int T, const float * x, const int * ids,
                 const float * w, const ExpertRef * slots, float * out,
                 void * scratch, cudaStream_t stream);
void experts_gpu_prefill(const Geometry &, int T, const float * x, const int * ids,
                         const float * w, const ExpertRef * slots, float * out,
                         void * scratch, cudaStream_t stream);

// Host-only prepared F32 activation rows, shared read-only by every expert.
// The exact IQ4/F32 AVX512 path avoids a lossy Q8 activation approximation.
// Scratch for prepare/eval must be float aligned and distinct from outputs.
// An eval call handles 1..8 rows of the same expert, reusing weight unpacking;
// tokens=null selects rows [0,n_rows), otherwise tokens indexes [0,T).
std::size_t cpu_expert_activation_bytes(const Geometry &, int T);
std::size_t cpu_expert_scratch_bytes(const Geometry &, int n_rows);
void cpu_expert_prepare(const Geometry &, const float * x, int T, void * act_scratch);
void cpu_expert_eval(const Geometry &, const ExpertRef &, const void * act, int T,
                     int n_rows, const int * tokens, float * out, void * private_scratch);

} // namespace strata::glm
