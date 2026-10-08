#pragma once
// Correctness-only routed experts for any GGUF expert type (Q4_K/Q5_K/Q6_K of the oracle model included), on the
// CPU through ggml-cpu's own type traits: the activation is quantized to the weight type's vec_dot_type and every
// row is one vec_dot, as llama.cpp's CPU backend does. Not a hot path: tests and the end-to-end parity check only.

#include <cstdint>

#include "model.hpp"

namespace strata::glm::test {

// out[t] = sum_k w[t][k] * down_e(silu(min(gate_e x, c)) * clamp(up_e x, -c, c)), e = ids[t][k], summed in k
// order like the oracle graph. Host pointers: x/out [T][n_embd], ids/w [T][n_expert_used].
void ref_experts_cpu(const Geometry & g, const HostExperts & ex, const float * x, const int32_t * ids, const float * w,
                     int T, float * out);
// The same with the planes of job j = t * n_expert_used + k at per_job[j].gate / up / down (e.g. a host arena
// blob); ex still gives the types.
void ref_experts_cpu(const Geometry & g, const HostExperts & ex, const float * x, const int32_t * ids, const float * w,
                     int T, float * out, const HostExperts * per_job);

}  // namespace strata::glm::test
