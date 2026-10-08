#pragma once
// Cross-component contracts for the GLM engine. Every component (KDA, MLA+DSA, experts, engine) talks through
// these types so they can be built in parallel and assembled without rewrites.

#include <cstdint>

#include "strata/glm/geometry.hpp"

typedef struct CUstream_st * cudaStream_t;

namespace strata::glm {

// A weight tensor as stored in the GGUF (ggml layout: ne[0] is the contiguous input dimension).
// `data` is device memory unless stated otherwise; `type` is the ggml_type id (GGML_TYPE_Q8_0, ...).
struct WeightRef {
    const void * data = nullptr;
    int          type = -1;
    int64_t      ne[4] = {1, 1, 1, 1};
    uint64_t     nbytes = 0;
};

// Activations are device float32, token-major: x[t * dim + i]. T = tokens in the batch (1 for plain decode,
// the speculative window for verify, a chunk for prefill). One sequence per engine (Strata serves one
// conversation at a time); positions are absolute token positions in that sequence.
struct Batch {
    int       T = 1;
    int64_t   pos0 = 0;   // absolute position of token 0 of this batch
};

}  // namespace strata::glm
