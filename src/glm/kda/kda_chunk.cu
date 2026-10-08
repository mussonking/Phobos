// src/glm/kda/kda_chunk.cu - chunked (block-parallel) KDA recurrence for prefill. Placeholder: sequential.
#include "kda_internal.hpp"

namespace strata::glm::kda {

size_t scan_chunked_work_bytes(int) { return 0; }

void scan_chunked(int T, const float * q, const float * k, const float * v, const float * g, const float * beta,
                  const float * S_in, float * S_out, float * o, void *, cudaStream_t st) {
    scan_sequential(T, q, k, v, g, beta, S_in, S_out, o, st);
}

}  // namespace strata::glm::kda
