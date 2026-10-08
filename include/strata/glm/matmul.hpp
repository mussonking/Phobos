#pragma once
// glm_matmul: the dense projections of the GLM engine, y = W x (+ beta * y), on the V100.
//
// Weights (WeightRef, device memory) in their raw GGUF layout: Q8_0, BF16 (the custom IQ4 pack keeps several
// small projections in BF16), F16, F32 (router, indexer projection); or Q8_0R, Q8_0 repacked once at load time by
// matmul_repack into 16-byte aligned quants + separate scales (same values, faster GEMV). ggml convention:
// ne[0] = n_in (contiguous reduction dimension), ne[1] = n_out (rows), ne[2] = batch (per-head matrices such as
// attn_k_b [256,512,64]; 1 otherwise).
//
// Activations are device F32, token-major (types.hpp): x[b * x_batch_stride + t * ldx + i], i < n_in, and
// y[b * y_batch_stride + t * ldy + o], o < n_out. The default strides are contiguous: ldx = n_in, ldy = n_out,
// x_batch_stride = T * ldx, y_batch_stride = T * ldy. Strides let a caller read per-head slices of a [T][heads][d]
// tensor without a permute (ldx = heads * d, x_batch_stride = d) and write into a column slice of a wider output.
//
// Paths (chosen from T and the weight type, transparent to the caller):
//   T <= kMatmulGemvMaxT : Q8_0/Q8_0R -> activations quantized to Q8 (32-blocks, same rounding as ggml's CPU
//                          Q8_0), integer-dot GEMV, bandwidth bound (decode, speculative verify).
//                          F32/F16/BF16 -> float GEMV on the raw F32 x (F32 accumulation).
//   larger T             : Q8_0/Q8_0R/BF16 -> weights dequantized to F16 in scratch (row/batch slices if the
//                          scratch is smaller than the matrix), F16 activations, cuBLAS tensor-core GEMM with F32
//                          accumulation (prefill); F16 the same without the dequantization. F32 -> cuBLAS SGEMM on
//                          the raw x. On this path |x| must stay below 65504.
//
// Every call is asynchronous on `stream`, allocates nothing and never synchronizes; the GEMV path is CUDA-graph
// capturable. The GEMM path uses one cuBLAS handle per host thread and device, created on first use or by
// matmul_init() (call it before any capture).

#include <cstddef>
#include <cstdint>

#include "strata/glm/types.hpp"

namespace strata::glm {

constexpr int kTypeQ8_0R = 1008;    // GLM-private WeightRef::type: Q8_0 after matmul_repack (not a ggml type)
constexpr int kMatmulGemvMaxT = 32;  // largest T on the GEMV path (one launch per 8 tokens); tuned on the V100

struct MatmulStrides {
    int64_t ldx = 0;              // 0 = n_in
    int64_t ldy = 0;              // 0 = n_out
    int64_t x_batch_stride = 0;   // 0 = T * ldx
    int64_t y_batch_stride = 0;   // 0 = T * ldy
    float   beta = 0.0f;          // y = W x + beta * y (beta = 1: accumulate into y)
};

// Scratch (device, 256-byte aligned) for one glm_matmul call that never slices the weights.
// A smaller scratch still works on the GEMM path as long as it holds the F16 activations plus one weight row
// (or one batch matrix); the weights are then dequantized in slices.
size_t matmul_scratch_bytes(const WeightRef & w, int T);

void glm_matmul(const WeightRef & w, const float * x, float * y, int T, void * scratch, size_t scratch_bytes,
                cudaStream_t stream, const MatmulStrides & s = {});

// Shared input: one activation preparation (Q8 or F16) reused by several weights with the same n_in, batch and T
// (q/k/v of a KDA layer, gate/up of an FFN). `prepared` must stay alive until the last matmul_run is enqueued.
struct MatmulInput {
    const float * x = nullptr;
    int     n_in = 0;
    int     T = 0;
    int     batch = 1;
    int64_t ldx = 0, x_batch_stride = 0;   // as in MatmulStrides
    void *  prepared = nullptr;            // inside the scratch given to matmul_prepare
    int     kind = 0;                      // internal: 0 none, 1 Q8, 2 F16
};
size_t matmul_prepare_bytes(int n_in, int T, int batch = 1);
MatmulInput matmul_prepare(const float * x, int n_in, int T, int batch, int64_t ldx, int64_t x_batch_stride,
                           void * scratch, cudaStream_t stream);
// Producer-fused preparation: a kernel that already holds the activations can write the Q8 prepared input itself
// (layout and rounding in q8_block.cuh) into a matmul_prepare_bytes(n_in, T) buffer, and matmul_input_q8 wraps it
// without any launch. Only on the Q8 GEMV path (matmul_input_is_q8); otherwise call matmul_prepare. With batch > 1
// the Q8 columns are batch-major (column b * T + t) as in matmul_prepare; x, ldx and x_batch_stride (0 = defaults)
// describe the raw activations, which float weights read.
bool        matmul_input_is_q8(int n_in, int T);
MatmulInput matmul_input_q8(const float * x, int n_in, int T, const void * prepared, int batch = 1, int64_t ldx = 0,
                            int64_t x_batch_stride = 0);
// Weight scratch for matmul_run (the GEMM path dequantizes there; the GEMV path needs none).
size_t matmul_run_bytes(const WeightRef & w, int T);
void matmul_run(const WeightRef & w, const MatmulInput & in, float * y, void * w_scratch, size_t w_scratch_bytes,
                cudaStream_t stream, int64_t ldy = 0, int64_t y_batch_stride = 0, float beta = 0.0f);

// Q8_0 -> Q8_0R into `dst` (device, 256-byte aligned, matmul_repack_bytes, not overlapping the source).
// Returns the WeightRef of the repacked copy; the source can be freed once the stream has run the copy.
size_t matmul_repack_bytes(const WeightRef & w);
WeightRef matmul_repack(const WeightRef & w, void * dst, cudaStream_t stream);

// Several weights on one prepared input (same type and n_in, batch 1), e.g. the KDA q/k/v: on the GEMV path up to
// 4 Q8_0R/BF16/F16/F32 weights run as ONE launch (less launch and tail overhead than separate calls); otherwise
// this is a loop of matmul_run. ldys may be null (contiguous outputs). y = W x (no beta).
void matmul_run_multi(const WeightRef * ws, float * const * ys, const int64_t * ldys, int n, const MatmulInput & in,
                      void * w_scratch, size_t w_scratch_bytes, cudaStream_t stream);

void matmul_init();   // optional: create this thread's cuBLAS handle now (before graph capture)

}  // namespace strata::glm
