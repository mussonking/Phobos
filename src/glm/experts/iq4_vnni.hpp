#pragma once

#include "strata/glm/experts.hpp"

namespace strata::glm {

// Optional expert helper: original x and clamped SwiGLU are quantized to Q8_K.
// Separate down-row APIs also support native Q8_0. Neither changes exact F32.
// automatic = cpu_isa() (strata/glm/cpu_isa.hpp): VNNI, else AVX2 (same integer dots, ggml's sign trick on
// vpmaddubsw; float sums in another order than VNNI), else scalar.
enum class IQ4VNNIPath { automatic, scalar, vnni, avx2 };
// cpu_isa() == vnni: AVX512 VNNI present and not lowered by GLM_CPU_ISA
bool iq4_vnni_available();
// cpu_isa() >= avx2: the int8 rows run vectorized (VNNI or AVX2)
bool iq4_int8_available();
std::size_t iq4_vnni_q8_bytes(int width);
std::size_t iq4_vnni_row_bytes(int type, int width);
// Storage must be at least float-aligned. AVX512 quantization preserves GGML
// reference bytes; exceptional blocks use that reference implementation.
// Zero-block bsums are initialized (ggml's reference leaves these untouched).
void iq4_vnni_quantize(const float * x, int width, void * q8);
// VNNI rows use vector/FMA accumulation and one final reduction. Results need
// not be bitwise equal to the scalar path; activation storage remains Q8_K.
float iq4_vnni_row(int type, const void * weights, int width, const void * q8,
                   IQ4VNNIPath path = IQ4VNNIPath::automatic);

// Consecutive weight rows sharing one Q8_K activation; dispatch/validation once.
void iq4_vnni_rows(int type, const void * weights, int width, const void * q8,
                    int n_rows, float * out, IQ4VNNIPath path = IQ4VNNIPath::automatic);
// The same rows against n_act <= 8 activations (out[n_act][n_rows]): each weight group is unpacked once for all of
// them (speculative verify: one missed expert, several tokens). Bitwise equal to n_act iq4_vnni_rows calls.
void iq4_vnni_rows_multi(int type, const void * weights, int width, const void * const * q8, int n_act,
                         int n_rows, float * out, IQ4VNNIPath path = IQ4VNNIPath::automatic);

// Separate native IQ4_NL down policy: GGML Q8_0 blocks (FP16 d, 32 int8s).
// These activations are NOT interchangeable with existing Q8_K buffers.
std::size_t iq4_vnni_q8_0_bytes(int width);
void iq4_vnni_quantize_q8_0(const float * x, int width, void * q8_0);
void iq4_vnni_nl_q8_0_rows(const void * weights, int width, const void * q8_0,
                           int n_rows, float * out, IQ4VNNIPath path = IQ4VNNIPath::automatic);

std::size_t vnni_expert_activation_bytes(const Geometry &, int T);
std::size_t vnni_expert_scratch_bytes(const Geometry &, int n_rows);
void vnni_expert_prepare(const Geometry &, const float * x, int T, void * act);
// act is shared immutable Q8_K prepared once per token across all eight jobs.
// Output is overwritten, unweighted [n_rows][n_embd]. Scratch is private,
// float-aligned, and disjoint from act/out/weights; eval performs no allocation.
void vnni_expert_eval(const Geometry &, const ExpertRef &, const void * act, int T,
                      int n_rows, const int * tokens, float * out, void * scratch,
                      IQ4VNNIPath path = IQ4VNNIPath::automatic);

} // namespace strata::glm
