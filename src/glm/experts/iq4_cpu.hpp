#pragma once

#include <cstddef>

namespace strata::glm {

// The row kernel keeps activations in F32. No Q8 activation approximation is used.
enum class CPUExpertPath { automatic, reference, avx512 };

struct CPUExpertShape {
    int hidden = 4096;
    int intermediate = 2048;
    int top_k = 8;
    int experts = 288;
};

struct CPUExpertRef {
    const void * gate_up = nullptr; // gate plane; up follows unless separately supplied
    const void * down = nullptr;
    int type_gate_up = 23;          // GGML_TYPE_IQ4_XS
    int type_down = 20;             // GGML_TYPE_IQ4_NL
    const void * up = nullptr;      // separate native GGUF plane; null means contiguous
};

struct CPUExpertJob {
    int token = 0;
    int route_index = 0;
    CPUExpertRef expert;
    float weight = 0.0f;
};

bool iq4_cpu_has_avx512();
std::size_t iq4_cpu_row_bytes(int type, int width);

// x[t] points to width floats; out[t] is overwritten. T must be in [1,8].
// The AVX512 implementation unpacks each 32-weight group once for all T inputs.
void iq4_cpu_row(int type, const void * row, int width, int T,
                 const float * const * x, float * out,
                 CPUExpertPath path = CPUExpertPath::automatic);

// Order-neutral reference: the same exact F32 weights, double activations, products and sums in double.
void iq4_cpu_row_f64(int type, const void * row, int width, int T, const double * const * x, double * out);

float expert_swiglu_clamp(float gate, float up);
double expert_swiglu_clamp_f64(double gate, double up);

// Unweighted evaluator for parallel pool jobs. Scratch contains n_rows *
// intermediate floats, aligned for float, and must be private to this call.
// tokens may be null to select rows [0,n_rows) of the token-major input.
void cpu_expert_iq4_eval(const CPUExpertShape & shape, const CPUExpertRef & expert,
                         int T, const float * x, int n_rows, const int * tokens,
                         float * out, float * scratch,
                         CPUExpertPath path = CPUExpertPath::automatic);

// Host-only, synchronous and additive: preserves the initial out[T][hidden].
// All jobs are validated before writing. Repeated expert pointers are grouped
// in chunks of at most 8 jobs to reuse unpacking. Callers own scheduling/NUMA;
// concurrent calls must use disjoint outputs or private reduction buffers.
void experts_cpu_iq4(const CPUExpertShape & shape, int T, const float * x,
                      int n_jobs, const CPUExpertJob * jobs, float * out,
                      CPUExpertPath path = CPUExpertPath::automatic);

} // namespace strata::glm
