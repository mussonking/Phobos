#include "iq4_cpu.hpp"

#include "ggml.h"
#include "strata/glm/cpu_isa.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define GLM_IQ4_X86 1
#define GLM_IQ4_TARGET __attribute__((target("avx512f,avx512bw,fma,f16c")))
#else
#define GLM_IQ4_X86 0
#endif

namespace strata::glm {
namespace {

uint16_t load_u16(const unsigned char * p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

void reference_row(int type, const void * row, int width, int T,
                   const float * const * x, float * out) {
    std::vector<float> weights(width);
    ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(row, weights.data(), width);
    for (int t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) sum += weights[i] * x[t][i];
        out[t] = sum;
    }
}

#if GLM_IQ4_X86
GLM_IQ4_TARGET float half_to_float(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

// SSSE3 byte shuffle is lane-local and available on Cascade Lake. Unlike the
// Qwen IQ kernel this path does not require AVX512VBMI or integer activations.
GLM_IQ4_TARGET void avx512_row(int type, const void * row, int width, int T,
                             const float * const * x, float * out) {
    const __m128i table = _mm_setr_epi8(-127, -104, -83, -65, -49, -35, -22, -10,
                                       1, 13, 25, 38, 53, 69, 89, 113);
    const __m128i mask = _mm_set1_epi8(15);
    __m512 acc[8];
    for (int t = 0; t < T; ++t) acc[t] = _mm512_setzero_ps();
    const auto * bytes = static_cast<const unsigned char *>(row);
    const bool xs = type == GGML_TYPE_IQ4_XS;
    const int block_width = xs ? 256 : 32;
    const int block_bytes = xs ? 136 : 18;
    for (int block = 0; block < width / block_width; ++block) {
        const auto * b = bytes + block * block_bytes;
        const float d = half_to_float(load_u16(b));
        const uint16_t scales_h = xs ? load_u16(b + 2) : 0;
        for (int group = 0; group < block_width / 32; ++group) {
            float scale = d;
            if (xs) {
                const int ls = ((b[4 + group / 2] >> (4 * (group % 2))) & 15)
                             | (((scales_h >> (2 * group)) & 3) << 4);
                scale *= ls - 32;
            }
            const auto * q = b + (xs ? 8 : 2) + group * 16;
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q));
            const __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
            const __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
            const __m512 s = _mm512_set1_ps(scale);
            // Scale before multiplying x: exactly ggml's dequantized F32 weights.
            const __m512 w0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), s);
            const __m512 w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), s);
            const int offset = block * block_width + group * 32;
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(x[t] + offset), acc[t]);
                acc[t] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(x[t] + offset + 16), acc[t]);
            }
        }
    }
    for (int t = 0; t < T; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

// iq4_cpu_row_f64: the same exact F32 weights, products and sums in double (8 lanes per vector)
GLM_IQ4_TARGET void avx512_row_f64(int type, const void * row, int width, int T,
                                 const double * const * x, double * out) {
    const __m128i table = _mm_setr_epi8(-127, -104, -83, -65, -49, -35, -22, -10,
                                       1, 13, 25, 38, 53, 69, 89, 113);
    const __m128i mask = _mm_set1_epi8(15);
    __m512d acc[8];
    for (int t = 0; t < T; ++t) acc[t] = _mm512_setzero_pd();
    const auto * bytes = static_cast<const unsigned char *>(row);
    const bool xs = type == GGML_TYPE_IQ4_XS;
    const int block_width = xs ? 256 : 32;
    const int block_bytes = xs ? 136 : 18;
    for (int block = 0; block < width / block_width; ++block) {
        const auto * b = bytes + block * block_bytes;
        const float d = half_to_float(load_u16(b));
        const uint16_t scales_h = xs ? load_u16(b + 2) : 0;
        for (int group = 0; group < block_width / 32; ++group) {
            float scale = d;
            if (xs) {
                const int ls = ((b[4 + group / 2] >> (4 * (group % 2))) & 15)
                             | (((scales_h >> (2 * group)) & 3) << 4);
                scale *= ls - 32;
            }
            const auto * q = b + (xs ? 8 : 2) + group * 16;
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q));
            const __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
            const __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
            const __m512 s = _mm512_set1_ps(scale);
            const __m512 w0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), s);
            const __m512 w1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), s);
            const __m512d w[4] = {
                _mm512_cvtps_pd(_mm512_castps512_ps256(w0)),
                _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(w0), 1))),
                _mm512_cvtps_pd(_mm512_castps512_ps256(w1)),
                _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(w1), 1)))};
            const int offset = block * block_width + group * 32;
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < 4; ++k)
                    acc[t] = _mm512_fmadd_pd(w[k], _mm512_loadu_pd(x[t] + offset + k * 8), acc[t]);
        }
    }
    for (int t = 0; t < T; ++t) out[t] = _mm512_reduce_add_pd(acc[t]);
}
#endif

} // namespace

bool iq4_cpu_has_avx512() {
#if GLM_IQ4_X86
    static const bool supported = [] {
        __builtin_cpu_init();
        // a GLM_CPU_ISA below the hardware's (the community path on a VNNI machine) keeps this AVX512 kernel off
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && cpu_isa() == cpu_isa_hardware();
    }();
    return supported;
#else
    return false;
#endif
}

std::size_t iq4_cpu_row_bytes(int type, int width) {
    const int block = type == GGML_TYPE_IQ4_XS ? 256 : 32;
    if ((type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_IQ4_NL) || width <= 0 || width % block)
        throw std::invalid_argument("IQ4 row: unsupported type or misaligned width");
    return static_cast<std::size_t>(width / block) * (type == GGML_TYPE_IQ4_XS ? 136 : 18);
}

void iq4_cpu_row(int type, const void * row, int width, int T,
                 const float * const * x, float * out, CPUExpertPath path) {
    iq4_cpu_row_bytes(type, width);
    if (!row || !x || !out || T < 1 || T > 8)
        throw std::invalid_argument("IQ4 row: null pointer or T outside [1,8]");
    for (int t = 0; t < T; ++t)
        if (!x[t]) throw std::invalid_argument("IQ4 row: null activation pointer");
    if (path == CPUExpertPath::avx512 && !iq4_cpu_has_avx512())
        throw std::runtime_error("IQ4 row: requested AVX512F/BW/FMA/F16C unavailable");
#if GLM_IQ4_X86
    if (path != CPUExpertPath::reference && iq4_cpu_has_avx512()) {
        avx512_row(type, row, width, T, x, out);
        return;
    }
#endif
    reference_row(type, row, width, T, x, out);
}

void iq4_cpu_row_f64(int type, const void * row, int width, int T, const double * const * x, double * out) {
    iq4_cpu_row_bytes(type, width);
    if (!row || !x || !out || T < 1 || T > 8)
        throw std::invalid_argument("IQ4 row f64: null pointer or T outside [1,8]");
#if GLM_IQ4_X86
    if (iq4_cpu_has_avx512()) {
        avx512_row_f64(type, row, width, T, x, out);
        return;
    }
#endif
    std::vector<float> weights(width);
    ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(row, weights.data(), width);
    for (int t = 0; t < T; ++t) {
        double sum = 0.0;
        for (int i = 0; i < width; ++i) sum += (double) weights[i] * x[t][i];
        out[t] = sum;
    }
}

double expert_swiglu_clamp_f64(double gate, double up) {
    gate = std::fmin(gate, 10.0);
    up = std::fmax(-10.0, std::fmin(up, 10.0));
    return (gate / (1.0 + std::exp(-gate))) * up;
}

float expert_swiglu_clamp(float gate, float up) {
    gate = std::fmin(gate, 10.0f);
    up = std::fmax(-10.0f, std::fmin(up, 10.0f));
    return (gate / (1.0f + std::exp(-gate))) * up;
}

} // namespace strata::glm
