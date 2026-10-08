#include "iq4_vnni.hpp"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define STRATA_VNNI 1
#define VNNI_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c")))
#define QUANT_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,f16c")))
#define AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#endif
#include "strata/glm/cpu_isa.hpp"

// Exported by ggml-base; Q8_K has no public from_float_ref trait.
extern "C" void quantize_row_q8_K_ref(const float *, block_q8_K *, int64_t);
extern "C" void quantize_row_q8_0_ref(const float *, block_q8_0 *, int64_t);

namespace strata::glm {
namespace {
constexpr int8_t codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                 1, 13, 25, 38, 53, 69, 89, 113};
void width_check(int width) {
    if (width <= 0 || width % 256) throw std::invalid_argument("Q8_K width must be a positive multiple of 256");
}
void shape_check(const Geometry & g, int T) {
    width_check(g.n_embd);
    width_check(g.ffn_expert);
    if (T < 1 || T > 8 || !std::isfinite(g.swiglu_clamp) || g.swiglu_clamp <= 0)
        throw std::invalid_argument("invalid VNNI expert geometry/tokens");
}
void dot32_scalar(const uint8_t * qs, const block_q8_K * const * inputs,
                  int block, int group, int n_rows, float scale, float * sums) {
    int8_t w[32];
    for (int j = 0; j < 16; ++j) {
        w[j] = codebook[qs[j] & 15];
        w[j + 16] = codebook[qs[j] >> 4];
    }
    for (int t = 0; t < n_rows; ++t) {
        const auto & x = inputs[t][block];
        int sum = 0;
        for (int j = 0; j < 32; ++j) sum += int(w[j]) * x.qs[group * 32 + j];
        sums[t] += (scale * x.d) * sum;
    }
}
#ifdef STRATA_VNNI
QUANT_TARGET __m512 quant_abs(__m512 v) {
    // Integer bitwise operations avoid an AVX512DQ requirement.
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7fffffff)));
}
QUANT_TARGET bool quant_nonfinite(__m512 v) {
    const auto bits = _mm512_castps_si512(quant_abs(v));
    return _mm512_cmp_epi32_mask(bits, _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GE) != 0;
}
QUANT_TARGET void quantize_q8_K_fast(const float * x, block_q8_K * y, int width) {
    // Scalar GGML owns exceptional values and non-default rounding/FTZ/DAZ.
    // These are compatibility fallbacks, not a new input rejection policy.
    if (_mm_getcsr() & 0xe040) {
        std::memset(y, 0, std::size_t(width / 256) * sizeof(*y));
        quantize_row_q8_K_ref(x, y, width);
        return;
    }
    for (int i = 0; i < width / 256; ++i, x += 256) {
        auto & b = y[i];
        __m512 v[16], am = _mm512_setzero_ps();
        bool nonfinite = false;
        for (int j = 0; j < 16; ++j) {
            v[j] = _mm512_loadu_ps(x + 16 * j);
            nonfinite |= quant_nonfinite(v[j]);
            am = _mm512_max_ps(quant_abs(v[j]), am);
        }
        if (nonfinite) {
            std::memset(&b, 0, sizeof(b));
            quantize_row_q8_K_ref(x, &b, 256);
            continue;
        }
        const float amax = _mm512_reduce_max_ps(am);
        if (amax == 0) {
            // The GGML zero branch leaves bsums untouched; our public contract
            // has always cleared those bytes as well as d/qs.
            std::memset(&b, 0, sizeof(b));
            continue;
        }
        float max = 0;
        const auto va = _mm512_set1_ps(amax);
        for (int j = 0; j < 16; ++j) {
            const auto m = _mm512_cmp_ps_mask(quant_abs(v[j]), va, _CMP_EQ_OQ);
            if (m) { max = x[16 * j + __builtin_ctz(unsigned(m))]; break; }
        }
        // The first signed maximum wins, including equal magnitudes of opposite sign.
        const float iscale = -127.0f / max;
        if (!std::isfinite(iscale)) {
            std::memset(&b, 0, sizeof(b));
            quantize_row_q8_K_ref(x, &b, 256);
            continue;
        }
        const auto scale = _mm512_set1_ps(iscale);
        for (int j = 0; j < 16; ++j) {
            const auto p = _mm512_mul_ps(scale, v[j]);
            // Reproduce ggml nearest_int's actual bias/mantissa arithmetic.
            const auto biased = _mm512_castps_si512(_mm512_add_ps(p, _mm512_set1_ps(12582912.0f)));
            const auto rounded = _mm512_sub_epi32(_mm512_and_si512(biased, _mm512_set1_epi32(0x007fffff)),
                                                  _mm512_set1_epi32(0x00400000));
            const auto q = _mm512_min_epi32(rounded, _mm512_set1_epi32(127));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(b.qs + 16 * j), _mm512_cvtepi32_epi8(q));
            b.bsums[j] = static_cast<int16_t>(_mm512_reduce_add_epi32(q));
        }
        b.d = 1.0f / iscale;
    }
}
QUANT_TARGET void quantize_q8_0_fast(const float * x, block_q8_0 * y, int width) {
    if (_mm_getcsr() & 0xe040) {
        std::memset(y, 0, std::size_t(width / 32) * sizeof(*y));
        quantize_row_q8_0_ref(x, y, width);
        return;
    }
    const auto half = _mm512_set1_ps(0.5f), one = _mm512_set1_ps(1.0f);
    for (int i = 0; i < width / 32; ++i, x += 32) {
        auto & b = y[i];
        const __m512 v[2] = {_mm512_loadu_ps(x), _mm512_loadu_ps(x + 16)};
        // Q8_0's scalar MAX has order-sensitive NaN behavior. Neither a vector
        // max nor an unordered-lane mask is equivalent; run the original block.
        if (quant_nonfinite(v[0]) || quant_nonfinite(v[1])) {
            std::memset(&b, 0, sizeof(b));
            quantize_row_q8_0_ref(x, &b, 32);
            continue;
        }
        const float amax = _mm512_reduce_max_ps(_mm512_max_ps(quant_abs(v[0]), quant_abs(v[1])));
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        if (!std::isfinite(id)) {
            std::memset(&b, 0, sizeof(b));
            quantize_row_q8_0_ref(x, &b, 32);
            continue;
        }
        b.d = _cvtss_sh(d, _MM_FROUND_TO_NEAREST_INT);
        const auto scale = _mm512_set1_ps(id);
        for (int j = 0; j < 2; ++j) {
            const auto p = _mm512_mul_ps(v[j], scale);
            const auto t = _mm512_roundscale_ps(p, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const auto step = _mm512_cmp_ps_mask(quant_abs(_mm512_sub_ps(p, t)), half, _CMP_GE_OQ);
            const auto signed_one = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(one),
                _mm512_and_si512(_mm512_castps_si512(p), _mm512_set1_epi32(int(0x80000000u)))));
            // roundf rounds halfway cases away from zero, not ties-to-even.
            const auto r = _mm512_mask_add_ps(t, step, t, signed_one);
            const auto q = _mm512_cvttps_epi32(r);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(b.qs + 16 * j), _mm512_cvtepi32_epi8(q));
        }
    }
}
VNNI_TARGET void dot32_vnni(const uint8_t * qs, const block_q8_K * const * inputs,
                            int block, int group, int n_rows, float scale, float * sums) {
    // Byte shuffles need only SSSE3/AVX512BW, not VBMI. Codebook + 128 is
    // unsigned; VPDPBUSD consumes unsigned weights and signed activations.
    const __m128i lut = _mm_setr_epi8(1, 24, 45, 63, 79, 93, 106, 118,
                                     char(129), char(141), char(153), char(166),
                                     char(181), char(197), char(217), char(241));
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i low = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
    const __m128i high = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    const __m256i w = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
    for (int t = 0; t < n_rows; ++t) {
        const auto & x = inputs[t][block];
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x.qs + group * 32));
        const __m256i products = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, a);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        const int correction = 128 * (int(x.bsums[2 * group]) + int(x.bsums[2 * group + 1]));
        sums[t] += (scale * x.d) * (_mm_cvtsi128_si32(s) - correction);
    }
}

VNNI_TARGET __m512i unpack_pair(const uint8_t * first, const uint8_t * second) {
    const __m128i lut = _mm_setr_epi8(1, 24, 45, 63, 79, 93, 106, 118,
                                     char(129), char(141), char(153), char(166),
                                     char(181), char(197), char(217), char(241));
    const __m256i packed = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(first))),
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(second)), 1);
    const __m512i nibbles = _mm512_and_si512(_mm512_inserti64x4(
        _mm512_castsi256_si512(packed), _mm256_srli_epi16(packed, 4), 1), _mm512_set1_epi8(15));
    const __m512i values = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(lut), nibbles);
    // [first low, second low, first high, second high] -> two contiguous groups.
    return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), values);
}
VNNI_TARGET __m512i signed_products(__m512i weights, __m512i x) {
    const __m512i zero = _mm512_setzero_si512();
    // Per-four-byte correction preserves -128; sign/abs negation would not.
    return _mm512_sub_epi32(_mm512_dpbusd_epi32(zero, weights, x),
                           _mm512_dpbusd_epi32(zero, _mm512_set1_epi8(char(128)), x));
}
VNNI_TARGET __m512i signed_products(__m512i weights, const int8_t * input) {
    return signed_products(weights, _mm512_loadu_si512(input));
}
VNNI_TARGET float half_unaligned(const uint8_t * bytes) {
    uint16_t half;
    std::memcpy(&half, bytes, sizeof(half));
    return _cvtsh_ss(half);
}
VNNI_TARGET float row_vnni(int type, const void * weights, int width, const block_q8_K * input) {
    const auto * bytes = static_cast<const uint8_t *>(weights);
    __m512 sums = _mm512_setzero_ps();
    for (int b = 0; b < width / 256; ++b) {
        const auto & x = input[b];
        if (type == GGML_TYPE_IQ4_XS) {
            const auto * block = bytes + std::size_t(b) * sizeof(block_iq4_xs);
            const int high = int(block[2]) | (int(block[3]) << 8);
            __m512i dots = _mm512_setzero_si512();
            for (int group = 0; group < 8; group += 2) {
                const int low = block[4 + group / 2];
                const int s0 = ((low & 15) | (((high >> (2 * group)) & 3) << 4)) - 32;
                const int s1 = ((low >> 4) | (((high >> (2 * (group + 1))) & 3) << 4)) - 32;
                const __m512i scale = _mm512_mask_blend_epi32(0xff00,
                    _mm512_set1_epi32(s0), _mm512_set1_epi32(s1));
                const __m512i w = unpack_pair(block + 8 + group * 16, block + 8 + (group + 1) * 16);
                dots = _mm512_add_epi32(dots, _mm512_mullo_epi32(
                    signed_products(w, x.qs + group * 32), scale));
            }
            // Integer subscales share one FP16/Q8_K scale for all 256 weights.
            const __m512 scale = _mm512_set1_ps(half_unaligned(block) * x.d);
            sums = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots), scale, sums);
        } else {
            for (int group = 0; group < 8; group += 2) {
                const auto * a = bytes + std::size_t(b * 8 + group) * sizeof(block_iq4_nl);
                const auto * c = a + sizeof(block_iq4_nl);
                const __m512 scale = _mm512_mask_blend_ps(0xff00,
                    _mm512_set1_ps(half_unaligned(a) * x.d), _mm512_set1_ps(half_unaligned(c) * x.d));
                const __m512i dots = signed_products(unpack_pair(a + 2, c + 2), x.qs + group * 32);
                sums = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots), scale, sums);
            }
        }
    }
    return _mm512_reduce_add_ps(sums);
}
// row_vnni for N activations at once: each 64-weight pair of groups is unpacked once for all of them, and the
// activation-only half of signed_products (-128 * x per 4 bytes) is precomputed per activation (neg) and used as the
// dpbusd accumulator. Same integer dots and float operation order as row_vnni: bitwise equal results.
template <int N>
VNNI_TARGET void rows_vnni_multi(int type, const uint8_t * bytes, std::size_t stride, int width,
                                 const block_q8_K * const * in, const __m512i * neg, int n_rows, float * out,
                                 int out_ld) {
    const int nb = width / 256;
    for (int r = 0; r < n_rows; ++r) {
        const uint8_t * row = bytes + std::size_t(r) * stride;
        __m512 sums[N];
        for (int t = 0; t < N; ++t) sums[t] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            if (type == GGML_TYPE_IQ4_XS) {
                const auto * block = row + std::size_t(b) * sizeof(block_iq4_xs);
                const int high = int(block[2]) | (int(block[3]) << 8);
                __m512i dots[N];
                for (int t = 0; t < N; ++t) dots[t] = _mm512_setzero_si512();
                for (int group = 0; group < 8; group += 2) {
                    const int low = block[4 + group / 2];
                    const int s0 = ((low & 15) | (((high >> (2 * group)) & 3) << 4)) - 32;
                    const int s1 = ((low >> 4) | (((high >> (2 * (group + 1))) & 3) << 4)) - 32;
                    const __m512i scale = _mm512_mask_blend_epi32(0xff00,
                        _mm512_set1_epi32(s0), _mm512_set1_epi32(s1));
                    const __m512i w = unpack_pair(block + 8 + group * 16, block + 8 + (group + 1) * 16);
                    for (int t = 0; t < N; ++t) {
                        const __m512i x = _mm512_loadu_si512(in[t][b].qs + group * 32);
                        const __m512i p = _mm512_dpbusd_epi32(neg[(t * nb + b) * 4 + group / 2], w, x);
                        dots[t] = _mm512_add_epi32(dots[t], _mm512_mullo_epi32(p, scale));
                    }
                }
                const float d = half_unaligned(block);
                for (int t = 0; t < N; ++t)
                    sums[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots[t]), _mm512_set1_ps(d * in[t][b].d), sums[t]);
            } else {
                for (int group = 0; group < 8; group += 2) {
                    const auto * a = row + std::size_t(b * 8 + group) * sizeof(block_iq4_nl);
                    const auto * c = a + sizeof(block_iq4_nl);
                    const float da = half_unaligned(a), dc = half_unaligned(c);
                    const __m512i w = unpack_pair(a + 2, c + 2);
                    for (int t = 0; t < N; ++t) {
                        const float xd = in[t][b].d;
                        const __m512 scale = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(da * xd), _mm512_set1_ps(dc * xd));
                        const __m512i x = _mm512_loadu_si512(in[t][b].qs + group * 32);
                        const __m512i p = _mm512_dpbusd_epi32(neg[(t * nb + b) * 4 + group / 2], w, x);
                        sums[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), scale, sums[t]);
                    }
                }
            }
        }
        for (int t = 0; t < N; ++t) out[std::size_t(t) * out_ld + r] = _mm512_reduce_add_ps(sums[t]);
    }
}
VNNI_TARGET void neg_correction(const block_q8_K * x, int nb, __m512i * neg) {
    const __m512i zero = _mm512_setzero_si512();
    for (int b = 0; b < nb; ++b)
        for (int q = 0; q < 4; ++q)
            neg[b * 4 + q] = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, _mm512_set1_epi8(char(128)),
                                                                         _mm512_loadu_si512(x[b].qs + q * 64)));
}
VNNI_TARGET float row_nl_q8_0_vnni(const void * weights, int width, const block_q8_0 * input) {
    const auto * bytes = static_cast<const uint8_t *>(weights);
    __m512 sums = _mm512_setzero_ps();
    const int groups = width / 32;
    for (int group = 0; group < groups; group += 2) {
        const auto * a = bytes + std::size_t(group) * sizeof(block_iq4_nl);
        const bool paired = group + 1 < groups;
        const auto * c = paired ? a + sizeof(block_iq4_nl) : a;
        const auto & x0 = input[group];
        const auto & x1 = input[paired ? group + 1 : group];
        const __m256i low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x0.qs));
        const __m256i high = paired ? _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x1.qs))
                                    : _mm256_setzero_si256();
        const __m512i x = _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
        const __m512 scale = _mm512_mask_blend_ps(0xff00,
            _mm512_set1_ps(half_unaligned(a) * _cvtsh_ss(x0.d)),
            _mm512_set1_ps(half_unaligned(c) * _cvtsh_ss(x1.d)));
        const __m512i dots = signed_products(unpack_pair(a + 2, c + 2), x);
        sums = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots), scale, sums);
    }
    return _mm512_reduce_add_ps(sums);
}

// AVX2 + FMA + F16C (no AVX512): the sign trick with the activation as the unsigned side, |x| x (w * sign x) on
// vpmaddubsw (|x| <= 128 as u8 keeps an activation byte of -128 exact, |w| <= 127: a pair <= 2 * 128 * 127, no
// saturation), vpmaddwd to int32: the integer dots of the VNNI kernels, summed in floats in their own order.
AVX2_TARGET __m256i nibbles_avx2(const uint8_t * qs) {
    // 16 bytes -> 32 signed codebook values: the low nibbles (values 0..15), then the high ones (16..31)
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codebook));
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qs));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i low = _mm_shuffle_epi8(lut, _mm_and_si128(packed, mask));
    const __m128i high = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}
AVX2_TARGET __m256i dot8_avx2(__m256i w, __m256i x) {
    // 8 int32 lanes, each the sum of 4 of the 32 products w * x
    const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}
AVX2_TARGET int hsum_epi32_avx2(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}
AVX2_TARGET float hsum_avx2(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
AVX2_TARGET float half_avx2(const uint8_t * bytes) {
    uint16_t half;
    std::memcpy(&half, bytes, sizeof(half));
    return _cvtsh_ss(half);
}
// dot32_scalar's float operations on the same integer dots: bitwise equal to it
AVX2_TARGET void dot32_avx2(const uint8_t * qs, const block_q8_K * const * inputs,
                            int block, int group, int n_rows, float scale, float * sums) {
    const __m256i w = nibbles_avx2(qs);
    for (int t = 0; t < n_rows; ++t) {
        const auto & x = inputs[t][block];
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x.qs + group * 32));
        sums[t] += (scale * x.d) * hsum_epi32_avx2(dot8_avx2(w, a));
    }
}
// rows x N activations, each group unpacked once for all of them. IQ4_XS: the 8 subscaled group dots of a block
// summed in int32 lanes (<= 8 * 4 * 128 * 127 * 32 < 2^24: exact as floats too), then one FMA per block.
template <int N>
AVX2_TARGET void rows_avx2_multi(int type, const uint8_t * bytes, std::size_t stride, int width,
                                 const block_q8_K * const * in, int n_rows, float * out, int out_ld) {
    const int nb = width / 256;
    for (int r = 0; r < n_rows; ++r) {
        const uint8_t * row = bytes + std::size_t(r) * stride;
        __m256 sums[N];
        for (int t = 0; t < N; ++t) sums[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            if (type == GGML_TYPE_IQ4_XS) {
                const auto * block = row + std::size_t(b) * sizeof(block_iq4_xs);
                const int high = int(block[2]) | (int(block[3]) << 8);
                __m256i dots[N];
                for (int t = 0; t < N; ++t) dots[t] = _mm256_setzero_si256();
                for (int group = 0; group < 8; ++group) {
                    const int low = block[4 + group / 2] >> (4 * (group & 1));
                    const __m256i scale = _mm256_set1_epi32(((low & 15) | (((high >> (2 * group)) & 3) << 4)) - 32);
                    const __m256i w = nibbles_avx2(block + 8 + group * 16);
                    for (int t = 0; t < N; ++t) {
                        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in[t][b].qs + group * 32));
                        dots[t] = _mm256_add_epi32(dots[t], _mm256_mullo_epi32(dot8_avx2(w, x), scale));
                    }
                }
                const float d = half_avx2(block);
                for (int t = 0; t < N; ++t)
                    sums[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots[t]), _mm256_set1_ps(d * in[t][b].d), sums[t]);
            } else {
                for (int group = 0; group < 8; ++group) {
                    const auto * a = row + std::size_t(b * 8 + group) * sizeof(block_iq4_nl);
                    const float da = half_avx2(a);
                    const __m256i w = nibbles_avx2(a + 2);
                    for (int t = 0; t < N; ++t) {
                        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(in[t][b].qs + group * 32));
                        sums[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot8_avx2(w, x)), _mm256_set1_ps(da * in[t][b].d),
                                                  sums[t]);
                    }
                }
            }
        }
        for (int t = 0; t < N; ++t) out[std::size_t(t) * out_ld + r] = hsum_avx2(sums[t]);
    }
}
AVX2_TARGET float row_nl_q8_0_avx2(const void * weights, int width, const block_q8_0 * input) {
    const auto * bytes = static_cast<const uint8_t *>(weights);
    __m256 sums = _mm256_setzero_ps();
    for (int group = 0; group < width / 32; ++group) {
        const auto * a = bytes + std::size_t(group) * sizeof(block_iq4_nl);
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input[group].qs));
        sums = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot8_avx2(nibbles_avx2(a + 2), x)),
                               _mm256_set1_ps(half_avx2(a) * _cvtsh_ss(input[group].d)), sums);
    }
    return hsum_avx2(sums);
}
void rows_avx2(int type, const uint8_t * bytes, std::size_t stride, int width, const block_q8_K * const * in,
               int n_act, int n_rows, float * out) {
    switch (n_act) {
        case 1: rows_avx2_multi<1>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 2: rows_avx2_multi<2>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 3: rows_avx2_multi<3>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 4: rows_avx2_multi<4>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 5: rows_avx2_multi<5>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 6: rows_avx2_multi<6>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        case 7: rows_avx2_multi<7>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
        default: rows_avx2_multi<8>(type, bytes, stride, width, in, n_rows, out, n_rows); break;
    }
}
#endif
float row_nl_q8_0_scalar(const void * weights, int width, const block_q8_0 * input) {
    const auto * bytes = static_cast<const uint8_t *>(weights);
    float sum = 0;
    for (int group = 0; group < width / 32; ++group) {
        const auto * block = bytes + std::size_t(group) * sizeof(block_iq4_nl);
        ggml_fp16_t half;
        std::memcpy(&half, block, sizeof(half));
        int dot = 0;
        for (int j = 0; j < 16; ++j) {
            dot += int(codebook[block[2 + j] & 15]) * input[group].qs[j];
            dot += int(codebook[block[2 + j] >> 4]) * input[group].qs[j + 16];
        }
        sum += (ggml_fp16_to_fp32(half) * ggml_fp16_to_fp32(input[group].d)) * dot;
    }
    return sum;
}
using Dot32 = void (*)(const uint8_t *, const block_q8_K * const *, int, int, int, float, float *);
Dot32 select(IQ4VNNIPath path) {
    if (path == IQ4VNNIPath::scalar) return dot32_scalar;
    const bool vnni = iq4_vnni_available(), avx2 = iq4_int8_available();
    if (path == IQ4VNNIPath::vnni && !vnni)
        throw std::invalid_argument("AVX512 VNNI/VL/F16C unavailable");
    if (path == IQ4VNNIPath::avx2 && !avx2)
        throw std::invalid_argument("AVX2/FMA/F16C unavailable");
#ifdef STRATA_VNNI
    if (path == IQ4VNNIPath::avx2) return dot32_avx2;
    if (vnni) return dot32_vnni;
    if (avx2) return dot32_avx2;
#endif
    return dot32_scalar;
}
void row_dots(int type, const void * weights, int width, const block_q8_K * const * inputs,
              int n_rows, float * sums, Dot32 dot) {
    std::fill_n(sums, n_rows, 0.0f);
    const auto * bytes = static_cast<const unsigned char *>(weights);
    for (int b = 0; b < width / 256; ++b) {
        // memcpy permits mapped or deliberately unaligned weight pointers.
        block_iq4_xs xs;
        if (type == GGML_TYPE_IQ4_XS) std::memcpy(&xs, bytes + b * sizeof(xs), sizeof(xs));
        for (int group = 0; group < 8; ++group) {
            float scale;
            const uint8_t * qs;
            block_iq4_nl nl;
            if (type == GGML_TYPE_IQ4_XS) {
                const int sub = ((xs.scales_l[group / 2] >> (4 * (group % 2))) & 15)
                              | (((xs.scales_h >> (2 * group)) & 3) << 4);
                scale = ggml_fp16_to_fp32(xs.d) * (sub - 32);
                qs = xs.qs + group * 16;
            } else {
                std::memcpy(&nl, bytes + (b * 8 + group) * sizeof(nl), sizeof(nl));
                scale = ggml_fp16_to_fp32(nl.d);
                qs = nl.qs;
            }
            dot(qs, inputs, b, group, n_rows, scale, sums);
        }
    }
}
float row_dot(int type, const void * weights, int width, const block_q8_K * x, Dot32 dot) {
    float sum;
    row_dots(type, weights, width, &x, 1, &sum, dot);
    return sum;
}
} // namespace

bool iq4_vnni_available() {
#ifdef STRATA_VNNI
    return cpu_isa() == CpuIsa::vnni;
#else
    return false;
#endif
}
bool iq4_int8_available() {
#ifdef STRATA_VNNI
    return cpu_isa() >= CpuIsa::avx2;
#else
    return false;
#endif
}
std::size_t iq4_vnni_q8_bytes(int width) {
    width_check(width);
    return std::size_t(width / 256) * sizeof(block_q8_K);
}
std::size_t iq4_vnni_row_bytes(int type, int width) {
    width_check(width);
    if (type == GGML_TYPE_IQ4_XS) return std::size_t(width / 256) * sizeof(block_iq4_xs);
    if (type == GGML_TYPE_IQ4_NL) return std::size_t(width / 32) * sizeof(block_iq4_nl);
    throw std::invalid_argument("VNNI experts require IQ4_XS/IQ4_NL");
}
void iq4_vnni_quantize(const float * x, int width, void * q8) {
    const auto bytes = iq4_vnni_q8_bytes(width);
    if (!x || !q8 || reinterpret_cast<uintptr_t>(q8) % alignof(block_q8_K))
        throw std::invalid_argument("invalid Q8_K quantization pointers/alignment");
#ifdef STRATA_VNNI
    if (iq4_vnni_available()) {
        quantize_q8_K_fast(x, static_cast<block_q8_K *>(q8), width);
        return;
    }
#endif
    std::memset(q8, 0, bytes);
    quantize_row_q8_K_ref(x, static_cast<block_q8_K *>(q8), width);
}
float iq4_vnni_row(int type, const void * weights, int width, const void * q8, IQ4VNNIPath path) {
    iq4_vnni_row_bytes(type, width);
    if (!weights || !q8 || reinterpret_cast<uintptr_t>(q8) % alignof(block_q8_K))
        throw std::invalid_argument("invalid VNNI row pointers/alignment");
    const auto dot = select(path);
    const auto * input = static_cast<const block_q8_K *>(q8);
#ifdef STRATA_VNNI
    if (dot == dot32_vnni) return row_vnni(type, weights, width, input);
    if (dot == dot32_avx2) {
        float out;
        rows_avx2(type, static_cast<const uint8_t *>(weights), 0, width, &input, 1, 1, &out);
        return out;
    }
#endif
    return row_dot(type, weights, width, input, dot);
}
void iq4_vnni_rows(int type, const void * weights, int width, const void * q8,
                    int n_rows, float * out, IQ4VNNIPath path) {
    const auto stride = iq4_vnni_row_bytes(type, width);
    if (n_rows < 1 || !weights || !q8 || !out
        || reinterpret_cast<uintptr_t>(q8) % alignof(block_q8_K))
        throw std::invalid_argument("invalid VNNI consecutive rows inputs");
    const auto dot = select(path);
    const auto * bytes = static_cast<const unsigned char *>(weights);
    const auto * input = static_cast<const block_q8_K *>(q8);
#ifdef STRATA_VNNI
    if (dot == dot32_vnni) {
        if (width / 256 <= 16) {   // one activation through the multi form: its correction is computed once per call
            iq4_vnni_rows_multi(type, weights, width, &q8, 1, n_rows, out, path);
            return;
        }
        for (int r = 0; r < n_rows; ++r)
            out[r] = row_vnni(type, bytes + std::size_t(r) * stride, width, input);
        return;
    }
    if (dot == dot32_avx2) {
        rows_avx2(type, bytes, stride, width, &input, 1, n_rows, out);
        return;
    }
#endif
    for (int r = 0; r < n_rows; ++r)
        out[r] = row_dot(type, bytes + std::size_t(r) * stride, width, input, dot);
}
void iq4_vnni_rows_multi(int type, const void * weights, int width, const void * const * q8, int n_act,
                         int n_rows, float * out, IQ4VNNIPath path) {
    const auto stride = iq4_vnni_row_bytes(type, width);
    if (n_rows < 1 || n_act < 1 || n_act > 8 || !weights || !q8 || !out)
        throw std::invalid_argument("invalid VNNI multi-activation rows inputs");
    const block_q8_K * in[8];
    for (int t = 0; t < n_act; ++t) {
        if (!q8[t] || reinterpret_cast<uintptr_t>(q8[t]) % alignof(block_q8_K))
            throw std::invalid_argument("invalid VNNI multi-activation rows inputs");
        in[t] = static_cast<const block_q8_K *>(q8[t]);
    }
    const auto dot = select(path);
#ifdef STRATA_VNNI
    constexpr int kMaxBlocks = 16;   // widths up to 4096 (n_embd, ffn_expert)
    if (dot == dot32_vnni && width / 256 <= kMaxBlocks) {
        const int nb = width / 256;
        alignas(64) thread_local __m512i neg_buf[8 * kMaxBlocks * 4];
        __m512i * neg = neg_buf;
        for (int t = 0; t < n_act; ++t) neg_correction(in[t], nb, neg + std::size_t(t) * nb * 4);
        const auto * bytes = static_cast<const uint8_t *>(weights);
        switch (n_act) {
            case 1: rows_vnni_multi<1>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 2: rows_vnni_multi<2>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 3: rows_vnni_multi<3>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 4: rows_vnni_multi<4>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 5: rows_vnni_multi<5>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 6: rows_vnni_multi<6>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            case 7: rows_vnni_multi<7>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
            default: rows_vnni_multi<8>(type, bytes, stride, width, in, neg, n_rows, out, n_rows); break;
        }
        return;
    }
    if (dot == dot32_avx2) {
        rows_avx2(type, static_cast<const uint8_t *>(weights), stride, width, in, n_act, n_rows, out);
        return;
    }
#endif
    for (int t = 0; t < n_act; ++t) iq4_vnni_rows(type, weights, width, q8[t], n_rows, out + std::size_t(t) * n_rows, path);
}
std::size_t iq4_vnni_q8_0_bytes(int width) {
    if (width <= 0 || width % 32)
        throw std::invalid_argument("Q8_0 width must be a positive multiple of 32");
    return std::size_t(width / 32) * sizeof(block_q8_0);
}
void iq4_vnni_quantize_q8_0(const float * x, int width, void * q8_0) {
    const auto bytes = iq4_vnni_q8_0_bytes(width);
    if (!x || !q8_0 || reinterpret_cast<uintptr_t>(q8_0) % alignof(block_q8_0))
        throw std::invalid_argument("invalid Q8_0 quantization pointers/alignment");
#ifdef STRATA_VNNI
    if (iq4_vnni_available()) {
        quantize_q8_0_fast(x, static_cast<block_q8_0 *>(q8_0), width);
        return;
    }
#endif
    std::memset(q8_0, 0, bytes);
    quantize_row_q8_0_ref(x, static_cast<block_q8_0 *>(q8_0), width);
}
void iq4_vnni_nl_q8_0_rows(const void * weights, int width, const void * q8_0,
                           int n_rows, float * out, IQ4VNNIPath path) {
    iq4_vnni_q8_0_bytes(width);
    if (n_rows < 1 || !weights || !q8_0 || !out
        || reinterpret_cast<uintptr_t>(q8_0) % alignof(block_q8_0))
        throw std::invalid_argument("invalid IQ4_NL/Q8_0 consecutive rows inputs");
    const auto stride = std::size_t(width / 32) * sizeof(block_iq4_nl);
    const auto * bytes = static_cast<const uint8_t *>(weights);
    const auto * input = static_cast<const block_q8_0 *>(q8_0);
    const auto dot = select(path);
#ifdef STRATA_VNNI
    if (dot == dot32_vnni) {
        for (int r = 0; r < n_rows; ++r)
            out[r] = row_nl_q8_0_vnni(bytes + std::size_t(r) * stride, width, input);
        return;
    }
    if (dot == dot32_avx2) {
        for (int r = 0; r < n_rows; ++r)
            out[r] = row_nl_q8_0_avx2(bytes + std::size_t(r) * stride, width, input);
        return;
    }
#endif
    for (int r = 0; r < n_rows; ++r)
        out[r] = row_nl_q8_0_scalar(bytes + std::size_t(r) * stride, width, input);
}
std::size_t vnni_expert_activation_bytes(const Geometry & g, int T) {
    shape_check(g, T);
    return std::size_t(T) * iq4_vnni_q8_bytes(g.n_embd);
}
std::size_t vnni_expert_scratch_bytes(const Geometry & g, int n_rows) {
    shape_check(g, n_rows);
    return std::size_t(n_rows) * (2 * std::size_t(g.ffn_expert) * sizeof(float)
                               + iq4_vnni_q8_bytes(g.ffn_expert));
}
void vnni_expert_prepare(const Geometry & g, const float * x, int T, void * act) {
    vnni_expert_activation_bytes(g, T);
    if (!x || !act) throw std::invalid_argument("null VNNI preparation input");
    const auto stride = iq4_vnni_q8_bytes(g.n_embd);
    for (int t = 0; t < T; ++t)
        iq4_vnni_quantize(x + std::size_t(t) * g.n_embd, g.n_embd,
                          static_cast<unsigned char *>(act) + t * stride);
}
void vnni_expert_eval(const Geometry & g, const ExpertRef & e, const void * act, int T,
                      int n_rows, const int * tokens, float * out, void * scratch, IQ4VNNIPath path) {
    shape_check(g, T);
    vnni_expert_scratch_bytes(g, n_rows);
    if (e.layout != 0)
        throw std::invalid_argument("VNNI experts: dispatch R8 weights before legacy evaluation");
    if (!act || !out || !scratch || !e.gate_up || !e.down
        || e.type_gate_up != GGML_TYPE_IQ4_XS || e.type_down != GGML_TYPE_IQ4_NL
        || (e.type_up != -1 && e.type_up != e.type_gate_up)
        || reinterpret_cast<uintptr_t>(act) % alignof(block_q8_K)
        || reinterpret_cast<uintptr_t>(scratch) % alignof(block_q8_K))
        throw std::invalid_argument("invalid VNNI expert inputs");
    for (int t = 0; t < n_rows; ++t) {
        const int token = tokens ? tokens[t] : t;
        if (token < 0 || token >= T) throw std::invalid_argument("VNNI selected token out of range");
    }
    const auto dot = select(path);
    const auto gu_stride = iq4_vnni_row_bytes(e.type_gate_up, g.n_embd);
    const auto down_stride = iq4_vnni_row_bytes(e.type_down, g.ffn_expert);
    const auto * gate_weights = static_cast<const unsigned char *>(e.gate_up);
    const auto * up_weights = e.up ? static_cast<const unsigned char *>(e.up)
                                  : gate_weights + gu_stride * g.ffn_expert;
    float * gate = static_cast<float *>(scratch);
    float * mid = gate + std::size_t(n_rows) * g.ffn_expert;
    auto * qmid = reinterpret_cast<block_q8_K *>(mid + std::size_t(n_rows) * g.ffn_expert);
    const auto * prepared = static_cast<const block_q8_K *>(act);
    const int xb = g.n_embd / 256, mb = g.ffn_expert / 256;
    const block_q8_K * inputs[8];
    float gate_row[8], up_row[8], down_row[8];
    for (int t = 0; t < n_rows; ++t) inputs[t] = prepared + (tokens ? tokens[t] : t) * xb;
    // Decode the nibble lookup once per group, reuse across selected tokens.
    for (int r = 0; r < g.ffn_expert; ++r) {
        row_dots(e.type_gate_up, gate_weights + r * gu_stride, g.n_embd, inputs, n_rows, gate_row, dot);
        row_dots(e.type_gate_up, up_weights + r * gu_stride, g.n_embd, inputs, n_rows, up_row, dot);
        for (int t = 0; t < n_rows; ++t) {
            gate[t * g.ffn_expert + r] = gate_row[t];
            const float gc = std::min(gate_row[t], g.swiglu_clamp);
            mid[t * g.ffn_expert + r] = (gc / (1.0f + std::exp(-gc)))
                                     * std::max(-g.swiglu_clamp, std::min(up_row[t], g.swiglu_clamp));
        }
    }
    for (int t = 0; t < n_rows; ++t) {
        iq4_vnni_quantize(mid + t * g.ffn_expert, g.ffn_expert, qmid + t * mb);
        inputs[t] = qmid + t * mb;
    }
    for (int r = 0; r < g.n_embd; ++r) {
        row_dots(e.type_down, static_cast<const unsigned char *>(e.down) + r * down_stride,
                 g.ffn_expert, inputs, n_rows, down_row, dot);
        for (int t = 0; t < n_rows; ++t) out[t * g.n_embd + r] = down_row[t];
    }
}
} // namespace strata::glm
