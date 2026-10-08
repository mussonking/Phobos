#include "r8.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define STRATA_R8_VNNI 1
#endif

namespace strata::glm {
namespace {
constexpr int kXsBlock = kR8XSBlock, kNlBlock = kR8NLBlock;
constexpr int kXsTile = kR8XSTile, kNlTile = kR8NLTile;
constexpr int kQ8KBlock = 292, kQ80Block = 34;
constexpr int8_t codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1, 13, 25, 38, 53, 69, 89, 113};
template <class T> T load(const void * p) {
    T v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("R8 byte extent overflow");
    return a * b;
}
void disjoint(const void * a, std::size_t na, const void * b, std::size_t nb) {
    const auto aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    if (!a || !b || na > UINTPTR_MAX - aa || nb > UINTPTR_MAX - bb
        || (aa < bb + nb && bb < aa + na))
        throw std::invalid_argument("R8 buffers must be nonnull, bounded and disjoint");
}
float half(const void * p) { return ggml_fp16_to_fp32(load<ggml_fp16_t>(p)); }
int64_t corr_of(const int8_t * x) {
    int s[2] = {};
    for (int k = 0; k < 32; ++k) s[(k / 4) % 2] += x[k];
    const uint64_t bits = uint32_t(-128 * s[0]) | (uint64_t(uint32_t(-128 * s[1])) << 32);
    return load<int64_t>(&bits);
}
int64_t correction(const R8Act & a, int group) {
    return load<int64_t>(reinterpret_cast<const uint8_t *>(a.corr)
                        + std::size_t(group) * a.cstride * sizeof(int64_t));
}
float scale_of(const R8Act & a, int block) {
    return load<float>(reinterpret_cast<const uint8_t *>(a.d)
                       + std::size_t(block) * a.dstride * sizeof(float));
}
int tile_scale(const uint8_t * tile, int group, int row) {
    const int j = 8 * group + row;
    const int lo = (tile[16 + j % 32] >> (4 * (j / 32))) & 15;
    const int hi = (tile[48 + j % 16] >> (2 * (j / 16))) & 3;
    return (lo | (hi << 4)) - 32;
}
int tile_weight(const uint8_t * q, int row, int k) {
    const int within = k % 16;
    const int lane = 2 * row + (within % 8) / 4;
    const int packed = q[64 * (k / 16) + 4 * lane + within % 4];
    return (packed >> (4 * (within / 8))) & 15;
}
inline int gguf_w(const uint8_t * q, int k) { return k < 16 ? q[k] & 15 : q[k - 16] >> 4; }
// the 64-B qs half h of a group (32 weights w[0..31]) in R8 lane order; rows' groups start at q[r]
inline void put_halves(const uint8_t * const q[8], uint8_t * out /* 128 B */) {
    for (int h = 0; h < 2; ++h)
        for (int r = 0; r < 8; ++r)
            for (int s = 0; s < 2; ++s)
                for (int i = 0; i < 4; ++i) {
                    const int k = 16 * h + 4 * s + i;
                    out[64 * h + 4 * (2 * r + s) + i] = uint8_t(gguf_w(q[r], k) | gguf_w(q[r], k + 8) << 4);
                }
}
inline void get_halves(const uint8_t * in /* 128 B */, uint8_t * const q[8]) {
    int8_t w[8][32];
    for (int h = 0; h < 2; ++h)
        for (int r = 0; r < 8; ++r)
            for (int s = 0; s < 2; ++s)
                for (int i = 0; i < 4; ++i) {
                    const int k = 16 * h + 4 * s + i;
                    const uint8_t b = in[64 * h + 4 * (2 * r + s) + i];
                    w[r][k] = int8_t(b & 15);
                    w[r][k + 8] = int8_t(b >> 4);
                }
    for (int r = 0; r < 8; ++r)
        for (int j = 0; j < 16; ++j) q[r][j] = uint8_t(w[r][j] | w[r][j + 16] << 4);
}
// one row group (8 GGUF rows of `width`) <-> R8; src and dst do not overlap
inline void repack_xs(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 256;
    const size_t row = size_t(nb) * kXsBlock;
    for (int b = 0; b < nb; ++b) {
        uint8_t * t = dst + size_t(b) * kXsTile;
        const uint8_t * blk[8];
        for (int r = 0; r < 8; ++r) blk[r] = src + r * row + size_t(b) * kXsBlock;
        uint8_t S[64];
        for (int r = 0; r < 8; ++r) {
            memcpy(t + 2 * r, blk[r], 2);
            const int sh = blk[r][2] | blk[r][3] << 8;
            for (int g = 0; g < 8; ++g)
                S[8 * g + r] = uint8_t(((blk[r][4 + g / 2] >> (4 * (g % 2))) & 15) | (((sh >> (2 * g)) & 3) << 4));
        }
        for (int j = 0; j < 32; ++j) t[16 + j] = uint8_t((S[j] & 15) | (S[j + 32] & 15) << 4);
        for (int j = 0; j < 16; ++j)
            t[48 + j] = uint8_t((S[j] >> 4) | (S[j + 16] >> 4) << 2 | (S[j + 32] >> 4) << 4 | (S[j + 48] >> 4) << 6);
        for (int g = 0; g < 8; ++g) {
            const uint8_t * q[8];
            for (int r = 0; r < 8; ++r) q[r] = blk[r] + 8 + 16 * g;
            put_halves(q, t + 64 + 128 * g);
        }
    }
}
inline void unpack_xs(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 256;
    const size_t row = size_t(nb) * kXsBlock;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * t = src + size_t(b) * kXsTile;
        uint8_t * blk[8];
        for (int r = 0; r < 8; ++r) blk[r] = dst + r * row + size_t(b) * kXsBlock;
        uint8_t S[64];
        for (int j = 0; j < 32; ++j) { S[j] = t[16 + j] & 15; S[j + 32] = t[16 + j] >> 4; }
        for (int j = 0; j < 16; ++j)
            for (int p = 0; p < 4; ++p) S[j + 16 * p] |= uint8_t(((t[48 + j] >> (2 * p)) & 3) << 4);
        for (int r = 0; r < 8; ++r) {
            memcpy(blk[r], t + 2 * r, 2);
            int sh = 0;
            for (int g = 0; g < 8; ++g) sh |= (S[8 * g + r] >> 4) << (2 * g);
            blk[r][2] = uint8_t(sh); blk[r][3] = uint8_t(sh >> 8);
            for (int k = 0; k < 4; ++k) blk[r][4 + k] = uint8_t((S[8 * (2 * k) + r] & 15) | (S[8 * (2 * k + 1) + r] & 15) << 4);
        }
        for (int g = 0; g < 8; ++g) {
            uint8_t * q[8];
            for (int r = 0; r < 8; ++r) q[r] = blk[r] + 8 + 16 * g;
            get_halves(t + 64 + 128 * g, q);
        }
    }
}
inline void repack_nl(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 32;
    const size_t row = size_t(nb) * kNlBlock;
    for (int b = 0; b < nb; ++b) {
        uint8_t * t = dst + size_t(b) * kNlTile;
        const uint8_t * q[8];
        for (int r = 0; r < 8; ++r) {
            const uint8_t * blk = src + r * row + size_t(b) * kNlBlock;
            memcpy(t + 2 * r, blk, 2);
            q[r] = blk + 2;
        }
        put_halves(q, t + 16);
    }
}
inline void unpack_nl(const uint8_t * src, uint8_t * dst, int width) {
    const int nb = width / 32;
    const size_t row = size_t(nb) * kNlBlock;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * t = src + size_t(b) * kNlTile;
        uint8_t * q[8];
        for (int r = 0; r < 8; ++r) {
            uint8_t * blk = dst + r * row + size_t(b) * kNlBlock;
            memcpy(blk, t + 2 * r, 2);
            q[r] = blk + 2;
        }
        get_halves(t + 16, q);
    }
}


void rows_scalar(bool xs, const uint8_t * w, int width, const R8Act * act,
                 int n_act, int n_rows, float * out) {
    const int nb = width / (xs ? 256 : 32), tile_bytes = xs ? kXsTile : kNlTile;
    for (int rg = 0; rg < n_rows / 8; ++rg) {
        for (int t = 0; t < n_act; ++t) {
            for (int r = 0; r < 8; ++r) {
                float acc[2] = {};
                for (int b = 0; b < nb; ++b) {
                    const uint8_t * tile = w + (std::size_t(rg) * nb + b) * tile_bytes;
                    float bacc[2] = {};
                    for (int g = 0; g < (xs ? 8 : 1); ++g) {
                        const int group = xs ? 8 * b + g : b;
                        const int ab = group / act[t].gpb;
                        const auto * x = act[t].qs + std::size_t(ab) * act[t].bstride
                                        + 32 * (group % act[t].gpb);
                        const uint8_t * q = tile + (xs ? 64 + 128 * g : 16);
                        int dot[2] = {};
                        for (int k = 0; k < 32; ++k)
                            dot[(k / 4) % 2] += int(codebook[tile_weight(q, r, k)]) * x[k];
                        const float wd = half(tile + 2 * r);
                        const float scale = xs ? wd * tile_scale(tile, g, r) : wd * scale_of(act[t], ab);
                        for (int s = 0; s < 2; ++s) {
                            if (xs) bacc[s] = std::fma(float(dot[s]), scale, bacc[s]);
                            else acc[s] = std::fma(float(dot[s]), scale, acc[s]);
                        }
                    }
                    if (xs)
                        for (int s = 0; s < 2; ++s)
                            acc[s] = std::fma(bacc[s], scale_of(act[t], b), acc[s]);
                }
                out[std::size_t(t) * n_rows + rg * 8 + r] = acc[0] + acc[1];
            }
        }
    }
}
#ifdef STRATA_R8_VNNI
#define R8_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c")))
R8_TARGET inline __m512i lut_u() {
    // codebook + 128 (unsigned weights for VPDPBUSD), in each 128-bit lane
    return _mm512_broadcast_i32x4(_mm_setr_epi8(1, 24, 45, 63, 79, 93, 106, 118, char(129), char(141), char(153),
                                                char(166), char(181), char(197), char(217), char(241)));
}
// lanes (r, s) -> row sums: out[r] = lane 2r + lane 2r+1
R8_TARGET inline void store_rows(__m512 acc, float * out) {
    const __m512 pair = _mm512_add_ps(acc, _mm512_permute_ps(acc, 0xB1));
    _mm256_storeu_ps(out, _mm512_castps512_ps256(_mm512_maskz_compress_ps(0x5555, pair)));
}
R8_TARGET inline __m512 dup8(__m256 v) {
    return _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7), _mm512_castps256_ps512(v));
}
// IQ4_XS R8 rows x N activations: n_rows (multiple of 8) rows from row group pointer w; out[t * ld + r]
template <int N>
R8_TARGET void xs_rows(const uint8_t * w, int width, const R8Act * act, int n_rows, float * out, int ld) {
    const int nb = width / 256;
    const __m512i lut = lut_u(), m4 = _mm512_set1_epi8(15);
    for (int rg = 0; rg < n_rows / 8; ++rg) {
        __m512 acc[N];
        for (int t = 0; t < N; ++t) acc[t] = _mm512_setzero_ps();
        const uint8_t * tile = w + size_t(rg) * nb * kXsTile;
        for (int b = 0; b < nb; ++b, tile += kXsTile) {
            const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(tile)));
            // the 64 sub-block scales S[8g + r] - 32, as int8
            const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tile + 16));
            const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tile + 48));
            const __m128i m3 = _mm_set1_epi8(3);
            const __m256i hA = _mm256_setr_m128i(_mm_and_si128(hi, m3), _mm_and_si128(_mm_srli_epi16(hi, 2), m3));
            const __m256i hB = _mm256_setr_m128i(_mm_and_si128(_mm_srli_epi16(hi, 4), m3), _mm_and_si128(_mm_srli_epi16(hi, 6), m3));
            const __m256i m4y = _mm256_set1_epi8(15), b32 = _mm256_set1_epi8(32);
            alignas(32) int8_t S[64];
            _mm256_store_si256(reinterpret_cast<__m256i *>(S),
                _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(lo, m4y), _mm256_slli_epi16(hA, 4)), b32));
            _mm256_store_si256(reinterpret_cast<__m256i *>(S + 32),
                _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 4), m4y), _mm256_slli_epi16(hB, 4)), b32));
            __m512 bacc[N];
            for (int t = 0; t < N; ++t) bacc[t] = _mm512_setzero_ps();
            const uint8_t * qs = tile + 64;
            for (int g = 0; g < 8; ++g, qs += 128) {
                const __m256 sg = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(S + 8 * g)))), d8);
                const __m512 scale = dup8(sg);
                const __m512i p0 = _mm512_loadu_si512(qs), p1 = _mm512_loadu_si512(qs + 64);
                const __m512i w0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, m4));
                const __m512i w1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), m4));
                const __m512i w2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, m4));
                const __m512i w3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), m4));
                for (int t = 0; t < N; ++t) {
                    const int8_t * x = act[t].qs + size_t(b) * act[t].bstride + 32 * g;
                    __m512i d = _mm512_set1_epi64(correction(act[t], b * 8 + g));
                    d = _mm512_dpbusd_epi32(d, w0, _mm512_set1_epi64(load<int64_t>(x)));
                    d = _mm512_dpbusd_epi32(d, w1, _mm512_set1_epi64(load<int64_t>(x + 8)));
                    d = _mm512_dpbusd_epi32(d, w2, _mm512_set1_epi64(load<int64_t>(x + 16)));
                    d = _mm512_dpbusd_epi32(d, w3, _mm512_set1_epi64(load<int64_t>(x + 24)));
                    bacc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), scale, bacc[t]);
                }
            }
            for (int t = 0; t < N; ++t) acc[t] = _mm512_fmadd_ps(bacc[t], _mm512_set1_ps(scale_of(act[t], b)), acc[t]);
        }
        for (int t = 0; t < N; ++t) store_rows(acc[t], out + size_t(t) * ld + rg * 8);
    }
}
// IQ4_NL R8 rows x N activations: Q8_0 (GPB 1) or Q8_K (GPB 8 groups per activation block)
template <int N, int GPB>
R8_TARGET void nl_rows(const uint8_t * w, int width, const R8Act * act, int n_rows, float * out, int ld) {
    const int nb = width / 32;
    const __m512i lut = lut_u(), m4 = _mm512_set1_epi8(15);
    for (int rg = 0; rg < n_rows / 8; ++rg) {
        __m512 acc[N];
        for (int t = 0; t < N; ++t) acc[t] = _mm512_setzero_ps();
        const uint8_t * tile = w + size_t(rg) * nb * kNlTile;
        for (int b = 0; b < nb; ++b, tile += kNlTile) {
            const __m512 dd = dup8(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(tile))));
            const __m512i p0 = _mm512_loadu_si512(tile + 16), p1 = _mm512_loadu_si512(tile + 80);
            const __m512i w0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, m4));
            const __m512i w1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), m4));
            const __m512i w2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, m4));
            const __m512i w3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), m4));
            for (int t = 0; t < N; ++t) {
                const int ab = b / GPB;
                const int8_t * x = act[t].qs + size_t(ab) * act[t].bstride + 32 * (b % GPB);
                __m512i d = _mm512_set1_epi64(correction(act[t], b));
                d = _mm512_dpbusd_epi32(d, w0, _mm512_set1_epi64(load<int64_t>(x)));
                d = _mm512_dpbusd_epi32(d, w1, _mm512_set1_epi64(load<int64_t>(x + 8)));
                d = _mm512_dpbusd_epi32(d, w2, _mm512_set1_epi64(load<int64_t>(x + 16)));
                d = _mm512_dpbusd_epi32(d, w3, _mm512_set1_epi64(load<int64_t>(x + 24)));
                acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_mul_ps(dd, _mm512_set1_ps(scale_of(act[t], ab))), acc[t]);
            }
        }
        for (int t = 0; t < N; ++t) store_rows(acc[t], out + size_t(t) * ld + rg * 8);
    }
}

#define R8_CASE(N) case N: { \
    if (xs) { xs_rows<N>(w, width, act, n_rows, out, n_rows); } \
    else if (act[0].gpb == 1) { nl_rows<N, 1>(w, width, act, n_rows, out, n_rows); } \
    else { nl_rows<N, 8>(w, width, act, n_rows, out, n_rows); } \
    break; }
void rows_vector(bool xs, const uint8_t * w, int width, const R8Act * act,
                 int n_act, int n_rows, float * out) {
    switch (n_act) {
        R8_CASE(1); R8_CASE(2); R8_CASE(3); R8_CASE(4);
        R8_CASE(5); R8_CASE(6); R8_CASE(7); R8_CASE(8);
    }
}
#undef R8_CASE
#undef R8_TARGET
#endif
void rows(bool xs, const void * weights, int width, const R8Act * act,
          int n_act, int n_rows, float * out, IQ4VNNIPath path) {
    const auto stride = r8_row_bytes(xs, width);
    if (!weights || !act || !out || n_act < 1 || n_act > 8 || n_rows < 8 || n_rows % 8)
        throw std::invalid_argument("invalid R8 row/activation count or pointers");
    if (path != IQ4VNNIPath::automatic && path != IQ4VNNIPath::scalar && path != IQ4VNNIPath::vnni)
        throw std::invalid_argument("invalid R8 execution path");
    const int gpb = act[0].gpb;
    if ((xs && gpb != 8) || (!xs && gpb != 1 && gpb != 8) || width % (32 * gpb))
        throw std::invalid_argument("R8 activation block geometry mismatch");
    const auto out_bytes = product(product(std::size_t(n_act), std::size_t(n_rows)), sizeof(float));
    disjoint(weights, product(std::size_t(n_rows), stride), out, out_bytes);
    disjoint(act, product(std::size_t(n_act), sizeof(R8Act)), out, out_bytes);
    for (int t = 0; t < n_act; ++t) {
        const auto & a = act[t];
        if (!a.qs || !a.corr || !a.d || a.gpb != gpb || a.bstride < 32 * gpb
            || a.cstride < 1 || a.dstride < 1)
            throw std::invalid_argument("invalid R8 activation geometry/strides");
        const auto qbytes = product(std::size_t(width / (32 * gpb) - 1), a.bstride) + 32 * gpb;
        const auto cbytes = product(product(std::size_t(width / 32 - 1), a.cstride) + 1, sizeof(int64_t));
        const auto dbytes = product(product(std::size_t(width / (32 * gpb) - 1), a.dstride) + 1, sizeof(float));
        disjoint(a.qs, qbytes, out, out_bytes);
        disjoint(a.corr, cbytes, out, out_bytes);
        disjoint(a.d, dbytes, out, out_bytes);
    }
    const bool vector = path != IQ4VNNIPath::scalar && r8_available();
    if (path == IQ4VNNIPath::vnni && !vector)
        throw std::invalid_argument("R8 AVX512 VNNI/VL/F16C unavailable");
#ifdef STRATA_R8_VNNI
    if (vector) {
        rows_vector(xs, static_cast<const uint8_t *>(weights), width, act, n_act, n_rows, out);
        return;
    }
#endif
    rows_scalar(xs, static_cast<const uint8_t *>(weights), width, act, n_act, n_rows, out);
}
void unpack_rows(bool xs, const void * src, std::size_t src_bytes, int width,
                 int plane_rows, int first, int count, void * dst, std::size_t dst_bytes) {
    const auto row = r8_row_bytes(xs, width);
    if (plane_rows < 8 || plane_rows % 8 || first < 0 || count < 0
        || first > plane_rows || count > plane_rows - first
        || src_bytes < product(std::size_t(plane_rows), row)
        || dst_bytes < product(std::size_t(count), row))
        throw std::invalid_argument("invalid R8 plane/row range or byte extents");
    disjoint(src, src_bytes, dst, dst_bytes);
    const auto * bytes = static_cast<const uint8_t *>(src);
    auto * out = static_cast<uint8_t *>(dst);
    const int nb = width / (xs ? 256 : 32), tile_bytes = xs ? kXsTile : kNlTile;
    const int block_bytes = xs ? kXsBlock : kNlBlock;
    for (int r = 0; r < count; ++r) {
        const int at = first + r, lane = at % 8;
        const auto * group = bytes + std::size_t(at / 8) * 8 * row;
        for (int b = 0; b < nb; ++b) {
            const auto * tile = group + std::size_t(b) * tile_bytes;
            auto * block = out + std::size_t(r) * row + std::size_t(b) * block_bytes;
            std::memcpy(block, tile + 2 * lane, 2);
            if (xs) {
                int high = 0;
                std::memset(block + 4, 0, 4);
                for (int g = 0; g < 8; ++g) {
                    const int s = tile_scale(tile, g, lane) + 32;
                    high |= (s >> 4) << (2 * g);
                    block[4 + g / 2] |= uint8_t((s & 15) << (4 * (g % 2)));
                }
                block[2] = uint8_t(high); block[3] = uint8_t(high >> 8);
            }
            for (int g = 0; g < (xs ? 8 : 1); ++g) {
                const auto * q = tile + (xs ? 64 + 128 * g : 16);
                auto * target = block + (xs ? 8 + 16 * g : 2);
                for (int k = 0; k < 16; ++k)
                    target[k] = uint8_t(tile_weight(q, lane, k) | (tile_weight(q, lane, k + 16) << 4));
            }
        }
    }
}
R8Act prep(const void * q8, int width, int64_t * corr, float * d, bool q8k) {
    const int unit = q8k ? 256 : 32, block = q8k ? kQ8KBlock : kQ80Block;
    if (width <= 0 || width % unit || !q8 || !corr || !d
        || reinterpret_cast<uintptr_t>(corr) % alignof(int64_t)
        || reinterpret_cast<uintptr_t>(d) % alignof(float))
        throw std::invalid_argument("invalid R8 activation preparation inputs");
    const auto qbytes = product(std::size_t(width / unit), block);
    const auto cbytes = product(std::size_t(width / 32), sizeof(int64_t));
    const auto dbytes = product(std::size_t(width / unit), sizeof(float));
    disjoint(q8, qbytes, corr, cbytes);
    disjoint(q8, qbytes, d, dbytes);
    disjoint(corr, cbytes, d, dbytes);
    const auto * bytes = static_cast<const uint8_t *>(q8);
    for (int b = 0; b < width / unit; ++b) {
        const auto * at = bytes + std::size_t(b) * block;
        d[b] = q8k ? load<float>(at) : half(at);
        for (int g = 0; g < unit / 32; ++g)
            corr[b * (unit / 32) + g] = corr_of(reinterpret_cast<const int8_t *>(at + (q8k ? 4 : 2) + 32 * g));
    }
    return {reinterpret_cast<const int8_t *>(bytes + (q8k ? 4 : 2)), block, unit / 32, corr, d, 1, 1};
}
} // namespace

bool r8_available() {
#ifdef STRATA_R8_VNNI
    return iq4_vnni_available();
#else
    return false;
#endif
}
std::size_t r8_row_bytes(bool xs, int width) {
    const int unit = xs ? 256 : 32;
    if (width <= 0 || width % unit)
        throw std::invalid_argument("R8 width must be positive and block divisible");
    return product(std::size_t(width / unit), xs ? kXsBlock : kNlBlock);
}
std::size_t r8_group_bytes(bool xs, int width) { return product(8, r8_row_bytes(xs, width)); }
void r8_repack_xs(const void * src, void * dst, int width) {
    const auto n = r8_group_bytes(true, width); disjoint(src, n, dst, n);
    repack_xs(static_cast<const uint8_t *>(src), static_cast<uint8_t *>(dst), width);
}
void r8_repack_nl(const void * src, void * dst, int width) {
    const auto n = r8_group_bytes(false, width); disjoint(src, n, dst, n);
    repack_nl(static_cast<const uint8_t *>(src), static_cast<uint8_t *>(dst), width);
}
void r8_unpack_xs(const void * src, void * dst, int width) {
    const auto n = r8_group_bytes(true, width); disjoint(src, n, dst, n);
    unpack_xs(static_cast<const uint8_t *>(src), static_cast<uint8_t *>(dst), width);
}
void r8_unpack_nl(const void * src, void * dst, int width) {
    const auto n = r8_group_bytes(false, width); disjoint(src, n, dst, n);
    unpack_nl(static_cast<const uint8_t *>(src), static_cast<uint8_t *>(dst), width);
}
void r8_unpack_xs_rows(const void * src, std::size_t src_bytes, int width, int plane_rows,
                      int first_row, int n_rows, void * dst, std::size_t dst_bytes) {
    unpack_rows(true, src, src_bytes, width, plane_rows, first_row, n_rows, dst, dst_bytes);
}
void r8_unpack_nl_rows(const void * src, std::size_t src_bytes, int width, int plane_rows,
                      int first_row, int n_rows, void * dst, std::size_t dst_bytes) {
    unpack_rows(false, src, src_bytes, width, plane_rows, first_row, n_rows, dst, dst_bytes);
}
R8Act r8_prep_q8k(const void * q8, int width, int64_t * corr, float * d) { return prep(q8, width, corr, d, true); }
R8Act r8_prep_q8_0(const void * q8, int width, int64_t * corr, float * d) { return prep(q8, width, corr, d, false); }
void r8_x64_store(const int8_t * qs, float scale, void * block64) {
    if (!qs || !block64) throw std::invalid_argument("null R8 x64 input");
    uint8_t block[64] = {};
    std::memcpy(block, qs, 32);
    const auto c = corr_of(qs);
    std::memcpy(block + 32, &c, 8);
    std::memcpy(block + 40, &scale, 4);
    std::memcpy(block64, block, 64);
}
void r8_x64_store_q8_0(const void * q8, void * block64) {
    if (!q8 || !block64) throw std::invalid_argument("null R8 native x64 input");
    // Stage native bits before writing, permitting overlapping input/output.
    const auto h = load<ggml_fp16_t>(q8);
    int8_t qs[32]; std::memcpy(qs, static_cast<const uint8_t *>(q8) + 2, 32);
    r8_x64_store(qs, ggml_fp16_to_fp32(h), block64);
    std::memcpy(static_cast<uint8_t *>(block64) + 44, &h, 2);
}
R8Act r8_x64_act(const void * row, int width) {
    if (!row || width <= 0 || width % 32) throw std::invalid_argument("invalid R8 x64 row/width");
    const auto * bytes = static_cast<const uint8_t *>(row);
    return {reinterpret_cast<const int8_t *>(bytes), kR8X64, 1,
            reinterpret_cast<const int64_t *>(bytes + 32), reinterpret_cast<const float *>(bytes + 40), 8, 16};
}
void r8_rows_xs(const void * w, int width, const R8Act * act, int n_act, int n_rows,
                float * out, IQ4VNNIPath path) { rows(true, w, width, act, n_act, n_rows, out, path); }
void r8_rows_nl(const void * w, int width, const R8Act * act, int n_act, int n_rows,
                float * out, IQ4VNNIPath path) { rows(false, w, width, act, n_act, n_rows, out, path); }
} // namespace strata::glm
