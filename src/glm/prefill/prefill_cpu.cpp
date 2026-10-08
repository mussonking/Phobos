// src/glm/prefill/prefill_cpu.cpp - see prefill_cpu.hpp.
//
// Kernel: a work item's 32 weight rows are unpacked once into u8 (codebook value + 128) in a lane-grouped layout,
// then every token of the expert is a register tile of pure vpdpbusd. Layout of a 512-value superblock (two IQ4
// blocks; both weights and activations): 8 steps of 64 bytes; dword lane l (16 per step) holds 4 values of the
// 32-value group l (values 4 st .. 4 st + 3 at step st < 4, values 16 + 4 (st - 4) .. at step st >= 4: the low then
// the high nibbles of the group's dword st % 4, so the unpack is one dword permute per pair of steps). After the 8
// steps lane l has summed its whole group: the weight scale and the activation scale (both per group) apply per lane,
// once per superblock. Activations are s8 with one F32 scale per 32 values; the 128 * sum(x) bias of the u8 weights
// is a per-lane constant of the activation row (computed once per token at quantization), the initial value of the
// int32 sums.
#include "prefill_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "iq4_cpu.hpp"
#include "iq4_vnni.hpp"
#include "pool.hpp"

#define PF_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c")))

namespace strata::glm {
namespace {

constexpr int kXsBlock = 136, kNlBlock = 18;
constexpr int kXsTile = 8 * kXsBlock, kNlTile = 8 * kNlBlock;   // R8: one block of 8 interleaved rows
constexpr int kRows = 32;   // rows per work item: the arena's NUMA chunk
constexpr int kTile = 5;    // tokens per register tile (at most)
constexpr int kPad = 64;    // row padding of R8-unpacked weights (see Unpacked::stride)
// IQ4_NL codebook + 128 (u8 operand of vpdpbusd)
constexpr uint8_t kCodebookU8[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};

[[noreturn]] void fail(const std::string & what) { throw std::runtime_error("glm prefill cpu: " + what); }

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 64-byte aligned host buffer.
struct Buf {
    void * p = nullptr;
    size_t bytes = 0;
    void reserve(size_t n) {
        if (n <= bytes) return;
        std::free(p);
        p = std::aligned_alloc(64, (n + 63) & ~size_t(63));
        if (!p) throw std::bad_alloc();
        bytes = n;
    }
    ~Buf() { std::free(p); }
};

// Quantized activation rows: s8 [rows][width] lane-grouped; per (row, superblock) 16 lane scales (the group's F32
// scale) and the 16 initial int32 sums (-128 * the group's sum of q). The R8 tiles read the same integers in value
// order (qn) with, per group, the initial sums of an R8 lane pair (c2: dword s = -128 * the sum of the group's values
// v with (v / 4) % 2 == s, the values lane s of a row multiplies; see LAYOUT.md).
struct Act {
    Buf q, d, c, qn, c2;
    int width = 0;
    void reserve(int rows, int w) {
        width = w;
        q.reserve((size_t) rows * w);
        d.reserve((size_t) rows * (w / 32) * 4);
        c.reserve((size_t) rows * (w / 32) * 4);
        qn.reserve((size_t) rows * w);
        c2.reserve((size_t) rows * (w / 32) * 8);
    }
    const int8_t * qrow(int r) const { return static_cast<const int8_t *>(q.p) + (size_t) r * width; }
    const float * drow(int r) const { return static_cast<const float *>(d.p) + (size_t) r * (width / 32); }
    const int32_t * crow(int r) const { return static_cast<const int32_t *>(c.p) + (size_t) r * (width / 32); }
    const int8_t * qnrow(int r) const { return static_cast<const int8_t *>(qn.p) + (size_t) r * width; }
    const int64_t * c2row(int r) const { return static_cast<const int64_t *>(c2.p) + (size_t) r * (width / 32); }
};

// Act rows copied in another order (qn, d, c only): the panel tiles read an expert's tokens as consecutive rows.
struct ActRows {
    Buf qn, d, c;
    int width = 0;
    void reserve(int rows, int w) {
        width = w;
        qn.reserve((size_t) rows * w);
        d.reserve((size_t) rows * (w / 32) * 4);
        c.reserve((size_t) rows * (w / 32) * 4);
    }
    int8_t * qnrow(int r) const { return static_cast<int8_t *>(qn.p) + (size_t) r * width; }
    float * drow(int r) const { return static_cast<float *>(d.p) + (size_t) r * (width / 32); }
    int32_t * crow(int r) const { return static_cast<int32_t *>(c.p) + (size_t) r * (width / 32); }
    void copy(int r, const Act & a, int ra) const {
        std::memcpy(qnrow(r), a.qnrow(ra), width);
        std::memcpy(drow(r), a.drow(ra), (width / 32) * 4);
        std::memcpy(crow(r), a.crow(ra), (width / 32) * 4);
    }
};

// Row r of a (value 16 h + j of group l of a superblock -> byte (4 h + j / 4) * 64 + 4 l + j % 4).
PF_TARGET void quantize_row(const float * x, int width, Act & a, int r) {
    int8_t * q = const_cast<int8_t *>(a.qrow(r));
    float * d = const_cast<float *>(a.drow(r));
    int32_t * c = const_cast<int32_t *>(a.crow(r));
    int8_t * qn = const_cast<int8_t *>(a.qnrow(r));
    int64_t * c2 = const_cast<int64_t *>(a.c2row(r));
    for (int sb = 0; sb < width / 512; ++sb) {
        int8_t * qb = q + sb * 512;
        for (int l = 0; l < 16; ++l) {
            const float * xg = x + sb * 512 + l * 32;
            const __m512 v0 = _mm512_loadu_ps(xg), v1 = _mm512_loadu_ps(xg + 16);
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
            const float inv = amax > 0 ? 127.0f / amax : 0.0f;
            const __m512 vi = _mm512_set1_ps(inv);
            int sum = 0, half[2] = {0, 0};
            for (int h = 0; h < 2; ++h) {
                const __m512i qi = _mm512_cvtps_epi32(_mm512_mul_ps(h ? v1 : v0, vi));   // round to nearest even
                alignas(16) int8_t q16[16];
                _mm_store_si128(reinterpret_cast<__m128i *>(q16), _mm512_cvtsepi32_epi8(qi));
                for (int st = 0; st < 4; ++st) std::memcpy(qb + (4 * h + st) * 64 + l * 4, q16 + st * 4, 4);
                std::memcpy(qn + sb * 512 + l * 32 + 16 * h, q16, 16);
                for (int j = 0; j < 16; ++j) half[(j / 4) % 2] += q16[j];
                sum += _mm512_reduce_add_epi32(qi);
            }
            d[sb * 16 + l] = amax / 127.0f;
            c[sb * 16 + l] = -128 * sum;
            c2[sb * 16 + l] = (int64_t) (uint32_t) (-128 * half[0]) | (int64_t) ((uint64_t) (uint32_t) (-128 * half[1]) << 32);
        }
    }
}

// SwiGLU with its clamps on 16 lanes, bitwise expert_swiglu_clamp: expf as glibc (2.27+) computes it on x86-64 with
// FMA (the same 2^(i/32) table, polynomial, shift rounding and fused steps in double; prefill_bench swiglu-exact
// checks every float gate on the host). Signaling NaN inputs excepted (fminf quiets them, vminps returns the clamp):
// float arithmetic never produces one. The scalar expf call was ~20 % of a panel item.
alignas(64) constexpr uint64_t kExp2fTab[32] = {   // bits of 2^(i/32) - (i << 47)
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b,
    0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715, 0x3feef1a7373aa9cb,
    0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
    0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d, 0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da, 0x3fefd0765b6e4540};

// expf of 8 floats widened to double (no overflow / underflow handling: the caller's)
PF_TARGET inline __m512d expf_pd(__m512d x) {
    constexpr double kShift = 0x1.8p+52;
    const __m512d z = _mm512_mul_pd(_mm512_set1_pd(0x1.71547652b82fep+0 * 32), x);
    const __m512d kd = _mm512_add_pd(z, _mm512_set1_pd(kShift));   // round to nearest integer
    const __m512i ki = _mm512_castpd_si512(kd);
    const __m512d r = _mm512_sub_pd(z, _mm512_sub_pd(kd, _mm512_set1_pd(kShift)));
    const __m512i t0 = _mm512_permutex2var_epi64(_mm512_load_si512(kExp2fTab), ki, _mm512_load_si512(kExp2fTab + 8));
    const __m512i t1 =
        _mm512_permutex2var_epi64(_mm512_load_si512(kExp2fTab + 16), ki, _mm512_load_si512(kExp2fTab + 24));
    const __m512i t = _mm512_mask_blend_epi64(_mm512_test_epi64_mask(ki, _mm512_set1_epi64(16)), t0, t1);
    const __m512d s = _mm512_castsi512_pd(_mm512_add_epi64(t, _mm512_slli_epi64(ki, 47)));
    const __m512d q = _mm512_fmadd_pd(_mm512_set1_pd(0x1.c6af84b912394p-5 / 32 / 32 / 32), r,
                                      _mm512_set1_pd(0x1.ebfce50fac4f3p-3 / 32 / 32));
    __m512d y = _mm512_fmadd_pd(_mm512_set1_pd(0x1.62e42ff0c52d6p-1 / 32), r, _mm512_set1_pd(1.0));
    y = _mm512_fmadd_pd(q, _mm512_mul_pd(r, r), y);
    return _mm512_mul_pd(y, s);
}

PF_TARGET inline __m512 swiglu16(__m512 g, __m512 u) {
    const __m512 ten = _mm512_set1_ps(10.0f);
    g = _mm512_min_ps(g, ten);   // NaN -> 10, as fminf
    u = _mm512_max_ps(_mm512_min_ps(u, ten), _mm512_set1_ps(-10.0f));
    const __m512 x = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(g), _mm512_set1_epi32(INT32_MIN)));
    const __m256 lo = _mm512_cvtpd_ps(expf_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(x))));
    const __m256 hi =
        _mm512_cvtpd_ps(expf_pd(_mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1)))));
    __m512 e = _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(lo)), _mm256_castps_pd(hi), 1));
    // x >= -10 here: no underflow; above ln(FLT_MAX) expf is +inf
    e = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, _mm512_set1_ps(0x1.62e42ep6f), _CMP_GT_OQ), e,
                             _mm512_set1_ps(INFINITY));
    return _mm512_mul_ps(_mm512_div_ps(g, _mm512_add_ps(_mm512_set1_ps(1.0f), e)), u);
}

// mid[t * F + i] = SwiGLU(gate, up) of the tile outputs gv / uv [i * ld + t], i < 32, t < n
PF_TARGET void swiglu_tile(const float * gv, const float * uv, int ld, int n, float * mid, int F) {
    const __m512i idx = _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                                           _mm512_set1_epi32(ld));
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < 2; ++h) {
            const size_t o = (size_t) 16 * h * ld + t;
            _mm512_storeu_ps(mid + (size_t) t * F + 16 * h,
                             swiglu16(_mm512_i32gather_ps(idx, gv + o, 4), _mm512_i32gather_ps(idx, uv + o, 4)));
        }
}
// the same from token-major tile outputs gv / uv [t * 32 + i] (the panel tiles)
PF_TARGET void swiglu_tile_t(const float * gv, const float * uv, int n, float * mid, int F) {
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < 32; h += 16)
            _mm512_storeu_ps(mid + (size_t) t * F + h,
                             swiglu16(_mm512_loadu_ps(gv + t * 32 + h), _mm512_loadu_ps(uv + t * 32 + h)));
}

// Unpacked weight rows of one work item: u8 [rows][width] lane-grouped and per (row, superblock) 16 lane scales.
struct Unpacked {
    Buf w, s;
    int width = 0;
    // w row stride. R8 (pad): width + one cache line, else rows exactly 4096 B apart all map to one L1 set and the
    // R8 unpack's 8 rows at once hit 8 lines of a set, 16 with both hyperthreads (48 threads: -2.4..3 % at 1-4 tokens
    // per expert). The GGUF unpack writes one row at a time: padding costs it 2-3 % there, so it keeps width. Not
    // the activations: padded rows, streamed from L3 / DRAM, straddle 4 KB pages (+2-3 % at 64-256 tokens per expert).
    size_t stride = 0;
    void reserve(int rows, int wd, bool pad) {
        width = wd;
        stride = (size_t) wd + (pad ? kPad : 0);
        w.reserve((size_t) rows * (wd + kPad));
        s.reserve((size_t) rows * (wd / 32) * 4);
    }
    const uint8_t * wrow(int r) const { return static_cast<const uint8_t *>(w.p) + (size_t) r * stride; }
    uint8_t * wrow(int r) { return static_cast<uint8_t *>(w.p) + (size_t) r * stride; }
    const float * srow(int r) const { return static_cast<const float *>(s.p) + (size_t) r * (width / 32); }
};

// One superblock's nibbles: a0 | a1 = groups 0..7, c0 | c1 = groups 8..15, 16 bytes each (low nibbles = values 0..15
// of the group, high = 16..31) -> w (8 x 64 bytes).
PF_TARGET inline void unpack_nibbles(__m512i a0, __m512i a1, __m512i c0, __m512i c1, uint8_t * w) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCodebookU8)));
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i base = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28);
    for (int st = 0; st < 4; ++st) {
        const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(st));
        const __m512i v = _mm512_mask_blend_epi32(0xFF00, _mm512_permutex2var_epi32(a0, idx, a1),
                                                  _mm512_permutex2var_epi32(c0, idx, c1));
        _mm512_store_si512(w + st * 64, _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4)));
        _mm512_store_si512(w + (4 + st) * 64, _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4)));
    }
}

PF_TARGET void unpack_xs_rows(const uint8_t * rows, int width, int n, Unpacked & u) {
    const int ns = width / 512;
    const size_t stride = (size_t) (width / 256) * kXsBlock;
    const __m512i grp = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i sh_lo = _mm512_slli_epi32(grp, 2), sh_hi = _mm512_slli_epi32(grp, 1);
    for (int r = 0; r < n; ++r)
        for (int sb = 0; sb < ns; ++sb) {
            const uint8_t * b0 = rows + r * stride + (size_t) 2 * sb * kXsBlock, * b1 = b0 + kXsBlock;
            // sub-scale g: 4 low bits = nibble g of bytes 4..7, 2 high bits = bits 2g of the u16 at byte 2
            uint32_t lo0, lo1;
            uint16_t d0, d1;
            std::memcpy(&lo0, b0 + 4, 4);
            std::memcpy(&lo1, b1 + 4, 4);
            std::memcpy(&d0, b0, 2);
            std::memcpy(&d1, b1, 2);
            const __m512i lo = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32((int) lo0), _mm512_set1_epi32((int) lo1));
            const __m512i hi = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32(b0[2] | (b0[3] << 8)),
                                                       _mm512_set1_epi32(b1[2] | (b1[3] << 8)));
            const __m512 d = _mm512_mask_blend_ps(0xFF00, _mm512_set1_ps(_cvtsh_ss(d0)), _mm512_set1_ps(_cvtsh_ss(d1)));
            const __m512i sub = _mm512_or_si512(
                _mm512_and_si512(_mm512_srlv_epi32(lo, sh_lo), _mm512_set1_epi32(15)),
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srlv_epi32(hi, sh_hi), _mm512_set1_epi32(3)), 4));
            const size_t o = (size_t) r * ns + sb;
            _mm512_store_ps(static_cast<float *>(u.s.p) + o * 16,
                            _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(sub, _mm512_set1_epi32(32))), d));
            unpack_nibbles(_mm512_loadu_si512(b0 + 8), _mm512_loadu_si512(b0 + 72), _mm512_loadu_si512(b1 + 8),
                           _mm512_loadu_si512(b1 + 72), u.wrow(r) + sb * 512);
        }
}

// 128-bit chunk c of halves i, i + 4, i + 8, i + 12 (half k at t + off[k]) in the vector's 4 slots.
PF_TARGET inline __m512i r8_chunks(const uint8_t * t, const size_t (&off)[16], int i, int c) {
    const auto ld = [&](int k) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(t + off[k] + 16 * c)); };
    __m512i v = _mm512_castsi128_si512(ld(i));
    v = _mm512_inserti32x4(v, ld(i + 4), 1);
    v = _mm512_inserti32x4(v, ld(i + 8), 2);
    return _mm512_inserti32x4(v, ld(i + 12), 3);
}

// The 16 groups' qs half h (dword 2r + s = row r, lane s) of a superblock -> row r's steps 4h + s (low nibbles) and
// 4h + 2 + s (high nibbles) for the 8 rows; half i is at t + off[i]. The 16 x 16 dword transpose (output j lane l =
// half l's dword j) as a 4 x 4 transpose of 128-bit chunks, done by the loads (inserts: ports 0 and 5), then a 4 x 4
// dword transpose inside each slot (unpacks): half the port-5 shuffles of four permutex2var stages (Cascade Lake).
PF_TARGET inline void r8_half_rows(const uint8_t * t, const size_t (&off)[16], int h, uint8_t * w0, size_t wstride) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCodebookU8)));
    const __m512i m4 = _mm512_set1_epi8(15);
    for (int c = 0; c < 4; ++c) {
        // v[i] slot I = chunk c of half 4 I + i: dword j' of slot I = half 4 I + i's dword 4 c + j'
        const __m512i v0 = r8_chunks(t, off, 0, c), v1 = r8_chunks(t, off, 1, c);
        const __m512i v2 = r8_chunks(t, off, 2, c), v3 = r8_chunks(t, off, 3, c);
        const __m512i t0 = _mm512_unpacklo_epi32(v0, v1), t1 = _mm512_unpackhi_epi32(v0, v1);
        const __m512i t2 = _mm512_unpacklo_epi32(v2, v3), t3 = _mm512_unpackhi_epi32(v2, v3);
        const __m512i a[4] = {_mm512_unpacklo_epi64(t0, t2), _mm512_unpackhi_epi64(t0, t2),
                              _mm512_unpacklo_epi64(t1, t3), _mm512_unpackhi_epi64(t1, t3)};
        for (int jj = 0; jj < 4; ++jj) {   // a[jj] lane l = half l's dword j = 4 c + jj = 2 r + s
            const int j = 4 * c + jj, r = j / 2, sl = j % 2;
            uint8_t * w = w0 + r * wstride;
            _mm512_store_si512(w + (4 * h + sl) * 64, _mm512_shuffle_epi8(lut, _mm512_and_si512(a[jj], m4)));
            _mm512_store_si512(w + (4 * h + 2 + sl) * 64,
                               _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(a[jj], 4), m4)));
        }
    }
}

// The 64 sub-scales S[8g + r] (group g, row r) of an R8 IQ4_XS tile tb: c[k] byte j = S[16k + j] (groups 2k, 2k + 1).
PF_TARGET inline void xs_r8_subscales(const uint8_t * tb, __m128i (&c)[4]) {
    const __m128i m4 = _mm_set1_epi8(15), m3 = _mm_set1_epi8(3);
    const __m128i lo0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tb + 16));
    const __m128i lo1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tb + 32));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(tb + 48));
    // sc_lo byte j = S[j] low | S[j + 32] low << 4; sc_hi byte j bits 2p = S[j + 16p] >> 4
    const auto hi2 = [&](int p) { return _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(hi, 2 * p), m3), 4); };
    c[0] = _mm_or_si128(_mm_and_si128(lo0, m4), hi2(0));
    c[1] = _mm_or_si128(_mm_and_si128(lo1, m4), hi2(1));
    c[2] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lo0, 4), m4), hi2(2));
    c[3] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lo1, 4), m4), hi2(3));
}

// R8 row groups (LAYOUT.md: 8 rows interleaved per block tile; row group q = the bytes of GGUF rows 8q..8q+7) into
// the same unpacked bytes and scales as unpack_xs_rows: qs half h of group g, lane 2r + s, byte i = w[16h + 4s + i] |
// w[16h + 4s + i + 8] << 4, i.e. the low nibbles are step 4h + s of row r, the high ones step 4h + 2 + s.
PF_TARGET void unpack_xs_rows_r8(const uint8_t * rows, int width, int n, Unpacked & u) {
    const int ns = width / 512;
    const size_t gstride = (size_t) 8 * (width / 256) * kXsBlock;
    size_t xs_off0[16], xs_off1[16];   // half h of lane l = group l % 8 of the superblock's tile l / 8
    for (int l = 0; l < 16; ++l) {
        xs_off0[l] = (size_t) (l / 8) * kXsTile + 64 + 128 * (l % 8);
        xs_off1[l] = xs_off0[l] + 64;
    }
    for (int q = 0; q < n / 8; ++q)
        for (int sb = 0; sb < ns; ++sb) {
            const uint8_t * t = rows + q * gstride + (size_t) 2 * sb * kXsTile;
            uint8_t * w0 = u.wrow(8 * q) + sb * 512;
            for (int h = 0; h < 2; ++h) r8_half_rows(t, h ? xs_off1 : xs_off0, h, w0, u.stride);
            // sub-scales S[8g + r] of both tiles, row-major: sr[bt][r / 2] qword r % 2 = S[8g + r] for g = 0..7
            __m128i sr[2][4];
            for (int bt = 0; bt < 2; ++bt) {
                __m128i c[4];
                xs_r8_subscales(t + bt * kXsTile, c);
                // c[k] = S[16k .. 16k + 15] = groups 2k, 2k + 1: 8 x 8 byte transpose to rows
                __m128i a2[4];
                for (int k = 0; k < 4; ++k) a2[k] = _mm_unpacklo_epi8(c[k], _mm_srli_si128(c[k], 8));
                const __m128i b0 = _mm_unpacklo_epi16(a2[0], a2[1]), b1 = _mm_unpackhi_epi16(a2[0], a2[1]);
                const __m128i b2 = _mm_unpacklo_epi16(a2[2], a2[3]), b3 = _mm_unpackhi_epi16(a2[2], a2[3]);
                sr[bt][0] = _mm_unpacklo_epi32(b0, b2);
                sr[bt][1] = _mm_unpackhi_epi32(b0, b2);
                sr[bt][2] = _mm_unpacklo_epi32(b1, b3);
                sr[bt][3] = _mm_unpackhi_epi32(b1, b3);
            }
            // lanes 0..7: tile 0's d[r], lanes 8..15: tile 1's
            const __m512 dd = _mm512_cvtph_ps(_mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(t))),
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(t + kXsTile)), 1));
            const __m512i dsel = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
            for (int r = 0; r < 8; ++r) {
                // lane l: (S[8 (l % 8) + r] - 32) * d of tile l / 8, as unpack_xs_rows computes it
                const __m128i s16 = (r & 1) ? _mm_unpackhi_epi64(sr[0][r / 2], sr[1][r / 2])
                                            : _mm_unpacklo_epi64(sr[0][r / 2], sr[1][r / 2]);
                const __m512 d = _mm512_permutexvar_ps(_mm512_add_epi32(dsel, _mm512_set1_epi32(r)), dd);
                _mm512_store_ps(static_cast<float *>(u.s.p) + ((size_t) (8 * q + r) * ns + sb) * 16,
                                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(s16),
                                                                                  _mm512_set1_epi32(32))), d));
            }
        }
}

PF_TARGET void unpack_nl_rows_r8(const uint8_t * rows, int width, int n, Unpacked & u) {
    const int ns = width / 512;
    const size_t gstride = (size_t) 8 * (width / 32) * kNlBlock;
    size_t nl_off0[16], nl_off1[16];   // half h of lane l = tile (32-value block) l of the superblock
    for (int l = 0; l < 16; ++l) {
        nl_off0[l] = (size_t) l * kNlTile + 16;
        nl_off1[l] = nl_off0[l] + 64;
    }
    for (int q = 0; q < n / 8; ++q)
        for (int sb = 0; sb < ns; ++sb) {
            const uint8_t * t = rows + q * gstride + (size_t) 16 * sb * kNlTile;
            uint8_t * w0 = u.wrow(8 * q) + sb * 512;
            for (int h = 0; h < 2; ++h) r8_half_rows(t, h ? nl_off1 : nl_off0, h, w0, u.stride);
            for (int r = 0; r < 8; ++r) {
                alignas(32) uint16_t dh[16];
                for (int l = 0; l < 16; ++l) std::memcpy(dh + l, t + (size_t) l * kNlTile + 2 * r, 2);
                _mm512_store_ps(static_cast<float *>(u.s.p) + ((size_t) (8 * q + r) * ns + sb) * 16,
                                _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(dh))));
            }
        }
}

PF_TARGET inline __m512i load_groups4(const uint8_t * blk) {   // 4 groups' 16 qs bytes (after each F16 scale)
    __m512i v = _mm512_castsi128_si512(_mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2)));
    v = _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + kNlBlock + 2)), 1);
    v = _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 2 * kNlBlock + 2)), 2);
    return _mm512_inserti32x4(v, _mm_loadu_si128(reinterpret_cast<const __m128i *>(blk + 3 * kNlBlock + 2)), 3);
}

PF_TARGET void unpack_nl_rows(const uint8_t * rows, int width, int n, Unpacked & u) {
    const int ns = width / 512;
    const size_t stride = (size_t) (width / 32) * kNlBlock;
    for (int r = 0; r < n; ++r)
        for (int sb = 0; sb < ns; ++sb) {
            const uint8_t * blk = rows + r * stride + (size_t) sb * 16 * kNlBlock;
            alignas(32) uint16_t dh[16];
            for (int g = 0; g < 16; ++g) std::memcpy(dh + g, blk + g * kNlBlock, 2);
            const size_t o = (size_t) r * ns + sb;
            _mm512_store_ps(static_cast<float *>(u.s.p) + o * 16,
                            _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(dh))));
            unpack_nibbles(load_groups4(blk), load_groups4(blk + 4 * kNlBlock), load_groups4(blk + 8 * kNlBlock),
                           load_groups4(blk + 12 * kNlBlock), u.wrow(r) + sb * 512);
        }
}

// Activation rows of a register tile.
struct Tok {
    const int8_t * q[kTile];
    const float * d[kTile];
    const int32_t * c[kTile];
};

// out[r * ldo + t] = row r . token t, for R rows from r0 and N tokens.
template <int R, int N>
PF_TARGET void tile(const Unpacked & u, int r0, const Tok & x, float * out, int ldo) {
    const int ns = u.width / 512;
    __m512 acc[R][N];
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < N; ++t) acc[r][t] = _mm512_setzero_ps();
    for (int b = 0; b < ns; ++b) {
        __m512i ai[R][N];
        for (int t = 0; t < N; ++t) {
            const __m512i c = _mm512_load_si512(x.c[t] + b * 16);
            for (int r = 0; r < R; ++r) ai[r][t] = c;
        }
        for (int st = 0; st < 8; ++st) {
            __m512i xv[N];
            for (int t = 0; t < N; ++t) xv[t] = _mm512_load_si512(x.q[t] + b * 512 + st * 64);
            for (int r = 0; r < R; ++r) {
                const __m512i wv = _mm512_load_si512(u.wrow(r0 + r) + b * 512 + st * 64);
                for (int t = 0; t < N; ++t) ai[r][t] = _mm512_dpbusd_epi32(ai[r][t], wv, xv[t]);
            }
        }
        __m512 s[R];
        for (int r = 0; r < R; ++r) s[r] = _mm512_load_ps(u.srow(r0 + r) + b * 16);
        for (int t = 0; t < N; ++t) {
            const __m512 xd = _mm512_load_ps(x.d[t] + b * 16);
            for (int r = 0; r < R; ++r)
                acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai[r][t]), _mm512_mul_ps(s[r], xd), acc[r][t]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int t = 0; t < N; ++t) out[r * ldo + t] = _mm512_reduce_add_ps(acc[r][t]);
}

// out[r * ldo + t] for the item's kRows rows and n <= kTile tokens.
void rows_x_tokens(const Unpacked & u, const Tok & x, int n, float * out, int ldo) {
    for (int r = 0; r < kRows; r += 2) {
        switch (n) {
            case 1: tile<2, 1>(u, r, x, out + r * ldo, ldo); break;
            case 2: tile<2, 2>(u, r, x, out + r * ldo, ldo); break;
            case 3: tile<2, 3>(u, r, x, out + r * ldo, ldo); break;
            case 4: tile<2, 4>(u, r, x, out + r * ldo, ldo); break;
            default: tile<2, 5>(u, r, x, out + r * ldo, ldo); break;
        }
    }
}

// R8 tiles: the item's row groups read in place (no unpack). One 64-B qs half of a group holds 8 rows x 2 lanes x 4
// bytes, and each byte holds 2 weights, so 4 vpdpbusd (one broadcast activation qword each) cover 8 rows x 32 values
// of a token. Per (token, group): the int32 lanes start at the token's c2 pair, then one cvt + fma with the weight
// scale times the token's group scale (the scales differ per group, so they apply per group, not per superblock).
// Accumulators stay in registers for the whole row: N tokens x 1 zmm.
constexpr int kTileR8 = 12;   // tokens per R8 tile (at most)
// Up to this many tokens per expert the R8 tiles beat R8 unpack + GGUF tiles; above, the unpack is amortized and the
// GGUF tile's 16-row accumulators (8 vpdpbusd per scale step vs 4 here) win. Quiet V100 host, 48 threads, U 192 (ms per
// layer, unpack -> tiles): gate_up 16 tok 24.4 -> 23.6, 32 tok 41.7 -> 48.9; down 16 tok 15.2 -> 13.0, 32 tok
// 25.2 -> 26.5.
constexpr int kR8MaxGateUp = 16, kR8MaxDown = 24;

struct TokR8 {
    const int8_t * q[kTileR8];
    const float * d[kTileR8];
    const int64_t * c2[kTileR8];
};

PF_TARGET inline __m512 dup8(__m256 v) {
    return _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7), _mm512_castps256_ps512(v));
}

// lanes (r, s) -> out[r * ldo + t] = lane 2r + lane 2r+1
PF_TARGET inline void store_r8(__m512 acc, float * out, int ldo) {
    alignas(32) float v[8];
    const __m512 pair = _mm512_add_ps(acc, _mm512_permute_ps(acc, 0xB1));
    _mm256_store_ps(v, _mm512_castps512_ps256(_mm512_maskz_compress_ps(0x5555, pair)));
    for (int r = 0; r < 8; ++r) out[r * ldo] = v[r];
}

// the 4 u8 weight vectors (codebook + 128) of a group's two qs halves at q: values 0-7, 8-15, 16-23, 24-31
PF_TARGET inline void r8_weights(const uint8_t * q, __m512i (&w)[4]) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCodebookU8)));
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i p0 = _mm512_loadu_si512(q), p1 = _mm512_loadu_si512(q + 64);
    w[0] = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, m4));
    w[1] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), m4));
    w[2] = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, m4));
    w[3] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), m4));
}

// acc[t] += (the 8 rows' group dot with token t) * ws * token t's scale of group gi
template <int N>
PF_TARGET inline void r8_group(const __m512i (&w)[4], __m512 ws, const TokR8 & x, int gi, __m512 (&acc)[N]) {
    for (int t = 0; t < N; ++t) {
        const int8_t * xq = x.q[t] + gi * 32;
        int64_t c, x0, x1, x2, x3;
        std::memcpy(&c, x.c2[t] + gi, 8);
        std::memcpy(&x0, xq, 8);
        std::memcpy(&x1, xq + 8, 8);
        std::memcpy(&x2, xq + 16, 8);
        std::memcpy(&x3, xq + 24, 8);
        __m512i s = _mm512_set1_epi64(c);
        s = _mm512_dpbusd_epi32(s, w[0], _mm512_set1_epi64(x0));
        s = _mm512_dpbusd_epi32(s, w[1], _mm512_set1_epi64(x1));
        s = _mm512_dpbusd_epi32(s, w[2], _mm512_set1_epi64(x2));
        s = _mm512_dpbusd_epi32(s, w[3], _mm512_set1_epi64(x3));
        acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(ws, _mm512_set1_ps(x.d[t][gi])), acc[t]);
    }
}

// IQ4_XS: the 8 rows of the row group at w (width / 256 tiles of 1088 B) x N tokens -> out[r * ldo + t]
template <int N>
PF_TARGET void r8_xs_tile(const uint8_t * w, int width, const TokR8 & x, float * out, int ldo) {
    __m512 acc[N];
    for (int t = 0; t < N; ++t) acc[t] = _mm512_setzero_ps();
    for (int b = 0; b < width / 256; ++b) {
        const uint8_t * tile = w + (size_t) b * kXsTile;
        const __m256 d8 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(tile)));
        // the 64 sub-block scales S[8g + r] - 32 as int8 (LAYOUT.md: sc_lo at 16, sc_hi at 48)
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
                           _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 4), m4y),
                                                           _mm256_slli_epi16(hB, 4)), b32));
        for (int g = 0; g < 8; ++g) {
            const __m512 ws = dup8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i *>(S + 8 * g)))), d8));
            __m512i wv[4];
            r8_weights(tile + 64 + 128 * g, wv);
            r8_group<N>(wv, ws, x, b * 8 + g, acc);
        }
    }
    for (int t = 0; t < N; ++t) store_r8(acc[t], out + t, ldo);
}

// IQ4_NL: the 8 rows of the row group at w (width / 32 tiles of 144 B) x N tokens -> out[r * ldo + t]
template <int N>
PF_TARGET void r8_nl_tile(const uint8_t * w, int width, const TokR8 & x, float * out, int ldo) {
    __m512 acc[N];
    for (int t = 0; t < N; ++t) acc[t] = _mm512_setzero_ps();
    for (int b = 0; b < width / 32; ++b) {
        const uint8_t * tile = w + (size_t) b * kNlTile;
        const __m512 ws = dup8(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(tile))));
        __m512i wv[4];
        r8_weights(tile + 16, wv);
        r8_group<N>(wv, ws, x, b, acc);
    }
    for (int t = 0; t < N; ++t) store_r8(acc[t], out + t, ldo);
}

// out[r * ldo + t] for the item's kRows rows (kRows / 8 R8 row groups from rows) and n <= kTileR8 tokens
template <int N>
void r8_rows_n(const uint8_t * rows, bool xs, int width, const TokR8 & x, float * out, int ldo) {
    const size_t gstride = xs ? (size_t) 8 * (width / 256) * kXsBlock : (size_t) 8 * (width / 32) * kNlBlock;
    for (int q = 0; q < kRows / 8; ++q)
        (xs ? r8_xs_tile<N> : r8_nl_tile<N>)(rows + q * gstride, width, x, out + 8 * q * ldo, ldo);
}
void r8_rows_x_tokens(const uint8_t * rows, bool xs, int width, const TokR8 & x, int n, float * out, int ldo) {
    switch (n) {
        case 1: r8_rows_n<1>(rows, xs, width, x, out, ldo); break;
        case 2: r8_rows_n<2>(rows, xs, width, x, out, ldo); break;
        case 3: r8_rows_n<3>(rows, xs, width, x, out, ldo); break;
        case 4: r8_rows_n<4>(rows, xs, width, x, out, ldo); break;
        case 5: r8_rows_n<5>(rows, xs, width, x, out, ldo); break;
        case 6: r8_rows_n<6>(rows, xs, width, x, out, ldo); break;
        case 7: r8_rows_n<7>(rows, xs, width, x, out, ldo); break;
        case 8: r8_rows_n<8>(rows, xs, width, x, out, ldo); break;
        case 9: r8_rows_n<9>(rows, xs, width, x, out, ldo); break;
        case 10: r8_rows_n<10>(rows, xs, width, x, out, ldo); break;
        case 11: r8_rows_n<11>(rows, xs, width, x, out, ldo); break;
        default: r8_rows_n<12>(rows, xs, width, x, out, ldo); break;
    }
}

// Panel tiles (many tokens per expert): the GGUF tile keeps one int32 and one F32 zmm per (row, token) (lanes =
// groups), so its register tile stops at 2 rows x 5 tokens and every vpdpbusd reloads ~45 B from L2. Here lanes =
// rows: a panel holds 16 rows, per group 8 zmm of u8 weights (zmm k lane r = row r's values 4k..4k+3, the
// unpacked bytes transposed) and the 16 rows' group scales. Per (group, token): the int32 sums start at the token's
// group bias, 8 vpdpbusd with the token's activation dwords broadcast from memory, then one cvt + fma with the 16
// weight scales times the token's group scale. One F32 zmm per (16 rows, token): 16 rows x up to kTileP tokens in
// registers, the group's weights loaded once per tile. Sums: per superblock (16 groups) in registers, then added to
// the output (the GGUF tile's order of magnitude of terms per float sum, see cpu-exact).
constexpr int kTileP = 16;    // tokens per panel tile (at most; prefill_cpu_dot_panels)
// tokens per panel tile in prefill_cpu_run: 8 interleaved chains keep acc + sums in registers (16 spill: 229 vs 188
// GMAC/s in cache)
constexpr int kTileRun = 8;
// tokens per expert from which the panels replace the GGUF / R8 tiles (48 threads, R8, U 64: a tie at 8, 1.2x at 12,
// 1.25-1.3x at 16-24 over the tiles; 2026-10-05 pf_ab48)
constexpr int kPanelMin = 8;

struct Panels {
    Buf w, s;
    int groups = 0;
    void reserve(int rows, int width) {
        groups = width / 32;
        w.reserve((size_t) rows * width);
        s.reserve((size_t) rows * groups * 4);
    }
    const uint8_t * wp(int p, int g) const { return static_cast<const uint8_t *>(w.p) + ((size_t) p * groups + g) * 512; }
    const float * sp(int p, int g) const { return static_cast<const float *>(s.p) + ((size_t) p * groups + g) * 16; }
};

// 16 x 16 dword transpose: a[r] lane l -> a[l] lane r
PF_TARGET inline void transpose16(__m512i (&a)[16]) {
    __m512i t[16], r[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i] = _mm512_unpacklo_epi32(a[2 * i], a[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(a[2 * i], a[2 * i + 1]);
    }
    // r[4 i + c] chunk L = column 4 L + c of rows 4 i .. 4 i + 3
    for (int i = 0; i < 4; ++i) {
        r[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        r[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int c = 0; c < 4; ++c) {
        const __m512i u0 = _mm512_shuffle_i32x4(r[c], r[4 + c], 0x88), u1 = _mm512_shuffle_i32x4(r[c], r[4 + c], 0xDD);
        const __m512i v0 = _mm512_shuffle_i32x4(r[8 + c], r[12 + c], 0x88);
        const __m512i v1 = _mm512_shuffle_i32x4(r[8 + c], r[12 + c], 0xDD);
        a[c] = _mm512_shuffle_i32x4(u0, v0, 0x88);
        a[4 + c] = _mm512_shuffle_i32x4(u1, v1, 0x88);
        a[8 + c] = _mm512_shuffle_i32x4(u0, v0, 0xDD);
        a[12 + c] = _mm512_shuffle_i32x4(u1, v1, 0xDD);
    }
}

// The unpacked rows (lane-grouped: step st lane l of superblock sb = values 4 st .. 4 st + 3 of group 16 sb + l)
// -> panels: group 16 sb + l, zmm st, lane r = row r's step st lane l; scales likewise
PF_TARGET void to_panels(const Unpacked & u, int rows, Panels & pn) {
    const int ns = u.width / 512;
    for (int p = 0; p < rows / 16; ++p)
        for (int sb = 0; sb < ns; ++sb) {
            __m512i a[16];
            for (int st = 0; st < 8; ++st) {
                for (int r = 0; r < 16; ++r) a[r] = _mm512_load_si512(u.wrow(16 * p + r) + sb * 512 + st * 64);
                transpose16(a);
                for (int l = 0; l < 16; ++l)
                    _mm512_store_si512(const_cast<uint8_t *>(pn.wp(p, sb * 16 + l)) + st * 64, a[l]);
            }
            for (int r = 0; r < 16; ++r) a[r] = _mm512_castps_si512(_mm512_load_ps(u.srow(16 * p + r) + sb * 16));
            transpose16(a);
            for (int l = 0; l < 16; ++l)
                _mm512_store_si512(const_cast<float *>(pn.sp(p, sb * 16 + l)), a[l]);
        }
}

// R8 row groups 2p, 2p + 1 straight to panel p, without the unpack and the 16 x 16 transposes (~7 us of the ~11 us of
// unpack + to_panels per 32 x 4096 rows): a panel's 16 rows are the two groups' 8 rows, and half h of a group's qs
// holds dword 2r + s = row r's steps 4h + s (low nibbles) and 4h + 2 + s (high ones), so one dword permute per
// (half, s) gives a step's 16 lanes. Bytes and scales identical to unpack_*_rows_r8 + to_panels.
PF_TARGET inline void r8_group_to_panel(const uint8_t * a, const uint8_t * b, uint8_t * w) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kCodebookU8)));
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i i0 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    for (int h = 0; h < 2; ++h) {
        const __m512i va = _mm512_loadu_si512(a + 64 * h), vb = _mm512_loadu_si512(b + 64 * h);
        for (int s = 0; s < 2; ++s) {
            const __m512i v = _mm512_permutex2var_epi32(va, _mm512_add_epi32(i0, _mm512_set1_epi32(s)), vb);
            _mm512_store_si512(w + (4 * h + s) * 64, _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4)));
            _mm512_store_si512(w + (4 * h + 2 + s) * 64,
                               _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4)));
        }
    }
}

PF_TARGET void r8_xs_to_panels(const uint8_t * rows, int width, int nrows, Panels & pn) {
    const int nb = width / 256;
    const size_t gstride = (size_t) 8 * nb * kXsBlock;
    for (int p = 0; p < nrows / 16; ++p)
        for (int b = 0; b < nb; ++b) {
            const uint8_t * t0 = rows + (size_t) 2 * p * gstride + (size_t) b * kXsTile, * t1 = t0 + gstride;
            for (int g = 0; g < 8; ++g)
                r8_group_to_panel(t0 + 64 + 128 * g, t1 + 64 + 128 * g, const_cast<uint8_t *>(pn.wp(p, 8 * b + g)));
            __m128i c0[4], c1[4];
            xs_r8_subscales(t0, c0);
            xs_r8_subscales(t1, c1);
            // lanes 0..7: group 2p's d[r], 8..15: group 2p + 1's
            const __m512 dd = _mm512_cvtph_ps(_mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(t0))),
                _mm_loadu_si128(reinterpret_cast<const __m128i *>(t1)), 1));
            for (int g = 0; g < 8; ++g) {
                const __m128i s16 = (g & 1) ? _mm_unpackhi_epi64(c0[g / 2], c1[g / 2])
                                            : _mm_unpacklo_epi64(c0[g / 2], c1[g / 2]);
                _mm512_store_ps(const_cast<float *>(pn.sp(p, 8 * b + g)),
                                _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(s16),
                                                                                  _mm512_set1_epi32(32))), dd));
            }
        }
}

PF_TARGET void r8_nl_to_panels(const uint8_t * rows, int width, int nrows, Panels & pn) {
    const int G = width / 32;
    const size_t gstride = (size_t) 8 * G * kNlBlock;
    for (int p = 0; p < nrows / 16; ++p)
        for (int g = 0; g < G; ++g) {
            const uint8_t * t0 = rows + (size_t) 2 * p * gstride + (size_t) g * kNlTile, * t1 = t0 + gstride;
            r8_group_to_panel(t0 + 16, t1 + 16, const_cast<uint8_t *>(pn.wp(p, g)));
            _mm512_store_ps(const_cast<float *>(pn.sp(p, g)),
                            _mm512_cvtph_ps(_mm256_inserti128_si256(
                                _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(t0))),
                                _mm_loadu_si128(reinterpret_cast<const __m128i *>(t1)), 1)));
        }
}

// A panel tile's activations: N consecutive rows of an Act of width W (value order qn, per group scale d and bias c),
// row t at qn + t * W, d / c + t * W / 32. 3 base pointers and immediate offsets in the kernel: one pointer per token
// and array does not fit the 16 GPRs (GCC reloads them from the stack every group), and packing the tile's rows per
// group cost ~20 % of an item.
struct RowsP {
    const int8_t * q;
    const float * d;
    const int32_t * c;
};

// out[t * ldo + r] = row 16 p + r . token t, r < 16, N tokens
template <int N, int W>
PF_TARGET void panel_tile(const Panels & pn, int p, RowsP x, float * out, int ldo) {
    constexpr int G = W / 32;
    for (int t = 0; t < N; ++t) _mm512_storeu_ps(out + t * ldo, _mm512_setzero_ps());
    const int8_t * xq = x.q;
    const float * xd = x.d;
    const int32_t * xc = x.c;
    for (int g0 = 0; g0 < G; g0 += 16) {
        __m512 acc[N];
        for (int t = 0; t < N; ++t) acc[t] = _mm512_setzero_ps();
        for (int g = g0; g < g0 + 16; ++g, xq += 32, ++xd, ++xc) {
            const uint8_t * wb = pn.wp(p, g);
            const __m512 ws = _mm512_load_ps(pn.sp(p, g));
            // fully unrolled (GCC keeps acc[] on the stack otherwise); 8 tokens interleaved per k: 8 independent
            // vpdpbusd chains (latency 5: one token's 8 chained ones leave the ports idle); w[k] loaded at its use
            // (L1), so the registers hold the 8 int32 sums and the N F32 accumulators
#pragma GCC unroll 2
            for (int t0 = 0; t0 < N; t0 += 8) {
                __m512i si[8];
#pragma GCC unroll 8
                for (int u = 0; u < 8; ++u)
                    if (t0 + u < N) si[u] = _mm512_set1_epi32(xc[(t0 + u) * G]);
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    const __m512i w = _mm512_load_si512(wb + 64 * k);
#pragma GCC unroll 8
                    for (int u = 0; u < 8; ++u)
                        if (t0 + u < N) {
                            int32_t v;
                            std::memcpy(&v, xq + (t0 + u) * W + 4 * k, 4);
                            si[u] = _mm512_dpbusd_epi32(si[u], w, _mm512_set1_epi32(v));
                        }
                }
#pragma GCC unroll 8
                for (int u = 0; u < 8; ++u)
                    if (t0 + u < N)
                        acc[t0 + u] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(si[u]),
                                                      _mm512_mul_ps(ws, _mm512_set1_ps(xd[(t0 + u) * G])), acc[t0 + u]);
            }
        }
        for (int t = 0; t < N; ++t) _mm512_storeu_ps(out + t * ldo, _mm512_add_ps(_mm512_loadu_ps(out + t * ldo), acc[t]));
    }
}

// out[t * ldo + r] for the panels' rows (rows / 16 panels) and n <= kTileP tokens
template <int N, int W>
void panels_n(const Panels & pn, int rows, RowsP x, float * out, int ldo) {
    for (int p = 0; p < rows / 16; ++p) panel_tile<N, W>(pn, p, x, out + 16 * p, ldo);
}
template <int W>
void panels_w(const Panels & pn, int rows, RowsP x, int n, float * out, int ldo) {
    switch (n) {
        case 1: panels_n<1, W>(pn, rows, x, out, ldo); break;
        case 2: panels_n<2, W>(pn, rows, x, out, ldo); break;
        case 3: panels_n<3, W>(pn, rows, x, out, ldo); break;
        case 4: panels_n<4, W>(pn, rows, x, out, ldo); break;
        case 5: panels_n<5, W>(pn, rows, x, out, ldo); break;
        case 6: panels_n<6, W>(pn, rows, x, out, ldo); break;
        case 7: panels_n<7, W>(pn, rows, x, out, ldo); break;
        case 8: panels_n<8, W>(pn, rows, x, out, ldo); break;
        case 9: panels_n<9, W>(pn, rows, x, out, ldo); break;
        case 10: panels_n<10, W>(pn, rows, x, out, ldo); break;
        case 11: panels_n<11, W>(pn, rows, x, out, ldo); break;
        case 12: panels_n<12, W>(pn, rows, x, out, ldo); break;
        case 13: panels_n<13, W>(pn, rows, x, out, ldo); break;
        case 14: panels_n<14, W>(pn, rows, x, out, ldo); break;
        case 15: panels_n<15, W>(pn, rows, x, out, ldo); break;
        default: panels_n<16, W>(pn, rows, x, out, ldo); break;
    }
}
// x: the tile's n consecutive activation rows (RowsP), width = the panels' width (4096 or 2048: the GLM geometry)
void panels_x_tokens(const Panels & pn, int rows, RowsP x, int n, float * out, int ldo) {
    switch (pn.groups * 32) {
        case 4096: panels_w<4096>(pn, rows, x, n, out, ldo); break;
        case 2048: panels_w<2048>(pn, rows, x, n, out, ldo); break;
        default: fail("panel width " + std::to_string(pn.groups * 32));
    }
}

// Size of the tile starting at job j0 of a list of m jobs [0, m): as few tiles of at most kt as possible, sizes
// within one.
inline int tile_size(int j0, int m, int kt = kTile) {
    // tile k of nt starts at k * m / nt (floor), so k = ceil(j0 * nt / m)
    const int nt = (m + kt - 1) / kt, k = (j0 * nt + m - 1) / m;
    return (k + 1) * m / nt - k * m / nt;
}

thread_local Unpacked tl_a, tl_b;
thread_local Panels tl_pa, tl_pb;

}  // namespace

struct PrefillCpu {
    Geometry g;
    int max_rows = 0, max_jobs = 0;
    Act y;     // [max_rows] of x
    Act mid;   // [max_jobs] of the SwiGLU output
    Buf midf;  // [max_jobs][F] F32 SwiGLU
    ActRows yj;              // y rows of the panel experts' jobs, expert u's from row pbase[u] (per run)
    std::vector<int> pbase;  // [U]: -1 = not a panel expert
    PrefillCpuTiming tm;
    bool r8_tiles = true;   // false: R8 experts always unpacked to the GGUF tiles (prefill_cpu_set_r8_tiles)
    bool panels = true;     // false: no panel tiles (prefill_cpu_set_panels)
};

PrefillCpu * prefill_cpu_create(const Geometry & g, int max_rows, int max_jobs) {
    if (!iq4_vnni_available()) fail("needs AVX512 VNNI");
    if (g.n_embd % 512 || g.ffn_expert % 512) fail("geometry not chunkable");
    auto * p = new PrefillCpu;
    p->g = g;
    p->max_rows = max_rows;
    p->max_jobs = max_jobs;
    p->y.reserve(max_rows, g.n_embd);
    p->mid.reserve(max_jobs, g.ffn_expert);
    p->midf.reserve((size_t) max_jobs * g.ffn_expert * 4);
    return p;
}

void prefill_cpu_destroy(PrefillCpu * p) { delete p; }
PrefillCpuTiming prefill_cpu_timing(const PrefillCpu * p) { return p->tm; }
void prefill_cpu_set_r8_tiles(PrefillCpu * p, bool on) { p->r8_tiles = on; }
void prefill_cpu_set_panels(PrefillCpu * p, bool on) { p->panels = on; }

void prefill_cpu_quantize(PrefillCpu * p, Pool & pool, const float * x, int n_rows, const int * rows) {
    if (n_rows < 0 || n_rows > p->max_rows) fail("rows out of range");
    const int D = p->g.n_embd;
    constexpr int kPer = 16;
    pool.run((n_rows + kPer - 1) / kPer, [&](int item) {
        const int r1 = std::min(n_rows, (item + 1) * kPer);
        for (int r = item * kPer; r < r1; ++r) quantize_row(x + (size_t) (rows ? rows[r] : r) * D, D, p->y, r);
    });
}

void prefill_cpu_dot(const uint8_t * rows, bool xs, bool r8, int width, const float * x, int n, float * out, int8_t * q,
                     float * d) {
    if (!iq4_vnni_available()) fail("needs AVX512 VNNI");
    if (width % 512 || n < 1 || n > (r8 ? kTileR8 : kTile)) fail("prefill_cpu_dot: width % 512 and 1 <= n <= 5 (R8: 12)");
    Act a;
    a.reserve(n, width);
    Tok tk;
    TokR8 tr;
    for (int t = 0; t < n; ++t) {
        quantize_row(x + (size_t) t * width, width, a, t);
        if (t < kTile) {
            tk.q[t] = a.qrow(t);
            tk.d[t] = a.drow(t);
            tk.c[t] = a.crow(t);
        }
        tr.q[t] = a.qnrow(t);
        tr.d[t] = a.drow(t);
        tr.c2[t] = a.c2row(t);
        // lane-grouped -> value order (value 16 h + j of group l of a superblock at byte (4 h + j / 4) * 64 + 4 l + j % 4)
        for (int sb = 0; sb < width / 512; ++sb)
            for (int l = 0; l < 16; ++l)
                for (int v = 0; v < 32; ++v)
                    q[(size_t) t * width + sb * 512 + l * 32 + v] =
                        a.qrow(t)[sb * 512 + (4 * (v / 16) + (v % 16) / 4) * 64 + 4 * l + v % 4];
        std::memcpy(d + (size_t) t * (width / 32), a.drow(t), (width / 32) * 4);
    }
    if (r8) {   // the R8 tiles, on the row groups in place
        r8_rows_x_tokens(rows, xs, width, tr, n, out, n);
        return;
    }
    Unpacked u;
    u.reserve(kRows, width, false);
    if (xs) unpack_xs_rows(rows, width, kRows, u);
    else unpack_nl_rows(rows, width, kRows, u);
    rows_x_tokens(u, tk, n, out, n);
}

void prefill_cpu_dot_panels(const uint8_t * rows, bool xs, bool r8, int width, const float * x, int n, float * out,
                            int8_t * q, float * d) {
    if (!iq4_vnni_available()) fail("needs AVX512 VNNI");
    if (width % 512 || n < 1 || n > kTileP) fail("prefill_cpu_dot_panels: width % 512 and 1 <= n <= 16");
    Act a;
    a.reserve(n, width);
    for (int t = 0; t < n; ++t) {
        quantize_row(x + (size_t) t * width, width, a, t);
        std::memcpy(q + (size_t) t * width, a.qnrow(t), width);
        std::memcpy(d + (size_t) t * (width / 32), a.drow(t), (width / 32) * 4);
    }
    Panels pn;
    pn.reserve(kRows, width);
    if (r8) (xs ? r8_xs_to_panels : r8_nl_to_panels)(rows, width, kRows, pn);
    else {
        Unpacked u;
        u.reserve(kRows, width, false);
        (xs ? unpack_xs_rows : unpack_nl_rows)(rows, width, kRows, u);
        to_panels(u, kRows, pn);
    }
    alignas(64) float o[kTileP * kRows];
    panels_x_tokens(pn, kRows, {a.qnrow(0), a.drow(0), a.crow(0)}, n, o, kRows);
    for (int r = 0; r < kRows; ++r)
        for (int t = 0; t < n; ++t) out[r * n + t] = o[t * kRows + r];
}

PF_TARGET void prefill_cpu_swiglu(const float * gate, const float * up, float * out, int n) {
    if (n % 16) fail("prefill_cpu_swiglu: n % 16");
    for (int i = 0; i < n; i += 16)
        _mm512_storeu_ps(out + i, swiglu16(_mm512_loadu_ps(gate + i), _mm512_loadu_ps(up + i)));
}

double prefill_cpu_kernel_bench(int what, int width, int n, int reps) {
    if (!iq4_vnni_available()) fail("needs AVX512 VNNI");
    // 32 IQ4_XS rows of random bytes (d kept finite), n random tokens; best of reps calls, ns per call
    std::vector<uint8_t> rows((size_t) kRows * (width / 256) * kXsBlock);
    uint32_t h = 12345;
    for (auto & b : rows) b = (uint8_t) ((h = h * 1664525u + 1013904223u) >> 24) & 0x7F;
    std::vector<float> x((size_t) n * width);
    for (auto & v : x) v = (float) ((h = h * 1664525u + 1013904223u) >> 8) / 16777216.0f - 0.5f;
    Act a;
    a.reserve(n, width);
    Tok tk{};
    for (int t = 0; t < n; ++t) {
        quantize_row(x.data() + (size_t) t * width, width, a, t);
        if (t < kTile) {
            tk.q[t] = a.qrow(t);
            tk.d[t] = a.drow(t);
            tk.c[t] = a.crow(t);
        }
    }
    // SwiGLU input: tile-like values (|v| up to ~12, both clamps hit)
    alignas(64) float sg[kRows * kTileP], su[kRows * kTileP], so[kRows * kTileP] = {};
    for (int i = 0; i < kRows * kTileP; ++i) {
        sg[i] = 24.0f * x[i % x.size()];
        su[i] = 24.0f * x[(i * 7 + 3) % x.size()];
    }
    Unpacked u;
    u.reserve(kRows, width, false);
    Panels pn;
    pn.reserve(kRows, width);
    unpack_xs_rows(rows.data(), width, kRows, u);   // the tiles read real weights and scales (no denormal assists)
    to_panels(u, kRows, pn);
    alignas(64) float o[kTileP * kRows] = {};
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        switch (what) {
            case 0: rows_x_tokens(u, tk, std::min(n, kTile), o, kTile); break;       // GGUF tile, unpacked rows
            case 1: panels_x_tokens(pn, kRows, {a.qnrow(0), a.drow(0), a.crow(0)}, n, o, kRows); break;
            case 2: unpack_xs_rows(rows.data(), width, kRows, u); break;             // unpack 32 rows
            case 3: unpack_xs_rows(rows.data(), width, kRows, u); to_panels(u, kRows, pn); break;   // + transpose
            case 6: r8_xs_to_panels(rows.data(), width, kRows, pn); break;   // R8 straight to panels (same bytes)
            case 4:   // scalar SwiGLU, 32 rows x n tokens
                for (int i = 0; i < kRows * n; ++i) so[i] = expert_swiglu_clamp(sg[i], su[i]);
                break;
            default: swiglu_tile(sg, su, n, n, so, kRows); break;   // vector SwiGLU, 32 rows x n tokens
        }
        best = std::min(best, now_ms() - t0);
    }
    volatile float sink = so[0] + o[0];
    (void) sink;
    return best * 1e6;
}

void prefill_cpu_run(PrefillCpu * p, Pool & pool, const CpuExpertJobs & jobs, int n_rows, float * acc) {
    const Geometry & g = p->g;
    const int D = g.n_embd, F = g.ffn_expert, U = jobs.experts(), J = jobs.jobs();
    if (n_rows < 0 || n_rows > p->max_rows) fail("rows out of range");
    if (J > p->max_jobs) fail("too many jobs");
    if ((int) jobs.start.size() != U + 1 || jobs.start[U] != J || (int) jobs.wt.size() != J) fail("bad job lists");
    const size_t gb = (size_t) (D / 256) * kXsBlock, db = (size_t) (F / 32) * kNlBlock;
    float * midf = static_cast<float *>(p->midf.p);

    // the panel experts' y rows in job order (each copied once; the phase 2 mid rows are in job order already)
    const double t0 = now_ms();
    p->pbase.assign(U, -1);
    int npj = 0;
    for (int u = 0; u < U && p->panels; ++u)
        if (jobs.start[u + 1] - jobs.start[u] >= kPanelMin) {
            p->pbase[u] = npj;
            npj += jobs.start[u + 1] - jobs.start[u];
        }
    if (npj > 0) {
        p->yj.reserve(npj, D);
        constexpr int kPerCopy = 64;
        pool.run((J + kPerCopy - 1) / kPerCopy, [&](int item) {
            const int j0 = item * kPerCopy, j1 = std::min(J, j0 + kPerCopy);
            // the experts covering [j0, j1): start is sorted
            int u = (int) (std::upper_bound(jobs.start.begin(), jobs.start.end(), j0) - jobs.start.begin()) - 1;
            for (int j = j0; j < j1; ++j) {
                while (jobs.start[u + 1] <= j) ++u;
                if (p->pbase[u] >= 0) p->yj.copy(p->pbase[u] + j - jobs.start[u], p->y, jobs.row[j]);
            }
        });
    }

    // phase 1: gate | up rows of every expert over its tokens -> clamped SwiGLU mid[job][F]
    const int c1 = F / kRows;   // even: item u * c1 + c is on node c % nodes
    if (U > 0)
        pool.run(U * c1, [&](int item) {
            const int u = item / c1, r0 = (item % c1) * kRows;
            const ExpertRef & e = jobs.refs[u];
            const uint8_t * gate = static_cast<const uint8_t *>(e.gate_up) + r0 * gb;
            const uint8_t * up = (e.up ? static_cast<const uint8_t *>(e.up)
                                       : static_cast<const uint8_t *>(e.gate_up) + (size_t) F * gb) + r0 * gb;
            const int js = jobs.start[u], m = jobs.start[u + 1] - js;
            // R8 host copy (layout 1): r0 is a multiple of kRows, so r0 * gb starts a group of 8 rows
            const auto unpack_gu = e.layout == 1 ? unpack_xs_rows_r8 : unpack_xs_rows;
            // many tokens: the rows unpacked once, transposed to panels
            if (p->pbase[u] >= 0) {
                tl_pa.reserve(kRows, D);
                tl_pb.reserve(kRows, D);
                if (e.layout == 1) {   // R8: straight to the panels
                    r8_xs_to_panels(gate, D, kRows, tl_pa);
                    r8_xs_to_panels(up, D, kRows, tl_pb);
                } else {
                    tl_a.reserve(kRows, D, false);
                    tl_b.reserve(kRows, D, false);
                    unpack_xs_rows(gate, D, kRows, tl_a);
                    unpack_xs_rows(up, D, kRows, tl_b);
                    to_panels(tl_a, kRows, tl_pa);
                    to_panels(tl_b, kRows, tl_pb);
                }
                alignas(64) float gv[kTileP * kRows], uv[kTileP * kRows];
                for (int j = 0; j < m;) {
                    const int n = tile_size(j, m, kTileRun), r = p->pbase[u] + j;
                    const RowsP x{p->yj.qnrow(r), p->yj.drow(r), p->yj.crow(r)};
                    panels_x_tokens(tl_pa, kRows, x, n, gv, kRows);
                    panels_x_tokens(tl_pb, kRows, x, n, uv, kRows);
                    swiglu_tile_t(gv, uv, n, midf + (size_t) (js + j) * F + r0, F);
                    j += n;
                }
                return;
            }
            // R8 host copy, few tokens: the tiles read the row groups in place (r0 starts a group)
            if (e.layout == 1 && m <= kR8MaxGateUp && p->r8_tiles) {
                float gv[kRows * kTileR8], uv[kRows * kTileR8];
                for (int j = 0; j < m;) {
                    const int n = tile_size(j, m, kTileR8);
                    TokR8 x;
                    for (int t = 0; t < n; ++t) {
                        const int row = jobs.row[js + j + t];
                        x.q[t] = p->y.qnrow(row);
                        x.d[t] = p->y.drow(row);
                        x.c2[t] = p->y.c2row(row);
                    }
                    r8_rows_x_tokens(gate, true, D, x, n, gv, kTileR8);
                    r8_rows_x_tokens(up, true, D, x, n, uv, kTileR8);
                    swiglu_tile(gv, uv, kTileR8, n, midf + (size_t) (js + j) * F + r0, F);
                    j += n;
                }
                return;
            }
            tl_a.reserve(kRows, D, e.layout == 1);
            tl_b.reserve(kRows, D, e.layout == 1);
            unpack_gu(gate, D, kRows, tl_a);
            unpack_gu(up, D, kRows, tl_b);
            float gv[kRows * kTile], uv[kRows * kTile];
            for (int j = 0; j < m;) {
                const int n = tile_size(j, m);
                Tok x;
                for (int t = 0; t < n; ++t) {
                    const int row = jobs.row[js + j + t];
                    x.q[t] = p->y.qrow(row);
                    x.d[t] = p->y.drow(row);
                    x.c[t] = p->y.crow(row);
                }
                rows_x_tokens(tl_a, x, n, gv, kTile);
                rows_x_tokens(tl_b, x, n, uv, kTile);
                swiglu_tile(gv, uv, kTile, n, midf + (size_t) (js + j) * F + r0, F);
                j += n;
            }
        });
    const double t1 = now_ms();
    constexpr int kPer = 16;
    pool.run((J + kPer - 1) / kPer, [&](int item) {
        const int j1 = std::min(J, (item + 1) * kPer);
        for (int j = item * kPer; j < j1; ++j) quantize_row(midf + (size_t) j * F, F, p->mid, j);
    });

    // phase 2: down rows; item = 32-row chunk c over every expert in order (each expert's down rows unpacked once)
    const double t2 = now_ms();
    const int c2 = D / kRows;   // even: item % nodes == c % nodes
    pool.run(c2, [&](int c) {
        const int r0 = c * kRows;
        for (int t = 0; t < n_rows; ++t) std::memset(acc + (size_t) t * D + r0, 0, kRows * 4);
        alignas(64) float dv[kRows * std::max(kTileR8, kTileP)];
        for (int u = 0; u < U; ++u) {
            const int js = jobs.start[u], m = jobs.start[u + 1] - js;
            if (!m) continue;
            const uint8_t * down = static_cast<const uint8_t *>(jobs.refs[u].down) + r0 * db;
            // many tokens: panels (the mid rows are in job order)
            if (p->pbase[u] >= 0) {
                tl_pa.reserve(kRows, F);
                if (jobs.refs[u].layout == 1) r8_nl_to_panels(down, F, kRows, tl_pa);
                else {
                    tl_a.reserve(kRows, F, false);
                    unpack_nl_rows(down, F, kRows, tl_a);
                    to_panels(tl_a, kRows, tl_pa);
                }
                for (int j = 0; j < m;) {
                    const int n = tile_size(j, m, kTileRun);
                    panels_x_tokens(tl_pa, kRows, {p->mid.qnrow(js + j), p->mid.drow(js + j), p->mid.crow(js + j)}, n,
                                    dv, kRows);
                    for (int t = 0; t < n; ++t) {
                        float * a = acc + (size_t) jobs.row[js + j + t] * D + r0;
                        const float w = jobs.wt[js + j + t];
                        for (int i = 0; i < kRows; ++i) a[i] += w * dv[t * kRows + i];
                    }
                    j += n;
                }
                continue;
            }
            // R8 host copy, few tokens: tiles in place
            if (jobs.refs[u].layout == 1 && m <= kR8MaxDown && p->r8_tiles) {
                for (int j = 0; j < m;) {
                    const int n = tile_size(j, m, kTileR8);
                    TokR8 x;
                    for (int t = 0; t < n; ++t) {
                        x.q[t] = p->mid.qnrow(js + j + t);
                        x.d[t] = p->mid.drow(js + j + t);
                        x.c2[t] = p->mid.c2row(js + j + t);
                    }
                    r8_rows_x_tokens(down, false, F, x, n, dv, kTileR8);
                    for (int t = 0; t < n; ++t) {
                        float * a = acc + (size_t) jobs.row[js + j + t] * D + r0;
                        const float w = jobs.wt[js + j + t];
                        for (int i = 0; i < kRows; ++i) a[i] += w * dv[i * kTileR8 + t];
                    }
                    j += n;
                }
                continue;
            }
            tl_a.reserve(kRows, F, jobs.refs[u].layout == 1);   // sets the row stride (no allocation after the first)
            (jobs.refs[u].layout == 1 ? unpack_nl_rows_r8 : unpack_nl_rows)(down, F, kRows, tl_a);
            for (int j = 0; j < m;) {
                const int n = tile_size(j, m);
                Tok x;
                for (int t = 0; t < n; ++t) {
                    x.q[t] = p->mid.qrow(js + j + t);
                    x.d[t] = p->mid.drow(js + j + t);
                    x.c[t] = p->mid.crow(js + j + t);
                }
                rows_x_tokens(tl_a, x, n, dv, kTile);
                for (int t = 0; t < n; ++t) {
                    float * a = acc + (size_t) jobs.row[js + j + t] * D + r0;
                    const float w = jobs.wt[js + j + t];
                    for (int i = 0; i < kRows; ++i) a[i] += w * dv[i * kTile + t];
                }
                j += n;
            }
        }
    });
    p->tm = {t1 - t0, t2 - t1, now_ms() - t2};
}

}  // namespace strata::glm
