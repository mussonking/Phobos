#pragma once

#include "iq4_vnni.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::glm {

// R8 interleaves eight rows within their original GGUF byte range. Four groups
// still form one 32-row NUMA chunk. No padding or second expert arena is needed.
constexpr int kR8Rows = 8;
constexpr int kR8XSBlock = 136, kR8NLBlock = 18;
constexpr int kR8XSTile = 1088, kR8NLTile = 144, kR8X64 = 64;

bool r8_available();
std::size_t r8_row_bytes(bool xs, int width);
std::size_t r8_group_bytes(bool xs, int width);

// Exactly one eight-row group. Buffers contain group_bytes and must not overlap.
// FP16 scale bits, all subscales and weight nibbles round-trip byte exactly.
void r8_repack_xs(const void * src, void * dst, int width);
void r8_repack_nl(const void * src, void * dst, int width);
void r8_unpack_xs(const void * src, void * dst, int width);
void r8_unpack_nl(const void * src, void * dst, int width);

// Bounded GGUF conversion for legacy F32/eval/prefill readers. Source is the
// whole R8 plane (plane_rows divisible by 8); output is n_rows ordinary rows.
// Arbitrary first/count, including partial groups, are supported without an
// intermediate full plane or an allocation. Rejects invalid bounds/overlap.
void r8_unpack_xs_rows(const void * src, std::size_t src_bytes, int width,
                      int plane_rows, int first_row, int n_rows,
                      void * dst, std::size_t dst_bytes);
void r8_unpack_nl_rows(const void * src, std::size_t src_bytes, int width,
                      int plane_rows, int first_row, int n_rows,
                      void * dst, std::size_t dst_bytes);

struct R8Act {
    const int8_t * qs = nullptr;
    int bstride = 0;                // bytes between activation blocks' qs
    int gpb = 0;                    // groups of 32 per block: 8 Q8_K, 1 per-32
    const int64_t * corr = nullptr; // packed correction pair per group
    const float * d = nullptr;      // exact activation scale per block
    int cstride = 1;                // correction stride in int64 units
    int dstride = 1;                // scale stride in float units
};

// Caller-owned side arrays: corr[width/32], d[width/256] or d[width/32].
// Native Q8_K blocks are 292 B (F32 d); native Q8_0 blocks are 34 B (FP16 d).
// Inputs may be byte-unaligned. No quantization or activation approximation.
R8Act r8_prep_q8k(const void * q8k, int width, int64_t * corr, float * d);
R8Act r8_prep_q8_0(const void * q8_0, int width, int64_t * corr, float * d);

// x64: qs[32] | correction[8] | F32 scale[4] | native FP16 bits[2] | padding.
// Each block is cache-line isolated when the row is 64-byte aligned. Native
// entry point preserves original FP16 bits (including signed zero) at offset44.
void r8_x64_store_q8_0(const void * native_q8_0_block, void * block64);
// CUSTOM per-32 F32-scale activation, NOT native Q8_0. No hidden FP16 rounding.
// Prefer the native entry point when comparing against existing Q8_0 APIs.
void r8_x64_store(const int8_t * qs32, float scale, void * block64);
R8Act r8_x64_act(const void * row, int width);

// R8 rows, overwritten out[activation][row], n_act 1..8, n_rows divisible by8.
// weights contain n_rows * row_bytes, starting at an eight-row boundary.
// Act arrays are immutable/private-prepared and must cover width: qs advances
// by bstride per block; corr covers width/32 entries at cstride; d covers blocks
// at dstride. XS requires gpb8; NL accepts homogeneous gpb1 or gpb8.
// Integer lane sums are exact, including int8 -128. F32 reassociation differs
// from old row kernels; compare against a float64 sum-of-absolute-terms bound,
// NOT bitwise equality. The scalar fallback implements the same R8 arithmetic.
void r8_rows_xs(const void * weights, int width, const R8Act * act, int n_act,
                int n_rows, float * out, IQ4VNNIPath path = IQ4VNNIPath::automatic);
void r8_rows_nl(const void * weights, int width, const R8Act * act, int n_act,
                int n_rows, float * out, IQ4VNNIPath path = IQ4VNNIPath::automatic);

} // namespace strata::glm
