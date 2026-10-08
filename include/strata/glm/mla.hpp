#pragma once
// GLM-5.3-Flash MLA attention + DSA lightning indexer, one trunk layer (blocks 3, 7, ..., 43) or the MTP block.
// Math: docs/glm5next-spec.md section 3 "MLA + DSA block" and "Lightning indexer / k-pool selection".
//
// Caches (one sequence, absolute positions starting at 0), all F16 like llama.cpp's default type_k:
//   latent  [max_ctx][512]       kv_cmpr row per token (K and V of the latent attention)
//   idx_kg  [max_ctx][256]       indexer key (LayerNorm'd) | pool gate, per token
//   pooled  [max_ctx/4][128]     pooled indexer key per complete pool of 4 tokens (pool p = positions 4p..4p+3)
// A pool is computed once, when its last member is written. Truncating to length n keeps pools p < n/4 and
// rows < n; a later append recomputes any pool it completes, so a cut to any position is O(1) and exact.
// idx_kg is only read to build pools, so the device keeps the last 1024 rows (a ring tagged with positions) and
// every row also goes to mapped pinned host memory, which the pooling reads for rows the ring no longer holds
// (after a cut far back): 512 B per token per layer leave VRAM, mla_cache_bytes counts the device part only.
// MlaLatent::Q8 stores the latent as int8 with one F16 scale per 64 dims (max |x| / 127, rounded to F16): 528 B per
// row instead of 1024. The attention dequantizes each row to F16 as it stages it (the F16 cache holds rounded F32,
// the Q8 one rounded F16 products q * scale). Measured on long-context logits (64K, teacher-forced): about one unit of
// the engine's own numeric noise (the F32 -> VNNI routed-miss switch); needle retrieval at 95K / 113K unchanged.

#include <cstddef>
#include <cstdint>

#include "strata/glm/types.hpp"

namespace strata::glm {

struct MlaWeights {
    WeightRef q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, out;
    WeightRef idx_q_b, idx_k, idx_k_norm_w, idx_k_norm_b, idx_proj, idx_gate, idx_ape;
};

struct MlaCache;
enum class MlaLatent { F16, Q8 };
struct MatmulInput;   // strata/glm/matmul.hpp

// Externally supplied attention cells (device memory): row t of `sel` (stride sel_ld) holds nsel[t] cache positions
// for token t. Used by the MTP draft chain with GGUF index_share_mtp (llama.cpp PR #27917): draft steps 1+ skip the
// indexer scoring and top-k and attend over step 0's selection verbatim (mla_taps sel / nsel / sel_ld copied out),
// with no extra tail cells. Projections, cache writes and pooling run as usual. Rows must stay valid until the
// stream reaches the attention; positions must be < the cache size after the batch; nsel[t] <= 2051.
struct MlaSelection {
    const int32_t * sel = nullptr;
    const int32_t * nsel = nullptr;
    int             sel_ld = 0;
};

// A cache owns a side stream (forked from and joined back into the caller's stream inside each call, also under
// graph capture), so one cache serves one caller stream at a time.
MlaCache * mla_cache_create(const Geometry &, int64_t max_ctx, MlaLatent latent = MlaLatent::F16);
void       mla_cache_destroy(MlaCache *);
void       mla_cache_truncate(MlaCache *, int64_t n_tokens);   // keep positions [0, n_tokens)
int64_t    mla_cache_size(const MlaCache *);                   // tokens currently cached
int64_t    mla_cache_capacity(const MlaCache *);
size_t     mla_cache_bytes(const Geometry &, int64_t max_ctx, MlaLatent latent = MlaLatent::F16);
size_t     mla_cache_host_bytes(int64_t max_ctx);   // its mapped pinned host part (512 B per token)

// Runs the attention branch for b.T tokens at positions [b.pos0, b.pos0 + b.T). b.pos0 must be <= the cache size;
// positions >= b.pos0 already in the cache are overwritten (implicit truncate). Afterwards the cache holds
// b.pos0 + b.T tokens. x_norm / out are device F32 [T][4096]; scratch is device memory of mla_scratch_bytes().
// Long batches are processed in internal sub-chunks, so scratch does not grow with T beyond the chunk size.
// shared: optional external selection (MlaSelection, rows for the batch's T tokens); the score / top / sel taps are
// then not written.
// xin: optional x_norm already prepared for glm_matmul ([T][4096], batch 1, e.g. the Q8 the trunk's hc_pre emits
// via matmul_input_q8); it replaces matmul_prepare(x_norm) for the five x projections. T <= 512 with it.
void   mla_forward(const Geometry &, const MlaWeights &, MlaCache *, const Batch &,
                   const float * x_norm, float * out, void * scratch, cudaStream_t,
                   const MlaSelection * shared = nullptr, const MatmulInput * xin = nullptr);
size_t mla_scratch_bytes(const Geometry &, int max_T, int64_t max_ctx);

// CUDA-graph capturable variant for the engine's captured decode / verify step. Same math as mla_forward, but the
// batch position is read from device memory (*d_pos0, an int64 the engine updates before each replay; one scalar
// can serve every MLA layer) and every launch configuration depends only on T and the cache capacity (blocks past
// the live length exit early), so a graph captured once replays at any position. 1 <= T <= 512; T <= 32 keeps
// every projection on glm_matmul's GEMV path (larger T needs matmul_init() before the capture). Create the cache
// before capturing (it sets kernel attributes). Preconditions as mla_forward (pos0 <= size, pos0 + T <= capacity)
// are the caller's: they cannot be checked on the host. The host-side size is not updated: after the step, call
// mla_cache_set_size(c, pos0 + T) so mla_forward, mla_cache_truncate and mla_taps see the new size.
void mla_forward_dev(const Geometry &, const MlaWeights &, MlaCache *, int T, const int64_t * d_pos0,
                     const float * x_norm, float * out, void * scratch, cudaStream_t,
                     const MlaSelection * shared = nullptr, const MatmulInput * xin = nullptr);
void mla_cache_set_size(MlaCache *, int64_t n_tokens);   // host-side size after mla_forward_dev; n <= capacity

// ---- host copies of a prefix (the engine's prefix cache) -------------------------------------------------
// A prefix of n tokens is rows blobs for consecutive ranges [0, a1), [a1, a2), ..., [ak, n) plus one tail blob at n.
// A rows blob [a, b) holds the latent rows (and Q8 scales) of positions [a, b) and the pooled rows of pools
// [a/4, b/4) (the pools completed inside the range: a chain of ranges holds every complete pool once, for any a).
// The tail blob holds the key | gate rows of the incomplete last pool (positions [4 (n/4), n), < 4 rows of 256
// halves), the only key | gate rows a later append reads. Blobs carry a header (latent type, range): an import
// whose blob does not match the cache or the range copies nothing and returns false.
// All copies are cudaMemcpyAsync on the given stream (host memory should be pinned); an export is complete when the
// stream reaches it, an import's blob must stay untouched until then.
// Import: rows blobs in order, each with a <= the cache size (the cache keeps its own rows below a; the size becomes
// b), then the tail blob at the final size, before any other call on the cache (forward, truncate, export). The
// import invalidates the device key | gate ring (it may hold another sequence's rows) and resets mla_taps (T = 0).
// The key | gate rows of an imported range are not held (only its tail): a later cut to a position m with m % 4 != 0
// whose pool [4 (m/4), m) falls in such a range would need them; mla_cache_can_truncate tells (truncate aborts). The
// cache tracks one interval for all imported ranges, so it may also refuse a cut whose rows are held. After an
// import, cut below the imported end only through mla_cache_truncate (checked): mla_forward_dev cannot check.
// Bit identity: the import reproduces the source's state when the rows the cache keeps below a are bitwise the
// source's (same tokens computed the same way, e.g. imported from the same blobs); rows of the same tokens computed
// with another batch shape differ in the last bits (the pool a/4 then mixes them with the blob's pooled row).
size_t mla_cache_rows_bytes(const MlaCache *, int64_t a, int64_t b);
void   mla_cache_export_rows(const MlaCache *, int64_t a, int64_t b, void * host, cudaStream_t);   // b <= size
bool   mla_cache_import_rows(MlaCache *, int64_t a, int64_t b, const void * host, cudaStream_t);
size_t mla_cache_tail_bytes(const MlaCache *);
void   mla_cache_export_tail(const MlaCache *, int64_t n, void * host, cudaStream_t);   // n <= size
bool   mla_cache_import_tail(MlaCache *, int64_t n, const void * host, cudaStream_t);   // n == size
bool   mla_cache_can_truncate(const MlaCache *, int64_t n_tokens);

// ---- introspection for parity tests and the engine's dump path ------------------------------------------
// Device pointers into the scratch of the LAST sub-chunk mla_forward processed on this cache (token-major rows).
// After mla_forward_dev, pos0 is known once mla_cache_set_size has been called. After an import, T = 0.
struct MlaTaps {
    int     T = 0;            // tokens in the last sub-chunk
    int64_t pos0 = 0;         // position of its first token
    int     n_pool_ld = 0;    // row stride of `score`
    const float *   qr = nullptr;     // [T][1536]   q_resid (after RMSNorm)
    const float *   q_abs = nullptr;  // [T][64][512]
    const float *   kv = nullptr;     // [T][512]    kv_cmpr (F32, before the F16 cache)
    const float *   iq = nullptr;     // [T][32][128]
    const float *   ik = nullptr;     // [T][128]
    const float *   ig = nullptr;     // [T][128]
    const float *   iw = nullptr;     // [T][32]     indexer weights (already / sqrt(32*128))
    const float *   score = nullptr;  // [T][n_pool_ld] pool scores, valid for pools visible to each token
    const int32_t * sel = nullptr;    // [T][sel_ld] selected cell positions; first nsel[t] valid
    const int32_t * nsel = nullptr;   // [T]
    int             sel_ld = 0;
    const int32_t * top = nullptr;    // [T][512]    selected pools (first min(nv,512) valid); sorted by descending
                                      //             score when nv <= 512, unordered when pruning
    const float *   o = nullptr;      // [T][64][512] attention output in latent space (before Wv_b)
    const float *   v = nullptr;      // [T][64][256] per-head output after Wv_b
};
MlaTaps mla_taps(const Geometry &, const MlaCache *, void * scratch);
const void * mla_cache_pooled(const MlaCache *);   // device F16 [capacity/4][128], for tests
MlaLatent    mla_cache_latent_type(const MlaCache *);
const void * mla_cache_latent(const MlaCache *);         // device F16 [capacity][512], or int8 [capacity][512] (Q8)
const void * mla_cache_latent_scale(const MlaCache *);   // Q8: device F16 [capacity][8], one per 64 dims; else null
// rows [pos0, pos0 + n) as the attention sees them (F16 [n][512], dequantized for Q8) into device memory `out`
void mla_cache_latent_f16(const MlaCache *, int64_t pos0, int64_t n, void * out, cudaStream_t);
const void * mla_cache_kg(const MlaCache *);       // mapped host F16 [capacity][256] indexer key | gate
// Benchmarks: fill positions [0, n) with pseudo-random latent / key / gate rows (N(0,1)-like) and pool them
// with the real pooling kernel, as if n tokens had been prefilled.
void mla_cache_fill_random(MlaCache *, const MlaWeights &, int64_t n, uint64_t seed, cudaStream_t);

}  // namespace strata::glm
