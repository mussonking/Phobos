// src/glm/mtp/mtp.cu - see include/strata/glm/mtp.hpp.
#include "strata/glm/mtp.hpp"
#include "mtp_internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "forward.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/ops.hpp"

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("mtp: ") + what + ": " + cudaGetErrorString(e));
}

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

constexpr int kSelCap = 4096;   // >= MlaTaps::sel_ld
constexpr int kArgmaxThreads = 256;
constexpr int kArgmaxBlocks = 160;

// Per-block (max, first index of the max) over a strided slice, then one block over the partials. Ties resolve to
// the lowest index, like a sequential scan.
__device__ __forceinline__ void better(float & v, int & i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

__device__ void block_argmax(float & v, int & i) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) better(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = v; si[warp] = i; }
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x >> 5;
        v = lane < nw ? sv[lane] : -INFINITY;
        i = lane < nw ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) better(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
    }
}

__global__ void __launch_bounds__(kArgmaxThreads) argmax_partial_kernel(const float * __restrict__ x, int n,
                                                                         float * pv, int * pi) {
    float v = -INFINITY;
    int i = 0x7fffffff;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) better(v, i, x[j], j);
    block_argmax(v, i);
    if (threadIdx.x == 0) { pv[blockIdx.x] = v; pi[blockIdx.x] = i; }
}

__global__ void __launch_bounds__(kArgmaxThreads) argmax_final_kernel(const float * __restrict__ pv,
                                                                       const int * __restrict__ pi, int n, int * out) {
    float v = -INFINITY;
    int i = 0x7fffffff;
    for (int j = threadIdx.x; j < n; j += blockDim.x) better(v, i, pv[j], pi[j]);
    block_argmax(v, i);
    if (threadIdx.x == 0) *out = i;
}

// a += b over n floats.
__global__ void add_kernel(float * __restrict__ a, const float * __restrict__ b, int64_t n) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) a[j] = a[j] + b[j];
}

mtp::AttnOverride g_attn_override;
mtp::ChainOverride g_chain_override;

void tap(const ForwardHooks & hk, int il, const char * node, const float * dev, int64_t n) {
    if (hk.tap) hk.tap(il, node, dev, n);
}

}  // namespace

struct MtpState {
    int il = 0;
    int max_T = 0;
    int64_t max_ctx = 0;
    bool share = false;
    MlaCache * cache = nullptr;
    void * arena = nullptr;
    float * cat = nullptr;    // [T][8192] e | h
    float * catn = nullptr;   // [T][8192] RMSNorm(e) * enorm | RMSNorm(h) * hnorm
    float * u = nullptr;      // [T][4096] eh_proj, later (moe + shexp) + f_in
    float * fin = nullptr;    // [T][4096] u + attention
    float * hout = nullptr;   // [T][4096] h_nextn
    const float * hlast = nullptr;   // last row of hout after the last forward
    float * pv = nullptr;     // argmax partials
    int * pi = nullptr;
    int * d_tok = nullptr;
    int32_t * sel = nullptr;  // shared selection of the first draft step (index_share_mtp)
    int32_t * nsel = nullptr;
    int sel_ld = 0;
    int * h_tok = nullptr;    // pinned
    float * e_host = nullptr; // pinned [T][4096]
    void * mla_scratch = nullptr;
    void * mm = nullptr;
    size_t mm_bytes = 0;
    WeightRef head;                 // the draft head: m.output, or the gathered subset rows
    void * head_buf = nullptr;      // device, the subset head (null: full head)
    std::vector<int32_t> head_ids;  // subset row -> token id (empty: full head)
};

MtpState * mtp_create(const Model & m, int64_t max_ctx, int max_T, MlaLatent latent) {
    const Geometry & g = m.g;
    if (g.n_layer_mtp != 1 || (int) m.layers.size() <= g.n_layer) throw std::runtime_error("mtp: model has no MTP block");
    const LayerWeights & L = m.layers[g.n_layer];
    if (!L.mtp || !L.eh_proj.data || !L.enorm.data || !L.hnorm.data || !L.shared_head_norm.data)
        throw std::runtime_error("mtp: MTP block weights not loaded (LoadOptions.load_mtp)");
    if (max_T < 1) throw std::runtime_error("mtp: max_T < 1");
    auto * s = new MtpState();
    s->il = g.n_layer;
    s->max_T = max_T;
    s->max_ctx = max_ctx;
    s->cache = mla_cache_create(g, max_ctx, latent);
    const size_t T = (size_t) max_T, D = (size_t) g.n_embd;
    s->mm_bytes = align256(std::max(matmul_scratch_bytes(L.eh_proj, max_T), matmul_scratch_bytes(m.output, 1)));
    const size_t mla_bytes = align256(mla_scratch_bytes(g, max_T, max_ctx));
    struct Buf { void ** p; size_t bytes; };
    const std::vector<Buf> bufs = {
        {(void **) &s->cat, T * 2 * D * 4},
        {(void **) &s->catn, T * 2 * D * 4},
        {(void **) &s->u, T * D * 4},
        {(void **) &s->fin, T * D * 4},
        {(void **) &s->hout, T * D * 4},
        {(void **) &s->pv, kArgmaxBlocks * 4},
        {(void **) &s->pi, kArgmaxBlocks * 4},
        {(void **) &s->d_tok, 4},
        {(void **) &s->sel, (size_t) kSelCap * 4},
        {(void **) &s->nsel, 4},
        {&s->mla_scratch, mla_bytes},
        {&s->mm, s->mm_bytes},
    };
    size_t total = 0;
    for (const auto & b : bufs) total += align256(b.bytes);
    ck(cudaMalloc(&s->arena, total), "arena");
    char * p = static_cast<char *>(s->arena);
    for (const auto & b : bufs) {
        *b.p = p;
        p += align256(b.bytes);
    }
    ck(cudaMallocHost(&s->h_tok, 4), "pinned token");
    ck(cudaMallocHost(&s->e_host, T * D * 4), "pinned embeddings");
    s->head = m.output;
    return s;
}

void mtp_destroy(MtpState * s) {
    if (!s) return;
    mla_cache_destroy(s->cache);
    cudaFree(s->arena);
    cudaFreeHost(s->h_tok);
    cudaFreeHost(s->e_host);
    cudaFree(s->head_buf);
    delete s;
}

void mtp_truncate(MtpState * s, int64_t n) { mla_cache_truncate(s->cache, n); }
int64_t mtp_size(const MtpState * s) { return mla_cache_size(s->cache); }
void mtp_set_share_selection(MtpState * s, bool on) { s->share = on; }
const float * mtp_last_hidden(const MtpState * s) { return s->hlast; }
MlaCache * mtp_cache(MtpState * s) { return s->cache; }

void mtp_set_draft_vocab(MtpState * s, const Model & m, const int32_t * ids, int n, cudaStream_t st) {
    ck(cudaStreamSynchronize(st), "draft vocab");
    cudaFree(s->head_buf);
    s->head_buf = nullptr;
    s->head = m.output;
    s->head_ids.clear();
    if (n <= 0) return;
    // gathered from the GGUF rows (m.output may be repacked), then prepared like the loader does
    const HostTensor src = m.gguf->tensor("output.weight");
    const uint64_t n_rows = src.shape.at(1), row = src.nbytes / n_rows;
    std::vector<uint8_t> rows((size_t) n * row);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || (uint64_t) ids[i] >= n_rows) throw std::runtime_error("mtp_set_draft_vocab: id out of range");
        std::memcpy(rows.data() + (size_t) i * row, src.data + (size_t) ids[i] * row, row);
    }
    HostTensor sub = src;
    sub.shape[1] = (uint64_t) n;
    sub.nbytes = (uint64_t) n * row;
    const bool repack = m.output.type == kTypeQ8_0R;
    void * raw = nullptr;
    ck(cudaMalloc(&raw, rows.size()), "draft vocab head");
    ck(cudaMemcpyAsync(raw, rows.data(), rows.size(), cudaMemcpyHostToDevice, st), "draft vocab upload");
    if (repack) {
        const WeightRef q = sub.ref(raw);
        ck(cudaMalloc(&s->head_buf, matmul_repack_bytes(q)), "draft vocab repack");
        s->head = matmul_repack(q, s->head_buf, st);
        ck(cudaStreamSynchronize(st), "draft vocab repack");
        cudaFree(raw);
    } else {
        s->head_buf = raw;
        s->head = sub.ref(raw);
        ck(cudaStreamSynchronize(st), "draft vocab upload");
    }
    if (matmul_scratch_bytes(s->head, 1) > s->mm_bytes) throw std::runtime_error("mtp_set_draft_vocab: scratch");
    s->head_ids.assign(ids, ids + n);
}

namespace {

// One forward of the MTP block over T rows (embeddings already in s->cat[:, :4096], h in s->cat[:, 4096:]).
// Writes h_nextn to s->hout and, if logits, the head of the last row to ws.logits.
void mtp_block(MtpState * s, const Model & m, Workspace & ws, const ForwardHooks & hk, int T, int64_t pos0,
               bool logits, const MlaSelection * shared, cudaStream_t st) {
    const Geometry & g = m.g;
    const LayerWeights & L = m.layers[s->il];
    const int D = g.n_embd, il = s->il;
    const int64_t TD = (int64_t) T * D;

    rms_norm(s->cat, nullptr, L.enorm, s->catn, nullptr, T, D, 2 * D, g.rms_eps, st);
    rms_norm(s->cat + D, nullptr, L.hnorm, s->catn + D, nullptr, T, D, 2 * D, g.rms_eps, st);
    glm_matmul(L.eh_proj, s->catn, s->u, T, s->mm, s->mm_bytes, st);
    tap(hk, il, "mtp_eh_proj", s->u, TD);

    rms_norm(s->u, nullptr, L.attn_norm, ws.y, nullptr, T, D, 0, g.rms_eps, st);
    tap(hk, il, "mtp_attn_norm", ws.y, TD);
    const Batch b{T, pos0};
    if (g_attn_override) {
        g_attn_override(ws.y, ws.a, T, st);
        mla_cache_set_size(s->cache, pos0 + T);   // keep the position bookkeeping of a real forward
    } else mla_forward(g, L.mla_w, s->cache, b, ws.y, ws.a, s->mla_scratch, st, shared);
    tap(hk, il, "mtp_attn_out", ws.a, TD);

    rms_norm(ws.a, s->u, L.ffn_norm, ws.y, s->fin, T, D, 0, g.rms_eps, st);
    tap(hk, il, "mtp_ffn_inp", s->fin, TD);
    tap(hk, il, "mtp_ffn_norm", ws.y, TD);

    const float * shexp = ffn_branch(m, ws, il, T, hk, st);
    if (!shexp) throw std::runtime_error("mtp: block is not MoE");
    add_kernel<<<(unsigned) ((TD + 255) / 256), 256, 0, st>>>(ws.a, shexp, TD);
    ck(cudaGetLastError(), "add");
    rms_norm(ws.a, s->fin, L.shared_head_norm, s->hout, s->u, T, D, 0, g.rms_eps, st);
    tap(hk, il, "mtp_post_ffn", s->u, TD);
    tap(hk, -1, "h_nextn", s->hout, TD);
    s->hlast = s->hout + (int64_t) (T - 1) * D;

    if (logits) {
        glm_matmul(s->head, s->hout + (int64_t) (T - 1) * D, ws.logits, 1, s->mm, s->mm_bytes, st);
        tap(hk, -1, "result_output", ws.logits, s->head.ne[1]);
    }
}

int32_t argmax_sync(MtpState * s, const float * x, int n, cudaStream_t st) {
    argmax_partial_kernel<<<kArgmaxBlocks, kArgmaxThreads, 0, st>>>(x, n, s->pv, s->pi);
    argmax_final_kernel<<<1, kArgmaxThreads, 0, st>>>(s->pv, s->pi, kArgmaxBlocks, s->d_tok);
    ck(cudaGetLastError(), "argmax");
    ck(cudaMemcpyAsync(s->h_tok, s->d_tok, 4, cudaMemcpyDeviceToHost, st), "argmax copy");
    ck(cudaStreamSynchronize(st), "argmax sync");
    return *s->h_tok;
}

// e rows of the host ids into s->cat[:, :D] (pinned staging, synchronous for the staging buffer).
void upload_embeddings(MtpState * s, const Model & m, const int32_t * ids, int T, cudaStream_t st) {
    const int D = m.g.n_embd;
    embed_rows_host(m, ids, T, s->e_host);
    ck(cudaMemcpy2DAsync(s->cat, 2 * D * 4, s->e_host, D * 4, D * 4, T, cudaMemcpyHostToDevice, st), "embd upload");
}

}  // namespace

void mtp_draft(MtpState * s, const Model & m, Workspace & ws, const ForwardHooks & hk, const int32_t * ids,
               const float * h_prev, int T, int64_t pos0, int k, int32_t * drafts, cudaStream_t st) {
    const Geometry & g = m.g;
    const int D = g.n_embd;
    if (T < 1 || T > s->max_T || T > ws.max_T) throw std::runtime_error("mtp_draft: T out of range");
    if (pos0 < 0 || pos0 > mla_cache_size(s->cache)) throw std::runtime_error("mtp_draft: pos0 beyond the cache");
    if (pos0 + T + std::max(k - 1, 0) > s->max_ctx) throw std::runtime_error("mtp_draft: cache capacity exceeded");

    upload_embeddings(s, m, ids, T, st);
    ck(cudaMemcpy2DAsync(s->cat + D, 2 * D * 4, h_prev, D * 4, D * 4, T, cudaMemcpyDeviceToDevice, st), "h copy");
    mtp_block(s, m, ws, hk, T, pos0, k > 0, nullptr, st);
    MlaSelection shared{};
    const bool share = s->share && k > 1 && !mtp::attn_overridden();
    if (share) {
        // the first step's selection (last row), reused verbatim by the later steps
        const MlaTaps tp = mla_taps(g, s->cache, s->mla_scratch);
        if (tp.sel_ld > kSelCap) throw std::runtime_error("mtp: selection row wider than the reserved buffer");
        ck(cudaMemcpyAsync(s->sel, tp.sel + (int64_t) (tp.T - 1) * tp.sel_ld, (size_t) tp.sel_ld * 4,
                           cudaMemcpyDeviceToDevice, st), "selection copy");
        ck(cudaMemcpyAsync(s->nsel, tp.nsel + tp.T - 1, 4, cudaMemcpyDeviceToDevice, st), "selection copy");
        shared = MlaSelection{s->sel, s->nsel, tp.sel_ld};
    }
    for (int i = 0; i < k; ++i) {
        drafts[i] = argmax_sync(s, ws.logits, (int) s->head.ne[1], st);
        if (!s->head_ids.empty()) drafts[i] = s->head_ids[drafts[i]];
        if (i + 1 == k) break;
        // chain: the draft at the next position with this step's h_nextn (last row)
        int32_t tok = drafts[i];
        ck(cudaMemcpyAsync(s->cat + D, s->hlast, D * 4, cudaMemcpyDeviceToDevice, st), "h chain");
        if (g_chain_override) g_chain_override(i + 1, tok, s->cat + D, st);
        upload_embeddings(s, m, &tok, 1, st);
        mtp_block(s, m, ws, hk, 1, pos0 + T + i, true, share ? &shared : nullptr, st);
    }
    // the staging buffers are reused by the next call
    ck(cudaStreamSynchronize(st), "mtp_draft");
}

void mtp_catch_up(MtpState * s, const Model & m, Workspace & ws, const ForwardHooks & hk, float * hrows, int T,
                  int64_t pos0, void * scratch, size_t scratch_bytes, cudaStream_t st) {
    const Geometry & g = m.g;
    const int D = g.n_embd;
    const LayerWeights & L = m.layers[s->il];
    if (T < 1 || T > ws.max_T) throw std::runtime_error("mtp_catch_up: T out of range");
    if (pos0 < 0 || pos0 > mla_cache_size(s->cache)) throw std::runtime_error("mtp_catch_up: pos0 beyond the cache");
    if (pos0 + T > s->max_ctx) throw std::runtime_error("mtp_catch_up: cache capacity exceeded");
    if (scratch_bytes < mla_scratch_bytes(g, T, s->max_ctx)) throw std::runtime_error("mtp_catch_up: scratch too small");
    if (matmul_scratch_bytes(L.eh_proj, T) > ws.mm_bytes) throw std::runtime_error("mtp_catch_up: matmul scratch");

    // the trunk's post-output_norm rows of the chunk (ws.x, left by forward_dev) -> hrows 1..T
    hc_head(g, ws.x, T, m.output_norm, hrows + D, st);
    // block 45 at large T: its T-row buffers carved from the now free mHC streams (ws.x and ws.x2, 4 T D floats
    // each), the chunk's embeddings from ws.e, MLA scratch and matmul scratch from the caller / the workspace
    MtpState big = *s;
    const size_t TD = (size_t) T * D;
    big.cat = ws.x;
    big.catn = ws.x + 2 * TD;
    big.u = ws.x2;
    big.fin = ws.x2 + TD;
    big.hout = ws.x2 + 2 * TD;
    big.mla_scratch = scratch;
    big.mm = ws.mm;
    big.mm_bytes = ws.mm_bytes;
    ck(cudaMemcpy2DAsync(big.cat, 2 * D * 4, ws.e, D * 4, D * 4, T, cudaMemcpyDeviceToDevice, st), "e rows");
    ck(cudaMemcpy2DAsync(big.cat + D, 2 * D * 4, hrows, D * 4, D * 4, T, cudaMemcpyDeviceToDevice, st), "h rows");
    mtp_block(&big, m, ws, hk, T, pos0, false, nullptr, st);
    // this block's last h_nextn into the state's own buffer (mtp_last_hidden), the chunk's last trunk h into row 0
    ck(cudaMemcpyAsync(s->hout, big.hlast, D * 4, cudaMemcpyDeviceToDevice, st), "h_nextn");
    s->hlast = s->hout;
    ck(cudaMemcpyAsync(hrows, hrows + TD, D * 4, cudaMemcpyDeviceToDevice, st), "carry");
}

namespace mtp {
void set_attn_override(AttnOverride f) { g_attn_override = std::move(f); }
bool attn_overridden() { return (bool) g_attn_override; }
void set_chain_override(ChainOverride f) { g_chain_override = std::move(f); }
}  // namespace mtp

}  // namespace strata::glm
