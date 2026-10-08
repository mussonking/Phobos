// src/glm/engine/forward.cpp - see forward.hpp.
#include "forward.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/glm/matmul.hpp"
#include "strata/glm/ops.hpp"

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm forward: ") + what + ": " + cudaGetErrorString(e));
}

size_t align256(size_t n) { return (n + 255) & ~size_t(255); }

HcBranch hc_attn(const LayerWeights & L) { return {L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale}; }
HcBranch hc_ffn(const LayerWeights & L) { return {L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale}; }

// Sinkhorn of the branch's combine on the side stream (it is needed only by hc_post), joined before hc_post or
// before anything that ends a graph capture segment.
void fork_comb(const Model & m, Workspace & ws, int T, cudaStream_t st) {
    ck(cudaEventRecord(ws.ev_fork, st), "fork");
    ck(cudaStreamWaitEvent(ws.side, ws.ev_fork, 0), "fork wait");
    hc_comb(m.g, ws.hcp, T, ws.side);
    ck(cudaEventRecord(ws.ev_join, ws.side), "join");
    ws.side_pending = true;
}
void join_comb(Workspace & ws, cudaStream_t st) {
    if (!ws.side_pending) return;
    ck(cudaStreamWaitEvent(st, ws.ev_join, 0), "join wait");
    ws.side_pending = false;
}

void tap(const ForwardHooks & hk, int il, const char * node, const float * dev, int64_t n) {
    if (hk.tap) hk.tap(il, node, dev, n);
}

// out = W x for the T tokens of x (contiguous rows).
void mm(Workspace & ws, const WeightRef & w, const float * x, float * y, int T, cudaStream_t st) {
    glm_matmul(w, x, y, T, ws.mm, ws.mm_bytes, st);
}

// Clamped SwiGLU FFN on ws.y (prepared as y_in, or here): gate/up share the prepared input, SwiGLU writes the Q8
// of h for down on the GEMV path, down writes `out`.
void swiglu_ffn(const Model & m, Workspace & ws, const WeightRef & gate, const WeightRef & up, const WeightRef & down,
                int T, float * out, const MatmulInput * y_in, cudaStream_t st) {
    const int F = (int) gate.ne[1];
    const MatmulInput in = y_in ? *y_in : matmul_prepare(ws.y, m.g.n_embd, T, 1, 0, 0, ws.prep, st);
    const WeightRef ws_[2] = {gate, up};
    float * const ys[2] = {ws.gu, ws.gu + F};
    const int64_t ld[2] = {2 * (int64_t) F, 2 * (int64_t) F};
    matmul_run_multi(ws_, ys, ld, 2, in, ws.mm, ws.mm_bytes, st);
    const bool hq = matmul_input_is_q8(F, T);
    swiglu_clamp(ws.gu, ws.gu + F, ws.h, T, F, 2 * (int64_t) F, F, m.g.swiglu_clamp, st, hq ? ws.prep_h : nullptr);
    if (hq) matmul_run(down, matmul_input_q8(ws.h, F, T, ws.prep_h), out, ws.mm, ws.mm_bytes, st);
    else mm(ws, down, ws.h, out, T, st);
}

}  // namespace

Workspace::~Workspace() {
    if (ev_fork) cudaEventDestroy(ev_fork);
    if (ev_join) cudaEventDestroy(ev_join);
    if (side) cudaStreamDestroy(side);
    if (arena) cudaFree(arena);
    if (e_host && own_e_host) cudaFreeHost(e_host);
}

namespace {

// Device buffers of a workspace for max_T tokens (sizes set in ws; workspace_create assigns the pointers).
struct WsBuf { void ** p; size_t bytes; };
std::vector<WsBuf> workspace_layout(const Model & m, Workspace * ws, int max_T) {
    const Geometry & g = m.g;
    ws->max_T = max_T;
    const size_t T = (size_t) max_T;
    const size_t F = (size_t) std::max(g.ffn_dense, g.ffn_shared);

    size_t mm = 0;
    for (const auto & L : m.layers) {
        for (const WeightRef * w : {&L.ffn_gate, &L.ffn_up, &L.ffn_down, &L.gate_inp, &L.gate_shexp, &L.up_shexp,
                                    &L.down_shexp})
            if (w->data) mm = std::max(mm, matmul_scratch_bytes(*w, max_T));
    }
    ws->head_rows = std::min(max_T, kMaxHeadRows);
    mm = std::max(mm, matmul_scratch_bytes(m.output, ws->head_rows));
    ws->mm_bytes = align256(mm);
    ws->prep_bytes = align256(matmul_prepare_bytes(g.n_embd, max_T));
    const size_t prep_h_bytes = align256(matmul_prepare_bytes((int) F, max_T));

    return {
        {(void **) &ws->e, T * g.n_embd * 4},
        {(void **) &ws->x, T * g.hc * g.n_embd * 4},
        {(void **) &ws->x2, T * g.hc * g.n_embd * 4},
        {(void **) &ws->part, hc_part_floats(max_T) * 4},
        {(void **) &ws->hcp, T * g.hc_mix * 4},
        {(void **) &ws->y, T * g.n_embd * 4},
        {(void **) &ws->a, T * g.n_embd * 4},
        {(void **) &ws->b, T * g.n_embd * 4},
        {(void **) &ws->gu, T * 2 * F * 4},
        {(void **) &ws->h, T * F * 4},
        {(void **) &ws->rl, T * g.n_expert * 4},
        {(void **) &ws->ids, T * g.n_expert_used * 4},
        {(void **) &ws->w, T * g.n_expert_used * 4},
        {(void **) &ws->logits, (size_t) ws->head_rows * g.n_vocab * 4},
        {(void **) &ws->tok, (size_t) ws->head_rows * 4},
        {(void **) &ws->hn, (size_t) ws->head_rows * g.n_embd * 4},
        {&ws->mm, ws->mm_bytes},
        {&ws->prep, ws->prep_bytes},
        {&ws->prep_h, prep_h_bytes},
    };
}

size_t layout_bytes(const std::vector<WsBuf> & bufs) {
    size_t total = 0;
    for (const auto & b : bufs) total += align256(b.bytes);
    return total;
}

}  // namespace

size_t workspace_bytes(const Model & m, int max_T) {
    Workspace ws;
    return layout_bytes(workspace_layout(m, &ws, max_T));
}

std::unique_ptr<Workspace> workspace_create(const Model & m, int max_T, void * mem, float * e_host) {
    auto ws = std::make_unique<Workspace>();
    const auto bufs = workspace_layout(m, ws.get(), max_T);
    if (!mem) {
        ck(cudaMalloc(&ws->arena, layout_bytes(bufs)), "workspace");
        mem = ws->arena;
    }
    char * p = static_cast<char *>(mem);
    for (const auto & b : bufs) {
        *b.p = p;
        p += align256(b.bytes);
    }
    if (e_host) {
        ws->e_host = e_host;
        ws->own_e_host = false;
    } else {
        ck(cudaMallocHost(&ws->e_host, (size_t) max_T * m.g.n_embd * 4), "pinned embeddings");
    }
    ck(cudaStreamCreateWithFlags(&ws->side, cudaStreamNonBlocking), "side stream");
    ck(cudaEventCreateWithFlags(&ws->ev_fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&ws->ev_join, cudaEventDisableTiming), "event");
    return ws;
}

const float * ffn_branch(const Model & m, Workspace & ws, int il, int T, const ForwardHooks & hk, cudaStream_t st,
                         const MatmulInput * y_in) {
    const Geometry & g = m.g;
    const LayerWeights & L = m.layers[il];
    const int64_t TD = (int64_t) T * g.n_embd;
    MatmulInput in_local;
    if (!y_in) {
        in_local = matmul_prepare(ws.y, g.n_embd, T, 1, 0, 0, ws.prep, st);
        y_in = &in_local;
    }
    if (!L.moe) {
        swiglu_ffn(m, ws, L.ffn_gate, L.ffn_up, L.ffn_down, T, ws.a, y_in, st);
        tap(hk, il, "ffn_out", ws.a, TD);
        return nullptr;
    }
    if (!hk.experts) throw std::runtime_error("ffn_branch: no experts hook");
    join_comb(ws, st);   // the experts hook may end a graph capture segment: no open fork across it
    matmul_run(L.gate_inp, *y_in, ws.rl, ws.mm, ws.mm_bytes, st);
    tap(hk, il, "ffn_moe_logits", ws.rl, (int64_t) T * g.n_expert);
    moe_route(g, ws.rl, L.exp_probs_b, T, ws.ids, ws.w, st);
    tap(hk, il, "ffn_moe_weights_scaled", ws.w, (int64_t) T * g.n_expert_used);
    hk.experts(il, ws.y, ws.ids, ws.w, ws.a, T, st);
    if (!hk.experts_finish) tap(hk, il, "ffn_moe_out", ws.a, TD);
    swiglu_ffn(m, ws, L.gate_shexp, L.up_shexp, L.down_shexp, T, ws.b, y_in, st);
    tap(hk, il, "ffn_shexp", ws.b, TD);
    if (hk.experts_finish) {
        hk.experts_finish(il, ws.a, T, st);
        tap(hk, il, "ffn_moe_out", ws.a, TD);
    }
    return ws.b;
}

void embed_upload(const Model & m, Workspace & ws, const int32_t * ids, int T, cudaStream_t st) {
    if (T <= 0 || T > ws.max_T) throw std::runtime_error("embed_upload: T out of range");
    embed_rows_host(m, ids, T, ws.e_host);
    ck(cudaMemcpyAsync(ws.e, ws.e_host, (size_t) T * m.g.n_embd * 4, cudaMemcpyHostToDevice, st), "embedding upload");
    ck(cudaStreamSynchronize(st), "embedding upload");   // e_host is reused by the next call
}

void forward_trunk(const Model & m, Workspace & ws, const int32_t * ids, int T, int64_t pos0, int il_end,
                   const ForwardHooks & hk, cudaStream_t st) {
    embed_upload(m, ws, ids, T, st);
    forward_dev(m, ws, T, pos0, il_end, hk, st);
}

void forward_dev(const Model & m, Workspace & ws, int T, int64_t pos0, int il_end, const ForwardHooks & hk,
                 cudaStream_t st, bool all_rows) {
    const Geometry & g = m.g;
    if (T <= 0 || T > ws.max_T) throw std::runtime_error("forward_dev: T out of range");
    if (il_end < 0 || il_end > g.n_layer) throw std::runtime_error("forward_dev: il_end out of range");
    if (!hk.attn && !hk.attn_in) throw std::runtime_error("forward_dev: no attention hook");
    const int64_t TD = (int64_t) T * g.n_embd;
    if (il_end == 0) return;
    // streams ping-pong between the two buffers (hc_post reads cur, writes nxt); 2 posts per block, so the result
    // ends in ws.x
    float * cur = ws.x, * nxt = ws.x2;
    // hc_pre writes the Q8 of ws.y for the branch projections (GEMV path); ws.prep holds it until the next hc_pre
    const bool yq = matmul_input_is_q8(g.n_embd, T);
    const MatmulInput y_q8 = yq ? matmul_input_q8(ws.y, g.n_embd, T, ws.prep) : MatmulInput{};
    const MatmulInput * y_in = yq ? &y_q8 : nullptr;
    hc_init(g, ws.e, cur, T, hc_attn(m.layers[0]), ws.part, st);

    for (int il = 0; il < il_end; ++il) {
        const LayerWeights & L = m.layers[il];

        // attention branch
        hc_pre(g, hc_attn(L), cur, ws.part, T, L.attn_norm, ws.hcp, ws.y, st, yq ? ws.prep : nullptr);
        fork_comb(m, ws, T, st);
        tap(hk, il, "attn_norm", ws.y, TD);
        if (hk.attn_in) hk.attn_in(il, ws.y, y_in, ws.a, Batch{T, pos0}, st);
        else hk.attn(il, ws.y, ws.a, Batch{T, pos0}, st);
        tap(hk, il, L.mla ? "attn_out" : "kda_out", ws.a, TD);
        const HcBranch fb = hc_ffn(L);
        join_comb(ws, st);
        tap(hk, il, "hcp_attn", ws.hcp, (int64_t) T * g.hc_mix);
        hc_post(g, ws.a, nullptr, ws.hcp, cur, nxt, T, &fb, ws.part, st);
        std::swap(cur, nxt);
        tap(hk, il, "hc_attn_post", cur, TD * g.hc);

        // FFN branch
        hc_pre(g, fb, cur, ws.part, T, L.ffn_norm, ws.hcp, ws.y, st, yq ? ws.prep : nullptr);
        fork_comb(m, ws, T, st);
        tap(hk, il, "ffn_norm", ws.y, TD);
        const float * second = ffn_branch(m, ws, il, T, hk, st, y_in);
        join_comb(ws, st);
        tap(hk, il, "hcp_ffn", ws.hcp, (int64_t) T * g.hc_mix);
        const bool last = il + 1 == g.n_layer;
        const HcBranch nb = last ? HcBranch{} : hc_attn(m.layers[il + 1]);
        hc_post(g, ws.a, second, ws.hcp, cur, nxt, T, last ? nullptr : &nb, ws.part, st);
        std::swap(cur, nxt);
        tap(hk, il, "l_out", cur, TD * g.hc);
    }

    if (il_end == g.n_layer && all_rows) {
        if (T > ws.head_rows) throw std::runtime_error("forward_dev: all-rows head needs T <= head_rows");
        const bool hq = matmul_input_is_q8(g.n_embd, T);
        hc_head(g, cur, T, m.output_norm, ws.hn, st, hq ? ws.prep : nullptr);
        const MatmulInput in = hq ? matmul_input_q8(ws.hn, g.n_embd, T, ws.prep)
                                  : matmul_prepare(ws.hn, g.n_embd, T, 1, 0, 0, ws.prep, st);
        matmul_run(m.output, in, ws.logits, ws.mm, ws.mm_bytes, st);
        argmax_rows(ws.logits, g.n_vocab, T, ws.tok, st);
    } else if (il_end == g.n_layer) {
        const float * xl = cur + (int64_t) (T - 1) * g.hc * g.n_embd;
        hc_head(g, xl, 1, m.output_norm, ws.y, st, ws.prep);   // T = 1 is always on the Q8 GEMV path
        tap(hk, -1, "result_norm", ws.y, g.n_embd);
        matmul_run(m.output, matmul_input_q8(ws.y, g.n_embd, 1, ws.prep), ws.logits, ws.mm, ws.mm_bytes, st);
        tap(hk, -1, "result_output", ws.logits, g.n_vocab);
        argmax_rows(ws.logits, g.n_vocab, 1, ws.tok, st);
    }
}

}  // namespace strata::glm
