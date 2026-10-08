#pragma once
// Trunk forward pass of the GLM engine: embedding -> 4 mHC streams -> 45 blocks (attention branch + FFN branch,
// each wrapped in mHC pre/post) -> stream mean -> output_norm -> head. The attention layers and the routed experts
// are hooks for now (oracle-fed in the parity tests, kda/mla/pack implementations in the engine).

#include <cstdint>
#include <functional>
#include <memory>

#include <cuda_runtime.h>

#include "model.hpp"
#include "strata/glm/matmul.hpp"

namespace strata::glm {

struct ForwardHooks {
    // Attention branch of block il: y = attn-normed input [b.T][4096] -> out [b.T][4096], positions from b.pos0.
    std::function<void(int il, const float * y, float * out, const Batch & b, cudaStream_t)> attn;
    // Optional, preferred over attn when set: the same plus y already prepared for glm_matmul (the Q8 written by
    // hc_pre; null when T is past the GEMV path), to hand to kda_forward / mla_forward_dev.
    std::function<void(int il, const float * y, const MatmulInput * y_in, float * out, const Batch & b, cudaStream_t)>
        attn_in;
    // Routed experts of MoE block il: ffn-normed y, ids/w [T][8] -> out [T][4096] = sum_k w_k expert_k(y).
    std::function<void(int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                       cudaStream_t)> experts;
    // Optional second half of experts: when set, ffn_branch enqueues the shared expert between experts and
    // experts_finish (so host-side miss work overlaps it), and capture_decode splits the graph at experts_finish
    // (experts is then captured and must be capturable).
    std::function<void(int il, float * out, int T, cudaStream_t)> experts_finish;
    // Optional: called (stream-ordered, device pointer) after each named node, oracle names without "-il".
    std::function<void(int il, const char * node, const float * dev, int64_t n)> tap;
};

// Device buffers for up to max_T tokens.
struct Workspace {
    int max_T = 0;
    float * e = nullptr;       // [T][4096] token embeddings (input of forward_dev)
    float * x = nullptr;       // [T][4][4096] mHC streams (ping-pong with x2; forward_dev leaves the result in x)
    float * x2 = nullptr;
    float * part = nullptr;    // mix partials
    float * hcp = nullptr;     // [T][24]
    float * y = nullptr;       // [T][4096] normed branch input
    float * a = nullptr;       // [T][4096] branch output (attention / routed experts)
    float * b = nullptr;       // [T][4096] shared expert output
    float * gu = nullptr;      // [T][2][F] gate | up (F = 12288 dense, 2048 shared)
    float * h = nullptr;       // [T][F] SwiGLU
    float * rl = nullptr;      // [T][288] router logits
    int32_t * ids = nullptr;   // [T][8]
    float * w = nullptr;       // [T][8]
    float * logits = nullptr;  // [head_rows][n_vocab] (last token, or every row with the all-rows head)
    int32_t * tok = nullptr;   // [head_rows] greedy argmax of logits
    float * hn = nullptr;      // [head_rows][4096] post-output_norm rows of the all-rows head (MTP h_prev)
    int head_rows = 1;         // min(max_T, kMaxHeadRows)
    void * mm = nullptr;       // matmul scratch
    size_t mm_bytes = 0;
    void * prep = nullptr;     // prepared y (Q8 from hc_pre / hc_head, or matmul_prepare)
    size_t prep_bytes = 0;
    void * prep_h = nullptr;   // prepared h (Q8 from swiglu_clamp)
    float * e_host = nullptr;  // pinned [T][4096] embeddings
    bool own_e_host = true;
    cudaStream_t side = nullptr;                          // hc_comb (Sinkhorn) in parallel with the branch
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    bool side_pending = false;
    void * arena = nullptr;
    ~Workspace();
};

constexpr int kMaxHeadRows = 8;   // the all-rows head (speculative verify) takes T <= 8
// mem: nullptr = own cudaMalloc arena; else workspace_bytes(m, max_T) bytes of device memory (256-aligned, not owned:
// it must outlive the workspace's use, e.g. a routed_lend loan for the prefill). e_host: nullptr = own pinned
// staging; else pinned host memory of >= max_T * n_embd floats (not owned; spares a large cudaMallocHost /
// cudaFreeHost per workspace for a workspace created per prompt).
std::unique_ptr<Workspace> workspace_create(const Model & m, int max_T, void * mem = nullptr, float * e_host = nullptr);
size_t workspace_bytes(const Model & m, int max_T);

// FFN branch of block il (0..45, including the MTP block) on the ffn-normed ws.y: dense SwiGLU into ws.a
// (returns null), or router + routed experts into ws.a and the shared expert into ws.b (returns ws.b; the branch
// output is ws.a + ws.b, added by hc_post).
// y_in (optional): ws.y already prepared (forward_dev passes hc_pre's Q8); otherwise prepared here.
const float * ffn_branch(const Model & m, Workspace & ws, int il, int T, const ForwardHooks & hk, cudaStream_t st,
                         const MatmulInput * y_in = nullptr);

// Host embedding rows of ids -> ws.e (synchronous: the pinned staging buffer is reused).
void embed_upload(const Model & m, Workspace & ws, const int32_t * ids, int T, cudaStream_t st);

// Device-only part of the forward on ws.e (no host sync unless a hook does one; capturable in a CUDA graph when the
// hooks are): blocks [0, il_end), then with il_end == n_layer the last token's logits in ws.logits and its greedy
// argmax in ws.tok. all_rows (T <= ws.head_rows): every row's output_norm row in ws.hn, logits in ws.logits[t], argmax
// in ws.tok[t] (speculative verify).
void forward_dev(const Model & m, Workspace & ws, int T, int64_t pos0, int il_end, const ForwardHooks & hk,
                 cudaStream_t st, bool all_rows = false);

// embed_upload + forward_dev on host token ids at positions [pos0, pos0 + T). With il_end == n_layer the logits of the last token land in
// ws.logits. Returns after enqueueing (the host embedding upload is synchronous).
void forward_trunk(const Model & m, Workspace & ws, const int32_t * ids, int T, int64_t pos0, int il_end,
                   const ForwardHooks & hk, cudaStream_t st);

}  // namespace strata::glm
