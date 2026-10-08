#pragma once
// Decode step as CUDA graphs, split at the routed experts: forward_dev is captured once per (T, KDA buffer parity)
// with the experts hook ending one graph segment and starting the next, so the routed experts (CPU reference or
// GPU kernels that are not capturable yet) run between segment replays. Interim design until the doorbell pattern
// (GPU hits in-graph, misses through host-mapped flags) replaces the split.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "forward.hpp"

namespace strata::glm {

struct SegmentedGraph {
    int T = 0;
    std::vector<cudaGraphExec_t> segs;   // experts.size() + 1 segments
    std::vector<int> expert_layers;      // block of the experts call after segment i
    int nodes = 0;
    ~SegmentedGraph();
};

// Captures forward_dev(m, ws, T, pos0 = unused by the captured attention, n_layer, hk) on st. hk.attn must be
// capturable (position from device memory); hk.tap must be empty. Without hk.experts_finish, hk.experts is only a
// split marker; with it, hk.experts is captured and the graph splits at experts_finish.
// all_rows: the all-rows head of forward_dev (speculative verify).
std::unique_ptr<SegmentedGraph> capture_decode(const Model & m, Workspace & ws, int T, const ForwardHooks & hk,
                                               cudaStream_t st, bool all_rows = false);

// Replays the segments, calling experts_finish(il, ws.a, T, st) if set, else experts(il, ws.y, ws.ids, ws.w, ws.a,
// T, st), between them.
void replay_decode(const SegmentedGraph & sg, Workspace & ws, const ForwardHooks & hk, cudaStream_t st);

}  // namespace strata::glm
