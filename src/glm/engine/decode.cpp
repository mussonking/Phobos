// src/glm/engine/decode.cpp - see decode.hpp.
#include "decode.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "strata/glm/matmul.hpp"

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm decode: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

SegmentedGraph::~SegmentedGraph() {
    for (auto g : segs) cudaGraphExecDestroy(g);
}

std::unique_ptr<SegmentedGraph> capture_decode(const Model & m, Workspace & ws, int T, const ForwardHooks & hk,
                                               cudaStream_t st, bool all_rows) {
    if (hk.tap) throw std::runtime_error("capture_decode: taps cannot be captured");
    matmul_init();
    auto sg = std::make_unique<SegmentedGraph>();
    sg->T = T;
    auto end_segment = [&]() {
        cudaGraph_t g = nullptr;
        ck(cudaStreamEndCapture(st, &g), "end capture");
        size_t n = 0;
        ck(cudaGraphGetNodes(g, nullptr, &n), "graph nodes");
        sg->nodes += (int) n;
        cudaGraphExec_t ge = nullptr;
        ck(cudaGraphInstantiateWithFlags(&ge, g, cudaGraphInstantiateFlagUseNodePriority), "instantiate");
        ck(cudaGraphDestroy(g), "graph destroy");
        sg->segs.push_back(ge);
    };
    ForwardHooks cap = hk;
    auto split = [&](int il) {
        end_segment();
        sg->expert_layers.push_back(il);
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
    };
    if (hk.experts_finish) cap.experts_finish = [&](int il, float *, int, cudaStream_t) { split(il); };
    else cap.experts = [&](int il, const float *, const int32_t *, const float *, float *, int, cudaStream_t) {
        split(il);
    };
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");
    try {
        forward_dev(m, ws, T, 0, m.g.n_layer, cap, st, all_rows);
    } catch (...) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(st, &g);
        if (g) cudaGraphDestroy(g);
        throw;
    }
    end_segment();
    return sg;
}

void replay_decode(const SegmentedGraph & sg, Workspace & ws, const ForwardHooks & hk, cudaStream_t st) {
    for (size_t i = 0; i < sg.segs.size(); ++i) {
        ck(cudaGraphLaunch(sg.segs[i], st), "graph launch");
        if (i >= sg.expert_layers.size()) continue;
        if (hk.experts_finish) hk.experts_finish(sg.expert_layers[i], ws.a, sg.T, st);
        else hk.experts(sg.expert_layers[i], ws.y, ws.ids, ws.w, ws.a, sg.T, st);
    }
}

}  // namespace strata::glm
