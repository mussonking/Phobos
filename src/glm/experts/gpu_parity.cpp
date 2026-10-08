// Same-weight ggml dequantization + F32 oracle. Run only under the coordinator's GPU lease.
#include "strata/glm/experts.hpp"
#include "ggml.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using strata::glm::ExpertRef;
using strata::glm::Geometry;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n)); }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
    void upload(const void* src, size_t n) { ck(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
};
std::vector<uint8_t> load(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(n)) throw std::runtime_error("wrong fixture size: " + path);
    f.seekg(0);
    std::vector<uint8_t> data(n);
    if (!f.read(reinterpret_cast<char*>(data.data()), n)) throw std::runtime_error("cannot read " + path);
    return data;
}
std::vector<uint8_t> load_slice(const std::string& path, size_t offset, size_t n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < std::streamoff(offset + n)) throw std::runtime_error("short fixture: " + path);
    f.seekg(offset);
    std::vector<uint8_t> data(n);
    if (!f.read(reinterpret_cast<char*>(data.data()), n)) throw std::runtime_error("cannot read " + path);
    return data;
}
void half_store(uint8_t* p, float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    std::memcpy(p, &h, sizeof(h));
}
struct Weights {
    std::vector<uint8_t> gu, down;
    std::vector<float> gu32, down32;
    Device dgu, dd, dup;
    Weights(const Geometry& g, int seed, const std::string& fixture, int fixture_expert = 0)
        : gu(size_t(2) * g.ffn_expert * (g.n_embd / 256) * 136),
          down(size_t(g.n_embd) * (g.ffn_expert / 32) * 18),
          gu32(size_t(2) * g.ffn_expert * g.n_embd),
          down32(size_t(g.n_embd) * g.ffn_expert), dgu(gu.size()), dd(down.size()), dup(gu.size() / 2) {
        if (!fixture.empty()) {
            if (std::ifstream(fixture + "/experts.bin", std::ios::binary)) {
                const size_t base = size_t(fixture_expert) * (gu.size() + down.size());
                gu = load_slice(fixture + "/experts.bin", base, gu.size());
                down = load_slice(fixture + "/experts.bin", base + gu.size(), down.size());
            } else {
                gu = load(fixture + "/gate_up.iq4xs", gu.size());
                down = load(fixture + "/down.iq4nl", down.size());
            }
        } else {
            std::mt19937 rng(seed);
            for (size_t i = 0; i < gu.size(); i += 136) {
                half_store(gu.data() + i, (seed % 2 ? -1 : 1) * 0.0002f);
                for (int b = 2; b < 136; ++b) gu[i + b] = uint8_t(rng());
            }
            for (size_t i = 0; i < down.size(); i += 18) {
                half_store(down.data() + i, (seed % 2 ? -1 : 1) * 0.002f);
                for (int b = 2; b < 18; ++b) down[i + b] = uint8_t(rng());
            }
        }
        // These are ggml's dequantizers, not a duplicate of the CUDA format decoder.
        ggml_get_type_traits(GGML_TYPE_IQ4_XS)->to_float(gu.data(), gu32.data(), gu32.size());
        ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(down.data(), down32.data(), down32.size());
        dgu.upload(gu.data(), gu.size()); dd.upload(down.data(), down.size());
        dup.upload(gu.data() + gu.size() / 2, gu.size() / 2);
    }
};
std::vector<float> oracle(const Geometry& g, int T, const std::vector<float>& x,
                          const std::vector<int>& ids, const std::vector<float>& w,
                          const std::vector<ExpertRef>& refs, const std::vector<Weights*>& weights,
                          const std::vector<float>& initial) {
    std::vector<float> out = initial, h(g.ffn_expert), y(g.n_embd);
    for (int t = 0; t < T; ++t) {
        // Evaluate each unique expert only once per token, even with deliberately duplicate selections.
        for (size_t e = 0; e < weights.size(); ++e) {
            bool needed = false;
            for (int s = 0; s < g.n_expert_used; ++s) {
                const int j = t * g.n_expert_used + s;
                needed |= ids[j] == int(e) && refs[j].gate_up && refs[j].down &&
                          refs[j].type_gate_up == 23 && refs[j].type_down == 20 &&
                          (refs[j].type_up < 0 || refs[j].type_up == 23);
            }
            if (!needed) continue;
            const auto& a = *weights[e];
            for (int r = 0; r < g.ffn_expert; ++r) {
                float gate = 0, up = 0;
                for (int k = 0; k < g.n_embd; ++k) {
                    gate += a.gu32[size_t(r) * g.n_embd + k] * x[size_t(t) * g.n_embd + k];
                    up += a.gu32[size_t(g.ffn_expert + r) * g.n_embd + k] * x[size_t(t) * g.n_embd + k];
                }
                gate = std::min(gate, g.swiglu_clamp);
                up = std::max(-g.swiglu_clamp, std::min(up, g.swiglu_clamp));
                h[r] = (gate / (1 + std::exp(-gate))) * up;
            }
            for (int r = 0; r < g.n_embd; ++r) {
                float v = 0;
                for (int k = 0; k < g.ffn_expert; ++k) v += a.down32[size_t(r) * g.ffn_expert + k] * h[k];
                y[r] = v;
            }
            for (int s = 0; s < g.n_expert_used; ++s) {
                const int j = t * g.n_expert_used + s;
                if (ids[j] != int(e) || !refs[j].gate_up || !refs[j].down ||
                    refs[j].type_gate_up != 23 || refs[j].type_down != 20 ||
                    (refs[j].type_up >= 0 && refs[j].type_up != 23)) continue;
                for (int r = 0; r < g.n_embd; ++r) out[size_t(t) * g.n_embd + r] += y[r] * w[j];
            }
        }
    }
    return out;
}
bool compare(const std::vector<float>& actual, const std::vector<float>& expected,
             int T, int hidden, const char* mode) {
    constexpr double atol = 0.02, rtol = 0.001, min_cos = 0.99999;
    double abs_err = 0, rel_err = 0, dot = 0, aa = 0, bb = 0;
    size_t bad = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i], d = std::abs(a - b);
        abs_err = std::max(abs_err, d); rel_err = std::max(rel_err, d / std::max(1e-6, std::abs(b)));
        dot += a * b; aa += a * a; bb += b * b;
        if (!std::isfinite(a) || !std::isfinite(b) || d > atol + rtol * std::abs(b)) ++bad;
    }
    const double cosine = aa && bb ? dot / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    const bool ok = bad == 0 && cosine >= min_cos;
    std::printf("%s H=%d T=%d max_abs=%.9g max_rel=%.9g cosine=%.12g bad=%zu %s (atol=%.3g rtol=%.3g cos>=%.6g)\n",
                mode, hidden, T, abs_err, rel_err, cosine, bad, ok ? "PASS" : "FAIL", atol, rtol, min_cos);
    return ok;
}
bool run(const Geometry& g, int T, const std::vector<Weights*>& weights, bool all_miss = false,
         const std::string& fixture = "", bool all_hits = false) {
    const size_t jobs = size_t(T) * g.n_expert_used, n = size_t(T) * g.n_embd;
    std::mt19937 rng(71 + T);
    std::uniform_real_distribution<float> dist(-2, 2);
    std::vector<float> x(n), initial(n), w(jobs), actual(n);
    std::vector<int> ids(jobs);
    std::vector<ExpertRef> refs(jobs);
    for (size_t i = 0; i < n; ++i) { x[i] = dist(rng); initial[i] = 0.01f * dist(rng); }
    if (!fixture.empty() && T == 4 && std::ifstream(fixture + "/x.f32", std::ios::binary)) {
        const auto data = load(fixture + "/x.f32", n * sizeof(float));
        std::memcpy(x.data(), data.data(), data.size());
    }
    for (size_t j = 0; j < jobs; ++j) {
        ids[j] = int(j % weights.size()); w[j] = 2.5f / g.n_expert_used;
        const auto* a = weights[ids[j]];
        refs[j] = {a->dgu.p, a->dd.p, 23, 20};
        // Exercise a genuinely separate allocation, not just an alias into gate | up.
        if (!all_hits && j % 2) refs[j].up = a->dup.p;
        if (all_miss) refs[j].gate_up = nullptr;
        if (!all_hits) {
            if (j % 7 == 0) refs[j].gate_up = nullptr;
            if (j % 11 == 3) refs[j].down = nullptr;
            if (j % 13 == 4) refs[j].type_gate_up = 0;
            if (j % 17 == 5) ids[j] = -1;
            if (j % 19 == 6) ids[j] = g.n_expert;
        }
    }
    size_t hit_jobs = 0;
    std::vector<bool> used(weights.size(), false);
    for (size_t j = 0; j < jobs; ++j) {
        if (ids[j] >= 0 && size_t(ids[j]) < weights.size() && refs[j].gate_up && refs[j].down &&
            refs[j].type_gate_up == 23 && refs[j].type_down == 20 &&
                          (refs[j].type_up < 0 || refs[j].type_up == 23)) {
            ++hit_jobs; used[ids[j]] = true;
        }
    }
    const size_t active_experts = std::count(used.begin(), used.end(), true);
    const auto expected = oracle(g, T, x, ids, w, refs, weights, initial);
    Device dx(n * 4), di(jobs * sizeof(int)), dw(jobs * 4), ds(jobs * sizeof(ExpertRef)), dout(n * 4),
           scratch(strata::glm::experts_gpu_scratch_bytes(g, T));
    dx.upload(x.data(), n * 4); di.upload(ids.data(), jobs * sizeof(int));
    dw.upload(w.data(), jobs * 4); ds.upload(refs.data(), jobs * sizeof(ExpertRef)); dout.upload(initial.data(), n * 4);
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    auto launch = [&] { strata::glm::experts_gpu(g, T, dx.as<float>(), di.as<int>(), dw.as<float>(),
                                               ds.as<ExpertRef>(), dout.as<float>(), scratch.p, stream); };
    launch(); ck(cudaGetLastError()); ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(actual.data(), dout.p, n * 4, cudaMemcpyDeviceToHost));
    bool ok = compare(actual, expected, T, g.n_embd, all_miss ? "all-miss" : "eager");
    dout.upload(initial.data(), n * 4);
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch(); ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(actual.data(), dout.p, n * 4, cudaMemcpyDeviceToHost));
    ok &= compare(actual, expected, T, g.n_embd, "graph");
    cudaEvent_t begin, end;
    ck(cudaEventCreate(&begin)); ck(cudaEventCreate(&end));
    ck(cudaEventRecord(begin, stream));
    for (int repeat = 0; repeat < 5; ++repeat) ck(cudaGraphLaunch(exec, stream));
    ck(cudaEventRecord(end, stream)); ck(cudaEventSynchronize(end));
    float elapsed = 0; ck(cudaEventElapsedTime(&elapsed, begin, end));
    std::printf("graph timing H=%d T=%d routing=%s hit_jobs=%zu/%zu unique_experts=%zu ms/layer=%.6f scratch_bytes=%zu (requires idle leased GPU)\n",
                g.n_embd, T, all_miss ? "all-miss" : (all_hits ? "all-hits" : "mixed"), hit_jobs, jobs,
                active_experts, elapsed / 5, strata::glm::experts_gpu_scratch_bytes(g, T));
    ck(cudaEventDestroy(begin)); ck(cudaEventDestroy(end));
    // Capture must consume live routing buffers, not a host snapshot. Change all slots to misses.
    for (auto& ref : refs) ref.gate_up = nullptr;
    ds.upload(refs.data(), jobs * sizeof(ExpertRef)); dout.upload(initial.data(), n * 4);
    ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    ck(cudaMemcpy(actual.data(), dout.p, n * 4, cudaMemcpyDeviceToHost));
    ok &= compare(actual, initial, T, g.n_embd, "graph-mutated-misses");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    return ok;
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string fixture;
        bool large_prefill = false, all_hits = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--fixture-dir" && i + 1 < argc) fixture = argv[++i];
            else if (std::string(argv[i]) == "--large-prefill") large_prefill = true;
            else if (std::string(argv[i]) == "--all-hits") all_hits = true;
            else throw std::runtime_error("usage: glm_experts_gpu_parity [--fixture-dir DIR] [--large-prefill] [--all-hits]");
        }
        ggml_init_params params = {1024 * 1024, nullptr, true};
        ggml_context* ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml initialization failed");
        bool ok = true;
        {
            Geometry small; small.n_embd = 256; small.ffn_expert = 64; small.n_expert = 3;
            Weights a(small, 1, ""), b(small, 2, "");
            const std::vector<Weights*> ws = {&a, &b};
            for (int T : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 65}) ok &= run(small, T, ws);
            ok &= run(small, 17, ws, true);
            if (large_prefill) ok &= run(small, 4096, ws);
        }
        {
            Geometry full;
            const bool multi_fixture = !fixture.empty() &&
                                       bool(std::ifstream(fixture + "/experts.bin", std::ios::binary));
            std::vector<std::unique_ptr<Weights>> storage;
            std::vector<Weights*> ws;
            for (int e = 0; e < (multi_fixture || all_hits ? 8 : 2); ++e) {
                storage.emplace_back(std::make_unique<Weights>(full, 3 + e, fixture, e));
                ws.push_back(storage.back().get());
            }
            for (int T : {1, 4, 8, 9}) ok &= run(full, T, ws, false, fixture);
            if (all_hits) {
                if (!fixture.empty() && !multi_fixture)
                    throw std::runtime_error("--all-hits requires eight distinct experts.bin fixture slices or synthetic weights");
                for (int T : {1, 4, 8, 9}) ok &= run(full, T, ws, false, fixture, true);
            }
        }
        ggml_free(ctx);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "glm GPU expert parity: %s\n", e.what());
        return 2;
    }
}
