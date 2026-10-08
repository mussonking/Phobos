// src/glm/tests/ref_experts.cpp - see ref_experts.hpp.
#include "ref_experts.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"

namespace strata::glm::test {
namespace {

// need_dot: weight types need vec_dot, activation types (vec_dot_type) only from_float.
const ggml_type_traits_cpu * traits(int type, bool need_dot) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
    const ggml_type_traits_cpu * t = ggml_get_type_traits_cpu((ggml_type) type);
    if (!t || (need_dot ? !t->vec_dot : !t->from_float))
        throw std::runtime_error("ref_experts: ggml-cpu cannot handle type " + std::to_string(type));
    return t;
}

}  // namespace

void ref_experts_cpu(const Geometry & g, const HostExperts & ex, const float * x, const int32_t * ids, const float * w,
                     int T, float * out) {
    ref_experts_cpu(g, ex, x, ids, w, T, out, nullptr);
}

void ref_experts_cpu(const Geometry & g, const HostExperts & ex, const float * x, const int32_t * ids, const float * w,
                     int T, float * out, const HostExperts * per_job) {
    const int D = g.n_embd, F = g.ffn_expert, K = g.n_expert_used;
    if (ex.type_up != ex.type_gate) throw std::runtime_error("ref_experts: gate/up types differ");
    const ggml_type_traits_cpu * tg = traits(ex.type_gate, true);
    const ggml_type_traits_cpu * td = traits(ex.type_down, true);
    const ggml_type_traits_cpu * ag = traits(tg->vec_dot_type, false);
    const ggml_type_traits_cpu * ad = traits(td->vec_dot_type, false);
    const size_t gu_row = ggml_row_size((ggml_type) ex.type_gate, D);
    const size_t d_row = ggml_row_size((ggml_type) ex.type_down, F);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, D);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, F);
    const float c = g.swiglu_clamp;

    std::vector<uint8_t> act((size_t) T * a_bytes);
    for (int t = 0; t < T; ++t) ag->from_float(x + (size_t) t * D, act.data() + (size_t) t * a_bytes, D);

    // one job per (token, slot): the expert output, weighted later in slot order
    std::vector<float> res((size_t) T * K * D);
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<float> h(F);
        std::vector<uint8_t> hq(h_bytes);
        for (int job; (job = next++) < T * K;) {
            const int t = job / K;
            const int e = ids[job];
            const uint8_t * gate = per_job ? per_job[job].gate : ex.gate + (size_t) e * ex.gate_bytes;
            const uint8_t * up = per_job ? per_job[job].up : ex.up + (size_t) e * ex.up_bytes;
            const uint8_t * down = per_job ? per_job[job].down : ex.down + (size_t) e * ex.down_bytes;
            const void * a = act.data() + (size_t) t * a_bytes;
            for (int r = 0; r < F; ++r) {
                float gv = 0, uv = 0;
                tg->vec_dot(D, &gv, 0, gate + (size_t) r * gu_row, 0, a, 0, 1);
                tg->vec_dot(D, &uv, 0, up + (size_t) r * gu_row, 0, a, 0, 1);
                gv = std::min(gv, c);
                uv = std::max(std::min(uv, c), -c);
                h[r] = gv / (1.0f + std::exp(-gv)) * uv;
            }
            ad->from_float(h.data(), hq.data(), F);
            float * o = res.data() + (size_t) job * D;
            for (int r = 0; r < D; ++r) td->vec_dot(F, &o[r], 0, down + (size_t) r * d_row, 0, hq.data(), 0, 1);
        }
    };
    const int nth = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i) th.emplace_back(worker);
    for (auto & t : th) t.join();

    for (int t = 0; t < T; ++t) {
        float * o = out + (size_t) t * D;
        for (int k = 0; k < K; ++k) {
            const float wk = w[t * K + k];
            const float * r = res.data() + ((size_t) t * K + k) * D;
            for (int i = 0; i < D; ++i) o[i] = k == 0 ? r[i] * wk : o[i] + r[i] * wk;
        }
    }
}

}  // namespace strata::glm::test
