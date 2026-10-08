#include "iq4_cpu.hpp"
#include "strata/glm/experts.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace strata::glm {
namespace {

CPUExpertShape cpu_shape(const Geometry & g) {
    if (g.n_embd <= 0 || g.ffn_expert <= 0 || g.n_expert_used <= 0 || g.n_expert <= 0
        || g.swiglu_clamp != 10.0f)
        throw std::invalid_argument("CPU experts: invalid Geometry or clamp other than 10");
    iq4_cpu_row_bytes(GGML_TYPE_IQ4_XS, g.n_embd);
    iq4_cpu_row_bytes(GGML_TYPE_IQ4_NL, g.ffn_expert);
    return {g.n_embd, g.ffn_expert, g.n_expert_used, g.n_expert};
}

std::size_t float_bytes(int rows, int width) {
    if (rows <= 0 || width <= 0
        || static_cast<std::size_t>(rows) > std::numeric_limits<std::size_t>::max() / sizeof(float) / width)
        throw std::invalid_argument("CPU experts: invalid or overflowing buffer dimensions");
    return static_cast<std::size_t>(rows) * width * sizeof(float);
}

} // namespace

std::size_t cpu_expert_activation_bytes(const Geometry & g, int T) {
    const auto shape = cpu_shape(g);
    return float_bytes(T, shape.hidden);
}

std::size_t cpu_expert_scratch_bytes(const Geometry & g, int n_rows) {
    const auto shape = cpu_shape(g);
    if (n_rows < 1 || n_rows > 8)
        throw std::invalid_argument("CPU experts: scratch rows outside [1,8]");
    return float_bytes(n_rows, shape.intermediate);
}

void cpu_expert_prepare(const Geometry & g, const float * x, int T, void * act_scratch) {
    const auto bytes = cpu_expert_activation_bytes(g, T);
    if (!x || !act_scratch)
        throw std::invalid_argument("CPU experts: null activation or prepared storage");
    std::memmove(act_scratch, x, bytes);
}

void cpu_expert_eval(const Geometry & g, const ExpertRef & expert, const void * act, int T,
                     int n_rows, const int * tokens, float * out, void * private_scratch) {
    const auto shape = cpu_shape(g);
    cpu_expert_activation_bytes(g, T);
    cpu_expert_scratch_bytes(g, n_rows);
    if (expert.layout != 0)
        throw std::invalid_argument("CPU experts: unpack R8 weights before legacy F32 evaluation");
    if (expert.type_up != -1 && expert.type_up != GGML_TYPE_IQ4_XS)
        throw std::invalid_argument("CPU experts: explicit up plane type must be IQ4_XS");
    const CPUExpertRef ref{expert.gate_up, expert.down, expert.type_gate_up, expert.type_down, expert.up};
    cpu_expert_iq4_eval(shape, ref, T, static_cast<const float *>(act), n_rows, tokens,
                        out, static_cast<float *>(private_scratch));
}

void cpu_expert_iq4_eval(const CPUExpertShape & shape, const CPUExpertRef & expert,
                         int T, const float * x, int n_rows, const int * tokens,
                         float * out, float * scratch, CPUExpertPath path) {
    const auto gate_bytes = iq4_cpu_row_bytes(GGML_TYPE_IQ4_XS, shape.hidden);
    const auto down_bytes = iq4_cpu_row_bytes(GGML_TYPE_IQ4_NL, shape.intermediate);
    if (T <= 0 || n_rows < 1 || n_rows > 8 || !x || !out || !scratch
        || !expert.gate_up || !expert.down || expert.type_gate_up != GGML_TYPE_IQ4_XS
        || expert.type_down != GGML_TYPE_IQ4_NL)
        throw std::invalid_argument("CPU expert evaluator: invalid argument or non-IQ4 weights");
    if (path == CPUExpertPath::avx512 && !iq4_cpu_has_avx512())
        throw std::runtime_error("CPU expert evaluator: requested AVX512 path unavailable");
    const float * inputs[8];
    const float * mid[8];
    for (int t = 0; t < n_rows; ++t) {
        const int token = tokens ? tokens[t] : t;
        if (token < 0 || token >= T)
            throw std::invalid_argument("CPU expert evaluator: token outside prepared input");
        inputs[t] = x + static_cast<std::size_t>(token) * shape.hidden;
        mid[t] = scratch + static_cast<std::size_t>(t) * shape.intermediate;
    }
    const auto * gate = static_cast<const unsigned char *>(expert.gate_up);
    const auto * up = expert.up ? static_cast<const unsigned char *>(expert.up)
                                   : gate + gate_bytes * shape.intermediate;
    float gates[8], ups[8], down[8];
    for (int row = 0; row < shape.intermediate; ++row) {
        iq4_cpu_row(GGML_TYPE_IQ4_XS, gate + row * gate_bytes, shape.hidden, n_rows, inputs, gates, path);
        iq4_cpu_row(GGML_TYPE_IQ4_XS, up + row * gate_bytes, shape.hidden, n_rows, inputs, ups, path);
        for (int t = 0; t < n_rows; ++t)
            scratch[static_cast<std::size_t>(t) * shape.intermediate + row]
                = expert_swiglu_clamp(gates[t], ups[t]);
    }
    const auto * weights = static_cast<const unsigned char *>(expert.down);
    for (int row = 0; row < shape.hidden; ++row) {
        iq4_cpu_row(GGML_TYPE_IQ4_NL, weights + row * down_bytes, shape.intermediate, n_rows, mid, down, path);
        for (int t = 0; t < n_rows; ++t)
            out[static_cast<std::size_t>(t) * shape.hidden + row] = down[t];
    }
}

void experts_cpu_iq4(const CPUExpertShape & shape, int T, const float * x,
                      int n_jobs, const CPUExpertJob * jobs, float * out,
                      CPUExpertPath path) {
    if (shape.hidden <= 0 || shape.intermediate <= 0 || shape.top_k <= 0 || shape.experts <= 0
        || T <= 0 || n_jobs < 0 || (n_jobs && (!x || !jobs || !out)))
        throw std::invalid_argument("CPU experts: invalid shape, count or pointer");
    const auto gate_bytes = iq4_cpu_row_bytes(GGML_TYPE_IQ4_XS, shape.hidden);
    const auto down_bytes = iq4_cpu_row_bytes(GGML_TYPE_IQ4_NL, shape.intermediate);
    if (path == CPUExpertPath::avx512 && !iq4_cpu_has_avx512())
        throw std::runtime_error("CPU experts: requested AVX512 path unavailable");
    for (int j = 0; j < n_jobs; ++j) {
        const auto & job = jobs[j];
        if (job.token < 0 || job.token >= T || job.route_index < 0 || job.route_index >= shape.top_k
            || !job.expert.gate_up || !job.expert.down || !std::isfinite(job.weight)
            || job.expert.type_gate_up != GGML_TYPE_IQ4_XS || job.expert.type_down != GGML_TYPE_IQ4_NL)
            throw std::invalid_argument("CPU experts: invalid job or non-IQ4 expert");
    }
    // Group the same mapped expert without copying weights or changing caller jobs.
    std::vector<int> order(n_jobs);
    std::iota(order.begin(), order.end(), 0);
    const std::less<const void *> less;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto & ea = jobs[a].expert;
        const auto & eb = jobs[b].expert;
        if (ea.gate_up != eb.gate_up) return less(ea.gate_up, eb.gate_up);
        if (ea.up != eb.up) return less(ea.up, eb.up);
        return less(ea.down, eb.down);
    });
    std::vector<float> activation(static_cast<std::size_t>(8) * shape.intermediate);
    for (int begin = 0; begin < n_jobs;) {
        const auto & expert = jobs[order[begin]].expert;
        int end = begin + 1;
        while (end < n_jobs && end - begin < 8
               && jobs[order[end]].expert.gate_up == expert.gate_up
               && jobs[order[end]].expert.up == expert.up
               && jobs[order[end]].expert.down == expert.down) ++end;
        const int count = end - begin;
        const float * inputs[8];
        const float * mid[8];
        for (int t = 0; t < count; ++t) {
            inputs[t] = x + static_cast<std::size_t>(jobs[order[begin + t]].token) * shape.hidden;
            mid[t] = activation.data() + static_cast<std::size_t>(t) * shape.intermediate;
        }
        const auto * gate = static_cast<const unsigned char *>(expert.gate_up);
        const auto * up = expert.up ? static_cast<const unsigned char *>(expert.up)
                                   : gate + gate_bytes * shape.intermediate;
        float gates[8], ups[8], down[8];
        for (int row = 0; row < shape.intermediate; ++row) {
            iq4_cpu_row(GGML_TYPE_IQ4_XS, gate + row * gate_bytes, shape.hidden, count, inputs, gates, path);
            iq4_cpu_row(GGML_TYPE_IQ4_XS, up + row * gate_bytes, shape.hidden, count, inputs, ups, path);
            for (int t = 0; t < count; ++t)
                activation[static_cast<std::size_t>(t) * shape.intermediate + row]
                    = expert_swiglu_clamp(gates[t], ups[t]);
        }
        const auto * weights = static_cast<const unsigned char *>(expert.down);
        for (int row = 0; row < shape.hidden; ++row) {
            iq4_cpu_row(GGML_TYPE_IQ4_NL, weights + row * down_bytes, shape.intermediate, count, mid, down, path);
            for (int t = 0; t < count; ++t) {
                const auto & job = jobs[order[begin + t]];
                out[static_cast<std::size_t>(job.token) * shape.hidden + row] += job.weight * down[t];
            }
        }
        begin = end;
    }
}

} // namespace strata::glm
