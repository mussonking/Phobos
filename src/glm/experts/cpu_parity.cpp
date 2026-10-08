#include "iq4_cpu.hpp"
#include "strata/glm/experts.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::glm;

namespace {

std::vector<unsigned char> load(const std::string & path, std::size_t size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(size))
        throw std::runtime_error("fixture size mismatch: " + path);
    std::vector<unsigned char> bytes(size);
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(bytes.data()), size))
        throw std::runtime_error("fixture read failed: " + path);
    return bytes;
}

std::vector<unsigned char> load_slice(const std::string & path, std::size_t offset, std::size_t size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < static_cast<std::streamoff>(offset + size))
        throw std::runtime_error("fixture slice outside file: " + path);
    std::vector<unsigned char> bytes(size);
    f.seekg(static_cast<std::streamoff>(offset));
    if (!f.read(reinterpret_cast<char *>(bytes.data()), size))
        throw std::runtime_error("fixture slice read failed: " + path);
    return bytes;
}

void put_u16(unsigned char * p, uint16_t v) { std::memcpy(p, &v, 2); }

std::vector<unsigned char> synthetic(int type, int width, int rows, std::mt19937 & rng) {
    const auto stride = iq4_cpu_row_bytes(type, width);
    std::vector<unsigned char> bytes(stride * rows);
    const bool xs = type == GGML_TYPE_IQ4_XS;
    const int block_bytes = xs ? 136 : 18;
    for (std::size_t i = 0; i < bytes.size(); i += block_bytes) {
        for (int j = 0; j < block_bytes; ++j) bytes[i + j] = static_cast<unsigned char>(rng());
        // Finite, positive/negative and zero FP16 scales, with all nibble and
        // six-bit subscale encodings sampled independently (no quantizer dependency).
        const uint16_t scales[] = {0x1000, 0x1400, 0x1800, 0x9000, 0x9400, 0};
        put_u16(bytes.data() + i, scales[rng() % 6]);
    }
    return bytes;
}

struct Metrics { double abs = 0, rel = 0, cosine = 1; bool pass = true; };

Metrics compare(const std::vector<float> & got, const std::vector<float> & expected,
                double atol = 0.002, double rtol = 0.0005) {
    Metrics m;
    double dot = 0, a2 = 0, b2 = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = expected[i], error = std::abs(a - b);
        m.abs = std::max(m.abs, error);
        m.rel = std::max(m.rel, error / std::max(std::abs(b), 1e-6));
        m.pass &= std::isfinite(a) && std::isfinite(b) && error <= atol + rtol * std::abs(b);
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
    }
    m.cosine = a2 == 0 || b2 == 0 ? (a2 == b2 ? 1 : 0) : dot / std::sqrt(a2 * b2);
    m.pass &= m.cosine >= 0.999999;
    return m;
}

void report(const std::string & name, const Metrics & m) {
    std::cout << name << " max_abs=" << m.abs << " max_rel=" << m.rel
              << " cosine=" << m.cosine << " " << (m.pass ? "PASS" : "FAIL") << '\n';
    if (!m.pass) throw std::runtime_error("parity failed: " + name);
}

// Independent ggml-dequantized F32 matmul, in caller job order, not the CPU
// implementation's grouped row dispatch or shared activation helper.
std::vector<float> reference(const CPUExpertShape & s, int T, const std::vector<float> & x,
                             const std::vector<CPUExpertJob> & jobs, float initial) {
    std::vector<float> out(static_cast<std::size_t>(T) * s.hidden, initial);
    std::vector<float> row(std::max(s.hidden, s.intermediate)), gate(s.intermediate), mid(s.intermediate);
    const auto gu_stride = iq4_cpu_row_bytes(GGML_TYPE_IQ4_XS, s.hidden);
    const auto down_stride = iq4_cpu_row_bytes(GGML_TYPE_IQ4_NL, s.intermediate);
    const auto * gu_traits = ggml_get_type_traits(GGML_TYPE_IQ4_XS);
    const auto * down_traits = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    for (const auto & job : jobs) {
        const float * input = x.data() + static_cast<std::size_t>(job.token) * s.hidden;
        const auto * gu = static_cast<const unsigned char *>(job.expert.gate_up);
        for (int r = 0; r < 2 * s.intermediate; ++r) {
            const auto * weights = r >= s.intermediate && job.expert.up
                ? static_cast<const unsigned char *>(job.expert.up) + (r - s.intermediate) * gu_stride
                : gu + r * gu_stride;
            gu_traits->to_float(weights, row.data(), s.hidden);
            float sum = 0;
            for (int i = 0; i < s.hidden; ++i) sum += row[i] * input[i];
            if (r < s.intermediate) gate[r] = sum;
            else {
                const int j = r - s.intermediate;
                const float g = std::min(gate[j], 10.0f), u = std::max(-10.0f, std::min(sum, 10.0f));
                mid[j] = (g / (1.0f + std::exp(-g))) * u;
            }
        }
        const auto * down = static_cast<const unsigned char *>(job.expert.down);
        for (int r = 0; r < s.hidden; ++r) {
            down_traits->to_float(down + r * down_stride, row.data(), s.intermediate);
            float sum = 0;
            for (int i = 0; i < s.intermediate; ++i) sum += row[i] * mid[i];
            out[static_cast<std::size_t>(job.token) * s.hidden + r] += job.weight * sum;
        }
    }
    return out;
}

void check_rows(std::mt19937 & rng, CPUExpertPath path) {
    for (int type : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL}) {
        const int width = type == GGML_TYPE_IQ4_XS ? 4096 : 2048;
        const auto bytes = synthetic(type, width, 1, rng);
        // Deliberately unaligned row and activations.
        std::vector<unsigned char> unaligned(bytes.size() + 1);
        std::memcpy(unaligned.data() + 1, bytes.data(), bytes.size());
        std::vector<float> x(8 * width + 1);
        for (auto & v : x) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000;
        const float * inputs[8];
        for (int t = 0; t < 8; ++t) inputs[t] = x.data() + 1 + t * width;
        for (int T = 1; T <= 8; ++T) {
            std::vector<float> got(T), expected(T);
            iq4_cpu_row(type, unaligned.data() + 1, width, T, inputs, expected.data(), CPUExpertPath::reference);
            iq4_cpu_row(type, unaligned.data() + 1, width, T, inputs, got.data(), path);
            report("row type=" + std::to_string(type) + " T=" + std::to_string(T), compare(got, expected));
        }
        // the double row (glm_eval --kernels f64): the same F32 weights, a scalar long double sum as the reference
        std::vector<float> w(width);
        ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(bytes.data(), w.data(), width);
        std::vector<double> xd(x.begin() + 1, x.end());
        const double * in64[8];
        for (int t = 0; t < 8; ++t) in64[t] = xd.data() + t * width;
        double got[8];
        iq4_cpu_row_f64(type, unaligned.data() + 1, width, 8, in64, got);
        double worst = 0;
        for (int t = 0; t < 8; ++t) {
            long double sum = 0, mag = 0;
            for (int i = 0; i < width; ++i) {
                sum += (long double) w[i] * in64[t][i];
                mag += std::fabs((long double) w[i] * in64[t][i]);
            }
            worst = std::max(worst, (double) (std::fabs(got[t] - sum) / mag));
        }
        std::cout << "row f64 type=" << type << " T=8 max |err| / sum|w x| = " << worst << "\n";
        if (!(worst < 1e-14)) throw std::runtime_error("iq4_cpu_row_f64: not double accurate");
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        CPUExpertShape shape;
        int fixture_T = 8;
        bool require_avx512 = false, quick = false;
        int fixture_expert = 0;
        std::string gu_path, down_path, x_path, expected_path, fixture_dir;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--require-avx512") require_avx512 = true;
            else if (arg == "--quick") quick = true;
            else if (arg == "--fixture-dir" && i + 1 < argc) fixture_dir = argv[++i];
            else if (arg == "--expert" && i + 1 < argc) fixture_expert = std::stoi(argv[++i]);
            else if (arg == "--fixture" && i + 3 < argc) {
                gu_path = argv[++i]; down_path = argv[++i]; x_path = argv[++i];
            } else if (arg == "--expected" && i + 1 < argc) expected_path = argv[++i];
            else if (arg == "--tokens" && i + 1 < argc) fixture_T = std::stoi(argv[++i]);
            else throw std::runtime_error("usage: glm_cpu_parity [--require-avx512] [--quick] "
                                          "[--fixture gate_up.iq4_xs down.iq4_nl x.f32] [--tokens 1..8] [--expected out.f32] "
                                          "[--fixture-dir DIR --tokens 4 --expert 0..7]");
        }
        if (!fixture_dir.empty()) {
            if (!gu_path.empty() || !expected_path.empty())
                throw std::runtime_error("--fixture-dir cannot be combined with --fixture or --expected");
            if (fixture_T != 4 || fixture_expert < 0 || fixture_expert >= 8)
                throw std::runtime_error("packed blk3 fixture requires --tokens 4 and --expert 0..7");
            gu_path = fixture_dir + "/experts.bin";
            down_path = gu_path;
            x_path = fixture_dir + "/x.f32";
            expected_path = fixture_dir + "/output.f32";
        }
        if (fixture_T < 1 || fixture_T > 8 || (!expected_path.empty() && gu_path.empty()))
            throw std::runtime_error("invalid fixture tokens or --expected without --fixture");
        if (require_avx512 && !iq4_cpu_has_avx512())
            throw std::runtime_error("AVX512 required but unavailable; this is not an AVX512 parity result");
        const auto path = iq4_cpu_has_avx512() ? CPUExpertPath::avx512 : CPUExpertPath::automatic;
        std::cout << std::setprecision(10) << "path=" << (iq4_cpu_has_avx512() ? "avx512-f32" : "ggml-f32-fallback")
                  << " tolerance=0.002+0.0005*abs(reference), cosine>=0.999999\n";
        std::mt19937 rng(0x47514c4d);
        check_rows(rng, path);
        for (float g : {-100.0f, -20.0f, -11.0f, 0.0f, 11.0f, 20.0f}) {
            const float gc = std::min(g, 10.0f);
            const float expected = gc / (1.0f + std::exp(-gc)) * 10;
            if (expert_swiglu_clamp(g, 20) != expected || expert_swiglu_clamp(g, -20) != -expected)
                throw std::runtime_error("asymmetric clamp failed");
        }
        if (quick && gu_path.empty()) { shape.hidden = 256; shape.intermediate = 32; }
        const auto gu_size = iq4_cpu_row_bytes(GGML_TYPE_IQ4_XS, shape.hidden) * 2 * shape.intermediate;
        const auto down_size = iq4_cpu_row_bytes(GGML_TYPE_IQ4_NL, shape.intermediate) * shape.hidden;
        const auto blob_offset = static_cast<std::size_t>(fixture_expert) * (gu_size + down_size);
        auto gu = gu_path.empty() ? synthetic(GGML_TYPE_IQ4_XS, shape.hidden, 2 * shape.intermediate, rng)
                  : fixture_dir.empty() ? load(gu_path, gu_size) : load_slice(gu_path, blob_offset, gu_size);
        auto down = down_path.empty() ? synthetic(GGML_TYPE_IQ4_NL, shape.intermediate, shape.hidden, rng)
                    : fixture_dir.empty() ? load(down_path, down_size) : load_slice(down_path, blob_offset + gu_size, down_size);
        const CPUExpertRef expert{gu.data(), down.data(), GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
        // The adjacent up plane is deliberately zero. A separate up pointer
        // must override it, and grouping must distinguish both references.
        std::vector<unsigned char> split_gate(gu.size(), 0);
        std::copy_n(gu.data(), gu.size() / 2, split_gate.data());
        const CPUExpertRef split_expert{split_gate.data(), down.data(), GGML_TYPE_IQ4_XS,
                                       GGML_TYPE_IQ4_NL, gu.data() + gu.size() / 2};
        const CPUExpertRef zero_up_expert{split_gate.data(), down.data(), GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
        std::vector<float> x(static_cast<std::size_t>(fixture_T) * shape.hidden);
        if (x_path.empty()) {
            for (auto & v : x) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 1000;
        } else {
            const auto bytes = load(x_path, x.size() * sizeof(float));
            std::memcpy(x.data(), bytes.data(), bytes.size());
        }
        const int first_T = gu_path.empty() ? 1 : fixture_T;
        for (int T = first_T; T <= fixture_T; ++T) {
            std::vector<CPUExpertJob> jobs;
            const float initial = gu_path.empty() ? 0.125f : 0;
            for (int t = T - 1; t >= 0; --t)
                jobs.push_back({t, 0, expert, gu_path.empty() ? 0.35f : 1.0f});
            if (gu_path.empty()) {
                for (int t = 0; t < T; ++t) jobs.push_back({t, 1, expert, -0.2f});
            }
            const auto expected = reference(shape, T, x, jobs, initial);
            std::vector<float> got(static_cast<std::size_t>(T) * shape.hidden, initial);
            experts_cpu_iq4(shape, T, x.data(), static_cast<int>(jobs.size()), jobs.data(), got.data(), path);
            report("expert T=" + std::to_string(T), compare(got, expected));
            std::vector<CPUExpertJob> unweighted_jobs;
            std::vector<int> selected(T);
            std::vector<float> reordered(static_cast<std::size_t>(T) * shape.hidden);
            for (int t = 0; t < T; ++t) {
                selected[t] = T - 1 - t;
                unweighted_jobs.push_back({t, 0, expert, 1.0f});
                std::copy_n(x.data() + static_cast<std::size_t>(selected[t]) * shape.hidden,
                            shape.hidden, reordered.data() + static_cast<std::size_t>(t) * shape.hidden);
            }
            const auto eval_expected = reference(shape, T, reordered, unweighted_jobs, 0);
            std::vector<float> eval_out(static_cast<std::size_t>(T) * shape.hidden, -1);
            std::vector<float> scratch(static_cast<std::size_t>(T) * shape.intermediate);
            cpu_expert_iq4_eval(shape, expert, T, x.data(), T, selected.data(), eval_out.data(), scratch.data(), path);
            report("unweighted selected rows T=" + std::to_string(T), compare(eval_out, eval_expected));
            Geometry geometry;
            geometry.n_embd = shape.hidden;
            geometry.ffn_expert = shape.intermediate;
            const ExpertRef public_ref{expert.gate_up, expert.down, expert.type_gate_up, expert.type_down};
            std::vector<float> prepared(cpu_expert_activation_bytes(geometry, T) / sizeof(float));
            cpu_expert_prepare(geometry, x.data(), T, prepared.data());
            if (!std::equal(prepared.begin(), prepared.end(), x.begin())
                || cpu_expert_scratch_bytes(geometry, T) != scratch.size() * sizeof(float))
                throw std::runtime_error("public prepare/scratch contract mismatch");
            const auto immutable_prepared = prepared;
            cpu_expert_eval(geometry, public_ref, prepared.data(), T, T, selected.data(), eval_out.data(), scratch.data());
            report("public prepared selected rows T=" + std::to_string(T), compare(eval_out, eval_expected));
            for (int t = 0; t < T; ++t) {
                cpu_expert_eval(geometry, public_ref, prepared.data(), T, 1, &selected[t],
                                eval_out.data() + static_cast<std::size_t>(t) * shape.hidden, scratch.data());
            }
            report("public prepared single jobs T=" + std::to_string(T), compare(eval_out, eval_expected));
            if (prepared != immutable_prepared)
                throw std::runtime_error("evaluator modified shared prepared activations");
            const ExpertRef split_public{split_expert.gate_up, split_expert.down,
                                         split_expert.type_gate_up, split_expert.type_down, split_expert.up};
            cpu_expert_eval(geometry, split_public, prepared.data(), T, T, selected.data(), eval_out.data(), scratch.data());
            report("public separate planes T=" + std::to_string(T), compare(eval_out, eval_expected));
            auto explicit_up = split_public;
            explicit_up.type_up = GGML_TYPE_IQ4_XS;
            cpu_expert_eval(geometry, explicit_up, prepared.data(), T, T, selected.data(), eval_out.data(), scratch.data());
            report("public explicit up type T=" + std::to_string(T), compare(eval_out, eval_expected));
            const auto before_invalid_up = eval_out;
            explicit_up.type_up = GGML_TYPE_IQ4_NL;
            bool up_rejected = false;
            try { cpu_expert_eval(geometry, explicit_up, prepared.data(), T, T, selected.data(), eval_out.data(), scratch.data()); }
            catch (const std::invalid_argument &) { up_rejected = true; }
            if (!up_rejected || eval_out != before_invalid_up)
                throw std::runtime_error("mixed up plane type was not rejected before writing");
            std::vector<CPUExpertJob> mixed_jobs;
            for (int t = 0; t < T; ++t) {
                mixed_jobs.push_back({t, 0, split_expert, 0.35f});
                mixed_jobs.push_back({t, 1, zero_up_expert, -0.2f});
            }
            const auto mixed_expected = reference(shape, T, x, mixed_jobs, 0.125f);
            std::vector<float> mixed_out(static_cast<std::size_t>(T) * shape.hidden, 0.125f);
            experts_cpu_iq4(shape, T, x.data(), static_cast<int>(mixed_jobs.size()), mixed_jobs.data(), mixed_out.data(), path);
            report("separate up grouping T=" + std::to_string(T), compare(mixed_out, mixed_expected));
            if (!expected_path.empty()) {
                const auto bytes = fixture_dir.empty() ? load(expected_path, got.size() * sizeof(float))
                    : load_slice(expected_path, static_cast<std::size_t>(fixture_expert) * got.size() * sizeof(float),
                                 got.size() * sizeof(float));
                std::vector<float> captured(got.size());
                std::memcpy(captured.data(), bytes.data(), bytes.size());
                report("fixture expected", compare(got, captured));
            }
            const auto before = got;
            experts_cpu_iq4(shape, T, nullptr, 0, nullptr, got.data(), path);
            if (got != before) throw std::runtime_error("empty jobs modified output");
            jobs.back().token = T;
            bool rejected = false;
            try { experts_cpu_iq4(shape, T, x.data(), static_cast<int>(jobs.size()), jobs.data(), got.data(), path); }
            catch (const std::invalid_argument &) { rejected = true; }
            if (!rejected || got != before) throw std::runtime_error("invalid jobs were not rejected before writing");
        }
        std::cout << "CPU IQ4 parity PASS (no timing claim)\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
