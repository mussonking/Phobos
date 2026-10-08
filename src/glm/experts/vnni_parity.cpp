#include "iq4_vnni.hpp"
#include "ggml.h"
#ifdef STRATA_VNNI_GGML_DOT
#include "ggml-cpu.h"
#endif
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

extern "C" void quantize_row_q8_K_ref(const float *, block_q8_K *, int64_t);
extern "C" void quantize_row_q8_0_ref(const float *, block_q8_0 *, int64_t);

using namespace strata::glm;
namespace {
struct Metrics { double abs = 0, rel = 0, cosine = 1; bool pass = true; };
Metrics compare(const std::vector<float> & a, const std::vector<float> & b, double atol, double rtol) {
    Metrics m;
    double dot = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double err = std::abs(double(a[i]) - b[i]);
        m.abs = std::max(m.abs, err);
        m.rel = std::max(m.rel, err / std::max(1e-6, std::abs(double(b[i]))));
        m.pass &= std::isfinite(a[i]) && std::isfinite(b[i]) && err <= atol + rtol * std::abs(b[i]);
        dot += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i];
    }
    m.cosine = aa && bb ? dot / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    return m;
}
void report(const std::string & name, const Metrics & m, bool enforce = true) {
    std::cout << name << " max_abs=" << m.abs << " max_rel=" << m.rel << " cosine=" << m.cosine
              << (enforce ? (m.pass ? " PASS" : " FAIL") : " APPROXIMATION_ONLY") << '\n';
    if (enforce && !m.pass) throw std::runtime_error(name + " failed");
}
std::vector<unsigned char> load(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) throw std::runtime_error("cannot load " + path);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) throw std::runtime_error("read failed " + path);
    return bytes;
}
std::vector<unsigned char> synthetic(int type, int width, int rows, std::mt19937 & rng) {
    const auto stride = type == GGML_TYPE_IQ4_NL ? std::size_t(width / 32) * sizeof(block_iq4_nl)
                                                : iq4_vnni_row_bytes(type, width);
    std::vector<unsigned char> bytes(stride * rows);
    const int bs = type == GGML_TYPE_IQ4_XS ? sizeof(block_iq4_xs) : sizeof(block_iq4_nl);
    const uint16_t scales[] = {0x1000, 0x1400, 0x1800, 0x9000, 0x9400, 0};
    for (std::size_t i = 0; i < bytes.size(); i += bs) {
        for (int j = 0; j < bs; ++j) bytes[i + j] = static_cast<unsigned char>(rng());
        const auto scale = scales[rng() % 6];
        std::memcpy(bytes.data() + i, &scale, sizeof(scale));
    }
    return bytes;
}
std::vector<float> dequant_q8(const void * data, int width) {
    std::vector<float> result(width);
    const auto * q = static_cast<const block_q8_K *>(data);
    for (int i = 0; i < width; ++i) result[i] = q[i / 256].d * q[i / 256].qs[i % 256];
    return result;
}
float f32_dot(int type, const void * row, int width, const float * x, std::vector<float> & unpacked) {
    ggml_get_type_traits(static_cast<ggml_type>(type))->to_float(row, unpacked.data(), width);
    // Double reduction isolates the integer-dot arithmetic from sequential
    // F32 accumulation error, using GGML's actual F32 dequantized operands.
    double sum = 0;
    for (int i = 0; i < width; ++i) sum += double(unpacked[i]) * x[i];
    return static_cast<float>(sum);
}
std::vector<float> model_reference(const Geometry & g, const ExpertRef & e, const std::vector<float> & x,
                                   int T, bool quantized, bool same_mid) {
    std::vector<float> out(std::size_t(T) * g.n_embd), row(std::max(g.n_embd, g.ffn_expert));
    std::vector<float> mid(g.ffn_expert), input(g.n_embd);
    std::vector<block_q8_K> qx(g.n_embd / 256), qm(g.ffn_expert / 256);
    const auto gs = iq4_vnni_row_bytes(e.type_gate_up, g.n_embd);
    const auto ds = iq4_vnni_row_bytes(e.type_down, g.ffn_expert);
    const auto * gate = static_cast<const unsigned char *>(e.gate_up);
    const auto * up = e.up ? static_cast<const unsigned char *>(e.up) : gate + gs * g.ffn_expert;
    for (int t = 0; t < T; ++t) {
        std::copy_n(x.data() + std::size_t(t) * g.n_embd, g.n_embd, input.data());
        if (quantized) {
            iq4_vnni_quantize(input.data(), g.n_embd, qx.data());
            input = dequant_q8(qx.data(), g.n_embd);
        }
        for (int r = 0; r < g.ffn_expert; ++r) {
            const float a = same_mid ? iq4_vnni_row(e.type_gate_up, gate + r * gs, g.n_embd, qx.data(), IQ4VNNIPath::scalar)
                                    : f32_dot(e.type_gate_up, gate + r * gs, g.n_embd, input.data(), row);
            const float b = same_mid ? iq4_vnni_row(e.type_gate_up, up + r * gs, g.n_embd, qx.data(), IQ4VNNIPath::scalar)
                                    : f32_dot(e.type_gate_up, up + r * gs, g.n_embd, input.data(), row);
            const float gc = std::min(a, g.swiglu_clamp);
            mid[r] = gc / (1 + std::exp(-gc)) * std::max(-g.swiglu_clamp, std::min(b, g.swiglu_clamp));
        }
        if (quantized) {
            iq4_vnni_quantize(mid.data(), g.ffn_expert, qm.data());
            mid = dequant_q8(qm.data(), g.ffn_expert);
        }
        for (int r = 0; r < g.n_embd; ++r)
            out[std::size_t(t) * g.n_embd + r] = f32_dot(e.type_down,
                static_cast<const unsigned char *>(e.down) + r * ds, g.ffn_expert, mid.data(), row);
    }
    return out;
}
#ifdef STRATA_VNNI_GGML_DOT
float ggml_dot(int type, const void * weights, int width, const void * q8) {
    // GGML IQ4_NL uses Q8_0, not Q8_K. Keep the same activation model:
    // this explicitly labelled hybrid baseline uses scalar Q8_K for down.
    if (type == GGML_TYPE_IQ4_NL)
        return iq4_vnni_row(type, weights, width, q8, IQ4VNNIPath::scalar);
    const auto * traits = ggml_get_type_traits_cpu(static_cast<ggml_type>(type));
    if (!traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K)
        throw std::runtime_error("GGML IQ4 vec_dot is not Q8_K");
    float sum;
    traits->vec_dot(width, &sum, 0, weights, 0, q8, 0, 1);
    return sum;
}
void ggml_eval(const Geometry & g, const ExpertRef & e, const block_q8_K * act,
               int T, float * out, float * scratch) {
    const auto gs = iq4_vnni_row_bytes(e.type_gate_up, g.n_embd);
    const auto ds = iq4_vnni_row_bytes(e.type_down, g.ffn_expert);
    const auto * gate = static_cast<const unsigned char *>(e.gate_up);
    const auto * up = e.up ? static_cast<const unsigned char *>(e.up) : gate + gs * g.ffn_expert;
    auto * mid = scratch;
    auto * qm = reinterpret_cast<block_q8_K *>(mid + std::size_t(T) * g.ffn_expert);
    for (int t = 0; t < T; ++t) {
        const auto * x = act + t * (g.n_embd / 256);
        for (int r = 0; r < g.ffn_expert; ++r) {
            const float a = ggml_dot(e.type_gate_up, gate + r * gs, g.n_embd, x);
            const float b = ggml_dot(e.type_gate_up, up + r * gs, g.n_embd, x);
            const float gc = std::min(a, g.swiglu_clamp);
            mid[t * g.ffn_expert + r] = gc / (1 + std::exp(-gc))
                                      * std::max(-g.swiglu_clamp, std::min(b, g.swiglu_clamp));
        }
        iq4_vnni_quantize(mid + t * g.ffn_expert, g.ffn_expert, qm + t * (g.ffn_expert / 256));
        for (int r = 0; r < g.n_embd; ++r)
            out[std::size_t(t) * g.n_embd + r] = ggml_dot(e.type_down,
                static_cast<const unsigned char *>(e.down) + r * ds, g.ffn_expert, qm + t * (g.ffn_expert / 256));
    }
}
#endif
using Quantize = void (*)(const float *, int, void *);
void check_quantizer_case(const std::vector<float> & values, const std::string & name) {
    const int width = static_cast<int>(values.size());
    // Input is deliberately not SIMD-aligned; output is float-aligned only.
    std::vector<float> x(width + 1);
    std::copy(values.begin(), values.end(), x.begin() + 1);
    for (int policy = 0; policy < 2; ++policy) {
        if (policy == 0 && width % 256) continue;
        const auto bytes = policy == 0 ? iq4_vnni_q8_bytes(width) : iq4_vnni_q8_0_bytes(width);
        std::vector<uint32_t> got((bytes + 3) / 4 + 2, 0xa5a5a5a5u), ref((bytes + 3) / 4 + 2, 0xa5a5a5a5u);
        auto * output = reinterpret_cast<unsigned char *>(got.data() + 1);
        auto * expected = reinterpret_cast<unsigned char *>(ref.data() + 1);
        std::memset(expected, 0, bytes); // Public Q8_K zero-bsums contract.
        if (policy == 0) quantize_row_q8_K_ref(x.data() + 1, reinterpret_cast<block_q8_K *>(expected), width);
        else quantize_row_q8_0_ref(x.data() + 1, reinterpret_cast<block_q8_0 *>(expected), width);
        const Quantize quantize = policy == 0 ? iq4_vnni_quantize : iq4_vnni_quantize_q8_0;
        quantize(x.data() + 1, width, output);
        if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(uint32_t))) {
            std::size_t byte = 0;
            while (byte < bytes && output[byte] == expected[byte]) ++byte;
            throw std::runtime_error("quantizer bitwise mismatch " + name + " policy="
                + (policy == 0 ? "Q8_K" : "Q8_0") + " byte=" + std::to_string(byte));
        }
    }
}
void check_quantizers(int random_cases) {
    int cases = 0;
    auto check = [&](const std::vector<float> & values, const std::string & name) {
        check_quantizer_case(values, name);
        ++cases;
    };
    for (int width : {32, 256, 2048, 4096}) {
        std::vector<float> x(width, 0);
        check(x, "zero");
        for (int j = 0; j < width; ++j) x[j] = j % 2 ? -0.0f : 0.0f;
        check(x, "signed-zero");
        for (int sign : {-1, 1}) {
            std::fill(x.begin(), x.end(), 0.25f);
            for (int b = 0; b < width; b += 32) {
                x[b + 3] = sign * 127.0f;
                x[b + 17] = -sign * 127.0f;
                x[b + 31] = sign * 127.0f;
            }
            check(x, "first-signed-maximum");
            for (int adjacent : {-1, 0, 1}) {
                for (int j = 0; j < width; ++j) {
                    const float v = float(j % 32 - 16) + 0.5f;
                    x[j] = adjacent == 0 ? v : std::nextafter(v,
                        adjacent < 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity());
                }
                for (int b = 0; b < width; b += 32) x[b] = sign * 127.0f;
                check(x, "rounding-half-and-neighbors");
            }
        }
        for (int exponent = -40; exponent <= 19; ++exponent) {
            const float scale = std::ldexp(1.0f, exponent);
            for (int j = 0; j < width; ++j) x[j] = float(j % 255 - 127) * scale;
            check(x, "power-of-two-scale=" + std::to_string(exponent));
        }
        const float denorm = std::numeric_limits<float>::denorm_min();
        for (int kind = 0; kind < 3; ++kind) {
            for (int j = 0; j < width; ++j)
                x[j] = (j % 2 ? -1.0f : 1.0f) * (kind == 0 ? denorm : std::ldexp(1.0f, -130)) * float(j % 127 + 1);
            if (kind == 2) for (int b = 0; b < width; b += 32) x[b] = 1.0f;
            check(x, "denormal-kind=" + std::to_string(kind));
        }
        for (int j = 0; j < width; ++j)
            x[j] = j % 2 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        check(x, "largest-finite");
        // Exceptional values must follow the actual legacy reference, including
        // Q8_0 MAX ordering and float-to-int behavior. Run with GGML Release:
        // its Debug nearest_int assertions reject some nonfinite/denorm inputs.
        for (float exceptional : {std::numeric_limits<float>::quiet_NaN(),
                                   -std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()}) {
            for (int position : {0, 15, 16, 31}) {
                std::fill(x.begin(), x.end(), 1.0f);
                for (int b = 0; b < width; b += 32) x[b + position] = exceptional;
                check(x, "nonfinite-position=" + std::to_string(position));
            }
            std::fill(x.begin(), x.end(), exceptional);
            check(x, "all-nonfinite");
        }
    }
    std::mt19937 rng(0x51384b30);
    std::normal_distribution<float> normal;
    for (int c = 0; c < random_cases; ++c) {
        const int width = c % 2 ? 4096 : 2048;
        std::vector<float> x(width);
        const float scale = std::ldexp(1.0f, int(rng() % 60) - 40);
        for (float & v : x) v = normal(rng) * scale;
        if (c % 4 == 1) { x[0] = 100 * scale; x[width - 1] = -100 * scale; }
        if (c % 4 == 2) std::fill_n(x.begin(), 256, 0.0f);
        if (c % 4 == 3) for (float & v : x) v = std::round(v / scale) * scale;
        check(x, "random=" + std::to_string(c));
    }
#if defined(__x86_64__) || defined(__i386__)
    struct RestoreCSR { unsigned csr = _mm_getcsr(); ~RestoreCSR() { _mm_setcsr(csr); } } restore;
    const std::vector<float> x(256, 1.0f);
    for (unsigned flags : {0x2000u, 0x4000u, 0x6000u, 0x8000u, 0x0040u}) {
        _mm_setcsr((restore.csr & ~0xe040u) | flags);
        check(x, "non-default-MXCSR=" + std::to_string(flags));
    }
#endif
    std::cout << "quantizer bitwise reference parity cases=" << cases
              << " random_cases=" << random_cases << " path="
              << (iq4_vnni_available() ? "avx512+block-reference-fallback" : "scalar-reference-fallback")
              << " full_Q8_K_d_qs_bsums_and_Q8_0_bytes=PASS\n";
}
void check_rows(std::mt19937 & rng, IQ4VNNIPath path) {
    for (int type : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL}) {
        const int width = type == GGML_TYPE_IQ4_XS ? 4096 : 2048;
        auto bytes = synthetic(type, width, 1, rng);
        std::vector<unsigned char> unaligned(bytes.size() + 1);
        std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
        std::vector<float> x(width), unpacked(width);
        std::vector<block_q8_K> q(width / 256);
        for (int trial = 0; trial < 12; ++trial) {
            for (float & v : x) v = trial == 0 ? 0 : float(int(rng() % 2001) - 1000) / 1000;
            std::memset(q.data(), 0xa5, q.size() * sizeof(block_q8_K));
            iq4_vnni_quantize(x.data(), width, q.data());
            if (trial == 1) { // Include -128, all signed-byte values, nonzero bsums.
                for (auto & b : q) {
                    for (int j = 0; j < 256; ++j) b.qs[j] = static_cast<int8_t>(j - 128);
                    for (int j = 0; j < 16; ++j) {
                        b.bsums[j] = 0;
                        for (int k = 0; k < 16; ++k) b.bsums[j] += b.qs[j * 16 + k];
                    }
                }
            }
            const float got = iq4_vnni_row(type, unaligned.data() + 1, width, q.data(), path);
            const float scalar = iq4_vnni_row(type, unaligned.data() + 1, width, q.data(), IQ4VNNIPath::scalar);
            report("row integer dispatch type=" + std::to_string(type) + " trial=" + std::to_string(trial),
                   compare({got}, {scalar}, 0.00002, 0.00002));
            const auto dx = dequant_q8(q.data(), width);
            const float expected = f32_dot(type, unaligned.data() + 1, width, dx.data(), unpacked);
            report("row GGML dequantized-Q8 F32", compare({got}, {expected}, 0.00002, 0.00002));
#ifdef STRATA_VNNI_GGML_DOT
            // GGML's x86 sign/abs byte implementation assumes Q8_K's
            // quantizer domain [-127,127]; the deliberate -128 stress row
            // remains checked against independent F32, not that baseline.
            if (type == GGML_TYPE_IQ4_XS && trial != 1) report("row GGML vec_dot Q8_K", compare({got},
                {ggml_dot(type, bytes.data(), width, q.data())}, 0.00002, 0.00002));
#endif
            if (trial == 0)
                for (const auto & b : q) for (int16_t sum : b.bsums)
                    if (sum) throw std::runtime_error("zero block has stale bsums");
        }
        const auto rows = synthetic(type, width, 32, rng);
        const auto stride = iq4_vnni_row_bytes(type, width);
        std::vector<unsigned char> unaligned_rows(rows.size() + 1);
        std::copy(rows.begin(), rows.end(), unaligned_rows.begin() + 1);
        for (int n_rows : {1, 2, 32}) {
            std::vector<float> got(n_rows), expected(n_rows);
            iq4_vnni_rows(type, unaligned_rows.data() + 1, width, q.data(), n_rows, got.data(), path);
            for (int r = 0; r < n_rows; ++r)
                expected[r] = iq4_vnni_row(type, unaligned_rows.data() + 1 + std::size_t(r) * stride,
                                          width, q.data(), path);
            report("consecutive weight rows type=" + std::to_string(type)
                   + " rows=" + std::to_string(n_rows), compare(got, expected, 0, 0));
            for (int r = 0; r < n_rows; ++r)
                expected[r] = iq4_vnni_row(type, unaligned_rows.data() + 1 + std::size_t(r) * stride,
                                          width, q.data(), IQ4VNNIPath::scalar);
            report("consecutive rows versus scalar", compare(got, expected, 0.00002, 0.00002));
        }
    }
}
void check_q8_0_rows(IQ4VNNIPath path) {
    std::mt19937 rng(0x513830);
    for (int width : {32, 64, 256, 2048}) {
        const auto bytes = synthetic(GGML_TYPE_IQ4_NL, width, 32, rng);
        std::vector<unsigned char> weights(bytes.size() + 1);
        std::copy(bytes.begin(), bytes.end(), weights.begin() + 1);
        const auto stride = std::size_t(width / 32) * sizeof(block_iq4_nl);
        std::vector<float> x(width), input(width), unpacked(width);
        std::vector<block_q8_0> q(width / 32);
        if (iq4_vnni_q8_0_bytes(width) != q.size() * sizeof(block_q8_0))
            throw std::runtime_error("Q8_0 storage contract mismatch");
        for (int trial = 0; trial < 3; ++trial) {
            for (float & v : x) v = trial == 0 ? 0 : float(int(rng() % 2001) - 1000) / 1000;
            iq4_vnni_quantize_q8_0(x.data(), width, q.data());
            if (trial == 1) {
                for (int b = 0; b < width / 32; ++b) {
                    q[b].d = ggml_fp32_to_fp16(0.5f);
                    for (int j = 0; j < 32; ++j) q[b].qs[j] = static_cast<int8_t>((b * 32 + j) % 256 - 128);
                }
            }
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), input.data(), width);
            for (int n_rows : {1, 2, 32}) {
                std::vector<float> got(n_rows), scalar(n_rows), expected(n_rows), singles(n_rows);
                iq4_vnni_nl_q8_0_rows(weights.data() + 1, width, q.data(), n_rows, got.data(), path);
                iq4_vnni_nl_q8_0_rows(weights.data() + 1, width, q.data(), n_rows, scalar.data(), IQ4VNNIPath::scalar);
                for (int r = 0; r < n_rows; ++r) {
                    const auto * row = weights.data() + 1 + r * stride;
                    expected[r] = f32_dot(GGML_TYPE_IQ4_NL, row, width, input.data(), unpacked);
                    iq4_vnni_nl_q8_0_rows(row, width, q.data(), 1, &singles[r], path);
                }
                const auto name = "native NL/Q8_0 width=" + std::to_string(width)
                                + " rows=" + std::to_string(n_rows) + " trial=" + std::to_string(trial);
                report(name + " F32", compare(got, expected, 0.00002, 0.00002));
                report(name + " scalar", compare(got, scalar, 0.00002, 0.00002));
                report(name + " singles", compare(got, singles, 0, 0));
#ifdef STRATA_VNNI_GGML_DOT
                if (trial != 1) {
                    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
                    if (!traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_0)
                        throw std::runtime_error("native IQ4_NL vec_dot policy is not Q8_0");
                    for (int r = 0; r < n_rows; ++r)
                        traits->vec_dot(width, &expected[r], 0, weights.data() + 1 + r * stride, 0, q.data(), 0, 1);
                    report(name + " GGML", compare(got, expected, 0.00002, 0.00002));
                }
#endif
            }
        }
        float sentinel = 53;
        bool rejected = false;
        try { iq4_vnni_nl_q8_0_rows(weights.data() + 1, width, q.data(), 0, &sentinel, path); }
        catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected || sentinel != 53) throw std::runtime_error("invalid Q8_0 rows modified output");
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        bool quick = false, require = false, quantizer_only = false;
        int T = 8, repeats = 0, quantizer_cases = -1;
        std::string dir;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--quick") quick = true;
            else if (arg == "--require-vnni") require = true;
            else if (arg == "--quantizer-only") quantizer_only = true;
            else if (arg == "--quantizer-cases" && i + 1 < argc) quantizer_cases = std::stoi(argv[++i]);
            else if (arg == "--tokens" && i + 1 < argc) T = std::stoi(argv[++i]);
            else if (arg == "--fixture-dir" && i + 1 < argc) dir = argv[++i];
            else if (arg == "--benchmark" && i + 1 < argc) repeats = std::stoi(argv[++i]);
            else throw std::runtime_error("usage: glm_vnni_parity [--quick] [--require-vnni] [--fixture-dir DIR --tokens 4] [--benchmark repeats] [--quantizer-only] [--quantizer-cases N]");
        }
        if (quantizer_cases < -1) throw std::runtime_error("invalid quantizer case count");
        if (T < 1 || T > 8 || repeats < 0 || (repeats && dir.empty())) throw std::runtime_error("invalid tokens/benchmark; benchmark requires real fixture");
        if (require && !iq4_vnni_available()) throw std::runtime_error("AVX512 VNNI required but unavailable");
        const auto path = iq4_vnni_available() ? IQ4VNNIPath::vnni
                        : iq4_int8_available() ? IQ4VNNIPath::avx2 : IQ4VNNIPath::scalar;
        std::cout << std::setprecision(10) << "path="
                  << (iq4_vnni_available() ? "avx512-vnni-q8k" : iq4_int8_available() ? "avx2-q8k" : "scalar-q8k")
                  << " Q8 activation approximation; NOT exact F32 baseline\n";
#ifdef STRATA_VNNI_GGML_DOT
        ggml_cpu_init(); // Initialize the CPU FP16 lookup table before direct vec_dot calls.
#endif
        check_quantizers(quantizer_cases >= 0 ? quantizer_cases : quantizer_only ? 20000 : quick ? 200 : 2000);
        if (quantizer_only) return 0;
        std::mt19937 rng(0x47564e4e);
        check_rows(rng, path);
        check_q8_0_rows(path);
        if (iq4_int8_available() && path != IQ4VNNIPath::avx2) {
            // the AVX2 int8 rows (GLM_CPU_ISA=avx2, CPUs without AVX512 VNNI) against the same scalar / F32 / GGML refs
            std::cout << "path=avx2-q8k rows\n";
            std::mt19937 rng2(0x41565832);
            check_rows(rng2, IQ4VNNIPath::avx2);
            check_q8_0_rows(IQ4VNNIPath::avx2);
        }
        Geometry g;
        if (quick && dir.empty()) { g.n_embd = 256; g.ffn_expert = 256; }
        const auto gu_size = iq4_vnni_row_bytes(GGML_TYPE_IQ4_XS, g.n_embd) * 2 * g.ffn_expert;
        const auto down_size = iq4_vnni_row_bytes(GGML_TYPE_IQ4_NL, g.ffn_expert) * g.n_embd;
        const auto blob_size = gu_size + down_size;
        const int experts = dir.empty() ? 1 : 8;
        std::vector<unsigned char> blob;
        if (dir.empty()) {
            blob = synthetic(GGML_TYPE_IQ4_XS, g.n_embd, 2 * g.ffn_expert, rng);
            auto down = synthetic(GGML_TYPE_IQ4_NL, g.ffn_expert, g.n_embd, rng);
            blob.insert(blob.end(), down.begin(), down.end());
        } else {
            blob = load(dir + "/experts.bin");
            if (blob.size() != experts * blob_size) throw std::runtime_error("packed eight-expert fixture size mismatch");
        }
        std::vector<float> x(std::size_t(T) * g.n_embd);
        if (dir.empty()) for (float & v : x) v = float(int(rng() % 2001) - 1000) / 1000;
        else {
            auto bytes = load(dir + "/x.f32");
            if (bytes.size() != x.size() * sizeof(float)) throw std::runtime_error("x fixture size mismatch");
            std::memcpy(x.data(), bytes.data(), bytes.size());
        }
        std::vector<float> captured;
        if (!dir.empty()) {
            const auto bytes = load(dir + "/output.f32");
            if (bytes.size() != std::size_t(experts) * x.size() * sizeof(float))
                throw std::runtime_error("output fixture size mismatch");
            captured.resize(std::size_t(experts) * x.size());
            std::memcpy(captured.data(), bytes.data(), bytes.size());
        }
        std::vector<float> captured_mid;
        if (!dir.empty() && std::ifstream(dir + "/activation.f32", std::ios::binary)) {
            const auto bytes = load(dir + "/activation.f32");
            if (bytes.size() != std::size_t(experts) * T * g.ffn_expert * sizeof(float))
                throw std::runtime_error("activation fixture size mismatch");
            captured_mid.resize(std::size_t(experts) * T * g.ffn_expert);
            std::memcpy(captured_mid.data(), bytes.data(), bytes.size());
        }
        std::vector<block_q8_K> act(vnni_expert_activation_bytes(g, T) / sizeof(block_q8_K));
        std::vector<float> scratch((vnni_expert_scratch_bytes(g, T) + 3) / 4);
        std::vector<float> out(x.size()), reordered(x.size());
        vnni_expert_prepare(g, x.data(), T, act.data());
        const auto immutable = act;
        std::vector<int> tokens(T);
        for (int t = 0; t < T; ++t) {
            tokens[t] = T - 1 - t;
            std::copy_n(x.data() + std::size_t(tokens[t]) * g.n_embd, g.n_embd, reordered.data() + std::size_t(t) * g.n_embd);
        }
        for (int expert = 0; expert < experts; ++expert) {
            const auto * base = blob.data() + expert * blob_size;
            ExpertRef e{base, base + gu_size, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
            for (int role = 0; role < 3; ++role) {
                const int type = role == 2 ? GGML_TYPE_IQ4_NL : GGML_TYPE_IQ4_XS;
                const int width = role == 2 ? g.ffn_expert : g.n_embd;
                const auto * weights = base + (role == 2 ? gu_size : role * gu_size / 2);
                const auto stride = iq4_vnni_row_bytes(type, width);
                std::vector<block_q8_K> q(width / 256);
                iq4_vnni_quantize(x.data(), width, q.data());
                std::vector<float> got(32), expected(32), unpacked(width);
                const auto input = dequant_q8(q.data(), width);
                iq4_vnni_rows(type, weights, width, q.data(), 32, got.data(), path);
                for (int r = 0; r < 32; ++r)
                    expected[r] = f32_dot(type, weights + r * stride, width, input.data(), unpacked);
                report("expert=" + std::to_string(expert) + " row API role=" + std::to_string(role),
                       compare(got, expected, 0.00002, 0.00002));
            }
            vnni_expert_eval(g, e, act.data(), T, T, tokens.data(), out.data(), scratch.data(), path);
            const auto reference = model_reference(g, e, reordered, T, true, true);
            report("expert=" + std::to_string(expert) + " SAME Q8 intermediate / GGML F32 down",
                   compare(out, reference, 0.0002, 0.00005));
            const auto independent = model_reference(g, e, reordered, T, true, false);
            report("expert independent dequantized-Q8 model", compare(out, independent, 0.003, 0.001));
            const auto full_f32 = model_reference(g, e, reordered, T, false, false);
            report("expert Q8 approximation versus all-F32", compare(out, full_f32, 0, 0), false);
            if (!captured.empty()) {
                std::vector<float> fixture_expected(x.size());
                for (int t = 0; t < T; ++t)
                    std::copy_n(captured.data() + std::size_t(expert) * x.size()
                                + std::size_t(tokens[t]) * g.n_embd, g.n_embd,
                                fixture_expected.data() + std::size_t(t) * g.n_embd);
                report("expert Q8 approximation versus captured F32 fixture",
                       compare(out, fixture_expected, 0, 0), false);
            }
            if (!captured_mid.empty()) {
                std::vector<block_q8_0> q0(g.ffn_expert / 32);
                std::vector<block_q8_K> qk(g.ffn_expert / 256);
                std::vector<float> native_down(x.size()), q8k_down(x.size()), expected(x.size());
                std::vector<float> mid_input(g.ffn_expert), unpacked(g.ffn_expert);
                const auto stride = iq4_vnni_row_bytes(GGML_TYPE_IQ4_NL, g.ffn_expert);
                for (int t = 0; t < T; ++t) {
                    const auto * mid = captured_mid.data() + (std::size_t(expert) * T + t) * g.ffn_expert;
                    iq4_vnni_quantize_q8_0(mid, g.ffn_expert, q0.data());
                    iq4_vnni_quantize(mid, g.ffn_expert, qk.data());
                    iq4_vnni_nl_q8_0_rows(e.down, g.ffn_expert, q0.data(), g.n_embd,
                                          native_down.data() + std::size_t(t) * g.n_embd, path);
                    iq4_vnni_rows(GGML_TYPE_IQ4_NL, e.down, g.ffn_expert, qk.data(), g.n_embd,
                                   q8k_down.data() + std::size_t(t) * g.n_embd, path);
                    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q0.data(), mid_input.data(), g.ffn_expert);
                    for (int r = 0; r < g.n_embd; ++r)
                        expected[std::size_t(t) * g.n_embd + r] = f32_dot(GGML_TYPE_IQ4_NL,
                            static_cast<const unsigned char *>(e.down) + r * stride,
                            g.ffn_expert, mid_input.data(), unpacked);
                }
                report("expert=" + std::to_string(expert) + " real native NL/Q8_0 down F32",
                       compare(native_down, expected, 0.00002, 0.00002));
                std::vector<float> saved(captured.begin() + std::size_t(expert) * x.size(),
                                         captured.begin() + std::size_t(expert + 1) * x.size());
                report("down-only Q8_0 approximation versus captured F32", compare(native_down, saved, 0, 0), false);
                report("down-only Q8_K approximation versus captured F32", compare(q8k_down, saved, 0, 0), false);
            }
            const auto batch = out;
            for (int t = 0; t < T; ++t)
                vnni_expert_eval(g, e, act.data(), T, 1, &tokens[t], out.data() + std::size_t(t) * g.n_embd, scratch.data(), path);
            report("single jobs vs batched", compare(out, batch, 0, 0));
            e.up = base + gu_size / 2;
            vnni_expert_eval(g, e, act.data(), T, T, tokens.data(), out.data(), scratch.data(), path);
            report("separate mapped up plane", compare(out, batch, 0, 0));
            e.type_up = GGML_TYPE_IQ4_XS;
            vnni_expert_eval(g, e, act.data(), T, T, tokens.data(), out.data(), scratch.data(), path);
            report("explicit matching up type", compare(out, batch, 0, 0));
            e.type_up = GGML_TYPE_IQ4_NL;
            bool mixed_rejected = false;
            try { vnni_expert_eval(g, e, act.data(), T, T, tokens.data(), out.data(), scratch.data(), path); }
            catch (const std::invalid_argument &) { mixed_rejected = true; }
            if (!mixed_rejected || out != batch) throw std::runtime_error("mixed up type not rejected before writes");
            e.type_up = -1;
            const auto before = out;
            tokens[0] = T;
            bool rejected = false;
            try { vnni_expert_eval(g, e, act.data(), T, T, tokens.data(), out.data(), scratch.data(), path); }
            catch (const std::invalid_argument &) { rejected = true; }
            tokens[0] = T - 1;
            if (!rejected || out != before) throw std::runtime_error("invalid token not rejected before writes");
        }
        if (std::memcmp(act.data(), immutable.data(), act.size() * sizeof(block_q8_K))) throw std::runtime_error("shared activation modified");
        if (repeats) {
            // Raw wall times only: the caller must establish idle runners and
            // low foreign CPU load before treating these as performance data.
#ifdef STRATA_VNNI_GGML_DOT
            const int modes = 3;
#else
            const int modes = 2;
            std::cout << "GGML vec_dot baseline unavailable (ggml-cpu target absent)\n";
#endif
            for (int mode = 0; mode < modes; ++mode) {
                const auto p = mode == 0 ? IQ4VNNIPath::scalar : path;
                const char * label = mode == 0 ? "scalar-q8k" : mode == 2 ? "ggml-xs-vecdot+scalar-nl-q8k"
                    : (iq4_vnni_available() ? "vnni-q8k" : "scalar-q8k");
                const auto start = std::chrono::steady_clock::now();
                for (int it = 0; it < repeats; ++it)
                    for (int expert = 0; expert < experts; ++expert) {
                        const auto * base = blob.data() + expert * blob_size;
                        ExpertRef e{base, base + gu_size, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
#ifdef STRATA_VNNI_GGML_DOT
                        if (mode == 2) ggml_eval(g, e, act.data(), T, out.data(), scratch.data());
                        else
#endif
                            vnni_expert_eval(g, e, act.data(), T, T, nullptr, out.data(), scratch.data(), p);
                    }
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
                std::cout << "RAW_TIMING path=" << label
                          << " real_experts=8 T=" << T << " ms_per_8experts=" << ms
                          << " weight_GBps=" << (8 * double(blob_size) / 1e6 / ms)
                          << " load_not_validated=1 no_performance_claim=1\n";
            }
        }
        std::cout << "VNNI quantized-model parity PASS; F32 approximation metrics are informational\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
