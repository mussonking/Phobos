// glm_miss_error: per-call error of the int8 miss paths (AVX2 vs AVX512 VNNI) against the F64 reference, on the
// model's real experts. One call = one routed expert on one token, as routed_finish computes a miss:
//   y -> Q8_K -> IQ4_XS gate / up rows -> SwiGLU clamp -> mid -> Q8_0 -> IQ4_NL down rows -> out[4096]
// VNNI path (prod): iq4_vnni_quantize / _q8_0 (AVX512 fast quantizers) + VNNI dots. The R8 kernels prod uses are the
// same integer dots in another float order (cpu-exact bound, ~1e-7), not modeled here.
// AVX2 path (GLM_CPU_ISA=avx2): ggml's reference quantizers (quantize_row_q8_K_ref / q8_0_ref) + AVX2 dots.
// F64: the exact F32 weights, double activations, products and sums (iq4_cpu_row_f64).
// Reported per amax / rms bin of the input row: relative L2 error of gate | up, of down alone (fed the F64 mid), of
// the whole call, both paths, the AVX2 - VNNI distance, and whether the quantized bytes of the two paths are equal.
// Inputs: synthetic rows (Gaussian plus outliers, amax / rms swept) or --dump FILE of real miss rows (records
// int32 layer, int32 expert, float y[n_embd]: GLM_MISS_DUMP). Single-threaded.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "iq4_cpu.hpp"
#include "iq4_vnni.hpp"
#include "strata/glm/gguf.hpp"

extern "C" void quantize_row_q8_K_ref(const float *, void *, int64_t);
extern "C" void quantize_row_q8_0_ref(const float *, void *, int64_t);

using namespace strata::glm;

namespace {

constexpr int kXs = 23, kNl = 20;   // GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL

struct Expert {
    const uint8_t * gate, * up, * down;
    size_t gb, db;   // row bytes
};

double rel_l2(const std::vector<float> & a, const std::vector<double> & ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e += (a[i] - ref[i]) * (a[i] - ref[i]);
        n += ref[i] * ref[i];
    }
    return n > 0 ? std::sqrt(e / n) : 0;
}

double rel_l2f(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        e += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        n += (double) b[i] * b[i];
    }
    return n > 0 ? std::sqrt(e / n) : 0;
}

double amax_rms(const float * x, int n) {
    double s = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        s += (double) x[i] * x[i];
        m = std::max(m, (double) std::fabs(x[i]));
    }
    return s > 0 ? m / std::sqrt(s / n) : 0;
}

struct Acc {
    std::vector<double> v[7];   // gate|up vnni, avx2; down vnni, avx2; call vnni, avx2; avx2-vnni
    int n = 0, qk_diff = 0, q0_diff = 0;
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * v.size()))];
}

}  // namespace

int main(int argc, char ** argv) {
    std::string gguf, dump;
    int calls = 48, seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--gguf") gguf = next();
        else if (a == "--dump") dump = next();
        else if (a == "--calls") calls = std::atoi(next().c_str());
        else if (a == "--seed") seed = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (gguf.empty()) {
        std::fprintf(stderr, "usage: glm_miss_error --gguf GGUF [--dump FILE] [--calls N] [--seed S]\n");
        return 2;
    }
    if (!iq4_vnni_available()) {
        std::fprintf(stderr, "needs an AVX512 VNNI CPU (both paths run here) and GLM_CPU_ISA unset\n");
        return 2;
    }
    try {
        const GgufSet f(gguf);
        // MoE layers and the expert geometry from the tensors
        std::vector<int> moe;
        int D = 0, F = 0, NE = 0;
        for (int il = 0; il < 128; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            if (!f.has(b + "ffn_gate_exps.weight")) continue;
            const HostTensor t = f.tensor(b + "ffn_gate_exps.weight");
            D = (int) t.shape[0];
            F = (int) t.shape[1];
            NE = (int) t.shape[2];
            moe.push_back(il);
        }
        if (moe.empty()) throw std::runtime_error("no routed expert tensors");
        auto expert = [&](int il, int e) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const HostTensor g = f.tensor(b + "ffn_gate_exps.weight"), u = f.tensor(b + "ffn_up_exps.weight"),
                             d = f.tensor(b + "ffn_down_exps.weight");
            if (g.type != kXs || u.type != kXs || d.type != kNl)
                throw std::runtime_error("expects IQ4_XS gate / up and IQ4_NL down");
            Expert x;
            x.gb = iq4_vnni_row_bytes(kXs, D);
            x.db = (size_t) F / 32 * 18;
            x.gate = g.data + g.nbytes / NE * e;
            x.up = u.data + u.nbytes / NE * e;
            x.down = d.data + d.nbytes / NE * e;
            return x;
        };

        // inputs: (layer, expert, y)
        struct In { int il, e; std::vector<float> y; };
        std::vector<In> ins;
        std::mt19937 rng(seed);
        if (!dump.empty()) {
            FILE * fp = std::fopen(dump.c_str(), "rb");
            if (!fp) throw std::runtime_error("cannot open " + dump);
            int32_t h[2];
            while (std::fread(h, 4, 2, fp) == 2) {
                In x{h[0], h[1], std::vector<float>(D)};
                if (std::fread(x.y.data(), 4, D, fp) != (size_t) D) break;
                ins.push_back(std::move(x));
            }
            std::fclose(fp);
            if ((int) ins.size() > calls) {   // an even sample of the dump
                std::vector<In> s;
                for (int i = 0; i < calls; ++i) s.push_back(std::move(ins[(size_t) i * ins.size() / calls]));
                ins.swap(s);
            }
        } else {
            // Gaussian rows with k outlier channels scaled so that amax / rms hits the target
            const double targets[5] = {4, 10, 30, 100, 300};
            std::normal_distribution<float> nd(0.f, 1.f);
            for (int c = 0; c < calls; ++c) {
                In x{moe[rng() % moe.size()], (int) (rng() % NE), std::vector<float>(D)};
                for (float & v : x.y) v = nd(rng);
                const double tgt = targets[c % 5];
                if (tgt > 4) {
                    const int k = 1 + (int) (rng() % 4);
                    double base = 0;
                    for (float v : x.y) base += (double) v * v;
                    // outlier value o on k channels: o / sqrt((base + k o^2) / D) = tgt
                    const double r2 = tgt * tgt / D;
                    const double o = std::sqrt(r2 * base / std::max(1e-9, 1 - r2 * k));
                    for (int j = 0; j < k; ++j) x.y[rng() % D] = (float) ((rng() & 1) ? o : -o);
                }
                ins.push_back(std::move(x));
            }
        }

        const size_t qk = iq4_vnni_q8_bytes(D), qm = iq4_vnni_q8_0_bytes(F);
        std::vector<uint8_t> qy_v(qk + 64), qy_a(qk + 64), qm_v(qm + 64), qm_a(qm + 64), qm_x(qm + 64);
        auto al = [](std::vector<uint8_t> & b) { return b.data() + (64 - ((uintptr_t) b.data() & 63)) % 64; };
        std::map<int, Acc> bins;   // by amax / rms bin (power of ~3)
        auto bin_of = [](double r) { return r < 6 ? 0 : r < 18 ? 1 : r < 55 ? 2 : r < 170 ? 3 : 4; };
        const char * bin_name[5] = {"<6", "6-18", "18-55", "55-170", ">=170"};
        std::vector<float> gv(F), uv(F), mid_v(F), mid_a(F), out_v(D), out_a(D), dv_f(D), da_f(D), g64f(F), mid64f(F);
        std::vector<double> y64(D), g64(F), u64(F), mid64(F), out64(D), gu64(2 * F);
        for (const In & x : ins) {
            const Expert ex = expert(x.il, x.e);
            for (int i = 0; i < D; ++i) y64[i] = x.y[i];
            // F64 reference
            const double * yp = y64.data();
            for (int r = 0; r < F; ++r) {
                iq4_cpu_row_f64(kXs, ex.gate + r * ex.gb, D, 1, &yp, &g64[r]);
                iq4_cpu_row_f64(kXs, ex.up + r * ex.gb, D, 1, &yp, &u64[r]);
                mid64[r] = expert_swiglu_clamp_f64(g64[r], u64[r]);
            }
            const double * mp = mid64.data();
            for (int r = 0; r < D; ++r) iq4_cpu_row_f64(kNl, ex.down + r * ex.db, F, 1, &mp, &out64[r]);
            for (int r = 0; r < F; ++r) {
                gu64[r] = g64[r];
                gu64[F + r] = u64[r];
            }
            Acc & A = bins[bin_of(amax_rms(x.y.data(), D))];
            ++A.n;
            // both paths: quantize, gate / up, mid, Q8_0, down
            uint8_t * qyv = al(qy_v), * qya = al(qy_a), * qmv = al(qm_v), * qma = al(qm_a), * qmx = al(qm_x);
            iq4_vnni_quantize(x.y.data(), D, qyv);
            std::memset(qya, 0, qk);
            quantize_row_q8_K_ref(x.y.data(), qya, D);
            A.qk_diff += std::memcmp(qyv, qya, qk) != 0;
            for (int p = 0; p < 2; ++p) {
                const IQ4VNNIPath path = p == 0 ? IQ4VNNIPath::vnni : IQ4VNNIPath::avx2;
                const uint8_t * q = p == 0 ? qyv : qya;
                iq4_vnni_rows(kXs, ex.gate, D, q, F, gv.data(), path);
                iq4_vnni_rows(kXs, ex.up, D, q, F, uv.data(), path);
                std::vector<float> gu(2 * F);
                for (int r = 0; r < F; ++r) {
                    gu[r] = gv[r];
                    gu[F + r] = uv[r];
                }
                A.v[p].push_back(rel_l2(gu, gu64));
                std::vector<float> & mid = p == 0 ? mid_v : mid_a;
                for (int r = 0; r < F; ++r) mid[r] = expert_swiglu_clamp(gv[r], uv[r]);
                uint8_t * qmid = p == 0 ? qmv : qma;
                if (p == 0) iq4_vnni_quantize_q8_0(mid.data(), F, qmid);
                else {
                    std::memset(qmid, 0, qm);
                    quantize_row_q8_0_ref(mid.data(), qmid, F);
                }
                std::vector<float> & out = p == 0 ? out_v : out_a;
                iq4_vnni_nl_q8_0_rows(ex.down, F, qmid, D, out.data(), path);
                A.v[4 + p].push_back(rel_l2(out, out64));
                // down alone: the F64 mid through this path's Q8_0 + down
                for (int r = 0; r < F; ++r) mid64f[r] = (float) mid64[r];
                if (p == 0) iq4_vnni_quantize_q8_0(mid64f.data(), F, qmx);
                else {
                    std::memset(qmx, 0, qm);
                    quantize_row_q8_0_ref(mid64f.data(), qmx, F);
                }
                std::vector<float> & dd = p == 0 ? dv_f : da_f;
                iq4_vnni_nl_q8_0_rows(ex.down, F, qmx, D, dd.data(), path);
                // reference for down alone: F64 down on the float-rounded F64 mid
                std::vector<double> m2(F), o2(D);
                for (int r = 0; r < F; ++r) m2[r] = mid64f[r];
                const double * m2p = m2.data();
                for (int r = 0; r < D; ++r) iq4_cpu_row_f64(kNl, ex.down + r * ex.db, F, 1, &m2p, &o2[r]);
                A.v[2 + p].push_back(rel_l2(dd, o2));
            }
            {   // Q8_0 bytes of the same mid through both quantizers
                uint8_t * a = al(qm_v), * b = al(qm_a);
                iq4_vnni_quantize_q8_0(mid_v.data(), F, a);
                std::memset(b, 0, qm);
                quantize_row_q8_0_ref(mid_v.data(), b, F);
                A.q0_diff += std::memcmp(a, b, qm) != 0;
            }
            A.v[6].push_back(rel_l2f(out_a, out_v));
        }

        std::printf("glm_miss_error: %zu calls (%s), D %d, F %d, %zu MoE layers\n", ins.size(),
                    dump.empty() ? "synthetic Gaussian + outliers" : dump.c_str(), D, F, moe.size());
        std::printf("relative L2 error vs F64 per call, median / p95 / max; quantized bytes AVX2 path != VNNI path\n");
        for (const auto & [b, A] : bins) {
            std::printf("amax/rms %-6s n=%3d  Q8_K y differ %d, Q8_0 mid differ %d\n", bin_name[b], A.n, A.qk_diff,
                        A.q0_diff);
            const char * nm[7] = {"gate|up VNNI", "gate|up AVX2", "down    VNNI", "down    AVX2", "call    VNNI",
                                  "call    AVX2", "call AVX2-VNNI"};
            for (int k = 0; k < 7; ++k)
                std::printf("  %-15s %.3e / %.3e / %.3e\n", nm[k], pct(A.v[k], 0.5), pct(A.v[k], 0.95),
                            pct(A.v[k], 1.0));
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "glm_miss_error: %s\n", e.what());
        return 1;
    }
    return 0;
}
