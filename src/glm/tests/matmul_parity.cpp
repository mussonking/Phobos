// glm_matmul parity against the llama.cpp oracle (CPU, UD-Q4_K_XL weights) + bandwidth on the V100.
//
//   glm_matmul_parity [--oracle DIR] [--gguf SHARD1] [--bench] [--case NAME]
//
// Every case takes a real weight from the GGUF, the oracle's input node and the oracle's output node of the
// 30-token prompt (phase 0). Token subsets (T = 1..30) and tilings of the 30 tokens (T = 64, 512) exercise the GEMV
// and the GEMM paths with exact expected values: a matmul is column-independent.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "strata/glm/matmul.hpp"
#include "strata/glm/oracle.hpp"

using namespace strata::glm;
using namespace strata::glm::test;

namespace {

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

struct Case {
    const char * label;
    const char * weight;
    int x_seq;        // phase-0 oracle node seq of the input (exact node, names repeat)
    int y_seq;        // phase-0 oracle node seq of the output
    int heads;        // batch of a 3D weight; the input is token-major [T][heads][n_in], the output [heads][T][n_out]
};

// Phase-0 node seqs from /mnt/models/glm-oracle/q4kxl-p32/index.tsv (llama.cpp 42d95816).
const Case kCases[] = {
    {"ffn_gate blk0 (4096->12288)", "blk.0.ffn_gate.weight", 126, 127, 1},
    {"ffn_down blk0 (12288->4096)", "blk.0.ffn_down.weight", 129, 130, 1},
    {"kda q blk0 (4096->8192)", "blk.0.attn_q.weight", 15, 26, 1},
    {"kda f_a blk0 (4096->128)", "blk.0.ssm_f_a.weight", 15, 75, 1},
    {"kda f_b blk0 (128->8192)", "blk.0.ssm_f_b.weight", 75, 76, 1},
    {"kda out blk0 (8192->4096)", "blk.0.attn_output.weight", 103, 104, 1},
    {"hc_attn_fn blk0 (16384->24)", "blk.0.hc_attn_fn.weight", 4, 5, 1},
    {"router blk3 F32 (4096->288)", "blk.3.ffn_gate_inp.weight", 546, 547, 1},
    {"mla q_a blk3 (4096->1536)", "blk.3.attn_q_a.weight", -1, 453, 1},
    {"mla q_b blk3 (1536->16384)", "blk.3.attn_q_b.weight", 455, 456, 1},
    {"mla k_b blk3 3D (256->512 x64)", "blk.3.attn_k_b.weight", 457, 459, 64},
    {"mla v_b blk3 3D (512->256 x64)", "blk.3.attn_v_b.weight", 518, 520, 64},
    {"mla out blk3 (16384->4096)", "blk.3.attn_output.weight", 523, 524, 1},
    {"shexp gate blk3 (4096->2048)", "blk.3.ffn_gate_shexp.weight", 546, 581, 1},
    {"output head (4096->154880)", "output.weight", 7727, 7728, 1},
};

struct Dev {
    void * p = nullptr;
    explicit Dev(size_t n) { CK(cudaMalloc(&p, std::max<size_t>(n, 256))); }
    ~Dev() { cudaFree(p); }
    template <class T> T * as() const { return static_cast<T *>(p); }
};

// The expected output of a tiled input: token t is the oracle's token t % T0.
std::vector<float> tile_x(const std::vector<float> & x, int T0, int T, size_t tok) {
    std::vector<float> v((size_t) T * tok);
    for (int t = 0; t < T; ++t) std::memcpy(&v[(size_t) t * tok], &x[(size_t) (t % T0) * tok], tok * 4);
    return v;
}
std::vector<float> tile_y(const std::vector<float> & y, int T0, int T, int heads, int n_out) {
    std::vector<float> v((size_t) heads * T * n_out);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < T; ++t)
            std::memcpy(&v[((size_t) h * T + t) * n_out], &y[((size_t) h * T0 + t % T0) * n_out], (size_t) n_out * 4);
    return v;
}

// Host conversions for the BF16/F16 variants.
std::vector<float> dequant_q8_0(const uint8_t * p, size_t n) {
    std::vector<float> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const float d = __half2float(*reinterpret_cast<const __half *>(p + b * 34));
        for (int j = 0; j < 32; ++j) v[b * 32 + j] = d * (float) (int8_t) p[b * 34 + 2 + j];
    }
    return v;
}
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1);   // round to nearest even
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t to_f16(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float from_f16(uint16_t u) {
    __half_raw r;
    r.x = u;
    return __half2float(__half(r));
}
// y[h][t][o] = sum_i w[h][o][i] * x[t][h][i], in double.
std::vector<float> host_ref(const std::vector<float> & w, const std::vector<float> & x, int n_in, int n_out, int heads,
                            int T) {
    std::vector<float> y((size_t) heads * T * n_out);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < n_out; ++o) {
                const float * wr = &w[((size_t) h * n_out + o) * n_in];
                const float * xr = &x[((size_t) t * heads + h) * n_in];
                double acc = 0;
                for (int i = 0; i < n_in; ++i) acc += (double) wr[i] * xr[i];
                y[((size_t) h * T + t) * n_out + o] = (float) acc;
            }
    return y;
}

struct Gate {
    int failures = 0;
    void check(const Stats & s, double min_cos, double max_rel_rms, const std::string & label) {
        // max_abs relative to the reference RMS: robust to near-zero references, catches any outlier.
        const bool ok = s.cosine >= min_cos && s.max_abs <= max_rel_rms * s.ref_rms && s.max_abs == s.max_abs;
        std::printf("  %s %-30s cos=%.9f max_abs=%.3e (%.2e x rms) max_rel=%.2e\n", ok ? "ok  " : "FAIL",
                    label.c_str(), s.cosine, s.max_abs, s.max_abs / s.ref_rms, s.max_rel);
        failures += !ok;
    }
};

float time_ms(cudaStream_t st, int iters, const std::function<void()> & f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    CK(cudaEventRecord(a, st));
    for (int i = 0; i < iters; ++i) f();
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string odir = "/mnt/models/glm-oracle/q4kxl-p32";
    std::string gguf = "/mnt/models-cold/glm53-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";
    bool bench = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--oracle" && i + 1 < argc) odir = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--bench") bench = true;
        else if (a == "--case" && i + 1 < argc) only = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    Oracle oracle(odir);
    GgufSet model(gguf);
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    matmul_init();
    Gate gate;

    for (const Case & c : kCases) {
        if (!only.empty() && std::string(c.label).find(only) == std::string::npos) continue;
        const HostTensor ht = model.tensor(c.weight);
        const int n_in = (int) ht.shape[0], n_out = (int) ht.shape[1], heads = c.heads;
        // q_a's input is attn_norm-3, which has no unique seq in this list: find it by name.
        const OracleNode & xn = c.x_seq >= 0 ? oracle.seq(c.x_seq) : oracle.node(0, "attn_norm-3");
        const std::vector<float> x0 = oracle.load(xn), y0 = oracle.load(oracle.seq(c.y_seq));
        const size_t tok = (size_t) n_in * heads;
        const int T0 = (int) (x0.size() / tok);
        if ((size_t) T0 * tok != x0.size() || y0.size() != (size_t) T0 * heads * n_out) {
            std::fprintf(stderr, "%s: oracle shapes do not match the weight\n", c.label);
            return 2;
        }
        std::printf("%s  [%s %dx%dx%d, %.1f MB]  oracle T=%d\n", c.label, ht.type == 8 ? "Q8_0" : "F32", n_in,
                    n_out, heads, ht.nbytes / 1e6, T0);
        Dev raw(ht.nbytes);
        CK(cudaMemcpy(raw.p, ht.data, ht.nbytes, cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());   // a pageable H2D copy may still be in flight on return

        // Variants: the GGUF weight and its Q8_0R repack against the oracle; for small Q8_0 matrices also BF16 and
        // F16 conversions of the dequantized weight against a float64 host reference with the converted values.
        struct Variant {
            std::string name;
            WeightRef w;
            std::shared_ptr<Dev> mem;
            std::vector<float> y_ref;   // [heads][T0][n_out]
            bool oracle_ref;
        };
        std::vector<Variant> vars;
        vars.push_back({ht.type == 8 ? "q8_0" : "f32", ht.ref(raw.p), nullptr, y0, true});
        if (ht.type == 8) {
            const WeightRef w = ht.ref(raw.p);
            auto m = std::make_shared<Dev>(matmul_repack_bytes(w));
            vars.push_back({"q8_0r", matmul_repack(w, m->p, st), m, y0, true});
            CK(cudaStreamSynchronize(st));
        }
        if (ht.type == 8 && (int64_t) n_in * n_out * heads <= 16 << 20) {
            const std::vector<float> wf = dequant_q8_0(ht.data, (size_t) n_in * n_out * heads);
            for (int ty : {30, 1}) {
                std::vector<uint16_t> h(wf.size());
                std::vector<float> wr(wf.size());
                for (size_t i = 0; i < wf.size(); ++i) {
                    h[i] = ty == 30 ? to_bf16(wf[i]) : to_f16(wf[i]);
                    wr[i] = ty == 30 ? from_bf16(h[i]) : from_f16(h[i]);
                }
                auto m = std::make_shared<Dev>(h.size() * 2);
                CK(cudaMemcpy(m->p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
                CK(cudaDeviceSynchronize());
                WeightRef w = ht.ref(m->p);
                w.type = ty;
                w.nbytes = h.size() * 2;
                vars.push_back({ty == 30 ? "bf16" : "f16", w, m, host_ref(wr, x0, n_in, n_out, heads, T0), false});
            }
        }

        std::vector<int> Ts = {1, 2, 3, 4, 5, 8, 9, 16, 30, 64, 512};
        if (n_out > 100000) Ts = {1};   // the head: the oracle has the last token only
        for (const Variant & v : vars) {
            const WeightRef & w = v.w;
            for (int T : Ts) {
                const std::vector<float> xs = tile_x(x0, T0, T, tok);
                const std::vector<float> ys = tile_y(v.y_ref, T0, T, heads, n_out);
                Dev xd(xs.size() * 4), yd(ys.size() * 4);
                CK(cudaMemcpyAsync(xd.p, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice, st));
                MatmulStrides s;
                if (heads > 1) {   // [T][heads][n_in] in, [heads][T][n_out] out
                    s.ldx = (int64_t) heads * n_in;
                    s.x_batch_stride = n_in;
                }
                const size_t sb = matmul_scratch_bytes(w, T);
                Dev scr(sb);
                glm_matmul(w, xd.as<float>(), yd.as<float>(), T, scr.p, sb, st, s);
                std::vector<float> out(ys.size());
                CK(cudaMemcpyAsync(out.data(), yd.p, out.size() * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                const bool gemv = T <= kMatmulGemvMaxT;
                // Oracle, GEMV: the same Q8 activations as the CPU oracle, only the summation order differs.
                // Oracle, GEMM: F16 activations/weights instead of the oracle's Q8 activations.
                // Host reference (BF16/F16 weights): GEMV is F32 end to end; GEMM rounds x (and BF16 w) to F16.
                const double min_cos = gemv ? (v.oracle_ref ? 0.999999 : 0.9999999) : 0.99995;
                const double max_rms = gemv ? (v.oracle_ref ? 2e-3 : 1e-4) : 1e-1;
                const std::string label = v.name + " T=" + std::to_string(T) + (gemv ? " gemv" : " gemm");
                gate.check(compare(out.data(), ys.data(), out.size()), min_cos, max_rms, label);

                if (T == 512 && w.type != 0) {   // sliced dequant (a third of the matrix per slice) and beta = 1
                    const size_t small = matmul_prepare_bytes(n_in, T, heads) +
                                         (size_t) std::max(1, n_out * heads / 3) * n_in * 2;
                    Dev scr2(small);
                    CK(cudaMemcpyAsync(yd.p, ys.data(), ys.size() * 4, cudaMemcpyHostToDevice, st));
                    s.beta = 1.0f;
                    glm_matmul(w, xd.as<float>(), yd.as<float>(), T, scr2.p, small, st, s);
                    s.beta = 0.0f;
                    CK(cudaMemcpyAsync(out.data(), yd.p, out.size() * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    std::vector<float> twice(ys);
                    for (auto & e : twice) e *= 2;
                    gate.check(compare(out.data(), twice.data(), out.size()), 0.99995, 1e-1,
                               v.name + " T=512 sliced+beta");
                }
                if (bench && (T == 1 || T == 2 || T == 4 || T == 8 || T == 16 || T == 30 || T == 64 || T == 512)) {
                    const int iters = T >= 512 ? 20 : 200;
                    const float ms = time_ms(st, iters, [&] {
                        glm_matmul(w, xd.as<float>(), yd.as<float>(), T, scr.p, sb, st, s);
                    });
                    const double gbs = w.nbytes / (ms * 1e-3) / 1e9;
                    const double tflops = 2.0 * n_in * n_out * heads * T / (ms * 1e-3) / 1e12;
                    std::printf("  bench %-5s T=%-4d %8.1f us  %7.1f GB/s  %6.2f TFLOP/s\n", v.name.c_str(), T,
                                ms * 1e3, gbs, tflops);
                }
            }
            if (bench && v.oracle_ref && w.type != 0 && heads == 1 && n_in == 4096 && n_out >= 8192) {   // prefill
                for (int T : {1024, 4096}) {
                    Dev xd((size_t) T * n_in * 4), yd((size_t) T * n_out * 4);
                    CK(cudaMemsetAsync(xd.p, 0, (size_t) T * n_in * 4, st));
                    const size_t sb = matmul_scratch_bytes(w, T);
                    Dev scr(sb);
                    const float ms = time_ms(st, 5, [&] {
                        glm_matmul(w, xd.as<float>(), yd.as<float>(), T, scr.p, sb, st);
                    });
                    std::printf("  bench %-5s T=%-4d %8.1f us  %6.2f TFLOP/s\n", v.name.c_str(), T, ms * 1e3,
                                2.0 * n_in * n_out * T / (ms * 1e-3) / 1e12);
                }
            }
        }
    }
    // matmul_run_multi: KDA q/k/v of block 0 (Q8_0R) in one launch vs the oracle, and timing vs three launches.
    if (only.empty() || only == "multi") {
        const char * names[3] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"};
        const int yseq[3] = {26, 34, 42};
        const std::vector<float> x0 = oracle.load(oracle.seq(15));
        std::vector<std::shared_ptr<Dev>> mem;
        WeightRef ws[3];
        for (int i = 0; i < 3; ++i) {
            const HostTensor ht = model.tensor(names[i]);
            auto raw = std::make_shared<Dev>(ht.nbytes);
            CK(cudaMemcpy(raw->p, ht.data, ht.nbytes, cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());
            auto rp = std::make_shared<Dev>(matmul_repack_bytes(ht.ref(raw->p)));
            ws[i] = matmul_repack(ht.ref(raw->p), rp->p, st);
            mem.push_back(rp);
            CK(cudaStreamSynchronize(st));
        }
        std::printf("multi: KDA q/k/v blk0 Q8_0R in one launch\n");
        for (int T : {1, 2, 4, 8, 30, 64}) {
            const std::vector<float> xs = tile_x(x0, 30, T, 4096);
            Dev xd(xs.size() * 4), yd((size_t) 3 * T * 8192 * 4);
            CK(cudaMemcpyAsync(xd.p, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice, st));
            const size_t pb = matmul_prepare_bytes(4096, T, 1), rb = matmul_run_bytes(ws[0], T);
            Dev scr(pb + rb);
            float * ys[3];
            for (int i = 0; i < 3; ++i) ys[i] = yd.as<float>() + (size_t) i * T * 8192;
            const MatmulInput in = matmul_prepare(xd.as<float>(), 4096, T, 1, 0, 0, scr.p, st);
            matmul_run_multi(ws, ys, nullptr, 3, in, static_cast<char *>(scr.p) + pb, rb, st);
            std::vector<float> out((size_t) 3 * T * 8192);
            CK(cudaMemcpyAsync(out.data(), yd.p, out.size() * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            const bool gemv = T <= kMatmulGemvMaxT;
            for (int i = 0; i < 3; ++i) {
                const std::vector<float> ys_ref = tile_y(oracle.load(oracle.seq(yseq[i])), 30, T, 1, 8192);
                gate.check(compare(&out[(size_t) i * T * 8192], ys_ref.data(), ys_ref.size()), gemv ? 0.999999 : 0.99995,
                           gemv ? 2e-3 : 1e-1, std::string("multi ") + "qkv"[i] + " T=" + std::to_string(T));
            }
            if (bench && gemv) {
                const float m1 = time_ms(st, 200, [&] {
                    matmul_run_multi(ws, ys, nullptr, 3, in, nullptr, 0, st);
                });
                const float m3 = time_ms(st, 200, [&] {
                    for (int i = 0; i < 3; ++i) matmul_run(ws[i], in, ys[i], nullptr, 0, st);
                });
                const double bytes = 3.0 * ws[0].nbytes;
                std::printf("  bench multi T=%-3d one launch %6.1f us (%5.1f GB/s)  three %6.1f us (%5.1f GB/s)\n", T,
                            m1 * 1e3, bytes / (m1 * 1e-3) / 1e9, m3 * 1e3, bytes / (m3 * 1e-3) / 1e9);
            }
        }
    }
    std::printf("%s (%d failure%s)\n", gate.failures ? "FAILED" : "PASSED", gate.failures,
                gate.failures == 1 ? "" : "s");
    return gate.failures ? 1 : 0;
}
