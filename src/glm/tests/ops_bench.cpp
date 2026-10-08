// glm_ops_bench: per-call GPU time of the shared trunk ops on synthetic weights (no model load), T = 1 by default.
//
//   glm_ops_bench [--T N] [--iters N] [--gemv] [--check] [--dump FILE]   (--gemv: Q8_0R GEMV shapes of the trunk, GB/s;
//                 GLM_MM_TPR / GLM_MM_R override the plan. --check: hc_init / hc_post at T (the token-tiled mix
//                 from T = 64) against T calls of one token: x_out and the mix partials bit-identical; hc_pre (hcp, y,
//                 its Q8) and hc_comb likewise. --dump FILE: the raw outputs of hc_init, hc_pre and hc_comb at T, to
//                 compare two builds byte for byte)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "strata/glm/matmul.hpp"
#include "strata/glm/ops.hpp"

using namespace strata::glm;

namespace {
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class V>
void * up(const V & v) {
    void * d = nullptr;
    ck(cudaMalloc(&d, v.size() * sizeof(v[0])), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice), "upload");
    return d;
}
WeightRef f32_ref(void * d, int64_t n) {
    WeightRef w;
    w.data = d;
    w.type = 0;
    w.ne[0] = n;
    w.nbytes = n * 4;
    return w;
}
// q8_block.cuh's q8_scales_offset (device header): the scales of a Q8 prepared input of C rows of n values
size_t q8_scales_offset(size_t C, int n) { return (C * (size_t) n + 255) & ~size_t(255); }
}  // namespace

int main(int argc, char ** argv) {
    int T = 1, iters = 2000;
    bool gemv = false, check = false;
    std::string dump;   // --dump FILE: hc_init, hc_pre (with Q8), hc_comb outputs at T, raw (bitwise A/B of builds)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--T" && i + 1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--gemv") gemv = true;
        else if (a == "--check") check = true;
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    Geometry g;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int D = g.n_embd, H = g.hc, M = g.hc_mix, N = H * D;

    // hc_fn as Q8_0R [24][16384]: int8 quants then fp16 scales at align256
    const size_t qbytes = (size_t) M * N, sc_off = (qbytes + 255) & ~size_t(255);
    std::vector<uint8_t> fn(sc_off + (size_t) M * (N / 32) * 2);
    for (size_t i = 0; i < qbytes; ++i) fn[i] = (uint8_t) (int8_t) (rng() % 255 - 127);
    for (size_t i = 0; i < (size_t) M * (N / 32); ++i) {
        const __half h = __float2half(1e-4f);
        std::memcpy(&fn[sc_off + i * 2], &h, 2);
    }
    HcBranch br;
    br.fn.data = up(fn);
    br.fn.type = kTypeQ8_0R;
    br.fn.ne[0] = N;
    br.fn.ne[1] = M;
    br.fn.nbytes = fn.size();
    std::vector<float> base(M), scale = {0.5f, 0.5f, 0.5f}, nw(D, 1.0f), bias(g.n_expert, 0.01f);
    for (auto & v : base) v = nd(rng) * 0.1f;
    br.base = f32_ref(up(base), M);
    br.scale = f32_ref(up(scale), 3);
    const WeightRef norm = f32_ref(up(nw), D);
    const WeightRef eb = f32_ref(up(bias), g.n_expert);

    std::vector<float> xs((size_t) T * N), br_in((size_t) T * D), logits((size_t) T * g.n_expert);
    for (auto & v : xs) v = nd(rng);
    for (auto & v : br_in) v = nd(rng);
    for (auto & v : logits) v = nd(rng);
    float * x = (float *) up(xs);
    float * x2 = (float *) up(xs);
    float * b = (float *) up(br_in);
    float * lg = (float *) up(logits);
    float *part, *hcp, *y, *h;
    int32_t * ids;
    float * w;
    ck(cudaMalloc(&part, hc_part_floats(T) * 4), "part");
    ck(cudaMalloc(&hcp, (size_t) T * M * 4), "hcp");
    ck(cudaMalloc(&y, (size_t) T * D * 4), "y");
    ck(cudaMalloc(&h, (size_t) T * 2048 * 4), "h");
    ck(cudaMalloc(&ids, (size_t) T * 8 * 4), "ids");
    ck(cudaMalloc(&w, (size_t) T * 8 * 4), "w");

    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    hc_init(g, b, x, T, br, part, st);
    hc_pre(g, br, x, part, T, norm, hcp, y, st);
    auto bench = [&](const char * name, const std::function<void()> & f) {
        // graph of `iters` calls: measures GPU time without host launch overhead
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
        for (int i = 0; i < iters; ++i) f();
        ck(cudaStreamEndCapture(st, &gr), "end");
        ck(cudaGraphInstantiate(&ge, gr, 0), "inst");
        ck(cudaGraphLaunch(ge, st), "warm");
        cudaEventRecord(e0, st);
        ck(cudaGraphLaunch(ge, st), "run");
        cudaEventRecord(e1, st);
        ck(cudaStreamSynchronize(st), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        std::printf("  %-34s %7.2f us/call\n", name, ms * 1000.0 / iters);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(gr);
    };
    if (gemv) {
        // Q8_0 random rows -> Q8_0R; x prepared once; time = one matmul_run
        struct S { const char * name; int n_in, n_out, batch; };
        const S shapes[] = {{"kda q/k/v 4096->8192", 4096, 8192, 1}, {"kda out 8192->4096", 8192, 4096, 1},
                            {"shexp down 2048->4096", 2048, 4096, 1}, {"dense down 12288->4096", 12288, 4096, 1},
                            {"mla out 16384->4096", 16384, 4096, 1}, {"mla Wv_b [64] 512->256", 512, 256, 64},
                            {"mla Wk_b [64] 256->512", 256, 512, 64}, {"router f32? q8 4096->288", 4096, 288, 1},
                            {"head 4096->154880", 4096, 154880, 1}};
        std::printf("glm_ops_bench --gemv T=%d\n", T);
        for (const S & sh : shapes) {
            const int64_t rows = (int64_t) sh.n_out * sh.batch;
            std::vector<uint8_t> q((size_t) rows * (sh.n_in / 32) * 34);
            for (size_t i = 0; i < q.size(); i += 34) {
                const __half h = __float2half(1e-3f);
                std::memcpy(&q[i], &h, 2);
                for (int k = 2; k < 34; ++k) q[i + k] = (uint8_t) (rng() & 0xff);
            }
            WeightRef w8;
            w8.data = up(q);
            w8.type = 8;
            w8.ne[0] = sh.n_in;
            w8.ne[1] = sh.n_out;
            w8.ne[2] = sh.batch;
            w8.nbytes = q.size();
            void * rp = nullptr;
            ck(cudaMalloc(&rp, matmul_repack_bytes(w8)), "repack");
            const WeightRef wr = matmul_repack(w8, rp, st);
            ck(cudaStreamSynchronize(st), "repack");
            cudaFree(const_cast<void *>(w8.data));
            std::vector<float> xv((size_t) T * sh.batch * sh.n_in);
            for (auto & v : xv) v = nd(rng);
            float * xd = (float *) up(xv);
            float * yd = nullptr;
            ck(cudaMalloc(&yd, (size_t) T * rows * 4), "y");
            void * prep = nullptr;
            ck(cudaMalloc(&prep, matmul_prepare_bytes(sh.n_in, T, sh.batch)), "prep");
            const MatmulInput in = matmul_prepare(xd, sh.n_in, T, sh.batch, 0, 0, prep, st);
            char label[64];
            std::snprintf(label, sizeof label, "%s", sh.name);
            const double mb = (double) wr.nbytes / 1e6;
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            const int it = std::max(10, iters / 20);
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
            for (int i = 0; i < it; ++i) matmul_run(wr, in, yd, nullptr, 0, st);
            ck(cudaStreamEndCapture(st, &gr), "end");
            ck(cudaGraphInstantiate(&ge, gr, 0), "inst");
            ck(cudaGraphLaunch(ge, st), "warm");
            cudaEventRecord(e0, st);
            ck(cudaGraphLaunch(ge, st), "run");
            cudaEventRecord(e1, st);
            ck(cudaStreamSynchronize(st), "sync");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000.0 / it;
            std::printf("  %-28s %8.2f MB %8.2f us %7.1f GB/s\n", label, mb, us, mb * 1e-3 / (us * 1e-6));
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(gr);
            cudaFree(rp);
            cudaFree(xd);
            cudaFree(yd);
            cudaFree(prep);
        }
        return 0;
    }
    const size_t q8_bytes = q8_scales_offset(T, D) + (size_t) T * (D / 32) * 4;
    void * yq;
    ck(cudaMalloc(&yq, q8_bytes), "yq");
    auto bytes = [&](const void * dp, size_t n) {   // after st's work (cudaMemcpy does not wait for a non-blocking stream)
        ck(cudaStreamSynchronize(st), "bytes");
        std::vector<uint8_t> v(n);
        ck(cudaMemcpy(v.data(), dp, n, cudaMemcpyDeviceToHost), "bytes");
        return v;
    };
    if (!dump.empty()) {
        hc_init(g, b, x2, T, br, part, st);
        hc_pre(g, br, x2, part, T, norm, hcp, y, st, yq);
        const std::vector<uint8_t> h_pre = bytes(hcp, (size_t) T * M * 4);
        hc_comb(g, hcp, T, st);
        ck(cudaStreamSynchronize(st), "dump");
        FILE * f = std::fopen(dump.c_str(), "wb");
        for (const auto & v : {bytes(part, hc_part_floats(T) * 4), h_pre, bytes(hcp, (size_t) T * M * 4),
                               bytes(y, (size_t) T * D * 4), bytes(yq, q8_bytes)})
            std::fwrite(v.data(), 1, v.size(), f);
        std::fclose(f);
        std::printf("dump T=%d -> %s\n", T, dump.c_str());
        return 0;
    }
    if (check) {
        auto host = [&](const float * dp, size_t n) {
            std::vector<float> v(n);
            ck(cudaMemcpy(v.data(), dp, n * 4, cudaMemcpyDeviceToHost), "x");
            return v;
        };
        float *xa, *xb, *pb;
        ck(cudaMalloc(&xa, (size_t) T * N * 4), "xa");
        ck(cudaMalloc(&xb, (size_t) T * N * 4), "xb");
        ck(cudaMalloc(&pb, hc_part_floats(T) * 4), "pb");
        bool ok = true;
        for (int mode = 0; mode < 2; ++mode) {
            if (mode == 0) {
                hc_init(g, b, xa, T, br, part, st);
                for (int t = 0; t < T; ++t)
                    hc_init(g, b + (size_t) t * D, xb + (size_t) t * N, 1, br, pb + hc_part_floats(t), st);
            } else {
                hc_post(g, b, b, hcp, x, xa, T, &br, part, st);
                for (int t = 0; t < T; ++t)
                    hc_post(g, b + (size_t) t * D, b + (size_t) t * D, hcp + (size_t) t * M, x + (size_t) t * N,
                            xb + (size_t) t * N, 1, &br, pb + hc_part_floats(t), st);
            }
            ck(cudaStreamSynchronize(st), "check");
            const std::vector<float> va = host(xa, (size_t) T * N), vb = host(xb, (size_t) T * N);
            const bool same = std::memcmp(va.data(), vb.data(), va.size() * 4) == 0;
            std::vector<float> pa(hc_part_floats(T)), pbh(hc_part_floats(T));
            ck(cudaMemcpy(pa.data(), part, pa.size() * 4, cudaMemcpyDeviceToHost), "part");
            ck(cudaMemcpy(pbh.data(), pb, pbh.size() * 4, cudaMemcpyDeviceToHost), "pb");
            const bool close = std::memcmp(pa.data(), pbh.data(), pa.size() * 4) == 0;
            std::printf("check %s T=%d: x_out %s, mix partials %s\n", mode ? "hc_post" : "hc_init", T,
                        same ? "identical" : "DIFFERENT", close ? "identical" : "DIFFERENT");
            ok = ok && same && close;
        }
        {   // hc_pre (with the Q8 output) and hc_comb: T rows in one call vs one call per row
            float *hb, *yb;
            void * yqb;
            ck(cudaMalloc(&hb, (size_t) T * M * 4), "hb");
            ck(cudaMalloc(&yb, (size_t) T * D * 4), "yb");
            ck(cudaMalloc(&yqb, q8_bytes), "yqb");
            hc_init(g, b, xa, T, br, part, st);
            hc_pre(g, br, xa, part, T, norm, hcp, y, st, yq);
            const size_t qs1 = q8_scales_offset(1, D);
            std::vector<uint8_t> q1(qs1 + (D / 32) * 4);
            std::vector<uint8_t> qa = bytes(yq, q8_bytes), qb(q8_bytes);
            for (int t = 0; t < T; ++t) {
                hc_pre(g, br, xa + (size_t) t * N, part + hc_part_floats(t), 1, norm, hb + (size_t) t * M,
                       yb + (size_t) t * D, st, yqb);
                ck(cudaStreamSynchronize(st), "check");
                q1 = bytes(yqb, q1.size());   // one-row layout -> the T-row layout (qs rows, then the scale rows)
                std::memcpy(qb.data() + (size_t) t * D, q1.data(), D);
                std::memcpy(qb.data() + q8_scales_offset(T, D) + (size_t) t * (D / 32) * 4, q1.data() + qs1,
                            (D / 32) * 4);
            }
            ck(cudaStreamSynchronize(st), "check");
            const bool sp = bytes(hcp, (size_t) T * M * 4) == bytes(hb, (size_t) T * M * 4);
            const bool sy = bytes(y, (size_t) T * D * 4) == bytes(yb, (size_t) T * D * 4);
            const bool sq = qa == qb;
            hc_comb(g, hcp, T, st);
            for (int t = 0; t < T; ++t) hc_comb(g, hb + (size_t) t * M, 1, st);
            ck(cudaStreamSynchronize(st), "check");
            const bool sc = bytes(hcp, (size_t) T * M * 4) == bytes(hb, (size_t) T * M * 4);
            std::printf("check hc_pre T=%d: hcp %s, y %s, y Q8 %s; hc_comb %s\n", T, sp ? "identical" : "DIFFERENT",
                        sy ? "identical" : "DIFFERENT", sq ? "identical" : "DIFFERENT", sc ? "identical" : "DIFFERENT");
            ok = ok && sp && sy && sq && sc;
        }
        return ok ? 0 : 1;
    }
    std::printf("glm_ops_bench T=%d (graph of %d calls each)\n", T, iters);
    bench("hc_post + next mix (Q8_0R)", [&] { hc_post(g, b, nullptr, hcp, x, x2, T, &br, part, st); });
    bench("hc_post, no next", [&] { hc_post(g, b, nullptr, hcp, x, x2, T, nullptr, part, st); });
    bench("hc_pre", [&] { hc_pre(g, br, x, part, T, norm, hcp, y, st); });
    bench("hc_comb (Sinkhorn x20)", [&] { hc_comb(g, hcp, T, st); });
    bench("hc_head", [&] { hc_head(g, x, T, norm, y, st); });
    bench("moe_route", [&] { moe_route(g, lg, eb, T, ids, w, st); });
    bench("swiglu_clamp F=2048", [&] { swiglu_clamp(b, b, h, T, 2048, 2048, 2048, 10.f, st); });
    bench("rms_norm 4096", [&] { rms_norm(b, nullptr, norm, y, nullptr, T, D, 0, 1e-5f, st); });
    bench("argmax 154880 (on x)", [&] { argmax_rows(x, 16384, T, ids, st); });
    return 0;
}
