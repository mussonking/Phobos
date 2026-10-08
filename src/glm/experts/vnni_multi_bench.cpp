// glm_vnni_multi_bench: iq4_vnni_rows_multi (one unpack for n activations) against n iq4_vnni_rows calls on random
// IQ4_XS gate rows (width n_embd) and IQ4_NL down rows (width ffn_expert): bitwise check + single-thread timing.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "iq4_vnni.hpp"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

using namespace strata::glm;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    std::mt19937 rng(7);
    int bad = 0;
    for (int type : {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL}) {
        const int width = type == GGML_TYPE_IQ4_XS ? 4096 : 2048, rows = 32, chunks = 64;
        const size_t stride = iq4_vnni_row_bytes(type, width);
        std::vector<uint8_t> w(stride * rows * chunks);
        for (auto & b : w) b = (uint8_t) rng();
        // sane fp16 scales (block d / group d)
        const size_t blk = type == GGML_TYPE_IQ4_XS ? sizeof(block_iq4_xs) : sizeof(block_iq4_nl);
        for (size_t o = 0; o < w.size(); o += blk) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(0.001f * (1 + rng() % 100));
            std::memcpy(&w[o], &h, 2);
        }
        std::vector<std::vector<uint8_t>> q(8, std::vector<uint8_t>(iq4_vnni_q8_bytes(width) + 64));
        const void * qp[8];
        std::normal_distribution<float> nd;
        for (int t = 0; t < 8; ++t) {
            std::vector<float> x(width);
            for (auto & v : x) v = nd(rng);
            void * p = q[t].data();
            while ((uintptr_t) p % 64) p = (uint8_t *) p + 1;
            iq4_vnni_quantize(x.data(), width, p);
            qp[t] = p;
        }
        for (int n : {1, 2, 4, 8}) {
            std::vector<float> a((size_t) n * rows), b((size_t) n * rows);
            for (int c = 0; c < chunks; ++c) {
                iq4_vnni_rows_multi(type, w.data() + c * rows * stride, width, qp, n, rows, a.data());
                for (int t = 0; t < n; ++t) iq4_vnni_rows(type, w.data() + c * rows * stride, width, qp[t], rows, b.data() + t * rows);
                if (std::memcmp(a.data(), b.data(), a.size() * 4)) { ++bad; std::printf("MISMATCH type %d n %d chunk %d\n", type, n, c); break; }
            }
            double best_m = 1e9, best_s = 1e9;
            for (int rep = 0; rep < 7; ++rep) {
                double t0 = now_ms();
                for (int c = 0; c < chunks; ++c) iq4_vnni_rows_multi(type, w.data() + c * rows * stride, width, qp, n, rows, a.data());
                best_m = std::min(best_m, now_ms() - t0);
                t0 = now_ms();
                for (int c = 0; c < chunks; ++c)
                    for (int t = 0; t < n; ++t) iq4_vnni_rows(type, w.data() + c * rows * stride, width, qp[t], rows, b.data() + t * rows);
                best_s = std::min(best_s, now_ms() - t0);
            }
            std::printf("%s width %d, %d rows, n %d: multi %.3f ms, %d x rows %.3f ms (%.2fx)\n",
                        type == GGML_TYPE_IQ4_XS ? "IQ4_XS" : "IQ4_NL", width, rows * chunks, n, best_m, n, best_s, best_s / best_m);
        }
    }
    std::printf("%s\n", bad ? "FAIL" : "PASS (bitwise)");
    return bad ? 1 : 0;
}
