// glm_load_check: load the real GGUF through the engine loader (metadata + tensor set + shapes validated), then
// spot-check the uploaded weights against the oracle: the token embedding of the prompt (host path) and, per layer
// kind, one projection through glm_matmul with the loaded (repacked) weight.
//
//   glm_load_check [--gguf SHARD1] [--oracle DIR] [--no-repack]
#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "model.hpp"
#include "strata/glm/matmul.hpp"
#include "strata/glm/oracle.hpp"

using namespace strata::glm;

int main(int argc, char ** argv) {
    std::string gguf = "/mnt/models-cold/glm53-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf";
    std::string odir = "/mnt/models/glm-oracle/q4kxl-p32";
    LoadOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--oracle" && i + 1 < argc) odir = argv[++i];
        else if (a == "--no-repack") opt.repack_q8 = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    std::unique_ptr<Model> m;
    try {
        m = load_model(gguf, opt, st);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "LOAD FAILED: %s\n", e.what());
        return 1;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("loaded: %zu layers, arena %.2f GB, VRAM free %.2f of %.2f GB\n", m->layers.size(),
                m->arena_bytes / 1e9, free_b / 1e9, total_b / 1e9);
    const LayerWeights & L3 = m->layers[3];
    std::printf("blk.3 experts: gate type %d %zu B/expert, up type %d, down type %d %zu B/expert\n",
                L3.experts.type_gate, L3.experts.gate_bytes, L3.experts.type_up, L3.experts.type_down,
                L3.experts.down_bytes);

    test::Oracle oracle(odir);
    int failures = 0;
    auto check = [&](const char * label, const test::Stats & s, double min_cos) {
        const bool ok = s.cosine >= min_cos;
        std::printf("  %s %-28s cos=%.9f max_abs=%.3e ref_rms=%.3e\n", ok ? "ok  " : "FAIL", label, s.cosine,
                    s.max_abs, s.ref_rms);
        failures += !ok;
    };

    // Embedding of the 30 prompt tokens (inp_embd, phase 0).
    std::vector<int32_t> ids;
    {
        FILE * f = std::fopen((odir + "/tokens.txt").c_str(), "r");
        int v;
        while (f && std::fscanf(f, "%d", &v) == 1) ids.push_back(v);
        if (f) std::fclose(f);
    }
    const int T = 30;
    std::vector<float> emb((size_t) T * m->g.n_embd);
    embed_rows_host(*m, ids.data(), T, emb.data());
    const std::vector<float> e0 = oracle.load(0, "inp_embd");
    check("token_embd (host)", test::compare(emb.data(), e0.data(), e0.size()), 0.9999999);

    // One projection per weight family, through the loaded weights.
    struct P {
        const char * label;
        const WeightRef * w;
        int x_seq, y_seq;
    } projs[] = {
        {"blk0 kda q", &m->layers[0].kda.q, 15, 26},
        {"blk0 kda out", &m->layers[0].kda.out, 103, 104},
        {"blk0 hc_attn_fn", &m->layers[0].hc_attn_fn, 4, 5},
        {"blk0 ffn_down", &m->layers[0].ffn_down, 129, 130},
        {"blk3 mla q_b", &m->layers[3].mla_w.q_b, 455, 456},
        {"blk3 router", &m->layers[3].gate_inp, 546, 547},
        {"blk3 shexp gate", &m->layers[3].gate_shexp, 546, 581},
        {"output head", &m->output, 7727, 7728},
    };
    for (const P & p : projs) {
        const std::vector<float> x = oracle.load(oracle.seq(p.x_seq)), y = oracle.load(oracle.seq(p.y_seq));
        const int Tn = (int) (x.size() / p.w->ne[0]);
        float *xd, *yd;
        cudaMalloc(&xd, x.size() * 4);
        cudaMalloc(&yd, y.size() * 4);
        const size_t sb = matmul_scratch_bytes(*p.w, Tn);
        void * scr;
        cudaMalloc(&scr, sb);
        cudaMemcpyAsync(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice, st);
        glm_matmul(*p.w, xd, yd, Tn, scr, sb, st);
        std::vector<float> out(y.size());
        cudaMemcpyAsync(out.data(), yd, out.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        check(p.label, test::compare(out.data(), y.data(), y.size()), 0.999999);
        cudaFree(xd);
        cudaFree(yd);
        cudaFree(scr);
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
