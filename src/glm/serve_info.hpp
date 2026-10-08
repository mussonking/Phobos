#pragma once
// glm_serve's INFO line before READY (serve/server.py: GET /metrics "engine", the web app's Monitor and About tabs;
// GET /health repeats arena / slots / kernels / down / sampling / cpu_isa). The keys Strata's own engine prints mean the same
// here: engine = the version, kv = the MLA latent cache (int8 | fp16), expert_slots / expert_cache_mib = the VRAM
// expert cache, spec = the verify window (the last token and spec - 1 MTP drafts; 0 = off), mtp_max = its cap.
// sampling=greedy: every token is the argmax, the GEN line's sampling keys are skipped (temperature, top_p, top_k,
// seed have no effect). cpu_isa = the CPU kernel path actually selected (vnni | avx2 | scalar, after GLM_CPU_ISA
// auto; strata/glm/cpu_isa.hpp): `kernels` names the miss rows' format (vnni | f32), which reads f32 on an AVX2 box
// too, so only cpu_isa tells a machine that fell back to AVX2. Header-only: tests/serve_info_test.cpp checks it without a GPU.
#include <cstddef>
#include <cstdio>
#include <string>

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace strata::glm {

struct ServeInfo {
    int    spec_k = 0;                       // MTP drafts per round
    int    adapt_every = 0, sessions = 0, snapshots = 0, prefill_min = 0;
    size_t snapshot_mb = 0;
    bool   q8 = false;                       // MLA latent int8 (else F16)
    bool   r8 = false, vnni = false, down_q8_0 = false;
    int    slots = 0;                        // VRAM expert slots
    size_t slot_bytes = 0;                   // one expert's blob (routed_blob_bytes)
    const char * cpu_isa = "scalar";         // cpu_isa_name(cpu_isa())
};

inline std::string serve_info_line(const ServeInfo & s) {
    const int spec = s.spec_k > 0 ? s.spec_k + 1 : 0;
    char b[640];
    std::snprintf(b, sizeof b,
                  "INFO engine=" STRATA_VERSION " spec_k=%d adapt_every=%d sessions=%d snapshots=%d snapshot_mb=%zu "
                  "prefill_min=%d kv_latent=%s arena=%s slots=%d kernels=%s down=%s kv=%s expert_slots=%d "
                  "expert_cache_mib=%zu spec=%d mtp_max=%d sampling=greedy cpu_isa=%s",
                  s.spec_k, s.adapt_every, s.sessions, s.snapshots, s.snapshot_mb, s.prefill_min, s.q8 ? "q8" : "f16",
                  s.r8 ? "r8" : "gguf", s.slots, s.vnni ? "vnni" : "f32", !s.vnni ? "f32" : s.down_q8_0 ? "q8_0" : "q8_k",
                  s.q8 ? "int8" : "fp16", s.slots, (size_t) s.slots * s.slot_bytes >> 20, spec, spec, s.cpu_isa);
    return b;
}

}  // namespace strata::glm
