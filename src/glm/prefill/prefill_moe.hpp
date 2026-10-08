#pragma once
// Routed experts of the prefill (ForwardHooks::experts / experts_finish at large T), hybrid per layer and expert:
//   - resident in the routed VRAM cache        -> GPU (prefill_experts, from the cache slot);
//   - not resident, many routed tokens         -> uploaded from the pinned host arena into a ring of VRAM slots
//                                                 lent by the routed cache, then GPU;
//   - not resident, few routed tokens          -> CPU (prefill_cpu on the routed miss pool), overlapped with the GPU.
// The upload / CPU split balances the two per layer with a cost model (upload_cost_ms per expert against
// cpu_expert_ms + cpu_job_ms per routed token): the most-routed non-resident experts are uploaded until the PCIe time
// would exceed the CPU time of the rest. The CPU side is calibrated online (running ratio of measured to modelled
// time per layer).
//
// start: routing download (one host sync per layer), plan, hands the CPU share to a driver thread (it waits for the
// activation download), enqueues the uploads + GPU experts on a side stream; finish: joins both, uploads and adds the
// CPU rows. ffn_branch enqueues the shared expert between, on the caller's stream (it overlaps the routed experts).
#include <cstdint>

#include <cuda_runtime.h>

#include "forward.hpp"
#include "model.hpp"
#include "routed.hpp"

namespace strata::glm {

struct PrefillMoeOptions {
    int    max_T = 8192;              // largest chunk
    int    group = 16;            // experts dequantized together on the GPU
    int    ring = 48;             // upload slots (>= 2 * group keeps uploads ahead of the dequant)
    // speculative uploads: after each layer, the next layer's most routed non-resident experts (counts over the past
    // prefills; at large T every expert, as nearly all get routed) are uploaded into the free ring while the GPU runs
    // the trunk up to that layer's routing (PCIe idle there). As many as fit spec_window of that trunk window
    // (measured per layer), at most spec and cap / 96 (the ring grows to hold them). History dependent like
    // calibrate: 0 for bit-reproducible prefills. Only in loans of >= 1024 tokens (at 512 the window holds a few
    // uploads and overrunning it delays the layer's own: measured slower).
    int    spec = 128;
    double spec_window = 1.0;
    double upload_ms = 1.108;     // one expert blob over PCIe (13.6 MB at 12.3 GB/s): the speculative uploads' window
    // The split's cost model, fitted on the 30K agentic conversation (R8 host arena, the R8 tiles for experts
    // of <= 16 / 24 tokens, 48 threads, glm_prefill_test --calibrate 0 "fit:" lines): CPU ms per layer = 0.045 per
    // expert + 0.0095 per routed token, within 3 % at 200 / 800 / 2000 new tokens and the 30K prompt (0.086 / 0.007
    // before: +30 % at 200, -21 % at 30K). upload_cost_ms is an uploaded expert's share of the GPU side (PCIe plus
    // its dequant and GEMMs competing with the trunk), swept 1.108 / 1.3 / 1.5 / 1.8 against the wall time: 1.5 is
    // the fastest everywhere (with 0.086 / 0.007 / 1.108: 200 new 1.49 -> 1.14 s, 800 3.13 -> 2.73 s, 2000 5.42 ->
    // 5.15 s, 30K 47.4 -> 47.4 s; uncalibrated, quiet V100 host).
    // Refit for the panel tiles (prefill_cpu from 8 tokens per expert), same least squares on the same 4 cases
    // (it gives back 0.0453 / 0.00939 on the old kernel's runs): 0.058 per expert + 0.0057 per routed token, within
    // 8 %. With the old constants the faster CPU share got the same experts and waited on PCIe (800 new -0.1 s only);
    // refit, calibrated, against the old kernel: 200 new 1.13 -> 1.10 s, 800 2.75 -> 2.30 s, 2000 5.17 -> 4.30 s,
    // 30K 47.8 -> 43.5 s (online calibration alone reaches about the same; the refit makes calibrate 0 match it).
    double upload_cost_ms = 1.5;
    double upload_tail_ms = 3.0;  // GPU work left after the last upload (its dequant group + GEMMs)
    double cpu_expert_ms = 0.058; // CPU fixed cost per expert
    double cpu_job_ms = 0.0057;   // CPU cost per routed token of an expert
    bool   cpu = true;            // false: every non-resident expert is uploaded
    bool   upload = true;         // false: every non-resident expert runs on the CPU
    bool   async_reclaim = true;  // end: routed_reclaim_async (the caller's routed_step admits the experts back)
    // online CPU calibration of the cost model. It moves the upload / CPU split with the measured CPU speed, so the
    // same prompt can take other expert paths (GPU F16 vs CPU int8) from one prefill to the next: results equal up to
    // near-tie flips. false (with async_reclaim false): bit-reproducible prefills and the decode right after.
    bool   calibrate = true;
    // Diagnostic (glm_prefill_test --r8-tiles 0): the CPU share's R8 experts always unpacked to the GGUF tiles.
    bool   r8_tiles = true;
};

struct PrefillMoeStats {
    int64_t calls = 0, jobs = 0, res_jobs = 0, up_jobs = 0, cpu_jobs = 0;
    int64_t res_experts = 0, up_experts = 0, cpu_experts = 0;
    int64_t spec_experts = 0, spec_jobs = 0, spec_wasted = 0;   // speculative uploads used / their jobs / unrouted
    double  window_ms = 0;      // measured trunk windows (previous layer's experts join -> this layer's routing)
    double  plan_wait_ms = 0;   // host wait for the routing download (GPU work ahead of it)
    double  cpu_wait_ms = 0;    // host wait for the activation download
    double  cpu_ms = 0;         // CPU share compute
    double  model_up_ms = 0, model_cpu_ms = 0;   // cost model totals of the chosen splits (CPU calibrated)
    double  lend_ms = 0, reclaim_ms = 0;         // prefill_moe_begin / prefill_moe_end (re-upload of the evicted)
    double  mtp_ms = 0;                          // prefill_prompt: MTP catch-up
};

// One prefill's device memory, for cap = min(n, max_T) tokens (bytes, each 256-aligned).
struct PrefillLoanBytes {
    int    cap = 0;
    int    ring_slots = 0, group = 0, spec_slots = 0;   // sized by cap (the ring holds spec_slots + group)
    size_t ring = 0, experts = 0, workspace = 0, scratch = 0, total = 0;
};

struct PrefillMoe;
// Host side only (streams, cuBLAS, pinned buffers for opt.max_T tokens, CPU calibration): no VRAM until begin.
PrefillMoe * prefill_moe_create(const Model & m, Routed & r, const PrefillMoeOptions & opt, cudaStream_t st);
void prefill_moe_destroy(PrefillMoe * p, cudaStream_t st);   // ends an open prefill first
size_t prefill_moe_device_bytes(const PrefillMoe * p);       // own allocations (the loan excluded)

// The device memory of one prefill of n tokens comes from a single routed_lend loan (pin_arena needed), taken by begin
// and returned by end (opt.async_reclaim: the evicted experts are uploaded back behind the decode and turn into hits
// at a later routed_step; else synchronously), so the routed cache keeps every slot between prefills. The loan holds the upload ring, the expert GEMM buffers, a trunk workspace and scratch_bytes of
// caller scratch (e.g. the attention scratch), all for cap = min(n, max_T) tokens: chunks of at most cap tokens go
// through forward_trunk on prefill_moe_workspace with the experts hooks on start / finish.
PrefillLoanBytes prefill_moe_loan_bytes(const PrefillMoe * p, int64_t n, size_t scratch_bytes);
void prefill_moe_begin(PrefillMoe * p, int64_t n, size_t scratch_bytes, cudaStream_t st);
void prefill_moe_end(PrefillMoe * p, cudaStream_t st);   // no-op without an open prefill
Workspace & prefill_moe_workspace(PrefillMoe * p);         // between begin and end
void * prefill_moe_scratch(PrefillMoe * p);                // the caller scratch (null if scratch_bytes was 0)

void prefill_moe_start(PrefillMoe * p, int il, const float * y, const int32_t * ids, const float * w, float * out,
                       int T, cudaStream_t st);
void prefill_moe_finish(PrefillMoe * p, int il, float * out, int T, cudaStream_t st);

PrefillMoeStats & prefill_moe_stats(PrefillMoe * p);
const PrefillMoeOptions & prefill_moe_options(const PrefillMoe * p);
// cost model / switches (not max_T, group, ring; spec at most the created count)
void prefill_moe_set_options(PrefillMoe * p, const PrefillMoeOptions & opt);

}  // namespace strata::glm
