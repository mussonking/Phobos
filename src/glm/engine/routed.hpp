#pragma once
// Routed experts of the GLM engine on top of pack's kernels (strata/glm/experts.hpp): a VRAM cache of expert blobs
// (gate | up | down, IQ4_XS / IQ4_XS / IQ4_NL) runs the hits with experts_gpu, and a host thread pool runs the
// misses on the host copy with pack's IQ4 row kernel, split by rows over all threads so one layer's misses finish
// in about one expert's worth of DRAM time. Interim split-graph form: routed_run is called between graph segments
// (the hits are enqueued, the host waits only for the routing download, computes the misses while the GPU runs the
// hits, then enqueues the upload + add). The doorbell form (one token graph, mapped flags) reuses the same pieces.
//
// Expert source: the model's mmapped GGUF experts (IQ4 GGUF, e.g. pack's glm53-strata-q4), or a synthetic pool
// of random IQ4 blobs (timing only: the real router ids are mapped onto the pool, outputs are meaningless).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "model.hpp"
#include "strata/glm/experts.hpp"

namespace strata::glm {

struct RoutedOptions {
    size_t vram_bytes = 0;                // expert cache budget; 0 = free VRAM minus reserve_bytes
    size_t reserve_bytes = (size_t) 1536 << 20;
    // > 0: exactly this many VRAM slots (overrides vram_bytes; fails if they don't fit past reserve_bytes). The
    // free-VRAM default varies by a few slots between processes, and the experts at the edge of the profile then flip
    // between GPU hit and CPU miss (kernels not bitwise equal): a fixed count makes runs comparable.
    int    slots = 0;
    int    cpu_threads = 32;              // miss pool size (the calling thread also works)
    size_t synthetic_pool_bytes = 0;      // > 0: synthetic blobs instead of the model's experts
    // Host arena (GGUF source): routed_load copies the non-resident experts out of the GGUF into anonymous memory,
    // each 32-row chunk on the NUMA node whose workers read it, with direct (uncached) sequential reads.
    // arena_max_bytes caps it (0 = every non-resident expert); past the cap, misses read the GGUF mapping (exact,
    // HDD-bound; counted in RoutedStats::spilled), or with arena_alias an arena blob (timing only, outputs wrong;
    // RoutedStats::aliased).
    bool   arena = false;
    size_t arena_max_bytes = 0;
    bool   arena_alias = false;
    // The arena is page-locked (cudaHostRegister) and also keeps a copy of the VRAM experts, so it can feed
    // uploads (prefill streaming, routed_reclaim). Implied by adapt_every > 0 at routed_load.
    bool   pin_arena = false;
    // Misses through pack's VNNI row kernels (int8 activations: y Q8_K for gate/up IQ4_XS, the SwiGLU output Q8_K
    // for down IQ4_NL, every token of an expert in one pass over its rows; ZMM 7d1b2a2: misses 27.6 vs 43.3 ms/step
    // F32) instead of the exact F32 one. Falls back to F32 on a CPU without AVX512 VNNI.
    bool   vnni = true;
    // VNNI down with the SwiGLU output in Q8_0 (ggml-cpu's own activation format for IQ4_NL), one token at a time;
    // false = Q8_K (one scale per 256) with the multi-token kernel: 1.3x faster at 1 token, 2.3-2.5x at 2-4 (verify
    // rows), 6-8 ms/token less of misses in decode, but farther from ggml (measured:
    // generated-region KL worse on 5/8 holdouts by 17-71 %, equal on 3, and farther than F32 on 7/8, where Q8_0 down
    // is a 4/4 split with F32). Quality first: on until a multi-token Q8_0 down kernel closes the speed gap.
    bool   down_q8_0 = true;
    // R8 arena (experts/r8.hpp and its kernels): routed_load repacks every arena chunk into 8-row
    // interleaved groups (same bytes, same 32-row NUMA chunks) and the VNNI misses run the R8 row kernels on them
    // (multi-token for gate / up and for the Q8_0 down: no T-fold cost at verify). The arena's host refs carry
    // ExpertRef::layout = 1; every upload converts back to GGUF on the GPU (VRAM slots stay GGUF); the exact F32
    // path and other GGUF readers unpack. Misses' sums differ from the GGUF kernels by float reassociation only.
    bool   r8 = false;
    // Diagnostic only (glm_eval --r8-gguf-down): R8 gate / up kernels, but the down rows of R8 arena experts are
    // unpacked per chunk and run through the GGUF down kernels (isolates the down kernel's reduction order).
    bool   r8_gguf_down = false;
    int    max_T = 1;
    bool   verbose = true;
    // Adaptive VRAM tier (Strata's generate.cpp adapt): every adapt_every routed_step calls, up to adapt_swaps of the
    // most-routed non-resident experts (decayed usage >= 2) replace the least-routed resident ones (usage at least
    // 1.5 lower). The victim turns into a miss at once; the newcomer is uploaded on a side stream (from the pinned
    // arena) and becomes a hit at a later routed_step, once its copy has landed. 0 = static residency. The usage
    // decays by adapt_decay at each adapt step.
    int    adapt_every = 0;
    int    adapt_swaps = 96;
    float  adapt_decay = 0.7f;
    // Called by routed_finish once the routing of block il has reached the host: ids/w [T][n_expert_used].
    std::function<void(int il, const int32_t * ids, const float * w, int T)> on_route;
};

struct RoutedStats {
    int64_t calls = 0, jobs = 0, hits = 0, misses = 0;
    int64_t aliased = 0;   // misses served by an aliased arena blob (capped arena, arena_alias)
    int64_t spilled = 0;   // misses past the arena cap read from the GGUF mapping
    int64_t swaps = 0;     // adaptive tier uploads
    int64_t distinct = 0, distinct_misses = 0;   // distinct experts per call, summed (vs jobs = T x K per call)
    double  host_ms = 0;   // hook wall time (routing download wait + misses + enqueue)
    double  miss_ms = 0;   // miss compute only
    double  wait_ms = 0;   // routing download wait only (the GPU still running the layer)
};

struct Routed;
class Pool;
void routed_destroy(Routed *);
struct RoutedDeleter { void operator()(Routed * r) const { routed_destroy(r); } };
using RoutedPtr = std::unique_ptr<Routed, RoutedDeleter>;

RoutedPtr routed_create(const Model & m, const RoutedOptions & opt);

// Blob per expert (gate | up | down) and the VRAM slot count the options allow.
size_t routed_blob_bytes(const Geometry & g);
int    routed_slots(const Routed &);
// Arena mode: the arena is page-locked (cudaHostRegister / cudaHostAlloc; WSL2 caps page-locked memory at 1 GiB per
// process). Without it the uploads (adaptive tier, prefill loan, reclaims) go through a pinned bounce ring.
bool   routed_arena_pinned(const Routed &);
// routed_lend can run: an arena keeping every expert (pin_arena), pinned or with the bounce ring.
bool   routed_can_lend(const Routed &);
// Page-locked host bytes this Routed holds besides a pinned arena (bounce ring, routing / accumulation buffers).
size_t routed_pinned_bytes(const Routed &);
// The options in effect: routed_create turns vnni off on a CPU without AVX512 VNNI and r8 off without its needs.
const RoutedOptions & routed_options(const Routed &);

// Fills the cache with the first slots of `ranked` (layer, expert) pairs (pairs of dense layers are skipped);
// replaces the current content. Synchronous.
void routed_fill(Routed &, const std::vector<std::pair<int, int>> & ranked, cudaStream_t st);
// Arena mode: the first slots of `ranked` go to VRAM, the next ones (up to the cap) to the host arena, in one
// sequential pass over the GGUF expert tensors. Replaces the cache and the arena content. Before the first load,
// misses read the mmapped GGUF. Synchronous.
void routed_load(Routed &, const std::vector<std::pair<int, int>> & ranked, cudaStream_t st);
// STRP v1 profile (tools/glm/make_profile.py): ranked (layer, expert) pairs.
std::vector<std::pair<int, int>> routed_read_profile(const std::string & path);
// Writes `ranked` as an STRP v1 profile (the format make_profile.py writes and reads).
void routed_write_profile(const std::string & path, const std::vector<std::pair<int, int>> & ranked, int n_layers,
                          int n_expert);

// The experts hook: out[T][4096] = sum_k w * expert_k(y) (overwrites out). ids/w/y are device; not capturable
// (the host waits for the routing download).
void routed_run(Routed &, int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                cudaStream_t st);
// The same in two halves, so the GPU runs the shared expert while the host runs the misses: routed_start enqueues
// the routing download and the hits (capturable); routed_finish waits for the download, runs the misses and
// enqueues their upload + add (host side). One start per finish, same il / T / out.
void routed_start(Routed &, int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                  cudaStream_t st);
void routed_finish(Routed &, int il, float * out, int T, cudaStream_t st);

// Per (layer, expert) selection counts since creation / the last reset (for profiles), and timing stats.
const std::vector<int64_t> & routed_counts(const Routed &);
RoutedStats & routed_stats(Routed &);
// Switches the miss kernel between VNNI and exact F32 (VNNI needs a Routed created with opt.vnni on a VNNI CPU).
void routed_set_vnni(Routed &, bool vnni);
// Diagnostic reference (glm_eval --kernels f64): with vnni off, the misses run in double (exact F32 weights, double
// activations, mids and sums; iq4_cpu_row_f64), order-neutral at the F32 level. The VRAM hits are unchanged.
void routed_set_f64(Routed &, bool f64);
// Diagnostic (glm_prefill_test decode1f64p*): the call with misses after `skip` such calls scales its token 0 miss
// sum by (1 + rel), once: a tiny deterministic perturbation, to measure how far the model itself amplifies a
// rounding-size change.
void routed_perturb_once(Routed &, float rel, int skip = 0);
// Between decode steps (the stream idle): admits the landed adaptive uploads and, every adapt_every calls, starts
// the next swaps (adapt_every == 0: no swaps). Also admits the landed routed_reclaim_async uploads, in any mode:
// call it between decode steps after a prefill loan.
void routed_step(Routed &, cudaStream_t st);
// Sets adapt_every / adapt_swaps at run time (0 = static) and clears the usage. The arena keeps a copy of the VRAM
// experts only if the Routed was loaded with adapt_every > 0.
void routed_set_adapt(Routed &, int every, int swaps);
// Device ExpertRef of expert (il, e) if resident in a VRAM slot, else a null one (gate_up == nullptr).
ExpertRef routed_slot_ref(const Routed &, int il, int e);
// Waits for the adaptive and reclaim uploads in flight and admits them (table updates on st): residency then stays as is until
// the next routed_step.
void routed_quiesce(Routed &, cudaStream_t st);
// Debug check (synchronous, ~1 s per 1000 slots): routed_quiesce, then every resident expert's VRAM blob read back and
// compared with its host copy, and the device table compared with the host mirror. Prints the first mismatches;
// returns their count (-1: not checkable, the host copies are R8 or there is no arena).
int routed_verify_slots(Routed &, cudaStream_t st);
// Lends the last ceil(bytes / blob) VRAM slots as one device range (returned): routed_quiesce, then their experts
// are evicted (misses until routed_reclaim) and the adaptive tier no longer uses those slots. Needs pin_arena (or
// adaptive) placement: every expert keeps a host copy. One loan at a time.
void * routed_lend(Routed &, size_t bytes, cudaStream_t st);
// Ends the loan: the evicted experts are uploaded back from the arena into the same slots and become hits again.
// Synchronous. The uploads are enqueued on st: every use of the loaned range must be complete or ordered before
// them on st (work on another stream, including the legacy default stream, must be synchronized first).
void routed_reclaim(Routed &, cudaStream_t st);
// Same, without the wait: the uploads run on a side stream after the work already on st (the borrower must not use
// the range after it), the slots return to the cache at once and their experts stay misses until a routed_step (or
// routed_quiesce / the next routed_lend) admits them once landed. Keeps the re-upload off the next turn's critical
// path (decode overlaps it).
void routed_reclaim_async(Routed &, cudaStream_t st);
// Host copy of expert (il, e): arena blob, synthetic blob or the GGUF mapping (up null: gate_up holds gate | up).
ExpertRef routed_host_expert(const Routed &, int il, int e);
// Enqueues on s the upload of a host expert (routed_host_expert) into a VRAM blob (gate | up | down, GGUF rows):
// the plane copies, then the R8 -> GGUF conversion in place when the host copy is R8. Every upload of expert bytes
// goes through it, so device blobs are always GGUF; publish dst (tables, events) only after it on s.
void routed_upload_expert(const Routed &, const ExpertRef & src, void * dst, cudaStream_t s);
// In-place R8 -> GGUF conversion of an uploaded plane of n_rows rows (multiple of 8): IQ4_XS rows of width 4096
// (xs) or IQ4_NL rows of width 2048 (r8_gpu.cu; layout: experts/r8.hpp).
void r8_plane_to_gguf(void * plane, bool xs, int width, int n_rows, cudaStream_t s);
// The miss thread pool (NUMA-pinned workers; item i of a run belongs to node i % nodes, as the arena's 32-row
// chunks): the prefill runs its CPU expert share on it, between decode steps.
Pool & routed_pool(Routed &);
// Ranked pairs from the counts (descending, ties by layer then expert).
std::vector<std::pair<int, int>> routed_rank_counts(const Routed &);

}  // namespace strata::glm
