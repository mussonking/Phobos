#pragma once
// CPU share of one MoE layer's routed experts during a prefill chunk: the experts with few routed tokens are cheaper
// on the host than uploaded (13.6 MB over PCIe each). AVX512 VNNI: the item's weight rows unpacked once to u8, every
// token a register tile of vpdpbusd; activations (x, then the clamped SwiGLU output) int8 with one F32 scale per 32
// values (the weights' group), F32 accumulation.
//
// Work items follow the arena's NUMA layout (32-row chunks of every plane alternate between nodes; pool item i
// belongs to node i % nodes): phase 1 = (expert, 32 gate/up rows) over all the expert's tokens, phase 2 = 32 down
// rows over every expert in a fixed order, so the sums are deterministic. From 8 tokens per expert the item's rows
// are laid out once as panels (lanes = 16 rows, one F32 accumulator per 16 rows x token; R8 row groups decode
// straight to them): see prefill_cpu.cpp.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "strata/glm/experts.hpp"
#include "strata/glm/geometry.hpp"

namespace strata::glm {

class Pool;

// Jobs of one layer grouped by expert (CSR): expert u owns jobs [start[u], start[u + 1]), each one an activation row
// and its router weight. Rows must be ascending inside an expert.
struct CpuExpertJobs {
    std::vector<ExpertRef> refs;   // host IQ4 blobs (up null: gate_up holds gate | up)
    std::vector<int> start{0};
    std::vector<int> row;
    std::vector<float> wt;
    void clear() {
        refs.clear();
        start.assign(1, 0);
        row.clear();
        wt.clear();
    }
    int experts() const { return (int) refs.size(); }
    int jobs() const { return (int) row.size(); }
};

struct PrefillCpu;
PrefillCpu * prefill_cpu_create(const Geometry & g, int max_rows, int max_jobs);
void prefill_cpu_destroy(PrefillCpu * p);

// Quantized activation rows for the next runs: row i = x[rows[i]] (rows null: x[i]), x F32 host [.][n_embd].
void prefill_cpu_quantize(PrefillCpu * p, Pool & pool, const float * x, int n_rows, const int * rows = nullptr);
// acc [n_rows][n_embd] = sum over the jobs of w * expert(x[row]) (overwritten; rows without a job are zero).
void prefill_cpu_run(PrefillCpu * p, Pool & pool, const CpuExpertJobs & jobs, int n_rows, float * acc);

// Test hook: the kernel's dot products of 32 rows (IQ4_XS when xs, else IQ4_NL; GGUF rows, or R8 row groups when r8;
// width % 512 == 0) with n <= 5 activation rows x [n][width]: out [32][n]. q [n][width] (value order) and
// d [n][width / 32] receive the activations as the kernel quantized them, for an exact reference on the same integers.
void prefill_cpu_dot(const uint8_t * rows, bool xs, bool r8, int width, const float * x, int n, float * out,
                     int8_t * q, float * d);

// The same through the panel tiles (many tokens per expert): GGUF rows, or R8 row groups when r8; 1 <= n <= 16.
void prefill_cpu_dot_panels(const uint8_t * rows, bool xs, bool r8, int width, const float * x, int n, float * out,
                            int8_t * q, float * d);
// Microbench, one thread, in cache (32 IQ4_XS rows of width): best of reps, ns per call of what = 0 the GGUF tile
// (n <= 5 tokens), 1 the panel tile (n <= 16), 2 the 32-row unpack, 3 unpack + panels, 4 / 5 the scalar / vector
// SwiGLU of 32 rows x n tokens, 6 R8 row groups straight to panels.
double prefill_cpu_kernel_bench(int what, int width, int n, int reps);
// Test hook: out[i] = the kernel's vector SwiGLU (meant bitwise expert_swiglu_clamp) of gate[i], up[i]; n % 16 == 0.
void prefill_cpu_swiglu(const float * gate, const float * up, float * out, int n);
// Diagnostic: false = no panel tiles (the GGUF / R8 tiles at any token count), for A/B timing.
void prefill_cpu_set_panels(PrefillCpu * p, bool on);

// Diagnostic (glm_prefill_test --r8-tiles 0): R8 experts unpacked to the GGUF tiles at any token count, the CPU
// share's numerics before the R8 tiles (default on).
void prefill_cpu_set_r8_tiles(PrefillCpu * p, bool on);

// Wall times (ms) of the last prefill_cpu_run phases.
struct PrefillCpuTiming { double gate_up = 0, quant = 0, down = 0; };
PrefillCpuTiming prefill_cpu_timing(const PrefillCpu * p);

}  // namespace strata::glm
