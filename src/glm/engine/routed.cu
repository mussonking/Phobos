// src/glm/engine/routed.cu - see routed.hpp.
#include "routed.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bounce.hpp"
#include "iq4_cpu.hpp"
#include "iq4_vnni.hpp"
#include "pinned_budget.hpp"
#include "pool.hpp"
#include "r8.hpp"
#include "strata/glm/cpu_isa.hpp"

namespace strata::glm {
namespace {

constexpr int kIq4Xs = 23, kIq4Nl = 20;
constexpr int kXsBlock = 136, kNlBlock = 18;   // bytes per 256 / 32 weights
constexpr int kRowsChunk = 32;   // rows per miss work item: 17 pages of gate/up, 9 of down

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm routed: ") + what + ": " + cudaGetErrorString(e));
}
[[noreturn]] void fail(const std::string & what) { throw std::runtime_error("glm routed: " + what); }

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Planes {
    size_t gate = 0, up = 0, down = 0;   // bytes per expert
};
Planes planes_of(const Geometry & g) {
    return {(size_t) g.ffn_expert * (g.n_embd / 256) * kXsBlock, (size_t) g.ffn_expert * (g.n_embd / 256) * kXsBlock,
            (size_t) g.n_embd * (g.ffn_expert / 32) * kNlBlock};
}

__global__ void gather_slots_kernel(const ExpertRef * __restrict__ table, const int32_t * __restrict__ ids, int n,
                                    int ne, ExpertRef * __restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int e = ids[j];
    out[j] = e >= 0 && e < ne ? table[e] : ExpertRef{nullptr, nullptr, kIq4Xs, kIq4Nl, nullptr, -1};
}

__global__ void add_kernel(float * __restrict__ out, const float * __restrict__ add, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += add[i];
}

// Uncached reads (O_DIRECT) of byte ranges into an aligned buffer: the experts are read once into the arena, so they
// should not also sit in the page cache.
class DirectReader {
public:
    ~DirectReader() {
        if (fd_ >= 0) ::close(fd_);
        std::free(buf_);
    }
    const uint8_t * read(const std::string & path, uint64_t off, size_t len) {
        if (path != path_) {
            if (fd_ >= 0) ::close(fd_);
            fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd_ < 0) fail("cannot open " + path + " with O_DIRECT");
            path_ = path;
        }
        constexpr uint64_t kAlign = 4096;
        const uint64_t a0 = off & ~(kAlign - 1), a1 = (off + len + kAlign - 1) & ~(kAlign - 1);
        const size_t n = a1 - a0;
        if (n > cap_) {
            std::free(buf_);
            buf_ = nullptr;
            if (posix_memalign(reinterpret_cast<void **>(&buf_), kAlign, n)) fail("direct read buffer");
            cap_ = n;
        }
        size_t done = 0;
        while (done < n) {
            const ssize_t got = pread(fd_, buf_ + done, n - done, (off_t) (a0 + done));
            if (got < 0) fail("read of " + path + " failed");
            if (got == 0) break;   // end of file inside the last aligned block
            done += (size_t) got;
        }
        if (done < off + len - a0) fail("short read of " + path);
        bytes_ += len;
        return buf_ + (off - a0);
    }
    uint64_t bytes() const { return bytes_; }

private:
    std::string path_;
    int fd_ = -1;
    uint8_t * buf_ = nullptr;
    size_t cap_ = 0;
    uint64_t bytes_ = 0;
};

}  // namespace

struct Routed {
    const Model * m = nullptr;
    Geometry g;
    RoutedOptions opt;
    Planes pl;
    size_t blob = 0;
    int n_layers = 0;   // m.layers.size() (trunk + MTP)
    // host source
    uint8_t * synth = nullptr;   // synthetic pool, anonymous mapping (null: the model's mmapped experts)
    size_t synth_blobs = 0, synth_bytes = 0;
    uint8_t * arena = nullptr;   // host arena (routed_load), anonymous mapping, same NUMA layout as the pool
    size_t arena_blobs = 0, arena_bytes = 0;
    std::vector<int> arena_idx;  // [layer][expert] arena blob or -1 (empty until the first load)
    // VRAM cache
    void * slots = nullptr;
    int n_slots = 0;
    ExpertRef * table = nullptr;   // device [n_layers][n_expert]
    std::vector<int> resident;     // host mirror: slot or -1
    ExpertRef * job_slots = nullptr;   // device [max_T * K]
    void * gpu_scratch = nullptr;
    // routing download + miss accumulation (pinned)
    int32_t * h_ids = nullptr;
    float * h_w = nullptr, * h_y = nullptr, * h_acc = nullptr;
    float * h_acc_dev = nullptr;   // h_acc's device alias: the add kernel reads the misses' sums over PCIe directly
    cudaEvent_t ev = nullptr;
    std::unique_ptr<Pool> pool;
    int prewake_us = 0;   // GLM_POOL_PREWAKE_US: routed_finish wakes the pool this long ahead (0: off)
    std::vector<float> mid;   // [K * max_T][ffn_expert]
    // routed_set_f64: y and the mids in double (allocated on first use)
    bool f64 = false;
    std::vector<double> y64, mid64;
    float perturb = 0;     // routed_perturb_once: pending relative scale of a miss sum (token 0)
    int perturb_skip = 0;  // calls with misses to let through first
    // VNNI path: Q8_K of y [max_T], Q8_0 of mid [K * max_T]
    std::vector<uint8_t> q8y, q8mid;
    size_t q8y_row = 0, q8mid_row = 0;
    // R8 arena (opt.r8): arena_r8 once routed_load has repacked it; the R8 kernels' activation side data: y's
    // corrections / scales per token, the Q8_0 mids as x64 blocks (one cache line each, written by the item that
    // quantizes it), the Q8_K mids' corrections / scales
    bool arena_r8 = false;
    std::vector<int64_t> r8_corr_y, r8_corr_mid;
    std::vector<float> r8_d_y, r8_d_mid;
    std::vector<uint8_t> r8_x64;   // 64 B aligned from r8_x64_at
    uint8_t * r8_x64_at = nullptr;
    std::vector<int64_t> counts;
    RoutedStats stats;
    // adaptive tier
    std::vector<float> usage;                          // decayed selections per (layer, expert)
    std::vector<ExpertRef> htable;                     // host mirror of table
    std::vector<std::pair<size_t, int>> pending;       // (key, slot) uploads in flight
    cudaStream_t adapt_st = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    int64_t steps = 0;
    bool arena_pinned = false;
    bool arena_hostalloc = false;   // the arena is a cudaHostAlloc block (GLM_ARENA_PIN), else an anonymous mapping
    // unpinned arena keeping every expert: its uploads go through this pinned ring (routed_upload_expert)
    std::unique_ptr<BounceRing> bounce;
    // routed_lend: slots [n_slots - lent, n_slots) are out of the cache; lent_keys = their evicted experts
    int lent = 0;
    std::vector<std::pair<size_t, int>> lent_keys;
    // routed_reclaim_async: (key, slot) uploads in flight on reclaim_st (misses until admitted)
    std::vector<std::pair<size_t, int>> reclaiming;
    cudaStream_t reclaim_st = nullptr;
    cudaEvent_t reclaim_ev = nullptr;

    ExpertRef host_ref(int il, int e) const {
        ExpertRef r;
        if (synth) {
            const size_t idx = ((size_t) il * g.n_expert + e) * 2654435761u % synth_blobs;
            const uint8_t * b = synth + idx * blob;
            r.gate_up = b;
            r.down = b + pl.gate + pl.up;
            return r;
        }
        const size_t key = (size_t) il * g.n_expert + e;
        if (!arena_idx.empty() && arena_blobs && (arena_idx[key] >= 0 || opt.arena_alias)) {
            const size_t idx = arena_idx[key] >= 0 ? (size_t) arena_idx[key] : key * 2654435761u % arena_blobs;
            const uint8_t * b = arena + idx * blob;
            r.gate_up = b;
            r.down = b + pl.gate + pl.up;
            r.layout = arena_r8 ? 1 : 0;
            return r;
        }
        const HostExperts & x = m->layers[il].experts;
        r.gate_up = x.gate + (size_t) e * x.gate_bytes;
        r.up = x.up + (size_t) e * x.up_bytes;
        r.type_up = kIq4Xs;
        r.down = x.down + (size_t) e * x.down_bytes;
        return r;
    }
};

size_t routed_blob_bytes(const Geometry & g) {
    const Planes p = planes_of(g);
    return p.gate + p.up + p.down;
}
int routed_slots(const Routed & r) { return r.n_slots; }
bool routed_arena_pinned(const Routed & r) { return r.arena_pinned; }
bool routed_can_lend(const Routed & r) { return !r.arena_idx.empty() && (r.arena_pinned || r.bounce); }
size_t routed_pinned_bytes(const Routed & r) {
    const size_t T = (size_t) r.opt.max_T, K = (size_t) r.g.n_expert_used, D = (size_t) r.g.n_embd;
    return (r.bounce ? r.bounce->bytes() : 0) + T * K * 8 + T * D * 8;   // h_ids, h_w, h_y, h_acc
}
const RoutedOptions & routed_options(const Routed & r) { return r.opt; }
const std::vector<int64_t> & routed_counts(const Routed & r) { return r.counts; }
RoutedStats & routed_stats(Routed & r) { return r.stats; }
Pool & routed_pool(Routed & r) { return *r.pool; }
void routed_set_vnni(Routed & r, bool vnni) {
    if (vnni && r.q8y.empty()) fail("routed_set_vnni: created without VNNI buffers");
    r.opt.vnni = vnni;
}
void routed_perturb_once(Routed & r, float rel, int skip) {
    r.perturb = rel;
    r.perturb_skip = skip;
}
void routed_set_f64(Routed & r, bool f64) {
    r.f64 = f64;
    if (f64 && r.mid64.empty()) {
        r.y64.resize((size_t) r.opt.max_T * r.g.n_embd);
        r.mid64.resize(r.mid.size());
    }
}

namespace {

// GLM_ARENA_PIN = auto (default) | register | hostalloc | off: how the arena gets page-locked (the adaptive uploads and
// the prefill's routed_lend need it). register: cudaHostRegister after the NUMA-placed fill (Linux). hostalloc: a
// cudaHostAlloc block before the fill, one piece, no NUMA placement: what WSL2 pins (it refuses cudaHostRegister). off:
// pageable (the unpinned fallbacks, testable on any box). auto: hostalloc under WSL, else register.
enum class ArenaPin { reg, hostalloc, off };
ArenaPin arena_pin_mode() {
    const char * e = std::getenv("GLM_ARENA_PIN");
    const std::string v = e && *e ? e : "auto";
    if (v == "register") return ArenaPin::reg;
    if (v == "hostalloc") return ArenaPin::hostalloc;
    if (v == "off") return ArenaPin::off;
    if (v != "auto") fail("GLM_ARENA_PIN=" + v + ": expected auto, register, hostalloc or off");
    std::ifstream f("/proc/version");
    std::string ver;
    std::getline(f, ver);
    return ver.find("icrosoft") != std::string::npos ? ArenaPin::hostalloc : ArenaPin::reg;
}

void arena_free(Routed & r) {
    if (r.arena_hostalloc) {
        cudaFreeHost(r.arena);
    } else {
        if (r.arena_pinned) cudaHostUnregister(r.arena);
        if (r.arena) munmap(r.arena, r.arena_bytes);
    }
    r.arena = nullptr;
    r.arena_pinned = r.arena_hostalloc = false;
}

}  // namespace

void routed_destroy(Routed * r) {
    if (!r) return;
    r->pool.reset();
    if (r->synth) munmap(r->synth, r->synth_bytes);
    if (r->adapt_st) cudaStreamSynchronize(r->adapt_st);
    if (r->reclaim_st) cudaStreamSynchronize(r->reclaim_st);
    arena_free(*r);
    if (r->ev) cudaEventDestroy(r->ev);
    if (r->adapt_ev) cudaEventDestroy(r->adapt_ev);
    if (r->adapt_st) cudaStreamDestroy(r->adapt_st);
    if (r->reclaim_ev) cudaEventDestroy(r->reclaim_ev);
    if (r->reclaim_st) cudaStreamDestroy(r->reclaim_st);
    for (void * p : {(void *) r->slots, (void *) r->table, (void *) r->job_slots, r->gpu_scratch})
        if (p) cudaFree(p);
    for (void * p : {(void *) r->h_ids, (void *) r->h_w, (void *) r->h_y, (void *) r->h_acc})
        if (p) cudaFreeHost(p);
    delete r;
}

RoutedPtr routed_create(const Model & m, const RoutedOptions & opt) {
    RoutedPtr r(new Routed);
    r->m = &m;
    r->g = m.g;
    r->opt = opt;
    const Geometry & g = m.g;
    r->pl = planes_of(g);
    r->blob = routed_blob_bytes(g);
    r->n_layers = (int) m.layers.size();
    if (g.n_embd % 256 || g.ffn_expert % 256) fail("geometry not IQ4_XS compatible");
    if (opt.max_T < 1 || opt.max_T > 8) fail("max_T must be in [1, 8] (decode / speculative verify)");
    if (opt.max_T * g.n_expert_used > 64 || g.ffn_expert > 16 * 256)
        fail("routed_finish's Q8_K block counters hold 64 missed experts x 16 blocks");

    r->pool = std::make_unique<Pool>(std::max(0, opt.cpu_threads - 1));
    if (const char * e = std::getenv("GLM_POOL_PREWAKE_US")) r->prewake_us = std::atoi(e);
    if (opt.arena && opt.synthetic_pool_bytes) fail("arena and synthetic pool are exclusive");
    if (opt.synthetic_pool_bytes) {
        r->synth_blobs = std::max<size_t>(1, opt.synthetic_pool_bytes / r->blob);
        r->synth_bytes = r->synth_blobs * r->blob;
        void * p = mmap(nullptr, r->synth_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                        -1, 0);
        if (p == MAP_FAILED) fail("synthetic pool mmap failed");
        madvise(p, r->synth_bytes, MADV_NOHUGEPAGE);   // 4 KiB pages: the row chunks alternate between nodes
        r->synth = static_cast<uint8_t *>(p);
        // random quants, small finite fp16 block scales. Chunk c of every plane (kRowsChunk rows, a whole number
        // of pages) is first-touched by the pool on node c % nodes, the node whose workers read it.
        const size_t cg = (size_t) kRowsChunk * (g.n_embd / 256) * kXsBlock;
        const size_t cd = (size_t) kRowsChunk * (g.ffn_expert / 32) * kNlBlock;
        const int n_g = g.ffn_expert / kRowsChunk, n_d = g.n_embd / kRowsChunk;
        const int nn = r->pool->nodes();
        // items: per node-aligned chunk index c, every blob; item i -> chunk c = i % (n_g or n_d)
        const int n_items = std::max(n_g, n_d);
        const std::function<void(int)> fill = [&](int c) {
            for (size_t b = 0; b < r->synth_blobs; ++b) {
                uint8_t * base = r->synth + b * r->blob;
                auto put = [&](uint8_t * q, size_t n, size_t blk) {
                    uint64_t x = 0x9E3779B97F4A7C15ull * (b * 1315423911u + (size_t) (q - base) + 1);
                    for (size_t i = 0; i + 8 <= n; i += 8) {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        std::memcpy(q + i, &x, 8);
                    }
                    const uint16_t d = 0x1400;   // ~1e-3
                    for (size_t o = 0; o < n; o += blk) std::memcpy(q + o, &d, 2);
                };
                if (c < n_g) {
                    put(base + c * cg, cg, kXsBlock);
                    put(base + r->pl.gate + c * cg, cg, kXsBlock);
                }
                if (c < n_d) put(base + r->pl.gate + r->pl.up + c * cd, cd, kNlBlock);
            }
        };
        if (n_g % nn || n_d % nn) fail("row chunks do not divide over the NUMA nodes");
        r->pool->run(n_items, fill);
    } else {
        for (int il = g.n_dense_lead; il < r->n_layers; ++il) {
            const HostExperts & x = m.layers[il].experts;
            if (!x.gate || !x.up || !x.down) fail("layer " + std::to_string(il) + ": experts not mapped");
            if (x.type_gate != kIq4Xs || x.type_up != kIq4Xs || x.type_down != kIq4Nl)
                fail("layer " + std::to_string(il) + ": experts are not IQ4_XS/IQ4_XS/IQ4_NL (types " +
                     std::to_string(x.type_gate) + "/" + std::to_string(x.type_up) + "/" +
                     std::to_string(x.type_down) + "); use an IQ4 GGUF or the synthetic pool");
            if (x.gate_bytes != r->pl.gate || x.up_bytes != r->pl.up || x.down_bytes != r->pl.down)
                fail("layer " + std::to_string(il) + ": unexpected expert plane sizes");
        }
    }

    // VRAM cache
    const int K = g.n_expert_used;
    const size_t total_experts = (size_t) (r->n_layers - g.n_dense_lead) * g.n_expert;
    ck(cudaMalloc(&r->table, (size_t) r->n_layers * g.n_expert * sizeof(ExpertRef)), "table");
    ck(cudaMemset(r->table, 0, (size_t) r->n_layers * g.n_expert * sizeof(ExpertRef)), "table");
    ck(cudaMalloc(&r->job_slots, (size_t) opt.max_T * K * sizeof(ExpertRef)), "job slots");
    ck(cudaMalloc(&r->gpu_scratch, experts_gpu_scratch_bytes(g, opt.max_T)), "gpu scratch");
    size_t budget = opt.vram_bytes;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "mem info");
    const size_t avail = free_b > opt.reserve_bytes ? free_b - opt.reserve_bytes : 0;
    if (opt.slots > 0) {
        budget = (size_t) opt.slots * r->blob;
        if (budget > avail) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "%d slots need %.2f GB, %.2f GB free past the reserve", opt.slots,
                          budget / 1e9, avail / 1e9);
            fail(msg);
        }
    }
    if (!budget) budget = avail;
    r->n_slots = (int) std::min(total_experts, budget / r->blob);
    if (r->n_slots > 0) ck(cudaMalloc(&r->slots, (size_t) r->n_slots * r->blob), "expert slots");
    r->resident.assign((size_t) r->n_layers * g.n_expert, -1);
    r->htable.assign((size_t) r->n_layers * g.n_expert, ExpertRef{});
    if (opt.adapt_every > 0) {
        r->usage.assign((size_t) r->n_layers * g.n_expert, 0.0f);
        ck(cudaStreamCreateWithFlags(&r->adapt_st, cudaStreamNonBlocking), "adapt stream");
        ck(cudaEventCreateWithFlags(&r->adapt_ev, cudaEventDisableTiming), "adapt event");
    }
    r->counts.assign((size_t) r->n_layers * g.n_expert, 0);

    ck(cudaMallocHost(&r->h_ids, (size_t) opt.max_T * K * 4), "pinned ids");
    ck(cudaMallocHost(&r->h_w, (size_t) opt.max_T * K * 4), "pinned w");
    ck(cudaMallocHost(&r->h_y, (size_t) opt.max_T * g.n_embd * 4), "pinned y");
    ck(cudaHostAlloc(&r->h_acc, (size_t) opt.max_T * g.n_embd * 4, cudaHostAllocMapped), "pinned acc");
    ck(cudaHostGetDevicePointer(&r->h_acc_dev, r->h_acc, 0), "acc device alias");
    ck(cudaEventCreateWithFlags(&r->ev, cudaEventDisableTiming), "event");
    // routed_finish's per-miss buffers: row u * 8 + t (missed expert u < K * max_T, token t < max_T <= 8)
    const size_t miss_rows = (size_t) K * opt.max_T * 8;
    r->mid.resize(miss_rows * g.ffn_expert);
    // opt.vnni = the int8 miss kernels: AVX512 VNNI, or AVX2 (cpu_isa(), GLM_CPU_ISA); exact F32 without either
    if (r->opt.vnni && !iq4_int8_available()) r->opt.vnni = false;
    if (r->opt.vnni) {
        r->q8y_row = iq4_vnni_q8_bytes(g.n_embd);
        r->q8mid_row = opt.down_q8_0 ? iq4_vnni_q8_0_bytes(g.ffn_expert) : iq4_vnni_q8_bytes(g.ffn_expert);
        r->q8y.resize((size_t) opt.max_T * r->q8y_row);
        r->q8mid.resize(miss_rows * r->q8mid_row);
    }
    if (r->opt.r8 && (!r->opt.vnni || !r8_available() || r->opt.synthetic_pool_bytes || !r->opt.arena)) {
        if (opt.verbose && r->opt.vnni && !r8_available())
            std::printf("routed: R8 arena off (CPU isa %s: R8 needs AVX512 VNNI)\n", cpu_isa_name(cpu_isa()));
        else if (opt.verbose)
            std::printf("routed: R8 arena off (needs the int8 misses, an AVX512 VNNI CPU and the arena)\n");
        r->opt.r8 = false;
    }
    if (r->opt.r8) {
        const size_t rows = miss_rows, nb32 = g.ffn_expert / 32;
        r->r8_corr_y.resize((size_t) opt.max_T * (g.n_embd / 32));
        r->r8_d_y.resize((size_t) opt.max_T * (g.n_embd / 256));
        r->r8_corr_mid.resize(rows * nb32);
        r->r8_d_mid.resize(rows * (g.ffn_expert / 256));
        r->r8_x64.resize(rows * nb32 * kR8X64 + kR8X64);
        r->r8_x64_at = r->r8_x64.data() + (kR8X64 - (uintptr_t) r->r8_x64.data() % kR8X64) % kR8X64;
    }
    if (opt.verbose)
        std::printf("routed: %d VRAM slots (%.1f GB, %.1f%% of %zu experts), %d miss threads on %d NUMA nodes, "
                    "source %s, %s miss kernel (CPU isa %s%s%s)\n",
                    r->n_slots, (double) r->n_slots * r->blob / 1e9, 100.0 * r->n_slots / total_experts,
                    total_experts, r->pool->size(), r->pool->nodes(),
                    r->synth ? ("synthetic pool " + std::to_string(r->synth_blobs) + " blobs").c_str()
                             : opt.arena ? "GGUF (host arena at routed_load, mmap until then)" : "GGUF (mmap)",
                    !r->opt.vnni ? "exact F32"
                    : r->opt.r8  ? (opt.down_q8_0 ? "R8 VNNI int8 (down x Q8_0 multi)" : "R8 VNNI int8 (down x Q8_K multi)")
                    : !iq4_vnni_available() ? (opt.down_q8_0 ? "AVX2 int8 (down x Q8_0 per token)" : "AVX2 int8 (down x Q8_K multi)")
                    : opt.down_q8_0 ? "VNNI int8 (down x Q8_0 per token)" : "VNNI int8 (down x Q8_K multi)",
                    cpu_isa_name(cpu_isa()), cpu_isa() != cpu_isa_hardware() ? ", lowered by GLM_CPU_ISA" : "",
                    cpu_has_avx_vnni() ? ", AVX-VNNI available, not used yet" : "");
    return r;
}

namespace {
// waits for the uploads in flight and forgets them (the cache content is about to be replaced)
void adapt_reset(Routed & r) {
    if (r.adapt_st) ck(cudaStreamSynchronize(r.adapt_st), "adapt sync");
    r.pending.clear();
    if (r.reclaim_st) ck(cudaStreamSynchronize(r.reclaim_st), "reclaim sync");
    r.reclaiming.clear();
    std::fill(r.usage.begin(), r.usage.end(), 0.0f);
}
}  // namespace

void routed_fill(Routed & r, const std::vector<std::pair<int, int>> & ranked, cudaStream_t st) {
    const Geometry & g = r.g;
    adapt_reset(r);
    if (r.lent) fail("routed_fill during a routed_lend loan");
    std::fill(r.resident.begin(), r.resident.end(), -1);
    std::vector<ExpertRef> table((size_t) r.n_layers * g.n_expert);
    int s = 0;
    const double t0 = now_ms();
    for (const auto & [il, e] : ranked) {
        if (s >= r.n_slots) break;
        if (il < g.n_dense_lead || il >= r.n_layers || e < 0 || e >= g.n_expert) continue;
        const size_t key = (size_t) il * g.n_expert + e;
        if (r.resident[key] >= 0) continue;
        uint8_t * dst = static_cast<uint8_t *>(r.slots) + (size_t) s * r.blob;
        routed_upload_expert(r, r.host_ref(il, e), dst, st);
        ExpertRef & t = table[key];
        t.gate_up = dst;
        t.down = dst + r.pl.gate + r.pl.up;
        r.resident[key] = s++;
    }
    ck(cudaMemcpyAsync(r.table, table.data(), table.size() * sizeof(ExpertRef), cudaMemcpyHostToDevice, st), "table");
    ck(cudaStreamSynchronize(st), "fill");
    r.htable = std::move(table);
    if (r.opt.verbose)
        std::printf("routed: filled %d / %d slots in %.1f s\n", s, r.n_slots, (now_ms() - t0) / 1000.0);
}

void routed_load(Routed & r, const std::vector<std::pair<int, int>> & ranked, cudaStream_t st) {
    if (!r.opt.arena || r.synth) fail("routed_load needs RoutedOptions::arena and a GGUF source");
    const Geometry & g = r.g;
    const int ne = g.n_expert;
    const double t0 = now_ms();
    // placement: VRAM first, then the arena up to the cap; with the adaptive tier the arena also keeps a copy of the
    // VRAM experts (an evicted expert's misses must not read the HDD)
    const bool keep_all = r.opt.adapt_every > 0 || r.opt.pin_arena;
    if (r.lent) fail("routed_load during a routed_lend loan");
    const size_t keys = (size_t) r.n_layers * ne;
    std::vector<int> slot(keys, -1), aidx(keys, -1);
    const size_t cap = r.opt.arena_max_bytes ? r.opt.arena_max_bytes / r.blob : keys;
    int s = 0;
    size_t a = 0;
    for (const auto & [il, e] : ranked) {
        if (il < g.n_dense_lead || il >= r.n_layers || e < 0 || e >= ne) continue;
        const size_t key = (size_t) il * ne + e;
        if (slot[key] >= 0 || aidx[key] >= 0) continue;
        const bool vram = s < r.n_slots;
        if (vram) slot[key] = s++;
        if ((!vram || keep_all) && a < cap) aidx[key] = (int) a++;
    }
    // the experts the ranking leaves out (the MTP block's: profiles rank the trunk's 45 layers) get an arena copy too,
    // behind the ranked ones: otherwise their misses, adaptive uploads and loan reclaims read the GGUF mapping, whose
    // pageable copies block the host until the stream drains (a lent MTP expert made every later reclaim_async wait
    // for the whole re-upload: +0.1 / 0.24 / 0.36 s TTFT at 200 / 800 / 2000 new tokens)
    for (size_t key = (size_t) g.n_dense_lead * ne; key < keys && a < cap; ++key)
        if (slot[key] < 0 && aidx[key] < 0) aidx[key] = (int) a++;
    size_t unplaced = 0;
    for (size_t key = (size_t) g.n_dense_lead * ne; key < keys; ++key) unplaced += slot[key] < 0 && aidx[key] < 0;
    adapt_reset(r);
    arena_free(r);
    r.arena_blobs = a;
    r.arena_bytes = std::max<size_t>(a, 1) * r.blob;
    r.arena_r8 = r.opt.r8;
    // refused before any allocation (pinned_budget.hpp arena_fits); each earlier arena of this process is already
    // filled, so several Routed loaded in turn are covered
    if (a) {
        const size_t margin = arena_margin_bytes(), avail = mem_available_bytes();
        if (!arena_fits(r.arena_bytes, margin, avail)) {
            char msg[320];
            std::snprintf(msg, sizeof msg,
                          "host arena of %.1f GB (%zu experts) does not fit: MemAvailable %.1f GB, margin %.1f GB "
                          "(GLM_ARENA_MARGIN_GB); cap it (RoutedOptions::arena_max_bytes) or free memory",
                          r.arena_bytes / 1e9, a, avail / 1e9, margin / 1e9);
            fail(msg);
        }
    }
    const ArenaPin pin = keep_all && a ? arena_pin_mode() : ArenaPin::reg;
    bool hostalloc_refused = false;
    if (pin == ArenaPin::hostalloc) {
        void * h = nullptr;
        r.arena_hostalloc = r.arena_pinned = cudaHostAlloc(&h, r.arena_bytes, cudaHostAllocDefault) == cudaSuccess;
        if (r.arena_hostalloc) r.arena = static_cast<uint8_t *>(h);
        else hostalloc_refused = true, cudaGetLastError();
    }
    if (!r.arena) {
        void * p = mmap(nullptr, r.arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) fail("arena mmap failed");
        madvise(p, r.arena_bytes, MADV_NOHUGEPAGE);   // 4 KiB pages: the row chunks alternate between nodes
        r.arena = static_cast<uint8_t *>(p);
    }

    // one pass over the expert planes in file order; runs of up to 32 needed experts per read
    struct Plane { std::string path; uint64_t off; int il, p; };
    std::vector<Plane> planes;
    for (int il = g.n_dense_lead; il < r.n_layers; ++il) {
        const HostExperts & x = r.m->layers[il].experts;
        const uint8_t * base[3] = {x.gate, x.up, x.down};
        for (int pi = 0; pi < 3; ++pi) {
            Plane pl{{}, 0, il, pi};
            if (!r.m->gguf->locate(base[pi], &pl.path, &pl.off)) fail("expert plane not in the GGUF mapping");
            planes.push_back(pl);
        }
    }
    std::sort(planes.begin(), planes.end(),
              [](const Plane & x, const Plane & y) { return x.path != y.path ? x.path < y.path : x.off < y.off; });
    const size_t pbytes[3] = {r.pl.gate, r.pl.up, r.pl.down};
    const size_t poff[3] = {0, r.pl.gate, r.pl.gate + r.pl.up};
    const size_t cbytes[3] = {(size_t) kRowsChunk * (g.n_embd / 256) * kXsBlock,
                              (size_t) kRowsChunk * (g.n_embd / 256) * kXsBlock,
                              (size_t) kRowsChunk * (g.ffn_expert / 32) * kNlBlock};
    const int nn = r.pool->nodes();
    DirectReader rd;
    double last = now_ms();
    for (size_t pi = 0; pi < planes.size(); ++pi) {
        const Plane & pl = planes[pi];
        const size_t pb = pbytes[pl.p], cb = cbytes[pl.p];
        const int nch = (int) (pb / cb);
        if (nch % nn) fail("row chunks do not divide over the NUMA nodes");
        auto needed = [&](int e) {
            const size_t key = (size_t) pl.il * ne + e;
            return slot[key] >= 0 || aidx[key] >= 0;
        };
        for (int e0 = 0; e0 < ne;) {
            if (!needed(e0)) { ++e0; continue; }
            int e1 = e0 + 1;
            while (e1 < ne && e1 - e0 < 32 && needed(e1)) ++e1;
            const uint8_t * src = rd.read(pl.path, pl.off + (uint64_t) e0 * pb, (size_t) (e1 - e0) * pb);
            std::vector<std::pair<const uint8_t *, uint8_t *>> to_arena;
            for (int e = e0; e < e1; ++e) {
                const size_t key = (size_t) pl.il * ne + e;
                const uint8_t * from = src + (size_t) (e - e0) * pb;
                if (slot[key] >= 0)
                    ck(cudaMemcpy(static_cast<uint8_t *>(r.slots) + (size_t) slot[key] * r.blob + poff[pl.p], from, pb,
                                  cudaMemcpyHostToDevice), "load slot");
                if (aidx[key] >= 0)
                    to_arena.emplace_back(from, r.arena + (size_t) aidx[key] * r.blob + poff[pl.p]);
            }
            // item j * nch + c copies chunk c of the j-th expert, first-touching its pages on node c % nodes (the
            // policy, not the thread: a stolen item or the unpinned caller misplaced 3-5 % of the pages); an R8 arena
            // gets the chunk's four 8-row groups repacked on the way (same bytes, same place)
            const bool xs = pl.p < 2;
            const int width = xs ? g.n_embd : g.ffn_expert;
            const size_t gbytes = cb / (kRowsChunk / kR8Rows);
            const std::function<void(int)> copy = [&](int item) {
                const int c = item % nch, j = item / nch;
                const uint8_t * from = to_arena[j].first + (size_t) c * cb;
                uint8_t * to = to_arena[j].second + (size_t) c * cb;
                if (nn > 1) numa_prefer(c % nn);
                if (!r.arena_r8) std::memcpy(to, from, cb);
                else
                    for (size_t o = 0; o < cb; o += gbytes)
                        xs ? r8_repack_xs(from + o, to + o, width) : r8_repack_nl(from + o, to + o, width);
                if (nn > 1) numa_prefer(-1);
            };
            r.pool->run((int) to_arena.size() * nch, copy);
            e0 = e1;
        }
        if (r.opt.verbose && now_ms() - last > 30000) {
            last = now_ms();
            std::printf("routed: load %zu / %zu planes, %.1f GB read, %.0f MB/s\n", pi + 1, planes.size(),
                        rd.bytes() / 1e9, rd.bytes() / 1e3 / (last - t0));
            std::fflush(stdout);
        }
    }

    std::vector<ExpertRef> table(keys);
    for (size_t key = 0; key < keys; ++key)
        if (slot[key] >= 0) {
            uint8_t * dst = static_cast<uint8_t *>(r.slots) + (size_t) slot[key] * r.blob;
            table[key].gate_up = dst;
            table[key].down = dst + r.pl.gate + r.pl.up;
        }
    ck(cudaMemcpyAsync(r.table, table.data(), table.size() * sizeof(ExpertRef), cudaMemcpyHostToDevice, st), "table");
    ck(cudaStreamSynchronize(st), "load");
    r.resident = slot;
    r.arena_idx = aidx;
    r.htable = std::move(table);
    // the adaptive tier uploads from the arena on a side stream: pinned, or every copy would stage synchronously
    if (keep_all && a) {
        const double p0 = now_ms();
        if (pin == ArenaPin::reg) {
            r.arena_pinned = cudaHostRegister(r.arena, r.arena_bytes, cudaHostRegisterDefault) == cudaSuccess;
            if (!r.arena_pinned) cudaGetLastError();
        }
        // not pinned (WSL2: 1 GiB of page-locked memory per process in total): the uploads stage through a pinned
        // ring of GLM_BOUNCE_SLOTS blobs (default 8, halved down to 2 if refused) filled by GLM_BOUNCE_THREADS
        // memcpy threads (default 4, the caller included)
        if (!r.arena_pinned && !r.bounce) {
            const char * es = std::getenv("GLM_BOUNCE_SLOTS"), * et = std::getenv("GLM_BOUNCE_THREADS");
            const int threads = et && *et ? std::max(1, std::atoi(et)) : 4;
            for (int n = es && *es ? std::max(2, std::atoi(es)) : 8; n >= 2 && !r.bounce; n /= 2) {
                try {
                    r.bounce = std::make_unique<BounceRing>(n, r.blob, threads);
                } catch (const std::exception &) {
                }
            }
        }
        if (r.opt.verbose) {
            const std::string how = r.bounce ? "uploads through a pinned bounce ring of " + std::to_string(r.bounce->slots()) +
                                                   " blobs (" + std::to_string(r.bounce->bytes() >> 20) + " MB, " +
                                                   std::to_string(r.bounce->threads()) + " copy threads)"
                                             : std::string("staged uploads, no pinned bounce ring: no prefill loan");
            std::printf("routed: arena %s (%.1f s)\n",
                        r.arena_hostalloc      ? "pinned (cudaHostAlloc)"
                        : r.arena_pinned       ? "pinned"
                        : hostalloc_refused    ? ("NOT pinned (cudaHostAlloc refused): " + how).c_str()
                        : pin == ArenaPin::off ? ("NOT pinned (GLM_ARENA_PIN=off): " + how).c_str()
                                               : ("NOT pinned (cudaHostRegister refused): " + how).c_str(),
                        (now_ms() - p0) / 1000.0);
        }
    }
    if (r.opt.verbose)
        std::printf("routed: loaded %d VRAM slots + %zu arena blobs (%.1f GB host), %zu experts unplaced%s; "
                    "%.1f GB read in %.0f s (%.0f MB/s)\n",
                    s, a, (double) a * r.blob / 1e9, unplaced, !unplaced              ? ""
                    : r.opt.arena_alias ? " (their misses alias: timing only)"
                                        : " (their misses read the GGUF mapping)",
                    rd.bytes() / 1e9, (now_ms() - t0) / 1000.0, rd.bytes() / 1e3 / (now_ms() - t0));
}

void routed_write_profile(const std::string & path, const std::vector<std::pair<int, int>> & ranked, int n_layers,
                          int n_expert) {
    std::vector<int32_t> table((size_t) n_layers * n_expert, -1);
    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto [il, e] = ranked[i];
        if (il < 0 || il >= n_layers || e < 0 || e >= n_expert || table[(size_t) il * n_expert + e] >= 0)
            fail("profile: invalid or duplicate pair");
        table[(size_t) il * n_expert + e] = (int32_t) i;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("cannot write profile " + path);
    const uint32_t h[5] = {1, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                           (uint32_t) ranked.size()};
    f.write("STRP", 4);
    f.write(reinterpret_cast<const char *>(h), sizeof h);
    for (const auto & [il, e] : ranked) {
        const uint16_t pr[2] = {(uint16_t) il, (uint16_t) e};
        f.write(reinterpret_cast<const char *>(pr), sizeof pr);
    }
    f.write(reinterpret_cast<const char *>(table.data()), (std::streamsize) (table.size() * 4));
    if (!f) fail("profile write failed: " + path);
}

std::vector<std::pair<int, int>> routed_read_profile(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open profile " + path);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 24 || std::memcmp(b.data(), "STRP", 4) != 0) fail(path + ": not a STRP profile");
    uint32_t h[5];
    std::memcpy(h, b.data() + 4, sizeof(h));
    if (h[0] != 1) fail(path + ": unsupported STRP version");
    const uint32_t count = h[4];
    if (b.size() < 24 + (size_t) 4 * count) fail(path + ": truncated");
    std::vector<std::pair<int, int>> out(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t p[2];
        std::memcpy(p, b.data() + 24 + 4 * i, 4);
        out[i] = {p[0], p[1]};
    }
    return out;
}

ExpertRef routed_host_expert(const Routed & r, int il, int e) { return r.host_ref(il, e); }

void routed_upload_expert(const Routed & r, const ExpertRef & src, void * dst, cudaStream_t s) {
    auto * d = static_cast<uint8_t *>(dst);
    const uint8_t * up = src.up ? static_cast<const uint8_t *>(src.up)
                                : static_cast<const uint8_t *>(src.gate_up) + r.pl.gate;
    if (r.bounce && !r.arena_pinned) {   // pageable arena: through the pinned ring, same bytes in the slot
        const BounceRing::Seg seg[3] = {{src.gate_up, r.pl.gate}, {up, r.pl.up}, {src.down, r.pl.down}};
        r.bounce->copy(d, seg, 3, s);
    } else {
        ck(cudaMemcpyAsync(d, src.gate_up, r.pl.gate, cudaMemcpyHostToDevice, s), "upload gate");
        ck(cudaMemcpyAsync(d + r.pl.gate, up, r.pl.up, cudaMemcpyHostToDevice, s), "upload up");
        ck(cudaMemcpyAsync(d + r.pl.gate + r.pl.up, src.down, r.pl.down, cudaMemcpyHostToDevice, s), "upload down");
    }
    if (src.layout == 1) {   // R8 host copy: back to GGUF rows in place before anything reads the blob
        r8_plane_to_gguf(d, true, r.g.n_embd, r.g.ffn_expert, s);
        r8_plane_to_gguf(d + r.pl.gate, true, r.g.n_embd, r.g.ffn_expert, s);
        r8_plane_to_gguf(d + r.pl.gate + r.pl.up, false, r.g.ffn_expert, r.g.n_embd, s);
    }
}

ExpertRef routed_slot_ref(const Routed & r, int il, int e) {
    const size_t key = (size_t) il * r.g.n_expert + e;
    return r.resident[key] >= 0 ? r.htable[key] : ExpertRef{nullptr, nullptr, kIq4Xs, kIq4Nl, nullptr, -1};
}

namespace {
void admit_pending(Routed & r, cudaStream_t st) {
    for (const auto & [key, slot] : r.pending) {
        uint8_t * dst = static_cast<uint8_t *>(r.slots) + (size_t) slot * r.blob;
        r.htable[key].gate_up = dst;
        r.htable[key].down = dst + r.pl.gate + r.pl.up;
        r.resident[key] = slot;
        ck(cudaMemcpyAsync(r.table + key, &r.htable[key], sizeof(ExpertRef), cudaMemcpyHostToDevice, st), "admit");
    }
    r.pending.clear();
}
// the reclaimed experts back as hits (their uploads have landed)
void admit_reclaimed(Routed & r, cudaStream_t st) {
    for (const auto & [key, slot] : r.reclaiming) {
        uint8_t * dst = static_cast<uint8_t *>(r.slots) + (size_t) slot * r.blob;
        r.htable[key].gate_up = dst;
        r.htable[key].down = dst + r.pl.gate + r.pl.up;
        r.resident[key] = slot;
        ck(cudaMemcpyAsync(r.table + key, &r.htable[key], sizeof(ExpertRef), cudaMemcpyHostToDevice, st), "admit");
    }
    r.reclaiming.clear();
}
// uploads of the evicted experts (lent_keys) back into their slots, on s
void reclaim_uploads(Routed & r, cudaStream_t s) {
    for (const auto & [key, slot] : r.lent_keys) {
        routed_upload_expert(r, r.host_ref((int) (key / r.g.n_expert), (int) (key % r.g.n_expert)),
                             static_cast<uint8_t *>(r.slots) + (size_t) slot * r.blob, s);
    }
}
}  // namespace

void routed_quiesce(Routed & r, cudaStream_t st) {
    if (!r.reclaiming.empty()) {
        ck(cudaEventSynchronize(r.reclaim_ev), "quiesce reclaim");
        admit_reclaimed(r, st);
    }
    if (r.pending.empty()) return;
    ck(cudaEventSynchronize(r.adapt_ev), "quiesce");
    admit_pending(r, st);
}

int routed_verify_slots(Routed & r, cudaStream_t st) {
    if (r.arena_r8 || r.arena_idx.empty() || r.synth) return -1;
    routed_quiesce(r, st);
    ck(cudaStreamSynchronize(st), "verify");
    std::vector<ExpertRef> dt(r.htable.size());
    ck(cudaMemcpy(dt.data(), r.table, dt.size() * sizeof(ExpertRef), cudaMemcpyDeviceToHost), "verify table");
    std::vector<uint8_t> b(r.blob);
    int bad = 0, n = 0;
    for (size_t key = 0; key < r.resident.size(); ++key) {
        const int il = (int) (key / r.g.n_expert), e = (int) (key % r.g.n_expert);
        const int slot = r.resident[key];
        const uint8_t * dev = slot >= 0 ? static_cast<uint8_t *>(r.slots) + (size_t) slot * r.blob : nullptr;
        if (dt[key].gate_up != (slot >= 0 ? dev : nullptr) ||
            (slot >= 0 && dt[key].down != dev + r.pl.gate + r.pl.up)) {
            if (bad++ < 8)
                std::printf("verify: layer %d expert %d: device table %p, host slot %d (%p)\n", il, e, dt[key].gate_up,
                            slot, (const void *) dev);
            continue;
        }
        if (slot < 0) continue;
        ++n;
        ck(cudaMemcpy(b.data(), dev, r.blob, cudaMemcpyDeviceToHost), "verify blob");
        const ExpertRef h = r.host_ref(il, e);
        const uint8_t * up = h.up ? static_cast<const uint8_t *>(h.up) : static_cast<const uint8_t *>(h.gate_up) + r.pl.gate;
        const bool same = !std::memcmp(b.data(), h.gate_up, r.pl.gate) && !std::memcmp(b.data() + r.pl.gate, up, r.pl.up) &&
                          !std::memcmp(b.data() + r.pl.gate + r.pl.up, h.down, r.pl.down);
        if (!same && bad++ < 8)
            std::printf("verify: layer %d expert %d in slot %d: VRAM blob differs from its host copy\n", il, e, slot);
    }
    std::printf("verify: %d resident experts, %d mismatches (slots %d, lent %d)\n", n, bad, r.n_slots, r.lent);
    return bad;
}

void * routed_lend(Routed & r, size_t bytes, cudaStream_t st) {
    if (r.lent) fail("routed_lend: a loan is already out");
    if (!routed_can_lend(r)) fail("routed_lend needs an arena keeping every expert (pin_arena), pinned or with the bounce ring");
    const int k = (int) ((bytes + r.blob - 1) / r.blob);
    if (k <= 0 || k > r.n_slots) fail("routed_lend: bad size");
    routed_quiesce(r, st);
    const int s0 = r.n_slots - k;
    for (size_t key = 0; key < r.resident.size(); ++key) {
        if (r.resident[key] < s0) continue;
        r.lent_keys.emplace_back(key, r.resident[key]);
        r.resident[key] = -1;
        r.htable[key] = ExpertRef{};
        ck(cudaMemcpyAsync(r.table + key, &r.htable[key], sizeof(ExpertRef), cudaMemcpyHostToDevice, st), "lend");
    }
    r.lent = k;
    r.n_slots = s0;   // the adaptive tier only sees the remaining slots
    ck(cudaStreamSynchronize(st), "lend");
    return static_cast<uint8_t *>(r.slots) + (size_t) s0 * r.blob;
}

void routed_reclaim(Routed & r, cudaStream_t st) {
    if (!r.lent) return;
    reclaim_uploads(r, st);
    r.reclaiming = std::move(r.lent_keys);
    admit_reclaimed(r, st);
    ck(cudaStreamSynchronize(st), "reclaim");
    r.n_slots += r.lent;
    r.lent = 0;
    r.lent_keys.clear();
}

void routed_reclaim_async(Routed & r, cudaStream_t st) {
    if (!r.lent) return;
    if (!r.reclaim_st) {
        ck(cudaStreamCreateWithFlags(&r.reclaim_st, cudaStreamNonBlocking), "reclaim stream");
        ck(cudaEventCreateWithFlags(&r.reclaim_ev, cudaEventDisableTiming), "reclaim event");
    }
    // after the borrower's work on st (the range is free once st reaches here)
    ck(cudaEventRecord(r.reclaim_ev, st), "reclaim fork");
    ck(cudaStreamWaitEvent(r.reclaim_st, r.reclaim_ev, 0), "reclaim fork wait");
    reclaim_uploads(r, r.reclaim_st);
    ck(cudaEventRecord(r.reclaim_ev, r.reclaim_st), "reclaim event");
    r.reclaiming = std::move(r.lent_keys);
    r.n_slots += r.lent;
    r.lent = 0;
    r.lent_keys.clear();
}

void routed_set_adapt(Routed & r, int every, int swaps) {
    adapt_reset(r);
    r.opt.adapt_every = every;
    r.opt.adapt_swaps = swaps;
    r.steps = 0;
    if (every > 0 && r.usage.empty()) {
        r.usage.assign((size_t) r.n_layers * r.g.n_expert, 0.0f);
        ck(cudaStreamCreateWithFlags(&r.adapt_st, cudaStreamNonBlocking), "adapt stream");
        ck(cudaEventCreateWithFlags(&r.adapt_ev, cudaEventDisableTiming), "adapt event");
    }
}

void routed_step(Routed & r, cudaStream_t st) {
    // landed reclaims (any mode); no adaptive swap while some are in flight (their keys are not resident yet)
    if (!r.reclaiming.empty()) {
        if (cudaEventQuery(r.reclaim_ev) != cudaSuccess) return;
        admit_reclaimed(r, st);
    }
    if (r.usage.empty() || r.opt.adapt_every <= 0) return;
    const size_t es = sizeof(ExpertRef);
    // admit the landed uploads
    if (!r.pending.empty() && cudaEventQuery(r.adapt_ev) == cudaSuccess) admit_pending(r, st);
    if (++r.steps % r.opt.adapt_every || !r.pending.empty()) return;
    // most-routed missing vs least-routed resident, over every layer (one blob size)
    std::vector<std::pair<float, size_t>> cand, vict;
    const size_t keys = r.usage.size();
    for (size_t key = (size_t) r.g.n_dense_lead * r.g.n_expert; key < keys; ++key) {
        if (r.resident[key] < 0) { if (r.usage[key] >= 2.0f) cand.emplace_back(r.usage[key], key); }
        else vict.emplace_back(r.usage[key], key);
    }
    const size_t nc = std::min({cand.size(), vict.size(), (size_t) std::max(0, r.opt.adapt_swaps)});
    if (nc) {
        std::partial_sort(cand.begin(), cand.begin() + (ptrdiff_t) nc, cand.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](const auto & a, const auto & b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            const size_t in = cand[i].second, out = vict[i].second;
            const int slot = r.resident[out];
            // the victim is a miss from the next step on: its table entry is cleared ahead of the upload
            r.resident[out] = -1;
            r.htable[out] = ExpertRef{};
            ck(cudaMemcpyAsync(r.table + out, &r.htable[out], es, cudaMemcpyHostToDevice, st), "evict");
            routed_upload_expert(r, r.host_ref((int) (in / r.g.n_expert), (int) (in % r.g.n_expert)),
                                 static_cast<uint8_t *>(r.slots) + (size_t) slot * r.blob, r.adapt_st);
            r.pending.emplace_back(in, slot);
            ++r.stats.swaps;
        }
        // the upload must not overwrite a slot the previous step's graph may still read: st is idle here, and the
        // evictions above are ordered before the next step on st
        if (!r.pending.empty()) ck(cudaEventRecord(r.adapt_ev, r.adapt_st), "adapt event");
    }
    for (float & v : r.usage) v *= r.opt.adapt_decay;
}

std::vector<std::pair<int, int>> routed_rank_counts(const Routed & r) {
    const int ne = r.g.n_expert;
    std::vector<int> idx;
    for (int k = r.g.n_dense_lead * ne; k < (int) r.counts.size(); ++k) idx.push_back(k);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return r.counts[a] > r.counts[b]; });
    std::vector<std::pair<int, int>> out;
    out.reserve(idx.size());
    for (int k : idx) out.emplace_back(k / ne, k % ne);
    return out;
}

void routed_start(Routed & r, int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                  cudaStream_t st) {
    const Geometry & g = r.g;
    const int K = g.n_expert_used, D = g.n_embd, ne = g.n_expert;
    if (T < 1 || T > r.opt.max_T) fail("T out of range");
    // routing + activations to the host first, then the hits on the GPU; the record is an external event node when
    // captured, so routed_finish can wait on it after the segment launch
    ck(cudaMemcpyAsync(r.h_ids, ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st), "ids d2h");
    ck(cudaMemcpyAsync(r.h_w, w, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st), "w d2h");
    ck(cudaMemcpyAsync(r.h_y, y, (size_t) T * D * 4, cudaMemcpyDeviceToHost, st), "y d2h");
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    ck(cudaStreamIsCapturing(st, &cs), "capture status");
    ck(cudaEventRecordWithFlags(r.ev, st, cs == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : 0), "event");
    ck(cudaMemsetAsync(out, 0, (size_t) T * D * 4, st), "zero out");
    gather_slots_kernel<<<1, 256, 0, st>>>(r.table + (size_t) il * ne, ids, T * K, ne, r.job_slots);
    ck(cudaGetLastError(), "gather slots");
    experts_gpu(g, T, y, ids, w, r.job_slots, out, r.gpu_scratch, st);
}

void routed_run(Routed & r, int il, const float * y, const int32_t * ids, const float * w, float * out, int T,
                cudaStream_t st) {
    routed_start(r, il, y, ids, w, out, T, st);
    routed_finish(r, il, out, T, st);
}

void routed_finish(Routed & r, int il, float * out, int T, cudaStream_t st) {
    const Geometry & g = r.g;
    const int K = g.n_expert_used, D = g.n_embd, F = g.ffn_expert, ne = g.n_expert;
    if (T < 1 || T > r.opt.max_T) fail("T out of range");
    const double t0 = now_ms();
    // the workers sleep through the GPU part of a layer: wake them now, while the routing download is still on its
    // way, so their futex wake-up (~35 us p50 at 48 threads) is not paid at the first miss run
    if (r.opt.vnni && r.prewake_us > 0) r.pool->prewake(r.prewake_us);
    ck(cudaEventSynchronize(r.ev), "routing download");
    r.stats.wait_ms += now_ms() - t0;
    if (r.opt.on_route) r.opt.on_route(il, r.h_ids, r.h_w, T);
    // misses grouped by expert (decode: at most K distinct)
    struct Miss { ExpertRef ref; int n; int tok[8]; float wt[8]; int job[8]; };
    std::vector<Miss> miss;
    bool is_miss[64] = {};
    for (int j = 0; j < T * K; ++j) {
        const int e = r.h_ids[j];
        if (e < 0 || e >= ne) continue;
        const size_t key = (size_t) il * ne + e;
        ++r.counts[key];
        if (r.opt.adapt_every > 0 && !r.usage.empty()) r.usage[key] += 1.0f;
        ++r.stats.jobs;
        if (r.resident[key] >= 0) { ++r.stats.hits; continue; }
        ++r.stats.misses;
        is_miss[j] = true;
        if (!r.arena_idx.empty() && r.arena_idx[key] < 0) ++(r.opt.arena_alias ? r.stats.aliased : r.stats.spilled);
        const ExpertRef ref = r.host_ref(il, e);
        Miss * mm = nullptr;
        for (auto & x : miss) if (x.ref.gate_up == ref.gate_up) mm = &x;
        if (!mm) { miss.push_back({ref, 0, {}, {}, {}}); mm = &miss.back(); }
        mm->tok[mm->n] = j / K;
        mm->wt[mm->n] = r.h_w[j];
        mm->job[mm->n] = j;
        ++mm->n;
    }
    ++r.stats.calls;
    {
        // distinct experts of this call (T rows x K share some under speculative verify)
        int ids_u[64], n_u = 0;
        for (int j = 0; j < T * K && j < 64; ++j) {
            const int e = r.h_ids[j];
            if (e < 0 || e >= ne) continue;
            bool dup = false;
            for (int q = 0; q < n_u && !dup; ++q) dup = ids_u[q] == e;
            if (!dup) ids_u[n_u++] = e;
        }
        r.stats.distinct += n_u;
        r.stats.distinct_misses += (int64_t) miss.size();
    }
    // GLM_MISS_DUMP=path (diagnostic, glm_miss_error --dump): every miss appended as int32 layer, int32 expert and
    // the token's float y row (the miss kernels' input)
    static FILE * miss_dump = std::getenv("GLM_MISS_DUMP") ? std::fopen(std::getenv("GLM_MISS_DUMP"), "ab") : nullptr;
    if (miss_dump)
        for (const Miss & x : miss)
            for (int t = 0; t < x.n; ++t) {
                const int32_t h[2] = {il, r.h_ids[x.job[t]]};
                std::fwrite(h, 4, 2, miss_dump);
                std::fwrite(r.h_y + (size_t) x.tok[t] * D, 4, D, miss_dump);
            }
    if (!miss.empty()) {
        const double m0 = now_ms();
        const int U = (int) miss.size();
        if (U > K * r.opt.max_T) fail("more missed experts than the per-miss buffers hold");
        const size_t gb = (size_t) (D / 256) * kXsBlock, db = (size_t) (F / 32) * kNlBlock;
        // phase 1: gate/up rows of every missed expert -> mid[u][n][F]
        constexpr int kRows1 = kRowsChunk;
        const int c1 = F / kRows1;   // a multiple of the node count: item u * c1 + c is on node c % nodes
        const std::function<void(int)> f1 = [&](int item) {
            const Miss & x = miss[item / c1];
            const int r0 = (item % c1) * kRows1;
            const float * in[8];
            for (int t = 0; t < x.n; ++t) in[t] = r.h_y + (size_t) x.tok[t] * D;
            float * mid = r.mid.data() + (size_t) (item / c1) * 8 * F;
            const uint8_t * gate = static_cast<const uint8_t *>(x.ref.gate_up);
            const uint8_t * up = x.ref.up ? static_cast<const uint8_t *>(x.ref.up) : gate + (size_t) F * gb;
            const uint8_t * g0 = gate + r0 * gb, * u0 = up + r0 * gb;   // the chunk's rows
            if (x.ref.layout == 1) {   // R8 host copy: this chunk's rows back to GGUF first (same bytes, exact)
                thread_local std::vector<uint8_t> un;
                un.resize(2 * kRows1 * gb);
                r8_unpack_xs_rows(gate, F * gb, D, F, r0, kRows1, un.data(), kRows1 * gb);
                r8_unpack_xs_rows(up, F * gb, D, F, r0, kRows1, un.data() + kRows1 * gb, kRows1 * gb);
                g0 = un.data();
                u0 = un.data() + kRows1 * gb;
            }
            if (r.f64) {
                const double * in64[8];
                for (int t = 0; t < x.n; ++t) in64[t] = r.y64.data() + (size_t) x.tok[t] * D;
                double * mid64 = r.mid64.data() + (size_t) (item / c1) * 8 * F;
                double gv[8], uv[8];
                for (int row = r0; row < r0 + kRows1; ++row) {
                    iq4_cpu_row_f64(kIq4Xs, g0 + (row - r0) * gb, D, x.n, in64, gv);
                    iq4_cpu_row_f64(kIq4Xs, u0 + (row - r0) * gb, D, x.n, in64, uv);
                    for (int t = 0; t < x.n; ++t) mid64[(size_t) t * F + row] = expert_swiglu_clamp_f64(gv[t], uv[t]);
                }
                return;
            }
            float gv[8], uv[8];
            for (int row = r0; row < r0 + kRows1; ++row) {
                iq4_cpu_row(kIq4Xs, g0 + (row - r0) * gb, D, x.n, in, gv);
                iq4_cpu_row(kIq4Xs, u0 + (row - r0) * gb, D, x.n, in, uv);
                for (int t = 0; t < x.n; ++t) mid[(size_t) t * F + row] = expert_swiglu_clamp(gv[t], uv[t]);
            }
        };
        const bool vn = r.opt.vnni;
        if (!vn && r.f64)
            for (size_t i = 0; i < (size_t) T * D; ++i) r.y64[i] = r.h_y[i];
        if (vn)
            for (int t = 0; t < T; ++t) iq4_vnni_quantize(r.h_y + (size_t) t * D, D, r.q8y.data() + t * r.q8y_row);
        R8Act ay[8];   // y per token for the R8 gate / up kernels
        if (vn && r.arena_r8)
            for (int t = 0; t < T; ++t)
                ay[t] = r8_prep_q8k(r.q8y.data() + t * r.q8y_row, D, r.r8_corr_y.data() + (size_t) t * (D / 32),
                                    r.r8_d_y.data() + (size_t) t * (D / 256));
        // the mids are quantized inside phase 1, not serially between the runs: a Q8_0 block (32 rows) is one item's
        // own rows; a Q8_K block (256 rows) is quantized by the last of its items to finish (q8_left[u][block])
        static_assert(kRows1 == 32 && 256 % kRows1 == 0, "one Q8_0 block per item, whole items per Q8_K block");
        const bool q80 = r.opt.down_q8_0;
        const int nb8 = F / 256;
        std::atomic<int> q8_left[64 * 16];   // U <= T * K = 64, F / 256 <= 16
        if (vn && !q80)
            for (int i = 0; i < U * nb8; ++i) q8_left[i].store(256 / kRows1, std::memory_order_relaxed);
        const std::function<void(int)> f1v = [&](int item) {
            const Miss & x = miss[item / c1];
            const int r0 = (item % c1) * kRows1;
            float * mid = r.mid.data() + (size_t) (item / c1) * 8 * F;
            const uint8_t * gate = static_cast<const uint8_t *>(x.ref.gate_up);
            const uint8_t * up = x.ref.up ? static_cast<const uint8_t *>(x.ref.up) : gate + (size_t) F * gb;
            // every token of this expert in one pass over the rows (verify: the weights are unpacked once)
            float gv[8 * kRows1], uv[8 * kRows1];
            const bool r8x = x.ref.layout == 1;
            if (r8x) {   // the chunk's four R8 row groups (same bytes as its GGUF rows)
                R8Act a[8];
                for (int t = 0; t < x.n; ++t) a[t] = ay[x.tok[t]];
                r8_rows_xs(gate + r0 * gb, D, a, x.n, kRows1, gv);
                r8_rows_xs(up + r0 * gb, D, a, x.n, kRows1, uv);
            } else {
                const void * q[8];
                for (int t = 0; t < x.n; ++t) q[t] = r.q8y.data() + (size_t) x.tok[t] * r.q8y_row;
                iq4_vnni_rows_multi(kIq4Xs, gate + r0 * gb, D, q, x.n, kRows1, gv);
                iq4_vnni_rows_multi(kIq4Xs, up + r0 * gb, D, q, x.n, kRows1, uv);
            }
            for (int t = 0; t < x.n; ++t)
                for (int i = 0; i < kRows1; ++i)
                    mid[(size_t) t * F + r0 + i] = expert_swiglu_clamp(gv[t * kRows1 + i], uv[t * kRows1 + i]);
            const int u = item / c1;
            uint8_t * qm = r.q8mid.data() + (size_t) u * 8 * r.q8mid_row;
            if (q80) {
                for (int t = 0; t < x.n; ++t) {
                    uint8_t * blk = qm + t * r.q8mid_row + (r0 / 32) * iq4_vnni_q8_0_bytes(32);
                    iq4_vnni_quantize_q8_0(mid + (size_t) t * F + r0, kRows1, blk);
                    // R8 down: the same Q8_0 block as an x64 line (this item's own cache line)
                    if (r8x) r8_x64_store_q8_0(blk, r.r8_x64_at + (((size_t) u * 8 + t) * (F / 32) + r0 / 32) * kR8X64);
                }
            } else {
                const int b = r0 / 256;
                if (q8_left[u * nb8 + b].fetch_sub(1, std::memory_order_acq_rel) == 1)
                    for (int t = 0; t < x.n; ++t) {
                        uint8_t * blk = qm + t * r.q8mid_row + b * iq4_vnni_q8_bytes(256);
                        iq4_vnni_quantize(mid + (size_t) t * F + b * 256, 256, blk);
                        if (r8x)   // the block's R8 corrections and scale
                            r8_prep_q8k(blk, 256, r.r8_corr_mid.data() + ((size_t) u * 8 + t) * (F / 32) + b * 8,
                                        r.r8_d_mid.data() + ((size_t) u * 8 + t) * (F / 256) + b);
                    }
            }
        };
        r.pool->run(U * c1, vn ? f1v : f1);
        // phase 2: down rows, all experts per row chunk. Each token sums its experts in its own slot order, so a
        // token's result does not depend on the other tokens of the call (verify rows = the T = 1 decode sums).
        constexpr int kRows2 = kRowsChunk;
        std::memset(r.h_acc, 0, (size_t) T * D * 4);
        const std::function<void(int)> f2 = [&](int item) {
            const int r0 = item * kRows2;
            float dj[64][kRows2];   // per job (token x slot) of a missed expert
            for (int u = 0; u < U; ++u) {
                const Miss & x = miss[u];
                const uint8_t * down = static_cast<const uint8_t *>(x.ref.down);
                const bool r8d = x.ref.layout == 1 && !r.opt.r8_gguf_down;
                const uint8_t * dch = down + r0 * db;   // the chunk's down rows, GGUF unless r8d
                if (x.ref.layout == 1 && !r8d) {   // R8 host copy, GGUF down kernels: the chunk's rows back to GGUF
                    thread_local std::vector<uint8_t> und;
                    und.resize(kRows2 * db);
                    r8_unpack_nl_rows(down, D * db, F, D, r0, kRows2, und.data(), kRows2 * db);
                    dch = und.data();
                }
                if (vn && r8d) {
                    // R8 IQ4_NL down, every token of this expert in one pass: Q8_0 mids as x64 lines, or Q8_K
                    R8Act a[8];
                    for (int t = 0; t < x.n; ++t) {
                        const size_t m = (size_t) u * 8 + t;
                        if (q80) a[t] = r8_x64_act(r.r8_x64_at + m * (F / 32) * kR8X64, F);
                        else a[t] = R8Act{reinterpret_cast<const int8_t *>(r.q8mid.data() + m * r.q8mid_row + 4),
                                          (int) iq4_vnni_q8_bytes(256), 8, r.r8_corr_mid.data() + m * (F / 32),
                                          r.r8_d_mid.data() + m * (F / 256), 1, 1};
                    }
                    float dr[8 * kRows2];
                    r8_rows_nl(dch, F, a, x.n, kRows2, dr);
                    for (int t = 0; t < x.n; ++t) std::memcpy(dj[x.job[t]], dr + t * kRows2, sizeof(float) * kRows2);
                    continue;
                }
                if (vn && q80) {
                    // native IQ4_NL x Q8_0 down, per token, into this job's row slice
                    for (int t = 0; t < x.n; ++t)
                        iq4_vnni_nl_q8_0_rows(dch, F, r.q8mid.data() + ((size_t) u * 8 + t) * r.q8mid_row,
                                              kRows2, dj[x.job[t]]);
                    continue;
                }
                if (vn) {
                    // IQ4_NL x Q8_K down, every token of this expert in one pass over the rows
                    const void * qm[8];
                    float dr[8 * kRows2];
                    for (int t = 0; t < x.n; ++t) qm[t] = r.q8mid.data() + ((size_t) u * 8 + t) * r.q8mid_row;
                    iq4_vnni_rows_multi(kIq4Nl, dch, F, qm, x.n, kRows2, dr);
                    for (int t = 0; t < x.n; ++t) std::memcpy(dj[x.job[t]], dr + t * kRows2, sizeof(float) * kRows2);
                    continue;
                }
                const float * mids[8];
                for (int t = 0; t < x.n; ++t) mids[t] = r.mid.data() + ((size_t) u * 8 + t) * F;
                const uint8_t * d0 = dch;
                if (r8d) {   // R8 host copy: this chunk's down rows back to GGUF first
                    thread_local std::vector<uint8_t> un;
                    un.resize(kRows2 * db);
                    r8_unpack_nl_rows(down, D * db, F, D, r0, kRows2, un.data(), kRows2 * db);
                    d0 = un.data();
                }
                if (r.f64) {
                    const double * mids64[8];
                    for (int t = 0; t < x.n; ++t) mids64[t] = r.mid64.data() + ((size_t) u * 8 + t) * F;
                    double dv[8];
                    for (int row = r0; row < r0 + kRows2; ++row) {
                        iq4_cpu_row_f64(kIq4Nl, d0 + (row - r0) * db, F, x.n, mids64, dv);
                        for (int t = 0; t < x.n; ++t) dj[x.job[t]][row - r0] = (float) dv[t];
                    }
                    continue;
                }
                float dv[8];
                for (int row = r0; row < r0 + kRows2; ++row) {
                    iq4_cpu_row(kIq4Nl, d0 + (row - r0) * db, F, x.n, mids, dv);
                    for (int t = 0; t < x.n; ++t) dj[x.job[t]][row - r0] = dv[t];
                }
            }
            for (int j = 0; j < T * K; ++j) {
                if (!is_miss[j]) continue;
                float * acc = r.h_acc + (size_t) (j / K) * D + r0;
                for (int i = 0; i < kRows2; ++i) acc[i] += r.h_w[j] * dj[j][i];
            }
        };
        r.pool->run(D / kRows2, f2);
        if (r.perturb != 0.0f && r.perturb_skip-- == 0) {
            for (int i = 0; i < D; ++i) r.h_acc[i] *= 1.0f + r.perturb;
            r.perturb = 0;
        }
        r.stats.miss_ms += now_ms() - m0;
        // no H2D copy: a copy would queue on the copy engine behind the adaptive tier's expert uploads (13.6 MB
        // each) and stall every MoE layer; the kernel's own PCIe reads (T x 16 KB) are not queued behind them
        const int64_t n = (int64_t) T * D;
        add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(out, r.h_acc_dev, n);
        ck(cudaGetLastError(), "add");
    }
    r.stats.host_ms += now_ms() - t0;
}

}  // namespace strata::glm
