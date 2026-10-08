// src/glm/prefill/prefill_moe.cu - see prefill_moe.hpp.
#include "prefill_moe.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "pool.hpp"
#include "iq4_vnni.hpp"
#include "prefill_cpu.hpp"
#include "prefill_experts.hpp"

namespace strata::glm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("prefill_moe: ") + what + ": " + cudaGetErrorString(e));
}
size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// out[rows[i]] += acc[i]
__global__ void scatter_add_kernel(float * __restrict__ out, const float * __restrict__ acc,
                                   const int * __restrict__ rows, int D) {
    const int i = blockIdx.x;
    float * o = out + (size_t) rows[i] * D;
    const float * a = acc + (size_t) i * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) o[d] += a[d];
}

}  // namespace

struct PrefillMoe {
    const Model * m = nullptr;
    Routed * r = nullptr;
    Geometry g;
    PrefillMoeOptions opt;
    PrefillExperts * pe = nullptr;
    PrefillCpu * pc = nullptr;
    size_t blob = 0, gate_bytes = 0;
    cudaStream_t up_st = nullptr, dl_st = nullptr;
    cudaStream_t ex_st = nullptr;   // the routed GPU experts (the shared expert runs on the caller's stream meanwhile)
    cudaEvent_t ev_route = nullptr, ev_ids = nullptr, ev_y = nullptr, ev_ex = nullptr;
    std::vector<cudaEvent_t> ev_full, ev_free;   // per ring slot
    ExpertRef * d_table = nullptr;   // device [n_expert]
    ExpertRef * h_table = nullptr;   // pinned [n_expert]
    int32_t * h_ids = nullptr;       // pinned [max_T][K]
    float * h_w = nullptr;           // pinned [max_T][K]
    float * h_y = nullptr;           // pinned [max_T][D]
    float * h_e = nullptr;           // pinned [max_T][D]: the trunk workspace's embedding staging
    float * h_acc = nullptr;         // pinned [max_T][D] (compact CPU rows)
    int * h_rows = nullptr;          // pinned [max_T]: token of compact CPU row i
    // the loan (begin -> end), for cap = min(n, max_T) tokens
    void * loan = nullptr;
    size_t loan_bytes = 0;
    int cap = 0;
    uint8_t * ring = nullptr;        // ring_n blobs (the speculative uploads go to its first slots between layers)
    float * d_acc = nullptr;         // [cap][D]
    int * d_rows = nullptr;          // [cap]
    std::unique_ptr<Workspace> ws;   // trunk workspace for cap tokens
    void * scratch = nullptr;        // caller scratch
    // current layer (start -> finish)
    int il = -1, T = 0, n_res = 0, n_pre = 0, n_rows = 0;
    // speculative uploads (issued at finish for layer spec_il, while the GPU runs the trunk up to its routing)
    std::vector<float> seen;         // [n_layer + 1][n_expert] routed counts over the past prefills
    std::vector<int> spec_e;         // expert in ring slot k (layer spec_il)
    std::vector<int> pre_slot;       // ring slot of the routed speculative uploads (order n_res .. n_pre - 1)
    int spec_il = -1;
    int spec_cap = 0;                // opt.spec at create (the most spec slots a loan holds)
    int ring_n = 0, group_n = 0, spec_n = 0;   // this loan's upload slots, dequant group, spec slots
    int up_base = 0;                 // ring slot of the layer's first upload (after its speculative ones)
    cudaEvent_t ev_spec = nullptr, ev_acc = nullptr;
    // trunk window per layer: previous layer's experts join (ev_w0) -> routing (ev_w1), with the T it was timed at
    cudaEvent_t ev_w0 = nullptr, ev_w1 = nullptr;
    int w0_il = -1, w0_T = 0, last_T = 0;
    double last_win = 0;
    std::vector<double> win;
    std::vector<int> win_T;
    double model_cpu = 0;    // cost model CPU ms of the current layer (before calibration)
    double cpu_scale = 1.0;  // measured / modelled CPU time, running average
    std::vector<int> counts, order, cand, row_of;
    CpuExpertJobs jobs;
    PrefillMoeStats stats;
    // CPU share driver: runs a layer's CPU experts (after its activation download) while the caller thread enqueues the
    // GPU side, so the CPU starts as soon as the activations land
    std::thread cpu_th;
    std::mutex mu;
    std::condition_variable cv;
    bool cpu_go = false, cpu_done = true, cpu_quit = false;
    std::exception_ptr cpu_err;
    double cpu_t0 = 0, cpu_t1 = 0, cpu_t2 = 0;
};

namespace {

void cpu_driver(PrefillMoe * p) {
    // the pool leaves CPU 0 to the host loop: the driver takes it (unpinned, it could share a worker's core)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv.wait(lk, [p] { return p->cpu_go || p->cpu_quit; });
            if (p->cpu_quit) return;
            p->cpu_go = false;
        }
        try {
            Pool & pool = routed_pool(*p->r);
            p->cpu_t0 = now_ms();
            ck(cudaEventSynchronize(p->ev_y), "y");
            p->cpu_t1 = now_ms();
            nvtxRangePushA("cpu experts");
            prefill_cpu_quantize(p->pc, pool, p->h_y, p->n_rows, p->h_rows);
            prefill_cpu_run(p->pc, pool, p->jobs, p->n_rows, p->h_acc);
            nvtxRangePop();
            p->cpu_t2 = now_ms();
        } catch (...) {
            p->cpu_err = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lk(p->mu);
            p->cpu_done = true;
        }
        p->cv.notify_all();
    }
}

void cpu_wait(PrefillMoe * p) {
    std::unique_lock<std::mutex> lk(p->mu);
    p->cv.wait(lk, [p] { return p->cpu_done; });
    if (p->cpu_err) {
        std::exception_ptr e = p->cpu_err;
        p->cpu_err = nullptr;
        std::rethrow_exception(e);
    }
}

}  // namespace

PrefillMoe * prefill_moe_create(const Model & m, Routed & r, const PrefillMoeOptions & opt, cudaStream_t st) {
    const Geometry & g = m.g;
    if (opt.ring < 2 * opt.group) throw std::runtime_error("prefill_moe: ring must be >= 2 * group");
    auto * p = new PrefillMoe;
    p->m = &m;
    p->r = &r;
    p->g = g;
    p->opt = opt;
    p->blob = routed_blob_bytes(g);
    p->gate_bytes = (size_t) g.ffn_expert * (g.n_embd / 256) * 136;
    const int K = g.n_expert_used, D = g.n_embd, NE = g.n_expert, T = opt.max_T;
    p->pe = prefill_experts_create_unbound(g, opt.group);
    // the CPU share needs AVX512 VNNI (cpu_isa(), GLM_CPU_ISA): without it every non-resident expert is uploaded
    if (iq4_vnni_available()) {
        p->pc = prefill_cpu_create(g, T, T * K);
        prefill_cpu_set_r8_tiles(p->pc, opt.r8_tiles);
    } else {
        p->opt.cpu = false;
        p->opt.upload = true;
    }
    ck(cudaStreamCreateWithFlags(&p->up_st, cudaStreamNonBlocking), "upload stream");
    ck(cudaStreamCreateWithFlags(&p->dl_st, cudaStreamNonBlocking), "download stream");
    ck(cudaStreamCreateWithFlags(&p->ex_st, cudaStreamNonBlocking), "experts stream");
    for (cudaEvent_t * e : {&p->ev_route, &p->ev_ids, &p->ev_y, &p->ev_ex, &p->ev_spec, &p->ev_acc})
        ck(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
    ck(cudaEventCreate(&p->ev_w0), "event");
    ck(cudaEventCreate(&p->ev_w1), "event");
    p->spec_cap = std::max(0, opt.spec);
    const int slots = std::max(opt.ring, p->spec_cap + opt.group);
    p->ev_full.resize(slots);
    p->ev_free.resize(slots);
    for (int s = 0; s < slots; ++s) {
        ck(cudaEventCreateWithFlags(&p->ev_full[s], cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&p->ev_free[s], cudaEventDisableTiming), "event");
    }
    ck(cudaMalloc(&p->d_table, NE * sizeof(ExpertRef)), "table");
    ck(cudaMallocHost(&p->h_table, NE * sizeof(ExpertRef)), "pinned table");
    ck(cudaMallocHost(&p->h_ids, (size_t) T * K * 4), "pinned ids");
    ck(cudaMallocHost(&p->h_w, (size_t) T * K * 4), "pinned w");
    ck(cudaMallocHost(&p->h_y, (size_t) T * D * 4), "pinned y");
    ck(cudaMallocHost(&p->h_e, (size_t) T * D * 4), "pinned embeddings");
    ck(cudaMallocHost(&p->h_acc, (size_t) T * D * 4), "pinned acc");
    ck(cudaMallocHost(&p->h_rows, (size_t) T * 4), "pinned rows");
    p->counts.resize(NE);
    p->row_of.assign(T, -1);
    p->seen.assign((size_t) (g.n_layer + 1) * NE, 0.0f);
    p->win.assign(g.n_layer + 1, 0.0);
    p->win_T.assign(g.n_layer + 1, 0);
    p->cpu_th = std::thread(cpu_driver, p);
    return p;
}

void prefill_moe_destroy(PrefillMoe * p, cudaStream_t st) {
    if (!p) return;
    prefill_moe_end(p, st);
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->cpu_quit = true;
    }
    p->cv.notify_all();
    p->cpu_th.join();
    prefill_experts_destroy(p->pe);
    prefill_cpu_destroy(p->pc);
    for (cudaEvent_t e : {p->ev_route, p->ev_ids, p->ev_y, p->ev_ex, p->ev_spec, p->ev_acc, p->ev_w0, p->ev_w1})
        cudaEventDestroy(e);
    for (cudaEvent_t e : p->ev_full) cudaEventDestroy(e);
    for (cudaEvent_t e : p->ev_free) cudaEventDestroy(e);
    cudaStreamDestroy(p->up_st);
    cudaStreamDestroy(p->dl_st);
    cudaStreamDestroy(p->ex_st);
    cudaFree(p->d_table);
    for (void * q : {(void *) p->h_table, (void *) p->h_ids, (void *) p->h_w, (void *) p->h_y, (void *) p->h_e,
                     (void *) p->h_acc, (void *) p->h_rows})
        cudaFreeHost(q);
    delete p;
}

PrefillLoanBytes prefill_moe_loan_bytes(const PrefillMoe * p, int64_t n, size_t scratch_bytes) {
    const int cap = (int) std::min<int64_t>(std::max<int64_t>(n, 1), p->opt.max_T);
    PrefillLoanBytes b;
    b.cap = cap;
    // a small prompt uploads few experts (few routed tokens each): a smaller dequant group and ring
    b.group = std::clamp(cap / 32, std::min(2, p->opt.group), p->opt.group);
    // speculative uploads: the trunk window grows with cap (measured ~1 upload per 110 tokens; ~1 per 96 kept);
    // smaller chunks: too short
    b.spec_slots = cap >= 1024 ? std::min(p->spec_cap, cap / 96) : 0;
    b.ring_slots = std::max(std::clamp(cap / 8, 2 * b.group, p->opt.ring), b.spec_slots + b.group);
    b.ring = align256((size_t) b.ring_slots * p->blob);
    b.experts = prefill_experts_bytes(p->g, cap, b.group) + align256((size_t) cap * p->g.n_embd * 4) +
                align256((size_t) cap * 4);
    b.workspace = workspace_bytes(*p->m, cap);
    b.scratch = align256(scratch_bytes);
    b.total = b.ring + b.experts + b.workspace + b.scratch;
    return b;
}

void prefill_moe_begin(PrefillMoe * p, int64_t n, size_t scratch_bytes, cudaStream_t st) {
    if (p->loan) throw std::runtime_error("prefill_moe_begin: a prefill is already open");
    const PrefillLoanBytes b = prefill_moe_loan_bytes(p, n, scratch_bytes);
    const double t0 = now_ms();
    p->loan = routed_lend(*p->r, b.total, st);
    p->loan_bytes = b.total;
    p->cap = b.cap;
    char * c = static_cast<char *>(p->loan);
    p->ring = reinterpret_cast<uint8_t *>(c);
    p->ring_n = b.ring_slots;
    p->group_n = b.group;
    p->spec_n = b.spec_slots;
    p->spec_il = -1;
    p->w0_il = -1;
    c += b.ring;
    const size_t pe_bytes = prefill_experts_bytes(p->g, b.cap, b.group);
    prefill_experts_bind(p->pe, b.cap, c, st, b.group);
    c += pe_bytes;
    p->d_acc = reinterpret_cast<float *>(c);
    c += align256((size_t) b.cap * p->g.n_embd * 4);
    p->d_rows = reinterpret_cast<int *>(c);
    c += align256((size_t) b.cap * 4);
    p->ws = workspace_create(*p->m, b.cap, c, p->h_e);
    c += b.workspace;
    p->scratch = scratch_bytes ? c : nullptr;
    // the ring slots start free
    for (cudaEvent_t e : p->ev_free) ck(cudaEventRecord(e, st), "slot free");
    ck(cudaStreamSynchronize(st), "begin");
    p->stats.lend_ms += now_ms() - t0;
}

void prefill_moe_end(PrefillMoe * p, cudaStream_t st) {
    if (!p->loan) return;
    const double t0 = now_ms();
    try {
        cpu_wait(p);   // a layer left started by an exception
    } catch (...) {
    }
    ck(cudaStreamSynchronize(st), "end");
    ck(cudaStreamSynchronize(p->ex_st), "end experts");
    ck(cudaStreamSynchronize(p->up_st), "end upload");
    ck(cudaStreamSynchronize(p->dl_st), "end download");
    p->ws.reset();
    prefill_experts_bind(p->pe, 0, nullptr, st);
    if (p->opt.async_reclaim) routed_reclaim_async(*p->r, st);
    else routed_reclaim(*p->r, st);
    p->loan = nullptr;
    p->loan_bytes = 0;
    p->cap = 0;
    p->ring = nullptr;
    p->spec_il = -1;
    p->w0_il = -1;
    p->d_acc = nullptr;
    p->d_rows = nullptr;
    p->scratch = nullptr;
    p->stats.reclaim_ms += now_ms() - t0;
}

Workspace & prefill_moe_workspace(PrefillMoe * p) {
    if (!p->ws) throw std::runtime_error("prefill_moe_workspace: no open prefill");
    return *p->ws;
}
void * prefill_moe_scratch(PrefillMoe * p) { return p->scratch; }
size_t prefill_moe_device_bytes(const PrefillMoe * p) { return p->g.n_expert * sizeof(ExpertRef); }
PrefillMoeStats & prefill_moe_stats(PrefillMoe * p) { return p->stats; }
const PrefillMoeOptions & prefill_moe_options(const PrefillMoe * p) { return p->opt; }
void prefill_moe_set_options(PrefillMoe * p, const PrefillMoeOptions & opt) {
    const int max_T = p->opt.max_T, group = p->opt.group, ring = p->opt.ring;
    p->opt = opt;
    p->opt.max_T = max_T;
    p->opt.group = group;
    p->opt.ring = ring;
    p->opt.spec = std::min(p->spec_cap, opt.spec);   // the loan has room for the created count
    if (p->pc) prefill_cpu_set_r8_tiles(p->pc, opt.r8_tiles);
    else {   // no CPU share on this CPU
        p->opt.cpu = false;
        p->opt.upload = true;
    }
}

void prefill_moe_start(PrefillMoe * p, int il, const float * y, const int32_t * ids, const float * w, float * out,
                       int T, cudaStream_t st) {
    const Geometry & g = p->g;
    const int K = g.n_expert_used, D = g.n_embd, NE = g.n_expert;
    const PrefillMoeOptions & o = p->opt;
    if (T < 1 || T > p->cap) throw std::runtime_error("prefill_moe_start: T out of range (or no open prefill)");
    p->il = il;
    p->T = T;

    // routing (and, for the CPU share, the activations) to the host on the download stream
    ck(cudaEventRecord(p->ev_w1, st), "window end");
    ck(cudaEventRecord(p->ev_route, st), "route event");
    ck(cudaStreamWaitEvent(p->dl_st, p->ev_route, 0), "route wait");
    ck(cudaMemcpyAsync(p->h_ids, ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, p->dl_st), "ids d2h");
    ck(cudaMemcpyAsync(p->h_w, w, (size_t) T * K * 4, cudaMemcpyDeviceToHost, p->dl_st), "w d2h");
    ck(cudaEventRecord(p->ev_ids, p->dl_st), "ids event");
    if (o.cpu) {
        ck(cudaMemcpyAsync(p->h_y, y, (size_t) T * D * 4, cudaMemcpyDeviceToHost, p->dl_st), "y d2h");
        ck(cudaEventRecord(p->ev_y, p->dl_st), "y event");
    }
    const double t0 = now_ms();
    nvtxRangePushA("routing wait");
    ck(cudaEventSynchronize(p->ev_ids), "ids");
    nvtxRangePop();
    p->stats.plan_wait_ms += now_ms() - t0;
    nvtxRangePushA("plan");
    // trunk window since the previous layer's experts joined (both events are behind the routing download)
    if (p->w0_il == il - 1 && p->w0_T == T) {
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, p->ev_w0, p->ev_w1), "window");
        p->win[il] = ms;
        p->win_T[il] = T;
        p->last_win = ms;
        p->last_T = T;
        p->stats.window_ms += ms;
    }
    p->w0_il = -1;

    // plan: resident experts, then the non-resident ones by descending count, split between upload and CPU
    std::fill(p->counts.begin(), p->counts.end(), 0);
    for (int j = 0; j < T * K; ++j) {
        const int e = p->h_ids[j];
        if (e >= 0 && e < NE) ++p->counts[e];
    }
    p->order.clear();
    p->cand.clear();
    for (int e = 0; e < NE; ++e) {
        p->h_table[e] = ExpertRef{nullptr, nullptr, 23, 20, nullptr, -1};
        if (!p->counts[e]) continue;
        const ExpertRef ref = routed_slot_ref(*p->r, il, e);
        if (ref.gate_up) {
            p->h_table[e] = ref;
            p->order.push_back(e);
            p->stats.res_jobs += p->counts[e];
        } else p->cand.push_back(e);
    }
    p->n_res = (int) p->order.size();
    // the speculative uploads of this layer that got routed: already on the GPU
    int n_spec = 0;
    p->pre_slot.clear();
    p->up_base = 0;
    if (p->spec_il == il) {
        // the uploads take the ring after the speculative slots: a slot holding a routed one is reused >= group
        // orders later (spec_e.size() <= ring_n - group), so its dequant (after) is enqueued before that upload
        p->up_base = (int) p->spec_e.size();
        for (size_t k = 0; k < p->spec_e.size(); ++k) {
            const int e = p->spec_e[k];
            if (!p->counts[e] || p->h_table[e].gate_up) continue;
            uint8_t * slot = p->ring + k * p->blob;
            p->h_table[e].gate_up = slot;
            p->h_table[e].down = slot + 2 * p->gate_bytes;
            p->order.push_back(e);
            p->pre_slot.push_back((int) k);
            p->stats.spec_jobs += p->counts[e];
            ++n_spec;
        }
        p->stats.spec_experts += n_spec;
        p->stats.spec_wasted += (int64_t) p->spec_e.size() - n_spec;
        p->cand.erase(std::remove_if(p->cand.begin(), p->cand.end(), [p](int e) { return p->h_table[e].gate_up; }),
                      p->cand.end());
    }
    p->n_pre = (int) p->order.size();
    for (int e = 0; e < NE; ++e) p->seen[(size_t) il * NE + e] += p->counts[e];
    std::stable_sort(p->cand.begin(), p->cand.end(), [&](int a, int b) { return p->counts[a] > p->counts[b]; });
    const int nc = (int) p->cand.size();
    int n_up = nc;
    double t_up = nc * o.upload_cost_ms, t_cpu = 0;
    if (!o.upload) {
        n_up = 0;
    } else if (o.cpu) {
        // n_up minimizing max(PCIe time of the uploads, CPU time of the rest)
        const double k = p->cpu_scale;
        double rest = 0;
        for (int e : p->cand) rest += k * (o.cpu_expert_ms + p->counts[e] * o.cpu_job_ms);
        double best = rest;
        n_up = 0;
        for (int i = 0; i < nc; ++i) {
            rest -= k * (o.cpu_expert_ms + p->counts[p->cand[i]] * o.cpu_job_ms);
            const double m = std::max((i + 1) * o.upload_cost_ms + o.upload_tail_ms, rest);
            if (m < best) {
                best = m;
                n_up = i + 1;
            }
        }
    }
    t_up = n_up ? n_up * o.upload_cost_ms + o.upload_tail_ms : 0;
    for (int i = n_up; i < nc; ++i) t_cpu += o.cpu_expert_ms + p->counts[p->cand[i]] * o.cpu_job_ms;
    p->model_cpu = t_cpu;
    p->stats.model_up_ms += t_up;
    p->stats.model_cpu_ms += t_cpu * p->cpu_scale;
    for (int i = 0; i < n_up; ++i) {
        const int e = p->cand[i];
        uint8_t * slot = p->ring + (size_t) ((p->up_base + i) % p->ring_n) * p->blob;
        p->h_table[e].gate_up = slot;
        p->h_table[e].down = slot + 2 * p->gate_bytes;
        p->order.push_back(e);
        p->stats.up_jobs += p->counts[e];
    }
    ++p->stats.calls;
    p->stats.jobs += (int64_t) T * K;
    p->stats.res_experts += p->n_res;
    p->stats.up_experts += n_up;
    p->stats.cpu_experts += nc - n_up;

    // CPU jobs: compact rows (tokens with a CPU expert, ascending), grouped by expert (rows ascending inside)
    p->jobs.clear();
    p->n_rows = 0;
    if (n_up < nc) {
        std::vector<int> slot(NE, -1);
        for (int i = n_up; i < nc; ++i) {
            const int e = p->cand[i];
            slot[e] = i - n_up;
            p->jobs.refs.push_back(routed_host_expert(*p->r, il, e));
        }
        const int U = nc - n_up;
        std::vector<int> start(U + 1, 0);
        for (int j = 0; j < T * K; ++j) {
            const int e = p->h_ids[j];
            if (e >= 0 && e < NE && slot[e] >= 0) ++start[slot[e] + 1];
        }
        for (int u = 0; u < U; ++u) start[u + 1] += start[u];
        p->jobs.start = start;
        p->jobs.row.resize(start[U]);
        p->jobs.wt.resize(start[U]);
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (int t = 0; t < T; ++t) {
            int row = -1;
            for (int k = 0; k < K; ++k) {
                const int e = p->h_ids[t * K + k];
                if (e < 0 || e >= NE || slot[e] < 0) continue;
                if (row < 0) {
                    row = p->n_rows++;
                    p->h_rows[row] = t;
                }
                const int q = fill[slot[e]]++;
                p->jobs.row[q] = row;
                p->jobs.wt[q] = p->h_w[t * K + k];
            }
        }
        p->stats.cpu_jobs += start[U];
    }

    nvtxRangePop();
    // CPU share: the driver thread starts it when the activations land
    if (p->jobs.experts() > 0) {
        {
            std::lock_guard<std::mutex> lk(p->mu);
            p->cpu_done = false;
            p->cpu_go = true;
        }
        p->cv.notify_all();
    }

    // GPU, on the experts stream (forked after the routing): table, then the planned experts with the uploads
    // interleaved (ring slot reuse waits for its dequant)
    cudaStream_t xs = p->ex_st;
    ck(cudaStreamWaitEvent(xs, p->ev_route, 0), "experts fork");
    ck(cudaMemcpyAsync(p->d_table, p->h_table, NE * sizeof(ExpertRef), cudaMemcpyHostToDevice, xs), "table");
    PrefillExpertsPlan plan;
    plan.counts = p->counts.data();
    plan.order = p->order.data();
    plan.n_order = (int) p->order.size();
    plan.before = [p, il, xs](int i) {
        if (i < p->n_res) return;
        if (i < p->n_pre) {
            if (i == p->n_res) ck(cudaStreamWaitEvent(xs, p->ev_spec, 0), "spec wait");
            return;
        }
        const int s = (p->up_base + i - p->n_pre) % p->ring_n;
        const int e = p->order[i];
        const ExpertRef src = routed_host_expert(*p->r, il, e);
        uint8_t * dst = p->ring + (size_t) s * p->blob;
        ck(cudaStreamWaitEvent(p->up_st, p->ev_free[s], 0), "slot free wait");
        routed_upload_expert(*p->r, src, dst, p->up_st);   // GGUF in the slot whatever the host layout
        ck(cudaEventRecord(p->ev_full[s], p->up_st), "slot full");
        ck(cudaStreamWaitEvent(xs, p->ev_full[s], 0), "slot full wait");
    };
    plan.after = [p, xs](int i) {
        if (i < p->n_res) return;
        const int s = i < p->n_pre ? p->pre_slot[i - p->n_res] : (p->up_base + i - p->n_pre) % p->ring_n;
        ck(cudaEventRecord(p->ev_free[s], xs), "slot free");
    };
    prefill_experts_run(p->pe, T, y, ids, w, p->d_table, plan, out, xs);
    ck(cudaEventRecord(p->ev_ex, xs), "experts done");
}

namespace {

// Speculative uploads for layer il + 1: its most routed non-resident experts so far, into the first ring slots, while
// the GPU runs the trunk up to that layer's routing (PCIe idle). As many as fit spec_window of that layer's trunk
// window (timed at an earlier chunk of the same T, else the last window of this chunk). The ring is free once layer
// il's GPU experts are done; the uploads start after the CPU rows' upload (on the critical path, it must not share
// the PCIe).
void spec_upload(PrefillMoe * p, int il, int T, bool after_acc) {
    p->spec_il = -1;
    p->spec_e.clear();
    const int nx = il + 1, NE = p->g.n_expert;
    int spec = std::min(p->opt.spec, p->spec_n);
    if (spec <= 0 || !p->opt.upload || nx >= p->g.n_layer) return;
    const double win = p->win_T[nx] == T ? p->win[nx] : p->last_T == T ? p->last_win : 0.0;
    spec = std::min(spec, (int) (win * p->opt.spec_window / p->opt.upload_ms));
    if (spec <= 0) return;
    // at large T nearly every expert gets routed: the never seen ones too (after the seen ones)
    const bool all = (int64_t) T * p->g.n_expert_used >= 8 * (int64_t) NE;
    const float * seen = p->seen.data() + (size_t) nx * NE;
    std::vector<int> c;
    for (int e = 0; e < NE; ++e)
        if ((all || seen[e] > 0) && !routed_slot_ref(*p->r, nx, e).gate_up) c.push_back(e);
    const int n = std::min<int>(spec, (int) c.size());
    if (!n) return;
    std::partial_sort(c.begin(), c.begin() + n, c.end(), [seen](int a, int b) { return seen[a] > seen[b]; });
    ck(cudaStreamWaitEvent(p->up_st, p->ev_ex, 0), "spec slots free");
    if (after_acc) ck(cudaStreamWaitEvent(p->up_st, p->ev_acc, 0), "spec after acc");
    for (int k = 0; k < n; ++k) {
        routed_upload_expert(*p->r, routed_host_expert(*p->r, nx, c[k]), p->ring + (size_t) k * p->blob, p->up_st);
        p->spec_e.push_back(c[k]);
    }
    ck(cudaEventRecord(p->ev_spec, p->up_st), "spec event");
    p->spec_il = nx;
}

}  // namespace

void prefill_moe_finish(PrefillMoe * p, int il, float * out, int T, cudaStream_t st) {
    if (il != p->il || T != p->T) throw std::runtime_error("prefill_moe_finish: not the started layer");
    ck(cudaStreamWaitEvent(st, p->ev_ex, 0), "experts join");
    if (p->jobs.experts() > 0) {
        cpu_wait(p);
        p->stats.cpu_wait_ms += p->cpu_t1 - p->cpu_t0;
        p->stats.cpu_ms += p->cpu_t2 - p->cpu_t1;
        // calibrate the CPU side of the cost model on layers with enough work to time
        if (p->opt.calibrate && p->model_cpu > 5.0)
            p->cpu_scale = 0.8 * p->cpu_scale + 0.2 * (p->cpu_t2 - p->cpu_t1) / p->model_cpu;
        const int D = p->g.n_embd;
        ck(cudaMemcpyAsync(p->d_rows, p->h_rows, (size_t) p->n_rows * 4, cudaMemcpyHostToDevice, st), "rows h2d");
        ck(cudaMemcpyAsync(p->d_acc, p->h_acc, (size_t) p->n_rows * D * 4, cudaMemcpyHostToDevice, st), "acc h2d");
        ck(cudaEventRecord(p->ev_acc, st), "acc event");
        scatter_add_kernel<<<p->n_rows, 256, 0, st>>>(out, p->d_acc, p->d_rows, D);
        ck(cudaGetLastError(), "scatter add");
    }
    ck(cudaEventRecord(p->ev_w0, st), "window start");
    p->w0_il = il;
    p->w0_T = T;
    spec_upload(p, il, T, p->jobs.experts() > 0);
}

}  // namespace strata::glm
