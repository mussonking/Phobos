#pragma once
// NUMA-aware host thread pool (moved out of routed.cu): the routed misses at decode and the CPU share of the prefill
// experts run on the same pinned workers.
#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::glm {

// NUMA topology: one logical CPU per physical core, per node (sysfs; one unpinned node if unavailable).
inline std::vector<int> parse_cpulist(const std::string & s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find_first_of(",\n", i);
        if (j == std::string::npos) j = s.size();
        const std::string r = s.substr(i, j - i);
        if (!r.empty()) {
            const size_t d = r.find('-');
            const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
            for (int c = a; c <= b; ++c) out.push_back(c);
        }
        i = j + 1;
    }
    return out;
}
inline std::string read_file(const std::string & path) {
    std::ifstream f(path);
    return f ? std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()) : std::string();
}
inline std::vector<std::vector<int>> numa_cores() {
    std::vector<std::vector<int>> nodes;
    for (int n = 0;; ++n) {
        const std::string list = read_file("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (list.empty()) break;
        std::vector<int> cores;
        for (int c : parse_cpulist(list)) {
            const std::vector<int> sib = parse_cpulist(
                read_file("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
            if (sib.empty() || sib[0] == c) cores.push_back(c);
        }
        nodes.push_back(cores);
    }
    return nodes;
}
// The physical cores of all the nodes (glm_serve's default --threads: a Pool of that size minus one, plus the caller,
// puts one thread on each core); the logical CPU count when sysfs has no NUMA topology.
inline int physical_cores(const std::vector<std::vector<int>> & nodes) {
    size_t n = 0;
    for (const auto & c : nodes) n += c.size();
    return n ? (int) n : (int) std::max(1u, std::thread::hardware_concurrency());
}
inline int physical_cores() { return physical_cores(numa_cores()); }

// The calling thread's next page allocations prefer NUMA node `node` (MPOL_PREFERRED: another node when it is full);
// -1 restores the default (local) policy. A pool item that first-touches chunk memory brackets the touch with it, so
// the pages land on the chunk's node even when a worker of the other node (or the caller) steals the item.
inline void numa_prefer(int node) {
    const unsigned long mask = node >= 0 ? 1ul << node : 0;
    syscall(SYS_set_mempolicy, node >= 0 ? 1 /* MPOL_PREFERRED */ : 0 /* MPOL_DEFAULT */, node >= 0 ? &mask : nullptr,
            node >= 0 ? 64 : 0);
}

// Work-sharing pool, NUMA-aware: workers are pinned one per physical core (core 0 left to the host loop), spread
// over the nodes. run(n, f) calls f(0..n-1); item i belongs to node i % n_nodes (the caller lays its chunks out so
// that chunk i's bytes live on that node), its workers drain their node's items first, then help the others. The
// caller drains too.
// Built for a loaded host (CI jobs share the cores): run returns when every ITEM is done, not when every worker has
// checked in, so a worker the scheduler preempted only costs the item it holds (none if it wakes late). Items are
// claimed by CAS on a per-node word tagged with the run's generation, so a late worker can never claim an item of
// another run. Idle workers spin for spin_us (the gap between decode calls) then sleep on a futex.
class Pool {
public:
    explicit Pool(int n) {
        if (const char * e = std::getenv("GLM_POOL_SPIN_US")) spin_us_ = std::atoi(e);
        std::vector<std::vector<int>> nodes = numa_cores();
        if (nodes.empty() || nodes[0].empty()) nodes = {{}};
        nn_ = (int) nodes.size();
        std::vector<std::pair<int, int>> place;   // (node, cpu), nodes interleaved, CPU 0 left to the host loop
        size_t longest = 0;
        for (const auto & c : nodes) longest = std::max(longest, c.size());
        for (size_t i = 0; i < longest; ++i)
            for (int k = 0; k < nn_; ++k)
                if (i < nodes[k].size() && nodes[k][i] != 0) place.emplace_back(k, nodes[k][i]);
        for (int i = 0; i < n; ++i) {
            const auto [node, cpu] = place.empty() ? std::pair<int, int>{0, -1} : place[i % place.size()];
            if (!place.empty() && i >= (int) place.size()) break;   // one worker per physical core at most
            th_.emplace_back([this, node = node, cpu = cpu] { loop(node, cpu); });
        }
        cpus_ = (int) th_.size();
    }
    ~Pool() {
        stop_.store(true);
        seq_.fetch_add(1, std::memory_order_seq_cst);
        futex_wake();
        for (auto & t : th_) t.join();
    }
    void run(int n, const std::function<void(int)> & f) {
        if (n <= 0) return;
        const uint64_t g = ++gen_;
        // close the counters to the previous generation before the task changes, then open them for this one
        for (int k = 0; k < nn_; ++k) next_[k].v.store(g << 32 | 0xffffffffu, std::memory_order_seq_cst);
        task_.store(&f, std::memory_order_seq_cst);
        n_.store(n, std::memory_order_seq_cst);
        done_.store(0, std::memory_order_seq_cst);
        for (int k = 0; k < nn_; ++k) next_[k].v.store(g << 32, std::memory_order_seq_cst);
        seq_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) futex_wake();
        work(0);
        while (done_.load(std::memory_order_acquire) < n) _mm_pause();
    }
    // Wake the sleeping workers now and keep them spinning for at least us microseconds, so the futex wake-up (tens of
    // us at decode) overlaps what the caller still waits for (the routing download) instead of the next run. Call it
    // from the thread that calls run, between runs: the workers find no item and go back to spinning.
    void prewake(int us) {
        spin_until_.store(now_us() + us, std::memory_order_relaxed);
        seq_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) > 0) futex_wake();
    }
    int size() const { return (int) th_.size() + 1; }
    int nodes() const { return nn_; }

private:
    static int64_t now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    void work(int home) {
        for (int d = 0; d < nn_; ++d) {
            const int k = (home + d) % nn_;
            for (;;) {
                uint64_t v = next_[k].v.load(std::memory_order_seq_cst);
                const uint32_t idx = (uint32_t) v;
                const int n = n_.load(std::memory_order_seq_cst);
                const int cnt = (n - k + nn_ - 1) / nn_;   // items k, k + nn, ...
                if ((int64_t) idx >= cnt) break;           // drained, closed, or a stale n (the CAS would fail)
                if (!next_[k].v.compare_exchange_weak(v, v + 1, std::memory_order_seq_cst)) continue;
                // claimed item idx of generation v >> 32: that run cannot return before the item is done, so its
                // task is still the current one
                (*task_.load(std::memory_order_seq_cst))(k + (int) idx * nn_);
                done_.fetch_add(1, std::memory_order_release);
            }
        }
    }
    void futex_wake() { syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0); }
    void loop(int node, int cpu) {
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        }
        uint32_t seen = 0;   // the sequence at construction: a worker that starts late still sees the first run
        for (;;) {
            uint32_t q;
            const int64_t t0 = now_us();
            int spins = 0;
            while ((q = seq_.load(std::memory_order_acquire)) == seen) {
                _mm_pause();
                if ((++spins & 255) == 0 && now_us() - t0 > spin_us_ &&
                    now_us() > spin_until_.load(std::memory_order_relaxed)) {
                    sleepers_.fetch_add(1, std::memory_order_seq_cst);
                    if (seq_.load(std::memory_order_seq_cst) == seen)
                        syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_), FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
                    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                }
            }
            seen = q;
            if (stop_.load()) return;
            work(node);
        }
    }
    struct alignas(64) Counter { std::atomic<uint64_t> v{0xffffffffu}; };
    std::vector<std::thread> th_;
    int nn_ = 1, cpus_ = 0;
    int spin_us_ = 100;
    uint64_t gen_ = 0;                   // caller side only
    alignas(64) std::atomic<uint32_t> seq_{0};   // futex word, bumped per run
    std::atomic<int> sleepers_{0};
    std::atomic<int64_t> spin_until_{0};   // prewake: workers keep spinning until then (steady clock, us)
    Counter next_[8];
    alignas(64) std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    std::atomic<const std::function<void(int)> *> task_{nullptr};
    std::atomic<int> n_{0};
};

}  // namespace strata::glm
