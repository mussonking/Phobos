#pragma once
// Page-locked host memory budget of glm_serve when the expert arena is not pinned. WSL2 caps a process's page-locked
// memory at 1 GiB in total (cudaHostAlloc, cudaMallocHost and cudaHostRegister all count; measured on Windows 10 22H2,
// WSL 3.0.1, driver 616.92), so the other pinned buffers must fit under it: the MLA caches' mapped host rows (per
// session, per context token), the session snapshots, the prefill's chunk buffers, the bounce ring. Pure arithmetic
// (glm_pinned_budget_test); glm_serve fills PinnedNeeds from the model and clamps its options with pinned_budget_fit.
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <string>

namespace strata::glm {

struct PinnedNeeds {
    size_t per_session = 0;       // a session at its context: MLA mapped rows, its pinned staging
    size_t per_snapshot = 0;      // one session snapshot (KDA states + an h row)
    size_t per_chunk_token = 0;   // prefill_moe's pinned buffers per chunk token
    size_t fixed = 0;             // the rest: bounce ring, routed and prefill tables
};

struct PinnedPlan {
    int sessions = 0, snapshots = 0, chunk = 0;   // snapshots per session; chunk 0 = no prefill
    size_t bytes = 0;
    bool fits = false;
};

inline size_t pinned_total(const PinnedNeeds & n, int sessions, int snapshots, int chunk) {
    return n.fixed + (size_t) sessions * (n.per_session + (size_t) snapshots * n.per_snapshot) +
           (size_t) chunk * n.per_chunk_token;
}

// Lowers the snapshots per session (to 1), then the prefill chunk (halving, to 64), then the sessions (to 1) until
// the total fits budget; then gives back what fits again, chunk first (doubling, up to the asked chunk), then
// snapshots. fits = false: 1 session x 1 snapshot x 64-token chunks does not fit (the context alone is too big).
inline PinnedPlan pinned_budget_fit(const PinnedNeeds & n, size_t budget, int sessions, int snapshots, int chunk) {
    PinnedPlan p{std::max(1, sessions), std::max(1, snapshots), std::max(0, chunk), 0, false};
    const auto total = [&] { return pinned_total(n, p.sessions, p.snapshots, p.chunk); };
    while (total() > budget && p.snapshots > 1) --p.snapshots;
    while (total() > budget && p.chunk > 64) p.chunk = std::max(64, p.chunk / 2);
    while (total() > budget && p.sessions > 1) --p.sessions;
    if (total() <= budget) {
        while (p.chunk && p.chunk * 2 <= chunk && pinned_total(n, p.sessions, p.snapshots, p.chunk * 2) <= budget)
            p.chunk *= 2;
        while (p.snapshots < snapshots && pinned_total(n, p.sessions, p.snapshots + 1, p.chunk) <= budget)
            ++p.snapshots;
    }
    p.bytes = total();
    p.fits = p.bytes <= budget;
    return p;
}

inline bool host_is_wsl() {
    std::ifstream f("/proc/version");
    std::string v;
    std::getline(f, v);
    return v.find("icrosoft") != std::string::npos;
}

// GLM_PINNED_BUDGET_MB if set (0 = no clamp), else 900 MB under WSL2 (its 1 GiB minus CUDA's own), else 0
inline size_t pinned_budget_bytes() {
    if (const char * e = std::getenv("GLM_PINNED_BUDGET_MB"); e && *e) return (size_t) std::max(0L, std::atol(e)) << 20;
    return host_is_wsl() ? (size_t) 900 << 20 : 0;
}

// The host expert arena is filled whole (and mostly page-locked) at routed_load: one past MemAvailable minus a margin
// would push the host into reclaim and swap under pinned pages (two full 169 GB arenas took 338 GB of a 376 GB host,
// Oct 7). /proc/meminfo MemAvailable in bytes, 0 when unreadable (no guard then).
inline size_t mem_available_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t kb = 0;
    while (f >> key >> kb) {
        if (key == "MemAvailable:") return kb * 1024;
        f.ignore(256, '\n');
    }
    return 0;
}

// GLM_ARENA_MARGIN_GB (GB of 1e9 bytes), default 4
inline size_t arena_margin_bytes() {
    const char * e = std::getenv("GLM_ARENA_MARGIN_GB");
    return (size_t) (std::max(0.0, e && *e ? std::atof(e) : 4.0) * 1e9);
}

// avail 0 = MemAvailable unknown: no refusal
inline bool arena_fits(size_t arena_bytes, size_t margin, size_t avail) {
    return !avail || arena_bytes + margin <= avail;
}

}  // namespace strata::glm
