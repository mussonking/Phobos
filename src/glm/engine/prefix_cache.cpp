// src/glm/engine/prefix_cache.cpp - see prefix_cache.hpp.
#include "prefix_cache.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace strata::glm {
namespace {

constexpr size_t kPage = 64 << 10;

size_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    size_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v << 10;
    return 0;
}

}  // namespace

PrefixSegment::~PrefixSegment() {
    for (auto & r : rows) pc->release(r);
    if (pc) {
        --pc->stats.segments;
        // free the chain iteratively: a long conversation's chain must not recurse once per segment
        std::shared_ptr<PrefixSegment> p = std::move(parent);
        while (p && p.use_count() == 1) {
            std::shared_ptr<PrefixSegment> next = std::move(p->parent);
            p.reset();
            p = std::move(next);
        }
    }
}

PrefixAlloc PrefixCache::alloc(size_t bytes) {
    PrefixAlloc a;
    if (bytes == 0) return a;
    const int need = (int) ((bytes + kPage - 1) / kPage);
    const int per_block = (int) (opt.block_bytes / kPage);
    if (need > per_block) return a;
    auto take = [&](int bi) {
        Block & b = blocks[bi];
        if (b.free_pages < need) return false;
        int run = 0;
        for (int i = 0; i < per_block; ++i) {
            run = b.used[i] ? 0 : run + 1;
            if (run == need) {
                const int p0 = i + 1 - need;
                std::fill(b.used.begin() + p0, b.used.begin() + p0 + need, 1);
                b.free_pages -= need;
                a.p = b.p + (size_t) p0 * kPage;
                a.bytes = bytes;
                a.block = bi;
                a.page0 = p0;
                a.pages = need;
                stats.used += (size_t) need * kPage;
                return true;
            }
        }
        return false;
    };
    for (int bi = 0; bi < (int) blocks.size(); ++bi)
        if (take(bi)) return a;
    // a new block: a spare the background thread pinned, else pinned here
    char * p = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!spare.empty()) {
            p = spare.back();
            spare.pop_back();
        }
    }
    if (!p) p = pin_block();
    if (!p) return a;
    cv.notify_one();   // pin the next spare
    Block b;
    b.p = p;
    b.used.assign(per_block, 0);
    b.free_pages = per_block;
    blocks.push_back(std::move(b));
    take((int) blocks.size() - 1);
    return a;
}

char * PrefixCache::pin_block() {
    {
        std::lock_guard<std::mutex> lk(mu);
        if (pinned + opt.block_bytes > opt.cap_bytes) return nullptr;
        pinned += opt.block_bytes;   // reserved while pinning
    }
    // the pages are faulted in by touching them, outside CUDA and outside mmap_lock (MAP_POPULATE or
    // MADV_POPULATE_WRITE hold it: the engine thread's mmaps waited for the whole populate), then registered
    // (~0.08 s/GB under the driver's lock: other threads' CUDA calls wait that long; cudaHostAlloc holds it for the
    // whole 0.46 s/GB)
    const bool room = mem_available() >= opt.reserve_bytes + opt.block_bytes;
    void * m = room ? mmap(nullptr, opt.block_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
                    : MAP_FAILED;
    char * p = m == MAP_FAILED ? nullptr : static_cast<char *>(m);
    if (p)
        for (size_t o = 0; o < opt.block_bytes; o += 4096) static_cast<volatile char *>(p)[o] = 0;
    if (p && cudaHostRegister(p, opt.block_bytes, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        munmap(p, opt.block_bytes);
        p = nullptr;
    }
    if (!p) {
        std::lock_guard<std::mutex> lk(mu);
        pinned -= opt.block_bytes;
        if (!clamped)
            std::fprintf(stderr, "prefix cache: %s, effective cap %.2f GB (ceiling %.2f GB)\n",
                         room ? "pinning failed" : "MemAvailable reserve reached", pinned / 1e9,
                         opt.cap_bytes / 1e9);
        clamped = true;
        return nullptr;
    }
    return p;
}

void PrefixCache::pin_loop() {
    cudaSetDevice(device);
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
        cv.wait(lk, [&] {
            return stop || (spare.size() * opt.block_bytes < opt.spare_bytes && !clamped &&
                            pinned + opt.block_bytes <= opt.cap_bytes);
        });
        if (stop) return;
        lk.unlock();
        char * p = pin_block();
        lk.lock();
        if (p) spare.push_back(p);
    }
}

void PrefixCache::release(PrefixAlloc & a) {
    if (!a.p) return;
    Block & b = blocks[a.block];
    std::fill(b.used.begin() + a.page0, b.used.begin() + a.page0 + a.pages, 0);
    b.free_pages += a.pages;
    stats.used -= (size_t) a.pages * kPage;
    a = PrefixAlloc{};
}

void PrefixCache::drop(PrefixEntry * e) {
    auto it = std::find_if(entries.begin(), entries.end(), [e](const auto & x) { return x.get() == e; });
    if (it == entries.end()) return;
    // a save may still be writing it: its pages go back only once the copies are done
    if (e->done) {
        cudaEventSynchronize(e->done);
        cudaEventDestroy(e->done);
    }
    for (auto & k : e->kda) release(k);
    release(e->small);
    entries.erase(it);   // the segments follow when no other entry references them
    --stats.entries;
}

PrefixAlloc PrefixCache::alloc_evict(size_t bytes, const PrefixEntry * keep) {
    for (;;) {
        PrefixAlloc a = alloc(bytes);
        if (a.p || bytes == 0) return a;
        PrefixEntry * lru = nullptr;
        for (auto & e : entries)
            if (e.get() != keep && (!lru || e->used < lru->used)) lru = e.get();
        if (!lru) return a;
        drop(lru);
        ++stats.evictions;
    }
}

PrefixCachePtr prefix_cache_create(const PrefixCacheOptions & opt) {
    PrefixCachePtr pc(new PrefixCache, prefix_cache_destroy);
    pc->opt = opt;
    pc->opt.block_bytes = std::max(kPage, opt.block_bytes / kPage * kPage);
    pc->stats.cap = opt.cap_bytes;
    if (cudaStreamCreateWithFlags(&pc->st, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("prefix cache: stream");
    cudaGetDevice(&pc->device);
    if (opt.spare_bytes > 0) pc->pinner = std::thread([p = pc.get()] { p->pin_loop(); });
    const size_t avail = mem_available();
    const size_t eff = avail > opt.reserve_bytes ? std::min(opt.cap_bytes, avail - opt.reserve_bytes) : 0;
    std::fprintf(stderr, "prefix cache: ceiling %.2f GB, now %.2f GB pinnable (MemAvailable %.1f GB - reserve %.0f GB), "
                         "blocks of %zu MB pinned on use\n", opt.cap_bytes / 1e9, eff / 1e9, avail / 1e9,
                 opt.reserve_bytes / 1e9, pc->opt.block_bytes >> 20);
    return pc;
}

void prefix_cache_destroy(PrefixCache * pc) {
    if (!pc) return;
    {
        std::lock_guard<std::mutex> lk(pc->mu);
        pc->stop = true;
    }
    pc->cv.notify_all();
    if (pc->pinner.joinable()) pc->pinner.join();
    if (pc->st) cudaStreamSynchronize(pc->st);
    while (!pc->entries.empty()) pc->drop(pc->entries.back().get());
    auto unpin = [pc](char * p) {
        cudaHostUnregister(p);
        munmap(p, pc->opt.block_bytes);
    };
    for (auto & b : pc->blocks) unpin(b.p);
    for (char * p : pc->spare) unpin(p);
    if (pc->st) cudaStreamDestroy(pc->st);
    delete pc;
}

PrefixEntry * prefix_cache_lookup(PrefixCache & pc, const int32_t * ids, int64_t n) {
    PrefixEntry * best = nullptr;
    for (auto & e : pc.entries) {
        const int64_t P = (int64_t) e->toks.size();
        if (P > n || (best && P <= (int64_t) best->toks.size())) continue;
        if (std::equal(e->toks.begin(), e->toks.end(), ids)) best = e.get();
    }
    if (best) {
        best->used = ++pc.clock;
        ++pc.stats.hits;
    } else {
        ++pc.stats.misses;
    }
    return best;
}

int64_t prefix_cache_common(const PrefixCache & pc, const int32_t * ids, int64_t n) {
    int64_t best = 0;
    for (const auto & e : pc.entries) {
        const int64_t m = std::min<int64_t>(n, (int64_t) e->toks.size());
        if (m <= best) continue;
        best = std::max<int64_t>(best, std::mismatch(e->toks.begin(), e->toks.begin() + m, ids).first - e->toks.begin());
    }
    return best;
}

PrefixCacheStats prefix_cache_stats(const PrefixCache & pc) {
    PrefixCacheStats s = pc.stats;
    std::lock_guard<std::mutex> lk(pc.mu);
    s.pinned = pc.pinned;
    s.clamped = pc.clamped;
    return s;
}

}  // namespace strata::glm
