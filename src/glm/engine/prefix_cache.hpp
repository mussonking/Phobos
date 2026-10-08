#pragma once
// Host-RAM prefix cache (docs/prefix-cache.md): conversation states saved from any session and restored into any
// session of the same model and options (session_save / session_load in session.hpp), shared by all sessions.
//
// An entry is the state after P tokens: the MLA rows as a chain of segments (rows blobs [a, b) of every MLA cache,
// mla_cache_export_rows), the key | gate tails at P, every KDA layer's state, hcat row 0 (h[P - 1]), the tokens and
// the pending token. Rows < P never change once written, so the entries of one conversation share their segments:
// a save copies only the rows past the session's own chain (the segments it restored or saved last). Segments are
// refcounted (shared_ptr): evicting an entry frees the segments no other entry uses.
//
// Memory: pinned host blocks of block_bytes, pinned lazily up to cap_bytes, and never past MemAvailable -
// reserve_bytes at the time of pinning (the effective cap is logged when that clamps). A block is mmap'd, its pages
// faulted in by touching them (~0.6 s/GB; the faults take per-VMA locks only: MAP_POPULATE held the process's
// mmap_lock throughout, so every mmap / brk of the engine thread, a large vector's malloc, waited ~90 ms per 256 MB),
// then cudaHostRegister'd (~0.08 s/GB holding the driver's lock and mmap_lock: the other threads' CUDA calls and mmaps
// wait up to ~6 ms per 64 MB block; cudaHostAlloc would hold the driver's lock for the whole 0.46 s/GB, 250 ms stalls
// measured). A background thread keeps spare_bytes pinned ahead of use (from create on): a save takes those
// blocks without waiting (with none ready it pins on the spot). Blocks are cut into 64 KB
// pages; every blob is one first-fit run of pages. A save that does not fit evicts the least recently used entries;
// if it still does not fit, it is skipped. The blocks are unpinned at destroy.
//
// Streams: every write into the cache memory is a D2H on the cache's own stream (saves never wait on the host);
// imports read on the session's stream and synchronize before returning, so a freed page is never read again.
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::glm {

struct PrefixCacheOptions {
    size_t  cap_bytes = 0;                   // ceiling of pinned memory
    size_t  reserve_bytes = 64ull << 30;     // keep MemAvailable >= this after each block is pinned
    size_t  block_bytes = 64ull << 20;       // bounds the register stall; >= the largest blob (~17 MB)
    int64_t segment_max = 16384;             // rows per segment (bounds one blob at ~17 MB per MLA cache)
    size_t  spare_bytes = 1ull << 30;        // pinned ahead by the background thread (0: pin on use)
};

struct PrefixCacheStats {
    size_t  cap = 0, pinned = 0, used = 0;   // bytes: ceiling, pinned blocks (spares included), pages in use
    int64_t entries = 0, segments = 0;
    int64_t hits = 0, misses = 0;            // prefix_cache_lookup with / without an entry
    int64_t saves = 0, save_skips = 0, evictions = 0;
    int64_t tokens_restored = 0;             // rows imported by session_load (past the rows it kept)
    size_t  bytes_saved = 0, bytes_restored = 0;
    bool    clamped = false;                 // MemAvailable stopped the pinning below cap
};

struct PrefixCache;
struct PrefixAlloc {
    char * p = nullptr;
    size_t bytes = 0;
    int    block = -1, page0 = 0, pages = 0;
};

struct PrefixSegment {
    PrefixCache * pc = nullptr;
    int64_t a = 0, b = 0;                    // rows [a, b); the parent holds the rows below a
    std::shared_ptr<PrefixSegment> parent;
    std::vector<PrefixAlloc> rows;           // one rows blob per MLA cache (trunk layers in order, then the MTP)
    ~PrefixSegment();
};

struct PrefixEntry {
    std::vector<int32_t> toks;               // the P fed tokens
    int32_t  next = -1;                      // pending token
    std::shared_ptr<PrefixSegment> last;     // rows [0, P) (null when P == 0)
    std::vector<PrefixAlloc> kda;            // one state per KDA layer
    PrefixAlloc small;                       // the tails (mla_cache_tail_bytes each, same cache order), then hcat row 0
    cudaEvent_t done = nullptr;              // the save's copies are complete when this event is
    uint64_t used = 0;                       // LRU clock
    size_t   bytes = 0;                      // pages owned by the entry alone (kda + small)
};

struct PrefixCache {
    PrefixCacheOptions opt;
    cudaStream_t st = nullptr;               // the saves' stream
    std::vector<std::unique_ptr<PrefixEntry>> entries;
    PrefixCacheStats stats;
    uint64_t clock = 0;
    struct Block {
        char * p = nullptr;
        std::vector<uint8_t> used;           // per page
        int free_pages = 0;
    };
    std::vector<Block> blocks;
    // background pinning: spare blocks, pinned bytes (blocks + spares), all under mu
    std::thread pinner;
    mutable std::mutex mu;
    std::condition_variable cv;
    std::vector<char *> spare;
    size_t pinned = 0;
    bool stop = false, clamped = false;
    int device = 0;
    char * pin_block();                      // a pinned block within the cap and MemAvailable (null if not)
    void   pin_loop();

    PrefixAlloc alloc(size_t bytes);         // empty on failure (no eviction)
    void        release(PrefixAlloc & a);
    // alloc, evicting LRU entries (never `keep`) until it fits; empty if nothing is left to evict
    PrefixAlloc alloc_evict(size_t bytes, const PrefixEntry * keep = nullptr);
    void        drop(PrefixEntry * e);       // remove an entry (its segments go when unreferenced)
};

using PrefixCachePtr = std::unique_ptr<PrefixCache, void (*)(PrefixCache *)>;
PrefixCachePtr prefix_cache_create(const PrefixCacheOptions &);
void           prefix_cache_destroy(PrefixCache *);
// the entry with the longest P whose tokens are ids[0, P), P <= n (counts a hit or a miss); null if none.
// Marks it used (LRU).
PrefixEntry *  prefix_cache_lookup(PrefixCache &, const int32_t * ids, int64_t n);
// the longest prefix ids[0, L) shares with any entry's tokens, usable or not (diagnostics; no stats, no LRU)
int64_t        prefix_cache_common(const PrefixCache &, const int32_t * ids, int64_t n);
PrefixCacheStats prefix_cache_stats(const PrefixCache &);

}  // namespace strata::glm
