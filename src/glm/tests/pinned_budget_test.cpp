// glm_pinned_budget_test: pinned_budget_fit arithmetic and the arena guard (engine/pinned_budget.hpp), no GPU. Prints
// PASS / FAIL lines, rc 0 when every case passes.
#include <cstdio>

#include "pinned_budget.hpp"

using namespace strata::glm;

namespace {

int bad = 0;
void expect(const char * what, const PinnedPlan & p, int sessions, int snapshots, int chunk, bool fits) {
    const bool ok = p.sessions == sessions && p.snapshots == snapshots && p.chunk == chunk && p.fits == fits;
    std::printf("%s %s: sessions %d snapshots %d chunk %d, %.1f MB, fits %d (want %d %d %d %d)\n", ok ? "PASS" : "FAIL",
                what, p.sessions, p.snapshots, p.chunk, p.bytes / 1048576.0, (int) p.fits, sessions, snapshots, chunk,
                (int) fits);
    bad += !ok;
}

}  // namespace

int main() {
    constexpr size_t MB = 1 << 20;
    // the full model's figures: 12 MLA caches x 512 B per token, ~146 MB snapshots, ~48 KB per chunk token, ring 109 MB
    const auto needs = [](long ctx) {
        PinnedNeeds n;
        n.per_session = (size_t) 12 * 512 * (ctx + 16) + 256 * 1024;
        n.per_snapshot = 146 * MB;
        n.per_chunk_token = 49220;
        n.fixed = 109 * MB + 2 * MB;
        return n;
    };
    const size_t b900 = 900 * MB;
    // unchanged when it fits
    expect("fits as asked", pinned_budget_fit(needs(16384), 64 * 1024 * MB, 2, 4, 8192), 2, 4, 8192, true);
    // defaults at 32K under 900 MB: snapshots 4 -> 1, chunk 8192 -> 2048 (2 x (192 + 146) + 111 + chunk), 2 sessions
    {
        const PinnedPlan p = pinned_budget_fit(needs(32768), b900, 2, 4, 8192);
        expect("32K x 2 sessions", p, 2, 1, 2048, true);
        bool tighter = pinned_total(needs(32768), 2, 1, 4096) > b900;
        std::printf("%s chunk 4096 would not fit\n", tighter ? "PASS" : "FAIL");
        bad += !tighter;
    }
    // 131K: one session is 768 MB of mapped rows: down to 1 session, its snapshot does not fit next to it
    expect("131K", pinned_budget_fit(needs(131072), b900, 2, 4, 8192), 1, 1, 64, false);
    // 64K x 1 session: 384 + 146 + 111 = 641 MB: chunk regrown to the largest that fits, then snapshots
    expect("64K x 1 session", pinned_budget_fit(needs(65536), b900, 1, 4, 8192), 1, 1, 4096, true);
    // no prefill (chunk 0) stays 0; the snapshots come back
    expect("no prefill", pinned_budget_fit(needs(16384), b900, 1, 4, 0), 1, 4, 0, true);
    // sessions only drop after snapshots and chunk are at their floors; then chunk regrows first (4096: 884 MiB; 8192
    // would be 1076), then the second snapshot
    expect("sessions last", pinned_budget_fit(needs(49152), b900, 3, 2, 8192), 1, 2, 4096, true);
    // the arena guard (routed_load): refuse past MemAvailable minus the margin, never when MemAvailable is unknown
    {
        constexpr size_t GB = 1000000000;
        const struct { const char * what; size_t arena, margin, avail; bool fits; } cases[] = {
            {"arena fits", 169 * GB, 4 * GB, 365 * GB, true},
            {"arena at the limit", 361 * GB, 4 * GB, 365 * GB, true},
            {"arena 1 byte over", 361 * GB + 1, 4 * GB, 365 * GB, false},
            {"second full arena (Oct 7)", 169 * GB, 4 * GB, 196 * GB, true},
            {"third full arena", 169 * GB, 4 * GB, 27 * GB, false},
            {"MemAvailable unknown", 400 * GB, 4 * GB, 0, true},
        };
        for (const auto & c : cases) {
            const bool ok = arena_fits(c.arena, c.margin, c.avail) == c.fits;
            std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.what);
            bad += !ok;
        }
        setenv("GLM_ARENA_MARGIN_GB", "400", 1);
        bool ok = arena_margin_bytes() == 400 * GB;
        unsetenv("GLM_ARENA_MARGIN_GB");
        ok = ok && arena_margin_bytes() == 4 * GB;
        std::printf("%s GLM_ARENA_MARGIN_GB 400 / default 4\n", ok ? "PASS" : "FAIL");
        bad += !ok;
        const size_t avail = mem_available_bytes();
        std::printf("%s MemAvailable read (%.1f GB)\n", avail ? "PASS" : "FAIL", avail / 1e9);
        bad += !avail;
    }
    std::printf("%s\n", bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}
