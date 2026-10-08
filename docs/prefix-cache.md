# Host-RAM prefix cache (design draft, kda, 2026-10-05)

## 1. What today's reuse covers (measured)

Our V100 host, glm_serve 26ec8cc, prod config (2 sessions x 131072, Q8 latent, --slots 1350, fast prefill), driver
a scenario script over GEN. S = a Claude Code-like system block (system text + 14 tools, 15550 tokens, up to the
first `<|user|>`); S2 / S3 = other system prompts of the same size. Engine recv -> first T:

| request | path | reused | new | first T |
|---|---|---|---|---|
| A1 new conversation, S, engine fresh | reset | 0 | 15853 | 30.9 s |
| A2 A follow-up | as-is | 15885 | 303 | 1.6 s |
| B1 new conversation, same S | snap (base) | 15550 | 303 | 1.5 s |
| A3 A follow-up after B | snap (base) | 15550 | 973 | 3.3 s (A's own turns re-read) |
| B2 B follow-up after A | snap (base) | 15550 | 638 | 2.4 s (same) |
| C1 new conversation, other system S2 | reset (slot 1) | 0 | 15853 | 29.6 s |
| E1 new conversation, S (C holds slot 1) | snap | 15550 | 303 | 1.5 s |
| D1 third system S3 | reset (evicts C) | 0 | 15853 | 29.6 s |
| F1 new conversation, S | snap | 15550 | 303 | 1.6 s |
| C2 C follow-up after D | reset | 0 | 16188 | 30.6 s |

Where the cold cost is paid today:
- Any prefix that is in no slot: the first conversation after a start, a third distinct system prompt with 2 slots
  (D1), and every conversation it evicts (C2: its whole history again). About 1.9 s per 1K tokens at 16K context
  (30 s for 15.5K; production log: 95K in 164 s, 113K in 193 s).
- Interleaved conversations with the same system block in one slot (main agent + subagent, two Claude Code windows):
  each switch keeps only S and re-reads the other conversation's own turns (A3, B2). At 100K conversations that is
  minutes per switch.
- The base snapshot is cut at the first `<|user|>`: a client whose shared part sits after it gets nothing (production
  log: "reused 9" on every request of a 95K / 113K run).
- Within one slot a new conversation with S already costs only its new tokens (B1, E1, F1: 1.5 s).

## 2. Design

A prefix cache in pinned host RAM, shared by all sessions; the slots become working buffers.

State of a prefix of P tokens (Q8 latent, 11 trunk MLA layers + the MTP block = 12 caches):
- per token per cache: latent 512 B + scales 16 B + pooled key 64 B (256 B per pool of 4) = 592 B of device rows,
  7.1 KB per token over the 12 caches;
- the idx_kg rows of the last incomplete pool (< 4 rows per cache; the rest is only read to build pools);
- KDA: 34 layers x state = 146 MiB, fixed per entry;
- small: position, tokens, pending token, hcat row 0 (h_prev of the MTP), MTP cache length.

Sizes / restore at 12.3 GB/s H2D (KDA 146 MiB restore measured 12.6 ms):

| prefix | rows | entry | restore | prefill avoided |
|---|---|---|---|---|
| 15.5K (system + tools) | 110 MB | 256 MB | ~22 ms | 30 s |
| 30K | 213 MB | 360 MB | ~30 ms | ~60 s |
| 100K | 710 MB | 860 MB | ~70 ms | ~170 s |

Structure:
- Segment tree: MLA / pooled rows are stored in segments [a, b) with a parent segment (rows below a); an entry =
  (its last segment, KDA states at P, small state). Rows < P never change after they are written, so entries of the
  same conversation share their prefix segments: a follow-up's save copies only its new rows (2K tokens = 14 MB) plus
  the KDA states (12 ms). Refcounted segments, LRU over entries by bytes.
- Pinned pool: --prefix-cache-gb (proposed 48 of our host's ~200 GB free after the 169 GB arena), 64 MB pinned blocks
  allocated by a background thread ahead of use (pinning measured 0.46 s per GB, see 2b), sub-allocated.
- Lookup: longest entry whose tokens prefix the request (entries filtered by a hash of their first 64 tokens, then a
  token compare: microseconds). Taken when it beats every slot's as-is / snapshot reuse.
- Save points:
  1. prefill checkpoints at the prompt's chunk boundaries (multiples of --chunk below the system block's end, the
     system block's end, then its end + multiples of --chunk): any client layout gets reuse at <= 8K granularity,
     and the chunking after a restore is the chunking of a full prefill (needed for the exact-mode gate);
  2. the end of every request (prompt + emitted tokens): a conversation evicted from its slot or interleaved with
     another comes back with only its new turn to read.
  KDA D2H is synchronous (12 ms, after DONE / after the early T: off the TTFT path); row D2H is queued on the
  session's stream before any later write to those rows.
- Restore into any slot: truncate the slot, H2D rows / pooled / kg tail, set the cache lengths, load KDA + hcat,
  pos / toks / pending; the decode graphs read the same buffers (no recapture).
- Slot choice: as today (most reusable), else the least recently used slot; its state is already in the cache.

Ownership:
- mla.hpp: `mla_cache_export(cache, a, b, host, st)` / `mla_cache_import(cache, n, ...)` for the latent +
  scales + pooled rows and the kg tail, and set the length (MTP: same API through mtp_cache()).
- engine/prefix_cache.{hpp,cpp} + session_export / session_import.
- serve.cpp: lookup across slots + cache, checkpoint / end-of-request saves, --prefix-cache-gb, /health
  counters (entries, GB, hits, tokens saved). spec: MTP state check (mtp_size, hcat).

Not in scope: surviving an engine restart (an NVMe tier, ~3 GB/s: a 15.5K entry in ~90 ms, would also remove the
first-start cost; later if wanted).

## 2b. Orch approval conditions (2026-10-05)

- the cap is a ceiling: blocks pinned lazily (a background thread keeps 1 GB pinned ahead so saves never pin;
  mmap, pages touched, then cudaHostRegister: ~6 ms driver-lock stall per 64 MB block vs 250 ms for cudaHostAlloc;
  MAP_POPULATE was dropped, see section 3),
  never past MemAvailable - 64 GB, effective cap logged; everything unpinned at stop; /health: cap, used, entries,
  hits, misses, bytes restored;
- saves on their own stream, never on the TTFT path; prove decode tok/s and first T with saves on / off (> 2 %
  regression blocks the merge);
- stage a) end-of-request + system-end entries + restore into any slot, report, then b) chunk checkpoints;
- gate below + a 1 GB cap eviction + the refcount case + a realistic interleave (2 conversations, same S, 30K+ each);
- before / after scen.py table here.

## 2c. Stage a) as built (kda, glm53-decode)

- mla.hpp (fbafec9): `mla_cache_{rows_bytes,export_rows,import_rows}` (range [a, b): latent
  rows + Q8 scales + pools [a/4, b/4), 64 B header checked on import), `mla_cache_{tail_bytes,export_tail,
  import_tail}` (key | gate rows of the incomplete last pool), `mla_cache_can_truncate`: imported ranges hold no
  key | gate rows, so a later cut at m % 4 != 0 inside them is refused (session_restorable / session_restore).
- engine/prefix_cache.{hpp,cpp}: 64 KB pages in pinned blocks, first fit; segments of <= segment_max rows owned
  (shared_ptr) by the entries; LRU eviction by entry; lookup = longest entry whose tokens prefix the request.
- session.hpp: `session_save(s, pc)` queues the copies on the cache's stream (the session's next call waits for them
  on its stream, the host never waits) and only copies rows past the session's own chain; `session_load(s, pc, e)`
  keeps the session's rows below the deepest segment both chains share (bitwise the same blobs), imports the rest,
  loads KDA + hcat. A chain is a weak reference: an evicted chain only costs a full copy at the next save.
- glm_decode --prefix-test (2 sessions, exact mode): save / generate unchanged, load into another conversation +
  follow-up = the same session as is, load of its own entry imports 0 rows, snapshot restorability after a load,
  eviction under a cap with segments kept by the surviving sibling. Q8, spec 1, 2048-token prompt: PASS; load
  of 2080 rows 14.4 ms (KDA 146 MiB is most of it), save copies ~12 ms.

## 3. Gate

Contract (2026-10-05). The cache adds no numerical path: it only copies state.
1. A restored state is bitwise the saved state, checked part by part (session_state_digest: per MLA cache the latent
   rows [0, P) + Q8 scales, the pooled rows of the complete pools and the key | gate tail; per KDA layer the state;
   hcat row 0 (MTP h_prev); position, tokens, pending token), not inferred from tokens.
2. The T stream after a restore is identical to the stream of the same session continuing as is.
3. Against a cold full prefill of the whole prompt: report only. The divergence must be no larger than today's
   accepted as-is reuse on the same scen.py rows (chunking and batch shapes differ exactly as they do for as-is reuse).

Runs (exact mode: --prefill-exact --adapt-every 0 --slots 1350; Q8 and F16 latents, spec 1):
- glm_decode --prefix-test: contract 1 + 2 on two sessions, eviction under a cap with shared segments;
- serve (once wired): gate.py's 20-follow-up chain unchanged with the cache on; S + U (system-end entry),
  S + U' after an eviction (load into the other slot), two interleaved conversations with the same S, 30K+ each
  (end-of-request entries), a 1 GB cap with evictions and a load sharing segments with an evicted sibling;
- saves on / off: decode tok/s, first T, and the inter-token gap max and p99 during decode with the pin-ahead thread
  active (cudaHostRegister holds the driver lock ~6 ms per 64 MB block); if the max gap rises > 25 ms vs off,
  blocks are registered only while no request decodes;
- the scen.py table of section 1 before / after, restore times and RAM.

Results (V100 host, 2026-10-05, glm_serve = 2e503e2 + 4aa1d66, --prefix-cache-gb 48; scripts pcgate.sh, pcgate.py,
pccompare.py, gate.py, scen.py, not in this repo):
- glm_decode --prefix-test, Q8 and F16: PREFIX OK; contract 1: 84 (Q8) / 72 (F16) state parts bitwise equal after
  a load of an entry whose evicted sibling's segments it shares; contract 2: load + generate = the session as is.
  Load of 2108 rows 14.7 / 15.6 ms.
- exact, two conversations with the same S, 4 turns of 4000 tokens each (30K+), interleaved A1 B1 A2 B2 ... vs the
  reference A1..A4 then B1..B4 as is: 2 sessions, 1 session (12 cache loads), 1 session with a 1 GB cap (4
  evictions, segments kept by the surviving entries): T streams identical (contract 2).
- exact, gate.py's 20-follow-up chain, cache off vs on: 21 / 22 streams identical. f16-2000 differs at T0: the
  rewind f06-rw200 took the cache path (the exact state at 33947 tokens) where the reference restored the base
  snapshot and re-read 227 tokens, so the two prefixes come from different chunkings (contract 3 class, accepted
  report-only). First T on = off (cold 50.8 s both; 0-token follow-ups 76-87 ms both); decode time on the 21
  identical streams 15.71 s on vs 15.77 s off (532 tokens, -0.4 %).
- prod fast mode, 1 session, the two interleaved 30K+ conversations (each switch: full re-read off, a cache load on):
  first T per switch 12.6-26.8 s off -> 7.6-7.8 s on; decode inter-T gaps median 55.4 / 55.2 ms, p99 66.1 / 67.4 ms,
  max 297 / 290 ms (both the cold request's first round), max elsewhere 72 / 79 ms: +7 ms, under the 25 ms rule, so
  blocks stay registered in the background. 8 entries: 1464 MB saved, 2432 MB pinned, 37150 tokens / 979 MB restored.
- Defect found and fixed on the way (4aa1d66): MAP_POPULATE held mmap_lock for each block's populate (~90 ms per
  256 MB) and the engine thread's large mallocs waited: "parse" 70-99 ms on about half the requests with the cache
  on. Now pages touched (per-VMA faults) and 64 MB blocks: parse 2-5 ms on every request.

Section 1's scen.py table with the cache (prod config, 2 sessions; engine recv -> first T, before -> after):

| request | path after | before | after |
|---|---|---|---|
| A1 new conversation, S, engine fresh | reset | 30.9 s | 29.2 s |
| A2 A follow-up | as-is | 1.6 s | 1.4 s |
| B1 new conversation, same S | snap | 1.5 s | 1.5 s |
| A3 A follow-up after B | cache (16220 reused) | 3.3 s | 1.3 s |
| B2 B follow-up after A | as-is (second slot) | 2.4 s | 1.4 s |
| C1 new conversation, other system S2 | reset | 29.6 s | 28.0 s |
| E1 new conversation, S (C holds slot 1) | snap | 1.5 s | 1.2 s |
| D1 third system S3 | reset (evicts C's slot) | 29.6 s | 27.8 s |
| F1 new conversation, S | snap | 1.6 s | 1.3 s |
| C2 C follow-up after D | cache (15885 reused) | 30.6 s | 1.3 s |
| G1 new conversation, S | snap | 1.6 s | 1.3 s |

A cache load of ~16K tokens takes 53-59 ms. A cold system prompt still costs its full read (A1, C1, D1: nothing to
reuse); stage b (chunk checkpoints) targets the shared parts past the first `<|user|>`.
