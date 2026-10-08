# KDA prefill: why the chunked form is deferred on the V100

Decision on 2026-10-01 (session kda, agreed with the orchestrator): prefill uses the sequential long-batch scan
(`kda_scan_rq_kernel`, 2 value rows x 32 key channels per thread). The chunked form (chunk 16, cumulative gates,
triangular solve, as in llama.cpp's `delta-net-base.cpp` CPU graph) is not implemented. `scan_chunked` in
`kda_chunk.cu` falls back to the sequential scan.

## Arithmetic

Per token and head (K = V = 128):

- sequential: decay multiply, k-dot, rank-1 update, q-dot = 4 x 128^2 FP32 ops.
- chunked (C = 16): W S0, QP S0, KE^T Delta = 3 x 128^2, plus the intra-chunk part (L, Qk, the 16x16 solve,
  W = M diag(beta) KP, U) at roughly 0.5 x 128^2 more. Each value-column block of a head recomputes that part.

On CUDA cores the two forms cost about the same. The FP32 floor is about 1.2 ms per 2048 tokens per layer
(64 heads, 80 SMs at about 1.3 GHz under load). The sequential scan reaches 2.55 ms. Its limit was
shared-memory bandwidth, not FP32 issue, until each thread kept 2 rows.

## Tensor cores

Chunking only pays if the three 128-wide products run on WMMA (sm_70: fp16 inputs, F32 accumulate). Keeping
F32-level accuracy needs split operands, A = A_hi + A_lo in fp16, which costs 3 MMAs per product. The delta rule
subtracts a prediction from v, so a single fp16 product loses about 1e-3 relative on that difference.

The blocker is that each head walks its chunks in order. So all blocks must be resident at once (at most about
80, one per SM). One such block per head or half-head needs, in shared memory, all of:

- the F32 state slice: 32-64 KB;
- its fp16 hi/lo copy: 32 KB;
- the W/QP/KE hi/lo operands: 24 KB;
- the pass-1 F32 inputs: 24-40 KB.

That exceeds the V100's 96 KB per block. Escape routes:

- keep the state in L2 between chunks (8 MiB of traffic per chunk step);
- store the state as hi/lo fp16, which is not F32. The brief forbids this.

Best case estimate: about 1.0-1.3 ms instead of 2.55 ms per 2048 tokens. That is at most about 10% of a KDA prefill
layer, where the projections take about 70%.

## What was done instead

Prep stays a separate kernel. Two attempts to fuse it into the scan were both slower than prep + scan
(3.7 ms per 2048 tokens):

- the scan's own staging computing conv / L2 / gates: 4.1 ms;
- a producer-warp double buffer: 13.8 ms with 1 producer warp, 4.5 ms with 4.

The recurrence warps are few (about 5 per SM) and latency-bound. Extra serial work in the same block costs more
than the 537 MB round trip it saves. What helped is a barrier-free prep, one warp per (token, head) row:
1.15 -> 0.84 ms per 2048 tokens, close to the bandwidth floor of about 0.65 ms. The fused attempts were not
committed (session kda, 2026-10-01).
