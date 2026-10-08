# GLM-5.3-Flash (`glm5-next`) forward-pass specification

Written from llama.cpp's `glm5-next` implementation (MIT, the ggml authors) before the engine: it is the reference
the GLM engine's math was checked against. File and line references point into that llama.cpp snapshot.

Source snapshot: a llama.cpp checkout, ggml-org master at `42d95816` (2026-10-01). Tensor values/shapes and byte totals below were taken from the six real `UD-Q4_K_XL` GGUF shard headers via `GGUFReader`; tensor payloads were not loaded. The op order below is derived from the llama.cpp source, not confirmed by a trace.

File/line references are relative to the llama.cpp checkout unless given as an absolute model path. Linear algebra notation uses column vectors: `W x` maps input dimension to output dimension; a GGUF shape `[in,out]` is stored in ggml's reversed matrix convention.

## 1. Real GGUF and layer inventory

### Model-wide values

| GGUF field / property | Real value | Notes |
|---|---:|---|
| `general.architecture` | `glm5-next` | maps to `LLM_ARCH_GLM5_NEXT` (`src/llama-arch.cpp:156`) |
| `general.name`, `general.size_label` | `GLM 5.3 Flash`, `288x10B` | Actual file advertises a 288-expert model. |
| `glm5-next.block_count` | 46 | Includes one MTP/NextN block. |
| `glm5-next.nextn_predict_layers` | 1 | `llama_hparams::n_layer() = n_layer_all - n_layer_nextn`, therefore 45 trunk blocks + block 45 MTP (`src/llama-hparams.cpp:348-350`). |
| Context | 1,048,576 | `glm5-next.context_length`. |
| Hidden / embedding | 4096 | `glm5-next.embedding_length`; token embedding and output matrix have this inner dimension. |
| Attention heads | 64 | `glm5-next.attention.head_count`; also KDA head count. |
| Generic attention key/value lengths | 512 / 512 | GGUF metadata; MLA-specific fields below are the dimensions used by its graph. |
| MLA key/value head lengths | 256 / 256 | `key_length_mla`, `value_length_mla`. |
| MLA Q LoRA rank / KV LoRA rank | 1536 / 512 | `q_lora_rank`, `kv_lora_rank`. |
| Dense FFN width | 12,288 | `feed_forward_length`, used by blocks 0-2. |
| Expert FFN width / shared expert width | 2048 / 2048 | `expert_feed_forward_length`, `expert_shared_feed_forward_length`. |
| Experts / routed experts | 288 / 8 | `expert_count`, `expert_used_count`. |
| Expert groups / selected groups | 1 / 1 | No multi-group filtering is performed. |
| Dense leading blocks | 3 | `leading_dense_block_count`; blocks 0, 1, 2 use dense FFN. |
| Shared experts | 1 | `expert_shared_count`. |
| Router function | sigmoid (`expert_gating_func = 2`) | Enum values in `src/llama-hparams.h:16-21`; `src/models/glm5-next.cpp:30-33`. |
| MoE selected-weight normalization / scale | true / 2.5 | `expert_weights_norm`, `expert_weights_scale`. |
| MoE selection correction bias | present | `blk.i.exp_probs_b.bias`, F32 vector length 288 on MoE blocks. It affects selection only, not the uncorrected probability used as the expert weight (`src/llama-graph.cpp:2084-2090`). |
| RMSNorm epsilon | 1e-5 | `attention.layer_norm_rms_epsilon`; `build_norm(..., LLM_NORM_RMS)` uses it (`src/llama-graph.cpp:1612-1613`). |
| LayerNorm epsilon | 1e-6 | `attention.layer_norm_epsilon`; used for the indexer key LayerNorm (`src/models/glm5-next.cpp:783-785`, `src/llama-graph.cpp:1612-1613`). |
| SwiGLU clamp | exp=10, shared/dense=10 | Both GGUF arrays have 46 entries, all 10.0. |
| Vocab | 154,880 | `glm5-next.vocab_size`. |
| Tied embedding/head | No | Both `token_embd.weight` and `output.weight` are present as separate Q8_0 tensors, each `[4096,154880]`. |
| Rope dimension count | 0 | `glm5-next.rope.dimension_count`. GLM5-next MLA asserts `n_rot()==0`; there is no RoPE in the trunk attention graph (`src/models/glm5-next.cpp:930-936`). |
| Rope theta / scale metadata | No model-specific theta/scaling keys | Generic loader fallbacks are theta 10000 and frequency scale 1 (`src/llama-model.cpp:1413-1430`), but inert because rotary dimension is 0. Effective RoPE variant is none; do not apply a rope permutation or rotation. |
| DSA indexer | 32 heads, 128 dims/head, top-k 2048 cells, kpool 4 | `attention.indexer.head_count`, `.key_length`, `.top_k`, `.kpool`. `kpool_select_tail` is absent in this GGUF and defaults true in the converter/runtime. |
| Indexer-MTP metadata | `index_share_mtp = true` | Pinned `glm5-next.cpp` does not read this key. See MTP caveat in section 5. |
| mHC | 4 streams, 20 Sinkhorn iterations, epsilon 1e-6 | `hyper_connection.count`, `.sinkhorn_iterations`, `.epsilon`. Runtime asserts `hc_mult==4` (`src/models/glm5-next.cpp:49-53`). |
| KDA | head dim 128, convolution kernel 4, gate lower bound -5 | `kda.head_dim`, `ssm.conv_kernel`, `kda.gate_lower_bound`. Thus `d_inner = 64*128 = 8192`. |

The per-block `attention.head_count_kv` array has 46 entries: zero on KDA blocks, one on MLA/DSA blocks and MTP block. `load_arch_hparams()` sets `is_recr[i] = (n_head_kv(i)==0)` (`src/models/glm5-next.cpp:21-23`). The real layer sequence is:

- Trunk KDA + dense FFN: blocks 0-2.
- Trunk KDA + MoE FFN: blocks 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38, 40-42, and 44.
- Trunk MLA + DSA indexer + MoE FFN: blocks 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 (11 total).
- MTP block 45: MLA/DSA-indexer + MoE tensors are present; no mHC tensors.

So the trunk has 34 KDA blocks + 11 MLA/DSA blocks = 45. The first-three-dense rule is independent of attention type. Header inventory confirms KDA tensors in 34 trunk blocks; `attn_q_a` and `indexer.proj` in the 11 trunk MLA blocks plus block 45; dense FFN weights in 0-2; and expert tensors in blocks 3-45.

The indexer type array is absent from the actual GGUF. `load_arch_hparams()` first fills it with 1 (full indexer) before reading the optional array (`src/models/glm5-next.cpp:46-47`), so all 11 trunk MLA blocks are full indexers. The graph reuses `prev_sel` only for a layer explicitly marked shared; that branch asserts a preceding full layer and reuses its top-k selection (`src/models/glm5-next.cpp:957-964`). The MTP block is forced full by the `il >= n_layer` condition. Do not infer shared trunk indexers for this GGUF.

## 2. Tensor name/conversion facts

The top-level `convert_hf_to_gguf.py` dispatches to `conversion/glm.py`; the converter-specific evidence is in `conversion/glm.py:407-599`, and mappings/constants in `conversion/tensor_mapping.py` and `gguf-py/gguf/constants.py`. Important transforms to preserve:

- No Q/K RoPE permutation; converter asserts nope-only (`conversion/glm.py:482`). `kv_b` is split into `attn_k_b` and `attn_v_b`: the key part is transposed to `[head,kv_rank,qk_nope]`; V remains `[head,value_dim,kv_rank]` (`conversion/glm.py:540-550`). Original fused `kv_b` is not emitted.
- Expert matrices are gathered in numeric expert order and stacked for `_exps` (`conversion/glm.py:522-538`).
- KDA convolution weights are reshaped to `[1,d_inner,1,d_conv]`; `A_log` becomes `ssm_a = -exp(A_log.float())`; `dt_bias` becomes `ssm_dt.bias` (`conversion/glm.py:552-567`).
- mHC tensors and KDA A/dt plus router correction are forced F32; vector/norm tensors, KDA convs and indexer projection also remain F32 as indicated by converter/base (`conversion/glm.py:575-580`, `conversion/base.py:1083-1121`). The actual quantized model subsequently uses mixed Q4_K/Q5_K/Q8_0 and F32 types.
- `lm_head` is only dropped when tied; this GGUF carries distinct `token_embd.weight` and `output.weight` (both Q8_0). Vision/projector tensors are omitted; `language_model.` is stripped; `e_score_correction_bias` maps to `exp_probs_b.bias` (`conversion/base.py:1387-1401`, `conversion/glm.py:590-599`).
- MTP names include `blk.45.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`. In this file, `nextn.embed_tokens` and `nextn.shared_head_head` are absent, so intended head/embed fallback is shared trunk embedding/head. `--no-mtp` removes the MTP block; default conversion preserves it (`conversion/glm.py:420-455`).

## 3. Trunk forward pass, exact branch order

Notation: `N` is the current ubatch token count; `D=4096`; `H=4` mHC streams. `x` has logical shape `[D,H,N]`. For GGML tensors, shapes below list feature dimensions first. `mat(W,x)` means `ggml_mul_mat(W,x)` with the weights as serialized. The entry graph repeats token embedding `e:[D,N]` four times to initialize streams (`src/models/glm5-next.cpp:554-581`). Every trunk block runs attention and FFN as separate mHC branches in this order (`src/models/glm5-next.cpp:585-650`).

### mHC branch wrapper used before attention and FFN

For each branch, save `residual = x`, then:

1. Flatten `x` from `[D,4,N]` to `[4D,N]`; RMSNorm over the flattened 16,384 features with epsilon 1e-5.
2. Apply the branch's `hc_*_fn` matrix: output `mixes:[24,N]`, because `(2+4)*4 = 24`.
3. Split first 4 entries as pre logits, next 4 as post logits, final 16 as combine logits. Apply affine params from `hc_*_scale[3]` and `hc_*_base[24]`:
   - `pre = sigmoid(mix_pre*scale[0] + base[0:4]) + eps`, `eps=1e-6`.
   - `post = 2*sigmoid(mix_post*scale[1] + base[4:8])`.
   - `comb_logits = reshape(mix_comb*scale[2] + base[8:24], [dst=4,src=4,N])`.
4. `comb = Sinkhorn(comb_logits)`: row softmax over `dst`, add eps, normalize columns once, then repeat row normalization + column normalization 19 times (20 total iterations); every sum denominator has `+eps` (`src/models/glm5-next.cpp:417-451`). The fused `ggml_dsv4_hc_comb` must match the same order.
5. Branch input is `branch_in[feature,n] = sum_s x[feature,s,n]*pre[s,n]` (`src/models/glm5-next.cpp:454-509`).
6. After the branch op, reconstruct each stream exactly as `x_new[dst] = branch_out*post[dst] + sum_src residual[src]*comb[dst,src]` (`src/models/glm5-next.cpp:512-547`).

The attention branch is executed first, and its output is mHC-post-mixed with the pre-attention residual. The FFN branch then independently repeats mHC-pre, applies FFN, and mHC-post-mixes with the post-attention residual. There is no conventional single-stream residual-add shortcut (`src/models/glm5-next.cpp:588-648`).

### Attention branch common prefix

`branch_in` -> RMSNorm with epsilon 1e-5 and `blk.i.attn_norm.weight` -> KDA if `n_head_kv(i)==0`, otherwise MLA+DSA (`src/models/glm5-next.cpp:595-605`). KDA and MLA do not share the same projection path.

### KDA block (34 trunk blocks)

Dimensions: heads `Hk=64`, head dim `d=128`, inner `8192`; Q/K/V per token are `[128,64]`. Weight names: `attn_q/k/v.weight`, `attn_output.weight`, `ssm_conv1d_{q,k,v}.weight`, `ssm_f_{a,b}.weight`, `ssm_beta.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_g_{a,b}.weight`, `ssm_norm.weight` (`src/models/glm5-next.cpp:96-121`; converter mapping in `conversion/tensor_mapping.py:838-948`).

1. Independently for q, k, v: project `branch_in` to 8192, reshape `[8192,Nseq,Nseqs]`, prepend that Q/K/V-specific cache history of 3 rows, causal depthwise Conv1D with kernel length 4, then SiLU. Reshape output to `[128,64,Nseq,Nseqs]`. The concat/history and graph ordering are in `src/models/glm5-next.cpp:195-235,694-705`; convolution is `ggml_ssm_conv` (`ggml/src/ggml-cpu/ops.cpp:9930-9998`).
2. Decay gate: `u = W_fa*branch_in`, `z = W_fb*u + dt_bias`, broadcast `A=ssm_a` over the 128 channels of each head, then `g = -5 * sigmoid(-(z*A))`. The converter stores `A=-exp(A_log)`. Since lower bound is -5, `g` is in `[-5,0]`. (The code's exact sign/order is `mul(z,A)`, `scale(-1)`, sigmoid, then scale by -5; `src/models/glm5-next.cpp:707-724`.)
3. Delta weight: `beta = sigmoid(W_beta*branch_in)`, shape `[1,64,Nseq,Nseqs]` (`src/models/glm5-next.cpp:728-731`).
4. Apply per-head Q/K L2 normalization with epsilon 1e-6. `build_gdn_l2_norm` is `RMSNorm(x,eps/d)*1/sqrt(d)`, algebraically `x/sqrt(sum(x^2)+eps)` (`src/models/models.h:13-18`; call order `glm5-next.cpp:737-742`).
5. Gated delta recurrence, independently per head; use F32 state `S` logically `[128,128,64,sequence]`. For token t and head h, with `g_t[i]` channelwise:
   - `decay_i = exp(g_t[i])`
   - `r[j] = sum_i decay_i * S_prev[i,j] * k_t[i]`
   - `delta[j] = beta_t * (v_t[j] - r[j])`
   - `S_t[i,j] = decay_i*S_prev[i,j] + k_t[i]*delta[j]`
   - `o_t[j] = sum_i S_t[i,j]*q_t[i] / sqrt(128)`.
   `S` is stored transposed for contiguous columns in CPU/CUDA kernels; preserve the math and do not transpose the learned matrices accidentally (`ggml/src/ggml-cpu/ops.cpp:11123-11263`, `ggml/src/ggml-cuda/gated_delta_net.cu:42-155`). The KDA gate has 128 independent decays per head; it is not a scalar-per-head GDA gate.
6. Output gate: `g2 = W_gb*(W_ga*branch_in)`, reshape `[128,64,N]`; apply per-head RMSNorm with epsilon 1e-5 to `o`, then elementwise multiply by `sigmoid(g2)`. Flatten to 8192, project through `attn_output.weight` back to 4096 (`src/models/glm5-next.cpp:746-760`).

#### KDA prefill/chunk form and fused op selection

`build_delta_net()` uses autoregressive arithmetic for one sequence token; for multiple tokens it uses the fused GDN op when `fused_gdn_ch` is enabled, otherwise the explicit chunk graph (`src/models/delta-net-base.cpp:425-447`). The generic explicit KDA chunk size is 16. It pads the tail to a multiple of 16, multiplies `k` by `beta`, computes cumulative `g` per channel, constructs lower-triangular causal decay masks `exp(cumsum(g)_j-cumsum(g)_i)`, forms within-chunk key-key matrices, and solves the lower triangular system `(I + tril(KB)) A = I - tril(KB)` (`src/models/delta-net-base.cpp:61-168`). Then it updates chunk values against the previous state, calculates per-token outputs, and advances the boundary state as `S_last = exp(g_last)*S_prev + (k*exp(g_last-cumsum(g)))^T * v_new` (`src/models/delta-net-base.cpp:170-286`). This is a block-parallel factorization of the recurrence above; it must preserve the same causal order and last-token state.

Fused `ggml_gated_delta_net` has `K=1` for ordinary calls; it can also emit K rollback snapshots when recurrent rollback is enabled (`src/models/delta-net-base.cpp:373-423,527-605`; op layout `ggml/src/ggml.c:6373-6427`). The CUDA GDN kernel currently loops over tokens sequentially and explicitly has `TODO: Add chunked kernel for even faster pre-fill` (`ggml/src/ggml-cuda/gated_delta_net.cu:169-219`). Do not assume its prefill arithmetic uses the CPU explicit chunk graph.

### MLA + DSA block (11 trunk blocks)

Weights (layer i): `attn_q_a [4096,1536]`, `attn_q_a_norm [1536]`, `attn_q_b [1536,64*256]`; `attn_kv_a_mqa [4096,512]`, `attn_kv_a_norm [512]`; `attn_k_b [256,512,64]`, `attn_v_b [512,256,64]`, `attn_output [64*256,4096]` (`src/models/glm5-next.cpp:123-139`). All rotary dims are zero.

1. `q_rank = Wq_a*x` -> RMSNorm(1e-5) -> `qr:[1536,N]`.
2. `q = Wq_b*qr`, reshape `[256,64,N]`; no rope split/rotation.
3. `kv = Wkv_a_mqa*x` -> RMSNorm(1e-5) -> latent `kv_cmpr:[512,1,N]`. The K cache stores this compressed latent only.
4. Absorb the key expansion into each query: `q_abs[h] = Wk_b[h]^T * q[h]`, producing `[512,64,N]`. The implementation explicitly multiplies by `wk_b` before sparse selection and then attends in latent space (`src/models/glm5-next.cpp:937-953`). No unabsorbed 256-d key is written into the cache.
5. Select/reuse the DSA sparse positions as described below. Main trunk GGUF marks all indexers full by default, so each of the 11 MLA layers scores its own selection.
6. Latent attention: score `q_abs[h]^T * cached_kv[:,pos]`, apply causal+sparse mask, softmax with `scale=1/sqrt(256)=1/16`, and weight cached latent values (V is a view of the same latent K rows). Apply `Wv_b[h]` to decompress each head's 512-d weighted latent to 256-d V. Concatenate 64 heads to 16384, apply `attn_output` back to 4096 (`src/models/glm5-next.cpp:930-936,966-1012`).
7. When `n_tokens<=16 && n_kv>n_sel`, `gather=true`: gather the selected latent rows, `mul_mat` qk, F32-accumulation qk, `soft_max_ext(scale=1/16)`, latent value matmul, then `wv_b` (`glm5-next.cpp:971-996`). Otherwise construct an additive sparse mask over the cached K rows and call standard `build_attn_mha` with the latent K/V view and `wv_b` (`glm5-next.cpp:997-1006`). This is masked conventional attention, not a separate custom sparse-attention ggml op.

#### Lightning indexer / k-pool selection

For each full indexer layer:

1. `iq = W_indexer_q_b * qr`, reshape `[128,32,N]` (`glm5-next.cpp:778-780`).
2. `ik = W_indexer_k*x`; ordinary LayerNorm with learned weight+bias and epsilon 1e-6. `ig = W_kpool_gate*x` (`glm5-next.cpp:783-788`). Cache each token's `key[128] | gate[128] | pooled_slot[128]` together.
3. For each newly completed pool of 4 consecutive tokens, use `gate[:,j] + kpool_ape[:,j]` and softmax along the 4-token pool axis per key dimension; `pooled_key[:,pool] = sum_j softmax(logits[:,j])*key[:,j]`. Cache the result in the representative (last member) row. The cache lifecycle is in `src/llama-memory-hybrid-idx.cpp:659-680`; graph pooling in `glm5-next.cpp:790-832`.
4. `weights = W_indexer_proj*x / sqrt(128*32)` (the reciprocal scale is applied to the 32 per-head weights). For pool p, `score[p,t] = pool_mask[p,t] + sum_h weights[h,t] * relu(dot(iq[:,h,t], pooled_key[:,p]))`. `ggml_lightning_indexer` is an optional fused form of this exact head-score calculation; the unfused math is explicit at `glm5-next.cpp:834-856`.
5. `top_k` selects `min(n_pools,2048/4)=up to 512` pools per token. GGML top-k is unordered, so gather their scores and argsort descending before expanding each chosen pool to its 4 member cells (`glm5-next.cpp:858-879`). With `kpool_select_tail=true`, append up to 3 visible incomplete-tail cells. Thus the nominal selected full-pool token budget is 2048, plus at most 3 tail positions.
6. Sparse application: selected cells become either gathered row indices (small decode path) or a mask with -infinity everywhere and zero at selected visible cells, then add the causal mask (`glm5-next.cpp:880-918`). Padding/dead slots are masked separately; they must never become live through a sentinel index. The pool mask and indices are populated from sequence positions by `llama-memory-hybrid-idx.cpp:789-968`.

If `is_indexer_full(il)==false`, the graph reuses `prev_sel` without recomputing `iq/ik/ig` top-k for that layer; a shared indexer must follow a full one (`glm5-next.cpp:957-964`). This GGUF has no `indexer.types` metadata, so this sharing branch is not used by the trunk.

## 4. FFN and head

After the attention result has been mHC-post-mixed, save `residual`; run a fresh FFN mHC-pre branch, then RMSNorm with epsilon 1e-5 (`src/models/glm5-next.cpp:609-615`).

- Blocks 0-2: parallel dense SwiGLU, dimensions `4096 -> 12288 -> 4096`. `up=W_up*x`, `gate=W_gate*x`, activation `silu(min(gate,10))*clamp(up,-10,10)`, then `down`. `build_ffn()` uses `swiglu_clamp_shexp` for this path too (`src/models/glm5-next.cpp:616-622`, `src/llama-graph.cpp:1808-1877`).
- Blocks 3-44: router logits `r=W_gate_inp*x`, `p=sigmoid(r)`. Add `exp_probs_b` only for top-8 selection; choose top-8 of 288 (one group, so no group stage). Get weights from original, unbiased `p`; normalize by `max(sum(p_selected),6.103515625e-5)`; multiply by 2.5. Each selected expert computes `down_expert(silu(min(gate_expert,10))*clamp(up_expert,-10,10))`, then multiply that expert result by its normalized/scaled route weight and sum the 8 outputs (`src/llama-graph.cpp:2044-2089,2128-2173,2210-2273,2326-2380`).
- Each MoE block also computes the single shared expert with the same clamped SwiGLU at width 2048; add it unweighted to routed output. The graph's shared branch and sum are at `src/models/glm5-next.cpp:637-645`; `build_ffn()` details at `src/llama-graph.cpp:1808-1877,1942-1968`.
- Apply mHC-post mix and optional `build_cvec` after FFN. After block 44, select requested output rows as needed, average the four streams, apply `output_norm` RMSNorm epsilon 1e-5, then `output.weight` matmul to 154880 logits (`src/models/glm5-next.cpp:648-681`). The pre-head stream mean is arithmetic mean, not a learned projection (`glm5-next.cpp:370-378,662-679`).

## 5. State/cache layouts and MTP

### Trunk caches

`llama-model.cpp` creates `llama_memory_hybrid_idx` for this arch with `type_r=F32`, `type_s=F32`, MLA attention cache only on non-recurrent trunk layers, index cache only on full indexer layers, and recurrent cache on KDA layers (`src/llama-model.cpp:2450-2492`).

- KDA recurrent matrix: per KDA block and sequence `[128,128,64]` F32 = 1,048,576 floats = 4 MiB per state slot. `n_embd_s = head_dim^2*n_head` (`src/llama-hparams.cpp:236-247`). With ordinary `K=1`, store final state. Speculative/recurrent rollback uses multiple state snapshot slots (`K=n_rs_seq+1`; `delta-net-base.cpp:563-605`).
- KDA convolution history: Q, K, V each retain `d_conv-1=3` rows x `d_inner=8192`, so packed cache per KDA block/sequence is `[3,3,8192]` = 73,728 F32 values = 288 KiB. `n_embd_r=3*(d_conv-1)*d_inner` (`src/llama-hparams.cpp:208-234`); `glm5_conv1d()` views the packed cache as three independent Q/K/V windows (`src/models/glm5-next.cpp:195-234`).
- MLA: one K-only cache row per token per MLA layer, containing `kv_cmpr[512]`; V is a view of these same compressed latent rows, then expanded via `wv_b` after attention. Cache tensor dtype follows runtime `type_k`; it is not the GGUF weight type (`src/models/glm5-next.cpp:944-947,966-969,1002-1005`; cache construction `src/llama-model.cpp:2474-2492`).
- Indexer: a side cache aligned to attention cell slots. GLM5's cache row has 3 x 128-d slots: raw normalized key, pool-gate vector, pooled-key slot. Newly completed pools are recomputed once; pooled keys persist and are gathered on later ubatches. `llama-memory-hybrid-idx.cpp:53-74,659-680`; API/layout comments `src/llama-memory-hybrid-idx.h:194-203`; graph pack/pool/scatter/gather `src/models/glm5-next.cpp:790-832`.
- Pool ids are groups of 4 cells in sequence order. The final member's row is the pooled-key representative. Incomplete current tail has separate indices; padded pools use sentinels and masks. Cache safety and stale-pool handling matter after sequence edits (`src/llama-memory-hybrid-idx.cpp:411-450,789-968`).

### MTP / NextN block 45

This section distinguishes observed from inferred. The pinned `glm5-next.cpp` loads the NextN tensors and block-45 attention, indexer, router, experts and shared expert, but `build_arch_graph()` throws `GLM5-Next NextN graph not implemented yet` for `LLM_GRAPH_TYPE_DECODER_MTP` (`src/models/glm5-next.cpp:67-72,177-190`). Therefore llama.cpp master at this snapshot does not execute a GLM5-next MTP forward graph; do not cite a guessed MTP as observed mainline math.

Actual block-45 MTP-specific tensors in this GGUF:

- `blk.45.nextn.eh_proj.weight [8192,4096] Q8_0`
- `blk.45.nextn.enorm.weight [4096] F32`, `hnorm.weight [4096] F32`
- `blk.45.nextn.shared_head_norm.weight [4096] F32`
- No `nextn.embed_tokens` or `nextn.shared_head_head` in the header. The loader marks them optional, so the intended path shares `token_embd` and `output` (`src/models/glm5-next.cpp:177-184`).
- The layer also has its own MLA matrices, indexer matrices, router correction, 288 experts, and one shared expert (real headers; shapes match the trunk but with `blk.45` prefix). It has no HC tensors.

Best-supported intended skeleton (inference from these tensors and the implemented GLM DSA NextN convention in `src/models/glm-dsa.cpp:532-538,581-615,707-767`; not verified for GLM5-next because its graph is missing):

```text
h = trunk post-output-norm hidden state (the main graph exports h_nextn)
e = token_embd(next draft token)   # MTP-specific embed_tokens absent
u = eh_proj(concat(RMSNorm(e, enorm), RMSNorm(h, hnorm)))
r = u
attn = one MLA/DSA block using block-45 matrices and its indexer tensors
f_in = r + attn
f = RMSNorm(f_in, blk.45.ffn_norm)
f_out = top8 sigmoid-routed expert FFN + shared expert
h_next = RMSNorm(f_in + f_out, shared_head_norm)
logits = output * h_next              # shared output head; shared_head_head absent
```

Block 45 has a dedicated indexer tensor set and `glm5-next.cpp` treats every `il>=n_layer` as a full indexer (`src/models/glm5-next.cpp:145-155,958-960`). GGUF also contains `index_share_mtp=true`, but no code in this pinned model reads that key; whether the reference model intended sharing of selected cells versus recomputing the block-45 indexer is unresolved in llama.cpp. The GLM DSA MTP skeleton is evidence for the general `enorm/hnorm -> concat -> eh_proj -> block -> shared-head norm/head` structure only, not proof of GLM5-next's missing branch details.

## 6. Non-standard ops and backend locations

| GGML op / path | Math/role | Implementation refs |
|---|---|---|
| `GGML_OP_SSM_CONV` (`ggml_ssm_conv`) | Per-channel causal width-4 convolution over prepended 3-token state plus current Q/K/V projections; output is then explicit SiLU in the graph (CUDA may fuse it). | API/op construction `ggml/src/ggml.c:5666-5691`; CPU `ggml/src/ggml-cpu/ops.cpp:9930-9998`; CUDA `ggml/src/ggml-cuda/ssm-conv.cu:160-206`; dispatch `ggml/src/ggml-cuda/ggml-cuda.cu:2357-2361,5522-5524`. |
| `GGML_OP_GATED_DELTA_NET` | F32 KDA/GDN recurrent update and token output using q,k,v,g,beta,state; returns output plus state snapshots. KDA is identified by vector gate shape `g.ne[0]==S_v`. | Construction/type/shape `ggml/src/ggml.c:6373-6427`; CPU recurrence `ggml/src/ggml-cpu/ops.cpp:11123-11334`; CUDA recurrence `ggml/src/ggml-cuda/gated_delta_net.cu:4-167,169-219`; dispatch `ggml/src/ggml-cuda/ggml-cuda.cu:2384-2385,5603-5604`. |
| `GGML_OP_LIGHTNING_INDEXER` | Fused DSA weighted sum of `relu(dot(indexer_q_head, pooled_key))` across indexer heads plus additive pool mask. It does not do top-k, pool expansion, or MLA attention; those remain graph ops. | Construction `ggml/src/ggml.c:6430-6464`; CPU `ggml/src/ggml-cpu/ops.cpp:12355-12435`; CUDA `ggml/src/ggml-cuda/lightning-indexer.cu:400-589`; graph scoring/top-k `src/models/glm5-next.cpp:834-918`. |
| `GGML_GLU_OP_SWIGLU_CLAMP` (used through `ggml_swiglu_clamp`) | `silu(min(gate,limit))*clamp(up,-limit,limit)`; GLM5 uses limit 10 for expert, shared and dense branches. | Construction `ggml/src/ggml.c:3132-3141`; CPU `ggml/src/ggml-cpu/ops.cpp:3569-3695`; CUDA `ggml/src/ggml-cuda/unary.cu:440-511`, scalar formula `unary.cuh:118-123`. |
| `GGML_OP_DSV4_HC_PRE` | Fused weighted sum across four streams using mHC pre weights. | Construction `ggml/src/ggml.c:6517-6570`; CPU `ggml/src/ggml-cpu/ops.cpp:11475-11553`; CUDA `ggml/src/ggml-cuda/dsv4-hc.cu:104-273`; dispatch `ggml/src/ggml-cuda/ggml-cuda.cu:2390-2392,5608-5610`. |
| `GGML_OP_DSV4_HC_COMB` | Fused affine 4x4 combine logits + row softmax + epsilon + Sinkhorn row/column normalization. | Construction `ggml/src/ggml.c:6466-6515`; CPU `ggml/src/ggml-cpu/ops.cpp:11337-11473`; CUDA `ggml/src/ggml-cuda/dsv4-hc.cu:8-102,197-238`; graph definition/order `src/models/glm5-next.cpp:417-451,493-506`. |
| `GGML_OP_DSV4_HC_POST` | Fused `post[dst]*branch + sum_src comb[dst,src]*residual[src]`. | Construction `ggml/src/ggml.c:6572-6611`; CPU `ggml/src/ggml-cpu/ops.cpp:11555-11648`; CUDA `ggml/src/ggml-cuda/dsv4-hc.cu:149-315`; graph equation `src/models/glm5-next.cpp:512-547`. |

The three DSV4 HC ops are optional fused routes; `glm5-next.cpp` supplies equivalent standard GGML-op decompositions when fused flags are off (`src/models/glm5-next.cpp:400-415,493-505,521-546`). There is no GLM5-specific sparse-attention custom op: small decode uses gather + matmul + softmax; the other route uses `build_attn_mha` with a sparse additive mask. `ggml_ssm_scan` is a different state-space operator and is not called by GLM5-next KDA.

### CUDA sm_70 note

The CUDA build's default architecture list includes `70-virtual` (`ggml/src/ggml-cuda/CMakeLists.txt:44-50`). The lightning-indexer WMMA path is gated on Turing (`TURING_MMA_AVAILABLE`, CC >= 750), but `lightning-indexer.cu` has a vector-kernel fallback with F32/BF16/F16 and supported quantized K types (`ggml/src/ggml-cuda/common.cuh:52-53,288-290`; `lightning-indexer.cu:5-7,449-489,490-530,536-588`). Thus its fast WMMA path is unavailable on sm_70, but the op itself is not inherently unavailable if built with sm_70 and a supported cache K type. The inspected GDN and DSV4-HC kernels have no explicit `__CUDA_ARCH__ >= 750` guard (`gated_delta_net.cu:4-219`, `dsv4-hc.cu:8-315`). No custom GLM5 op has a source-level sm_70 hard ban in this snapshot.

## 7. Per-token active parameter and weight-byte accounting

These are calculations from actual GGUF header dimensions, tensor quant types and `tensor.n_bytes`, not a claim that the runtime fetches an entire matrix from storage per token. The active routed-expert estimate uses 8/288 experts in each of the 42 trunk MoE layers; expert slices are equal-sized per tensor. MTP block 45 is excluded from trunk inference totals.

| Category | Active params/token (conventional count) | GGUF weight bytes used/token (approx.) | Actual stored types / scope |
|---|---:|---:|---|
| KDA blocks (34) | 4,682,897,792 | 4,986,234,368 | Q8_0 projections; F32 KDA auxiliary/conv weights |
| MLA attention (11) | 1,291,868,160 | 1,372,676,096 | Q8_0 projections, F32 norms |
| DSA indexers (11) | 82,190,592 | 91,587,584 | Q8_0 plus F32 projector/norm/APE |
| Routed experts (42 layers x 8) | 8,455,716,864 | 5,152,178,176 | all experts stored: 304,405,807,104 params / 185,478,414,336 B; per-token estimate scales by 8/288; Q4_K/Q5_K/Q6_K |
| Shared experts (42) | 1,056,964,608 | 1,123,024,896 | Q8_0 |
| Dense FFN (3) | 452,984,832 | 481,296,384 | Q8_0 |
| mHC (45) | 35,391,870 | 37,611,000 | F32 vectors + Q8_0 mixing matrices |
| Router + norm vectors | 49,930,048 | 199,720,192 | routers/correction biases F32; norms F32 |
| Token embedding | 634,388,480 parameter-table count; one lookup touches 4,096 values | 674,037,760 table; 4,352 B for one Q8_0 row | Separate Q8_0 table `[4096,154880]` |
| Output head | 634,388,480 | 674,037,760 | Separate Q8_0 matrix `[4096,154880]`; full matrix is used for logits |

Conventional model-active count with the full embedding table counted is **17,376,721,726 parameters** (~17.38B, which rounds to the advertised ~18B active). Actual main-trunk weight bytes touched per token, counting one input embedding row and the full output head, are approximately **14,118,370,808 B** (~14.12 GB, ~13.14 GiB); counting the full embedding table as an active parameter table rather than a row adds 674,033,408 B, for 14,792,404,216 B. These byte totals exclude activations, cache traffic, allocator padding, and reuse/cache effects. The size label and total stored expert tensors should not be confused with per-token routed weight use. The 46th block/MTP weights are excluded.

Header sample checks: `output.weight` and `token_embd.weight` each `[4096,154880] Q8_0`; dense block 0 FFN `[4096,12288] Q8_0`; block 3 experts `[4096,2048,288] Q4_K`, `[2048,4096,288] Q5_K`; shared expert `[4096,2048] Q8_0`; KDA block 0 Q/K/V `[4096,8192] Q8_0`, conv `[4,1,8192] F32`; MLA block 3 Q rank `[4096,1536]`, q-b `[1536,16384]`, key-b `[256,512,64]`, value-b `[512,256,64]`; indexer `[4096,128]` and `[4096,32]`.

## 8. Numerical matching checklist

1. Use exactly 45 trunk layers, not 46; layer 45 is separately stored MTP and is not in the main decoder graph.
2. Preserve four independent mHC streams. mHC first RMSNorms the flattened 4x4096 vector; pre has `sigmoid + 1e-6`; post is `2*sigmoid`; comb uses row-softmax, epsilon additions, one column norm then 19 row/column rounds. Do not move mHC outside the attention/FFN branch residual topology.
3. Main RMSNorm epsilon is 1e-5; indexer k uses ordinary LayerNorm epsilon 1e-6 with bias. KDA q/k L2 normalization is equivalent to `x/sqrt(sum(x*x)+1e-6)`; output per-head RMSNorm is epsilon 1e-5.
4. KDA `A` is already `-exp(A_log)` from converter; gate sequence is `g=-5*sigmoid(-(z*A))`, not `softplus(z)*A` for this finite lower-bound model. Then use `exp(g)` in the delta rule. `beta=sigmoid(...)`, and the output attention scale is `1/sqrt(128)`.
5. GDN q/k/v, g, beta, recurrent state, and outputs are F32 by operator contract (`ggml/src/ggml.c:6384-6416`); model cache `type_r/type_s` is F32. Preserve channelwise KDA decay and the state layout/transpose convention.
6. Conv is causal, kernel 4 over three cached history values plus current token, followed by SiLU. It is three independent convs, not a fused QKV convolution or a learned output nonlinearity.
7. MLA has zero rope dimensions. Do not rotate q or k, do not RoPE-permute weights, and do not add a rotary tail to the `kv_cmpr` latent. Attention scale is `1/sqrt(256)`, not `1/sqrt(512)` (the absorbed latent rank is 512, but the MLA key head dimension is 256).
8. DSA score applies ReLU to each query/pool dot product before multiplying by the learned per-head weight and summing. Apply pool/causal mask before top-k. Sort top-k pool ids by descending selected score because `ggml_top_k` itself is unordered. Add at most three live tail cells only when selected by `kpool_select_tail`.
9. MoE bias is selection-only. Select using `sigmoid(router_logits)+exp_probs_b`, but weight using sigmoid probabilities without the correction bias. Normalize the 8 weights before multiplying by 2.5; clamp the sum from below to 6.103515625e-5 first.
10. SwiGLU clamp semantics are asymmetric: clamp gate only above at 10 (`min(gate,10)`), clamp up to `[-10,10]`, then `SiLU(gate)*up`. Do not clamp the gate's lower tail to -10. Both `swiglu_clamp_exp` and `swiglu_clamp_shexp` are 10 in this GGUF.
11. Shared expert output is added after routed expert sum and is not multiplied by router weights. Dense layers use the shared/dense clamp array.
12. The gather DSA path explicitly requests F32 precision for qk matmul and uses `soft_max_ext` with scale 1/16 and zero softcap; match the sparse mask and gather-mask dtype. The fallback MLA path delegates precision to `build_attn_mha` and the selected backend.
13. Quantized GGUF weights are mixed Q4_K/Q5_K/Q6_K/Q8_0/F32; exact outputs depend on dequantization and backend accumulators. Do not expect bitwise equality across kernels. Keep F32 state/decay/softmax accumulations where source specifies them; do not silently downcast cache state to match weight quantization.
14. The MTP body is not executable in this llama.cpp snapshot. Its high-level tensor interpretation is an explicitly labeled inference, not a source-verified forward pass. `index_share_mtp=true` has no implemented consumer here.
