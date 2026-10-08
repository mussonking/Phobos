# GLM-5.3-Flash on Phobos

Phobos (a fork of Strata) runs **GLM-5.3-Flash** (288 experts, 8 active per token, 4-bit IQ4 weights) on one NVIDIA card plus a large
amount of system RAM. The attention trunk and the most used experts live on the GPU; the other experts stay in RAM and
the CPU computes them. You get a local server with a web chat and OpenAI / Anthropic compatible endpoints, usable from
agents such as Claude Code.

This is a workstation / server setup, not the gaming-PC setup of Strata's Qwen models: the experts alone take
~165 GB of RAM.

## Requirements

| | minimum | better |
|---|---|---|
| OS | Linux (x86-64); Windows through WSL2 is being verified | |
| GPU | NVIDIA, compute capability 8.0+ (RTX 30 / 40 / 50, A-series), 24 GB VRAM | more VRAM: more experts on the GPU |
| | sm_70 (V100) builds as experimental (`-DSTRATA_EXPERIMENTAL_SM75=ON`) | |
| RAM | 192 GB | 256 GB+ (room for the prefix cache) |
| CPU | AVX2 | AVX-512 VNNI (Xeon Cascade Lake or newer, AMD Zen 4 / Zen 5) |
| Disk | ~180 GB free for the model file, NVMe | |

- **192 GB of RAM rules out most desktop platforms.** AM4 and LGA1200 boards take 128 GB at most. Realistic desktops:
  - AMD AM5 (Zen 4 / Zen 5): has AVX-512 VNNI, so it runs the full VNNI path.
  - Intel LGA1700 / LGA1851: no AVX-512, so it runs the AVX2 path today (AVX-VNNI kernels are planned).
- **RAM bandwidth sets the speed.** Each generated token reads the experts the GPU does not hold from RAM (~3 GB per
  token on our machine). More memory channels = faster.
- **AVX-512 VNNI or AVX2.** The CPU computes its experts with int8 kernels: VNNI on the R8 expert layout when the
  CPU has AVX-512 VNNI, AVX2 on the plain GGUF layout otherwise (`GLM_CPU_ISA=auto|vnni|avx2|scalar` forces one).
  Both paths are validated on long conversations (95-113 K tokens of context).
- **192 GB of RAM** runs the model with the prefix cache effectively off: it is capped at free RAM minus 64 GB.
- **Windows: WSL2 is being verified.** On an RTX 3080 10 GB (sm_86) under WSL2, a mini model (the real engine, a few
  experts per layer) passes all 18 checks of our test package (v13.1, Oct 8), the prompt reading through the WSL2
  pinned-memory path included. The full model under WSL2 is untested. Native Windows is not supported.
- One GPU only.

## How fast is it?

### Measured

On our machine: 2 x Xeon Gold 6252 (12 DDR4-2933 channels in total, 2 NUMA nodes), 376 GB RAM, one Tesla V100 32 GB
(sm_70, the experimental path, PCIe 3). It holds 1353 experts (17.2 GB) in VRAM, with 2 conversations of 131072
tokens each. Quiet machine unless noted.

Writing (decode), by CPU kernel path:

| CPU path | tokens/s |
|---|---|
| AVX-512 VNNI, R8 layout (the default on this CPU) | 26.0 |
| AVX-512 VNNI, GGUF layout | ~21 |
| AVX2 (AVX-512 switched off with `GLM_CPU_ISA=avx2`) | 21.1 |

Time to the first token for a follow-up turn of N new tokens:

| new tokens | VNNI | AVX2 |
|---|---|---|
| 200 | 1.1 s | 12.3 s |
| 800 | 2.3 s | 14.8 s |
| 2000 | 4.3 s | 16.3 s |

The AVX2 path uploads its prompt experts to the GPU instead of sharing the work with the CPU, so here it is bound by
PCIe 3.

Other measurements (under background load, so indicative; `--spec 1`):

| | VNNI, R8 | AVX2 |
|---|---|---|
| writing at 95-113 K tokens of context | 21-25 tokens/s | 22-25 tokens/s |
| reading a new long prompt (cold) | ~600-680 tokens/s (a 113 K-token prompt: 166 s) | ~440 tokens/s |
| loading the model at start | ~2 min (more on a slower disk) | not measured |

At 95-113 K tokens, a whole AVX2 turn (reading the prompt, then writing) takes ~1.3-1.4x as long as on VNNI.

The same long contexts on a quiet machine (VNNI, R8, `--spec 1`; 6 needle-in-a-haystack prompts of 95 K and 113 K
tokens, Oct 7):

| | quiet | under load (above) |
|---|---|---|
| writing | 29-33 tokens/s | 21-25 tokens/s |
| reading a new long prompt (cold) | 616-626 tokens/s (a 113 K-token prompt: 181 s) | ~600-680 tokens/s |

The needle answers are short and predictable, which favours speculative decoding: expect less on free-form writing.

### Projected (not measured)

- **AVX2 on PCIe 4:** first token after 200 / 800 / 2000 new tokens in ~6.3 / 7.4 / 7.8 s.
- **Desktop memory is the floor.** 2-channel DDR5 gives ~55-60 GB/s in practice, against ~280 GB/s peak here. That
  limits every placement of the experts:
  - Writing: roughly **10-15 tokens/s**. A 24 GB card also holds fewer experts than our 32 GB one.
  - The VNNI path's first token on short prompts: about 3x slower than our 1.1 s.

We will replace these projections with measurements.

## Get the model

From [mussonking/GLM-5.3-Flash-Phobos-GGUF](https://huggingface.co/mussonking/GLM-5.3-Flash-Phobos-GGUF):

- `GLM-5.3-Flash-Phobos-IQ4.gguf` (179.1 GB): the weights, the IQ4 quantization made for this engine.
- `expert-profile-agent.strp` (optional, 100 KB): which experts to put on the GPU first, measured on agent traffic.
  Without it the engine starts from its own ranking, and the adaptive tier learns from your traffic.
- `tokenizer/`: the tokenizer and chat template, already extracted from the GGUF.

```sh
pip install -U huggingface_hub
hf download mussonking/GLM-5.3-Flash-Phobos-GGUF \
    --include "GLM-5.3-Flash-Phobos-IQ4.gguf" "expert-profile-agent.strp" "tokenizer/*" --local-dir /models/glm
```

`SHA256SUMS` in the same repo lists the checksums.

## Build

```sh
git clone https://github.com/mussonking/Phobos
cd Phobos
python3 -m venv .venv && . .venv/bin/activate && pip install jinja2 cmake ninja
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target glm_serve
```

- `CMAKE_CUDA_ARCHITECTURES`: your card's compute capability without the dot (RTX 30: 86, RTX 40: 89, RTX 50: 120,
  A100: 80). V100: `70` plus `-DSTRATA_EXPERIMENTAL_SM75=ON`.
- Needs Python 3.10+, the CUDA toolkit (12.x) and a C++20 compiler. CMake fetches the pinned ggml (llama.cpp) commit
  on the first configure.

The tokenizer comes with the download. To extract it from the GGUF yourself instead:

```sh
.venv/bin/python tools/strata_tokenizer.py --gguf /models/glm/GLM-5.3-Flash-Phobos-IQ4.gguf --out /models/glm
# writes /models/glm/tokenizer/ (vocab, merges, chat_template.jinja)
```

## Start it

Write a config, e.g. `glm.json`:

```json
{
  "exe": "build/src/glm/glm_serve",
  "args": ["--gguf", "/models/glm/GLM-5.3-Flash-Phobos-IQ4.gguf", "--profile", "/models/glm/expert-profile-agent.strp",
           "--ctx", "131072", "--sessions", "2", "--kv-latent", "auto", "--spec", "1",
           "--prefix-cache-gb", "48"],
  "tokenizer": "/models/glm/tokenizer",
  "family": "glm",
  "model_name": "<the name clients see in /v1/models>",
  "fit_max_tokens": true,
  "log": "glm-serve.log"
}
```

```sh
STRATA_API_KEY=<a key you choose> .venv/bin/python -m serve.server --engine strata --config glm.json --port 8142
```

- Loading takes a few minutes; `curl http://127.0.0.1:8142/health` answers once it is ready.
- The server listens on `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in the config) opens it to your
  network: set an API key then.
- Stop it with Ctrl+C or SIGTERM. The engine needs ~20-25 s to release its memory before it exits.

| engine argument | meaning |
|---|---|
| `--ctx N` | tokens per conversation, prompt + answer (131072 max tested) |
| `--sessions N` | conversations kept resident at once (default 2: an agent's main conversation plus its side requests) |
| `--kv-latent auto\|f16\|q8` | attention cache precision; `auto` picks int8 at long contexts |
| `--spec 1` | MTP speculative decoding: one drafted token per step (`--spec 0` turns it off) |
| `--prefix-cache-gb N` | host RAM for conversations that left the GPU (a ceiling, used on demand; 0 = off) |
| `--threads N` | CPU threads for the experts (default: one per physical core; the engine log shows the count) |
| `--no-r8` | keep the plain GGUF expert layout in RAM |

## Use it

**Web chat.** Open `http://127.0.0.1:8142/`. Settings shows the GPU, the engine and how many experts sit in VRAM.
The Thinking setting offers Low, High (default) and Max.

**Endpoints.** `/v1/*` needs `Authorization: Bearer <key>` (or `x-api-key: <key>`) when a key is set; `/health` is
open.

| endpoint | |
|---|---|
| `GET /health` | ready state, context size, expert layout, prefix cache counters |
| `GET /v1/models` | the model name |
| `POST /v1/chat/completions` | OpenAI: streaming, tool calls, `usage`, the thinking in `reasoning_content` |
| `POST /v1/messages` | Anthropic: streaming, tools, thinking blocks |

The `model` field of a request is ignored (one model per server).

**Claude Code and other agents.**

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8142 ANTHROPIC_API_KEY=<your key>
export API_TIMEOUT_MS=3600000 CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
claude
```

The first request reads the whole system prompt and the tool list (~11 K tokens for Claude Code: ~20 s); later turns
read only what is new. Keep the context limit of the client under the server's (we use 122880 + 8192 output with
`--ctx 131072`).

## Thinking effort

GLM always thinks before it answers. Its template has three levels:

| effort | API value | behaviour (our tests, 2 K-token cap) |
|---|---|---|
| Low | `low`, also `none` / `disabled` | short thinking; complete answers in 10-40 s |
| High | `high` (also `medium`) | complete answers in 10-40 s; the web chat's default |
| Max | `max`, or no effort given | can think for minutes: 2 K to 8 K+ tokens even for a short story |

- OpenAI: `"reasoning_effort": "low"`. Anthropic: `"output_config": {"effort": "low"}`, or a `thinking` budget
  (under 2048 tokens = Low, otherwise High).
- A request that names no effort gets **Max**, the model's own default. Set the effort, and a `max_tokens` large
  enough for thinking plus answer, or a long thinking can use the whole budget and leave the answer empty.
- Thinking cannot be turned off (the template has no switch for it): a request for no thinking (`none`,
  `disabled`, `enable_thinking: false`, Anthropic `thinking: {"type": "disabled"}`) gets Low.

## Prefix cache

A conversation's state stays on the GPU between turns, so a follow-up reads only the new tokens. With
`--sessions 2`, two conversations stay resident; a third one evicts the least recently used. The evicted state goes
to the host prefix cache (`--prefix-cache-gb`), and coming back costs tens of milliseconds instead of re-reading the
whole prompt. `/health` reports `"prefix_cache"` (entries, hits, misses, MB used).

## Limits

- Greedy decoding only: temperature, top-p, top-k and seed are ignored (the web chat hides them).
- One GPU. No multi-GPU split for GLM.
- One request generates at a time; others wait in line.
- Output is not bit-reproducible by default: which experts sit on the GPU changes with the traffic, and a GPU and a
  CPU expert round differently. `--prefill-exact --adapt-every 0 --slots N` gives a reproducible mode.
- Linux. Windows through WSL2 is being verified; no native Windows.

## License and attribution

- Phobos: MIT ([LICENSE](../LICENSE)). Copyright (c) 2026 Niko1221 and the Strata contributors; the GLM-5.3-Flash
  engine Copyright (c) 2026 mussonking.
- GLM-5.3-Flash weights, tokenizer and chat template: Z.ai, MIT
  ([zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)). The IQ4 GGUF was made with llama.cpp's
  `llama-quantize` from unsloth's BF16 GGUF, with unsloth's importance matrix (MIT).
- llama.cpp / ggml (MIT): the i-quant formats and their codebooks, the `glm5-next` implementation our forward pass
  was written from ([glm5next-spec.md](glm5next-spec.md)); see `third_party/ggml/LICENSE`.
- The full credits, the research papers included, are in the [README](../README.md#credits).
