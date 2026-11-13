# Compact committed prompt snapshots

With a small prompt-cache budget, unnecessary bytes in each saved state can evict a reusable conversation prefix.
Patch 0122 adds `TF_GLM_SNAPSHOT_FORMAT=committed`: at `pending == 0`, save each KDA layer's active FP32 recurrent
bank, selected by its saved `cur` bit, and omit the projection/replay scratch. Position, convolution, latent KV,
indexer state, prompt hidden state and available DFlash2 context keep their existing representation. Restore uses
the saved format and per-layer bits even when the destination slot currently has different bits.

The live state layout, arithmetic, tensor types, weights and 4096-token restore grid are unchanged. The default
is `legacy`. The new format supports unsharded TP2 in memory; it refuses disk learning and other rank counts.
The rank fingerprint includes the setting. Disk-image entry points also reject a compact lease, so it cannot be
mistaken for the legacy disk format.

```sh
export TF_GLM_SNAPSHOT_FORMAT=committed
KV_SHARD=0 LEARN_GIB=0 ./start.sh restart
```

To return to legacy packing, unset `TF_GLM_SNAPSHOT_FORMAT` and restart. In-memory kept states expire on restart.
No persisted cache is converted. Use the same weight and precision settings for either mode.

## Current-release measurements

Measured on 2026-11-13 with the **same release binary in both modes** (SHA below), on two DGX Sparks:
original EXL3 weights, dense/head q8, FP8 latent KV, DFlash2 q4, TP2, server parallel 8, context 1,048,576.
The memory settings were `CACHE_FIRST=1`, `MEMORY_RESERVE_GIB=6.5`, `HOST_CACHE_GIB=0`, `KV_SHARD=0`,
`LEARN_GIB=0`, with a 2.5 or 5 GiB device prompt-cache budget per rank. Cache-first, reserve and 5 GiB
match this base's memory defaults. The earlier original-service configuration had reserve 12.5 and no
cache-first switch; these are explicitly changed for **both** formats in this comparison.
Actual `/metrics` budget equalled the requested cap in every arm.

Eight independent synthetic sessions ran **serially**, four turns each, with actual assistant replies in their
histories. Input lengths were 9651/11676/13701/15726 tokens. Each response produced 8 tokens (`max_tokens=32`),
greedy, seed 20261112, thinking off. Each arm started fresh; order was A0/B0/B1/A1 at 2.5 GiB, then the same
order at 5 GiB. Foreground wall sums the 32 HTTP request durations and excludes the fixed idle checks between
requests. Background warming stayed enabled and stepped aside on all 32 attempts in every arm.

| Cache GiB/rank | Arm | Format | Warm hits (turns 2–4) | Foreground wall, s | Rank-0 KV pages |
|---:|---|---|---:|---:|---:|
| 2.5 | A0 | legacy | 2/24 | 227.561 | 6262 |
| 2.5 | B0 | committed | 24/24 | 109.031 | 6263 |
| 2.5 | B1 | committed | 24/24 | 108.736 | 6264 |
| 2.5 | A1 | legacy | 2/24 | 226.564 | 6263 |
| 5 | A0 | legacy | 24/24 | 108.986 | 5253 |
| 5 | B0 | committed | 24/24 | 108.354 | 5388 |
| 5 | B1 | committed | 24/24 | 108.377 | 5361 |
| 5 | A1 | legacy | 24/24 | 108.407 | 5422 |

- **2.5 GiB:** paired wall reductions 52.09% and 52.01%; pooled 52.05%. Warm wall fell 65.47% and warm TTFT
  67.12%. Each committed arm reused 212,992 more input tokens. Cold wall increased 0.85% pooled.
- **5 GiB:** both formats hit 24/24 warm requests. Paired wall reductions were 0.58% and 0.03%; pooled 0.30%.
  The reverse pair's warm wall changed by -0.003%. This does **not** establish a stable speedup at 5 GiB.

All 256 requests matched the first arm's corresponding input history, full `choices`, output token SHA-256,
and prompt/completion/total token counts. An independent offline check reconciled raw responses with completion
logs, cache logs and metrics, verified ordering and identical configurations apart from format/cap, and found
zero cache refusals/failures, client disconnections or preemptions. The compact per-request values and input,
choice and output hashes are in [requests.csv](requests.csv). Full synthetic responses and machine logs were
retained privately; the CSV lets reviewers recompute the reported foreground/TTFT/prefill totals.

This is a **cache-capacity threshold** result on short-output, repetitive synthetic prompts. Two repetitions
per format provide no statistical significance claim. KV page counts vary with available memory at startup
(rank 1 does not emit the same page-count line); the workloads fit, but maximum residency is unqualified.
There is no claim of a decode-kernel speedup, broad model quality, or a healthy whole-model sparkDash gain.
The separate 32K functional check below covers longer outputs and eight concurrent restores.

### Reproduce an arm

Build the stated pin with patches 0001–0122 and keep the same model, tokenizer/template, quantization,
DFlash2 and TP2 settings. Set `cap` to 2.5, then 5. For each cap restart both ranks in the fixed order
`legacy`, `committed`, `committed`, `legacy`; save the startup and request logs for each arm.

```sh
cap=2.5
export TF_GLM_SNAPSHOT_FORMAT=legacy  # select the arm before each restart
PROMPT_CACHE_GIB=$cap CACHE_FIRST=1 MEMORY_RESERVE_GIB=6.5 \
  HOST_CACHE_GIB=0 KV_SHARD=0 LEARN_GIB=0 ./start.sh restart
python3 docs/speed/compact-snapshots/benchmark.py \
  --url http://127.0.0.1:8888 --cap-gib "$cap" --out /tmp/compact-2p5-A0
```

Use a new `--out` directory per arm and add `--reference /tmp/compact-2p5-A0` for every later arm, including
5 GiB. The client requires an unused process (`prompt_tokens_total=0`), checks actual budget, saves full
responses, and fails on an input/output mismatch. It reproduces the measured request history and timing
boundaries; private machine orchestration is not part of the serving recipe. Do not run unrelated GPU work.

## Snapshot space accounting

The CPU probe used the actual loaded model config and the release's real paged FP8 layout/snapshot owners,
with 34 KDA layers and 32 KDA heads per rank. The copied tree's 3,810 `zig/**/*.zig` files matched the build
inputs before replacing only the diagnostic test root. The probe ran one test and produced 12 observations
(six prefixes, both ranks). Both ranks have equal sizes. [sizes.csv](sizes.csv) records exact bytes.

Each committed state omits **71,303,168 bytes (68 MiB)** of inactive FP32 recurrent bank and a further
**72,069,120 bytes (68.73046875 MiB)** of projection/replay scratch per rank. Live buffers remain allocated;
these are saved-state bytes. The table includes a 40 MiB DFlash2 context ring and rounds charges to 2 MiB pages:

| Saved prefix tokens | Legacy MiB/rank | Bank-only counterfactual MiB/rank | Committed MiB/rank |
|---:|---:|---:|---:|
| 4096 | 296 | 228 | 158 |
| 8192 | 342 | 274 | 206 |
| 12288 | 366 | 298 | 230 |
| 16384 | 392 | 324 | 256 |
| 28672 | 466 | 398 | 330 |
| 126976 | 1064 | 996 | 928 |

The 4096-row values equal the minimum-state metric in all eight arms; the 8192-row values equal the first
single-state held charge in all eight runtime logs. Larger rows are owner-computed sizes, not a new GPU
long-context performance measurement. The ring is an upper bound when no usable DFlash context is present.
The bank-only column is an arithmetic counterfactual, **not a benchmark or qualification of PR #42**.

To repeat the accounting in a disposable checkout of the patched engine, copy [size_probe.zig](size_probe.zig)
over `zig/glm_compact_snapshot_host.zig`, copy the loaded model's `config.json` to `zig/release-model-config.json`,
then run `zig build test-glm-compact-snapshots-host -j1 --summary all` with Zig 0.17.0. Require `1/1 tests passed`
and exactly 12 `SIZE,` lines. The driver records addresses without allocating GPU memory. This test-root
replacement is for accounting only; do not use that diagnostic tree to build the serving binary.

## Release integration

Base: recipe `80be4d6882837d8e0840827ae2aab6124fd48da3`, TensorFold
`f350cd3ea253a042e205746928271f9d8f71a82d` plus patches 0001–0121. Patch 0122 adapts packing to the current owners,
retaining the complete Plan metadata and legacy KV-sharding/disk paths. It is independent of PR #42.

- Zig 0.17.0: `test-glm-compact-snapshots-host`, 6/6 passed. Actual snapshot owners with the recording CUDA driver:
  poisoned omitted regions, saved parity, window/paged FP8 transfer, device/host copies, DFlash2 ring, mixed layer
  parity, failed-copy cleanup, invalid formats, disk rejection and rank configuration agreement.
- `test-cuda-disk-open-host`, 9/9 passed, including committed-format admission through GLM's actual
  `TF_GLM_WORLD`/`TF_GLM_RANK` parser, invalid rank/world refusal and disk-learning refusal. These are host checks.
- `test-glm-snapshots-host`: the initial integration's control and candidate both 22/26. The same four failures remain: minimum-prefix sizing
  on the old 2048-row fixture, the borrowed-state store hit, the 0.3 GiB fixture (expected 203423744, actual
  297795584), and the host-tier store hit. This suite is not green. The final revision changed only native admission
  and its test; the six compact-owner checks and nine disk/open checks above passed for that revision.
- `zig build native -j2 -Doptimize=fast -Dcpu=native -Dstrip=false -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
  -Dxgrammar=/src/xgrammar`: passed, with all CUDA assets rebuilt. CUDA 13.3.73;
  xgrammar `97787376faee5ed8466cfad57c99855e4ce2f6aa` (v0.2.8).
- Release binary SHA-256: `b64a3858b8b823852a73e25d7e0dd42971b33cf4c19247238b5c8e185cec1aa3`.
- All 122 patches apply to a fresh archive of the full pin; 5056 Zig/build source files match the native build
  inputs byte for byte. `bash scripts/tests/run.sh` passed, including 11 new configuration cases. Shellcheck was
  unavailable and skipped. Text was exported with LF from Git to avoid Windows checkout conversion.

On 2026-11-13 the release binary was baked into the original 1104 runtime image on both Sparks, with its newly
compiled, embedded SM121 assets. This validates that runtime derivative; a clean rebuild of the entire recipe
Dockerfile was not performed. The weights, dense/head q8, FP8 latent KV, DFlash2 q4, TP2 and parallel 8 were retained,
with a 2.5 GiB prompt cache per rank. The same binary was started in legacy mode, tested and stopped, then started
in committed mode. The committed pair remained healthy after the checks.

| Mode and request | Responses | Cached input tokens per response |
|---|---:|---:|
| legacy cold / warm | 1 / 1 | 0 / 28672 |
| committed cold / warm | 1 / 1 | 0 / 28672 |
| committed, eight concurrent warm requests | 8 | 28672 |

All 12 responses had 32070 input tokens and 128 output tokens, matching full `choices` and output-token SHA-256
`cd9edb806d60e7074f0200751e6ed6b8c3f6ab3d8031c78c20adf2dd4557b650`. This is a functional restore and fallback check;
it complements the release ABBA above; maximum residency and general model quality remain unqualified.

Generate the exact synthetic request with `python3 docs/speed/compact-snapshots/request.py > /tmp/compact-request.json`.
After each fresh mode start, send it twice (cold, then warm), saving each full response:

```sh
curl -fsS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @/tmp/compact-request.json
```

For committed mode, then send eight copies concurrently. Compare `choices`, `usage.prompt_tokens`,
`usage.completion_tokens` and `tensorfold.token_sha256` against the legacy cold response; inspect
`usage.prompt_tokens_details.cached_tokens` against the table. Timing fields and request IDs naturally differ.

## Relationship to PR #42

[PR #42](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/pull/42) includes an upstream live-half port
and a shared-page store. Its live-half path keeps both banks when layer parity differs and retains replay scratch.
This patch uses the saved bit for each layer and also removes the committed window's replay scratch. It does not
include the shared-page store or claim qualification with it. Both changes touch snapshot owners and rank
configuration, so whichever lands second needs integration and validation against the resulting main. Patch
number 0122 is the next number on this PR's base and must be updated if main advances before merge.

A patch-application check against #42 head `5b598c2241e4836ad5fb62d6dbe091930652b6e8` confirms text conflicts:
ours first prevents #42's 0125 from applying to `cuda_snapshots.zig`; #42 first prevents this patch from applying
to `cuda_snapshots.zig`, `cuda_ranks.zig` and `cuda_refusal.zig`. Each stack applies independently to the stated base.
