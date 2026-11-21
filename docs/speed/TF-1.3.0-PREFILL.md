# TensorFold 1.3.0 on this pair: where the prompt time went, and what brings it back

Measured on the two Sparks (`dgxspark5` = 10.0.4.2, `dgxspark4` = 10.0.4.1), GLM-5.3-Flash EXL3 4 bpw, TP=2,
`MEMORY_RESERVE_GIB=12`, `KV_SHARD=1`, `DRAFTER=dflash2`, GPU clock capped at 1,982 MHz, 63-70 °C.
Every number is a cold prompt (`cached 0`) through `tools/bench.py` of the DeepSeek-V4.1 kit against `:8888`,
2 reps, medians unless stated; `greedy` is the six-prompt probe (short/math/code/jp/think/tool).

## The port, and what it cost

`a691916` (this repository's 1.3.0 port: engine `81de26f6`, 16 engine patches) replaces the 121-patch 1.1.9
stack with a much smaller one. On this pair it drops prompt throughput by a quarter:

| build | prefill 8K | prefill 32K | TTFT 32K | decode | multi-4 |
|---|---:|---:|---:|---:|---:|
| `d114466` (1.1.9 + 121 patches, what this repository ships) | 1,584 | **1,719-1,724** | 18.3 s | 42.0 | 82.2 |
| `a691916` (1.3.0 + 16 patches) as-is | 1,235 | **1,277** | 24.6 s | 39.0 | 75.3 |

Decode loses 3 points; the prompt loses 26 %. Three of those points are settings and kernel details the port did not
carry, and each is a patch here:

| patch | what it restores | measured |
|---|---|---|
| `0017` — `prefill_rows` 2,048 → 4,096 | the 4,096-row prompt pass the 1.1.9 stack used (0094-0097) | **+2.7 %** (1,374 → 1,411) |
| `0018` — `absorb_q` / `expand_v` on tensor cores | 1.3.0's `cuda_latent.absorbQ`/`expandV` always take the SIMT `rowBlocks`; `TF_GLM_ABSORB_MMA` is read nowhere in the tree | **+7.4 %** (1,280 → 1,374), matching upstream's published +8.0 % |
| `0020` — the FP8 key-tile pipeline in `common/mla.cuh` | 1.3.0 loads an E4M3 tile with two byte loads a code pair, converts, stores; 1.1.9 issued one 8-byte `__ldg` a unit and fetched tile + 2 ahead (`fetch8`/`store8`) | **+7.2 %** (1,411 → 1,512) |

Together: **1,277 → 1,512 (32K) and 1,274 → 1,526 (64K), +18 %, with the six greedy replies byte-identical
to the 1.1.9 build's.** TTFT 32K 24.6 s → 20.8 s.

One patch that looks like the same class of loss and is **not**: `0019` restored the fused sparse-chunk kernel
(`tf_glm_chunks_*_fused`, the old `fuses()` selection, 13 chunk variants against 1.3.0's 6) and measured **0 %**
(1,365 / 1,380 against 1,374 / 1,379). Do not port it again.

## What is left (-10 %), and why it is not another kernel

Both builds profiled with Nsight Systems on the same 32K cold prompt, capture bracketed to the request
(`--delay`/`--duration`; a whole-session capture mixes in the load and the start smoke test and reverses these totals):

| | 1.1.9 + 121 | 1.3.0 + 0017/0018/0020 |
|---|---:|---:|
| wall (TTFT) | 17.99 s | 20.55 s |
| GPU kernel time, all streams | 23.12 s | 23.41 s (+1.3 %) |
| **overlap (GPU time ÷ wall)** | **1.29×** | **1.14×** |
| 50 ms buckets with no kernel at all | 39 % | 0 % |
| `cuStreamSynchronize` | 82 calls / 5.02 s | 39 calls / 7.19 s |

The GPU does the same work and never idles in 1.3.0 — it simply stops running the transport beside the compute.
The per-stream split names the mechanism:

| stream | 1.1.9 + 121 | 1.3.0 |
|---|---|---|
| main (18) | 18.06 s | 20.02 s |
| side (19) | 5.05 s = `SendRecv` 3.91 s + **`hc_post` 0.56 s + `hc_partial` 0.36 s** | 3.39 s = `SendRecv` only |

1.1.9 runs the glue kernels (`hc_pre`/`hc_post`/`hc_partial`/`hc_finish`) on the pieces' side stream, under the
main stream's mixer/ffn. 1.3.0 runs them on the main stream: +1.96 s of serial work, which is the whole wall gap.
The side-stream path exists only for a **paired prompt step** (two chunks of one sequence, each chunk's exchange and
glue under the other's sites), and 1.3.0 refuses that step whenever a drafter is loaded
(`cuda_prompt_pair.pairedAt`: `e.drafter != null`), because the port also dropped the paired path's DFlash2 taps
(`tapPiece`) and made `pairBody` return `BadPair` when taps exist. This pair runs `DRAFTER=dflash2`, so it lands in
the serial path by default. `0023` + `0024` give the paired path its taps back, which is what lets the glue return
to the side stream with a drafter loaded:

| patch | what it does | measured |
|---|---|---|
| `0023` | `pairedAt` and the paired chunk stop refusing a loaded drafter | (with 0024) |
| `0024` | restores `tapPiece` and the per-piece taps `exchangeGlue` issued on the side stream, and `shareTaps` after both chunks; drops `f.taps != null` from `pairBody`'s `BadPair` gate | **32K 1,512 → 1,554, 64K 1,526 → 1,572** (+2.8 % / +3.0 %); decode 39.5-41.4 and multi-4 74-75 against 40.4 / 80.0 (inside this pair's run-to-run spread); the six greedy replies stay byte-identical |

Best 1.3.0 build here is therefore `0017` + `0018` + `0020` + `0023` + `0024`: **1,554 (32K) / 1,572 (64K)**
against the 121-patch build's 1,719-1,724 — **-9 %**, with identical greedy replies.

## Flags that look relevant and are not (all measured, 32K cold prompt)

`TF_GLM_PDL=0` 1,529 · `TF_GLM_INDEX_CUDA=0` 1,528 · `TF_GLM_PROMPT_STEPS=0` 1,520 · `TF_GLM_SHARED=0` 1,530 ·
`TF_GLM_LINK_WAIT=0` 1,528 · `TF_GLM_OVERLAP_PIECES=8` 1,534 (decode 40.4 → 39.3) · `TF_GLM_CHUNK_SOLO=4096`
**1,423** (it doubles the prompt buffers; it only starts with `PROMPT_CACHE_MIN_GIB` lowered to 4) ·
`TF_GLM_EXPERT_FORM=classic` 1,512-1,526 · `DRAFTER=none` 1,590 but decode **40.4 → 25.8** (that +4.5 % is the
drafter's own prompt work leaving, the same gain measured on its own). Baselines for all of the above: 1,512-1,530.

`TF_GLM_PDL_CHAIN`, `TF_EXL3_PDL`, `TF_GLM_DSEL`, `TF_GLM_TOPK_SUM`, `TF_GLM_DSCORES`, `TF_EXL3_SHFL` and
`TF_GLM_MM_TWO` are inert too (1,280-1,284 at stock 1.3.0, i.e. inside the noise floor).

`TF_GLM_CHUNK_SOLO`, `TF_GLM_PROMPT_PAIR` and `TF_GLM_OVERLAP_PIECES` are read in `cuda_ranks.zig`
(`s.pair` defaults **true**, `solo` false, `pieces` 3), not in `platform/environment.zig` — grep the whole tree.

## Reproducing

```bash
# images: tensorfold-glm53-spark:800ae46bf8c4 (1.1.9 + 121) and tensorfold-glm53-spark:<built> (1.3.0 + patches)
git checkout a691916 && ./scripts/prepare.sh --image && ./start.sh restart
python3 -I ../DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/tools/bench.py --url http://127.0.0.1:8888/v1 \
  --model GLM-5.3-Flash --suite prefill --prefill-sizes 32768,65536 --reps 2
```

The Nsight Systems recipe (host `nsys` bind-mounted into the container, `libbpf`/`libcap`/`libelf` mounts, the
`target-linux-sbsa-armv8` path, `--delay`/`--duration` to bracket one request) is in the `gpu-kernel-tuning` skill's
`references/nsys-in-container.md`.
