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

Together the first three are **1,277 → 1,512 (32K) and 1,274 → 1,526 (64K), +18 %**, with the six greedy replies
byte-identical to the 1.1.9 build's; the paired-path and GEMM patches below take it to **1,618 / 1,628 (+27 %)**. TTFT 32K 24.6 s → 20.8 s.

One patch that looks like the same class of loss and is **not**: `0019` restored the fused sparse-chunk kernel
(`tf_glm_chunks_*_fused`, the old `fuses()` selection, 13 chunk variants against 1.3.0's 6) and measured **0 %**
(1,365 / 1,380 against 1,374 / 1,379). Do not port it again.

## Where the rest of the gap went: first the overlap, then the GPU work

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

Best 1.3.0 build here is therefore `0017` + `0018` + `0020` + `0023` + `0024` + `0025`: **1,618 (32K) / 1,628 (64K)**
against the 121-patch build's 1,719-1,724 — **-6 %**, with identical greedy replies, decode 40.9 and multi-4 78.3
against 42.0 / 82.2.

Profiled again after `0023` + `0024`: the side stream carries `SendRecv` 4.71 s + `hc_post` 0.52 s +
`hc_partial` 0.29 s + `hc_finish` 0.15 s, the main stream 20.09 s, and the overlap factor is back to **1.29×**.
After `0025` the totals are **24.20 s of GPU work in 19.40 s (1.25×)** against 1.1.9's 23.12 s in 17.99 s (1.29×), so
what is left is mostly GPU work again, in three small places: `tf_exl3_prompt_gu` 3.55 s against 3.08 (the expert
prompt GEMM is ~15 % slower a row), `tf_glm_merge` 0.66 s against 0.11 — it is launched as a dependent kernel now, so
part of that number is the wait it clocks rather than work — and `tf_glm_combine` 1.09 against 1.05. `SendRecv` is
in fact cheaper than 1.1.9's (3.61 against 3.91).

The last piece is the packed prompt GEMM. 1.1.9 fed 4/6/8-bit prompt weights to a 128-row kernel with a two-stage
pipeline and `__launch_bounds__(256, 2)` (41 KiB of shared memory: two blocks an SM, so a 768-row piece's 384 blocks
fit one wave of 376 slots). 1.3.0 dropped that variant and sends the same matmuls through the three-stage bucket
(60 KiB, one block an SM). Restoring the shape on 1.3.0's own template (`0025`) is worth
**1,554 → 1,618 (32K) and 1,572 → 1,628 (64K)**, +4.1 % / +3.6 %, greedy identical.

## Flags that look relevant and are not (all measured, 32K cold prompt)

`TF_GLM_PDL=0` 1,529 · `TF_GLM_INDEX_CUDA=0` 1,528 · `TF_GLM_PROMPT_STEPS=0` 1,520 · `TF_GLM_SHARED=0` 1,530 ·
`TF_GLM_LINK_WAIT=0` 1,528 · `TF_GLM_OVERLAP_PIECES=8` 1,534 (decode 40.4 → 39.3) · `TF_GLM_CHUNK_SOLO=4096`
**1,423** (it doubles the prompt buffers; it only starts with `PROMPT_CACHE_MIN_GIB` lowered to 4) ·
`TF_GLM_EXPERT_FORM=classic` 1,512-1,526 · `DRAFTER=none` 1,590 but decode **40.4 → 25.8** (that +4.5 % is the
drafter's own prompt work leaving, the same gain measured on its own). Baselines for all of the above: 1,512-1,530.

`TF_GLM_PDL_CHAIN`, `TF_EXL3_PDL`, `TF_GLM_DSEL`, `TF_GLM_TOPK_SUM`, `TF_GLM_DSCORES`, `TF_EXL3_SHFL` and
`TF_GLM_MM_TWO` are inert too (1,280-1,284 at stock 1.3.0, i.e. inside the noise floor). Re-measured **with the paired
step enabled** (the first round was taken without it): `TF_GLM_OVERLAP_PIECES=8` 1,606 and
`TF_GLM_PROMPT_STEPS=0` **1,537 (-5 %)** against a 1,617 baseline, so 1.3.0's own chunk interleaving and three pieces
are already the right setting.

One negative result worth recording: the 1104 stack's `exl3_prompt.cuh` main loop is 13 % faster a row-layer on
`tf_exl3_prompt_gu` (2.33 ns against 2.63), but only as a set — its 8-member column-block specialisation *and* its
two-set trellis decode. Taking the decode pairing alone into 1.3.0's `tmain`, keeping its slot-write interface, `cnt`,
`stage` and `acc[TMT][TNB]`, measured **1,589 / 1,597 against 1,618 / 1,628: a regression**, because the second A set
costs registers in a kernel pinned at two blocks an SM. `TSTAGES` is 3 in both trees, so pipeline depth is not the
difference either; restoring the whole loop would mean reverting the slot-write prompt experts, which is a design
reversal rather than a patch.

`TF_GLM_CHUNK_SOLO`, `TF_GLM_PROMPT_PAIR` and `TF_GLM_OVERLAP_PIECES` are read in `cuda_ranks.zig`
(`s.pair` defaults **true**, `solo` false, `pieces` 3), not in `platform/environment.zig` — grep the whole tree.

## Configuration room, measured (nothing here needs an engine change)

Warm prompts are at parity between stacks: same 12,493-token prompt, `cached 12,288`, **0.28-0.31 s on both**; 25K
0.40-0.66 s both; five turns 0.29-0.33 s a turn each. So the -6 % below is cold-prompt only.

Every configuration lever either does nothing or trades concurrency for prompt-cache room: `PARALLEL` 8 → 7.6 GiB of
cache, 6 → 8.5, 4 → 9.4, 2 → 10.2 (and multi-4 falls to 55.8 tok/s); `MEMORY_RESERVE_GIB=10` → 9.1 GiB with multi-4 at
76.8. Prefill sits at 1,618-1,623 in all of them. Asking for `PROMPT_CACHE_GIB=16` with `MIN=12` refuses to start
(`E_CACHE`), and `KV_SHARD=0` refuses too (4.3 GiB funded against `PROMPT_CACHE_MIN_GIB=6`). `TF_GLM_L2PF=1`
(decode 39.9, multi-4 77.6) and `TF_ROCE=1` (38.0, 79.9) are inside the noise of 40.0 / 81.0, and RoCE is the default
here already. With 7.6 GiB holding roughly two hundred 30K-token states, cache capacity is not the constraint.

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
