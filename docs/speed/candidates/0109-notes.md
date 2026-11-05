# 0109: the busy prompt step (upstream's `TF_GLM_CHUNK_SOLO`, ported onto the 4,096-row pin)

Engine patch `patches/engine/0109-glm-cuda-TF_GLM_CHUNK_SOLO-...patch`, written 2026-11-04 against `f350cd3e` + patches
0001-0108 (engine branch `pin-chunk-solo` on `tp4-mesh-roce`), **not built or run on hardware** (no Zig toolchain where it
was written; reviewed for compile-correctness by reading). A candidate: the gates below decide whether it ships, and the
recipe's README/CHANGELOG are untouched until they are measured.

## What upstream did, and what was ported

| Upstream (ashhart/TensorFold `main`) | What |
| --- | --- |
| `04171ccf` glm cuda: TF_GLM_CHUNK_SOLO takes two grid chunks in one pass while no other stream decodes (#1823, option C of #1554; v1.2.1) | the mechanism: `cuda_prompt_solo.zig` (`othersDecode`, `takesTwo`, `ringsAgree`, `at`/`allowed`, `mtpRows`), `Leader.chunk` returning `Chunk{first, rows}`, `Lanes.prefillStepFn` passing `!othersDecode`, `Settings.solo` + fingerprint field, `glm-solo-check` |
| `d229c4f0` TF_GLM_CHUNK_SOLO=4096 beside --drafter is refused by name (#1842) | **not ported** (no counterpart, see below) |
| `d6e9569a`, `2914b688`, `d0695af5` docs: the recipe table and the TP=3 / ROCE_MULTI notes | the numbers quoted here |

Upstream's grid is 2,048 rows: `prefill_rows` = 2,048, `index_ring` = 4,096, kept prompt states cut every 2,048. Unset,
every pass is 2,048; `TF_GLM_CHUNK_SOLO=4096` lets rank 0 take **two grid chunks in one pass** while no other stream is
past its prompt (a lane with `fill == null`), sized at load (`Buffers.init(solo_rows)`, +1.6 to 1.9 GiB), and every
rank must agree (fingerprint). Their measurement (two GB10s, TP=2, XSFP4, `--parallel 4`): cold first token 6 to 9 %
sooner from 8K to 128K; a decoding stream's gaps beside a 32K prompt unchanged (median 1,474 vs 1,478 ms, the busy pass
is still 2,048); a request arriving during a solo pass waits for it (1.27 -> 2.48 s mean).

**How the pin differs.** Patches 0094/0095 put this tree's grid at 4,096 (`cuda_state.prefill_rows`), with `index_ring`
8,192, `dflash.MAX_CTX` 4,096, `cuda_native.snap_grid` 4,096, and patches 0050-0062 already give a step alone **two**
4,096-row chunks in one pass (`TF_GLM_PROMPT_PAIR`, on by default, same `Lanes.othersDecode` predicate). So upstream's
"solo pass = two grid chunks" is already here, one grid size up, and the thing this tree lacks is the other half of
upstream's behaviour: **a short pass while a decoder is present**. Pasting upstream's code would have moved the grid to
2,048 (every kept state cut, every resume, `SnapshotOutOfStep`) and dropped the pair. The port therefore inverts the
direction and keeps the grid:

| | upstream `main` | this pin before 0109 | this pin with 0109, `CHUNK_SOLO=1` |
| --- | --- | --- | --- |
| kept-state grid / `prefill_rows` | 2,048 | 4,096 | 4,096 (unchanged) |
| step alone (no other stream decoding) | 2,048, or 4,096 with `TF_GLM_CHUNK_SOLO=4096` | 4,096, or a pair of 4,096 (8,192) | 4,096, or a pair (unchanged) |
| step beside a decoding stream | 2,048 | 4,096 | `TF_GLM_CHUNK` rows: 2,048 (default) or 1,024 |
| extra memory | +1.6-1.9 GiB prompt buffers | - | none (a busy step is part of the 4,096-row buffers) |
| `--drafter` | refused beside the setting (#1842) | - | allowed: DFlash2 ingests each step as a paired step's second chunk (0057, `dflash.promptRun`) |

## Knobs and defaults

| Recipe (`scripts/config.sh`) | Engine (every rank, fingerprinted as `chunk`, layout 15) | Default |
| --- | --- | --- |
| `CHUNK_SOLO=1` | `TF_GLM_CHUNK_SOLO=4096` (upstream's name and its only "on" value; `2048`, upstream's off, is refused by name because this grid cannot spell it) | `0`: unset, every step a whole chunk, bit-for-bit today's engine path (the new code runs only on `settings.chunk != 0`) |
| `CHUNK_BUSY=2048 \| 1024` | `TF_GLM_CHUNK` (needs the switch; the role of upstream's `TF_FLASHNEXT_CHUNK`) | `2048`, upstream's busy pass |

Both start scripts add `-e TF_GLM_CHUNK_SOLO=4096 -e TF_GLM_CHUNK=$CHUNK_BUSY` in `build_env_args` only when
`CHUNK_SOLO=1` (as `TF_GLM_HOST_CACHE_GIB` / `TENSORFOLD_PREFILL_BUDGET` are added), and `prepare.sh` validates the two
values. Rank 0 logs at load: `glm: prompt steps of N rows while another stream decodes, 4096 alone (TF_GLM_CHUNK_SOLO)`.

## Mechanism (engine, `zig/src/families/glm/`)

- `cuda_prompt_solo.zig` (new): `busyAt(e, start, len, solo, from)` is rank 0's choice for a step: the setting is on,
  `solo` is false (`Lanes.othersDecode`: another lane has `fill == null`), nothing reads a chunk's rows (no
  `prompt_logprobs`, no image rows, not the 2048-row `TF_GLM_PROF` bucket: upstream's exclusions), and `ringsAgree`.
  **Off the chunk grid a step is a busy one whatever `solo` says**: a chunk split into busy steps is finished in busy
  steps, so a whole chunk, a pair, a resume and a kept state all start on the 4,096 grid as before. `allowed(e, start,
  len)` is the same rule on every rank against its own state (the follower refuses a step the rule would not give).
- `ringsAgree(base, ring, c, rows, busy)`: `c + busy > base + ring or c + rows <= base + ring`. The DSA index rings
  (`cuda_state.State.ready` -> `rebase`) rebase when a pass would overflow them, to the pass's start; busy steps over a
  chunk may rebase only where the whole chunk would (at the chunk's start, by its first step) or not at all, else a
  later step would rebase part way in and the pools would be read from another base. With a 4,096 grid and an 8,192
  ring every aligned chunk passes (both sides of the inequality fall on the grid); the rule still guards a short ring
  (`ring_rows = min(capacity, 8192)`). Upstream's `ringsAgree` is the same statement for the doubling direction.
- `cuda_engine.zig`: `Engine.chunk_busy` (rank 0's choice for the next step, the chunk header's **flag 4**, consumed
  and cleared at the top of `prefillChunk` exactly as `index_split`/flag 2 is) and `busy_steps` (a counter for
  `glm-solo-check`). `prefillChunk`: a busy step relaxes `start % prefill_rows` to `allowed`'s `start % chunk`, takes
  `rows = min(len - start, settings.chunk)`, never pairs; a cancel at the gate part way into a chunk calls
  `snapshots.interrupted` with no marks (nothing kept off the grid; the gate's control exchange still runs on every
  rank). `prefillStart` (a resume) is unchanged: `r.at % 4096`.
- `cuda_follow.zig`: `Leader.chunk(ids, start, solo, ...)` (the old `pair: bool` renamed; `true` = no other stream
  decodes) picks `n = settings.chunk` when `busyAt`, else a pair when `pairedAt`, else a chunk, sends flag 4, and
  returns the rows taken (`Lanes.prefillStepFn` already advances `l.fill` by `step.rows`). The follower sets
  `e.chunk_busy = busyStep(h)` before `prefillChunk`.
- `cuda_ranks.zig`: `Settings.chunk` (0 off), the two readers with their refusals, `Fingerprint.chunk`, layout 15,
  `explain`. The `TF_GLM_CHUNK_SOLO=4096` spelling was kept so a configuration written for upstream does the same thing
  here (busy 2,048, alone 4,096), and so a value this grid cannot honour is refused instead of silently ignored.
- `cuda_lanes.zig`: comments only; `othersDecode` and the step loop are the pair patches' and were already generic.
- Untouched: `cuda_state.zig` (`prefill_rows`, `index_ring`, `snap_grid`), `dflash.zig`, the lane host
  (`TENSORFOLD_PREFILL_SHARE` spaces steps by time and `TENSORFOLD_PREFILL_BUDGET` counts rows a visit, so both see
  smaller steps as more frequent, shorter visits), `prompt_pair.zig`.

## Exactness

A chunk boundary does not change the bytes, and this patch adds boundaries only at multiples of 2,048 (or 1,024) inside
chunks that already exist:

1. The common EXL3 experts: "a pair's sums do not depend on which item holds it, so the bits are the same"
   (`cuda_exl3.zig`'s note on `chunk_rows`), which is the ground patch 0094 shipped on (2,048 -> 4,096 with identical
   greedy replies; `docs/speed/RESULTS.md`).
2. Attention and KDA: a prompt pass's rows attend over the committed cache and their own causal prefix in the same
   fixed tiles whatever the pass width (the pair's second chunk is any width from `min_rows` up); `glm-pair-check` shows
   8,192-row passes byte-equal to 4,096-row chunks on this tree (states, logits, the rings' bases), and upstream's
   `glm-solo-check` shows 4,096 byte-equal to 2,048 on `main`'s kernels, which this tree shares.
3. The one state-dependent reader is the DSA index ring base, which `ringsAgree` pins to the whole chunk's.
4. DFlash2 drafts only propose; the target verifies, so verified tokens cannot change. The drafter's own ingest sees the
   same rows in the same order (patch 0057's argument for the pair applies: `promptRun` keeps a step's rows from
   `len - WIN` on and `noteRun` continues a run across steps).

**The test:** `glm-solo-check` (`zig/tests/cuda/glm_solo_tests.zig`, registered in `tests/cuda/main.zig`) runs, on one
Engine and its ranks with `TF_GLM_CHUNK_SOLO=4096`, eight prompts twice: twin 0 a whole chunk a step (`solo` true, pair
off), twin 1 busy steps (`solo` false). It requires equal first token, equal next-window logits and tokens, equal rings'
bases and MTP base, equal KDA parity, every state byte equal, the kept-state marks called at the same positions (a case
with marks at chunks 1 and 3), and the busy-step count `prompt_solo`'s rule predicts (`Engine.busy_steps`). Run it at
`TF_GLM_CHUNK=2048` and `=1024`. Production hashes: `7a00f3a54e85` / `11e523fbec9a` (prose / code, `acc.py`, current
config) must repeat with `CHUNK_SOLO=1`, both with a concurrent decoder (the busy path) and without (the unchanged
path), and with `CHUNK_SOLO=0`.

## Expected effect, and the pre-registered gates

Pause of a decoder beside a cold prefill ~= busy rows / prefill rate. At ~2,450 tok/s a 4,096-row step is ~1.7 s of
GPU time (measured worst gap 1.96-2.02 s, p99 1.80-1.86 s, `benchmarks/2026-11-04-glm-1104-tp4-switch/README.md`);
2,048 ~= 0.85 s; 1,024 ~= 0.42 s. The prompt's own cost while sharing: the expert trellis decode amortises over fewer
rows (upstream measured a 2,048-row pass ~7% dearer than 4,096 at TP=2; at TP=4 with the pieces' exchanges per step the
delta may be larger, and the time-based share `TENSORFOLD_PREFILL_SHARE=0.68` then owes the decoders proportionally
less per step, so the cold-prefill-beside-decoders wall time is the number to read, not a per-pass figure). Alone,
nothing changes.

| Gate | Pass |
| --- | --- |
| `stall_during_200k_prefill` (glm_gates, 3 decoders beside a cold ~200k prompt) | max gap and p99 clearly under today's 1.96-2.02 / 1.80-1.86 s: expected ~1.0-1.1 s max at 2048, ~0.6 s at 1024; the prompt's admit time within ~10% of 95-101 s at 2048 |
| Cold prefill tok/s with no decoders (prompt ~6k / 25k / 51k / 100k, max of 2) | unchanged (same pairs and chunks; any difference is noise or a bug) |
| Greedy hashes `7a00f3a54e85` / `11e523fbec9a` | unchanged, with and without a concurrent decoder, at 2048 and 1024 |
| `glm-solo-check` at TP=4, `TF_GLM_CHUNK=2048` and `=1024` | PASS on every rank |
| Retention (`glm_retention.py`, 3 x ~275k, 768k window) | 3/3 as today (the kept-state grid is unchanged; the prompt cache holds the same states) |
| `concurrent == solo` 4/4, disconnect clean, 1-/4-/8-stream decode tok/s | unchanged |
| Rank 0's load line | `glm: prompt steps of 2048 rows while another stream decodes, 4096 alone (TF_GLM_CHUNK_SOLO)`; with `CHUNK_SOLO=0` no such line and today's `TF_GLM_PROF` profile byte for byte |

Also worth a look while measuring: a request that arrives during a busy step now waits ~0.85 s instead of ~1.7 s
(upstream's "first token of a short request sent into a 32K prompt" row), and the warm-turn resume TTFT (unchanged grid,
so unchanged).

## Not verified here

- Nothing was compiled or run: no Zig toolchain, no GPU, no Docker. Reviewed by reading against the pin's code paths
  (`prefillChunk`, `Leader.chunk`/`follow`, `Lanes.prefillStepFn`, `State.ready`/`rebase`, `snapshots.interrupted`,
  `dflash.promptRun`). Likely compile risks: the `if` expressions mixing `u32` and `comptime_int` in `Leader.chunk` and
  `prefillChunk` (given an explicit `usize` result type), and `std.ArrayList(u32) = .empty` + `append(gpa, v)` in the test
  (the 0.17 unmanaged API the engine itself uses).
- The exactness of 1,024-row boundaries is argued, not yet shown: `glm-solo-check` at `TF_GLM_CHUNK=1024` is the proof.
- The +7% per-pass cost is upstream's TP=2 number; TP=4 on the switch with 3 exchange pieces a step has not been measured.
- The engine branch `tp4-mesh-roce` lags the recipe's `0108` by one line (`glm_cuda_records.zig`'s `df_sumn` register
  figures, recipe commit `600cd93`); 0109 touches neither, and the full series 0001-0109 replays onto fresh `f350cd3e`
  with `git apply` (checked here).

## Build and test commands

```bash
# on the Spark (6f90), recipe checkout on branch tp4-mesh-roce-engine with 0109 in patches/engine/:
scripts/prepare.sh --image                     # builds f350cd3e + 0001-0109; the image tag hashes the patch set
# host tests in the image's build stage (no GPU): the Zig test blocks the patch touches
zig build test-glm-protocol-host               # cuda_follow's flag tests (flag 4 added)
zig build test-glm-startup-host test-glm-tp3-host test-glm-exl3-host   # the glm module's test block: cuda_ranks_test, cuda_prompt_solo, cuda_lanes_test
zig build test                                 # everything host-side, as the 2026-11-04 record did
# GPU, on all four ranks (as glm-pair-check / glm-split-check are run), with the production checkpoint:
TF_GLM_CHUNK_SOLO=4096 TF_GLM_CHUNK=2048 tf-cuda-test glm-solo-check <ckpt> <dc.json> <rank> 4 <idfile>
TF_GLM_CHUNK_SOLO=4096 TF_GLM_CHUNK=1024 tf-cuda-test glm-solo-check <ckpt> <dc.json> <rank> 4 <idfile>
# serve: CHUNK_SOLO=1 [CHUNK_BUSY=2048|1024] ./start-tp4.sh, then acc.py hashes, glm_gates, glm_bench cold prefill, glm_retention
```
