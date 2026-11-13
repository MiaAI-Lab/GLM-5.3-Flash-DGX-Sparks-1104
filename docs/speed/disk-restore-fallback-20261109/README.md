# One shorter disk state after a refused read

Patch 0122 (originally numbered 0114) backports TensorFold [#2442](https://github.com/ashhart/TensorFold/pull/2442),
commit [`12d61cb6`](https://github.com/ashhart/TensorFold/commit/12d61cb6b96c968aab76b8c53444380e66a76f53),
onto this recipe's `f350cd3e` pin and the 111 patches at recipe main `841e7497`.
The upstream author's attribution is retained in the patch.

The PR now merges main `80be4d6` (121 patches). Its patch is numbered **0122** because main now uses
0114 for KV sharding. The engine diff is unchanged; the numbered 0114 measurements below describe the
historical fallback candidate, not the current KV-sharding patch. No historical GPU measurement is relabeled
as a run of the 122-patch stack.

The previous landing base was `d4226a1` (112 patches): the maintainer added a server-only tool-envelope fix as 0112
while these measurements ran. Its six changed engine files are in `zig/src/server/`; the rows reader and GPU
kernels are unchanged. The tables below retain their measured `841e7497` baseline rather than relabeling it.

Previously, one unreadable longest disk state ended the disk lookup even when a shorter compatible state remained.
The backport makes one shorter attempt, preserves the existing two-tier lease budget checks, and leaves a valid
RAM fallback available if both disk reads fail. The last point is a pinned-stack adaptation: blindly keeping the
upstream terminal `catch null` would discard that RAM fallback. Its regression failed before the adaptation
(`expected 1000, found 0`) and passed afterward. Neither cache format nor model arithmetic changes.

## Fixed inputs

The hardware was two 128 GB NVIDIA DGX Sparks, one CUDA rank per GB10 (`sm_121`), over the existing dual-HCA RoCE link.
The runtime was Linux/aarch64, CUDA 13.3.1 and NCCL 2.30.7; driver versions were 580.159.03 and 580.173.02.
Builds used Zig 0.17.0 and xgrammar v0.2.8 (`97787376faee5ed8466cfad57c99855e4ce2f6aa`). All 103 base fatbins were rebuilt from the pinned source plus patches 0001–0111;
both arms embedded those same files. This is a complete native/asset rebuild, not a clean-image packaging receipt.

- Model: `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`, revision `76c0b5173166d2795dd48860f45d8224817f894c`.
- Drafter: `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`, at 4 bits.
- Serving: TP=2, parallel=8, FP8 KV pages, pre-existing q8 dense/head packing, `ABSORB_MMA=1`, `W8A8=0`.
- Disk cells: `CACHE_FIRST=1`, 8 GiB prompt cache per rank, 24 GiB learned-state cap, 15 GiB minimum free disk.
- Unsharded cells used a 32,768-token window; the sharded combination used 1,048,576. Comparisons within each cell
  used identical settings. The original recipe template's `EFFORT_LINE=thinking` rewrite was applied in both arms.
- Recovery requests: greedy, seed 101, thinking off, draft on, 128 output tokens; 19,294 rendered input tokens for
  the long case. The 300/600-note prefix-seeding requests were capped at 8 output tokens.

| Arm | Native binary SHA-256 |
| --- | --- |
| 111-patch control | `0de4883ad5d6bffd9c6a0eeca0b20f2a2f8490c7eb29fd5d0fc5f6b4dd50caea` |
| Control + 0114 | `fab34ef0692508091d419fa5049f38b428350efe07e0feecd7ce406bca9c4e08` |

## Device recovery result

The input generator is in [fixture.py](fixture.py), with selected machine-readable measurements in [results.json](results.json). Shorter compatible prefixes were requested separately, producing
4,096 / 8,192 / 12,288 / 16,384-token disk states. The stopped control's checkpoint directory was frozen and copied
into fresh task directories. Each corruption was injected only after the new process had opened its catalog, with
RAM cleared by restart. One byte in the selected `.st` payload was flipped; no trailer or metadata was rewritten.

| Cell | Control cached tokens / TTFT | 0114 cached tokens / TTFT |
| --- | --- | --- |
| Healthy restart | 16,384 / 2.219 s | 16,384 / 2.234 s |
| Longest `.st` damaged | 0 / 11.031 s | 12,288 / 4.734 s |
| Longest two damaged; a third remains healthy | 0 / 11.047 s | 0 / 11.234 s |

There was one paired run per fault cell. The one-damaged fixture reduced TTFT by 57.1%; this is a recovery-case
observation, not a claim of faster ordinary decode. Two failed attempts still go cold, so the extra bounded read
can cost time when no attempted disk state works.

All these responses, the original cold response, and the `LEARN + KV_SHARD=1` recovery cases had the same complete
output-token SHA-256: `81bf153f9dea1917b70bad88609ebccb41a5b64714e66d39525c1b38cbb56e29`.
The sharded run also passed 18 plain/drafted/resumed/concurrent/post-cancellation comparisons. All 18 matched a
separate genuine `--prompt-cache-gib 0` control, whose reported cached-token counts were all zero. The long
cache-disabled response also matched the cold/cache-on response. Client cancellation was observed server-side
as `reason=client-left` after 11 tokens; the next request still matched its solo reference.

The sharded files were checked as format 3 with two parts; the retained files in this fixture had home part 0.
The recipe's existing `scripts/cache-replay.py` tool-loop smoke passed: the first response made one tool call,
and the following user-message cases reported 8,192 cached tokens. This is a tool-loop smoke, not a function-calling
accuracy benchmark. Original containers were restored and their identities, configuration, binary hashes and
idle health checked after each device window.

## Host validation at the measured baseline

- `test-rows-host`: 116 passed. The four added tests cover one shorter read, the retry bound, compatible-prefix
  selection, allocation refusal, and retaining an existing RAM state after two failed disk reads.
- `test-lane-api-host`: 412 passed. `test-glm-scheduling-host`: 33 passed, 2 skipped.
- Removing the second attempt produced four expected assertion failures (112/116 passed), not a compile failure.
- Full `test-cuda-host`: control 4,086/4,100 passed, 10 skipped, 4 assertion failures; candidate 4,092/4,108 passed,
  10 skipped, 6 assertion failures. Both also reported the existing invalid-`TF_GLM_MM_STAGES` error-log fixture.
  Four assertion failures reproduced on the control: the GLM paged-store and draft-source fixtures, and two voice
  directory-permission fixtures. The candidate's two additional benchmark mock failures were not reproduced in
  isolated checks: both revisions actually ran all 17 selected tests five times and passed every run, with run
  caching disabled. Their cause is not established; the full suite is not reported as green.
- The same UTF-8 lean checker found 1,030 inherited problems on both control and candidate, zero new problems.
- The full root command `zig build test test-kernel-contract test-golden fuzz-cortheon -j1 --summary all
  -Dfatbins=FATBINS -Dsm=121` passed 5,946/5,978 tests, with 28 skipped and the same four named assertion failures
  already reproduced in the CUDA control. The existing invalid-`TF_GLM_MM_STAGES` error-log fixture also remained.
  The `test-kernel-contract`, `test-golden` and `fuzz-cortheon` targets succeeded; the root `test` target was not green.
- Plain `zig build -j1 --summary all -Doptimize=fast -Dcpu=native -Dstrip=false -Dfatbins=FATBINS -Dsm=121
  -Dxgrammar=XGRAMMAR` succeeded: 24/24 build steps, including the CUDA test and TP tools.
- Native builds and strict patch-application checks passed. No CPU-only check is presented as GPU qualification.
- The recipe's `scripts/tests/run.sh` passed locally under Git Bash: syntax, Python parsing, cache minimum,
  detached memory guard, NCCL interface handling and patch numbers. Shellcheck was unavailable and skipped.

## Historical integration on d4226a1

After rebasing onto `d4226a1`, all 113 published patches applied strictly to the full pin. The five rows files and
six updated server files matched the intended engine source by SHA-256. The current control passed 611 checks
with 2 skipped; the candidate passed 619 with 2 skipped: rows 116, lane API 412, GLM scheduling 33 plus 2 skipped,
and 58 tool-parser/guard integration checks. The same UTF-8 lean checker found 1,043 base findings and zero new.

The rebuilt candidate binary was `b083fb7bef995372fd6dcc21495c481fa405334bb798cbeea962bb2eb31f7849`.
On the pair, its healthy / one-damaged / two-damaged restarts again resumed 16,384 / 12,288 / 0 tokens, with the
same complete 128-token hash as the frozen cold reference. All 18 output checks matched the genuine cache-off
reference. The current recipe's tool-loop smoke returned one tool call and reused 8,192 tokens on later user turns.
Original services were restored and audited. This is a latest-stack integration rerun, not a new paired timing
baseline; the 57.1% figure above remains attached to the explicitly frozen 111-patch A/B.

Only the combinations above were exercised. This backport does not claim other backends, models, drafter types,
or all possible owner-placement and failure combinations.

## Conflict refresh on 80be4d6 (2026-11-13)

Main's TP4 integration renumbered the existing patches and assigned 0114 to KV sharding. This refresh merges
main, retains its README/settings/TP4 documentation, and moves the fallback patch to 0122. The fallback's
`diff --git` payload is byte-identical after line-ending normalization to the former 0114 patch; only its filename
and explanatory mail header change. The original source attribution and the measurements above are preserved.

All 122 engine patches passed `git apply --check` and applied in filename order on a fresh checkout of the full
`f350cd3ea253a042e205746928271f9d8f71a82d` pin. `bash scripts/tests/run.sh` passed under Ubuntu/WSL, including
the new main-branch graft and KV-shard configuration tests and unique patch-number checking. Shellcheck was not
installed and was skipped. This conflict refresh did not rebuild the image or repeat host-engine/GPU tests;
the earlier numerical and recovery results remain tied to their documented historical builds.
