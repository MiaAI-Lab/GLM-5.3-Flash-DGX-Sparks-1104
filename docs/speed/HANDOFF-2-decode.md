# Handoff 2: the decode campaign's verdict, the shipped state, and the one lever left (written 2026-11-04, ~13:30Z; state lines and item 2 updated 2026-11-04 ~11:45 local after KDA C1 was measured)

Supersedes HANDOFF-decode-4stream.md (2026-11-03) - that campaign ran to its measured end. Everything below is
in docs/speed/RESULTS.md with full tables; this file is the operating summary. Standing orders unchanged:
measured numbers or blockers only; consult Codex on direction, bits, allocator/drafter, contradictions, and
"unreachable" declarations; never build on spark3/spark4; candidate patches stay out of patches/engine until
their image is on both Sparks; commit by path.

## The goal, and the answer

"spark3+4 must decode faster than spark1+2 at 4 streams, without losing the wins." Final measured shape:

| test | A: spark3+4 (102 patches) | B: spark1+2 | verdict |
| --- | --- | --- | --- |
| 1 stream prose / code | 38.4-39.2 / 55.6-56.8 | 40.5 / 50.6 | par; A wins code |
| 4 streams total | 78-83 (n=20+ reps, 3 images) | 93-95 | **B holds; not closable by policy** |
| 6 / 8 streams | 91-94 / 105-107 | (untested on B) | **A's new ground** (was capped at 4 streams) |
| cold prefill 16K / 64K | 1737-1780 | 1624-1632 | A |
| chat beside 50K prefill | 11.2-11.7 tok/s, ~2.7 s worst gap | frozen | A |

Every number greedy, same prompts, hashes `64c908210a9e` (prose) / `315341215112` (code) in every run.
**Sampled decoding is a different story (2026-11-04 12:05, RESULTS.md):** at temperature 0.7 A stays at 36.7-38.9 /
75-76 (1 / 4 streams) while B drops to 21.3 / 30-31 - its round doubles then triples. B's 4-stream lead is greedy-only;
for any sampled workload A is 1.8-2.4x faster.

**Why 4 streams stays at ~80 (all measured, RESULTS.md 2026-11-03 21:40Z through 2026-11-04 12:30Z):** a round
is ~104 ms and A lands 2.09-2.11 tok/round against B's 2.32; B's edge is per-offer draft quality at every depth
(0.665 vs 0.600 offered-weighted at 1 stream too - engine-intrinsic, not a 4-stream effect). Thirteen
configurations (cut 0.3/0.5/0.7, selector edge 0.6/1/4, tpm rule, mm-stages, drafter bf16, arena 4/8, forced
width 4, confidence-temperature mimic of B, 32-group weights) all land 76.9-83: the policy space is exhausted.
The calibration study bounds cut-tuning at ~84; Codex-checked arithmetic: 93 needs 2.42 tpr at 104 ms, and even
B's exact tpr on A's latency gives 89.2. KDA C1/C4 add +1-2 ms (82-84 ceiling). The campaign's stop rules fired
on their pre-registered gates; this is a measured verdict, not a guess.

## TP=4 (2026-11-04 evening): pair A is DOWN by the user's choice; spark1 serves all four Sparks

`./start-tp4.sh` on spark1 (rank 0, port 8888; spark2/spark3/spark4 ranks 1-3 round the switchless ring; weights from
spark1 over NFS) is LIVE on image `74e19a63a9a5` (106 patches: 0106 = the mode-aware peer warm-up; `daa6f2cde063` is the same code under the label before patch 0078's credit text changed). RESULTS.md
"TP=4 on the four-Spark ring" has the whole story: the four boots, Codex's read, the NCCL eager-connect and the
follower environ panic, the LAN bootstrap. Stop it with `./stop-tp4.sh` on spark1; pair A restarts with `./start.sh`
on spark3 (its local.sh untouched) once the four ranks are down. Check rank 0's startup line after every TP=4 start:
"16392 pages", "prompt cache: 3.6 GiB", no "cannot keep one state".

Since 22:30 local the four Sparks serve **unsloth/GLM-5.3-Flash-FP8** instead (`FP8=1 ./start-tp4.sh` on spark1, image
`cb33a3d5fc68` = 107 patches with 0107 block FP8, reserve 9.5): startup line "6147 pages", "prompt cache: 1.2 GiB".
`./start-tp4.sh restart` without FP8=1 goes back to the EXL3 pack on the same image. RESULTS.md "Block FP8 on the
ring" has the gates, the numbers, the decode profile, the DeepSeek-recipe levers and Codex's RoCE-ring design.

Since 00:20 local (2026-11-05) the image is `5d37e5287d7b` (108 patches: 0108, the four-rank RoCE ring all-gather)
on all four Sparks. **Live since 01:05 (2026-11-05): the EXL3 pack, the default (`./start-tp4.sh`), 16392 pages, 3.1 GiB
prompt cache, loaded in 135 s, 56.2 / 85.6 tok/s, the known hashes; `FP8=1` stays the opt-in and `ROCE_RING=on` the
opt-in gather path.** The FP8 serve with `ROCE_RING=on` (00:20-01:00) measured: same-image A/B +1-2% at one stream, +5-7% at four,
prefill, hashes and NLL identical, no ring warning in any rank log. The knob's default stays off (the pre-registered
one-stream line was 3%; RESULTS.md "The four-rank RoCE ring all-gather"). Boot time: every FP8 boot before 00:20 ran
with `TF_GLM_LOAD_PREFETCH=0` and took 686-742 s, set by spark3 (two NFS hops, page-fault reads); the EXL3 boots with
the default 8 readers loaded in ~2 min. With the knob left alone the FP8 boot loads in 245 s (RESULTS.md "The FP8
boot: 742 s to 245 s"); never set it to 0 again. Next engine levers, in order (RESULTS.md "Baseten's ... read against this engine"): the FP8
router and expert launches onto the PDL chain (1-2 ms a round), a fused router (up to 2.5 ms), then the 11 ms
outside the GPU sections (measure its composition first).

## What shipped (all exact, on image `cd1de29726c3`, both Sparks; **superseded 2026-11-04 13:20 by image `1fdae27149cc` (104 patches: 0103 tool-parse fix, 0104 warm turns, both ported from the rtx recipe, plus MEMORY_RESERVE_GIB 9.5 in scripts/local.sh) and at 15:10 by `8c4ca7bd8c39` (105 patches: 0105, untyped tool parameters decode as the JSON object the model wrote - the sibling recipe's issue #180) - RESULTS.md 2026-11-04 12:00, 13:20 and 15:30 sections; A serves 8c4ca7bd8c39 now (an upstream lane fix tried as 0106 at 15:30 was killed: rounds 50% longer, a ghost stream; RESULTS.md). Pair B (spark1+2) was STOPPED by the user at 13:35 after the Python-engine comparison and is down.**)

- **PARALLEL=8** in scripts/local.sh: the 8-stream arena. Free at every served shape; 6 streams 91-94, 8 streams
  105-107 tok/s. The one user-visible gain of the campaign.
- 0100 mm-stages (default 6 = 3 measured), 0098 selector-edge env, 0101 chance-temperature env, 0102 Q32
  32-group drafter linears - all env-gated, all defaults behavior-identical to the 98-patch image, hashes
  unchanged at defaults. Image lineage: b9d260ebc60b (98) -> 8156bfb7d08e (100) -> d9f47d5bc28f (101) ->
  cd1de29726c3 (102, current, on both Sparks).
- The **paired-metrics instrument** (no patch, both engines): snapshot `/metrics`
  (`tensorfold:mtp_drafted_total/mtp_accepted_total/decode_rounds_total`), run a fixed workload, snapshot again
  - exact drafts/round, accepted/offered, tokens/round per phase. Harness: docs/speed/harness/acc.py (tok/s,
  tok/round, hashes), acc4only.py / acc1only.py (phase-clean acceptance), acc8.py (6/8 streams), bench2.py (full
  suite). This killed the "counters may differ" caveat: A and B share the core's accounting; B's edge is real.

## The M1/M2 record (the drafter investigation, closed)

M1 (docs/speed/m1/m1diff.zig, f64 host harness on the real checkpoint): A quantizes the drafter's linears at
64-group affine, B at 32 - weight relrms 0.090 vs 0.078, cross-engine 0.103; the control isolates group size as
the whole gap. M2 shipped the 32-group path (0102) and the gate said stop: acceptance +0.4pt at 4 streams,
tpr flat - a 16-candidate argmax is robust to that noise; **weight precision is not the acceptance lever**.
S1 (conditioning) equivalent by source both trees; S2 (context restarts) measured zero in 2048 rounds. What
remains, untested and priced: the drafter's attention-kernel numerics - A's taps through the fused fp8 target
pipelines vs B's dedicated bf16 drafter pass (S3/M3: port B's cuda_dflash_forward, 2-3 days, materials staged
in docs/speed/s3-b-src/, design + gates in docs/speed/s3-port.md). The output head's 64 groups (welded to the
shared glue rmsnorm's sums) is the bounded cheap piece if anyone revives it.

## If work resumes, in order of expected value

1. **Nothing - it is a good stopping point.** The pair is faster where it was beatable, protected everywhere
   else, and the 4-stream question has a measured answer. The decay series (docs/speed/decay-*.log) was
   invalidated by this week's restarts if anyone cares about it again.
2. ~~KDA C1~~ **DONE, closed by measurement (RESULTS.md 2026-11-04, two sections):** the chain is ~1.1 ms of a
   1-stream round and ~3.2 ms of a 4-stream round, so "+1-2 ms" was its whole cost. Standalone on a GB10 (bench in
   docs/speed/kda-c1/): fusing prep into step LOSES, the single fused kernel is no better, and dependent launches
   (PDL) on the unchanged kernels save 5-7 us a layer (~0.2 ms/round), bit-identical. Built as 0199 (image
   45a1d17d74d7, loaded on both Sparks, unused) and A/B'd in the engine OFF/ON/OFF/ON: hashes stable, no resolvable
   change in the TF_GLM_PROF kda section (-0.13 then +0.06 ms at verify 2 rows; nothing at shared 8-13 rows; the
   section's own run-to-run spread is 0.19 ms). SHIP-OFF fired: 0199 stays in docs/speed/candidates/ as an exact,
   opt-in-by-name knob (TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe,kda); A went back to cd1de29726c3 (and later that day to
   1fdae27149cc, the rtx port). The KDA block's time is
   its input projections (~0.35 ms a layer at decode, ~half of bandwidth), not the chain. Codex consult archived
   (docs/speed/codex-kda-c1*.md). Do not reopen C1/C4 without a new mechanism.
3. S3/M3, the full-forward port, only as a deliberate project with its gate (paired tpr >= 2.25 at <= 104 ms
   twice, hashes unchanged; partial credit 2.18): the sole measured path to ~89+.
4. B-side instrumented image (add A's dft-trace to B's 0004 patch) for the matched-prefix A/B landing
   comparison - only if someone needs the last word on WHY B's picks land more often.

## Operating notes (unchanged in force, plus 2026-11-04 late additions)

**MEMORY_RESERVE_GIB is 9.5 in scripts/local.sh (untracked, site-local) since 2026-11-04 13:00** - at 8 the room had
grown past the page planner's third-window threshold (19.5 GiB) and the prompt cache was ZERO (every chat turn
re-prefilled its history). 9.5 keeps two 1M windows (4098 pages) and a 3.6 GiB cache. After EVERY start read rank 0's
startup line: "4098 pages", "prompt cache: N GiB", no "cannot keep one state" warning (RESULTS.md 2026-11-04 12:00).

**Since 2026-11-04 ~10:00 local the rtx box refuses ssh from spark3** (its ED25519 host key changed; spark1 and spark4
see the same new key SHA256:8boCMvwNt+Laujb3djxVmnsQTNdKucBbHUqx1S6IyqE and spark1 already trusts it - a stale entry in
spark3's ~/.ssh/known_hosts, lines for 192.168.1.122 and rtx, for the user to clear). Until then **spark5** (idle GB10,
not in either pair) carries the rtx roles: Codex (~/.local/bin/codex, logged in; not on the non-login PATH), nvcc 13.0
for sm_121 microbenchmarks, zig 0.17.0 at ~/c1bench/zig/zig (sha-checked download) for the cold host-test compile
(its baseline differs from the rtx box's: 16-29 failing binaries on the UNPATCHED tree too - compare against a base
checkout on the same host, docs/speed/kda-c1 shows how), and the harness client (~/c1bench/harness; acc.py from spark5
reproduces the standing band: 38.8/56.9 at 1 stream, 77.8-81.8 at 4). The image build worked from spark1 with
`~/tf-build/build-<label>.sh` / `ship-<label>.sh` (kept there as templates: stop B, build with the recipe's args and
the label, save|load to both Sparks, restart B). The label replica (sha256 over the TF_* pins + Dockerfile + sorted
patches) reproduces prepare.sh --label; promote a candidate into patches/engine on spark3 only after its image is on
both Sparks - start.sh then finds it by label and never builds.

Builds on spark1 only, pair stopped (`cd ~/newMod/GLM-5.3-Flash-EXL3-TensorFold && ./stop.sh`), tree at
`~/tf-build/recipe` (docker/ + patches/engine; label = `scripts/prepare.sh --label` computed on spark3 with the
intended set - replicate its sha256 over TF_* constants + Dockerfile + sorted patches); ship
`docker save | ssh sparkN docker load`; the serving host then finds it by label. Never let start.sh build on
spark3/spark4 (memguard stops the pair). Local compile check before any image:
`zig build test-glm-tp3-host test-glm-fp8-kv-host test-glm-exl3-host test-cuda-host` with the rtx toolchain
(/home/zurih/MiaAI_Lab/TensorFold/toolchain/zig-0.17.0/zig; the working clone is this session's scratchpad
.../e6decf6c.../scratchpad/tf, branch q32 @ 12e267d3 = shipped 0102) - but ONLY trust a cold compile (touch the
files; the cached 'success' hid a missing struct field once this week). Pre-existing test-cuda-host failures
(cuda_mm env-log noise, draft_sources check, glm_fp8_pages) fail identically without any diff - not ours.
spark3's repo has no remote; commit by path, never `git add -A`. The B pair restarts with
`PATH=$HOME/.local/bin:$PATH DENSE=fp8 ENGINE=zig ./start.sh` from ~newMod/... on spark1 (nohup + detach it, or
the ssh timeout kills it mid-start - happened once).

Session trail: RESULTS.md sections 2026-11-03 21:40Z onward; three Codex consults archived
(docs/speed/codex-{decode-op-20261103,endgame,final}.md); 0102/0101 in patches/engine and candidates/;
M1 harness docs/speed/m1/; B source refs docs/speed/s3-b-src/; scratch copies of everything in the rtx box's
/tmp/tfwork (regenerate the harness from docs/speed/harness/ - /tmp dies on reboot).

**2026-11-05 overnight (RESULTS "The overnight push"):** profiled TP=4 EXL3 (verify 28.0 ms of a 37.3 ms round;
prompt chunks wait 25% on gathers, HC glue 17%); ROCE_RING=on measured on EXL3 (+2% / +3% / -1.5% at 1 / 4 / 8
streams, exact: stays opt-in); candidate 0109 (per-cable link communicators for PREFILL_REDUCE=ring) built, made to
work on the wire, and KILLED by its gate (prefill -25 to -30%). The serve is back on 5d37e5287d7b, EXL3, gather.
Next prefill lever if anyone resumes: the exchange as an NCCL reduce-scatter on the world ring (pipelined like gather,
a quarter of the bytes) with the glue reading one summed partial; gate on NLL (the sum order changes).
