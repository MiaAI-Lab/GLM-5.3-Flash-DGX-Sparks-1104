# Results: the GB10 decode/prefill campaign

Live record for the plan in `AGREEMENT.md`. Every number here is measured on spark3 + spark4, engine
`tensorfold-native 1.1.9`, image `tensorfold-glm53-spark:503f271d5532` (95 patches) unless a row says
otherwise. Appended as legs finish; nothing is summarised before it is measured.

## Harness added for this campaign

| File | What it does |
| --- | --- |
| `docs/dec.py` | decode at 1/2/4 streams on **fixed** prompts; reports tok/s, `round_ms`, `tokens_per_round`, acceptance, and `token_sha` per request. `--contract` runs the frozen 8-prompt contract corpus |
| `docs/trace.py` | fits GB10's verify-cost line `a + b*(rows-1)` from `TF_GLM_DFLASH2_TRACE` records, lone and shared separately, with residuals; derives per-position conditional landing rates |
| `docs/optwin.py` | replicates `draft_rule.zig`'s `offer()` so the predicted optimal window and the engine's choice can be compared, and tabulates implied throughput by width |
| `docs/leg.sh` | one measured leg: env -> `./start.sh restart` -> decode sweep -> contract corpus -> `token_sha` diff vs baseline |
| `docs/traceleg.sh` | a trace leg: restart with tracing, drive a workload, harvest `dft` records from the container's own output, fit |

## Method correction, Phase 0.1 (this changed the measurement design)

The first baseline run randomised prompt content per trial, as the prefill protocol requires. For
**decode** that was wrong, and it showed:

| 1 stream, random content | tok/s | tokens/round | acceptance | rounds/s |
| --- | ---: | ---: | ---: | ---: |
| trial 1 | 34.7 | 1.969 | 0.517 | 17.60 |
| trial 2 | 38.5 | 2.193 | 0.625 | 17.56 |
| trial 3 | 39.4 | 2.232 | 0.659 | 17.65 |

Throughput moved 12.2% while **rounds per second was stable to under 0.5%**. All of the variance is
`tokens_per_round`, which is content-dependent: how far the drafter happens to stay right on that
text. Greedy decode on a *fixed* prompt is deterministic, so fixed prompts remove the variance
entirely and leave only machine timing. `dec.py` was changed accordingly, and the decode gate is read
on identical content.

This matters beyond hygiene: **an exact draft-policy change cannot alter the reply, so on fixed
prompts it must show up as fewer rounds for the same tokens.** That is a far sharper signal than
tok/s on random text, and it is what the Phase 1 gate now reads.

## Phase 0.1 baseline (image `503f271d5532`, the shipped configuration)

Configuration captured in `baseline_env.txt`: `TF_GLM_DENSE_BITS=8`, `TF_GLM_HEAD_BITS=16`,
`TF_GLM_ABSORB_MMA=1`, `TF_GLM_W8A8=0`, `TENSORFOLD_MEMORY_RESERVE_GIB=8`, no `TF_GLM_DRAFT_*` set
(so: the `cut` rule at its default confidence 0.7). Startup line: `model 83.58 GiB`, 4,098 pages
(12.47 GiB), 4.4 GiB prompt cache, 18.1 GiB left after the 8.0 GiB reserve, prompts in passes of up
to 4,096 rows.

**Decode, random content (superseded by the fixed-prompt method, kept for the noise analysis):**
1 stream 38.5 tok/s median (spread 12.16%), 2 streams 51.7 (5.19%), 4 streams 68.1 (3.73%).

**Contract corpus, `token_sha` per prompt** (`contract-baseline.json`) - the output gate every later
leg is diffed against:

| prompt | prompt tok | out tok | token_sha |
| --- | ---: | ---: | --- |
| prose | 25 | 90 | `87fa94aff290` |
| code | 30 | 103 | `2e49486f633d` |
| spelling | 19 | 41 | `39fcf5dc3ba3` (answers `yrassece…`, the correct spelling) |
| math | 26 | 120 | `690b2a33a2d6` |
| zig | 31 | 48 | `60b55b74b419` |
| json | 32 | 21 | `29a950f3434c` |
| boundary | 10,674 | 2 | `fbe213fd977e` |
| longctx | 45,604 | 2 | `fbe213fd977e` |

**The drafting state the campaign is aiming at**, from the same baseline requests:
`drafted/round 1.74-1.89`, `tokens_per_round 1.97-2.23`, acceptance 0.52-0.66, against `max_held = 7`
drafts available. The `cut` rule at 0.7 stops a chain as soon as the product of its chances falls
under 0.7, which at per-draft chances near 0.8 is after one or two drafts.

## Phase 0.1 baseline, prefill and quality: reproduces the repo's record

Cold prefill, fresh random content, `cached_tokens = 0` on every trial:

| | trial 1 | trial 2 | trial 3 | median | spread | recorded in `docs/prefill/` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~23.5K | 1,733 | 1,735 | 1,739 | **1,735** | 0.35% | 1,735 |
| ~95K | 1,756 | 1,756 | 1,755 | **1,756** | 0.06% | 1,759 |

**Prefill is a very low-noise measurement here (0.06-0.35%)**, so a prefill gate can be read at the
1% level - unlike decode on random content. Quality baseline: NLL **0.7523** (per passage 0.5188 /
0.7838 / 0.2802 / 2.1669), `answers_sha` **`11ceddde19d4`**, and the spelling probe answers
`yrassecen` correctly with its own verification. All three match the recorded baseline exactly, so
the starting state is the one the repo documents and every later leg is comparable to it.

## Phase 0.1b: the fixed-prompt baseline, and a usable gate

| load | tok/s | round ms | tokens/round | drafted/round | acceptance | tok/s spread | tokens / rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 stream | **37.1** | 53.5 | 2.083 | 1.83 | 0.591 | **1.02%** | identical |
| 2 streams | **52.7** | 76.3 | 2.119 | 1.80 | 0.620 | **0.21%** | identical |
| 4 streams | **68.3** | 111.9 | 2.109 | 1.78 | 0.618 | **0.29%** | identical |

Tokens *and* rounds are bit-identical across trials at every load, so the G1 gate is a real
max(2%, 2x spread) = ~2% instead of the 24% the random-content noise would have forced.

## Phase 0.3: the verify-cost line, measured. **This refutes the campaign's central thesis.**

`TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` cycled (one leg samples every width, drift-free), tracing on,
**587 records**. Before fitting, two source facts were checked rather than assumed:

- `v=` is the wall ms of exactly one verify forward (`cuda_lanes.zig:452-454` for lone,
  `:489-493` for shared, where the same ms is recorded against every lane in the forward), and
  `lanes/config.zig:30,37` define the tables as "forward ms by exact window width" and "shared
  forwards' ms by total rows". **Same quantity: the comparison is like-for-like.**
- `costTable`'s `one` is the ms at width 1 (`i = 0` gives `width = 1`), the same convention as the
  fit, so there is **no one-row offset** between table and measurement.
- Shared records were regrouped into forwards and fitted against the forward's **total** rows, which
  is what `shared_costs.get(rows)` is indexed by (`lanes/depth.zig:305`). Fitting shared `v` against
  one lane's rows would have been wrong.

Lone windows, 309 observations, medians at 1..8 rows: 37.97, 44.77, 50.89, 56.03, 61.27, 66.23,
73.05, 78.55 ms.

| | fit | b/a | r2 |
| --- | --- | ---: | ---: |
| lone windows | **38.73 + 5.68 (rows-1) ms** | **0.147** | 0.9983 (residuals within +-0.9 ms) |
| shared forwards (2 lanes) | **47.38 + 5.05 (rows-1) ms** | 0.107 | 0.9934 |
| drafter pass | **5.92 ms** lone, 6.83 shared | | (sm_120 seed says 1.37) |
| sm_120 seed, for comparison | 9.40 + 1.31 | 0.139 | |
| the GB10 table being used | 52.0 + 6.8 | 0.131 | |

**A verify row is not cheap on GB10.** `b/a = 0.147` against sm_120's 0.139 - 6% apart, and slightly
*worse* than the table's 0.131. The mechanism all three experts reasoned from was wrong: a draft row
re-reads the shared dense and attention weights for free, but **each row selects its own top-8 of 288
routed experts**, so expert traffic scales nearly per row (~2.1 GB/rank/row, ~7.8 ms at 273 GB/s;
measured 5.68 ms, so the rows do overlap partially). This is the same expert-divergence mechanism
that caps 1->4 stream scaling at 2.05x. The brief contained that arithmetic and none of us applied it
to draft rows.

Per-position conditional landing (forced regime): 0.701, 0.524, 0.537, 0.434, 0.278. The sixth point
(0.750) rests on n=4 and is **discarded** from any fit at GLM's insistence - too small to anchor a
decay curve.

**Throughput confirms it directly.** The forced-width leg itself ran at `drafted/round 3.43`:

| | 1 stream | 2 streams |
| --- | ---: | ---: |
| shipped `cut` 0.7 (`drafted/round` 1.83) | **37.1** | **52.7** |
| forced deep windows (`drafted/round` 3.43) | 30.7 (**-17%**) | 42.7 (**-19%**) |

with acceptance collapsing 0.591 -> 0.294. And feeding the measured line through a replica of
`draft_rule.zig`'s `offer()` (`docs/optwin.py`) puts the optimum at **k=2 drafts, 36.93 tok/s
predicted**, against the shipped rule's k=1.83 measuring **37.1**. The rule is already within noise
of the best width.

**G0 fires: the constant-mispricing thesis is dead.** The `tpm` leg and the rebuild it was meant to
justify are dropped, as both reviewers pre-agreed.

## Phase 1: the `cut` bracket (the insurance both reviewers demanded on G0-fail)

| `TF_GLM_DRAFT_CONFIDENCE` | 1 stream | 2 streams | 4 streams | drafted/round | acceptance | contract |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.70 (shipped) | 37.1 | 52.7 | 68.3 | 1.83 | 0.591 | reference |
| 0.85 | 37.2 (+0.3%) | 52.4 (-0.6%) | 69.1 (+1.2%) | 1.52 | 0.667 | **identical on all 8** |

Nothing clears the ~2% gate at any load. A shallower cut trades fewer drafts (1.83 -> 1.52) for
higher acceptance (0.591 -> 0.667) and nets zero: **the policy sits on a broad plateau**, and the
shipped 0.7 is on it. The identical `token_sha` across all 8 corpus prompts also confirms
empirically that a draft-policy change is exact, which the source claims but had never been checked
here.

## The one lever that survived, and the mechanism measured before building

The *ratio* is right but the *absolute* prices are badly wrong, which is a different defect from the
one G0 killed, and it meets D4's independently pre-set >20%-per-coefficient test:

| | table | measured | error |
| --- | ---: | ---: | ---: |
| lone intercept | 52.0 ms | 38.73 ms | +34% |
| shared intercept | 76.0 ms | 47.38 ms | +60% |
| shared slope | 4.1 ms/row | 5.05 ms/row | -19% |

`lanes/depth.zig:304-306` learns the scheduler's overhead as `ms - shared_costs.get(rows)`, clamped
at zero, and feeds it to `allocateDrafts` - the adaptive input to cross-stream draft allocation. With
the forward over-priced, that subtraction goes negative. Checked against the 139 measured two-lane
forwards, taking `v + drafter` as a lower bound on the round ms:

| | forwards that could fire the learner |
| --- | ---: |
| old table | **4 of 139** |
| measured table | **133 of 139**, learning 3-8 ms |

So the learner is **starved, not dead** (the round ms it is actually given exceeds that lower bound,
so the true old rate may be above 4/139). Correcting the tables lets it learn for the first time.
Patches `0096` (the tables; behaviour) and `0097` (the `draft_rule.zig` seed and its stale `sm_120`
comment; inert under the default `cut` rule) are written and both pre-verified with
`git apply --check` against the 95-patch tree, kept as separate files so `0096` stays independently
revertible. **No throughput claim is attached: the direction is unknown and the A/B decides.**

## Patch 0096 + 0097: built, A/B'd, mechanism confirmed, throughput inside noise

New image `42298ca48d80` (the rollback image `503f271d5532` stays on disk). Fixed prompts, tracing
off, 3 trials, against the same `base2` capture:

| | 1 stream | 2 streams | 4 streams |
| --- | ---: | ---: | ---: |
| tok/s | 37.16 (+0.20%) | 52.85 (+0.26%) | 68.77 (+0.65%) |
| gate needed | +2.05% | +2.00% | +2.00% |
| round ms | 53.28 (-0.45%) | 75.95 (-0.47%) | 110.50 (-1.22%) |
| **`drafted/round`** | 1.833 -> **1.833** | 1.797 -> **1.784** | 1.781 -> **1.726** |
| `tokens/round` | 2.083 -> 2.083 | 2.119 -> 2.119 | 2.109 -> 2.097 |

`token_sha` identical on all 8 contract prompts; prefill 1,730/1,753 against 1,735/1,756 (inside its
0.06-0.35% noise); NLL 0.7523 and `answers_sha 11ceddde19d4` unchanged.

**The mechanism is confirmed and the signature is specific**: `drafted/round` moved at 2 and 4
streams and was bit-identical at 1 stream - which is exactly right, because a lone window never goes
through the shared forward whose table was wrong. The learner now fires and changes allocation. It
trades draft attempts for round time (drafts -3.1%, tokens/round -0.6%, round -1.22%, net +0.65%),
which is what a cost-aware scheduler should do once a draft attempt is priced at its true cost.

**It does not clear the +2% gate.** The panel split: GLM KEEP (a correctness fix to a cost model is
not a performance claim and should not need a throughput gate, only an absence of regression), Codex
REVERT (do not relax a gate after seeing the result). Resolved as **keep provisionally under a
pre-committed rule** (`PRECOMMIT-0096.md`, written before any coverage leg reported), on the ground
that the correctness framing is in 0096's own header, written before the build - *"Whether that
improves throughput is a question for the A/B, not a claim of this patch"* - so it is pre-registered
rather than post-hoc. Codex's risk-coverage objection is answered by the paired ABBA coverage suite
(10 trials at 4 streams, long-context decode, mixed prefill+decode, `PARALLEL=8` stress, tails).
**Both reviewers were explicit that "marginal win" overstates this and that no +2% language may be
used.**

## The forward's byte budget, from this checkpoint's own shape

Needed because, as Codex put it, the 38.73 ms verify forward and a round-level byte estimate "need
matching boundaries before they can be compared". Derived from `config.json` (45 layers, hidden 4096,
34 KDA layers at 64 heads x 128, 11 full-attn MLA layers, `first_k_dense_replace` 3, 288 routed
experts at top-8, `moe_intermediate` 2048, vocab 154,880) and the engine's own tensor list
(`weights.zig:18-20`: `kda_in` is `q_proj k_proj v_proj f_a_proj g_a_proj b_proj`, so KDA's gates are
**rank-128, not full width**; and `indexer.wk` / `indexer.wq_b` sit in `mla_x` / `mla_qr`, so the DSA
indexer is on the **11 full-attn layers only, not all 45**). At TP=2 each rank holds half the heads,
half the MoE intermediate and half the vocabulary.

| term | GB a rank | share |
| --- | ---: | ---: |
| KDA projections x34 | 2.357 | 35.8% |
| routed experts, top-8 x42 | 2.114 | 32.1% |
| MLA + indexer x11 | 0.720 | 10.9% |
| LM head (bf16) | 0.634 | 9.6% |
| shared expert x42 | 0.528 | 8.0% |
| dense MLP x3 | 0.226 | 3.4% |
| **one forward, one row** | **6.581** | |

| | floor | measured 38.73 ms is |
| --- | ---: | ---: |
| at peak 273 GB/s | 24.10 ms | 62% of the forward |
| at 85% (232 GB/s) | 28.36 ms | 73% |
| at 80% (218 GB/s) | 30.19 ms | 78% |

### This corrects my earlier "20-30 ms of every round is not weight streaming"

It closes the accounting instead. The shipped round runs at ~2.83 rows, so the forward is
`38.73 + 5.68 x 1.83 = 49.1 ms`, and the drafter adds 5.92 ms, giving **55.0 ms against a measured
53.5 ms round - within 3%**. So a round *is* its verify forward plus (most of) its drafter pass:
**there is no large hidden host-time term.** The identifiable slack is ~8.6-14.6 ms inside the
forward (38.73 against a 24.1-30.2 ms byte floor) plus ~3 ms in the drafter, i.e. roughly **22-33% of
a round**, not the ~50% I claimed from the wrong arithmetic. `TF_GLM_PROF` will attribute the
in-forward part.

### What the byte budget says about the remaining levers

- **`DENSE=q6` is the largest byte lever available.** The q8 terms (KDA + MLA + shared + dense MLP)
  are 3.83 GB, **58% of the forward**, so q8 -> q6 removes ~0.96 GB = **14.6% of forward bytes**.
- **`HEAD=q8`** removes 0.317 GB = **4.8% of forward bytes**.
- **The marginal row is cheaper than its own expert bytes**, and that is the interesting number:

| | |
| --- | ---: |
| a row's 8 experts, if disjoint | 2.114 GB = 7.74 ms at peak, 9.11 ms at 85% |
| measured marginal row | **5.68 ms** = 73% of the peak estimate, 62% of the 85% estimate |
| random overlap between two rows' top-8 of 288 | 0.22 experts = **2.8%** |

Random overlap explains under 3 points of a 27-38 point gap. So either **routing correlates strongly
across adjacent positions**, or the expert gather runs nearer peak than the rest of the forward, or
expert streaming overlaps other work. Those are separable by logging the actual top-8 per position
and measuring a window's union size - which is exactly milestone 1 of the expert-traffic project, and
it is also the measurement that decides whether GLM's lag-1 routing predictor has anything to
predict. **Not claimed as overlap:** GLM's caution stands that L2 residency and shared-expert caching
produce the same signature.

<!-- APPEND BELOW: ABBA coverage, profile, drafter probe, numerics -->

## Host-state drift is larger than the effects being measured (2026-11-03, 06:20Z)

spark4 wedged and was power-cycled (see the restart bug below). The pair came back **bit-identical**: all eight
contract `token_sha` match the pre-stall baseline, including the 10,674 and 45,604-token prompts, and decode's
`sha@1` is `5a0be163e6f3` in every run before and after. Nothing needs re-qualifying on quality.

But the same config measured **+1.5% faster after the reboot**, at every load:

| config | 1 str | 2 str | 4 str | round ms @1 |
| --- | ---: | ---: | ---: | ---: |
| base2 (pre-stall, no 0096) | 37.09 | 52.71 | 68.32 | 53.52 |
| p0096 (pre-stall, 0096 in) | 37.16 | 52.85 | 68.77 | 53.28 |
| postreboot (0096 in) | **37.73** | **53.79** | **69.79** | **52.64** |

postreboot vs p0096, same image and patches, across the reboot only: **+1.52 / +1.78 / +1.49%**.

Only spark4 was cycled (it had been up since 2026-10-30; spark3 was not restarted). Trial spreads within each run are
0.14-0.69% and tokens are bit-identical, so this is not trial noise -- it is **host state**.

### What this invalidates

0096's A/B read +0.20 / +0.26 / +0.65%. A reboot moves the number ~3x further. **The drift band is wider than the
effect**, so that A/B cannot support a verdict in either direction, and the "postreboot vs base2" column (+1.7 / +2.0 /
+2.1%) is confounded by the reboot and is not 0096's effect either. 0096 stays unresolved, not kept and not reverted.

Consequence: **only interleaved A->B->A measurement is valid here** at the 1-2% level, with a restart between passes.
That is what `abba.sh` does, and what both advisors asked for; it turns out to be the measurement floor, not diligence.

### A free lever this exposes

If long uptime costs ~1.5% of decode -- plausible, since weights, KV pages and the prompt cache all draw on one
128 GB pool and fragmentation accumulates -- then a scheduled restart is worth more than 0096 was. **Untested**: the
clean experiment is to cycle spark3 (never restarted here) and re-measure, and to re-measure spark4 again after days
of uptime. Confounders not yet excluded: page-cache state, the 5-hour-old container, and driver-side allocator state.

## The restart bug that took spark4 down (2026-11-03, 05:07Z)

`start.sh restart` issues `docker run` on the worker without confirming the previous container died. The worker's
journal for that boot:

```
07:07:07 dockerd: cafdbe5c... cleanup: failed to delete container from containerd:
                  cannot delete running task cafdbe5c...: failed precondition
07:07:07 dockerd: failed to delete container from containerd ... DeadlineExceeded
07:07:10 containerd: connecting to shim d67d3658...        <- the NEW container, old task still live
```

Two ~100 GiB containers on a 128 GB machine. No OOM kill and no panic: root cron kept running to 07:30 while `zurih`
ssh logins stalled `[preauth]` from 07:28 (fork no longer completing), the journal stops at 07:30:24, and the host
never recovered -- it needed a power cycle. Ping answered and TCP/22 accepted throughout, which is why it looked alive.

**Fix before any further multi-restart programme**: poll `docker inspect` on the worker until the container is
genuinely absent, and abort the leg if it is not, rather than firing `docker run` and hoping. The overnight programme
does seven restarts, so the race is near-certain to recur. Open question for the advisors: hard-`kill` after a stop
timeout, or abort and leave it for a human.

## [RETRACTED -- see the retraction below] The profile (2026-11-03, 06:35Z): prefill is router-bound

`TF_GLM_PROF=1`, 25 reports. Per 4,096-row prompt chunk (21 reports, 84 calls), wall 1,981.78 ms a call:

| section | ms/call | % of wall |
| --- | ---: | ---: |
| **router** | **1,720.38** | **86.8%** |
| routed | 641.07 | 32.3% |
| head | 12.13 | 0.6% |
| embed / gate / draw / mtp / commit | < 1.3 | ~0% |

(gpu totals 120.5% of wall and "NOT GPU" is -20.5%, i.e. sections overlap on concurrent streams, so the column
sums past 100%. The absolute attribution therefore needs one independent confirmation before anyone acts on it.)

**The router needs ~4 ms of arithmetic and is taking 1,720 ms.** Gating 288 experts is a 4096x288 GEMM a layer,
~217 GFLOP over 45 layers, a few ms at any plausible rate; top-8 of 288 over 184K rows is trivial. So the prefill
router is ~400x off its own roofline and is the largest single number in this engine.

### Why it stayed hidden, and why it is not just a wait on another stream

Per row, between the 64-row and 4,096-row buckets:

| section | 64 rows | 4,096 rows | amortisation |
| --- | ---: | ---: | ---: |
| routed | 1.511 ms/row | 0.157 ms/row | **9.6x** |
| router | 0.727 ms/row | 0.420 ms/row | **1.7x** |

Widening the chunk (patch 0094) amortised the expert trellis decode 9.6x and left the router as the wall -- which is
why +7% was the ceiling there. And if `router` were merely blocking on the expert stream its curve would track
`routed`'s; it does not, so it is doing real row-proportional work that does not amortise. 0.420 ms/row over 45 layers
is 9.3 us a row-layer.

**Next step, before any claim**: read the prefill router path, try the existing off switches (`TF_GLM_TOPK_SUM=0`,
`TF_GLM_DSEL=0`) to see what the section contains, and time the router alone. If it can be brought even to 10x off
roof, prefill is a multiple, not a percentage -- far beyond anything else measured in this exercise.

## Decode attribution, and the byte model confirmed

Verify forwards, same profile. GPU is 99.9% of wall at both widths: **decode has no host-time component at all.**

| section | 2 rows (ms) | 3 rows (ms) | marginal |
| --- | ---: | ---: | ---: |
| routed | 17.09 | 22.66 | **+5.57** |
| kda | 12.99 | 13.17 | +0.18 |
| mla | 4.95 | 5.00 | +0.05 |
| router | 3.32 | 3.37 | +0.05 |
| head | 2.74 | 2.74 | 0.00 |
| gather_a + gather_b | 1.85 | 2.36 | +0.51 |
| dense | 1.17 | 1.18 | +0.01 |
| shared | 0.37 | 0.39 | +0.02 |
| total | 44.50 | 50.89 | **+6.39** |

**87% of a marginal verify row is routed-expert bytes** -- the expert-divergence mechanism measured directly, which
closes the open question from the trace fit (the 5.68 ms marginal row *is* the routed experts). **KDA is flat in rows**,
so it is a fixed per-forward cost, not a per-row one.

Extrapolated to one row (~38.09 ms): kda 33.6%, routed 30.2%, mla 12.9%, router 8.6%, head 7.2%, gathers 3.5%,
dense 3.0%, shared 0.9%. Against the byte budget's predictions of 35.8 / 32.1 / 10.9 / 9.6 / 3.4% -- **the byte model
holds** for everything except the shared expert, predicted at 8.0% of bytes and measuring 0.9%, i.e. successfully
hidden on its side stream. That predicts `TF_GLM_SHARED_SIDE=0` will be *worse*, which is a falsifiable test of the
whole accounting.

## Drafter probes (2026-11-03, 06:50Z)

| leg | 1 str | 2 str | 4 str | round ms @1 | tok/round | accept | drafted/round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped (DFlash2, 4-bit) | 37.73 | 53.79 | 69.79 | 52.64 | 2.083 | 0.591 | 1.83 |
| `DRAFTER=none` | 26.57 | 26.45 | 26.40 | 36.46 | 1.004 | 0 | 0 |
| `DRAFTER_BITS=0` (bf16) | 36.18 | 50.60 | 67.60 | 56.71 | 2.155 | 0.610 | 1.88 |
| `DRAFTER=mtp` | 37.47 | 45.62 | 57.60 | 47.37 | 1.866 | 0.661 | 1.30 |

- **Speculation is worth +41%** (37.7 vs 26.7 at 1 stream), and a plain 1-row forward is 36.46 ms, consistent with the
  38.1 ms the verify line extrapolates to.
- **Without a drafter, 1->4 streams scales 1.00x** (rounds 36.5 / 74.3 / 149.2 ms, exactly linear). So *all*
  multi-stream throughput here comes from the lane scheduler sharing verify forwards, not from batching as such. A
  4-row batched forward should cost ~55 ms and serve 4 tokens; the no-drafter path takes 149 ms, so that path does not
  batch. Irrelevant to shipped behaviour, but it suggests checking whether draft-less lanes at load also fall back to
  unbatched forwards.
- **bf16 drafter is -3.7%**: acceptance does improve (0.591 -> 0.610) but the round grows 4.07 ms and eats it. 4-bit
  confirmed. The drafter pass is substantially bandwidth-bound, which was the question this leg was built to answer.
- **MTP is a wash at 1 stream and -15 / -17.5% at load.** Notably it runs 10% shorter rounds at higher acceptance with
  29% fewer drafts, then its calibration collapses at depth (2.96 drafts at 0.429 acceptance). A cut threshold
  re-tuned for MTP at low load is a plausible follow-up; it is not a conclusion.

## RETRACTION: the prefill "router" finding was an instrument artifact (2026-11-03, 07:10Z)

**Withdrawn.** `cuda_prof.zig:71` says it in one line:

```zig
/// The work queued on `s` since the last mark belongs to `section`.
pub fn mark(p: *Prof, s: cuda.Stream, section: Section) void {
```

`mark()` is **cumulative since the previous mark**, not a bracketed span. The pieced prompt path never marks the
mixer, the attention, the output projections or the cross-rank exchanges, so the next mark -- `.router` -- absorbs all
of it. The 1,720 ms is "most of the prompt layer loop", not gating. Codex reached this independently and pointed at
the line. The ~400x-off-roofline claim is void: it compared an arithmetic bound against the wrong interval.

Also invalid, from the same review: my 64-row vs 4,096-row per-row amortisation argument. The last bucket takes
**anything above 1,024 rows** (patch 0094 documents the stale "to 2048" label), and the raw log shows that bucket
mixing 3064, 3073, 3100 and 4096-row chunks, while the "to 64" bucket held a 24-row chunk. So the per-row figures
compared two mixed denominators, and the 1.7x-vs-9.6x asymmetry I used to argue "not a wait" is not established.

### What the fine instrument says instead

There is a separate, properly bracketed per-stage trace in the same log (`glm prompt stage ... site= stage= stream=
ms=`). Filtered to **exactly** 4,096-row chunks on rank 0, 23 passes, ms a pass:

| site | stage | stream | ms/pass |
| --- | --- | --- | ---: |
| mixer | wait_filled | side | 3755.91 |
| ffn | wait_filled | side | 2963.37 |
| ffn | output_projection | main | 933.00 |
| ffn | shared_expert | main | 882.13 |
| ffn | partials | side | 875.39 |
| mixer | partials | side | 689.93 |
| mixer | output_projection | main | 576.89 |
| mixer | glue | main | 294.47 |
| ffn | glue | main | 292.69 |
| ffn | glued | side | 213.00 |
| mixer | glued | side | 208.07 |
| ffn | wait_shared | main | 127.96 |
| mixer | wait_shared | main | 122.17 |
| mixer | wait_gathered | main | 121.64 |
| ffn | wait_gathered | main | 101.20 |

main-stream total 3,452 ms a pass against 1,982 ms of wall, so these **overlap and are not additive** -- relative
weights only. Gating (`glue`) is 587 ms, ~17% of main-stream time, not 86.8%. The large real terms are the **output
projections (1,510 ms)** and the **shared expert (882 ms)**, with **473 ms (13.7%) of main-stream waits** while the
side stream sits in `wait_filled` for 6.7 s, i.e. largely idle.

### The blocking methodological problem

`TF_GLM_PROF` **disables prompt pairing** (`cuda_prompt_pair.zig:83`: `or e.prompt_profs[3] != null) return false`).
So every prefill number in this profile is from a path production does not run, which is also why it read 1,700 tok/s
against the 1,735 baseline. **No prefill attribution from a `TF_GLM_PROF` run can be trusted as production.** The
agreed next step is an Nsight Systems timeline of the unprofiled prefill on both ranks, using the stage spans only to
pick candidate kernels, and validating any change by unprofiled interleaved measurement.

### Other claims this review weakened (all accepted)

- **"Decode has no host-time term"** is not established by GPU = 99.9% of wall: event intervals can include GPU idle
  waiting on host submission and stream dependencies. Needs a timeline.
- **"87% of a marginal verify row is routed-expert bytes"** overstates it. The attribution is to the *routed interval*;
  it is consistent with expert traffic but also contains kernel and scheduling effects. Bytes need hardware counters.
- **"KDA is flat in rows"** holds between 2 and 3 rows in this one measurement, not across widths or concurrency. The
  one-row column is an extrapolation, not a profile.
- **"All multi-stream scaling comes from shared verify forwards"** -- the no-drafter leg shows no aggregate scaling in
  that experiment; it does not prove the engine never batches. Needs forward-width and lane traces.

### What survives unharmed

- The **shared-side falsification**: predicted worse, measured worse (-1.4 / -1.0 / -0.7%). The side-stream overlap is
  real and working.
- The **drafter probes**, which are end-to-end throughput and need no attribution: speculation +41%, bf16 drafter
  -3.7%, MTP a wash at 1 stream and -15/-17.5% at load.
- The **bit-identical recovery** through the power cycle, and the **~1.5% host-state drift** finding.

## Numerics, measure-only (2026-11-03, 06:54-07:05Z): HEAD=q8 is the one real candidate

### My own contamination, first

`codex exec` was run at 07:03:00Z to consult an advisor. **`~/.codex-shim/models.json` points at
`http://127.0.0.1:8888/v1` -- the benchmark server itself.** Codex's model calls therefore took decode streams from
the pair while it was being measured. That window (07:03-07:06) covers the tail of the `DENSE=q6` sweep and the start
of the `restored` leg, which is why those two show 25-57% trial spreads and the restore leg read 31.6 tok/s at 2
streams against 53.8 at the identical config. Not contention from the host, not a regression: self-inflicted load.

**Rule for the rest of this campaign: never run an advisor consultation against the measured endpoint.** Either
sequence it outside a leg, or point the shim at a different model host.

`HEAD=q8` (06:54:50-06:59:18Z) and everything before it are **clean** -- they finished before 07:03.

### HEAD=q8: +3.1% at 1 stream, and exact on every check available

| evidence | result |
| --- | --- |
| decode 1 / 2 / 4 streams | 38.9 / 54.3 / 70.6 tok/s = **+3.1% / +0.9% / +1.2%** |
| trial spreads | 1.40% / 0.19% / 0.01% |
| round ms @1 | 52.64 -> 51.50 (**-2.2%**) |
| 8 contract `token_sha` | **all identical to baseline** |
| 10 greedy answers | `11ceddde19d4` -- **identical to the reference** |
| paired per-token NLL, 1,051 tokens | delta **-0.00050**, se 0.00054, 95% CI **-0.00156 .. +0.00056** |
| `quality.py` 4-passage NLL | 0.752643 vs 0.752338 baseline (+0.0003) |

The head is 0.634 GB of the 6.58 GB forward (4.8% of bytes); q8 halves it, predicting ~2.4% off the forward, and
-2.2% of round time is measured. **The prediction and the measurement agree**, which is the first time a byte-budget
prediction has been confirmed end-to-end here.

The 1-stream vs 4-stream asymmetry (+3.1% vs +1.2%) is a consistency check that passes: the head is a **fixed
per-forward cost**, like KDA, so its share falls as rows grow -- matching its measured 7.2% of a 1-row forward against
6.2% at 2 rows. A per-row cost would have behaved the opposite way.

At +3.1% this is also **the first effect measured in this campaign that clears the ~1.5% host-drift band**, i.e. the
first one this rig can actually resolve. The paired CI (+-0.0016/token) is ~10x tighter than the bf16-vs-q8 interval in
`docs/prefill/README.md` (+-0.015), so the quality evidence is stronger than anything used to clear `ABSORB_MMA`.

**Status: NOT shipped.** It is a numerics change and needs the user's explicit approval. It also wants A->B->A
confirmation of the +3.1%, since a single non-interleaved pass is exactly what the drift finding invalidated.

### DENSE=q6: rejected on quality, by the repo's own rule

| evidence | result |
| --- | --- |
| 10 greedy answers | `81de7e0a307d` -- **changed** from `11ceddde19d4` |
| 8 contract `token_sha` | **3 of 8 changed**: prose, math, zig |
| `zig` prompt | 48 -> 120 output tokens, different function name (`sum` -> `sumSlice`) |
| `quality.py` 4-passage NLL | 0.771079 vs 0.752338 -- **+0.0187 worse** |
| paired aggregate NLL | 1.50565 vs 1.50609 -- marginally *better* |
| paired per-token delta | +0.00043, se 0.00550, 95% CI -0.0103 .. +0.0112 (uninformative) |

**This is the W8A8 trap exactly as the record describes it**: the aggregate paired NLL says q6 is slightly better while
the 4-passage NLL is 0.019 worse and the greedy answers moved. Judge per passage, never on the mean. For scale, q4 was
+0.053, so q6 does ~35% of q4's damage -- the same failure mode, smaller.

Its throughput is unusable (contaminated, 57% spread at 1 stream) and there is no reason to re-measure speed for a
config that fails the quality gate. **The largest byte lever on the table is closed on quality, and the small one is
the winner** -- the opposite of what the byte budget alone predicted, because bytes do not decide whether output moves.

## The 0096 ABA (2026-11-03, 07:13-07:52Z): contract clean, throughput inconclusive, and a second noise regime

`chain2.sh` -> `abba.sh` ran to completion on its own: a1 07:13:32-07:25:49Z (0096 image `eeb98fe202ff`),
b1 07:25:49-07:40:40Z (baseline image `503f271d5532`), a2 07:40:40-07:51:46Z (`eeb98fe202ff` again).
Final state verified: 0096/0097 back in `patches/engine/`, serving `eeb98fe202ff`. No restarts lost, no
wedged stops; `safestart.sh` held on all three transitions.

**The contract half of the pre-commit gate passes outright.** All 8 `token_sha` values equal the baseline
(`87fa94aff290 2e49486f633d 39fcf5dc3ba3 690b2a33a2d6 60b55b74b419 29a950f3434c fbe213fd977e fbe213fd977e`)
in every pass, and the pair served healthily throughout -- precommit items 1 and 6.

**The throughput half is unresolvable at the gate's 1% thresholds, exactly as HANDOFF sec 6.1 predicted.**
10-trial 4-stream medians: 49.6 (a1) / 48.5 (b1) / 40.5 tok/s (a2) -- means 49.16 / 46.39 / 43.11.

- `|A1 - A2|` = **18.3% on medians (12.3% on means) between two runs of the identical config 26 min apart**,
  with within-pass trial spreads of 52-104%. Every 1%-threshold comparison in `PRECOMMIT-0096.md` (items 2-5)
  is smaller than the instrument's own wander. Read alone, a1 vs b1 (-2.2%) would have "shown" a 0096
  regression; a2 dissolves it. This is the drift trap in miniature.
- The a1 -> b1 -> a2 means decline monotonically, 49.2 -> 46.4 -> 43.1 (-12% in 38 min). Whatever this component
  is, it is slow and one-directional within a run of hours -- direct input to the uptime/drift lever (sec 6.5).
- **Two distinct noise regimes now documented.** The coverage 4-stream bench is bimodal in round ms *within* a
  single pass (modes near 110 ms and 150-185 ms; per-trial tok/s 26-68). The leg-sweep 4-stream benchmark after
  the same restarts is 0.01-3% stable (108.4 ms constant in `leg-head_q8`). Coverage passes remain useful for
  contract/contention coverage; effect sizes come only from the clean leg sweep, interleaved.
- Mixed prefill-under-load drifted 1576 -> 1638 -> 1650 tok/s across passes; same caveat.
- The KV-pool fit itself wanders with free host memory: 4098 pages (a1) vs 6062 pages (a2), identical config and
  image, 28 min apart -- the unified-memory fit probe sees whatever the host is holding at that moment. Benign for
  the decode bench (both pools dwarf its working set), but it means "same config" does not mean "same memory fit".

**Verdict recorded: 0096 is inconclusive on throughput** -- not keep, not revert; per the precommit rule it stands
as *a correctness fix with an unmeasurable-at-this-noise throughput effect*, and per HANDOFF sec 6.1 no further
restarts are spent on it. 0097 stays (inert under the default `cut` rule, as pre-agreed).

Harness note: a2's long-context leg died in `dec.py`'s summary print (`round p50` formatted a `None` after every
1-stream longctx response came back without usable seconds/prefill timestamps; `spec` itself was present). Fixed
in place: the summary and noise-floor prints are None-safe (`n/a`) instead of raising, so one degenerate load can
no longer cost the rest of a pass. a2 therefore has no 4-stream longctx point; the 1-stream longctx legs that did
run (a1 9.6 vs b1 14.4 tok/s, spreads 71/92%) are too noisy to gate on and are not used.

In flight as this is written: `head-aba.sh` (launched 07:54:27Z) -- the A->B->A confirmation of `HEAD=q8`.

## HEAD=q8 confirmed by interleaved A->B->A (2026-11-03, 07:54-08:07Z): +3.7 / +2.2 / +2.0%

`head-aba.sh`: baseline (a1) -> `HEAD=q8` (b) -> baseline (a2), all three legs on the 0096 image
`eeb98fe202ff`, full `leg.sh` each (verified restart, 3 trials a load on the fixed prompts, contract corpus).
Arm identities verified from the engine banner: model 83.58 / 83.29 / 83.58 GiB -- the B leg really loaded the
halved head (0.29 GiB a rank less).

| load | a1 | b (HEAD=q8) | a2 | drift \|A1-A2\| | effect B vs pooled A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 stream | 37.42 | **38.59** | 37.40 | 0.04% | **+3.69%** |
| 2 streams | 52.92 | **54.14** | 53.04 | 0.22% | **+2.21%** |
| 4 streams | 68.89 | **70.20** | 68.90 | 0.02% | **+2.01%** |

- Round ms @1 stream: 53.41 / 53.37 -> 51.92 = **-2.8%**, against the byte-budget prediction of -2.4%. The
  prediction has now been confirmed twice (06:54: -2.2%, now: -2.8%).
- Tokens and per-trial `token_sha` are **identical across all three arms** on the sweep prompts, and the contract
  corpus is 8/8 identical in every leg. `aba.py` prints no IDENTITY warnings.
- The A arms landed within 0.25% of each other at every load -- the interleaved design does its job; compare with
  the same-config 18% wander of the coverage bench the same night.
- Replicates the 06:54 non-interleaved estimate (+3.1 / +0.9 / +1.2) in direction and scale, larger at 2/4 streams.
- B's 1-stream trial 1 was the slowest B trial (37.06, right after restart) -- the cold-first-trial pattern the
  old baseline leg also showed; including it, B still wins every load.

Quality evidence stands from the 06:54 pass (unchanged config, re-confirmed contract tonight): 10/10 greedy
answers `11ceddde19d4`, paired per-token NLL delta -0.00050, 95% CI -0.00156..+0.00056, `quality.py` +0.0003.

**Status: NOT shipped. Numerics -- user approval pending.** Presented with the full evidence same day.

## Round 7: quality re-bound to the ship artifact; panel consulted (2026-11-03, 08:11-08:40Z)

Codex's round-7 point: the paired-NLL evidence predated the ABA image. Re-run on `eeb98fe202ff` itself
(`nll-baser7` on the live baseline server, `HEAD=q8` restart, `nll-headq8r7`, then restored to baseline-serving):

- Aggregate NLL 1.50609 -> 1.50659. Paired per-token delta **-0.00050, se 0.00054, 95% CI -0.00156..+0.00056 --
  identical to the 06:54 result to every printed digit**, on the exact image that would ship.
- Per-passage: worst |mean d| 0.00448 (medical), 0.00428 (dialogue); sign split 438 better / 446 worse / 167
  identical of 1,051. No domain moves. The baseline arm's aggregate equals the 06:54 capture bit-for-bit
  (1.50609): 0096/0097 are output-neutral on this corpus too.

Panel (docs/speed/panel/brief-r7.md, glm-r7.md, codex-r7.md):

- GLM: wanted an 8-stream point -- answered in the package instead of measured: the served shape caps at
  `max_batch_size: 4` (`/health`), PARALLEL=8 is a different memory fit the precommit itself called a stress probe;
  the shipped shape is fully covered by 1/2/4. Wanted the round-ms spread stated: B's 1-stream round p95 (53.35 ms)
  is below the A arms' p50 (53.41/53.37) -- the distributions barely overlap. Endorsed weighting B (interleaved)
  for effect size and A (06:54) for quality + first sign. No NLL re-run needed, one was done anyway.
- Codex: quality must bind to the shipped artifact -- done, above. "The speed evidence is otherwise ready."

**Package complete. Presented to the user for ship approval 2026-11-03.** q8 remains measure-only until then.

## SHIPPED: HEAD=q8 is the serving default (2026-11-03, user-approved)

Ship act, on the user's "ship it": `HEAD=q8` added to `scripts/local.sh` with the evidence comment (the file that
already carries DENSE=q8 / ABSORB_MMA=1 / W8A8=0 / the reserve). README updated to match the shipped state:
measured-with line, decode table now 38.6 / 54.1 / 70.2 tok/s (the interleaved ABA numbers), settings-table cell
`DENSE, HEAD | q8, q8`. Verified by a clean `safestart` with **no** env override: rank banner reads model
**83.29 GiB** (the q8-head footprint) and the contract corpus is **8/8 identical** to baseline
(`contract-shipped-q8.json`). The pair is serving q8 as of 08:4xZ.

Notes for the next legs:
- `docs/leg.sh` with no overrides now measures the q8 configuration (local.sh wins); a bf16 comparison must pass
  `HEAD=bf16` explicitly.
- `contract-baseline.json` (captured at bf16) remains the valid reference -- the hashes are identical under q8,
  which is the point.
- Rollback = set `HEAD=bf16` in `scripts/local.sh` (or export it) and `bash docs/safestart.sh`.
- Still uncommitted: everything in docs/speed/, the docs/* harness, patches 0096/0097, scripts/local.sh, README.

## The coverage-bench bimodality, phase 1 (2026-11-03, 08:42-09:10Z): a host-memory regime, not an instrument

User directive: the coverage bench's slow mode (rounds 110 vs 150-185 ms within one pass, 26-68 tok/s trials)
was classified as instrument noise; is it a real serving regime? It would dwarf the shipped +3.7% and is
invisible to the leg sweep. Findings, in the order they landed:

- **Correction to "two noise regimes" (the 0096-ABA section above).** The head-aba legs ran stable
  (A arms 0.04-0.22% apart) at 07:54-08:07, the tail of the same morning the coverage passes wobbled; and the
  exact coverage sequence ran clean at 08:52 (10-trial median 69.7, spread 0.67%, p50 109.2 ms - the best
  coverage pass ever recorded here). The wobble follows the clock, not the benchmark: it is a host-state
  regime, and until proven otherwise the two benches measure the same thing in different windows.
- **The bimodal era was the morning memory fill.** sar (sysstat, 10-min grain): 07:20 the box showed 111 GB
  free / 5% used; by 07:50, 91% used - ~108 GB committed during precisely the a1/b1/a2 wobble (07:13-07:51),
  and it stayed at 91% after. Clean-era runs: free RAM flat 1.2-1.9 GB, SM clocks 2164-2184 (no thermal
  throttle, 88 C peak), memguard never fired (no log), heavy paging harmless (the clean pass pulled 474 MB/s
  page-in while holding 109 ms rounds).
- **The traced clean pass has no slow mode at all.** The probe (TF_GLM_DFLASH2_TRACE=1, 7,284 dft records,
  docs/speed/bimodal-analysis.py) joined per-window verify times to trials: med v 98.6-102.1 ms across all ten
  trials, and fast-half vs slow-half at matched r=2 differs by +0.47 ms (n=1,125/1,127). When the slow mode
  next appears, the same instrument answers same-window-slower vs different-regime directly; the drafter term
  (med d 9.2 ms) and the row mix (med r 2.0) were constant.
- **Reproduction failed twice at 08:45-09:04** (traced probe, then untraced coverage.sh): clean both times.
  Whatever the 07:13-07:51 state was, it had passed by 08:45. Best surviving hypothesis: unified-memory
  commit churn while the map fills after the night's idle - which the abba restarts would perpetually renew,
  and which a fresh boot maximizes. This is also the uptime/drift phenomenon's mechanism candidate: not
  "long uptime decays speed" but "the memory map churns while filling and settles after".

Instruments left in place: docs/speed/bimodal-probe.sh (traced block + sampler + stability sweep),
docs/speed/bimodal-analysis.py, per-stop log archives (since the safestart fix). The post-reboot battery
runs the traced probe first thing - the freshest possible fill, the likeliest capture.

## Nsight on the unprofiled path: the capture rig works, CUPTI activity does not (2026-11-03, 11:00-12:10Z)

Built the full wrap-and-capture rig: the engine's entrypoint is env-overridable (`TF_BIN`), so a wrapper on the
mounted state volume runs the rank under nsys with zero start.sh edits; nsys trees (2026.1.3 host, 2026.3.1 from
the dsv41 image) live in `zig-home/nsys*/`, mounted at `/state` on both ranks. The wrapped pair serves and benches
identically (prefill 1734-1738 / 1754-1759 tok/s, cached=0 - the shipped numbers), capture and finalize to
`.nsys-rep` on stop all work (`nsys-r0/r1.nsys-rep`, 11:26Z).

The hole: **CUPTI activity collection is empty everywhere** - kernel/memcpy tables absent from every capture,
host or container, nsys 2026.1.3 and 2026.3.1 alike, and also under a from-scratch minimal CUPTI activity
logger (`docs/speed/cuptitrace.c` -> `docs/libcuptitrace.so`, LD_PRELOAD, CSV per record): header written,
zero records, same on host. NVTX capture works; activity does not. Two independent consumers agreeing points at
the platform (driver 580.159.03 / CUDA 13.3 / GB10), not the tooling. Nsight kernel-level prefill attribution is
parked until an nsys/CUPTI pairing that matches this driver exists; the rig above is the thing to retry with.

Fallback in use meanwhile: the engine's own bracketed per-stage trace (`TF_GLM_PROF=1`, the `glm prompt stage`
lines) - profiled-run caveats apply (~2% prefill cost, prompt pairing disabled, per trap 2/5), so it carries the
structural claims (who waits on whom, side-stream idle) with numbers labeled profiled, never headline rates.

## Prefill stage structure on the shipped config, profiled-run caveats attached (2026-11-03, 09:34-09:41Z)

`TF_GLM_PROF=1` run, pf2 23K x3 + 95K x3 (1702/1697/1686 and 1693/1705/1712 tok/s = the clean 1735/1759 minus
the documented ~2% prof tax; cached=0 throughout); 239,690 `glm prompt stage` lines captured
(`docs/speed/stages-q8.log`), rank 0's 4096-row chunks aggregated:

| site | stage | stream | total s | ms/call |
| --- | --- | --- | ---: | ---: |
| mixer | wait_filled | side | 86.9 | 7.67 |
| ffn | wait_filled | side | 69.2 | 6.10 |
| ffn | ffn | main | 60.3 | 15.95 |
| ffn | experts | main | 54.9 | 15.56 |
| mixer | projections | main | 31.5 | 6.70 |
| mixer | mla | main | 24.5 | 24.73 |
| ffn+mixer | output_projection | main | 35.1 | 1.2-1.9 |
| ffn | shared_expert | main | 20.5 | 1.94 |

Main-stream stage sum 12.14 s/chunk, side 8.71 s/chunk, main waits only 0.42 s/chunk - and the trap-5 weights
reproduce exactly (wait_filled-side dominant, then output projections and the shared expert). The headline
refinement: **the side streams are the waiters, not the waited-on** - 156 s of side `wait_filled` against 9.6 s
of main waits. The side lane's inputs are produced per layer by the main path, and the fill granularity is
per-layer with no cross-layer buffering, so ~6.8 s/chunk of side capacity sits serialized behind the main
pipeline. That is the concrete prefill lever (layer-pipelined side fill); it is an engine scheduling change and
has not been designed, let alone approved. Per chunk wall is not derivable from these sums (stages overlap; trap 5);
relative weights and the wait asymmetry are the claim. All numbers profiled-run: structure yes, rates no.

## Session close 2026-11-03 ~12:00Z: state and the exact resumption points

Done this session, beyond the record above: campaign committed (10 commits, clean tree; NO git remote exists -
push is a no-op until one is added); codex-shim re-pointed to rtx:8888 and verified live; stop ladder in
stop.sh + start.sh abort, safestart now stops through stop.sh (log archives restored; the pre-fix gap cost the
ABA-era raw logs); bimodality phase 1 (memory-fill regime; tracer staged); Nsight rig built and the CUPTI-activity
platform gap documented; prefill stage structure measured on shipped q8 (side streams wait per-layer on main).

**Pending user action: `sudo reboot`** (I hold no passwordless reboot). Immediately after the box returns:
`bash docs/speed/uptime-post.sh` (~18 min: traced fill-churn capture -> bf16 uptime leg vs the 37.42/52.92/68.89
A-arms -> shipped-q8 check; leaves the pair serving q8). Compare with `aba.py`; the drift verdict and the
bimodal-capture analysis fall out of those three jsons.

Next session's queue, in order:
1. Run/analyze the post-reboot battery above (the uptime/drift verdict; possibly the same phenomenon as the
   morning fill).
2. `kda` (12.99 ms of the 44.5 ms 2-row verify, flat in rows - the largest fixed per-forward cost). Entry
   points found: the decode step is the generated kernel `kda_rows` ("the fused step over one Mac's 32 heads"),
   built comptime in `families/glm/wide.zig` via `kernels.generatedSource("kda_rows")` / `generatedFunction`;
   variants `kda_pre` / `kda_fg[nw]` (f/g dot widths) at wide.zig:35-36,71. The prompt-side sibling trio is
   `kda_prep/kda_scan/kda_post` (kernels.zig:91-96). Byte model first, then `ncu` - NOTE ncu uses the profiler
   API, not the activity API, so it may work where nsys does not; the rigged capture paths above all still hold.
3. Nsight/CUPTI activity retry when an nsys matching driver 580 exists (rig ready, section above).
4. The layer-pipelined side-fill idea (prefill), and MTP-at-low-load cut retune, in that order of promise.

## Pre-reboot addenda (2026-11-03, ~12:10Z; user directives)

**Bimodality status correction.** The memory-fill story in the phase-1 section above is the leading hypothesis,
not a closed finding: what exists is correlational (fill rising through the wobble window, one clean pass an
hour later). Confirmation lands with the post-reboot battery's traced capture (`docs/speed/uptime-post.sh`,
leg 1) - the freshest possible fill, instrumented. Until that reproduces the slow mode with the tracer
watching, "probable cause identified, confirmation pending".

**compass-pg.** The postgres container (`compass-pg`, up 3 days, unrelated to this campaign, restart=no
policy) dies with the reboot and will not come back on its own. Flagged for the user to restart; not ours to
start. Also pre-existing on this host, exited, NOT touched by this work: sparkDash, the nexus stack (5), the
autoclip stack (7), slate, nexusui, local-llm, model-trader pair, metube, promptwiki, newsdesk, sig pair,
bolt-dev.

**Campaign backup.** `git bundle` (all refs, 22 commits, 7.3 MB, verified complete) copied to
`spark4:~/backups/tf-campaign/campaign-20261103T1006.bundle`; restore proven by a test clone on spark4
(log and clean tree verified, temp removed). The single-copy risk is retired for the reboot window.
User to decide the real remote; candidates when they are: any bare repo on spark4 or the rtx box
(`git init --bare` + `git push --mirror`), or a hosting provider - the bundle re-seeds any of them with
`git clone <bundle>`.

**Reboot order (policy: never reboot with the pair live; sudo reboot stops the very container whose
docker stop failed at 05:07Z).** `./stop.sh` -> verify absent on BOTH boxes -> `sudo reboot` (user-run) ->
`bash docs/speed/uptime-post.sh`.

**Priority order after the battery, user-set:** the prefill side-fill structural lever is worked first or
alongside `kda` - as a DESIGN PROPOSAL consulted with codex and glm before any code; numerics measure-only
until approved, per standing rule. `ncu` approved to attempt - it uses the profiler API, not the activity
API, so it may succeed where nsys's empty captures did not.

## Post-reboot battery (2026-11-03, boots 12:10 local / 10:12Z; battery 10:12-10:27Z): same-sign twice, below resolution - the decay series decides

Three legs, every contract 8/8 identical, pair left serving the shipped q8 configuration.

**The drift verdict (the battery's purpose).** Same configs, before vs ~10 min after a clean reboot of
spark3 (up 4d22h, never previously restarted):

| config | pre-reboot settled (evening) | post-reboot | delta |
| --- | --- | --- | --- |
| bf16 head | 37.42 / 52.92 / 68.89 | 37.7 / 53.3 / 69.7 | **+0.8 / +0.7 / +1.2%** |
| q8 head (shipped) | 38.59 / 54.14 / 70.20 | 38.6 / 54.7 / 70.8 | +0.0 / +1.0 / +0.9% |

Second independent observation of the effect (the first was +1.5% after spark4's power cycle), same sign,
consistent across loads, spreads 0.1-1.4%. **Honest size: consistent in sign across two independent boot
events, below resolution** - sigma ~=1.5% was measured across restarts, which is exactly the fresh-vs-aged
comparison, so ~1% needs tens of restarts per arm to establish (the campaign's own sizing: 2% ~9, so 1% ~35;
six non-negative deltas, five positive, is a sign test at p~0.11). Directionally consistent, not established.

**The instrument is redesigned to sidestep across-restart drift entirely (user directive): a within-boot decay
series off boot 12:10:43 - repeated multi-trial sweeps at +1h, +3h, +12h, +24h with NO restart between points,
every point sharing one boot.** The 71.1 -> 70.8 @4 between minute 6 and 10 was the first sample of exactly
this. At every point, alongside the mean: **trial spread and slow-mode incidence** (the share of trials in the
slow round-ms mode), same runs, no extra cost - see the reframe below. Uptime vs accumulated daemon load is
NOT separated: the restart lever addresses both, so the confound does not change the decision.

**The reframe that carries (user): bimodality and uptime may be one phenomenon.** Fresh boot: zero slow modes,
the cleanest spread ever (0.51%). Aged box: the morning wobble, rounds 110 vs 150-185 ms. If slow-mode
incidence climbs with uptime, the restart lever's value is not ~1% on the mean - it is removing a 40-70%
p95 cliff at 4 streams, which dwarfs everything in this campaign including the shipped q8. A mean-only decay
curve would not see it; hence incidence and spread co-recorded at every point above. If the +1/+3/+12/+24h
incidence trend is real, it becomes the campaign headline and reorders priorities.

**Bonus: a third independent q8-vs-bf16 A/B, same fresh box 5 min apart** (leg 2 bf16 10:18-10:22Z, leg 3 q8
10:22-10:27Z): +2.4 / +2.6 / +1.6% at 1/2/4 streams, q8 wins every load, both contracts identical. Retrospective
strength for the shipped decision: three A/Bs, three sweeps-worth of signs, all q8.

**Bimodality, second traced capture (leg 1).** The post-boot window is the fastest and cleanest yet measured:
71.1 tok/s @4 (fastest of the campaign), spread 0.51%, and the tracer shows no slow mode - med verify 94-101 ms,
fast-vs-slow halves under 1.1 ms apart, row mix constant. The simple fill-churn story (fill => slow rounds) is
NOT confirmed: fresh was fast immediately. Refined hypothesis: the morning wobble lives in the transition - the
host memory map growing under daemon load (111 GB free -> 91% used in 40 min) while the engine runs - not in the
engine's own steady state, fresh or settled. Still confirmation-pending: the wobble has never been caught with
the tracer on; the instrument stays permanent and the next instability window gets the same-window-vs-regime
verdict. status: probable-cause REFINED, capture still owed.

Operations: compass-pg came back up at boot (user-side); all pre-existing exited containers untouched; bundle
refreshed to spark4 after these commits.

## Side-fill probe stage: pre-registered gates (2026-11-03, 10:40Z - BEFORE any probe runs)

User directive: the threshold is a number written down before P2, with Codex's null preference as the gate.
The advisors' disagreement (GLM +8-12% likely/+16% best; Codel per-branch <=8%, "prefer null at 2%") is not to
be resolved by argument - P2 measures it. Pre-registered, for the best candidate that survives P1:

- **>=3% repeatable end-to-end prefill gain** (P2 additivity + P3 joined-makespan, extrapolated with explicit
  contention allowance): proceed to an engine-change design proposal for approval.
- **2-3%**: null UNLESS the scheduling change is demonstrably trivial (<=1 engineer-day in existing
  stream/event plumbing) - Codex's qualifier, operationalized; otherwise null with the trace attached.
- **<2%**: null, outright, no discussion.
- "Repeatable" = holds across interleaved x3 runs (the bimodality rule) with contract 8/8 intact at every step.

Sequence locked with the user: the decay series owns the pair until the +24h point (~10:12Z tomorrow);
P1 now is CPU-only analysis of the existing 239,690-line stage trace plus source reading - no rebuild, no GPU,
no restart. P2/P3 only after +24h lands. Any forced restart voids the series from the new boot (announce,
do not patch across). If the series dies, its partial points still stand as within-boot pairs up to that time.

## Decay series: anchor ruled comparable; decision rules pre-registered (2026-11-03, ~11:25Z - BEFORE the +3h point)

**Schedule correction (user): +3h lands 13:10:43Z, not ~14:10Z - boot 10:10:43Z; the sched log's 6969 s
remainder agrees.**

**t+0.2h anchor admitted, with its caveats in the open.** The battery's leg-1 block is the SAME instrument as
the series' incidence block (10 trials, 4 streams, fixed prompts, 478 rounds everywhere, cached=0 on every
trial both sides, acceptance 0.603 and drafted/round 1.709 vs 1.72 - same compute shape). Two deltas, both
ruled benign: the battery leg ran with `TF_GLM_DFLASH2_TRACE=1`, t+1h without - trace overhead would slow the
battery, and the battery is faster anyway, so the measured decline is conservative (a floor); pool/cache state
differences do not touch the decode blocks (no cache hits possible, cached=0 verified). **Anchor: 71.01 mean at
t+0.2h vs 70.55 at t+1h = -0.65% in the first hour** (medians 71.1 -> 70.5, -0.8% on the stricter reading).
Early-slope implication, stated as the fork it is: if most of the ~1% total lands in hour one, scheduled
restarts are near-useless (one would be restarting continuously); if the decline is spread across 24 h, a
daily restart captures it. The +3h point begins to separate these; no conclusion before it.

**Pre-registered decision rules for the series (written before +3h, per the user's demand - the same fix the
side-fill got):**

- CONFIRM uptime->slow-mode (one-phenomenon) iff: at ANY two consecutive points from +3h onward, slow-mode
  incidence (round-mean > 130 ms trials, the 10-trial 4-stream block) is **>= 3/10 at one and >= 5/10 at the
  next**, AND the t+1h->that-point mean decline at 4 streams is >= 1%. Incidence alone without decline, or
  decline without incidence, does NOT confirm - they must move together to be one phenomenon.
- KILL uptime->slow-mode iff: incidence stays **0/10 at every point through +24h** (t+1h already 0/10) while
  the 4-stream mean stays within +-1% of the t+1h value. On kill: the p95-cliff question is reassigned to the
  host-map/daemon-load transition (the morning-fill regime of the 0096-ABA section), this line CLOSES, and the
  closure says so in one paragraph - no hunting for a residual effect without a new, pre-registered rule.
- MIDDLE cases (incidence 1-2/10 sustained, or decline 1-2% with incidence flat): the series EXTENDS by one
  pre-registered point at +48h, same rules, decided once - no serial extension.
- Cadence rule (the practical output either way): if CONFIRM, restart cadence = the interval at which the
  4-stream mean has lost >=1% AND incidence >=3/10, read off the fitted curve; if the curve flattens before
  -1%, no scheduled restarts - the reboot lever dies with the same one-paragraph closure.
- All readings from the untraced sweep numbers (trace-env runs are anchors only, marked); every point's
  contract set remains a precondition read before any number.

## Decay rules v2: the two questions decoupled; the mean lever bounded by breakeven (2026-11-03, user)

Ops rule first, learned from the bundle gap: **commit + re-bundle after EVERY decay point**, not at series end -
the t+1h artifacts existed on one disk (the box under test, wedged once this week) until c1efd78;
campaign-t1h.bundle now carries them, restore-tested on spark4.

**The KILL rule's conjunction is split - incidence and mean answer different questions and get their own gates:**

- P95/one-phenomenon question - gate on INCIDENCE alone: KILL iff 0/10 at every point through +24h (t+1h 0/10
  already) => p95 cliff reassigned to the host-map/daemon transition, line closed in one paragraph. CONFIRM iff
  the incidence ramp AND the mean decline move together (>=3/10 then >=5/10 at consecutive points, >=1% decline)
  - the conjunction stays here, where "one phenomenon" is the claim.
- Cadence/lever question - gate on MEAN alone: a confirmed mean decline stands on its own evidence at any
  magnitude; incidence 0/10 does not weaken it. The +48h extension goes to WHICHEVER question its own gate
  leaves unresolved - a 2% decline over 0/10 incidence is a clean DOUBLE RESULT (p95 story dead, lever confirmed
  larger), not MIDDLE, and spends nothing.

**The lever's economic bound, pre-registered (user's arithmetic, adopted):** a verified restart costs 133 s
(2m13s, safestart-measured). At interval T hours, availability loss 3.69/T %; mean decline r %/h costs ~rT/2
before a restart; total ~rT/2 + 3.69/T, minimised at T* = sqrt(7.38/r), optimum loss ~2*sqrt(7.38*r)/2...
at r=0.042 (1%/24h linear): T* ~13 h, total loss ~0.55%. Consequences, written down to prevent over-investment:
(i) restarting pays at declines well BELOW 1%/24h-equivalent - the action threshold is the breakeven, not 1%;
(ii) **the entire mean-decay lever is CAPPED at ~0.55% of availability even if decline is real and linear
unchecked - less if the curve plateaus.** The p95/incidence question is therefore the only part of this series
that can exceed rounding error; if MIDDLE triggers, the +48h point resolves INCIDENCE, not the mean.

## Standing orders (2026-11-03, user; in force)

Report ONLY: a measured number or a blocker. No memos, no priors, no candidate rankings. Decay points as they
fire (self-committing) and the P3 verdict with the gate applied are the deliverables. kda analysis SHELVED -
if kda is ever worked, it starts with ncu on the GPU, not more source reading. Silence otherwise.

## Decode attribution campaign on the pair, and the spark1+2 comparison (2026-11-03, 15:00-21:30 local)

The question: spark1+2 (the GB10 recipe's `zig/python-parity-defaults` branch, TensorFold 1.0.7 Zig port, fp8
dense, bf16 drafter, 8-stream arena) decodes 95 tok/s at 4 streams against this pair's 80, and 40.5 against 38.4
at one stream, on the same checkpoint weights (snapshots 76c0b517 and 078455ff differ only in model-card commits).
This pair prefills faster (1,715-1,780 vs 1,624-1,632 tok/s) and keeps a chat alive beside a long prefill
(14.5 tok/s; the branch freezes it). Every claim below is exact: the greedy token hashes (prose 64c908210a9e,
code 315341215112) were identical in every run.

**Where a round goes (TF_GLM_PROF=1, the lone and, with 0099, the shared forwards):**

| forward | wall ms | routed | kda | mla | router | gathers | head | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| verify 2 rows | 43.8 | 17.2 | 13.0 | 5.0 | 3.3 | 2.5 | 1.5 | 1.2 |
| verify 4 rows | 56.5 | 28.4 | 13.3 | 5.1 | 3.5 | 3.1 | 1.5 | 1.2 |
| verify 8 rows | 83.1 | 51.9 | 14.2 | 5.5 | 3.7 | 4.4 | 1.5 | 1.2 |
| shared 8 rows (4 streams) | 82.4 | 49.4 | 16.4 | 5.5 | 3.5 | 4.1 | 1.5 | 1.2 |
| shared 10 rows | 93.1 | 58.5 | 16.8 | 5.8 | 3.7 | 4.6 | 1.5 | 1.2 |
| shared 14 rows | 110.8 | 74.6 | 17.4 | 6.2 | 4.0 | 5.0 | 1.5 | 1.2 |

Routed experts cost 4.2-5.8 ms a row whether the rows come from one stream or four: 8 fresh experts a row a
layer at 4 bpw is 1.4 GiB a rank, 5.3 ms at 273 GB/s, so the per-row cost is the bytes. The drafter pass
(TF_GLM_DFLASH2_PROF=1) is ingest 0.5 + block 4.5 + select 0.05 ms at one stream, block 7.2 at four, plus ~3 ms
of rank exchange the phases don't see. KDA mixers 13 ms flat in rows (the projections, 50 MB a layer a rank at
q8, run at ~60% of bandwidth: 0.308 ms a layer from the prompt-stage trace; the chain is the rest). A 4-stream
round is forward + drafter + ~3 ms: 93 + 8 + 3 = 104 ms measured.

**What was tried, each a restart on production, none a gain:**

| lever | 1 stream prose / code tok/s | 4 streams tok/s | note |
| --- | --- | --- | --- |
| baseline (cut 0.7) | 39.0 / 57.3 | 78-80 | |
| bf16 drafter (DRAFTER_BITS=0) | 36.9 / 53.3 | 76.7 | same acceptance, heavier pass |
| TF_GLM_DRAFT_CONFIDENCE=0.5 | 37.5 / 55.3 | 78 | |
| TF_GLM_DRAFT_CONFIDENCE=0.3 (the branch's) | 34.6 / 52.0 | 76-77 | 2.22 tok/round but 64 ms rounds |
| TF_GLM_DRAFT_RULE=tpm | 38.7 / 55.9 | 80 | |
| 0098 TF_GLM_DFLASH2_EDGE=0.6 (dflash2.py's) | 39.0 / 56.8 | 81-82 | parked: docs/speed/candidates |
| 0100 TF_GLM_MM_STAGES=5/6/8 (cp.async depth) | 38.1-38.5 / 56.6-56.8 | 79-84 | parked: docs/speed/candidates |
| DENSE=bf16 | 30.5 / 46.5 | 70 | +14 ms a round for +1.5 GiB |
| stock TensorFold v1.2.1, no patches | 35.8 / 53.3 | 67-71 | prefill 1,058-1,086; refuses fp8 KV on pages |

Ten essay topics at one stream: this pair 39.8 +- 1.3 tok/s at 60 +- 3% accepted, the branch 40.5 +- 2.1 at
64 +- 4%; equal within noise, and this pair wins code (54.1 vs 50.6). The branch's margin is only at 4 streams,
where it yields 2.34-2.47 tokens a round against 2.07-2.11 here at the same 102-105 ms. Its drafts per stream
per round are 2.0 against 1.7, and it lands 68-71% of them against 61-64%; the same chain rule at its threshold
(0.3) on this engine offers more rows but lands fewer and lengthens the round. Kernel-level attribution of the
branch is not possible here (CUPTI activity is empty on this driver, see above), so the difference stays as
measured, not explained.

**Shipped:** 0099 (TF_GLM_PROF=1 brackets shared windows: "shared N rows"), diagnostic only. **Parked**
(docs/speed/candidates/, exact by construction, neutral in speed): 0098 (selector edge weight), 0100 (decode
dense linears at 5/6/8 pipeline stages). **Not pursued:** v1.2.1 as a base (its stock numbers above; 20 of 97
patches apply, 77 conflict); it would be a selective re-port later, with TF_GLM_CHUNK_SOLO=4096 the one thing
worth having from it (a 4,096-row chunk only while nothing decodes).

**Operational notes:** an image build on a serving Spark trips the memguard (2.2 GiB available under the 2.5
floor at 16:59Z stopped the pair; memguard.log in that archive). Build on spark1 with its pair stopped, ship
with `docker save | ssh docker load`; keep candidate patches out of patches/engine on the serving host so
start.sh never builds there. The local toolchain's `zig build test-glm-*-host` steps catch Zig errors in a
minute (a hand-built Kernels table in cuda_kda_tp3_test needs new tables to be optional); they do not catch a
`[:0]const u8` kernel name because the test mocks take `[]const u8`.

## Direction 1 and 2 answered; the knob matrix is flat; 0098+0100 promoted; PARALLEL=8 shipped (2026-11-03, 21:40-23:59 local)

Continuation of the campaign above, same rules. Every number below from the rtx box with docs/speed/harness/acc.py
against the live pair; hashes prose 64c908210a9e / code 315341215112 identical in every run.

**Direction 1 (PARALLEL=8), measured on image b9d260ebc60b, env-free:** 1 stream 38.3-38.4 prose / 56.5-56.6 code;
4 streams 80.6-82.2 (equal to PARALLEL=4's 76.9-83.0); prefill 1745/1780; conc2 58.6; mixed1 11.2 tok/s, gap_max
2.67 s - all within noise of PARALLEL=4, window stays 1M (5,890-5,908 pages). What it buys: 6 concurrent 91.4-93.5
tok/s, 8 concurrent 105.0-106.9 (2.00 tok/round, 149-152 ms) - shapes the 4 arena capped at 78-80 or refused
(max_batch was 4). The 4-stream number is not an arena artefact: at 4 streams the shared-row budget binds, not the
arena. **Shipped as the default (scripts/local.sh): PARALLEL=8.**

**Direction 2, the free-running trace (TF_GLM_DFLASH2_TRACE=1, 4 streams, 400 rank-0 windows):** offered 1.94
drafts/drafted window (1.80/window with the 29 plain), 1.19 land - 61%, matching the counters. By offered depth:
r=2 n=172 a=0.576; r=3 n=90 a=1.300; r=4 n=79 a=1.886; r=5 n=23 a=2.130. Conditional landing stays healthy at
depth (r=3 windows land 1.30 > 2x the r=2 population's first-draft 0.576 would give if depth-2 were weak); the
0.7 chain cut binds before the allocator's ~0.47 break-even. Forced-3 shared lands 1.22 < free-r3's 1.30: the
selector+cut pick better windows than uniform depth - selection works; the deficit vs B is not in the cut.

**Codex consult (gpt-6.1-sol, xhigh, 23:00Z, full text in the session scratchpad as codex-decode-op.md):** ranks
H2 (B produces better deep proposals) > H3-conditioning > H1 (confidence bookkeeping) as the explanation of the
whole gap; the free/forced contrast shows selection bias, not depth-quality; widening the 1-draft windows alone
caps at +0.248 tok/stream/round for +8.7 ms (perfect-continuation bound ~83.3 total) - insufficient for B's +12;
the decisive missing measurement is a matched-prefix forced-width A/B landing trace, which B's image cannot run
(no TF_GLM_DFLASH2_TRACE in its 0004 patch - verified on spark1); a static landing table's failure modes are
censored calibration and lost per-domain selection; equal mean NLL does not bear on argmax/acceptance. My
decisions taken on this: (a) measure the one untested cell combination before any port (below); (b) do NOT port
B's drafting now - the source-level comparison (B zig/patches/0004 cuda_draft_policy.zig vs A dflash_select.zig)
shows the same pick form (value+edge, chained product cut), B's differences being edge 0.6 vs 1.0, beta=2
confidence, cut 0.3, and a final expected-ms depth choice - each covered by an A knob measured below; (c) the
H2 test (B-side instrumentation or a conditioning crossover) is the next session's item, with KDA C1 as the
parallel exact lever.

**The knob matrix on the new image 8156bfb7d08e (0098+0100 promoted into patches/engine; image built on spark1
with its pair stopped, shipped docker save/load to both Sparks, label verified before any restart on spark3):**
4-stream total tok/s, reps as listed. 1-stream and hashes unchanged everywhere.

| cell (rule x edge x stages) | reps | mean | note |
|---|---|---|---|
| cut0.7 x e1.0 x mm3 (old image b9d2, this morning's baseline) | 7 reps today | 80.6 | |
| cut0.7 x e1.0 x mm6 (8156 as built) | 78.6, 80.6 | 79.6 | 0100's default stages: no gain, no loss |
| cut0.7 x e0.6 x mm6 | 80.5, 80.3, 81.8 | 80.9 | 0098's Python edge weight: within noise of baseline |
| tpm x e1.0 (on b9d2, edge env inert there - re-labelled) | 79.1, 81.8 | 80.5 | |
| tpm x e0.6 x mm6 | 81.4, 80.5 | 81.0 | best cell; still within noise of baseline |
| cut0.5 x e1.0 (b9d2) | 78 | | deeper offers, longer rounds |
| cut0.3 (b9d2) | 76-77 | | |
| EDGE=4.0 sensitivity probe (b9d2) | 80.1, stats = baseline | | env inert on b9d2 (0098 absent): earlier "edge" cells re-labelled above |

**Conclusion, measured:** on this engine the cut/edge/depth-rule/stages/drafter-bits/arena axes are all flat at
79.5-81.5 against B's 93-95. The gap needs ~+0.25-0.3 tok/stream/round at equal ms; no env-reachable lever
found; per Codex the remaining candidates are proposal quality (selector inputs/candidate space - needs the
matched-prefix A/B or a conditioning crossover) and the exact ms-levers KDA C1/C4 (bound +1-3 ms/round ~ +1-2
total). **Shipped state:** image 8156bfb7d08e (100 patches = old 98 + 0098 + 0100, both exact, defaults
behavior-preserving except 0100's mm6, measured equal), PARALLEL=8, env-free; final verification 38.3-38.4 / 56.6
1-stream, 79.7-79.9 at 4 streams, hashes unchanged. B (spark1+2) restored after the build and LIVE.

## The paired-metrics instrument, the drafting plateau closed with it, and the campaign's measured verdict (2026-11-04, 00:10-03:40 local)

Overnight continuation. Every claim measured; hashes prose 64c908210a9e / code 315341215112 unchanged in every
run. Three Codex consults (gpt-6.1-sol xhigh; full texts docs/speed/codex-endgame.md, codex-final.md, and
codex-decode-op-20261103.md).

**The instrument (new, zero-patch, works on BOTH engines):** the served /metrics counters
(tensorfold:mtp_drafted_total / mtp_accepted_total / decode_rounds_total) are cumulative per-finished-request
totals from the SHARED lanes core - the same accounting code on A and B (verified in source both sides: offered =
window rows-1, accepted = kept-1 per head window). Snapshot / run a fixed workload / snapshot: exact per-offer
acceptance and drafts/round per phase, on the production pair, no restart, no patch. The handoff's "counters may
count differently" warning is thereby resolved: they do not differ; bookkeeping is not part of the gap.
 harness/acc1only.py and acc4only.py are the phase-clean runners.

**The paired result (identical prompts, both engines, tonight):**

| engine, phase | drafts/round | accepted/offered | tokens/round |
|---|---|---|---|
| A 1-stream prose | 1.870 | 0.600 | 2.122 |
| B 1-stream prose | 2.087 | 0.665 | 2.388 |
| A 4-stream prose | 1.769 | 0.619 | 2.096 |
| B 4-stream prose | 1.949 | 0.677 | 2.319 |
| A 4-stream, full mimic (edge0.6+TCONF2+cut0.3) | 1.292 | 0.739 | 1.954 |

B leads tokens/round +12.6% at 1 stream and +10.6% at 4 - the gap is engine-intrinsic, not a 4-stream effect.
The handoff's "equal at 1 stream" holds for tok/s only via B's slower per-token code path; tpr never matched.

**The plateau (13 configurations today, every one 79.5-83 total at 4 streams):** cut 0.3/0.5/0.7; selector edge
0.6/1.0/4.0; tpm rule; the full B confidence mimic on the new patch 0101 (TF_GLM_DFLASH2_TCONF, exact, default
1: the chance softmax's temperature, pick untouched); mm-stages 3/6; drafter bf16; arena 4/8; forced width 4
(68.5-72.7: deeper unconditional offering LOSES at 4 streams - the shared forward's marginal row returns less
than the lone 5.05 ms table past ~3 rows/stream). The calibration sub-experiment (Codex-designed): A's stated
chances are OVERCONFIDENT (stated 0.475 vs realized 0.307 joint at depth 2, x0.65; x0.54 at depth 3); correcting
the cut on measured landing moves along the frontier, not above it - extension of stopped-at-1 windows caps at
~84 total (Codex-checked arithmetic; my first 0.102 break-even ignored the shared 4-stream denominator, true
value 0.408).

**The structural candidates, adjudicated:**
- S1 conditioning (A's verify-taps vs B's re-run): EQUIVALENT by source on both sides - A's keptRuns ingests
  exactly the accepted prefix's tap rows and prefix acceptance makes their states accepted-only; B commits
  taps[:keep] likewise (Codex read both trees; concurs). No cheap win.
- S2 context restarts: measured lost=0 over 2,048 rounds - excluded by Codex's own pre-registered gate, no patch.
- S3 drafter-forward numerics (A's drafter states via taps through the target's fused fp8 pipelines vs B's
  dedicated bf16 drafter forward): the ONE mechanism left standing; untested; its remedy is porting B's
  cuda_dflash_forward as A's drafter path - a structural project (B's 400-line forward + weights path + ring
  integration), not a patch.

**The verdict, with the arithmetic that makes it honest:** 93 total at 4 streams needs 2.42 tpr at A's 104 ms,
or 2.25 at <=96.8 ms (Codex-checked). Even porting B's exact tpr (2.319) onto A's latency gives ~89.2. The
policy/patch-level space is exhausted at ~83 (13 cells + calibration bound + S1/S2 exclusions); KDA C1/C4 add
+1-2 ms (81.4-83.8 ceiling, Codex) and cannot close it alone. **The goal "beat 93-95 at 4 streams" is not
reachable on this engine by any change short of the S3 drafter-forward port** - measured, not conjectured, with
the paired instrument to verify any future attempt in 20 minutes.

**Shipped by the campaign (all exact, verified, on both Sparks as image d9f47d5bc28f = 101 patches):** 0098
(selector edge weight env; default = old behavior), 0100 (mm pipeline stages; default 6, measured = 3 within
noise), 0101 (chance temperature env; default 1). PARALLEL=8 in scripts/local.sh (8 streams: 105-107 tok/s, 6:
91-94; everything else unchanged). Final bench2 on the shipped state: 38.6/56.8 1-stream, prefill 1737/1776,
conc4 80.4, mixed1 11.7 tok/s with 2.66 s worst gap, conc2 55.0 - every protected win intact, hashes unchanged.

**Left for the user (not taken unattended):** the S3 port (pre-registered go gate: paired tpr >=2.25 at
<=104 ms twice, hashes unchanged; else stop) and KDA C1 graphs (+1-2 ms, 81.4-82.2 ceiling) as an independent
quality-of-engine win. Both pairs LEFT SERVING: A on d9f47d5bc28f env-free, B restored on its image.

## M1 ran: the drafter weights themselves differ - the 64 vs 32-group quantizer owns the gap (2026-11-04, 10:20Z)

The S3 port's M1 (state diff) executed as a pure-host f64 harness (docs/speed/m1/m1diff.zig, zig 0.17 run, the
real DFlash2 safetensors) after the source reading collapsed the candidate set: A's dflash linears ARE the
python engine's qmm_group kernel (same family B's lane path serves), the conditioning and ring semantics are
alike, so the weight representation was the first stage tested - and it owns the delta.

| tensor | relrms A64 (shipped) | B32 (reference engine) | 64-grp f32-scale control | A-vs-B |
|---|---|---|---|---|
| q_proj 4096x4096 | 0.0904 | 0.0786 | 0.0905 | 0.1033 |
| o_proj | 0.0897 | 0.0782 | 0.0980* | 0.1017 |
| down_proj 4096x12288 | 0.0975 | 0.0840 | 0.0976 | 0.1106 |
| gate_proj | 0.0900 | 0.0783 | 0.0901 | 0.1019 |
| fc 4096x20480 | 0.0903 | 0.0786 | 0.0904 | 0.1033 |
(*o control 0.0898; table rounded) A quantizes every drafter linear at 64-group affine with the scale/bias
bf16-ROUNDED before quantizing; the reference engine uses 32-groups at f32. The control column isolates it: the
rounded-scale nuance contributes ~0.1%; the group size is the whole 13% fidelity difference. The engines'
weights disagree (AB ~0.10) by MORE than either differs from the source. Through one linear (fp64 dot, bf16
activations, 16 rows) that is ~10-14% relative output deviation - compounding through 5 layers into the top16
boundary and the selector's chances, exactly the measured acceptance gap's shape. B's loader takes the plain
32-group path (its MSE variant is another family's flag); no hidden knob there.

Consequences for the plan: the S3 "port the forward" premise reduces to a LOAD-TIME variant - M2 is now "32-group
mode for A's dflash linears": a tf_glm_quantize4_32 (the existing kernel at 32 groups), the q4 pack/size math at
k/32, a qmm_group GS=32 template instantiation (the template is already GS-parameterized; smem re-derived), the
host layout offsets, behind TF_GLM_DFLASH2_Q32=1 default off. Drafts only propose: replies/hashes unchanged by
construction; the gate stays the paired tpr >=2.25 at <=104 ms twice. The full-forward M3 likely never needed.
Caveats, stated: synthetic gaussian activations not real tapped states; dot trees emulated fp64 (real fp32 fma
chains differ at 1e-6, immaterial); B's lane-kernel fold order not emulated (same per-group fp32 structure).
The harness and this table are the M1 deliverable; both pairs untouched and serving throughout.

## M2 ran and closed the small-change campaign by its own gate (2026-11-04, 12:30Z)

Patch 0102 (TF_GLM_DFLASH2_Q32, env-gated, default off): the drafter's body linears at the reference engine's
32-group granularity - tf_glm_quantize4_32, the GS 32 pack tile, the LaneTile<32,...> instantiation, 32-wide
group sums, loader/scratch branches. Built as image cd1de29726c3 (102 patches), shipped to both Sparks, run on
the pair with Q32=1. The mechanism verified end to end: hashes stable in every run (the path is drafts-only as
designed), acceptance normal, no errors, the offered mix changed (the chances moved - the g32 path is live).

The paired verdict, against the 64-group numbers taken with the same instrument:

| config | drafts/round | accepted/offered | tokens/round |
|---|---|---|---|
| 64-group (shipped), 1 stream | 1.870 | 0.600 | 2.122 |
| Q32, 1 stream | 1.830 | 0.595 | 2.089 |
| 64-group, 4 streams | 1.769 | 0.619 | 2.096 |
| Q32, 4 streams | 1.747 | 0.623 | 2.088 |

4-stream acceptance +0.4pt with the better weights; tokens/round flat; totals 76.9-80.0, inside the standing
band. The pre-registered gates (GO 2.25 tpr at <=104 ms twice; partial 2.18) are NOT met: **the M2 stop rule
fires.** Measured conclusion: the 13% drafter-weight fidelity gap is real but does not move acceptance at this
operating point - a 16-candidate argmax is robust to the sub-percent proposal changes weight noise makes; B's
acceptance edge is NOT weight precision. What remains untested is only the deep numerics (the drafter's
attention kernels' state paths - S3's M3 full-forward territory) and the output head (still 64 groups here; its
sums are welded to the shared glue rmsnorm - a bounded follow-up if anyone revives this).

State: image cd1de29726c3 (102 patches; at env-free its behavior is identical to 101 - the union variant and
buffer sizing only) on both Sparks; A serving env-free on it now; B restored and UP. 0102 in patches/engine and
candidates/ with the full record here. The campaign's final shape: PARALLEL=8 (+25-30% at 6-8 streams), the
paired-metrics instrument, three exact env knobs, the M1 quantizer attribution, and this closure - every one
measured, hashes 64c908210a9e / 315341215112 throughout.

## KDA C1 measured before any build: fusion loses, the chain's floor is its bytes, dependent launches win 5-9 us a layer; 0199 pre-registered (2026-11-04, 09:00-11:15 local)

Resuming from HANDOFF-2's item 2 (KDA C1: "merge the verify chain's 3 launches into 1-2, cuda graphs, +1-2 ms/round,
bit-exact"). Operating changes this session, both forced: the rtx box refuses ssh from spark3 (its ED25519 host key
changed; the new key is what spark1 and spark4 also see, and spark1 already trusts it - a stale entry on spark3, left
for the user to clear; updating known_hosts from here was denied), so **spark5** (an idle GB10, not a member of either
pair; Codex 0.166 and nvcc 13.0 present, zig 0.17.0 fetched and sha-checked) is this session's advisor host, kernel
bench host, compile-check host and harness client. Both pairs were left serving throughout.

**What the handoff's number rested on.** The engine's TF_GLM_PROF "kda" bucket (13.0 ms at verify 2 rows, 16.4 at
shared 8 rows) brackets the whole KDA block minus o_proj: the input projections AND the chain. The record (2026-11-03
21:40Z) priced the projections at ~10.5 ms from the prompt-stage trace, leaving ~2.5 ms for the chain; "+1-2 ms" for
C1 was therefore most of the chain's entire cost, not a schedule's share of it. CUPTI is dead on this driver, so the
chain was measured standalone instead.

**The microbenchmark** (docs/speed/kda-c1/bench_kda_c1.cu; spark5, sm_121, driver 580.178.04, nvcc 13.0, the engine's
flags -O3 --fmad=false --ftz=false; the shipped glm_kda.cu included verbatim; 34 layers each with its own 2 MiB
state pair, conv window and rows - and after Codex caught the first version letting 3 of 4 segment states stay in L2,
every layer x segment with its own state; 60 reps x 34 layers, two passes). Every candidate's out, state_out, k, v,
decay, beta saves (and the segments' states) byte-compared against the shipped chain on every layer: **same bits in
every cell of every run.** Mean us per layer-chain:

| shape | shipped prep+step+out | all three as dependents | **prep plain, step+out dependents (0199's form)** | prep fused into step (2 launches) | single fused kernel |
|---|---|---|---|---|---|
| lone 1 row (wp 2) | 32.3 | 25.4 | 27.5-29.2 | 40.7 | 32.0 |
| lone 2 rows (wp 4) | 33.1 | 26.0 | 28.2 | 38.6 | 36.3 |
| lone 3 rows | 34.5 | 27.2 | 29.3 | 45.9 | 42.6 |
| lone 4 rows | 36.4 | 28.7 | 30.4 (one pass 35.0) | 52.6 | 44.9 |
| lone 8 rows | 43.0 | 36.1 | 38.5 (one pass 44.6) | 78.9 | 63.4 |
| lone 11 rows (wp 8) | 45.2 | 38.0 | 39.4 | 57.5 | 75.2 |
| shared 4 segs, 8 rows | 93.5 | 84.1 | 86.5 | 97.4 | - |
| shared 4 segs, 10 rows | 95.9 | 87.3 | 90.1 | 99.8 | - |
| shared 4 segs, 12 rows | 100.0 | 89.3 | 90.1 | 102.8 | - |
| shared 4 segs, 14 rows | 102.6 | 90.1 | 92.2 | 106.8 | - |
| shared 2 segs, 4 rows | 52.9 | 44.3 | 46.5 | 54.4 | - |

Read: (1) **merging prep into step loses** - every step block (8 a head at 4 warps) recomputes the head's prep;
(2) the single fused kernel is no better than three launches anywhere; (3) the chain's floor is its state traffic
(4 MiB a layer = 15 us; 16 MiB at 4 segments = 61 us) against 33-100 us measured, so a schedule can give back at most
the launch gaps; (4) programmatic dependent launches on the UNCHANGED kernels do that: -7 to -10 us a layer with all
three dependent, -5 to -7 with prep launched plainly. Over 34 layers that is **0.17 ms a round at 1 stream and
0.20-0.24 ms at 4 streams: 0.3% and 0.2%**, against the handoff's 1-2 ms. KDA C1's honest ceiling is ~0.3 ms/round.
(The first bench version, states shared across segments, read the shared chain at 57 us - the 16 MiB figure is the
one to trust; the engine's 4-stream chain is ~3.2 ms of a 104 ms round.)

**Codex (gpt-6.1, xhigh, from spark5; brief docs/speed/codex-kda-c1-brief.md, answer docs/speed/codex-kda-c1.md):**
agrees fusion is dead and the PDL gain is ~0.235 ms/round (~80 -> 80.2 tok/s); caught the segment-state cache reuse
(fixed above); notes ~1.4 ms of the inferred 2.5 ms engine chain cost is unexplained (host submission gaps,
attribution) and that graphs are neither shown nor excluded by a standalone bench; judges the dependency placement
correct under the engine's wait-before-trigger invariant; flags waiting CTAs squatting on the kernel before prep (the
glm_mm matmul does trigger early, qmm_frag.cuh) - so 0199 launches prep plainly and lets only step and out depend,
keeping prep's own trigger; recommends out carry no trigger (done); says a build is "defensible as integration
validation, weak as a standalone performance investment" and would bundle it into a next planned image, default off.
**My decision:** build it now, once, env-gated, default on as built - there is no other planned image, the patch is
exact and compiled, and the engine-level A/B is the only instrument that decides whether the chain's engine cost is
GPU-side (then PDL recovers ~0.2 ms and graphs have nothing left) or host-side (then a graph form of C1 is the real
lever, up to the unexplained ~1.4 ms). The gain itself is below the harness's resolution and is recorded as such.

**0199** (docs/speed/candidates/0199-glm-cuda-TF_GLM_PDL_CHAIN-s-kda-...patch - numbered 0103 when built, renumbered 0199 as a parked candidate once the rtx port took 0103/0104; label of the 103-patch set
45a1d17d74d7, replica of prepare.sh --label verified to reproduce cd1de29726c3 on the 102 set): tf_glm_kda_prep_p /
step_p / step4_p / step2_p / out_p and the _segs_p forms (prep waits then triggers at entry; step loads its state slice,
waits, triggers; out loads the gate and norm weight, waits, no trigger; masked segment CTAs wait before exiting),
TF_GLM_PDL_CHAIN gains `kda` (in `all`; `TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe` is the control on the same image).
Cold compile (touched files; spark5, zig 0.17.0, aarch64): test-glm-tp3-host test-glm-fp8-kv-host test-glm-exl3-host
test-cuda-host give the same failure set as the unpatched 102-patch tree on the same host (16-29 failing binaries
either way, identical FAIL lines - this host's baseline differs from the rtx box's three), plus the new Chain parse
test passing. Applies cleanly on the 102-patch base.

**Pre-registered gate (before the build):** image 45a1d17d74d7 on both Sparks; A restarted with TF_GLM_PROF=1 in the
order kda OFF (TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe), ON (unset), OFF, ON; each leg runs acc.py from spark5 (1-stream
prose and code, 4-stream prose x2) and the "glm prof" lines are read from rank 0's log. GO: hashes 64c908210a9e /
315341215112 in every leg; the kda section at "verify 2 rows" lower by >= 0.12 ms and at "shared 8 rows" or "shared
10 rows" (the same label in both legs of a pair) by >= 0.15 ms, ON vs OFF, in both pairs; ON-leg tok/s not below the
OFF legs by more than the OFF legs differ from each other. Then a final env-free leg (profile off) with acc.py and
bench2.py against the standing band (1-stream 38.4-39.2 / 55.6-56.8, 4-stream 77-83, prefill 1737-1780, mixed >=11.2
tok/s). SHIP-OFF: hashes hold but the section does not clear both thresholds in both pairs -> A back to cd1de29726c3,
0199 stays a candidate with `all` to be amended (kda off) before any later image. KILL: any hash change, CUDA error or
hang -> A back to cd1de29726c3, the patch marked wrong here. Predicted from the bench: -0.17 ms at verify 2 rows,
-0.20 to -0.24 at shared 8-10 rows.

## 0199 measured in the engine: exact, no resolvable gain; SHIP-OFF fired, KDA C1 closed (2026-11-04, 11:00-11:45 local)

Image 45a1d17d74d7 (103 patches) built on spark1 with pair B stopped (zig build 381 s), shipped by docker save/load
to both Sparks, label verified on each; B restored on its own image (UP, healthy). Pair A ran the four pre-registered
legs on that image (TF_GLM_PROF=1; client spark5; acc.py 2 reps + acc4only; the "glm prof" reports since each
restart averaged per label; docs/speed/kda-c1/legs.log, acc-*.log, prof-*.log):

| leg | TF_GLM_PDL_CHAIN | 1-stream prose / code tok/s | 4-stream totals | kda ms: verify 2 rows (6 reports) | verify 3 rows | shared 8 rows | shared 10 rows | hashes |
|---|---|---|---|---|---|---|---|---|
| off1 | reduce,hc,p2p,moe | 37.3-37.6 / 55.7-55.8 | 78.4 78.5 81.5 78.2 | 13.180 | 13.385 | 16.700 | 16.975 (2) | both |
| on1 | unset (all, with kda) | 37.8-37.9 / 55.9-56.0 | 79.8 80.9 79.6 78.3 | 13.052 | 13.220 | 16.670 | 17.110 (1) | both |
| off2 | reduce,hc,p2p,moe | 37.7-37.9 / 55.7-55.9 | 81.2 79.2 77.8 77.4 | 12.993 | 13.195 | 16.340 | 16.810 (2) | both |
| on2 | unset | 37.8-37.9 / 55.8-56.0 | 79.7 78.5 80.3 76.7 | 13.052 | 13.215 | 16.650 | 17.125 (2) | both |

ON minus OFF: verify 2 rows -0.128 then +0.059 ms; verify 3 rows -0.165 then +0.020; shared 8 rows -0.03 then +0.31;
shared 10 rows +0.14 then +0.32. The OFF legs differ from each other by 0.19 ms at verify 2 rows - the section's
run-to-run spread exceeds the standalone prediction (-0.17 ms), and the shared windows never dropped. Hashes
64c908210a9e / 315341215112 in all four legs; no CUDA error; throughput within noise either way. **SHIP-OFF by the
pre-registered rule**: 0199 is exact (bit-identical standalone on every output, hashes stable in the engine) but buys
nothing the engine can resolve; pair A back on cd1de29726c3 env-free (final leg below); the candidate amended so `kda`
is opt-in by name only (not in Chain.all; TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe,kda enables it), re-applied on the base,
cold-recompiled on spark5. Image 45a1d17d74d7 stays loaded on both Sparks, unused; spark1's recipe tree is back at the
102 shipped patches.

**What the legs also settle (Codex's open point):** at verify 2 rows the kda section is 13.0-13.2 ms with the chain
measured standalone at 1.13 ms over 34 layers, so the decode-shaped input projections cost ~0.35 ms a layer (the
0.308 figure was a prefill-trace number); there is no ~1.4 ms of hidden host gap for a graph form of C1 to recover.
**KDA C1 is closed with a measured answer:** the chain is ~1.1 ms of a 1-stream round and ~3.2 ms of a 4-stream round;
launch scheduling can return at most ~0.2 ms of it, below the engine's resolution; fusion loses. The handoff's +1-2 ms
was the chain's whole cost. What is left in the KDA block is the projections (~12 ms a round, ~0.35 ms a layer at
~50 MB a layer a rank = ~145 GB/s, about half the box's bandwidth) - a dense-linear question (0100's territory), not a
chain question.

**Final state (2026-11-04 11:45 local):** A restarted env-free on cd1de29726c3 (102 patches): 1-stream 37.9-38.3 /
56.4-56.7, 4-stream 77.8-79.9, hashes 64c908210a9e / 315341215112 (docs/speed/kda-c1/acc-final102.log; the 1-stream
prose is ~0.5-1 tok/s under this morning's 38.8 from the same client, within the day's drift seen in every leg). B UP on
its image. The amended 0199 (kda opt-in by name) cold-compiles on spark5 with the base tree's failure set, its Chain test
passing. Both pairs LEFT SERVING.

## Sampled decode (temperature 0.7): A holds, B collapses - the "prose is very slow on spark1+2" report explained (2026-11-04, 12:05 local)

User report: prose feels very slow on spark1+2. Every number in this file's A-vs-B comparison was GREEDY (temperature 0,
thinking off). Measured now from spark5 with docs/speed/harness/acc_sampled.py (acc.py's essay prompt, temperature 0.7,
top_p 0.95, thinking off, 600 tokens at 1 stream, 4 x 400 at 4 streams; docs/speed/kda-c1/acc-sampled-t07.log):

| shape | A: spark3+4 (cd1de29726c3) | B: spark1+2 (zig-v1.0.7, dense fp8) | greedy, for reference (A / B) |
|---|---|---|---|
| 1 stream prose, T=0.7 | 36.7, 38.9 tok/s; 54-55 ms/round; 1.98-2.16 tok/round | **21.3, 21.3 tok/s; 108.3 ms/round**; 2.31 tok/round | 38.8 / 40.2 |
| 4 streams prose, T=0.7 | 76.1, 75.4 total; 106-107 ms/round | **31.3, 30.3 total; 296-303 ms/round** | 78-82 / 89.1 |

A's sampled decode is within 1-2 tok/s of its greedy decode (the sampler costs it nothing visible). B's round doubles
at 1 stream (+53 ms) and triples at 4 streams (+200 ms) while its tokens per round stay at its greedy level: the
reference engine's sampling path adds ~50 ms a round per stream (host-side sampling or a per-row synchronous draw,
not measured here - B is frozen and read-only). B's two sampled reps also returned identical tokens/round and
ms/round to the decimal, which suggests a fixed seed or temperature not honoured; not pursued. **For any non-greedy
workload A is the faster pair by ~1.8x at 1 stream and ~2.4x at 4 streams**; the B advantage at 4 streams (89-95 vs 78-82)
exists only under greedy decoding. The comparison table in HANDOFF-2 is annotated accordingly.

## The rtx recipe's two patches checked for porting; pair A's prompt cache found dead and restored (2026-11-04, 12:00-13:00 local)

The sister recipe (2x RTX PRO 6000, same engine pin and base patches; reached via spark5 as zurih@rtx) has two engine
patches this pair lacks, both server/host-side: its 0096 (tool_parse: a call after prose that quotes a tool-call
opener is the call) and 0097 (GLM warm turns: after a reply, a background prompt-only pass of the next turn's rendering
so a user message after an agent loop resumes past the loop). PREFILL_SHARE=0.68 and HEAD=q8 are already here. Both
apply cleanly on the 102-patch base (together and alone); cold host suites on spark5 give the base tree's failure set;
the patches' own steps pass (test-tool-parse-host, test-tool-visibility-http-host, test-json-answer-host,
check-native-cuda-host). Ported as 0103/0104 (the parked PDL candidate renumbered 0105). Codex consult:
docs/speed/codex-cache-port.md (brief -brief.md).

**Codex's parser case, confirmed on this tree, fixed in the port:** with 0096 as written, a reply whose JSON call
quotes a whole call inside a string argument - `<tool_call>{"name":"lookup","arguments":{"q":"<tool_call>lookup
<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>"}}</tool_call>` - invoked lookup(n=7) (host test:
"calls=1 arguments={"n":7}", the cut outer call left as content); before the patch it was kept as content. Our 0103
adds insideValue: an opener inside a value still open in the text before it (unbalanced `{`, an open <arg_value>, an
open <parameter=) is never retried. The four cut shapes (JSON, GLM, XML, after prose) never yield the inner call; the
ported positive tests pass unchanged (42/42). **The rtx recipe's 0096 carries the bug as shipped there.**

**The replay showed cached=0 on EVERY turn of an agent loop on pair A** (the loop's own next step included, 6 s a turn
at 9.3K tokens), and rank 0's startup line said "no prompt cache" with "warning: GLM CUDA prompt cache budget 0
bytes cannot keep one state". Mechanism (core/kv_pool.zig plan: cache = min(want, left - held); held = the whole
1M-token windows the room would hold beyond the first): the room had grown from 18.2 GiB (yesterday, "4,098 pages +
4.5 GiB cache") to 19.8 GiB (the engine now counts 0.53 GiB of per-CPU page lists; less else on the box), crossing the
threshold for a THIRD window (room >= 6.23 + 2 x 6.64 = 19.5 GiB): 5580 pages, cache 0. A model of the planner with the
pair's constants reproduces 5569/0 and predicted reserve 9.5 -> 4096 pages (two 1M windows, the shipped pool), ~3 GiB
cache, 1.2 GiB below the threshold. Codex: agrees (the cliff is real, 9.5 the right immediate trade; --prompt-cache-gib
cannot help since held() is computed first; the durable fix is a minimum cache reservation in the planner; check the
startup line at every start; the cliff returns if the room grows another ~1.2 GiB).

**Measured after MEMORY_RESERVE_GIB 8 -> 9.5 (scripts/local.sh), same image cd1de29726c3** (docs/speed/cache/):
startup "prompt cache: 3.6 GiB of device pages", "4098 pages of 512 rows (12.47 GiB) ... of 18.9 GiB left after a 9.5
GiB reserve", no warning; MemAvailable spark3 9.6 -> 10.5 GiB, spark4 12.4 -> 13.2 at start. Replay: loop step cached
4096 of 9279 (was 0), next user message 4096 (the pre-0104 behaviour), the same message again 8192 (was 0); turn
latency 5.9-6.2 s -> 3.5 s then 1.2-1.3 s. Decode: 38.3-38.4 / 56.4-56.5 at 1 stream, 79.8-80.8 at 4, hashes
64c908210a9e / 315341215112; 6 streams 94.4-95.4, 8 streams 105.4-106.4 (the record's 91-94 / 105-107); bench2 single
38.3 / 56.5, prefill 16K 1703 (one rep, 2% under the 1737-1780 band), 64K 1755, conc2 57.4, conc4 78.1; mixed1 (a chat
beside a 50K prefill) 11.8 tok/s during, worst gap 2.68 s (record 11.2-11.7 / ~2.7 s); mixed3 6.8-9.7 tok/s a decoder;
prefill pair 35.0 / 22.2 s TTFT for two 30K prompts (17.4 s alone). MemAvailable after the whole run: spark3 9.9 GiB,
spark4 13.2 (the reserve-8 low-water was 8.3). **Every protected win intact; shipped: the reserve at 9.5 (recipe only,
scripts/local.sh is untracked).**

**Pre-registered for the 104-patch image 1fdae27149cc (0103 amended + 0104):** on a quiet engine the replay's next
user message (R3) cached >= 8192 of ~9.3K (now 4096) and rank 0's log shows "next turn: background prefill of N tokens
done" after an answer and "stepped aside" for a request within 0.3 s; hashes and the decode band unchanged (no numerics
touched); prefill 16K/64K and the mixed case within band; TTFT of back-to-back requests (acc.py's sequence) not worse;
greedy output of the same final prompt identical cold and after a completed warm pass. KILL: a hang or CUDA error on
either rank, a hash change, or a warm pass holding a foreground request past one chunk. SHIP-OFF: no cache gain at R3
-> 0104 out of patches/engine (0103 may stay: exact, tested).

## 0103 + 0104 live on pair A: the next user message after an agent loop resumes at 8192 (2026-11-04, 12:40-13:20 local)

Image 1fdae27149cc (104 patches: 0103 the amended parser fix, 0104 warm turns) built on spark1 with B stopped (zig
build 399 s), shipped to both Sparks, B restored and healthy; the two candidates promoted into spark3's patches/engine
only (pair B's own recipe on spark1 is a different lineage and untouched); A restarted env-free. Gate (docs/speed/cache/
port-leg.log, coldwarm-port.log, bench2-port.log):

| check | result |
|---|---|
| startup | 4098 pages, prompt cache 3.6 GiB, no warning (the reserve-9.5 plan) |
| replay R3, next user message after the loop (quiet engine) | **cached 8192 of 9330** (4096 before 0104, 0 before the reserve), 1.61 s (3.51 / 5.94) |
| replay R2 loop step / R4 / R5 | 4096 / 8192 / 8192 |
| rank 0 log | "next turn: background prefill of 9317 tokens done"; "stepped aside: other work came first" when the next request was already there |
| decode band, hashes (acc.py, back-to-back) | 38.4 / 56.6 at 1 stream, 78.9-81.3 at 4, 64c908210a9e / 315341215112 |
| warm passes during acc.py's back-to-back traffic | 17 of 18 stepped aside (the pass yields to the next request as designed) |
| back-to-back TTFT, 6 short requests, no pause | 0.145-0.154 s (bench2 on the previous image: 0.17-0.19) |
| greedy identity | the answer right after a completed warm pass and the answer from the kept state on the same prompt: identical. A cold control with the same prompt is not possible without a restart; my reworded-filler control answered differently because its prompt differs, not evidence either way |
| bench2 | single 38.4 / 56.5 (TTFT 0.17 / 0.19), prefill 16K 1707, 64K 1760 (the 16K figure was 1703 on the reserve leg too: ~2% under yesterday's 1737-1780, same client and prompt - today's drift, not the patches), conc2 57.0, conc4 79.0, mixed1 12.9 tok/s during the 50K prefill (worst gap 2.68 s), prefill pair 22.2 / 35.0 s |

Codex's refinements taken: the gate measures retention (cached counts), the yield under arrival (stepped-aside lines
and TTFT), and the band; not taken today (noted as follow-ups): an arrival DURING an active 4,096-row warm chunk
measured end to end (the engine's chunk gate yields at the next chunk boundary, ~2.3 s at 1,750 tok/s), two
conversations competing for the 3.6 GiB cache until eviction, and the long-context page-pressure case for 8 streams.
**Shipped: A serves image 1fdae27149cc, 104 patches, reserve 9.5, env-free.**

## Pair B on the Python engine (TensorFold v0.6.0) vs pair A's new recipe, sampled prose at T=0.7 (2026-11-04, 13:35-13:55 local)

At the user's request pair B (spark1+2) was brought up on its recipe's Python engine (`ENGINE=python ./start.sh`, image
tensorfold-glm53:v0.6.0, 4 streams, 1M window, fp8 KV, DFlash2) in place of the frozen Zig v1.0.7 reference, and the
sampled prose bench (acc_sampled.py: the essay prompt, temperature 0.7, top_p 0.95, thinking off, 600 tokens at 1
stream, 4 x 400 at 4 streams) run from spark5 against both pairs within the same half hour (docs/speed/python-vs/):

| shape, prose, T=0.7 | A: spark3+4, image 1fdae27149cc (104 patches) | B: spark1+2, Python engine v0.6.0 | B on Zig v1.0.7 (earlier today) |
|---|---|---|---|
| 1 stream | **39.1, 39.3 tok/s**; 55 ms/round; 2.15 tok/round | 29.6, 29.6; 77 ms/round; 2.29 tok/round | 21.3; 108 ms/round |
| 4 streams total | **77.6, 75.4**; 106 ms/round | 67.7, 71.0; 131-134 ms/round | 30-31; ~300 ms/round |

A is 1.32x the Python engine at 1 stream and 1.07-1.15x at 4 streams under sampling; the Python engine lands more
tokens a round (2.29 vs 2.15) but its round is 40% longer at 1 stream and 25% at 4. The Zig v1.0.7 reference remains
the slowest of the three when sampling. Greedy point on the Python engine below. B is LEFT on the Python engine (the
user's choice; `PATH=$HOME/.local/bin:$PATH DENSE=fp8 ENGINE=zig ./start.sh` from ~/newMod/GLM-5.3-Flash-EXL3-TensorFold
on spark1 restores the Zig reference).
Greedy on the Python engine (one rep): prose 30.5 tok/s (73 ms/round, 2.23 tok/round), code 42.0, 4 streams 73.8 -
so the Python engine pays almost nothing for sampling (30.5 -> 29.6) but is slower than A under greedy too (A today:
38.4 / 56.6 / 79-81). Its greedy hashes differ from A's (c40faa1b46fe / 8f59d62ea2c9: a different engine and dense
format), as expected.

## Sweep of the sibling repos' merged PRs and issues for compatibility and stability items (2026-11-04, 13:30-15:30 local)

At the user's request: every merged PR and every issue of MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold
(11 merged, 15 issues), Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold (7 merged, 42 issues) and
GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (90 merged, 110 issues), triaged against this recipe (native Zig engine at
f350cd3e = 1.1.9, 2x GB10, TP2). No banned account appears among the authors. Most Python-engine items (spill tier,
kept-state caps, Triton variants, nucleus union, DSML) have no counterpart here; the Zig-port items (zig/patches
0006-0016 on v1.0.7) are older than this engine pin. What was checked on this pair, and the outcome:

| item (source) | here | outcome |
|---|---|---|
| chat template renders quadratically, `ns.out = ns.out + ...` (DeepSeek #17/#18, TensorFold#1633) | our template has no such append | n/a |
| stop.sh feeds /health's count to bash arithmetic (Qwen #82) | the count is validated in Python (int or '?') | n/a |
| memory check parses `free -g` by locale (Qwen #37) | /proc/meminfo + parked pages | n/a |
| rails() misses a second cabled QSFP port (GLM #243) | the f0 ports here are on different subnets (10.0.33 vs 10.0.41), not cabled to each other | n/a; would matter if both ports were cabled |
| SSE keepalives on silent streams (DeepSeek PR #13, GLM #106) | the pin has keepalive vtables on all three APIs | present |
| fixed default sampling seed (Qwen #75) | a random request seed; two unseeded T=1 replies differed | n/a |
| penalties ignored, invalid sampling values accepted (Qwen #68) | presence/frequency implemented; `temperature: -1` accepted as greedy (200) | design choice, not adopted |
| any model id answered as the resident model (GLM #195) | `model: not-a-real-model` -> 200 | design choice, not adopted |
| Responses `include` (GLM #73), chunked bodies (GLM #67), 400 framing (GLM #74) | 200 / 200 / Content-Length 92 = body 92 | fine |
| switching reasoning_effort re-prefills (GLM #93) | high -> low on a 16.6K prompt: 16,384 cached | n/a |
| a control token quoted in a thinking turn ends it (Qwen #76) | reasoning 242 chars, content 1,104, finish=length | n/a |
| NV_ERR_NO_MEMORY in the kernel log (Qwen #33/PR #59, GLM #143) | 5 lines on spark3 in 3 days, all at restarts (Nov 3 19:06, 20:27 x2; Nov 4 09:48 x2), none on spark4, no Xid, no OOM-killer; PR #59's mitigation is TP=1-only | benign; watch |
| idle CPU of the engine (DeepSeek #2/#40) | tensorfold-native ~50% of one core on each rank at idle; GPU 50 C, 9-10 W | note; keep_warm loop suspected, not confirmed |
| overcommit posture (DeepSeek #36) | vm.overcommit_memory=0, ratio 50: CommitLimit 80.5 GB, Committed_AS 6-10 GB | fine for this engine |
| DHCP moves the RoCE GID under a running engine (GLM #196) | static addresses on the direct link | n/a |
| .env / local.sh in git (GLM #226) | both gitignored | n/a |
| **an untyped or union-typed object argument comes back as a JSON string (GLM #180)** | **reproduced**: `"payload":"{\"id\": 7, ...}"` for a description-only parameter (the union case already decoded on this pin) | **fixed: 0105** (below) |

**0105** (docs/speed/candidates/0105-server-an-untyped-tool-parameter-...patch): tool_params.decode takes a JSON object
or array for a schema that names no kinds (`{}`, description-only, unresolvable `$ref`), keeps everything else as text
(the maintainer's rule from #180; upstream keeps text - noted at the gate); the streamer holds an untyped key until the
value's first byte, buffers `{`/`[` values for the close and streams text as before; six witness cases updated. Host
tests pass (43 parser, 7 params; cold suites = base). Image of the 105 set: 8c4ca7bd8c39, built on spark1 with B
already stopped (the user stopped B after the Python-engine comparison), shipped to both Sparks, A restarted; the probe
that reproduced the bug must return `"payload":{"id":7,"tags":["alpha","beta"]}`; hashes and the decode band unchanged
(server-only change).

**0105 live (2026-11-04, 15:00-15:40 local):** image 8c4ca7bd8c39 (105 patches) built on spark1 (B stopped by the user,
no stop/start needed; zig build 355 s), shipped to both Sparks, 0105 promoted into patches/engine, A restarted env-free
(docs/speed/cache/leg-0105.log, probe-0105.log, acc-0105.log). The probe that reproduced #180 now returns
`"payload":{"id":7,"tags":["alpha","beta"]}`, and the same streamed is valid JSON with the payload an object; hashes
64c908210a9e / 315341215112; decode 38.6-38.7 / 56.7 at 1 stream, 78.7-80.4 at 4 (a first pass had one low rep each,
31.1 and 34.4, right after the restart, not seen again); the replay still caches 8192 at the next user message; startup
4098 pages, 3.6 GiB cache. A GLM-markup text argument arrives in one delta at the end as before (the streamer streams
only the XML `<parameter=` form; GLM's own form is parsed whole). **Port to the rtx recipe staged** as its 0098
(docs/speed/rtx-port-0098/: the patch, the CHANGELOG entry, an apply script that commits only those two paths there),
apply-checked and parser-tested on that recipe's exact stack (base + 0001-0093 + its 0096/0097: 79 tests pass) - not
applied: the rtx box answers ping but its sshd does not (connection timed out at 15:20 local).

## Upstream since the pin: what is worth carrying (2026-11-04, 15:45-16:15 local)

Upstream TensorFold main (85e53093b, 2026-11-03) is 1,059 commits past this recipe's pin f350cd3e (v1.2.0, v1.2.1 and
v1.2.2 were tagged in between). A rebase is out: 67 of the files our 105 patches touch changed upstream, and yesterday 77
of 97 patches conflicted with v1.2.1. Individual fixes carried as patches instead. From upstream's CHANGELOG and the
commits touching families/glm, server and core/lanes:

| upstream | relevance here | decision |
|---|---|---|
| ed39acc5a: a stream that finishes on its first token leaked its lane when its trace failed (caches held until restart) | stability | **carried as 0106** (applies cleanly, 91 lines; lane steps 73/73, cold suites = base) |
| abd5ac350 (#1975): GLM tool calls stream their arguments as the model writes them (today GLM-markup arguments arrive whole at the end; a 573-char text argument came in one delta) | agent-client UX | not now: 814 lines, conflicts with 0105's streamer changes; a manual port of 1-2 h if wanted |
| c4dc99e28 (#1396): one-call-mode streams carry the whole reply's content | correctness with `parallel_tool_calls: false` only | not now (conflicts; niche) |
| 4f7219a6e: GLM image prefixes reused by content key | vision prompt cache; images not checked on this pair | not now |
| dd326f4f2 (#1577): a prompt chunk's MTP pass runs its last row only (5-14% sooner cold prompts at TP=3) | our stack runs no MTP pass beside DFlash2 (0008) | n/a |
| #1623 DFlash2 taps at TP>1 | already carried as 0013 | n/a |
| TF_GLM_KDA_PAD (TP=3), TF_GLM_KV_HOST (96 GB cards), csf0_v1 NVFP4 checkpoints, --tp CLI, KV_SHARE idle line | other topologies, checkpoints or cosmetic | n/a |
| -Dgpu=<gpu> single-GPU stripped build (~69% smaller image), Flash Next O_DIRECT loader, TF_LOAD_PREFETCH (Gemma/Qwen) | build size; faster loads for other families (GLM #245 asks for the loader port) | not in this pin's build; later |

**0106 built, measured, KILLED (2026-11-04, 15:30-16:00 local):** image fd1647ce8d0f (106 patches) on both Sparks, A
restarted on it (docs/speed/cache/leg-0106.log). Hashes and the replay fine, startup fine, but after twenty one-token
replies /health showed connections 2, waiting 1 with no client, and the decode band collapsed: 28.2 / 22.2 prose,
36.3 / 36.9 code at 1 stream, 50.9 / 65.7 at 4 (rounds 83-166 ms against 54-105). Reverted to 8c4ca7bd8c39: 38.4 / 56.5
/ 80.6-81.8. Attribution: the same twenty one-token replies then acc.py on the 105-patch image give 38.3 / 56.6 /
77.5-80.2, so the regression is 0106 on this stack, not the warm passes those replies start. Reading: a stream whose
lane upstream now releases deferred around its trace is left decoding on this stack's lane host (0093 frees grammars
on streams that cannot start; 0104 added the fill-queue yield checks), so every round carries a ghost stream. Not
pursued further: the leak it fixes needs a failed trace, which the pair has never logged. 0106 stays in candidates as
a record; the recipe is back at 105 patches; spark1's build tree likewise.

## TP=4 on the four-Spark ring: the launcher, the first two boots, Codex's read, 0106 pre-registered (2026-11-04, 16:00-19:30 local)

**The ask.** Pair A stopped (16:17 local; stop.sh archived its logs), the recipe cloned to spark1 as
`~/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig` (mia's login), and a `start-tp4.sh` that serves on
spark1 (rank 0) with spark2, spark3, spark4 as ranks 1-3, every rank reading the weights from spark1. The kit is cabled as
a ring without a switch (spark1-spark2 10.0.22.x, spark2-spark3 10.0.33.x, spark3-spark4 10.0.34.x, spark4-spark1
10.0.41.x; a port and its PCIe twin at .N/.10N; 10.0.0.N aliases on lo routed over the cables; the LAN on enP7s7).
The "ideas" source: the lab's own TensorFold recipe on spark1 (`~/newMod/GLM-5.3-Flash-EXL3-TensorFold`,
start-tp4-switchless.sh + scripts/nodes.sh), which ran TP=4 on this ring with v1.0.7 and its own ring patches.

**What was built** (commit e7ae4ef, then 349b9f7): `scripts/tp4.sh` (ring discovery from each Spark's ConnectX-7
addresses, NCCL graph files per rank with channels on the devices toward the previous rank, NCCL_ALGO=Ring,
NCCL_CROSS_NIC=1, NCCL_IB_SUBNET_AWARE_ROUTING=1, NCCL_IB_SUBNET_PREFIX_LEN=26, no NIC fusion / P2P / SHM, image shipping
to the three workers, NFS volumes per worker from the head's address on that worker's route, a memory guard over four),
`start-tp4.sh` (start.sh's flow for four ranks; TF_GLM_PREFILL_REDUCE=gather, TF_ROCE=0, NCCL_COMM_ID on what spark1's
name resolves to, 10.0.0.1) and `stop-tp4.sh`. config.sh gains the four-Spark settings; prepare.sh validates them only
under start-tp4.sh (TP=2 unchanged). Weights: spark1's Hub snapshot 078455ff is byte-identical to the pin 76c0b517 in
every shard and config file (checked against spark3's served copy; the commits between them touch the model card and
licenses), so a flat hard-linked dir `~/.cache/huggingface/glm53-1104/{model,dflash2}` inside the directory spark1's
ablit-nfs container already exports as `/huggingface` serves rank 0 locally and the workers over NFS
(NFS_MODEL_PATH=/huggingface/glm53-1104/model; spark3, across the ring, mounts 10.0.0.1 through spark2). Image
d3acd712b31c shipped to spark2 (spark3/4 had it). DRY_RUN on spark1 found the ring in rank order, GID 3 on every device,
prefix 26, and the right devices per direction.

**Boot 1 (16:34 local, 105 patches):** all four ranks up to NCCL; rank 2 (spark3) `ncclCommInitRank: unhandled system
error`, rank 0 `rank 2 stopped (ncclCommInitRank)`, 91 s in. **Boot 2 with NCCL_DEBUG=INFO:** the rings are right
(`Ring 00 : 3 -> 0 -> 1`, `1 -> 2 -> 3`; subnet-aware routing moves each send onto the next rank's /26), but NCCL's
tree for four ranks is `Tree 0 : -1 -> 0 -> 2`, and the engine sets NCCL_RUNTIME_CONNECT=0 (cuda_native.zig:205, so a
full GB10 never registers a channel late), which connects every algorithm's channels at init regardless of
NCCL_ALGO: `ibv_modify_qp failed with 110 Connection timed out ... local GID ::ffff:10.0.41.1, remote GID
::ffff:10.0.33.3` (rank 0's port toward spark4 trying to reach spark3), 35 retries, then the error.

**Codex (gpt-6.1, xhigh, read-only over the engine tree + the 105 patches; docs/speed/codex-tp4.md):** it named the
same two blockers and one I had not: (1) `warmPeers` (cuda_native.zig:138) posts a 4 MiB send+recv to EVERY rank
before the weights load, so even with the trees out of the way the two ranks across the ring would be connected; its
least-risk shape is mode-aware (split: every peer; ring: send next / receive previous; gather: no p2p, the rings connect
at the first collective), not "warm and ignore failures"; (2) the eager connect above - NCCL 2.30.7's eager branch
connects rings, trees and PAT, so NCCL_RUNTIME_CONNECT=1 with NCCL_CUMEM_ENABLE=1 (host cuMem buffers still off) is the
experiment; (3) the rest of the GLM family and the 105 patches hold no other all-pairs p2p at world 4 (0034's p2p and
0035/0063's copy-engine exchange are guarded out; 0051's "pair" is two prompt chunks; "TP3" in 0033/0063/0065 are test
fixtures covering worlds 1-8); DFlash2 splits its MLP only at world 2 (full MLP per rank at four, more replicated
work); (4) broadcasts from non-zero roots are fine under Ring; subnet-aware routing applies to p2p too but "retains the
default device" when no subnet matches, so it never repairs a cross-ring choice; (5) the lo alias 10.0.0.1 works as
NCCL_COMM_ID (the explicit-COMM_ID matcher does not exclude lo), the LAN address would be the simpler choice;
tp_watch wants a literal IPv4; (6) memory: 6,380 KV bytes a token a rank at any TP, one 1M window 2,049 pages = 6.23
GiB, eight windows + seven fixed parts ~51.0 GiB of the limiting rank's room; with ~55 GiB more room than TP=2 expect
eight windows and the full 16 GiB default prompt cache; there is no ninth-window cliff at --parallel 8; (7) TP=2's
token_sha is not a TP=4 acceptance gate (different partitioning, different fp grouping) - use a TP=4 greedy, draft-off,
cache-cold baseline and require repeatability, draft on/off equality, solo/concurrent equality.

**Decision:** both of Codex's fixes, as it shaped them. Launcher (349b9f7): NCCL_RUNTIME_CONNECT=1,
NCCL_CUMEM_ENABLE=1, NCCL_CUMEM_HOST_ENABLE=0 on every rank. Engine candidate **0106**
(docs/speed/candidates/0106-glm-cuda-the-start-s-peer-warm-up-follows-the-prefil.patch; the killed lane fix that
carried the number 0106 earlier today stays in candidates as a record and never entered the recipe): warmPeers takes
the rank Settings; `!set.split` returns before any posting; ring posts send(next) and recv(prev) only; split is
unchanged. Built on spark1 as image **daa6f2cde063** (106 patches, ~7 min); cold host tests on spark5
(docs/speed/tp4/hosttest-0106.log).

**Pre-registered gates for boot 3 (image daa6f2cde063, gather, reserve 12, PARALLEL=8, CONTEXT=auto):**
- GO-1 (boot): all four ranks reach the API and the smoke passes; the four NCCL logs show no `ibv_modify_qp` retry
  and no QP toward a GID off the rank's two cables; rank 0's startup line names 8 streams, the pages and a prompt
  cache above 0 with no "cannot keep one state".
- GO-2 (quality, from spark5's harness): acc.py greedy run twice gives the same token_sha each time; `draft: false`
  on the same prompts gives the same tokens; the 4-stream contract hashes equal the 1-stream ones.
- KILL: a rank that fails to start, any cross-ring QP, a smoke answer that is garbage, or a token_sha that differs
  between two greedy runs. Speed is recorded, not gated (no TP=4 band exists yet; TP=2 is 38.4 / 56.6 at one stream).

**Boots 3 and 4 (image daa6f2cde063, 17:00-18:15 local):** past NCCL init (no cross-ring QP: the trees are never
connected under NCCL_RUNTIME_CONNECT=1), rank 0 hashed the checkpoint in 44.7 s, but all three workers stopped three
seconds in with `thread N panic: attempt to use null value` and no trace (the panic handler hung). Bisected on the
spark3+spark4 pair with the same image: NFS weights alone fine (spark4 read the 163.6 GiB from spark1 in 41.9 s),
gather + TF_ROCE=0 fine, the ring/routing NCCL knobs fine, **NCCL_RUNTIME_CONNECT=1 + NCCL_CUMEM_ENABLE=1 reproduce
it on the follower**. gdb inside that container (every thread's backtrace, docs/speed/tp4/gdb-follower.txt): the
follower's main thread is in Zig's standard library, `Io.Threaded.Environ.scan` under `lockStderr` from the first
`std.log` after NCCL init (`p2p_report.logSink`), unwrapping a null entry of the process environment. The engine's
follower had just called `unsetenv("NCCL_COMM_ID")` (cuda_native.zig:210, when NCCL_SOCKET_IFNAME is set), which
shortens libc's environ under the std library's fixed-length view of it; with eager connect the first stderr lock
happened earlier and never saw it. Launcher fix (b9e87c2), no rebuild: NCCL_COMM_ID on spark1's LAN address
(192.168.1.143, what the default route leaves with; Codex's choice too) and NCCL_SOCKET_IFNAME to rank 0 only, so a
follower keeps NCCL_COMM_ID and NCCL picks its bootstrap interface by that subnet. The engine-side bug (an unsetenv
after start is unsafe under Zig 0.17's std) is noted for upstream, not patched here. The lab's DeepSeek 4x recipe,
checked on the user's pointer, runs the same ring design (graph files, Ring-only, /26, NFS from spark1) on a different
engine family whose own rendezvous never unsets NCCL_COMM_ID.

**Boot 5 (image 74e19a63a9a5 = the same 106 patches relabelled after patch 0078's credit text changed on main at 19:07; the launcher built it on spark1 and shipped it to the three workers itself; LAN bootstrap, 19:22 local): GO-1 met.** All four ranks up, loaded in
125.6 s (the workers' digests over NFS 40-45 s each), smoke "ready". Every rank's NCCL report names exactly its two
neighbours (`rank 0 NET/IB (4 connections); rank 2 NET/IB (4 connections)` on rank 1, and so round the ring): nothing
crosses the ring. Rank 0's startup line: model 45.61 GiB a rank; 8 streams, 16,392 pages of 512 rows (49.87 GiB, eight
1M windows, as Codex computed) of 55.8 GiB left after the 12 GiB reserve, 331 MiB fixed a stream, a **3.6 GiB prompt
cache** a rank, no "cannot keep one state". KV pool 8,392,704 tokens shared by 8 streams, one request up to
1,048,576. Available memory with the server idle: spark1 12.4 GiB, spark2 14.5, spark3 12.2, spark4 14.5 (the
memguard floor is 2.5). Settings on spark1 (scripts/local.sh, untracked): PARALLEL=8, CONTEXT=auto (1,048,576 fit
first try), KV=fp8, DENSE=q8, HEAD=q8, ABSORB_MMA=1, HOST_CACHE_GIB=0, MEMORY_RESERVE_GIB=12, WORKER_WEIGHTS=nfs,
NFS_MODEL_PATH=/huggingface/glm53-1104/model, MODEL_REVISION=078455ff (byte-identical to the pin), PORT=8888.

**GO-2 met (19:30-20:05 local, spark5's harness against spark1:8888; logs in docs/speed/tp4/):** two acc.py runs give
the same token_sha each time (prose `b6ae240144bc`, code `9b68d8a77151`; not the pair's hashes, as pre-registered: a
different partition); `draft: false` gives the same 600 prose tokens and hash as drafts on; the 4-stream contract
runs returned whole. The TP=4 band, greedy, against pair A's on the same harness:

| | pair A (TP=2, 105 patches) | four Sparks (TP=4, 106 patches) |
|---|---|---|
| 1 stream prose / code (tok/s) | 38.4 / 56.6 | 56.0-56.2 / 84.5-85.3 (ms/round 36.6 / 44.3, accepted 57% / 74%) |
| 4 streams prose, total | 79-81 | 115-120 |
| 6 / 8 streams prose, total | 94-95 / 105-106 | 139-141 / 163-166 |
| prefill 16K / 64K (tok/s) | ~1,600-1,800 | 1,752 / 1,889 (TTFT 9.1 s / 33.8 s) |
| conc2 / conc4 total | - / 79-81 | 86.6 / 118.4 |
| mixed1: decoder before / during / after a 50K prefill | - | 63.6 / 19.8 / 58.4 tok/s (gap p95 53 ms, max 2.6 s) |
| prefill pair (30K + 30K) | - | 16.4 s alone; 16.4 s and 33.8 s as a pair |

Decode +46% at one stream, +48% at four, +55% at eight; prefill about the pair's (the ring's exchanges go round four
hops). Available memory after the benches: spark1 12.2 GiB, spark2 14.0, spark3 11.9, spark4 14.2. Not measured yet:
PREFILL_REDUCE=ring (the split prefill's rows passed along the ring; the engine's own neighbours-only mode) against
gather, sampled decode at T=0.7, and a long soak. 0106 is in patches/engine; the served image is 74e19a63a9a5 on all four Sparks (daa6f2cde063 is the same code under the earlier label).

## TP=4 on a switch: FABRIC=mesh, five boots, the ring's hashes reproduced (2026-11-04, 14:20-16:00 CST, another kit)

**The kit.** Four DGX Sparks (spark-6f90 rank 0, spark-de02, spark-8fe0, spark-8458), each with one 200G cable to a
MikroTik CRS804-4DDQ; a port and its PCIe twin addressed on 192.168.120.0/24 and 192.168.121.0/24, MTU 9000, RoCE v2 GID 3
on every device; every pair of Sparks shares both subnets, so `detect_ring` as shipped stops with "Every pair of these four
Sparks shares a RoCE subnet (a switch)". Weights: the Ablit checkpoint `57edefd2` (and, for the hash check, the published
`078455ff`, byte-identical to the pin as recorded above) served from flat hard-linked copies of the Hub blobs on every
Spark (`WORKER_WEIGHTS=check`; a snapshot directory is symlinks into `../../blobs`, which dangle inside the container:
boot 1 stopped at `cannot read /model/tokenizer.json`). Image `74e19a63a9a5` built on rank 0 (12 min) and shipped.
`PARALLEL=8`, `CONTEXT=auto` (1,048,576 fit first try), `KV=fp8`, `DENSE=q8`, `HEAD=q8`, reserve 12 GiB, thinking off.

**What was changed** (`FABRIC=auto|ring|mesh`): `detect_ring` calls a fabric with no missing pair a mesh (an explicit
`FABRIC=ring` or `mesh` that disagrees fails); on a mesh `ring_graphs` writes nothing and `tp4_rank_env` sets
`NCCL_CROSS_NIC=1`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_SUBNET_PREFIX_LEN` from `pair_links` (24 here),
`NCCL_IB_MERGE_NICS=0`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, the channel count, and nothing else: no graph file, no
`NCCL_ALGO` (`MESH_NCCL_ALGO` pins one), no `NCCL_RUNTIME_CONNECT` (eager connect, the engine's default; every edge is
reachable). `NCCL_SOCKET_IFNAME` stays on rank 0 only. `PREFILL_REDUCE=auto` resolves to `split` on a mesh and `gather` on
a ring; the launcher reads the fabric before the banner and keys the context fit on it. A ring runs exactly as before.

**Pre-registered gates** (an advisor's review of the patch before the first boot): GO-1 boot: NET/IB to every peer on
every rank, no `ibv_modify_qp` retry, no NET/Socket channel, no queue pair between the two subnets, rank 0's line with 8
streams and a prompt cache; GO-2 correctness: greedy `token_sha` repeats, drafts on == off, 4 streams == 1, `split` ==
`gather`, and on the published checkpoint the ring's hashes above (`b6ae240144bc`, `9b68d8a77151`). KILL: a cross-subnet
queue pair, a rank that exits, a hash that differs between greedy runs. Speed recorded, not gated.

**Boots (all with the patch; `./start-tp4.sh restart`):**

| Boot | Settings | Result |
|---|---|---|
| 2 | Ablit, split, NCCL algo unset | LIVE in 121.8 s (digest 44.7 s); `p2p: NCCL transport: rank N NET/IB (12 connections)` to every peer on every rank; 8 streams, 16,392 pages, **4.4 GiB** prompt cache; smoke `ready`. acc.py: prose 53.8 (`fab1d3b6894b`), code 83.0 (`e71492bf766b`), 4 streams 117.6-122.4, both reps equal |
| 3 | gather, `NCCL_DEBUG=INFO,SUBSYS=INIT,NET,GRAPH,ENV` | LIVE 77.2 s (warm). Every rank `Using [0]rocep1s0f0:1/RoCE [1]roceP2p1s0f0:1/RoCE`, two virtual devices of one physical each, 12 QP lines all GID 3, 0 `modify_qp`, 0 NET/Socket, 0 QPs 120↔121, sends 6/6 over NET/IB/0 and /1, rings and trees connected (the trees span non-neighbours, as a mesh allows). **GO-1 met.** acc.py: same hashes, 53.5 / 82.6. Prompt cache 3.8 GiB |
| 4 | split, `MESH_NCCL_ALGO=Ring` | same hashes, 53.6 / 82.5, 4 streams 120.6-124.2: no difference, `NCCL_ALGO` stays unset |
| 5 | `DRAFTER=none` | same hashes; 44.0 tok/s serial (22.7 ms a round) |
| 6 | **published `078455ff`, gather** | **`b6ae240144bc` / `9b68d8a77151`, the ring's hashes**; 55.6 / 83.2 tok/s (ring: 56.0-56.2 / 84.5-85.3), 4 streams 119.9 |

**GO-2 met.** The fabric does not change the arithmetic: gathers are exact and the hashes are the ring's.

**Speed (greedy; the lab's harness and the site's own battery, client on a desktop over 1 GbE):**

| | ring (above) | switch, split | switch, gather |
|---|---:|---:|---:|
| 1 stream prose / code | 56.0-56.2 / 84.5-85.3 | 55.6 / 83.2 (published), 53.8 / 83.0 (Ablit) | 53.5 / 82.6 |
| 4 streams prose total | 115-120 | 117.6-124.2 | 119.2-121.3 |
| 8 streams prose total | 163-166 | 162.7 (site battery, T=0) | |
| prefill 6K / 25K / 51K / 100K tok/s | 1,752 (16K) / 1,889 (64K) | **2,070 / 2,421 / 2,462 / 2,457** | 1,498 / 1,865 / 1,878 / 1,882 |
| cold ~200K prefill beside 3 decoders | | 100.8 s; worst pause 2.02 s, p99 1.86 s | |
| 4 prompts concurrent == solo at T=0 | | 4/4 | |

Decode equals the ring's; `split`, which a ring cannot carry, prefills ~30% faster than `gather`. The 2 s decode pause
during a cold prefill is the 4,096-row prompt chunk (the ring's mixed1 showed 2.6 s). With drafts off, four streams
decoded at 91 ms a round against 22.7 ms for one (no batching gain without drafts; noted, not pursued).

**More A/Bs (boots 7-10, Ablit, split):** `NCCL_CHANNELS=8` changed nothing (52.5 / 81.3 at one stream; prefill 2,390 /
2,424 / 2,416 against 2,421 / 2,462 / 2,457 at 4); `MESH_NCCL_ALGO=Ring` nothing (above). Decode at world 4 is the engine's
36-37 ms round, not an NCCL setting.

**Prompt cache and retention (not switch-specific; worth knowing at TP=4).** With `PARALLEL=8` and the 1M window the page
pool took 16,392 pages and the prompt cache got 4.4 GiB a rank (~0.7M tokens): three ~275k-token conversations filled and
each resent with a short new turn retained **0 of 3** (every resend re-prefilled, 117 s). `CONTEXT=786432` leaves 12,296
pages (6.3M tokens, one request up to 768k) and the cache takes its 16 GiB default: **3 of 3** retained, resends answered in
0.5 s, decode unchanged. For agent sessions of several hundred thousand tokens the smaller window is the better trade on
GB10; the README's "the page pool takes whole windows before the prompt cache" applies at four as at two.

**Open:** a four-rank RoCE all-gather (roce.zig is two-rank) would take the decode-sized exchanges off NCCL on a mesh;
`NCCL_CROSS_NIC=0` (rail-only) needs a launcher knob to test (the mesh branch sets 1); `DENSE=q4` was not tried (rejected
on quality above).
