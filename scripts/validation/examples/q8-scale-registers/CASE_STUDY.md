# Q8 scale register queue: positive local screen, unconfirmed serving gain

## Mechanism and experiment

The candidate preloads original FP16 scales into a warp-distributed register queue, removing shared-memory scale transfers/reads. The original q8 codes, layout, input/weight pipeline and MMA/FP32 order are retained. Registers rise from 58 to 66; the recorded residency remains two blocks. This does not reduce global logical weight bytes.

Historical screen: two physical GB10 ranks, one captured N12576/K4096 matrix per rank, seven captured M2..8 inputs, two cache controls, three A0/B/A1 legs and six repeats = **504 complete-call measurements**. No-forced-scrub is not proof of production warm cache; 128 MiB scrubbing does not establish production cache coverage. Equal-M weighting is a screening mix, not a production estimate.

| Rank | Cache control | Cost reduction vs A0 | vs A1 |
|---:|---|---:|---:|
| 0 | no forced scrub | 2.756% | 2.540% |
| 0 | scrub 128 MiB | 2.133% | 2.215% |
| 1 | no forced scrub | 1.977% | 1.908% |
| 1 | scrub 128 MiB | 1.897% | 2.207% |

Each percentage compares sums of seven row medians. All 28 cells and all repeats are included in the analyzer output. These are local latency reductions, not whole-model throughput improvements.

## Evidence stages

- Bundled recorded results: 126 numeric records and seven dependency-control sets per rank, both resource records per rank. They include full-output/bitwise checks and independent wrong-code controls; the Python tool checks recorded outcomes, not CUDA arithmetic.
- Historical standard qualification, **not rerun here**: selected FP32/BF16 products, M1..16/N12576/K4096 plus auxiliary N64, both ranks plain/memcheck/synccheck; four sanitizer reports recorded zero errors without suppression. This is not all-repository CI.
- Historical native integration and greedy tests: both ranks hit the candidate; tested text/token/counters matched. A first startup was refused because prompt cache was 3.3 GiB instead of the 3.4 GiB requirement; the failure was retained. Full 1M-token input capability was not tested.
- Independent service window: original sparkDash C1 ABBA/C4 ABA, three jobs per leg, 400 output tokens, temperature 0.7, thinking on, top_p=1, top_k=20. All 21 job throughput values are included. Serving results are separate from greedy and local screening.

## Independent serving result

| Scenario/leg | Median aggregate decode tok/s |
|---|---:|
| C1 A0 | 51.25 |
| C1 B1 | 47.53 |
| C1 B2 | 48.88 |
| C1 A1 | 46.98 |
| C4 A0 | 87.61 |
| C4 B | 88.27 |
| C4 A1 | 87.48 |

C1 B1/B2 are below A0 and above A1; baseline drift is -8.33%. C4 is about +0.75%/+0.90% against its two controls. The C4 signal does not override the unconfirmed C1 result. Do not attribute all drift to hardware or the kernel: sampled outputs and verification work differ.

**Final historical decision: NOT_PROMOTED_NO_INDEPENDENT_PRIMARY_GAIN.** The candidate was not adopted; the original service was restored in the experiment. This package does not inspect or attest current service state. No failed candidate is restarted and no extra measurement window is selected.

## Reproduction boundary

Run the README command to independently recompute every bundled local and service comparison on CPU. `provenance.json` records the original source hashes. The original source logs remain local. This is a reproducible analysis of extracted historical measurements, not a fresh GPU measurement or independently verified standard-qualification run. No prompts, captured weights or host credentials are included.
