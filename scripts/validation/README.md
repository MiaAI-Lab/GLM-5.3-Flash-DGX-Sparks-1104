# Offline inference validation tools

A dependency-free Python 3.10+ for reviewing inference optimization experiments. It separates full-call kernel screening, recorded numerical checks, independent service measurements, and instrumented cost attribution. It does not turn a positive microbenchmark into an end-to-end acceptance claim.

## Run the complete historical case

From this repository root:

```sh
python scripts/validation/validate.py analyze scripts/validation/examples/q8-scale-registers --output q8-review.json
python -m unittest discover -s scripts/validation -p test_validate.py -v
python scripts/validation/validate.py attribute scripts/validation/examples/intervals.csv --output interval-review.json
```

Choose a new output filename for each analysis. The analyzer refuses existing outputs. It performs no network calls, CUDA launches, remote service changes or deployment actions. The bundled example needs neither a checkpoint nor DGX hardware.

Expected q8 result: 504 timings, 28 cells and four rank/cache aggregates; local screening positive, C1 directional confirmation false, C4 directional confirmation true, **no end-to-end gain claim**. See [the case study](examples/q8-scale-registers/CASE_STUDY.md).

## Input contract

`plan.json`: unique, nonempty `ranks`, `rows`, `cache_modes`; integer `repeats` >= 2. The plan should be frozen before collection. Every Cartesian cell must contain every A0/B/A1 repeat exactly once. The tool detects missing, unexpected, duplicate, nonfinite and nonpositive measurements. It cannot detect an omitted experiment or a plan rewritten after observing results.

`timings.csv`: `rank,rows,cache,leg,repeat,ms`. ms is the **complete measured call**, including preparation and consumers required by the candidate. A0/B/A1 are baseline/candidate/baseline; repeat is zero-based. For each rank/cache, sum the row-wise medians, with equal row weights. Do not sum ranks or interpret this screening mix as production traffic. No outlier trimming, retry selection or baseline pooling is performed.

`checks.json`: recorded numeric, resource and dependency-control records. Numeric records supply `worst_tol` (error divided by the original tolerance), `negative_tol` (wrong-result negative control), `full_outputs`, and `bitwise_repeats`. Dependency controls supply `delayed_producer`, `zero_input`, `nonfinite`, `restored`. The tool validates reported outcomes; it does not independently recompute numerical references. Coverage, aliasing, layout, resource limits and sanitizer evidence still require their original harness and review.

`service.json`: scenario -> leg -> throughput samples. A0/A1 and one or more B legs are required, with equal repeat counts per scenario. All values must be finite and positive. Report medians, each candidate against **both** controls, and baseline drift. An all-candidates-above-both flag is descriptive only; even true does not establish statistical or deployment acceptance. The tool intentionally never sets `end_to_end_gain_confirmed` to true. Declare any acceptance contract separately before collection.

`attribute` accepts CSV columns `rank,request,round,actual_tokens,category,start_ms,end_ms`. Times share one clock and use milliseconds. Use actual committed output tokens, not accepted+1. Within a category, overlapping intervals are merged before division. Categories and ranks remain separate and may overlap. The bundled interval data is **synthetic**, not historical performance evidence. Input events must already have been attributed to the correct request and round; this tool is not a CUPTI/Nsight collector and does not prove trace completeness.

## Re-export from the original experiment

```sh
python scripts/validation/export_q8_case.py /path/to/experiment-root /path/to/new-case
```

The adapter reads the two original screen logs and the independent confirmation analysis. It validates rank/variant identity, copies all timing repeats, extracts recorded checks, and records SHA256 source identities. It excludes deployment and host records; it does not include model weights or captured prompts. Its source layout is specific to this historical experiment; the analyzer itself is source-independent.

## Scope

This is an **offline analysis and evidence-format contribution**, with one complete historical analysis case. It is not a portable GPU benchmark or a fresh hardware reproduction. The original CUDA harness depends on captured checkpoint inputs, fatbins, the pinned patched engine and standard registry; those are deliberately not bundled as if they were freely redistributable fixtures. A GPU replay contribution requires a separately reviewed capture/export contract and license/provenance review.

No kernel patch or default serving setting is proposed. Source code is covered by the repository Apache-2.0 license. Do not publish captured inputs merely because their aggregate measurements are included here.
