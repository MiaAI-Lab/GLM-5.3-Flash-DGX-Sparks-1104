Your plan is viable for **block-FP8 weights with BF16 activations**, with three material corrections: admission must change before `config.parse`, dense MLP execution needs a new FP8 arm, and this tree **does not load layer 45 beside DFlash2**. I would reuse the existing FP8 kernels initially. Their performance and model fidelity on this checkpoint remain unmeasured.

This is a static review; I changed nothing and ran no device tests. References below point into the supplied snapshot.

A. The loader and forward plan is mostly sound.

Setting `c.fp8` in `config.zig` alone will not admit the checkpoint. `Engine.init` calls EXL3/ModelOpt admission **before** parsing the family config. Add a family-specific block-FP8 admission branch there, with named refusals for unsupported `fmt`, block geometry, and activation scheme. Keep ModelOpt’s existing interpretation intact. I would call the flag `block_fp8` to distinguish stored weights from FP8 KV. See [cuda_engine.zig:142](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:142).

There are also two separate scale-name consumers to change. `Loader.formatOne` probes only `.weight_scale`, and `Fp8pb.read` independently reads that same name. Updating only the reader still fails format resolution. Use one resolver for both, selected by the admitted checkpoint convention; reject ambiguous dual names rather than silently choosing one. ModelOpt uses `.weight_scale` here. Its existing FP8 reader copies the FP32 scales without taking reciprocals, so your multiply interpretation matches the existing packing machinery. See [cuda_weights.zig:121](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:121) and [cuda_fp8pb.zig:16](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_fp8pb.zig:16).

Add a distinct `formats.Declared` case permitting BF16 and block-FP8 under this admission. Today a missing ModelOpt recipe table means `.nvfp4`, which rejects block-FP8 even when its tensor metadata is valid. The existing tensor classifier already recognizes E4M3 plus FP32 block scales, including the two-dimensional shape you describe. Preserve those checks, then additionally check each expert against its expected GLM dimensions before slicing. See [cuda_formats.zig:25](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_formats.zig:25) and [cuda_formats.zig:65](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_formats.zig:65).

Your routed-expert cuts and scale cuts are correct. At TP=4, gate/up scales become `[4,32]` per projection and down scales `[32,4]`. Store gate/up scales as `[E][2][4][32]`, while interleaving their **codes** through `packGateUp`. These are different layouts. Do not use the lane projection’s expanded scale packing for routed experts. `Fp8Experts` expects the compact block-scale tables documented in [experts.zig:24](/home/zurih/c1bench/tp4/tf/zig/src/cuda/experts.zig:24).

Use 128-unit MoE sharding for this mode and validate alignment before calling the packers: their geometry checks are assertions. TP=4 is clean; this also prevents an unsupported uneven cut from silently truncating scale blocks. Shared projections can retain their existing 64-unit cuts because their packer explicitly resolves scales from the original row/input origins, including cuts inside a 128 block. See [cuda_shard.zig:38](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_shard.zig:38), [experts.zig:305](/home/zurih/c1bench/tp4/tf/zig/src/cuda/experts.zig:305), and [cuda_fp8pb.zig:122](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_fp8pb.zig:122).

The proposed optional `Fp8Experts` member and `Grouped.fp8g` dispatch are sufficient for routed execution. Pass the mode through `Modules.load`, initialize/unload the optional member on every success/error path, and update recording fixtures that construct these structs. Currently module loading has no config argument and loads the NVFP4 and EXL3 modules unconditionally. See [cuda_forward.zig:60](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_forward.zig:60).

**The non-EXL3 prompt path remains intact.** `moe.body` creates a plan with `count + 1` experts and `top_k + 1` slots, routes it, then invokes gate/up and down with `skip = count`. Prompt down produces BF16 slots; decode down produces FP32 slots. `fill` already selects the matching byte width whenever `l.exl3 == null`, which includes your proposed FP8 layer. Keep that relationship intact. See [cuda_moe.zig:113](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_moe.zig:113) and [cuda_moe.zig:124](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_moe.zig:124).

The router and bias require no format-specific change. The loader preserves the router as BF16 and widens the correction bias to FP32. Selection uses `sigmoid(logit) + bias`, while routed weights use the selected **unbiased sigmoid scores**, optional normalization, and routed scaling. The shared slot gets ID `count` and weight one. See [cuda_weights.zig:438](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:438) and [glm_glue_moe.cuh:74](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/glm_glue_moe.cuh:74).

The FP8 kernels check `skip` before computing the expert’s weight address. Thus the shared sentinel does not require a 289th weight entry. They leave its output rows untouched; `sharedRows` subsequently fills that slot. Test this explicitly with poisoned buffers. See [fp8_experts.cu:113](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/fp8_experts.cu:113) and [fp8_experts.cu:159](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/fp8_experts.cu:159).

For projections, the situation is narrower than step 4 suggests:

- DSA `q_a`, `kv_a`, `q_b`, and `o_proj` **already allow `.fp8_pb`**, including MTP. Their execution fields are `Proj`, and their consumers call `projection.matmul`.
- Absorb/expand read BF16 `wk/wv` derived from **kv_b**, not raw `q_b` or `o_proj` weights. Quantized `q_b` produces ordinary BF16 query rows before absorb.
- The two-problem BF16 optimization returns false for FP8 projections, allowing the ordinary projection calls to run.
- KDA’s input split specifically represents MXFP8 QKV. It is not a general block-FP8 input path, but your checkpoint keeps those weights BF16.

These are explicit in [cuda_weights.zig:246](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:246), [cuda_dsa_block.zig:108](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_dsa_block.zig:108), [cuda_proj.zig:30](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_proj.zig:30), and [cuda_kda_block.zig:12](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_kda_block.zig:12).

**Dense MLPs need an execution change.** Their weight union has only NVFP4, BF16, and MXFP8. Add an FP8 arm, implement gate/up and down using `projection.matmul`, and extend `cuda_tile`’s exhaustive switch to leave that arm on the lane. The current dtype heuristic would misidentify this checkpoint’s dense E4M3 weights as MXFP8. Resolve their actual format instead. See [cuda_mlp.zig:10](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_mlp.zig:10), [cuda_weights.zig:340](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:340), and [cuda_tile.zig:49](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_tile.zig:49).

Your settings separation is correct. Existing tiling skips FP8 projections; absorb MMA consumes BF16 activations and kv_b halves; KV representation is selected independently. HC/split exchanges transport FP32 partial rows and BF16 residual/normed rows, not expert bytes. The pieced and paired paths still call the same FFN body/fill functions. See [cuda_latent.zig:33](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_latent.zig:33), [cuda_forward_pieces.zig:23](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_forward_pieces.zig:23), and [cuda_forward_pair.zig:46](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_forward_pair.zig:46).

Finally, patch 0014 does the opposite of your description: beside DFlash2, **the target MTP layer, its caches, and its window buffers stay out**; only the draft logits head loads. Plain MTP drafting still loads layer 45 through `moeLayer`, so implement and test its FP8 experts, but omit their memory from the DFlash2 configuration. See [cuda_engine.zig:52](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:52), [cuda_weights.zig:566](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:566), and [0014 patch:4](/home/zurih/c1bench/tp4/patches/0014-glm-cuda-beside-DFlash2-the-target-MTP-layer-its-cac.patch:4).

B. The FP8 expert arithmetic is BF16 × widened FP8, with FP32 accumulation.

Both kernels read BF16 activations. E4M3 codes are widened to BF16 pairs, consumed by `mma.sync…f32.bf16.bf16.f32`, accumulated over each 128-input scale block, then multiplied by that block’s FP32 scale into the final accumulator. There is **no dynamic activation quantizer** in this path. See [fp8_experts.cu:32](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/fp8_experts.cu:32), [fp8_experts.cu:78](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/fp8_experts.cu:78), and [experts.cuh:43](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/experts.cuh:43).

Consequently, this establishes storage support for your checkpoint, not equivalence to a reference implementation that dynamically rounds activations to E4M3. Treat that arithmetic distinction as a separate fidelity comparison.

`swiglu_limit` is supported. The epilogue rounds gate/up sums to BF16, clamps gate above `limit`, clamps up to `[-limit,limit]`, computes the activation, and stores BF16. That is the existing GLM-style asymmetric clamp, not a symmetric gate clamp. See [experts.cuh:55](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/experts.cuh:55).

The prompt kernel is complete in source: 64-pair items, staged activation/weight loads, partial-item masks, scale application, skip handling, and all three epilogues are instantiated. `Fp8Experts.load` resolves all six decode/prompt symbols, and `Grouped.prompt` dispatches FP8 directly. The old NVFP4 refusal does not describe this tree: NVFP4 also has `runPrompt` now. See [fp8_experts.cu:135](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/fp8_experts.cu:135), [experts.zig:259](/home/zurih/c1bench/tp4/tf/zig/src/cuda/experts.zig:259), and [experts.zig:437](/home/zurih/c1bench/tp4/tf/zig/src/cuda/experts.zig:437).

I found no sm_121-specific refusal for these kernels. The compile records include sm_121 prompt instances: 128 threads, 32 KiB static shared memory, and 218/221 registers per thread. That establishes recorded compilability, not good throughput. This path uses BF16 MMA and does not gain native FP8 activation throughput automatically. See [registry_media.zig:60](/home/zurih/c1bench/tp4/tf/zig/src/core/contract/registry_media.zig:60).

C. The early budget estimate is not a resident-memory calculation.

The server sums actual safetensor file sizes named by the index, then GLM multiplies that total by its largest sharded-axis fraction. At TP=4 this is approximately **305.8 / 4 = 76.45 GiB**. It does not account accurately for replication, prepared layouts, skipped MTP weights, or subsequent BF16 packing. The comment describing this as an upper bound is therefore too strong. See [cuda_memory.zig:17](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda_memory.zig:17), [cuda_native.zig:106](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:106), and [native/cuda.zig:305](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda.zig:305).

Your expert-code arithmetic is right:

- Backbone routed codes: **70.875 GiB/rank**.
- MTP routed codes, when loaded: **1.6875 GiB/rank**.
- Backbone routed block scales: approximately **17.7 MiB/rank**.

Routed packing is a permutation without code expansion. Lane projections differ: their scale tables occupy one FP32 per output row per 64 inputs, or **6.25% of code bytes**, plus padding. This is much larger than the compact checkpoint scale tables, although small beside routed experts. See [cuda_fp8pb.zig:36](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_fp8pb.zig:36).

I would treat your ~82 GiB as a planning estimate, not a verified weights total; the “~9 GiB replicated” component needs an inventory. Also distinguish weights from prompt scratch, drafter allocations, sequence state, and caches. This tree allocates a 4096-row prompt arena, including expert slots and exchange buffers. DFlash2 and transport allocations are made before ranks report their remaining room, and stream admission uses the smallest rank’s room. See [cuda_state.zig:279](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_state.zig:279) and [cuda_native.zig:275](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:275). Three 1M windows should be admitted from that measured remainder, not from `128 − estimated weights`.

For host staging, **do not copy `nvfp4Experts`’ whole-layer allocation pattern**. Its eight-expert batching is inside the BF16 quantization helper; the surrounding loader still allocates full-layer packed arrays and copies them through `L.host`. An analogous FP8 implementation would hold 1.6875 GiB of packed arrays, plus a large upload copy. Allocate final device destinations once, pack eight experts, upload at offsets, and reuse the chunk. Your 48 MiB figure then describes the three rank-local code matrices for eight experts; add packer temporaries and scales. See [cuda_weights.zig:475](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:475) and [cuda_weights.zig:515](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_weights.zig:515).

Shard mapping does not eagerly allocate 306 GiB of host RAM: mappings use `populate=false`. Nevertheless, default prefetch reads **every tensor in the current layer**, using eight 8 MiB reader buffers, before the layer’s pages are released. One complete routed layer is 6.75 GiB. Strided down cuts can also fault most source pages despite retaining one quarter of their bytes. These are material NFS and unified-memory costs. See [safetensors.zig:231](/home/zurih/c1bench/tp4/tf/zig/src/core/safetensors.zig:231), [cuda_load_ahead.zig:13](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_load_ahead.zig:13), and [checkpoint_prefetch.zig:42](/home/zurih/c1bench/tp4/tf/zig/src/core/checkpoint_prefetch.zig:42).

For first boot, I would use `TF_GLM_LOAD_PREFETCH=0`, disable the optional pinned prompt-cache tier, and measure physical memory alongside CUDA allocations. The pinned tier shares the Spark’s RAM too; its automatic allowance is separate from device weights. See [cuda_snapshot_host.zig:16](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_snapshot_host.zig:16).

D. I would use three reviewable patches, with tests accompanying each.

1. **Metadata and scales:** strict block-FP8 admission/parser helper, shared scale-name resolution, declared-format case, malformed-input tests.
2. **Routed experts:** a separate loader helper, 128-unit cuts, bounded staging, FP8 module ownership and dispatch, upload/launch tests.
3. **Dense execution and integration:** FP8 dense arm, tiler exclusion, final admission wiring, diagnostics, and synthetic end-to-end coverage.

Keep generic kernels unchanged initially. An improved resident-weight estimator can be a separate patch; it should price replication and packing explicitly rather than adding a checkpoint-specific fudge factor.

For host tests, model the expert helper on `cuda_exl3_test.zig`’s recording loader: it exercises real loading logic and records upload bytes, names, and pointers. Use `cuda_fp8pb.zig`’s fake safetensor construction for actual `_scale_inv` lookup. See [cuda_exl3_test.zig:50](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_exl3_test.zig:50) and [cuda_fp8pb.zig:90](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_fp8pb.zig:90).

Cover every rank at worlds 2 and 4, distinct gate/up/down scales across both block axes, expert counts crossing the eight-expert chunk boundary, and independent unpacking of uploaded results. Include wrong shapes/dtypes, missing or ambiguous scales, unsupported config, allocation/upload failures, and `.none/.draft/.mtp` loading. A `.draft` test should succeed with layer-45 tensors absent.

Add recording dispatch tests for 16- versus 64-pair plans, BF16 prompt versus FP32 decode down outputs, `skip=count`, slots nine, and clamp ten. Existing FP8 device fixture tests exercise packing, plans, prompt/decode outputs, but use default clamp zero and skip −1; they do not establish GLM’s sentinel/clamp behavior. See [fp8_experts_tests.zig:76](/home/zurih/c1bench/tp4/tf/zig/tests/cuda/fp8_experts_tests.zig:76).

Extend `glm-synth` with a genuinely non-ModelOpt FP8 variant: its present `fp8pb` variant still uses NVFP4 routed experts and dense layers. Extend the independent loader gate and `glm-fp8g-shares` to TP=4 and the new projection classes. See [glm_synth.zig:17](/home/zurih/c1bench/tp4/tf/zig/tests/cuda/glm_synth.zig:17), [glm_loader_gate.zig:440](/home/zurih/c1bench/tp4/tf/zig/tests/cuda/glm_loader_gate.zig:440), and [glm_fp8g_shares.zig:81](/home/zurih/c1bench/tp4/tf/zig/tests/cuda/glm_fp8g_shares.zig:81).

My first-boot gates would be:

- **Load correctness:** exact code/scale upload checks pass; all ranks report the expected cuts and formats; no requantization of FP8 experts.
- **Operator correctness:** independent FP64 checks on served expert shapes, clamp/skip cases, prompt tails, and lane projections pass.
- **Model fidelity:** teacher-force identical token IDs against a trusted implementation of this **same FP8 checkpoint**, preferably with a BF16-activation comparison as well as its dynamic-FP8 mode.
- **Serving correctness:** same-checkpoint drafted/plain greedy equality, then solo/concurrent, cold/warm, paired/unpaired prompt, and resume checks.
- **Capacity:** increase context and concurrency only after measured room and physical-memory behavior support it.

For fidelity, report mean NLL, per-document loss deltas, top-1 agreement, reference top-two margins, and worst logprob discrepancies. Use fixed prose/code/multilingual continuations and enough tokens to expose systematic drift. The existing TP prompt-logprob path scores supplied next-token IDs and gathers the full vocabulary across ranks; it can support this harness. See [cuda_prompt_logprobs.zig:66](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_prompt_logprobs.zig:66).

As provisional **investigation triggers**, I would use a mean loss increase above 0.02 nats/token or disagreement above 1% on positions whose reference top-two margin exceeds 0.5. Those are suggested starting points, not measured tolerances for GLM. Calibrate them against known-good arithmetic differences. EXL3 TP=4 is useful for operational and performance comparisons; token equality with it is not a fidelity gate.

Enable `TF_CUDA_MEMORY_RECEIPTS=1` and retain `TF_MEMORY` load boundaries from every rank. These receipts cover process-owned CUDA buffers, so also collect RSS/PSS and `/proc/meminfo`. See [memory_receipt.zig:8](/home/zurih/c1bench/tp4/tf/zig/src/cuda/memory_receipt.zig:8).

Grep for named admission/format failures, `FormatNotLoaded`, `SlicePartialsNotHeld`, `RankMismatch`, CUDA allocation/illegal-address errors, and prompt-cache refusals. Add a concise loader summary naming **weight format**, expert count, local width, scale convention, loaded MTP status, and allocated bytes. Existing “FP8 KV” logs cannot prove FP8 weight dispatch.

E. The patch series contains format-specific optimizations, but they are mostly isolated.

The supplied numbering differs from your hints: **0047 is the mm-policy/prompt host-fake patch; 0046 concerns context/snapshot tests; 0049 moves lane host tests.** The EXL3 load/cache changes are 0026–0029 and 0081–0082.

The important assumptions to retain are:

- **0001/0003/0004/0007/0078:** EXL3 loading, rotations, prompt eligibility, scratch, and epilogues. FP8 should bypass them.
- **0030/0074:** fused select/group/combine is explicitly gated by `l.exl3 != null`. Leave FP8 on the existing ordinary router/Grouped path; do not broaden that gate. [cuda_moe.zig:57](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_moe.zig:57)
- **0006/0010:** shared-expert side-stream ordering is generic and remains relevant to FP8. Preserve the fork, join, and error drain. [cuda_moe.zig:77](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_moe.zig:77)
- **0009/0064/0073/0100:** packed BF16-derived linear optimizations do not automatically apply to FP8 lane weights; two-problem execution already declines them.
- **0026–0029/0081–0082:** EXL3 trellis loading and dependent-launch optimizations give no evidence about FP8 performance.
- **0094:** the 4096-row prompt arena affects memory regardless of expert format.

Useful audit terms are `c.exl3`, `l.exl3`, `bf16_slots`, `grouped(`, `prompt.eligible`, `mostScratch`, `nvfp4`, `mxfp8`, `addProj`, `b16Shapes`, `skip`, and `chain.moe`. One residual config-based assumption is `b16Shapes` choosing dense BF16 scratch from `c.exl3`; FP8 lane projections currently require zero external slice partials, so that is not an immediate FP8 blocker, but it deserves a recording test. See [cuda_state.zig:119](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_state.zig:119).

I am confident in the loader/dispatch findings above. Sustainable decode rate, cold-prefill throughput, dynamic-FP8 fidelity differences, and three-window capacity are still measurement questions.