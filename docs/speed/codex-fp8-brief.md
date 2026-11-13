# Consult: serving unsloth/GLM-5.3-Flash-FP8 (block-FP8, Z.AI layout) on this GLM engine at TP=4

Same directory as before: `tf/` is the engine (upstream f350cd3 + the recipe's 106 patches, branch port; ignore the
one extra "lanes" commit at HEAD), `patches/` the patch files. Read-only. Answer in prose with file:line references.
I take your answer as advice; measurements outrank it.

## The checkpoint

`unsloth/GLM-5.3-Flash-FP8` @ a160e229: `quantization_config = {"quant_method": "fp8", "fmt": "e4m3",
"activation_scheme": "dynamic", "weight_block_size": [128, 128], "modules_to_not_convert": [...]}`; 62 shards, 305.8
GiB, tensor names `model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight` (e4m3 [n, k]) with
`...weight_scale_inv` (fp32 [n/128, k/128]; dequant = w * scale_inv, DeepSeek's convention). Block-FP8 modules: the
routed experts of all 43 MoE layers (the MTP layer 45 included), the shared experts (43), the three dense MLPs
(layers 0-2: gate/up/down), and in the 12 DSA layers `self_attn.q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
`o_proj`. bf16 everywhere else: kv_b_proj, the KDA layers' q/k/v/o and conv, hyper-connections, norms, gates,
embed_tokens, lm_head. GLM-5.3-Flash: hidden 4096, moe_intermediate 2048, 288 experts top-8, first_k_dense_replace 3,
45 layers + 1 MTP.

## What I found in the engine

- Today the GLM family admits exactly: ModelOpt NVFP4, ModelOpt MIXED_PRECISION (recipes table, core/modelopt.zig),
  and EXL3 (`quant_method: exl3`, config.zig:153 sets c.exl3, head_unit 128). Everything else: NotModelOpt /
  UnsupportedQuant (cuda_native.zig:459).
- A generic block-FP8 expert format exists: `cuda/experts.zig` Format.fp8g with `Fp8Experts` (kernels/cuda/
  fp8_experts.cu, decode and prompt kernels, `packOne`/`packGateUp` host packers, scale block 128 both ways, cols 32,
  prompt tile 64), dispatched by `experts.Grouped.decode/prompt` when `Layer.format == .fp8g`. kolibri1 uses it end to
  end (families/kolibri1/pack.zig:88 experts(): gate/up interleaved, down alone, scales [E][2][NI/128][D/128] fp32
  then [E][1][D/128][NI/128]; forward.zig:327 `Grouped{ .fp8g = &o.k.fp8x }`).
- GLM's NVFP4 path: families/glm/cuda_experts.zig builds `Grouped{ .nvfp4 = &ks.nvfp4 }` only; cuda_weights.zig
  moeLayer (line ~430) loads routed experts as EXL3 (c.exl3) or `nvfp4Experts(i)` (bf16 rows quantized on device in
  chunks of 8 experts, packed with Nvfp4Experts.packGateUp/packOne). Shared experts go through
  `formats.cutProj(..., &.{ .bf16, .fp8_pb }, ...)`, i.e. block-FP8 (`.fp8_pb`, cuda_fp8pb.zig: e4m3 codes + fp32
  128x128 block scales, cut by rows or inputs, packed for the lane as a `qmmf` fp8g weight) is ALREADY a lane
  projection format for the shared expert, kv_b (line ~246) and, in the mixed-precision tests, kv_a and mla_o.
  Dense MLP (line ~340) decides `.bf16` (exl3) / `.mxfp8` (an e4m3 gate_proj: FP4-Local's 1x32 MXFP8) / `.nvfp4`.
  q/k/v (`.mxfp8` or `.bf16`), o_proj (`.bf16, .mxfp8`).
- Sharding: cuda_shard.zig:38 cuts moe_inter at unit 128 for EXL3 else `unit`; at TP=4 a rank holds 512 of the 2048
  expert rows (gate/up) and 512 of down's 2048 inputs, both multiples of the 128 scale block.
- The MTP head (cuda_mtp_load.zig) loads layer 45 as DSA + MoE through the same moeLayer; the draft head reads
  lm_head bf16 (fp8g refused there, not relevant: lm_head is bf16 in this checkpoint). With DFlash2 (`--drafter`)
  this stack runs no target MTP prompt pass (patch 0008) but still loads the head (0014).
- Memory at TP=4: 42 x 288 x (512x4096 + 512x4096 + 4096x512) bytes = 70.9 GiB of routed experts a rank, the MTP
  layer's 1.7 GiB, plus ~9 GiB replicated: ~82 GiB a rank, ~21 GiB of room for KV (three 1M windows on FP8 KV).

## The plan I intend (please check and improve)

1. Admission (config.zig): `c.fp8 = true` for quant_method "fp8" with fmt e4m3 and weight_block_size [128, 128];
   refuse other block sizes with a named reason. Format resolution for a non-ModelOpt, non-EXL3 checkpoint: an e4m3
   tensor whose `weight_scale_inv` exists with shape [n/128, k/128] is `.fp8_pb`; bf16 tensors `.bf16`. The reader
   cuda_fp8pb.Fp8pb.read must accept the `weight_scale_inv` name (ModelOpt's name?) with the same multiply semantics.
2. Routed experts: a new `fp8gExperts(i)` in cuda_weights.zig: per expert, read the e4m3 [n, k] codes and the fp32
   block scales, cut the rank's rows (gate/up: [lo, hi) of n; down: [lo, hi) of k, a strided copy per row), cut the
   matching scale blocks, pack with `Fp8Experts.packGateUp` / `packOne` into `Layer{ .format = .fp8g, .up, .down,
   .up_scale, .down_scale, .width = ni, .dims = d, .experts = 288, .limit = c.swiglu_limit }`; the chunked host
   staging as nvfp4Experts does (8 experts a chunk, ~48 MiB each). Shard unit 128 for c.fp8 as for EXL3.
3. Kernels: cuda_experts.Kernels gains `fp8: ?Fp8Experts` loaded when c.fp8; `run` builds `Grouped{ .nvfp4 = ...,
   .fp8g = ... }` so decode and prompt dispatch by the layer's format. Is anything else in the GLM forward
   format-specific for routed experts (the EXL3 path has its own prompt kernels from patch 0001 and friends; does the
   NVFP4/Grouped path on this patched tree still run prompt rows through experts.Grouped.prompt, with the staged
   router plans, under patches 0004/0009/0020/0028/0049's prefill changes)?
4. Dense MLPs (0-2) and the 12 DSA layers' q_a/q_b/kv_a/o_proj: accept `.fp8_pb` where today only bf16/mxfp8/nvfp4
   are wanted, giving the same `qmmf` fp8g lane weights the shared expert already gets. Which consumers of those
   projections (DSA absorb, the prompt kernels, the KDA/DSA front, the "FP8 lane") take a Proj.fp8g today, and which
   assume bf16 or mxfp8 (cuda_proj.zig:9 Proj = bf16 | fp8g | mxfp8)? Anything reading q_b or o_proj as raw bf16 rows?
5. Settings: DENSE/HEAD q8 packing applies to bf16 linears only (unchanged); TF_GLM_ABSORB_MMA unchanged; KV fp8 on
   pages unchanged; the DFlash2 drafter unchanged (it reads the target's residual taps).

## Questions

A. Correctness of the plan above: what did I miss (the MTP layer's FP8 experts beside DFlash2, the e_score bias,
   the router, swiglu_limit on fp8g, the `skip` row semantics, hc/HC split exchanges that may assume expert bytes)?
B. The fp8g kernels: what activation precision do they use (bf16 x, e4m3 dynamic per 128 block as the checkpoint's
   "activation_scheme dynamic" implies, or bf16 x fp8 weights widened)? Any sm_121 caveat? Do the prompt kernels
   exist and run for 64-pair items on this tree (the Oct 10 commit said grouped NVFP4 prompt rows were refused until
   a staged kernel landed; is fp8g's prompt kernel complete)?
C. Memory accounting: where does the "weights do not fit the budget" estimate come from (file bytes a rank?), and
   will it be right for a 306 GiB checkpoint split four ways (~82 GiB a rank)? Host RAM during the load (staging of
   e4m3 reads over NFS, the pack buffers) on a 128 GB unified Spark that also holds the GPU memory.
D. The least-risk patch set and order (one patch or several), the host tests to add (fake checkpoint with fp8 block
   experts at worlds 2 and 4: which existing tests to model on), and the GO/KILL gates for the first boot (what to
   compare the FP8 serve against: the EXL3 TP=4 baseline's greedy tokens will differ; propose the NLL/agreement
   check and what to grep in the logs).
E. Anything in the 106 patches that assumes EXL3 or NVFP4 experts specifically (grep hints), e.g. 0001, 0003, 0047
   "exl3 decode loads", 0004/0009 prompt kernels, 0046 L2 prefetch.
Keep it concrete; say what you are sure of and what you are guessing.
