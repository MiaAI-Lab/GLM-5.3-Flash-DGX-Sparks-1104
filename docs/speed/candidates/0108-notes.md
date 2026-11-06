# 0108: the busy prompt step (`TF_GLM_CHUNK_SOLO` on this pin's 4,096-row chunk)

What it does: with `CHUNK_SOLO=1` a prompt step beside a decoding stream takes `TF_GLM_CHUNK` rows (2,048 or 1,024) of the
4,096-row chunk instead of the whole chunk. A step alone is unchanged: a whole chunk, or a pair (`TF_GLM_PROMPT_PAIR`). Kept
prompt states stay on the 4,096-row grid, so the prompt cache, resumes and `snapshots` are unchanged. A decoder's pause beside
a long prefill is about the busy step's rows over the prefill rate, so it shrinks with them. Default off.

Upstream's `TF_GLM_CHUNK_SOLO` goes the other way (a 2,048-row grid, two chunks in one pass when alone). This pin's grid is
already 4,096 with the pair, so the port keeps the grid and adds the short step. The recipe's `CHUNK_SOLO=1` sets
`TF_GLM_CHUNK_SOLO=4096` (upstream's only "on" value; 2048, its off, is refused by name) and `TF_GLM_CHUNK=$CHUNK_BUSY`,
on both ranks (the setting is part of the ranks' start-up fingerprint).

## Mechanism

`cuda_prompt_solo.zig` (new): `busyAt` is rank 0's choice for a step (setting on, another lane decoding, no prompt log-probs, no
image rows); off the chunk grid a step is a busy one, so a chunk split into busy steps is finished in busy steps. `ringsAgree`
keeps the DSA index rings' rebase at the chunk's start. Rank 0 sends the choice as flag 4 of the chunk header and the followers
obey it. The MTP head absorbs in `TF_GLM_CHUNK`-row pieces whenever the setting is on (its index-key ring rebased on the step
length; with `--drafter`, the recipe default, that head's absorb never runs, so tokens cannot move from it).

## Exactness

A chunk boundary does not change the bytes: the EXL3 expert sums do not depend on which pass holds a row (patch 0094 shipped on
that), attention and the KDA records read the committed cache and the row's own causal prefix in fixed tiles, the one
state-dependent reader (the index ring's base) is pinned to the chunk by `ringsAgree`, and the DFlash2 drafter only proposes.
`tf-cuda-test glm-solo-check <ckpt> <dc.json> <rank> <world> <idfile>` runs busy twins against whole-chunk twins on one engine
(first token, next-window logits and tokens, ring bases, the KDA records read next, kept-state marks), with
`TF_GLM_CHUNK_SOLO=4096` and `TF_GLM_CHUNK=2048` or `1024`. Its first run found the MTP ring base following the step length
(hence the pieces above); it passes at TP=4 at both sizes.

## Measured

`docs/speed/RESULTS.md`, "Patch 0108": worst pause beside a cold ~200k prefill 2.77 s -> 1.55 s (2048) -> 0.88 s (1024) on the
pair, the prefill itself 143 -> 145 -> 151 s, hashes, concurrent == solo, retention, cold prefill alone and decode alone
unchanged. Upstream measured a 2,048-row pass about 7% dearer than a 4,096-row one at TP=2, which is the cost of sharing.
