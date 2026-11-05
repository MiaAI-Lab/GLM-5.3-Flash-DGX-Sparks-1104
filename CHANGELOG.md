# Changelog

- **2026-11-04** -- engine patch `0109`: a prompt step beside a decoding stream takes `TF_GLM_CHUNK` rows (2,048 or 1,024) of the
  4,096-row chunk instead of the whole chunk; alone, the chunk (and the pair) run as before, and kept states stay on the 4,096-row grid.
  Upstream's `TF_GLM_CHUNK_SOLO` (v1.2.1) ported onto this pin inverted: the pin already takes two chunks a step alone (0050-0062), it
  lacked the short step beside a decoder. The MTP head absorbs in `TF_GLM_CHUNK`-row pieces whenever the setting is on, so its
  index-key ring's base no longer follows the step length (found by `tf-cuda-test glm-solo-check`, which now compares only the KDA
  records the next step reads and names the state region at the first differing byte). `CHUNK_SOLO=0|1` (default 0) and
  `CHUNK_BUSY=2048|1024` in `scripts/config.sh`, plumbed by `start.sh` and `start-tp4.sh`. Measured on four DGX Sparks behind a switch
  (TP=4, q6 dense, RoCE gathers): `glm-solo-check` PASS on every rank at 2048 and 1024; served, three decoders beside a cold ~200k
  prefill: worst decode pause 1.96 s -> 1.17 s (2048) -> **0.65 s (1024)**, p99 1.80 -> 0.98 -> 0.57 s; that prefill 95.3 s -> 97.7 /
  99.9 s (+2.5% / +4.8%); cold prefill alone and greedy hashes unchanged; four concurrent prompts equal solo; 3 of 3 ~275k conversations
  resume. Details: docs/speed/RESULTS.md, "Engine patch 0109".
- **2026-11-04** -- engine patches `0107` and `0108`, for four Sparks on a switch. `0107`: the one-shot RoCE all-gather
  (`zig/src/cuda/roce.zig`, `roce_gather.cu`) at any world instead of two: one RC queue pair per peer per HCA, a receive region
  `[slot][world-1]` with a flag per peer per HCA, the shard written to every peer, the kernel waiting for `(world-1) x n_hca`
  flags and copying each peer's stripes into rank-ordered slots; a connect-time probe writes a token to every peer and refuses
  (`PeerUnreachable`) when one cannot be written to, so a ring never opens it; world 2 keeps the old layout bit for bit
  (a test pins it). `tf-cuda-test roce-gather <rank> [world]` compares every size with `ncclAllGather` and times a call: on
  four GB10s behind a CRS804, 16 KiB a rank 18.2 us against NCCL's 32.9, 32 KiB 22.0 against 39.4, 0 mismatches in 3,000
  mixed calls. `0108`: the DFlash2 drafter MLP split across every rank (`ff = FF / world` when 256-aligned, a rank-ordered fp32
  `df_sumn`), world 2 bit-identical to `df_sum2`. In the model (TP=4, greedy, Ablit): the per-layer gathers 4.77 -> 3.31 ms a
  verify call, the drafter block 5.5 -> 3.5 ms a round, the decode round 36.4 -> 33.0 ms: one stream 53.8 / 83.0 -> 59.4 / 90.8
  tok/s (prose / code), four streams 118 -> 131.5; the harness's greedy hashes are unchanged (`fab1d3b6894b`,
  `e71492bf766b`), acceptance unchanged. `0108` alone moves nothing (its block overlaps other work) and is kept as exact and
  lighter. `start-tp4.sh`: on a switch `MESH_ROCE=auto` turns `TF_ROCE` on when `0107` is in the image (`on` / `off` insist);
  a ring stays on NCCL. Details: docs/speed/RESULTS.md, "Engine patches 0107 and 0108".
- **2026-11-04** -- `./start-tp4.sh` on a switch: `FABRIC=auto|ring|mesh` (README "4 Sparks"). When every pair of the four
  Sparks shares a RoCE subnet (every ConnectX-7 port on one switch) the launcher runs them as a mesh: no NCCL graph files,
  NCCL's own algorithm choice (`MESH_NCCL_ALGO` pins it), every RoCE device toward every peer with `NCCL_CROSS_NIC=1`,
  subnet-aware routing and `NCCL_IB_MERGE_NICS=0`, the engine's default eager connect, workers in any order, and
  `TF_GLM_PREFILL_REDUCE=split` (`PREFILL_REDUCE=auto`: `gather` on a ring, `split` on a switch; a ring refuses `split`
  before the start). The fabric is read before the banner, so the reduce mode and the context fit key (now keyed on the
  fabric too) follow it; a ring runs exactly as before. Measured on four DGX Sparks behind a MikroTik CRS804, one 200G cable
  each: decode equal to the ring's (55.6 / 83.2 tok/s prose / code at one stream, 163 at eight), prefill 2,421-2,462 tok/s
  with `split` against 1,865-1,882 with `gather`; the published checkpoint gives the ring's greedy hashes (`b6ae240144bc`,
  `9b68d8a77151`), drafts on or off, `split` or `gather`; `NCCL_DEBUG=INFO` shows no retry, no socket channel and no queue
  pair between the two subnets. Details: docs/speed/RESULTS.md, "TP=4 on a switch".
- **2026-11-04** -- `WORKER_WEIGHTS=check` accepts a local checkpoint with no `.revision` marker when its config.json
  and shards are there, as `worker_weights_ready`'s comment always said; `MODEL_REVISION`'s default made the marker
  mandatory even for `MODEL_SOURCE=local` (issue 3, failure 1). The `E_WEIGHTS` hint names the marker.
- **2026-11-04** -- the worker's state paths (`WORKER_CACHE_DIR` and the state and overlay built on it) default under
  the worker's own home when its login differs from this Spark's and every default is untouched; an explicitly set
  path wins as before (issue 3, failure 2). The `E_START` failure to create the directory says what to set.
- **2026-11-04** -- `start.sh` runs rank 1 as the worker's own login when `CONTAINER_USER` is this Spark's (the default), as
  `scripts/tp4.sh` already does. With different ids on the two Sparks (uid 1000 and 1001 on ours), rank 1 could not write
  to its state folder and stopped at `hashing the checkpoint failed at model-00001-of-00083.safetensors` with
  `AccessDenied`. Equal ids, root and another `CONTAINER_USER` run as before; a `CONTAINER_USER` set to this Spark's own ids
  counts as the default, as in `tp4.sh`. With different ids the start stops, with the fix, when a folder in rank 1's `HOME`
  (`WORKER_CACHE_DIR/home`) is not writable by that login, which a workaround that let an earlier rank write there can leave
  behind. A worker reading shards only the head's id can read (mode 0600 over NFS without `all_squash`) now needs shards its
  login can read.
- **2026-11-04** -- CREDITS and NOTICE: the Aevonix entry removed. On reflection it earns no line here: the serving scripts are
  this lab's own (Aevonix's recipe reworked the lab's GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold scripts, and this recipe's
  copies reach it through the lab's 2x RTX PRO 6000 recipe), the banner was never theirs, and the lab's own 2x RTX PRO 6000
  repository keeps the credit where its scripts are Aevonix's rework outright. The one debt that survives is the K12 loop, now
  recorded where it is used: patch 0078's comment names patches/aevonix/0073 in Aevonix's
  GLM-5.3-Flash-EXL3-4x-RTX-PRO-6000-TensorFold (it said MiaAI-Lab/Aevonix, a repository that does not exist).
- **2026-11-04** -- CREDITS and NOTICE, the Aevonix entry corrected: it said this recipe's `start.sh`, `stop.sh`,
  `scripts/config.sh`, `scripts/prepare.sh` and `scripts/banner.sh` are adapted from Aevonix's 4x RTX PRO 6000 recipe, which
  states the direction backwards. The adaptation began from this lab's own scripts: Aevonix's NOTICE names the lab's
  `scripts/config.sh` and `start-tp4.sh` (GLM-5.3-Flash-EXL3-4x-DGX-Sparks-TensorFold, created 2026-10-01, the day before
  Aevonix's repository) as "the basis" of their `start.sh` and `scripts/config.sh`, and their `stop.sh` and
  `scripts/prepare.sh` as "adapted from Mia's scripts"; and `scripts/banner.sh` here is the lab's own art - the two-Spark and
  four-Spark recipes carry the same MIA AI LAB banner file and this recipe recolours it, while Aevonix's banner is their own
  artwork, so it never came from them. The old text also cited patch 0073 to "MiaAI-Lab/Aevonix", a repository that does not
  exist; the K12 loop patch 0078 follows is `patches/aevonix/0073` in Aevonix's repository. What survives of the credit: this
  recipe's four serving scripts do pass through Aevonix's rework (the 2x RTX PRO 6000 recipe adopted it - their texts are the
  closest matches to ours), so Aevonix stays credited for the intermediate rework and the K12 loop, as the reworker of the
  lab's scripts rather than their origin.
- **2026-11-04** -- `./start-tp4.sh` and `./stop-tp4.sh`: the recipe over four DGX Sparks cabled as a ring without a
  switch (README "4 Sparks"). The ring is found from the Sparks' ConnectX-7 addresses; NCCL runs ring-only with a graph
  file per rank and subnet-aware routing, connected at first use (the engine's connect-everything-at-init would join
  rank 0 to the Spark across the ring, where the first boot timed out); the engine's prefill exchanges run as all-gathers
  between neighbours; the workers read the head's checkpoint over NFS. Engine patch `0106`: the start's peer warm-up
  follows `TF_GLM_PREFILL_REDUCE` (split: every peer; ring: next and previous; gather: none) instead of sending to every
  rank. NCCL connects at first use (`NCCL_RUNTIME_CONNECT=1`) and bootstraps on the head's LAN address, so neither
  NCCL's tree nor the engine's follower start-up reaches the Spark across the ring. Measured on this kit (greedy, the
  recipe's harness): 56 / 85 tok/s prose / code at one stream against the pair's 38 / 57, 115-120 at four streams
  against 79-81, 163-166 at eight against 105-106, prefill 1,750-1,890 tok/s, eight 1M windows and a 3.6 GiB prompt
  cache a rank. The two-Spark `./start.sh` is unchanged. Details: docs/speed/RESULTS.md, "TP=4 on the four-Spark ring".
- **2026-11-04** -- Upstream TensorFold since this recipe's pin (`f350cd3e`, 1,059 commits behind main) was reviewed for
  fixes worth carrying (docs/speed/RESULTS.md, "Upstream since the pin"). The one that applied cleanly, the lane fix
  `ed39acc5a` (a stream finishing on its first token released its lane only after its trace), was built as `0106` and
  **killed on the pair**: with it, decode fell to 28 / 36 tok/s at one stream and 51-66 at four, with a connection and a
  waiter left open after twenty one-token replies; the same sequence on the 105-patch image is normal (38 / 57 / 78-80).
  Upstream's deferred release does not fit this stack's lane host (0093, 0104). The candidate stays in
  docs/speed/candidates as a record; the served image is the 105-patch `8c4ca7bd8c39`.
- **2026-11-04** -- `0105`: a tool parameter whose schema names no type (`{}`, a description alone, a `$ref` that resolves
  nowhere) takes the JSON object or array the model wrote instead of coming back as a JSON string of it; scalars, plain
  words and quoted strings stay text, and a tool declared without a parameter schema keeps streaming its arguments as
  before. The rule is the maintainer's from the 2x DGX Sparks recipe's issue #180 (upstream keeps untyped values as text);
  a union that admits an object already decoded one on this engine pin. Found by a sweep of the sibling repos' issues
  (docs/speed/RESULTS.md, "Sweep of the sibling repos"), reproduced here, host-tested (43 parser + 7 parameter tests).
- **2026-11-04** -- `0104`, warm turns for GLM (ported from the 2x RTX PRO 6000 recipe): after a reply the server prefills the
  next turn's rendering in the background (a prompt-only pass that steps aside for real work), so a user message after an
  agent loop resumes past the loop instead of re-reading it. Measured here (docs/speed/RESULTS.md): the next user message
  after a 9.3K-token tool loop cached 8,192 tokens (was 4,096), 1.6 s to answer (was 3.5 s); 17 of 18 passes stepped aside
  under back-to-back requests; first-token latency 0.15 s unchanged; decode 38.4 / 56.6 at 1 stream, 79-81 at 4 streams,
  greedy hashes unchanged. Each pass logs `next turn: background prefill of N tokens done` or `stepped aside`.
- **2026-11-04** -- `0103` (ported from the RTX recipe's `0096`, with a guard): a tool call that follows prose quoting
  tool-call markup ("Qwen's `<tool_call>{}`") is parsed as the call instead of coming back as content. The guard: an opener
  inside a value still open in the text before it (an unbalanced `{`, an open `<arg_value>` or `<parameter=`) is never
  retried - without it a call quoting a whole call inside a JSON string argument invoked the quoted call (found by Codex,
  confirmed by a host test). `zig build test-tool-parse-host`: 42 pass.
- **2026-11-04** -- `MEMORY_RESERVE_GIB` 8 -> 9.5 (scripts/local.sh): the prompt cache had silently gone to ZERO. The page
  planner takes whole 1M-token windows of pages before the cache, and the room (19.8 GiB at reserve 8, up from 18.2 the
  day before) crossed the threshold for a third window; rank 0 logged "no prompt cache" and every chat turn re-prefilled
  its history (6 s a turn at 9.3K tokens). At 9.5 the pool is two windows (4,098 pages, 2.1M tokens) and the cache 3.6 GiB;
  decode, prefill, mixed and 8-stream numbers unchanged, 1.5 GiB more free memory. Check rank 0's startup line after
  every start: "4098 pages", "prompt cache: N GiB", no "cannot keep one state" warning.
- **2026-11-04** -- Decode campaign items closed by measurement (docs/speed/RESULTS.md): KDA C1 (merging the KDA chain's
  launches) - fusion loses, dependent launches save ~0.2 ms a round, below the engine's resolution; the exact PDL
  variant is parked as candidate `0199` (opt-in `TF_GLM_PDL_CHAIN=...,kda`). Sampled decoding at temperature 0.7: this
  pair 36.7-38.9 tok/s at 1 stream and 75-76 at 4, the spark1+2 reference 21 and 30-31 (its round doubles under
  sampling); its 4-stream lead over this pair is greedy-only.
- **2026-11-03** -- `0099`: `TF_GLM_PROF=1` also brackets shared windows (several streams in one forward), reported as
  `glm prof shared N rows` by total rows; off unless set, no kernel or token change. With it, a 4-stream round is
  attributed (docs/speed/RESULTS.md, "Decode attribution campaign"): routed experts 4.2-5.8 ms a row at the GB10's
  memory bandwidth, KDA mixers 16-17 ms flat, drafter 7-8 ms. Two exact candidates measured neutral and parked in
  docs/speed/candidates: `0098` (`TF_GLM_DFLASH2_EDGE`) and `0100` (`TF_GLM_MM_STAGES`). The chain cut, the tpm rule,
  a bf16 drafter, bf16 dense and stock TensorFold v1.2.1 were all slower or equal; the numbers are in that section.
- **2026-11-03** -- `PREFILL_SHARE` (default `0.68`, passed to both ranks as `TENSORFOLD_PREFILL_SHARE`): a stream
  decoding beside another request's long prefill no longer gets one decode round per prompt chunk. A chunk is atomic
  (4,096 rows with 0094/0095, about 2.5 s here), and at the engine's default share of 1 a chat beside a long prefill
  produced a token or two per chunk. At `0.68` each chunk first owes the decoders about 1.2 s of decode rounds (~22),
  and the long prompt's first token comes about 47% later. Replies are byte-identical at any value: the share only
  moves when the next chunk runs. The rows in the README are the knob's arithmetic on this pair's measured chunk and
  round times; the mixed-traffic measurement is from the 2x RTX PRO 6000 recipe (2.3 -> 27.2 chat events/s during a
  cold 175K prefill, that prompt's first token 33.4 -> 49.8 s). `PREFILL_BUDGET` is passed as
  `TENSORFOLD_PREFILL_BUDGET` when set; it cannot split a chunk. Takes effect at the next `./start.sh restart`
  ([README](README.md#prefill-beside-decoding)).
