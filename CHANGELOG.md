# Changelog

- **2026-11-06** -- engine patch 0108: `CHUNK_SOLO=1` (with `CHUNK_BUSY=2048|1024`), off by default: a prompt step beside a decoding stream takes `CHUNK_BUSY` rows of the 4,096-row chunk (upstream's `TF_GLM_CHUNK_SOLO`, ported to this pin's grid). On this pair (TP=2, Ablit, DFlash2, 4 streams, 262,144 window) the worst decode pause beside a cold ~200k prefill falls 2.77 -> 1.55 s (2048) and 0.88 s (1024) for that prefill taking 143 -> 145 s and 151 s; greedy hashes, concurrent == solo, retention (3 x 80k, 3/3), cold prefill alone and decode alone are unchanged. Numbers and method: `docs/speed/RESULTS.md`, `docs/speed/candidates/0108-notes.md`.
- **2026-11-05** -- engine patch 0107: `TF_TOOL_CALL_GUARD=1`, opt-in. A Hermes agent showed raw
  `<tool_call>session_search<arg_key>…` text: its turn after two tool rounds was sent without `session_search`, and
  the server (as vLLM) keeps a call to a tool the request does not offer as text. With the guard, such a request gets a
  closing system note that tools are unavailable and the markup hidden (no tools, or `tool_choice: "none"`), or only
  the unoffered call hidden (other tools offered); streamed or whole. Measured on a live serve with the patch: off,
  every case replies as before and the greedy hashes are unchanged; on, no markup in 12 of 12 such replies (prose or
  empty), calls to offered tools still parse (4 of 4), hashes and speed unchanged. README: the patch row and a
  troubleshooting row.
- **2026-11-05** -- the memory guard `start.sh` leaves running holds none of the caller's stdin, stdout or stderr: it
  kept the pipe of `ssh spark './start.sh'` open, so that ssh session never returned after LIVE (issue 7, crystal8194).
  Its one message now goes to `memguard.log` in the run directory (it went to the terminal and the log). A test,
  `scripts/tests/memguard-detached.sh`, reads the guard's descriptors from /proc; `scripts/tests/run.sh` runs the
  syntax check, shellcheck when installed, and the tests.
- **2026-11-05** -- README: the prompt cache is funded from the remainder after the page pool and the streams' fixed
  parts, so `PROMPT_CACHE_GIB` only caps it (eight streams at 1M leave under 1 GiB; `PARALLEL=4` funds 8 GiB), the
  room differs by a few GiB between Sparks so the reserve that holds two windows is a per-pair check, hits land on the
  4,096-row chunk grid, and nothing survives a restart at this pin (issue 3: jds1-ai, micachi; issue 7: crystal8194).
  Video input: the template accepts `video_url` parts but the image ships no `libtfvideo.so`, so they return HTTP 500;
  the settings table, the "What is checked" line and a troubleshooting row say so, with the out-of-image build that
  works (issue 8, chai-md). Building the video runtime into the image is a Dockerfile change with an FFmpeg stage,
  not taken here.
- **2026-11-04** -- `WORKER_WEIGHTS=check` accepts a local checkpoint with no `.revision` marker when its config.json
  and shards are there, as `worker_weights_ready`'s comment always said; `MODEL_REVISION`'s default made the marker
  mandatory even for `MODEL_SOURCE=local` (issue 3, failure 1). The `E_WEIGHTS` hint names the marker.
- **2026-11-04** -- the worker's state paths (`WORKER_CACHE_DIR` and the state and overlay built on it) default under
  the worker's own home when its login differs from this Spark's and every default is untouched; an explicitly set
  path wins as before (issue 3, failure 2). The `E_START` failure to create the directory says what to set.
- **2026-11-04** -- `start.sh` runs rank 1 as the worker's own login when `CONTAINER_USER` is this Spark's (the default).
  With different ids on the two Sparks (uid 1000 and 1001 on ours), rank 1 could not write
  to its state folder and stopped at `hashing the checkpoint failed at model-00001-of-00083.safetensors` with
  `AccessDenied`. Equal ids, root and another `CONTAINER_USER` run as before; a `CONTAINER_USER` set to this Spark's own ids
  counts as the default. With different ids the start stops, with the fix, when a folder in rank 1's `HOME`
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
