<h1 align="center">GLM-5.3-Flash on DGX Sparks with TensorFold</h1>

<img src=".github/image.png" alt="GLM-5.3-Flash on two DGX Sparks" width="100%">

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> [!IMPORTANT]
> **This is a brand-new recipe, version 1104.** It is not an update of
> [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
> (the Python-engine recipe and its `zig-engine` branch, which stay as they are). It descends from the lab's
> [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold): the same engine pin
> and 93 of its patches unchanged, plus a two-Spark launch over the ConnectX-7 link, unified-memory budgeting and twelve
> GB10 patches, every default measured on the pair. Different engine lineage, different patch stack, different numbers.

**Engine: TensorFold 1.1.9** (`tensorfold-native --version`), upstream commit
[`f350cd3e`](https://github.com/ashhart/TensorFold/commit/f350cd3ea253a042e205746928271f9d8f71a82d) plus this recipe's
112 engine patches (`patches/engine/`).

This recipe serves GLM-5.3-Flash from an EXL3 checkpoint on **two NVIDIA DGX Sparks** (GB10, 128 GB unified memory
each) at tensor parallel 2, one rank on each Spark, over their ConnectX-7 link. It uses TensorFold's native Zig/CUDA
server (`tensorfold-native`) built in Docker for `sm_121`. The default checkpoint is the pinned 4 bpw EXL3 quant,
downloaded on the first start and mounted read-only. The pair's KV pool holds about 2M tokens in total (fp8 on shared
pages, 12.5 GiB on each rank). Rank 0 serves the OpenAI-compatible API; rank 1 follows its steps.

```sh
./start.sh            # check both Sparks, build the image once, send it to the worker, check the weights, start both ranks, smoke-test
./stop.sh             # stop both ranks, archive the logs (logs/glm53-spark-*.tar.gz)
DRY_RUN=1 ./start.sh  # print the whole plan: the link it finds, the docker build, both ranks' command lines, the health wait
```

The engine is TensorFold 1.1.9 at commit `f350cd3e` ([ashhart/TensorFold](https://github.com/ashhart/TensorFold)), plus the
`patches/engine/*.patch` files, compiled with Zig 0.17.0 for `sm_121` by `docker/Dockerfile`.


## Why this engine on a Spark

- `sm_121` is the engine's default `-Dsm` target, GLM at tensor parallel over hosts is an upstream-measured
  configuration, and the engine's **RoCE all-gather** (`zig/src/cuda/roce.zig`) carries the decode and verify
  exchanges as one-shot RDMA writes over the Sparks' cable: same bytes, same bits as NCCL, less time.
- **Unified-memory budgeting** is upstream: the engine budgets from `/proc/meminfo` plus the pages parked on the
  kernel's per-CPU lists, not from a card's free bytes.
- The two single-host optimisations in the stack (peer-store all-gather, copy-engine prompt exchange) step aside across
  hosts instead of failing.

## What is checked

112 patches apply to `f350cd3e` with no rejects; the image builds on aarch64 for `sm_121` (about 6 minutes) and
reaches the worker by content; link discovery finds both PCIe twins of the cabled port and the RoCE v2 GID that
holds each port's own IPv4; NCCL and the engine's RDMA all-gather run over the cable; the full 1,048,576-token window
serves 8 streams beside a 3.6 GiB prompt cache; greedy replies are byte-stable across every change
([CHANGELOG.md](CHANGELOG.md)); sparkDash's quality suite scores the served template's effort line
([The effort line](#the-effort-line)). Image input was checked by a user (issue 8: PNG, and four frames in order); video input
needs the `WITH_VIDEO=1` image (checked here: a three-colour clip described in order). `W8A8` is stashed on quality.

## Performance

Served configuration: `PARALLEL=8`, `CONTEXT=1048576`, `KV=fp8` on shared pages, `DENSE=q8`, `HEAD=q8`,
`ABSORB_MMA=1`, DFlash2 drafts at 4 bits, `MEMORY_RESERVE_GIB=9.5`, image `8c4ca7bd8c39` (105 patches). The complete
measured record, traps and retractions included, is [docs/speed/RESULTS.md](docs/speed/RESULTS.md).

**Official numbers** (sparkDash, this pair, 2026-11-04). Decode is prose at temperature 0.7 with **thinking on**;
prefill is a cold prompt of the given size:

| Decode, prose, T=0.7, thinking on | 1 stream | 2 streams | 3 streams | 4 streams |
| --- | ---: | ---: | ---: | ---: |
| Total tok/s | **47.8** | **66.5** | **77.0** | **87.3** |
| tok/s per stream | 47.8 | 33.6 | 27.3 | 22.6 |
| TTFT | 200 ms | 324 ms | 437 ms | 557 ms |

| Prefill, prompt tokens | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 1,601 | 1,688 | 1,767 | 1,801 | 1,812 | 1,778 | 1,708 |
| TTFT | 2.8 s | 5.1 s | 9.5 s | 18.4 s | 36.4 s | 74.0 s | 153.7 s |

**The lab's harness** (`docs/speed/harness/`, greedy, thinking off, fixed prompts, the client on a third Spark), the
shape every engine change is gated on:

| Decode, greedy, thinking off | 1 stream | 4 streams | 6 streams | 8 streams |
| --- | ---: | ---: | ---: | ---: |
| Prose, total tok/s | 38.4 | 79-81 | 94-95 | 105-106 |
| Code, 1 stream | 56.6 | | | |
| Sampled prose, T=0.7, total | 39.1-39.3 | 75-78 | | |

A chat decoding beside a 50K-token prefill keeps 12.9 tok/s with a worst pause of 2.7 s (one 4,096-row chunk). On the
same checkpoint, prompts and day, prose at T=0.7 reads 39.1 / 75-78 tok/s (1 / 4 streams) here against 29.6 / 68-71
on TensorFold's Python engine v0.6.0 and 21.3 / 30-31 on the Zig v1.0.7 branch; under greedy decoding the v1.0.7
branch leads at 4 streams (89-95 against 79-81).

> [!WARNING]
> **Measure cold prefill with fresh prompt content.** The prompt cache resumes a shared prefix and reports a rate
> that is not prefill; `prompt_tokens_details.cached_tokens` is 0 when nothing was resumed.

Prefill was tuned on 2026-11-02 (docs/speed/RESULTS.md): `ABSORB_MMA` is +8.0% with identical answers and is on; the
4,096-row prompt chunk (0094/0095) is +7% with byte-identical replies; `W8A8` reaches +10.5% but rounds activations
(one of ten answers wrong) and stays off; `q4` dense is rejected on quality; the cable is about 3% of a long prompt's
time. At equal dense width the engine is at parity with the v1.0.7 build's prefill.

## Requirements

- **Two DGX Sparks** (GB10, 128 GB, CUDA 13 driver) cabled to each other on their ConnectX-7 ports, with IPv4
  addresses on one subnet. One cable is two netdevs and two RDMA devices (the port's two PCIe links); both are used.
- Key-based ssh from this Spark to the worker; your login in `docker` on both; Docker with the NVIDIA Container
  Toolkit and BuildKit; `rdma-core` (without it every exchange stays on NCCL, more slowly).
- `curl`, `python3`, `flock`, `tar`, `gzip`, `sha256sum`, `nvidia-smi`, `ssh`, `rsync`, `ip`, `awk`; `hf` for downloads.
- The checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@76c0b517` (about 164 GiB, downloaded when missing);
  both Sparks read every shard, so the worker gets it over the link (`WORKER_WEIGHTS=rsync`) or by NFS. The default
  drafter `incoai/GLM-5.3-Flash-DFlash2@bf582e4e` is CC BY-NC-ND 4.0: for commercial use set `DRAFTER=mtp` or `none`.
- Disk: about 40 GiB for the image on each Spark, 180 GiB for the checkpoint on each that holds it.

## How it starts

1. **Checks** on both Sparks: Linux aarch64, Docker, the NVIDIA runtime, a GB10, memory as the engine counts it
   (`scripts/memavail.sh`), disk, the ports.
2. **Image**: `docker build` tagged `tensorfold-glm53-spark:<hash>` over the Dockerfile, the patches and the pins;
   skipped when that label exists; the worker gets the same image by content over the link.
3. **Weights**: downloaded when missing, placed on the worker by `WORKER_WEIGHTS`; both ranks mount `/model` and
   `/dflash2`, so their startup fingerprints agree.
4. **The link**: the route to the worker gives the netdev, sysfs the port's RDMA devices and the RoCE v2 GID holding
   each side's IPv4; a second rail is probed with a full-size ping before use.
5. **Launch**: rank 1 on the worker first, then rank 0 here, each PID 1 in its own container with `--network host
   --ipc=host --device /dev/infiniband`; rank 0 serves the API and watches the follower (a rank that dies stops the
   other).
6. **Ready**: `/health`, then `/v1/models`, then a greedy smoke test.

Every rank carries the recipe's `TF_*`/`TENSORFOLD_*`/`NCCL_*` settings; any other such variable you export reaches
both ranks by name (`TENSORFOLD_API_KEY` included, never printed).

### Prefill beside decoding

A long prompt is read in atomic 4,096-row chunks (about 2.5 s each); other streams decode between chunks.
`PREFILL_SHARE=p` makes each chunk owe the decoders `(1 - p) / p` of its own time in decode rounds first. At the
default `0.68` a chat beside a long prefill keeps about 12-13 tok/s with a worst pause of one chunk, and the long
prompt's first token comes about 47% later than at `1`. Replies are byte-identical at any value.

## Memory on a 128 GB unified Spark

- VRAM and system RAM are one pool. The budget is `MemAvailable` plus parked pages, minus `MEMORY_RESERVE_GIB`; per
  Spark on the served pair: model 83.3 GiB, KV pool 12.5 GiB (4,098 pages, 2.1M tokens), prompt cache 3.6 GiB, 9.5 GiB
  reserve kept free.
- **The page pool takes whole 1M windows before the prompt cache**, so a larger room can silently leave the cache at
  zero and every chat turn re-reads its history. The 9.5 GiB reserve holds the pool at two windows; after a start,
  rank 0's startup line must show `4098 pages` (with `CACHE_FIRST=1 PROMPT_CACHE_GIB=12` below: one window,
  `2056 pages` at `PARALLEL=8`), `prompt cache: N GiB` and no `cannot keep one state` warning.
- **The cache is funded from the remainder**, so `PROMPT_CACHE_GIB` only caps it: room after the reserve, minus the
  pool, minus about 0.42 GiB of fixed state for each stream past the first (measured on issue 3). Eight streams at
  the 1M window leave 0.7-0.9 GiB whatever the cap says; `PARALLEL=4` at the same window funds an 8 GiB cap, and a
  190k-token prompt then resumes at 99%. The room itself differs by a few GiB between Sparks (DGX OS builds), so a
  pair with more room than ours needs a larger reserve to stay at two windows, and one with less room gets a smaller
  cache (1.2 GiB with 2.4 GiB less room, issue 7); the startup line is the check.
- **What the cache keeps, at this pin** (issues 3 and 7): hits land on the 4,096-row prompt-chunk grid, so a
  4,870-token prompt resumes 4,096 and up to 4,095 trailing tokens are prefilled again each turn (about 2.5 s); the
  cache is device memory only, and without `LEARN_GIB` nothing survives a restart (the CUDA GLM family refuses
  `--learn-conversations` at `f350cd3e`; engine patch 0109 adds it, see below). A 1.2 GiB cache kept one conversation up to 252k tokens but not 378k, and two of eight ~38k
  conversations recalled newest first. For many long-lived agent sessions, size `PARALLEL` and `CONTEXT` for the
  cache (see above) rather than for the stream count.
- **Long agent sessions: `CACHE_FIRST=1 PROMPT_CACHE_GIB=12`** (engine patch 0108). The cache is set aside before
  the room is counted in whole windows, so the eight streams share one 1M window of pages (2,056 pages, any request
  still up to 1,048,576 tokens) and the cache gets the rest. Measured on the pair (2026-11-06) at the served
  `PARALLEL=8` and 1M window: rank 0 funds a 9.3-9.7 GiB cache (the startup log says when the room funds less than
  `PROMPT_CACHE_GIB` asks); a 380k-token conversation, then eight unique ~31k ones, then the long one's next turn
  resumes 99% cached (3.0 s instead of 234 s), and all eight short ones resume. Decode and reply hashes are unchanged,
  and three concurrent agent loops ran as fast as without it. Without it, the served config resumes 1 of 8 and not the
  long one. The trade: the streams together hold one window's tokens at a time instead of two, so a second
  request past ~1M tokens of live context waits. `PARALLEL=2 CONTEXT=524288 PROMPT_CACHE_GIB=32` also keeps them
  (12.4 GiB), at two streams. Add `LEARN_GIB` (below) so the states also survive eviction and restarts.
  With `CACHE_FIRST=1`, `PROMPT_CACHE_GIB` is a priority, not a floor: it is taken off the room only to count the
  streams' whole windows, and the cache then gets what those windows leave, up to the ask. So a larger ask can fund
  more cache than a smaller one, by costing the streams a window, and a lower `CONTEXT` alone can buy a second window
  instead of more cache. On a second pair (issue 16, 17.8-18.9 GiB of room) `CONTEXT=524288` funded 9.7 GiB with
  `PROMPT_CACHE_GIB=12` (two windows, 2,050 pages) and 12.9 GiB with `14` (one window, 1,032 pages). Read the page
  count and the funded cache in the startup line after every change.
- **Conversations on disk: `LEARN_GIB=30`** (engine patch 0109, `--learn-conversations`, off by default). Kept prompt
  states also go to disk in `CACHE_DIR/learn` on the head Spark, every rank's part, so a conversation whose device
  copy was evicted, or that outlived a restart, resumes from disk instead of prefilling again. No state is written
  while the disk has under `LEARN_MIN_FREE_GIB` (15) free. Measured on the pair (2026-11-06, `PARALLEL=8`, 1M window,
  DFlash2, Ablit): an 18.7k-token conversation's next turn after a restart has its first token in 1.9 s (cold: 10.7 s)
  and after an eviction in 2.5 s; a 102.5k-token one after a restart in 1.8 s (1,013 MiB read in 1.4 s; cold: 58.4 s).
  Replies equal the in-memory resume's. A state is written behind the request (713 MiB in 1.5 s for 53k tokens) and
  costs a stream decoding beside it about 1% (34.57 against 34.77 tok/s). A state read from disk restarts DFlash2 at
  its mark, as `TF_GLM_SNAP_DRAFTER=0` does for one in memory. Damaged files fail their check and the turn prefills.
  About 0.5 GiB a 16k-token state, less for later turns, which share earlier turns' rows. Off: replies byte-identical.
  Two limits decide whether a long conversation imprints at all, and both bit this pair's 380k and 500k agent sessions:
  the write permit (engine patch 0112, below) and `LEARN_DAILY_GIB`, the store's write budget a day. The engine's
  default budget is 32 GiB, which two such sessions spend in hours while `LEARN_GIB` still has room, after which every
  save is skipped with `DailyBudgetSpent`. Measured on the pair (2026-11-09, `PARALLEL=8`, 1M window,
  `CACHE_FIRST=1 PROMPT_CACHE_GIB=6 LEARN_GIB=100 LEARN_DAILY_GIB=100 KV_SHARD=1`): a 385k-token state written in
  16.5 s (2,559 MiB), a 520k one read back in 3.2 s (3,556 MiB), and those sessions' per-turn first token went from
  183-250 s to 2.7-9 s. A cold 500k prompt still costs about 650 s with two streams prefilling.
- **KV sharding: `KV_SHARD=1`** (engine patch 0110, `TF_GLM_KV_SHARD`, off by default). Each stream's FP8 latent rows
  live on one Spark, its owner, instead of on both (the pooled index keys stay on both, as the indexer runs on both),
  so a token costs 3,542 bytes a rank instead of 6,380 and the same memory holds about twice the tokens. The other
  Spark's queries go to the owner for attention, so decode costs 1-4%. Measured on the pair (2026-11-06,
  `PARALLEL=8`, 1M window, DFlash2, Ablit): at the defaults the pool goes from 4,098 to 8,196 pages (2.10M to 4.20M
  tokens); decode 34.93 to 34.37 tok/s at one stream and 85.0 to 82.0 aggregate at eight; cold first token unchanged;
  every greedy reply identical (decode, 1-8 concurrent streams, eight 30k contexts at once, a 102.5k conversation).
  The pool then takes most of the room and the prompt cache gets 0.5-1.2 GiB, too little to keep a 100k-token
  conversation (880 MiB on its owner), so **use it with `CACHE_FIRST=1 PROMPT_CACHE_GIB=6`**: a 4.1 GiB cache a
  rank and 6,147 pages (3.15M tokens), both more than unsharded at the defaults (2.4 GiB, 2.10M). With `LEARN_GIB`
  each Spark writes the rows it holds and a read lands on an owner chosen then: the 102.5k conversation resumed
  after a restart with its first token in 1.13 s (1,013 MiB in 698 ms). A store written sharded is not read
  unsharded or the reverse (its identity differs), so turning `KV_SHARD` on or off starts an empty one.
- `HOST_CACHE_GIB=0` on GB10 (a pinned host tier takes the pool's own memory). The memory guard stops the pair in
  order when a Spark stays under 2.5 GiB available, and logs that to `logs/current/memguard.log`. To free memory:
  lower `PARALLEL`, `DENSE`/`HEAD` bits or `VISION=0`; raise `MEMORY_RESERVE_GIB` on out-of-memory errors.

## CONTEXT=auto

The engine refuses a window that does not fit after loading, naming bytes. `CONTEXT=auto` asks for `CONTEXT_MAX`
(1,048,576) and, on a refusal with figures, restarts both ranks at the window those figures allow (times
`CONTEXT_FIT_MARGIN`, the tighter rank winning), up to `CONTEXT_RETRIES` (3) times; a window that served is kept in
`.state/context-fit.tsv` for the same settings. `CONTEXT_RETRY=0` turns it off.

## Settings

Layered: `scripts/config.sh` defaults, then `.env`, then `scripts/local.sh` (trusted Bash, gitignored), then exported
variables. **The served pair's `local.sh`** sets `PORT=8888`, `PARALLEL=8`, `CONTEXT=auto`, `KV=fp8`, `DENSE=q8`,
`HEAD=q8`, `ABSORB_MMA=1`, `HOST_CACHE_GIB=0`, `MEMORY_RESERVE_GIB=9.5`, `MODEL_SOURCE=local`, `WORKER_WEIGHTS=check`.

### The pair

| Setting | Default | Meaning |
| --- | --- | --- |
| `WORKER` | **required** | The second Spark's ssh target on the direct link, e.g. `zurih@10.0.34.4` |
| `TP` | `2` | Two Sparks with `./start.sh` |
| `FABRIC_PEER`, `FABRIC_SELF`, `NCCL_SOCKET_IFNAME`, `TF_ROCE_HCA`, `NCCL_IB_GID_INDEX` | auto | Found from the route to `WORKER`, the port's RDMA devices and its RoCE v2 GID table |
| `NCCL_CHANNELS` | `4` | `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` |
| `ROCE`, `ROCE_MAX_KB` | `on`, `4096` | The engine's one-shot RDMA all-gathers, and the largest shard they carry; `off` puts everything on NCCL |
| `PORT`, `HOST`, `NCCL_PORT` | `8020`, `0.0.0.0`, `29583` | The API address; rank 0's bootstrap port on the link (the watch line uses the next one) |
| `WORKER_WEIGHTS` | `rsync` | `rsync` the checkpoint over the link, `nfs` (the worker mounts it; `NFS_*` settings), or `check` (refuse unless present) |
| `WORKER_MODEL_DIR`, `WORKER_DFLASH2_DIR`, `WORKER_CACHE_DIR` | the same paths | Where the worker keeps them; the home-based defaults re-root on the worker's own home when the two logins differ |
| `SSH_OPTS` | `-o BatchMode=yes -o ConnectTimeout=10` | For every call to the worker |

### Weights and serving

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_SOURCE`, `MODEL_DIR` | `download`, `DATA_DIR/model` | Fetch the pinned 4 bpw checkpoint when missing, or `local` to serve `MODEL_DIR` as it is (a tree with no `.revision` marker is checked by its config.json and shards) |
| `ABLIT` | `0` | `1`: the gated [Ablit weights](#ablit-weights); needs `HF_TOKEN` |
| `DRAFTER`, `DRAFTER_BITS` | `dflash2`, `4` | `dflash2`, `mtp` (the checkpoint's own head) or `none` |
| `PARALLEL` | `4` | `--parallel`; admitted only if it fits, `auto` serves up to 8. The pair runs 8: free at 1-4 streams, 94-95 tok/s at 6, 105-106 at 8 |
| `PREFILL_SHARE`, `PREFILL_BUDGET` | `0.68`, empty | See [Prefill beside decoding](#prefill-beside-decoding) |
| `CONTEXT`, `CONTEXT_MAX` | `auto`, `1048576` | See [CONTEXT=auto](#contextauto) |
| `KV`, `KV_PAGES` | `fp8`, `on` | `--kv-cache`, one shared pool of 512-row pages with `CONTEXT` as one request's cap |
| `DENSE`, `HEAD` | `q8`, `bf16` | `bf16`, `q8`, `q6` or `q4` for the dense linears and the LM head; the pair's `local.sh` sets `HEAD=q8` |
| `MEMORY_RESERVE_GIB`, `HOST_CACHE_GIB`, `PROMPT_CACHE_GIB` | `12`, `0`, empty | See [Memory](#memory-on-a-128-gb-unified-spark); the pair runs the reserve at 9.5 |
| `LEARN_GIB`, `LEARN_MIN_FREE_GIB`, `LEARN_DAILY_GIB` | `0`, `15`, `0` | `LEARN_GIB` > 0 keeps the prompt states on disk too, up to that many GiB (`--learn-conversations`, patch 0109); `LEARN_DAILY_GIB` is what the store may write a day (`--learn-daily-gib`; 0 keeps the engine's 32 GiB default, which long agent sessions spend in hours); the write permit is one whole state at the served context (patch 0112). See [Memory](#memory-on-a-128-gb-unified-spark) |
| `KV_SHARD` | `0` | `1` keeps each stream's latent rows on one Spark (`TF_GLM_KV_SHARD`, patch 0110): about twice the tokens, 1-4% decode; use with `CACHE_FIRST=1 PROMPT_CACHE_GIB=6`, see [Memory](#memory-on-a-128-gb-unified-spark). Needs `KV=fp8`, KV pages, DFlash2 or no drafter |
| `CACHE_FIRST` | `0` | `1` sets `PROMPT_CACHE_GIB` (required) aside before the streams' windows (`TF_PROMPT_CACHE_FIRST`, patch 0108), so the cache comes first, still capped by what the room leaves (set `PROMPT_CACHE_MIN_GIB` to enforce a floor); see [Memory](#memory-on-a-128-gb-unified-spark) |
| `PROMPT_CACHE_MIN_GIB` | empty | The smallest prompt cache a start may serve with. `start.sh` reads each rank's startup line and stops both ranks before LIVE (`E_CACHE`) when one funds less; the cache is what that start's room leaves, so `PROMPT_CACHE_GIB` caps it but does not guarantee it, even with `CACHE_FIRST=1` (issue 16). Empty: no check |
| `WITH_VIDEO` | `0` | `1` builds `libtfvideo.so` (FFmpeg 9.0.2, static, LGPL; h264/hevc/vp8/vp9/mpeg4 in mp4/mkv) into its own image tag, so `VISION=1` serves `video_url` parts; a few minutes more at build |
| `VISION`, `VISION_MAX_IMAGES`, `CHAT_TEMPLATE` | `1`, `50`, `auto` | Image input (`image_url` parts) on rank 0's tower, with `templates/glm-5.3-flash-vision.jinja`. `video_url` parts too with `WITH_VIDEO=1` (otherwise HTTP 500, issue 8); audio parts are unsupported and an mp4's audio track is ignored |
| `THINKING`, `REASONING_EFFORT`, `MAX_TOKENS`, `CLEAR_THINKING` | `0`, `xhigh`, `32768`, `1` | Request defaults (`xhigh` renders the checkpoint's trained "Reasoning Effort: Max" line); earlier turns' reasoning is dropped from the prompt |
| `EFFORT_LINE` | `thinking` | The template's `<|system|>Reasoning Effort: Max` line with thinking on only, as TensorFold v0.6.0's recipe wrote it (the served template is written to `.state/chat_template.jinja`); `always` keeps the checkpoint's template, which writes it into every prompt and makes a thinking-off reply reason in its answer. See [The effort line](#the-effort-line) |
| `ABSORB_MMA`, `W8A8` | `1`, `0` | Prompt numerics: `ABSORB_MMA` is free (+8.0%, identical answers); `W8A8` rounds activations and stays off |
| `CE_EXCHANGE` | `0` | Needs CUDA IPC between ranks, which two Sparks do not have |
| `SERVED_NAME`, `SERVE_EXTRA_ARGS`, `TENSORFOLD_API_KEY` | `GLM-5.3-Flash`, empty, unset | The model id clients send; more `serve` flags; a bearer key for every route but `/health`. Requests that send no sampling fields get the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95) and no `top_k`; the pair's `local.sh` sets `SERVE_EXTRA_ARGS='--top-k 20'`, the old deployment's default (greedy replies unchanged) |

### Build, checks and timeouts

| Setting | Default | Meaning |
| --- | --- | --- |
| `SM`, `ZIG_CPU` | `121`, `native` | GB10's target; `baseline` builds an image that runs elsewhere |
| `TF_REPO`, `TF_REF`, `ZIG_VERSION`, `ZIG_SHA256`, `NCCL_WHEEL` | pinned | All hashed into the image tag |
| `IMAGE`, `IMAGE_REPO`, `DOCKER_BUILD_FLAGS`, `CONTAINER_NAME` | `tensorfold-glm53-spark:<hash>`, `glm53-spark` | |
| `MEMGUARD`, `MEMGUARD_FLOOR_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_INTERVAL_S`, `MIN_AVAIL_GIB` | `1`, `2.5`, `4`, `2`, `100` | See [Memory](#memory-on-a-128-gb-unified-spark) |
| `DATA_DIR`, `CACHE_DIR`, `LOG_DIR`, `LOG_KEEP` | `~/.cache/aevonix-glm53`, `DATA_DIR/zig-home`, `logs`, `10` | Downloads, the ranks' `HOME`, run logs |
| `WAIT_TIMEOUT`, `SMOKE_TIMEOUT`, `STOP_TIMEOUT` | `2400`, `180`, `45` | Seconds |
| `DRY_RUN` | `0` | `1` prints the plan and changes nothing |

## Hermes and other agents: the tool-call guard

The server turns the `<tool_call>` markup GLM writes into OpenAI `tool_calls` only for tools the request offers, as
vLLM does. Agents sometimes send a turn without the tool the model then calls, and that call reaches the agent as raw
text. With [Hermes Agent](https://github.com/NousResearch/hermes-agent) it looks like this in the reply:

```
<tool_call>session_search<arg_key>after</arg_key><arg_value>7d</arg_value><arg_key>query</arg_key><arg_value>…</arg_value></tool_call>
```

It happens on a turn sent with no `tools`, with `tool_choice: "none"`, or with a tool list that lacks the called tool:
typically the final-summary turn after the agent reaches its iteration limit, or a toolset that is switched off.
Calls to offered tools are not affected, and parallel calls parse.

**First, on the Hermes side:** raise its iteration limit if long tasks hit it, and keep the toolset that holds the tool
(for `session_search`, session search) enabled for the agent you run.

**Then, to keep the markup out of replies,** turn on the guard (engine patch 0107, off by default):

```bash
TF_TOOL_CALL_GUARD=1 ./start.sh restart
```

or, to keep it on for every start, add this line to `scripts/local.sh` (it must be exported to reach the ranks):

```bash
export TF_TOOL_CALL_GUARD=1
```

The first start after pulling this version builds the new image (the patch set changed). Check it took effect with
`docker exec glm53-spark env | grep TF_TOOL_CALL_GUARD`.

What it does, streamed or not:

- A request that offers no tools (none sent, or `tool_choice: "none"`) whose history called tools gets a closing system
  note that tools are unavailable for this reply, and any `<tool_call>` block the model writes anyway is removed. The
  reply is then prose (the model answers from what it has) or empty (it only wanted to call).
- A request that offers tools has calls to tools it did not offer, and malformed calls, removed; calls to offered tools
  come back as `tool_calls` as before.
- Each removal is logged: `docker logs glm53-spark 2>&1 | grep "tool-call guard"`.
- Off (the default), every reply is byte-identical to a build without the patch. On, replies to ordinary requests are
  unchanged too (same greedy hashes, same speed): the guard only acts on the requests above.

Leave it off for a client that reads tool calls out of the reply text itself (tools described in its own system prompt,
no `tools` field): the guard would remove the calls that client is waiting for.

## The effort line

The checkpoint's chat template (and this recipe's vision copy of it) opens every prompt with
`<|system|>Reasoning Effort: Max` unless a request's `reasoning_effort` is `low` or `high`: its second line maps
anything else, an absent value included, to Max. The engine renders the template as written, so a thinking-off
prompt carries the line too (7 tokens: `/tokenize` on a one-line user message counts 27 against 20 without it), and
with thinking off the model reasons in its answer instead of answering. TensorFold v0.6.0's recipe, whose server
rendered GLM's prompt in its own code, wrote the line with thinking on only.

Measured with sparkDash's quality suite on this pair (greedy, seed fixed, the same 755 items; the v0.6.0 run served
`DENSE=fp8 KV=bf16`): with the line in every prompt, QA 79.3% (119/150) against v0.6.0's 84.0%, with five
"binary representation, digits only" items cut by `max_tokens` 64 mid-division and four string reversals transposed;
MMLU 91.2% (260/285) against 88.1%, the longer visible reasoning helping under its 1,024-token budget. Replaying the
server's exact prompt through `/v1/completions` with and without the line reproduces every one of those flips, and
the thinking-on categories (reason, arith, track, GSM8K) render the same prompt on both servers. With
`EFFORT_LINE=thinking`, the same image: QA 86.0% (129/150), 140 of 150 replies byte-identical to v0.6.0's; MMLU
88.4% (252/285); reason, arith, track and GSM8K byte-identical to the run with the line, and no reply ends on
`length`.

`EFFORT_LINE=thinking` (default) writes the line with thinking on only: `start.sh` copies the chosen template to
`.state/chat_template.jinja` with its second line taking the effort from `reasoning_effort`, which the engine passes
with thinking on and leaves null with thinking off, and serves that copy on both ranks. `EFFORT_LINE=always` serves
the template as it is. A `CHAT_TEMPLATE` of your own is served as it is either way.

## Engine patches

`patches/engine/` holds 112 patches on TensorFold `f350cd3e`, applied in filename order at image build: 0001-0093 are
the [RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold)'s, unchanged;
0094-0112 are this pair's own or ported from that recipe or from upstream. None is GB10-specific: the arch-gated kernels test
`__CUDA_ARCH__ >= 1200`, which `sm_121` satisfies.

| Patches | What they do |
| --- | --- |
| 0001–0012 | EXL3 experts at mixed bit widths, 3-bit prompt kernels; FP8 attention loads, the shared expert on a side stream; one expert call a prompt pass, no MTP pass beside DFlash2 |
| 0013–0016 | DFlash2 reads complete prompt taps at TP>1 (upstream #1623); no target MTP layer beside DFlash2; a pinned host prompt-state tier (off here) |
| 0017–0033 | FP8 KV on shared pages; EXL3 load-ahead, fused MoE glue, KDA in one launch, one broadcast a follower step |
| 0034–0035 | Peer-store all-gather and copy-engine prompt exchange, both stepping aside across hosts |
| 0036–0048 | DFlash2 copy rounds, `ABSORB_MMA` (opt-in), "fits:" refusals, the device prompt-cache share |
| 0049–0066 | Paired prompt chunks (upstream #1423) extended to DFlash2; `W8A8` (opt-in) |
| 0067–0091 | DSA select on many CTAs, audited programmatic dependent launches along the decode chain, glue fusions, EXL3 prompt and decode kernels; each keeps an off switch (`TF_GLM_DSEL=0`, `TF_GLM_PDL_CHAIN=0`, `TF_GLM_MM_TWO=0`, `TF_GLM_TOPK_SUM=0`, `TF_GLM_SHARED_SEGS=0`, `TF_EXL3_SHFL=0`, `TF_EXL3_PDL=0`, `TF_GLM_KDA_STEP_WARPS=8`, `TF_GLM_HC_SPLIT=0`, `TF_GLM_SHARED_WRITES=0`, `TF_GLM_HC_PART=0`, `TF_GLM_DSCORES=0`). Never ship 0086 or 0089 without 0090 |
| 0092–0093 | A JSON answer stays JSON beside offered tools on chat, Responses and Messages; a reply grammar is freed when its stream cannot start |
| 0094–0097 | **This pair's.** A 4,096-row prompt chunk with DFlash2's `MAX_CTX` to match (+7% prefill, byte-identical replies); the GB10's measured cost tables and draft-rule seed |
| 0098–0102 | **This pair's,** env-gated, defaults behaviour-identical: `TF_GLM_DFLASH2_EDGE`, `TF_GLM_PROF` on shared windows, `TF_GLM_MM_STAGES`, `TF_GLM_DFLASH2_TCONF`, `TF_GLM_DFLASH2_Q32` (the decode campaign's exact knobs, each measured neutral) |
| 0103–0104 | From the RTX recipe: a tool call after prose that quotes a tool-call opener is the call (with a guard so a call quoted inside a call's value is never invoked); GLM warm turns, so a user message after an agent loop resumes past the loop (8,192 of 9.3K tokens cached, was 4,096) |
| 0105 | An untyped tool parameter takes the JSON object or array the model wrote; text stays text (sibling recipe issue #180) |
| 0107 | `TF_TOOL_CALL_GUARD=1` (opt-in, off by default): tool-call markup never reaches the client as text. A request that offers no tools (none sent, or `tool_choice: "none"`) although its history called them gets a closing system note that tools are unavailable, and any `<tool_call>` the model writes anyway is hidden; a request that offers tools has calls to tools it did not offer, and malformed calls, hidden. Streamed or whole; each hide is logged (`tool-call guard:`). Off: replies byte-identical |
| 0108 | `TF_PROMPT_CACHE_FIRST=1` (`CACHE_FIRST=1`, opt-in): the prompt cache is set aside before the room is counted in whole windows; off, the plan is as before. The startup log says when the room funds less cache than `--prompt-cache-gib` asks, and what frees more |
| 0109 | `--learn-conversations` for GLM on CUDA (`LEARN_GIB`, off by default), a backport of upstream [#1411](https://github.com/ashhart/TensorFold/pull/1411): kept prompt states on disk, every rank's part, written behind the request and read back (ahead of a queued turn when it can) after an eviction or a restart. On this stack the disk image copies 0015's two-tier lease page by page and rank 1's part travels the follow link of 0031. `--spill-gib` and `--checkpoint-slots` are refused for GLM. Off: replies byte-identical |
| 0110 | KV sharding (`KV_SHARD=1`, `TF_GLM_KV_SHARD`, off by default) at TP=2, toward upstream [#1431](https://github.com/ashhart/TensorFold/issues/1431): each stream's FP8 latent rows on its owner rank, a second page table pointing off-owner at a discard page, owner-computed decode and verify (no new kernel), prompts on a borrowed twin, kept states carried to another owner in 32 MiB pieces, and 0109's disk image under it. Refused by name with the MTP drafter, `--parallel 1`, other KV caches, or a part under 131k tokens. `tf-cuda-test glm-kv-shard-check` checks it against an unsharded twin byte for byte. Off: replies byte-identical |
| 0111 | **This pair's.** The clean image build prints no compiler warning: two fp8 expert kernels' unread kernel-local `RW`, `prefill_attention`'s readerless `SMEM_MAX`, `glm_mm`'s `Tile` `SMEM` member no instantiation uses, and `flashnext_fact` / `flashnext_attn` unroll loops that sat behind an `if constexpr` they could not reach (now behind the inverted guard, so each instantiation compiles only its reachable side). No arithmetic, launch shape or output changes: all six files' fatbins compile byte-identical |

| 0112 | GLM's disk write permit takes one whole state at the served context (toward upstream [#2337](https://github.com/ashhart/TensorFold/pull/2337); carries the `coldCharge` sizing from [#2382](https://github.com/ashhart/TensorFold/pull/2382)): `rows_write.coldCharge` prices one cold state and `cuda_glm_disk.attach` sets the permit to it, never past the RAM store's budget (5,705 MiB on this pair), so states over about 135k tokens can be written instead of skipped with `StagingFull`. The engine's startup line names the permit. Off: replies byte-identical |

Every default-on patch was A/B tested against the image before it: greedy replies identical plus a measured gain.

## Logs and stopping

`logs/current` holds `rank0.log` and `rank1.log`, collected from each Spark's Docker output. `./stop.sh` stops rank 0
first (it drains for `TF_DRAIN_S`, 20 s), then rank 1, archives both logs as `logs/glm53-spark-<time>.tar.gz`, and
only ever stops containers labelled `tf.recipe=glm53-spark`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `[E_WORKER]` | ssh, Docker or the NVIDIA runtime is not ready on the worker; the message names which |
| A thinking-off reply works the sum out in its answer and `finish_reason` is `length` on a short `max_tokens` | The prompt carries the template's `Reasoning Effort: Max` line (`EFFORT_LINE=always`, or a `CHAT_TEMPLATE` of your own); `EFFORT_LINE=thinking` (default) writes it with thinking on only. `/tokenize` on a one-line user message shows it: 7 tokens more |
| `[E_FABRIC]` | No directly reached address, or no RoCE v2 GID holding an IPv4 on the link's port: cable the Sparks on one subnet, or set `FABRIC_PEER` / `NCCL_IB_GID_INDEX` |
| `[W_FABRIC]` a rail left out | The second rail does not carry a full-size ping; check the cable, the twin's address and MTU |
| `[E_MEMORY]`, `[E_PORT]` | Less than `MIN_AVAIL_GIB` available (another rank, usually), or a port in use |
| `RankMismatch`, `RankNeverCame`, NCCL timeouts | The ranks were started differently, or rank 1 did not reach rank 0 within 300 s: check `rank1.log` and the ports; `NCCL_DEBUG=INFO` for detail |
| `vendor_err 0x81` | A rail without a path to its partner was set by hand through `TF_ROCE_HCA` |
| A long prompt dies in a RoCE gather | The gather has no wait setting at this pin; `ROCE=off` puts every exchange on NCCL |
| `--parallel N needs … GiB`, `[memguard]` | The window or stream count does not fit, or a Spark went under the floor: lower `CONTEXT`/`PARALLEL` or raise `MEMORY_RESERVE_GIB` |
| No prompt cache in the startup line, `cannot keep one state` | The page pool took a third window: raise `MEMORY_RESERVE_GIB` (the engine's own hint to lower it is the wrong direction here; see [Memory](#memory-on-a-128-gb-unified-spark)) |
| A prompt cache far under `PROMPT_CACHE_GIB`; `prompt cache: the room funds X GiB of the Y GiB` | The cap never binds: the cache is the remainder after the pool and the streams' fixed parts. Set `CACHE_FIRST=1`, or lower `PARALLEL` or `CONTEXT` (see [Memory](#memory-on-a-128-gb-unified-spark)). With `CACHE_FIRST=1` already on, the pool kept a whole window more than the cache left room for: raise `PROMPT_CACHE_GIB` until the page count drops by a window (issue 16) |
| `[E_CONFIG] NCCL_SOCKET_IFNAME must be a netdev name.`, or `[W_CONFIG] NCCL_SOCKET_IFNAME=^… is an NCCL exclusion list` | An exported value wins over `scripts/local.sh`. An exclusion list such as `^lo,docker` from `/etc/profile.d` is set aside with that warning and the link is found as when empty (issue 17); any other value must be one netdev: `NCCL_SOCKET_IFNAME=<netdev> ./start.sh`, or `env -u NCCL_SOCKET_IFNAME ./start.sh` |
| `git pull` will not fast-forward a clone from before 2026-11-04 | The history was rewritten that day, so a clone from then (such as `0049d7c`) shares no commit with `main`: take a fresh clone and carry `scripts/local.sh` and `.env` over (issue 16) |
| `[E_CACHE] Rank N funds a X GiB prompt cache, under PROMPT_CACHE_MIN_GIB` | That start's room funded less cache than the minimum (the room varies by a GiB or more between starts). Lower `PARALLEL` or `CONTEXT`; with `CACHE_FIRST=1`, a larger `PROMPT_CACHE_GIB` can fund more, because it can cost the streams a whole window (issue 16); or lower the minimum |
| An exact value (an id, a hash) copied wrong in a JSON reply on `ABLIT=1` | The Ablit weights' own error (issue 9): the published weights copy it right on the same engine and request, and the Ablit weights do too with thinking on. See [Ablit weights](#ablit-weights) |
| `libtfvideo.so is required for video inputs` (HTTP 500 on a `video_url` part) | The default image builds no video runtime. Set `WITH_VIDEO=1` and `./start.sh restart` (it builds its own image tag); a server that already refused one video request keeps refusing until restarted |
| Raw `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` text in an agent's reply (Hermes and others) | The client sent that turn without the tool the model called: no `tools`, `tool_choice: "none"` (an agent's final-summary turn after its iteration limit, typically), or a list without that tool. The server only turns calls to offered tools into `tool_calls`. Fix the client (raise its iteration limit, enable the toolset), or turn on the guard: [Hermes and other agents](#hermes-and-other-agents-the-tool-call-guard) |
| The first start takes long | Each rank hashes all 164 GiB once (cached in `CACHE_DIR`); raise `WAIT_TIMEOUT` |

## Ablit weights

`ABLIT=1` serves
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit),
the same checkpoint abliterated, at the revision pinned in `scripts/config.sh`. The repository is gated: `start.sh`
checks that `HF_TOKEN` reaches it before building or downloading. Thinking off, the recipe's default, is also how
these weights give their best results. One measured exception (issue 9): asked to return a record id
`LANTERN-ca6241dd2684` in schema-constrained JSON, the Ablit weights drop a digit, with drafting on or off; the
published weights copy it exactly on the same engine and request, and the Ablit weights do too with thinking on. For
replies that must copy ids, hashes or numbers exactly, serve `ABLIT=0` or turn thinking on for those requests.
DFlash2 accepts fewer drafts on these weights; measure them at your own memory settings before choosing a drafter.
One pair (issue 17) saw drafting turn into a net loss at `local.sh.example`'s `PARALLEL=4 MEMORY_RESERVE_GIB=12`, and
about 15% below the published weights (74.6% against 79.8% accepted) at the served `PARALLEL=8` with `CACHE_FIRST=1`.

## Credits

TensorFold and its native engine are by Ash Hart and contributors
([ashhart/TensorFold](https://github.com/ashhart/TensorFold)). The GLM-5.3-Flash EXL3 checkpoints are by Mia's AI Lab,
from zai-org's GLM-5.3-Flash. DFlash2 is by incoai. The two-host launch, the link discovery and the unified-memory
accounting are adapted from Mia's AI Lab's
[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(branch `zig-engine`), and the engine's RoCE transport is the GPU side of b12x's "RoCEnante" exchange
([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), Apache 2.0).

Patches 0103-0104 are ported from the lab's own [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold).
Full credits, including the authors of the carried upstream patches (Adam Hillson, ajensenwaud, Bizuayeu,
SvangenStudios, with their names in each patch's `From:` line), are in [CREDITS.md](CREDITS.md). Patch 0109 is a backport of upstream #1411, and 0110 (KV sharding) is toward upstream #1431, both by Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)). Required notices are
in [NOTICE.md](NOTICE.md).

## License

Apache License 2.0: see [LICENSE](LICENSE). Downloaded software and weights keep their own terms; DFlash2's weights are
CC BY-NC-ND 4.0.
