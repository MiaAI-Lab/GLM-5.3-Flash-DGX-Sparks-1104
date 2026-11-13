**I would treat the first boot as KILL pending a small transport test.** Your `warmPeers` concern is valid, but there is another likely blocker: the engine defaults to eager NCCL connection setup, which can connect Tree/PAT edges even with `NCCL_ALGO=Ring`.

This is a static review of the supplied files and 105 patches. The supplied copy has no Git metadata, so I could not independently verify commit ancestry or isolate HEAD’s extra commit.

1. **`warmPeers`: choose (a), with no explicit p2p warm-up for `gather`.**

   The call is unconditional at world > 1, before `Engine.init`; it posts both directions to every peer and synchronizes. Failure propagates out of startup. The later `probeStart` fix therefore does not protect this entry point. See [cuda_native.zig:138](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:138), [cuda_native.zig:225](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:225), and [cuda_ranks.zig:329](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_ranks.zig:329).

   On a genuinely neighbours-only RDMA fabric, that is a startup blocker. TCP reachability to the opposite rank does not establish that these RDMA connections work.

   My least-risk patch shape would be:

   - Direct `split`: retain the existing all-peer, large-message warm-up.
   - Engine `ring`: warm the actual directed exchange—send to next, receive from previous. Avoid adding reverse-direction connections unnecessarily.
   - `gather`: post no explicit `ncclSend`/`ncclRecv`; warm the Ring collectives instead.

   Put the mode-dependent warm-up **after the existing fingerprint agreement and before weight loading**. Otherwise mismatched modes can produce mismatched postings before the engine gets to report `RankMismatch`. That insertion point exists at [cuda_engine.zig:168](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:168), before the weight load at line 203.

   This retains the registration rationale: establish the connections actually used while memory is available. For `gather`, a representative large AllGather before loading is preferable to warming unrelated p2p connections. Keep synchronization and fail startup on a required connection’s failure.

   I would reject (b). A failed grouped NCCL operation is not a reliable “this pair is unavailable” capability query: it can hang or leave communicator/stream work unsuitable for continuation. Logging and proceeding also abandons the early-registration guarantee.

   **Routed RoCEv2 is possible in principle**, including through an IP router; it is not inherently restricted to one subnet. [NVIDIA’s RoCE documentation](https://networking-docs.nvidia.com/mlnxofedswum/586042lts/rdma-over-converged-ethernet-roce) establishes that. Your proposed gateway-MAC mechanism is plausible, but success requires routing to the selected **port GIDs**, correct source devices and return paths, and working transit forwarding. Routes to `10.0.0.N` alone do not prove those conditions. I would design the first boot around neighbour traffic and qualify opposite-rank RoCE separately with explicitly selected GIDs/devices, then NCCL.

2. **The remaining GLM/patch audit mostly supports your inventory.**

   I found these application-level all-peer p2p sites: `warmPeers`, direct split’s startup probe, and direct split’s partial/glued exchanges. DFlash tap sharing also reaches the direct glued exchange when direct split is selected. Engine ring uses next/previous; gather uses NCCL AllGather. See [cuda_relay.zig:31](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_relay.zig:31), [cuda_relay.zig:83](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_relay.zig:83), and [cuda_forward_pieces.zig:98](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_forward_pieces.zig:98). The 105 patch files contain no literal `ncclSend` or `ncclRecv` calls.

   The specific patches you named look as follows:

   - **0034 and the PDL p2p chain:** the two-rank peer-store implementation is guarded out at world 4. Copy-engine exchange from **0035/0063** is likewise guarded out. See [cuda_p2p.zig:34](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_p2p.zig:34), [cuda_ce.zig:70](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_ce.zig:70), and [tp_group.zig:59](/home/zurih/c1bench/tp4/tf/zig/src/cuda/tp_group.zig:59).
   - **0051 prompt pair:** “pair” means two prompt chunks. It propagates rank 0’s indexer-split decision to both forwards; the surrounding code uses `world`, including minimum rows per rank. Gather gets its own twin exchange room. See [cuda_prompt_pair.zig:52](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_prompt_pair.zig:52), [cuda_engine.zig:226](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:226), and [patch 0051:42](/home/zurih/c1bench/tp4/patches/0051-glm-cuda-a-paired-step-takes-rank-0-s-indexer-split-.patch:42).
   - **0033/0065 “TP3”:** these repair host-test fixtures. The current test actually covers worlds 1, 2, 3, **4**, 5 and 8. “comm3/prefill3” in 0063 are development-topic names, not three-rank restrictions. See [cuda_kda_tp3_test.zig:61](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_kda_tp3_test.zig:61).
   - **PDL reductions and KDA:** I found no extra inter-rank p2p exchange. Reduction kernels iterate over `world`; KDA derives its dimensions from the local head count. TP=4 has 16 KDA heads per rank. See [glm_glue_rows.cuh:33](/home/zurih/c1bench/tp4/tf/zig/kernels/cuda/glm_glue_rows.cuh:33) and [cuda_kda_block.zig:115](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_kda_block.zig:115).
   - **0093 lane host / 0104 warm turns:** grammar lifetime and scheduling/cancellation changes, respectively. Neither adds a rank-1 transport assumption. Warm cancellation enters the existing all-rank gate. See [patch 0093:15](/home/zurih/c1bench/tp4/patches/0093-lanes-a-reply-grammar-is-freed-when-its-stream-cannot-start.patch:15), [patch 0104:147](/home/zurih/c1bench/tp4/patches/0104-glm-warm-turns-the-next-user-message-resumes-past-the-agent-loop.patch:147), and [cuda_engine.zig:525](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:525).

   One material world-2 special case remains: **DFlash2 splits its MLP only at world 2**. At world 4 it runs the full MLP per rank; its two-input `sum2` path is bypassed, while vocabulary candidates still gather across the world. That appears intentional, but costs additional replicated memory/computation. See [dflash_load.zig:161](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/dflash_load.zig:161) and [dflash_block.zig:161](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/dflash_block.zig:161).

   I found no further reachable, unguarded TP=2 assumption in those CUDA paths. That is source evidence, not TP=4 device qualification.

3. **The NCCL plan fits collectives better than explicit p2p—and eager initialization needs attention.**

   **The additional blocker:** unless overridden by the launcher, this engine sets `NCCL_RUNTIME_CONNECT=0`, `NCCL_CUMEM_HOST_ENABLE=0`, and `NCCL_CUMEM_ENABLE=0`. See [cuda_native.zig:205](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:205).

   In upstream NCCL 2.30.7, `runtimeConn` requires both cuMem support and runtime-connect enabled. Its eager branch connects rings, trees, and—on one GPU per node—PAT. These calls are not conditioned on `NCCL_ALGO=Ring`. Thus **fixing `warmPeers` alone may still fail inside `ncclCommInitRank`**. Simply exporting runtime-connect 1 while retaining cuMem 0 is insufficient. See [NCCL init.cc:957 and 1425](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/init.cc#L957) and [generic.cc:45](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/generic.cc#L45).

   I would first test the exact image under the proposed environment. If this eager path blocks, one experiment is explicit `NCCL_RUNTIME_CONNECT=1`, `NCCL_CUMEM_ENABLE=1`, retaining `NCCL_CUMEM_HOST_ENABLE=0`, plus pre-weight collective warm-up. That revisits an existing GB10 workaround and needs qualification. Preserving cuMem 0 instead requires an NCCL-side solution that avoids unused connection setup.

   Otherwise:

   - **One communicator and grouped neighbour send/recv are valid.** `NCCL_P2P_DISABLE=1` disables CUDA peer-access transport; it does not prohibit the `ncclSend` API.
   - **Non-zero Broadcast roots work with Ring.** NCCL forwards through the same ring from the selected root. The sampler’s broadcasts from roots 0–3 therefore do not inherently require opposite-rank RDMA edges. See [cuda_sampler.zig:160](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_sampler.zig:160) and [NCCL broadcast.h:11](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/broadcast.h#L11).
   - **One-GPU graph files are meaningful.** Use the local GPU device identity, and verify the resulting global ring order on every channel. With one GPU, NCCL’s graph device selection uses the graph’s first network endpoint for that rank. Explicit p2p setup instead uses a null graph and local-NIC selection, so your graph does **not** pin its receiver NIC. See [NCCL search.cc:1307](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/graph/search.cc#L1307) and [group.cc:127](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/group.cc#L127).
   - **Subnet-aware routing applies to p2p too.** It runs in the IB connection layer, below collective/p2p distinction. It searches for a device matching the peer GIDs; without a match it retains the default device. It does not select an IP next hop. This tag also contains an accept-side remapping step, so the sibling note is not a universal description of all receiver behavior. None of that guarantees neighbour p2p NIC selection. See [NCCL connect.cc:483, 734 and 1344](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/net_ib/connect.cc#L483).

   Your `/26` separates `.N` from `.10N` as intended without changing the actual `/24` addresses. Keep merge-NICs disabled for this fabric. Verify each device’s selected IPv4 RoCEv2 GID, and map graph NIC numbers from NCCL’s actual enumeration: `NCCL_IB_HCA` is a filter, not a promise that list order defines numbering. [NCCL HCA selection documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca).

   **I would start with `gather`.** Engine `ring` needs its own p2p connection test even after the collective ring works.

4. **Bootstrap: `10.0.0.1` can work; the LAN address is the simpler first-boot choice.**

   Rank 0 retains `NCCL_COMM_ID`; followers naming `NCCL_SOCKET_IFNAME` unset it before NCCL initialization and receive rank 0’s ID over `tp_watch`. Your reading is correct. See [cuda_native.zig:210](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:210).

   NCCL’s explicit COMM_ID path performs subnet matching. That matcher does not exclude `lo`: spark1’s `10.0.0.1/32` can match itself and be advertised. This differs from automatic interface selection’s usual loopback exclusion. See [NCCL bootstrap.cc:103](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/bootstrap.cc#L103) and [linux.cc:459](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/os/linux.cc#L459).

   I would nevertheless use **`NCCL_COMM_ID=192.168.1.143:PORT`**, with exact LAN-interface selection on all ranks. It makes bootstrap independent of the routed ring.

   Use a **literal IPv4 address**: `tp_watch` parses the host as an IP address rather than resolving a hostname. Resolve spark1’s hostname in the launcher if necessary. Its port+1 listener binds that address specifically. See [tp_watch.zig:38](/home/zurih/c1bench/tp4/tf/zig/src/cuda/tp_watch.zig:38).

5. **Memory: with the stated extra room, expect eight windows and probably the full default device cache.**

   Assuming “1M” means `--context 1048576`, the full backbone, FP8 KV and DFlash2 without target-MTP caches:

   Each rank uses **6,380 KV bytes/token**, independent of TP size. Pages contain 512 rows; including the verify allowance, one window takes **2,049 pages, about 6.2335 GiB**. Eight windows take **16,392 pages, about 49.8681 GiB**. See [cuda_pages.zig:15](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_pages.zig:15), [cuda_state.zig:77](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_state.zig:77), and [cuda_latent.zig:8](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_latent.zig:8).

   Let `R` be the limiting post-load room, `W` one rounded window’s bytes, and `F` one additional stream’s fixed allocation. With eight admitted lanes, the planner computes the window count as:

   `k = min(8, 1 + floor((R − W)/(W + F)))`.

   It then gives the cache at most `want`, after the first window, seven fixed parts, and the remaining held windows. That is the sawtooth you observed. **There is no ninth-window threshold at parallel 8.** See [kv_pool.zig:73](/home/zurih/c1bench/tp4/tf/zig/src/core/kv_pool.zig:73) and [kv_pool.zig:98](/home/zurih/c1bench/tp4/tf/zig/src/core/kv_pool.zig:98).

   For the standard TP=4 layout, I calculate `F ≈ 0.1665 GiB`, including DFlash context rings. Thus eight windows plus seven fixed parts cost approximately **51.03 GiB** of `R`. At that point, device cache is approximately `min(want, R − 51.03 GiB)`: about 4 GiB at `R=55`, 9 GiB at `R=60`, and the full default 16 GiB once `R≈67.03`. The default cap comes from [cuda_memory.zig:171](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda_memory.zig:171).

   If TP=4 really adds roughly **55 GiB to the TP=2 room**, I expect eight windows and the full default cache. Exact room needs the logs; weights alone cannot establish it.

   Keep reserve 9.5 initially. `--prompt-cache-gib` **caps** cache; it cannot force a reservation ahead of windows. A smaller context reduces window cost. Reserve changes move `R`, with sawtooth effects before eight windows; after eight, additional room can increase cache normally. Rank 0 uses the smallest rank’s reported room: [cuda.zig:363](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda.zig:363).

   Read rank 0’s `CUDA sm_...: model ...; 8 streams at once, ... pages ... of ... GiB left after ... reserve`, the `prompt cache: ... device pages ... pinned host memory a rank` line, and any `GLM CUDA prompt cache budget ... cannot keep one state` warning. Their sources are [cuda_startup.zig:2](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda_startup.zig:2), [cuda_native.zig:367](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:367), and [cuda_kv_pool.zig:328](/home/zurih/c1bench/tp4/tf/zig/src/native/cuda_kv_pool.zig:328). Account for pinned host cache too on unified memory.

6. **My first-boot GO/KILL gates would be these.**

   Before weights, run a bounded four-rank harness with the **exact library and environment**: communicator initialization, small and large AllGather, Broadcast roots 0–3, and one-double MAX AllReduce. Add grouped send-next/recv-previous before qualifying engine `ring`. This catches eager Tree/PAT setup and p2p NIC selection without an expensive NFS load.

   Collect `NCCL_DEBUG=INFO` with `INIT,BOOTSTRAP,GRAPH,NET,ENV` on all four ranks. Check actual channel rank order, NIC/GID mapping, four collective channels, `Subnet-aware routing: overriding dev`, and data channels using `NET/IB`. Bootstrap/proxy TCP is expected. KILL on unintended opposite-rank RDMA connections, data-channel socket fallback, registration/QP/completion errors, or stalled progress.

   Engine-side, grep for `REFUSE rank settings`, `RankMismatch`, `SplitUnreachable`, `RankNeverCame`, `RankNeverReady`, `CheckpointBytesDiffer`, `following rank 0 stopped`, allocation failures, and `kv pool idle (UNEVEN)`. Require the planned eight streams/cache budget and successful repeated requests. The existing fake-fabric test only exercises `probeStart`; it does not cover native warm-up or NCCL initialization: [cuda_ranks_test.zig:159](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_ranks_test.zig:159).

   For correctness, establish a **TP=4 gather, greedy, request-level `draft:false`, cache-cold baseline**. Require repeatability, then exact token hashes for DFlash2 on versus off, solo versus concurrent, and fresh versus resumed/cache paths. Exercise prompts beyond the pairing/index-split thresholds; add a sampled or structured-output request because ordinary greedy uses top-1 AllGather and does not exercise the sampler’s root-by-root broadcasts. See [cuda_engine.zig:713](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_engine.zig:713).

   **Correct: TP=2’s `token_sha` should not be an exact TP=4 acceptance gate.** Different partitioning changes floating-point grouping. Equality may happen; it is not guaranteed. Keep TP=2 as a fidelity comparison, investigate substantial differences, and use the TP=4 gather baseline when later testing engine `ring`.