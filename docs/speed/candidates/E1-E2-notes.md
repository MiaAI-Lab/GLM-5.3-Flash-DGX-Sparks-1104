# E1 / E2: the four-rank RoCE all-gather and the drafter MLP split at world N

Two engine patches for TP=4 decode on a switch (mesh), written 2026-11-04 against `f350cd3e` + patches 0001-0106,
not yet built or run on hardware (no Zig toolchain where they were written). They are candidates: the microbench's
GO gate decides whether 0107 goes on at four ranks, the hashes decide whether either ships.

| Patch | What | Expected on the 36.4 ms round (one stream) |
| --- | --- | --- |
| `patches/engine/0107-cuda-roce-...patch` | `roce.zig` / `roce_gather.cu` at any world on a mesh (per-peer RC queue pairs, a receive slot and a flag a peer an HCA, the blob exchange as a group all-gather, a connect probe that refuses an unreachable rank) | the ~90 gathers of a verify call, 4.7 ms on NCCL, at the two-rank path's 10-18 us each: about -3 ms (-8%) |
| `patches/engine/0108-glm-cuda-...patch` | `dflash.mlpSplit(world)`: gate/up rows and down columns at `rank * FF / world`, `df_sumn` sums the fp32 partials in rank order; the gather rides 0107 | the drafter block 5.5 ms (8.7 at four streams) toward a quarter of the MLP work: -1 to -2 ms |

Both apply with `git apply` after 0106 (checked by replaying all 108 onto a fresh `f350cd3e`: the tree equals the
branch they were cut from).

## E1 design (0107)

- A rank's peers are the other ranks in rank order: `peerIndex(rank, peer)` / `senderOf(rank, p)`. Every rank opens one
  RC queue pair a peer an HCA (`Hca.qp[peer]`), one completion queue an HCA.
- Region: `recv[slot][world-1][slot_bytes]`, `flags[slot][world-1][n_hca]` (128 B apart), `send[slot]`, `ctrl`. At world 2
  every offset is the old two-rank layout's (a unit test pins it to the old formula), so two Sparks keep bits and speed.
- `post()`: per HCA a stripe of the shard, written to every peer on this rank's queue pair toward it, the sequence flag
  behind it on the same queue pair (never ahead of its stripe). Credits: at most `send_depth / 4` signalled writes
  outstanding an HCA across its queue pairs (each with its unsignalled stripe ahead: at most `send_depth / 2` entries on
  one queue pair of depth `send_depth`).
- Kernel: thread `tid < (world-1) * n_hca` waits for flag `[slot][tid / n_hca][tid % n_hca]`; every block copies its own
  shard and each peer's stripes into the output slot of the sender's rank (ncclAllGather's layout). A timed-out wait
  poisons the exchange and records the HCA and the rank whose stripe never came (`ctrl[C_PEER]`).
- Setup over the group's NCCL communicator (`Group.gather`, already N-rank): blobs (region address, rkeys, per-peer qpns,
  GIDs), then "queue pairs in RTS" from every rank, then the connect probe, then the probe's verdict. The cap is the
  minimum over every rank's. `fits()`, `TF_ROCE`, `TF_ROCE_MAX_KB`, `TF_ROCE_HCA`, `TF_ROCE_GID_INDEX` unchanged.
  `max_world` is 8.
- Connect probe (fail closed): RC `modify_qp` succeeds whether or not a cable reaches the peer, so once every rank's
  queue pairs are in RTS each rank RDMA-writes a token (`rank + 1`) to every peer on every HCA (`ctrl[c_probe + peer *
  2 + hca]`) and waits (about 5 s, 20 us naps) for its writes' completions and every peer's token. On a ring of four the
  far rank's write ends in a retry-exhausted completion (or its token never comes); every rank then returns
  `error.PeerUnreachable` and logs `REFUSE TF_ROCE: rank N has no RDMA path to every other rank ... set TF_ROCE=0`,
  before the weights load. The proxy thread starts only after the probe (no completion queue shared across threads).
- PDL: the RoCE kernel is launched plainly after the kernel that writes its input (stream order), exactly as before;
  the decode chain's dependent launches (0069-0071: `glm_mm_reduce`, hc, the p2p gather) are untouched.
- Exactness: a data move; every rank's bytes land in its rank's slot. Tokens cannot change.

## E2 design (0108)

- `dflash.mlpSplit(world)`: split when `FF % world == 0` and `FF / world` is 256-aligned (2: 6144 as before, 3: 4096,
  4: 3072, 6, 8, 12, 16); otherwise the MLP runs whole (world 1, 5, 7, ...). `rawBytes`, the weight estimate, the loader's
  slices (`rank * ff`), `df_swiglu`'s grid (`cdiv(ff, 1024)`) and the partials buffer `[world][MAX_M][D]` follow it.
- `df_sumn(g, out, R, world)` replaces `df_sum2`: `g[0] + g[1] + ... + g[world-1]` in rank order, one fp32 add a rank,
  one bf16 rounding; at world 2 it is `df_sum2`'s single add on the same operands (same bits).
- The extra gather (`M * D * 4` bytes a rank: 128 KiB at one stream, 1 MiB at eight) goes through `Group.allGather`,
  so it takes the RoCE path when 0107 is on (cap 4 MiB by default), NCCL otherwise.
- Exactness: drafts only propose; the target verifies, so the verified tokens cannot change. The drafter's own
  arithmetic changes (K-slices summed in a different order), so candidates may differ in a logit's last bit and the
  acceptance rate may move by a fraction of a percent.
- The launch record for `df_sumn` carries `df_sum2`'s register figures; refresh with `tf-cuda-test contract-limits`.

## Tests that exist

| Where | What | Runs |
| --- | --- | --- |
| `zig/src/cuda/roce.zig` | two-rank layout == the old formula; four-rank slots and flags do not overlap and sit in the sender's rank order, both sides agree on every peer number; cap = min over four; fabric contract (mesh of four reaches all, ring of four leaves rank r without r+2, two on a cable are neighbours) | `zig build test` (cuda module) |
| `zig/src/families/glm/cuda_ranks_test.zig` | the same contract on the fake `Fabric` (switched / ring) | `zig build test` (glm module) |
| `zig/glm_dflash_host.zig` | `mlpSplit` at every world 1-16 (halves, quarters, 4096 at three, whole at 5 and 7, every slice inside FF, shares sum to FF, 256-aligned); the raw scratch follows it; every `df_*` kernel has a launch record | `zig build test-glm-dflash-host` |
| `tf-cuda-test roce-gather <rank> [world]` | on real ranks: every size against `ncclAllGather` byte for byte, a captured graph replayed 64x, 3,000 mixed-size calls (16/32/64 KiB a rank, tiny between) each against NCCL's rows, time a call at 4-64 KiB; last line `RESULT roce-gather world N rank R: 32 KiB a rank RoCE X us a call (GO at most 25.0), 3000 mixed-size calls, M mismatch(es) (GO 0)` | the microbench below |

None of these has been executed yet: no Zig 0.17 toolchain was available where the patches were written. The first
image build (`zig build install native grammar`) is the compile check; `zig build test` inside the build stage (or a
`docker run --entrypoint zig ... build test` is not possible: the image carries no compiler) is owed on a Spark.

## The microbench inside the image (the GO gate for ROCE=on at four)

Build the image with 0107-0108 in `patches/engine/` (the launcher's `PATCHES_HASH` changes, so `./start-tp4.sh` with
`DRY_RUN=1` or `scripts/prepare.sh`'s build step produces `tensorfold-glm53-spark:<hash>`; ship it to the workers as the
launcher does). Then, with the model server **stopped** (`./stop-tp4.sh`; the bench takes the GPUs and the NICs), one
container per Spark, mirroring `start-tp4.sh`'s rank env (`scripts/tp4.sh tp4_rank_env`) with `tf-cuda-test` as the
entrypoint instead of `tensorfold-native serve`:

```bash
# on every Spark; RANK 0 = this Spark (6f90), 1 = de02, 2 = 8fe0, 3 = 8458 as start-tp4.sh numbers them
IMAGE=tensorfold-glm53-spark:<hash>      # the image built with 0107-0108
MASTER_ADDR=192.168.120.10               # rank 0's fabric address (the launcher's MASTER_ADDR), port 29583 as the launcher's
RANK=0                                   # 0..3
docker run --rm --name tf-roce-bench --gpus all --ipc=host --network host --shm-size 16g --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband \
  -e TF_CUDA_DEVICE=0 -e "TF_GLM_RANK=$RANK" -e TF_GLM_WORLD=4 -e "NCCL_COMM_ID=$MASTER_ADDR:29583" \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e TF_ROCE_GID_INDEX=3 \
  -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 \
  $([[ $RANK == 0 ]] && printf -- '-e NCCL_SOCKET_IFNAME=enp1s0f0np0') \
  --entrypoint tf-cuda-test "$IMAGE" roce-gather "$RANK" 4
```

(`TF_ROCE` unset = on; `TF_ROCE_HCA` falls back to `NCCL_IB_HCA`, in that order on every node, so HCA 0 is the 120.x
rail and HCA 1 the 121.x rail on all four. Start rank 0 first or within NCCL's bootstrap timeout of the others.)
Expected: four `PASS roce-gather rank R of 4:` lines a rank and one `RESULT` line; the whole run is a minute or two.

**Pre-registered gates**

| Gate | Pass | Kill |
| --- | --- | --- |
| GO-A (bytes) | `0 mismatch(es)` on every rank over the 3,000 mixed calls, every size case and the graph replays | any mismatch, any `FAIL`, a rank that does not finish |
| GO-B (time) | RoCE at 32 KiB a rank <= 25 us a call on every rank (NCCL's figure printed beside it for the record) | > 25 us: do not turn `ROCE=on` at four; the path is then not faster than NCCL's LL at this size and world (record the numbers anyway) |
| GO-C (fail closed, optional, needs a ring cabling or a blocked pair) | a rank without a path to another makes every rank print `REFUSE TF_ROCE: ...` and exit non-zero within ~10 s, no hang | a hang or a kernel-spin timeout instead |
| Model boot (after GO-A and GO-B) | with a launcher change that passes `TF_ROCE=1` on `FABRIC=mesh` (not in this change): `RoCE all-gather: 4 ranks, 2 HCA(s), shards up to 4096 KiB by RDMA writes` on every rank's log; acc.py greedy hashes `fab1d3b6894b` (prose) / `e71492bf766b` (code) on the Ablit checkpoint, drafts on, split prefill, identical over two runs; `TF_GLM_PROF=1` gather_a + gather_b under 4.7 ms a call and the drafter block under 5.5 ms a round | any other hash = a bug in 0107 or 0108, revert that patch; a `RoCE all-gather failed (sequence ...)` line; 4- or 8-stream outputs that differ from 1-stream |
| E2 alone (a boot with `TF_ROCE=0`, both patches in) | hashes unchanged; acc.py `accepted` within 1 point of 53% / 72%; drafter block ms a round below 5.5 / 8.7 | hashes changed (E2 touched the target: bug); acceptance down by more than ~2 points (then the split costs more than it saves at these stream counts) |

Record the numbers in the benchmarks folder with the image hash and both patches' sha256 (BUILD-INFO lists them).

## Not verified here (needs the toolchain or the hardware)

- Compilation of every changed file under Zig 0.17.0 and nvcc 13.3 (read for correctness only; the idioms follow the
  files they live in).
- The RC credit accounting across several queue pairs on one completion queue under sustained load (the bench's 2,000
  back-to-back calls a size and the 3,000 mixed calls are the test).
- The probe's wall time on a real ring (RC retry: `timeout 14`, `retry_cnt 7`, about 0.5-1 s to a retry-exhausted
  completion; the 5 s wait is the ceiling).
- Whether the flag wait with 6 waiting threads (3 peers x 2 HCAs) costs anything against 2: the kernel's time at
  32 KiB is the measurement.
- `df_sumn`'s register usage (the record carries `df_sum2`'s 16 / 38).
- The GB10 numbers themselves: the two-rank path's 10.4-17.4 us were measured on two Sparks with one cable; four on a
  switch with a 3-way incast at each port is a different fabric, hence the gate.

## What this change does not do

- It does not touch `start-tp4.sh` / `scripts/tp4.sh`: they still pass `TF_ROCE=0` at four. A separate launcher change
  adds `ROCE=on` for `FABRIC=mesh` (and keeps it off on a ring) once GO-A and GO-B pass.
- It does not change the README's patch table or the CHANGELOG (that goes with the launcher change, with the numbers).
- `cuda_p2p.zig` (two GPUs on one host, peer stores) keeps its `world != 2` check: it is not this path.
