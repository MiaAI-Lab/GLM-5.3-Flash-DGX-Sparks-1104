# Consult: GLM-5.3-Flash at TP=4 on four DGX Sparks cabled as a switchless ring, with this pinned TensorFold engine

You are read-only in a directory that holds: `tf/` (the engine tree: upstream TensorFold at f350cd3ea253 with this
recipe's 105 patches applied as commits on branch `port`; HEAD carries one extra commit, a killed candidate that is NOT
served, ignore it) and `patches/` (the same 105 patches as files, for grep). Answer in prose with file:line references.
I will take your answer as advice; measurements outrank it.

## The kit and what runs today

Four DGX Sparks (GB10, 128 GB unified each), one ConnectX-7 card each with two QSFP ports; every port shows up as two
netdevs and two RDMA devices (a PCIe twin: `enp1s0f0np0`/`rocep1s0f0` and `enP2p1s0f0np0`/`roceP2p1s0f0`). No switch.
Cabled as a ring, one /24 per cable, the twin addressed .N and .10N in the same /24:

    spark1 (rank 0) 10.0.22.1/.101 (f1) <-> 10.0.22.2/.102 (f1) spark2 (rank 1)
    spark2 (rank 1) 10.0.33.2/.102 (f0) <-> 10.0.33.3/.103 (f0) spark3 (rank 2)
    spark3 (rank 2) 10.0.34.3/.103 (f1) <-> 10.0.34.4/.104 (f1) spark4 (rank 3)
    spark4 (rank 3) 10.0.41.4/.104 (f0) <-> 10.0.41.1/.101 (f0) spark1 (rank 0)

Each Spark also has a /32 alias 10.0.0.N on lo, routed over the cables (ip_forward=1, rp_filter=0 everywhere), so TCP
between the two Sparks across the ring (spark1<->spark3, spark2<->spark4) goes through a neighbour's kernel. A LAN
(192.168.1.x, enP7s7, the default route) joins all four as well. MTU 9000 on every CX7 port. NCCL in the image: 2.30.7.

Today the engine serves at TP=2 on spark3+spark4 (one cable, both rails) with the recipe's start.sh: NCCL_COMM_ID on the
link, NCCL_SOCKET_IFNAME/NCCL_IB_HCA/NCCL_IB_GID_INDEX per rank, 4 channels, TF_ROCE=1 (the two-rank RoCE one-shot
all-gather), TF_KV_PAGES=on, --kv-cache fp8, --parallel 8, DFlash2 drafter at 4 bits.

## What I am building

`start-tp4.sh`: rank 0 on spark1 (the API), ranks 1-3 on spark2, spark3, spark4 in ring order; all four read the same
checkpoint from spark1 (rank 0 local, the workers over NFS). The networking follows the lab's sibling recipe that ran
TP=4 on this very ring (a different engine build, v1.0.7 + its own patches), which measured and documented:

- NCCL_ALGO=Ring (PAT or Tree would send two ranks away), plus a per-rank NCCL graph file whose channels name the RDMA
  device toward the PREVIOUS rank (where the node listens), alternating port and PCIe twin per channel, and
  NCCL_IB_SUBNET_AWARE_ROUTING=1 so each send goes out the device on the next rank's subnet; NCCL_CROSS_NIC=1,
  NCCL_IB_MERGE_NICS=0 (a fused two-port device sends part of a connection to a port without that peer),
  NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1, NCCL_IB_SUBNET_PREFIX_LEN=26 (keeps a port and its twin apart so both carry
  traffic), NCCL_IB_HCA = every device toward any peer in uverbs order, 4 channels.
- NCCL bootstrap socket (NCCL_SOCKET_IFNAME) on each node's LAN netdev; the rendezvous address (their --master, our
  NCCL_COMM_ID) = what spark1's hostname resolves to, which on this kit is the lo alias 10.0.0.1.
- Their engine had its own patches to pass split-prefill exchanges round the ring over separate two-rank communicators
  per cable and to relay the small RoCE all-gathers through the neighbours; this engine has neither, and I am not
  porting them. Their note: "the sending side cannot repair a receiver's choice" of device, hence graph files.

For this engine I plan: TF_GLM_WORLD=4, TF_GLM_PREFILL_REDUCE=gather (the engine's own neighbours-only mode; `ring`
is the other candidate, measured after), TF_ROCE=0 (tp_group.openRoce returns at world != 2 anyway), everything else
as at TP=2.

## What I found in the engine and want checked

1. `zig/src/families/glm/cuda_native.zig:138-154` `warmPeers`: at any world > 1, before the weights load, every rank
   posts a 4 MiB ncclSend+ncclRecv with EVERY other rank in one NCCL group ("every peer's send/recv and NCCL's rings
   connect before the weights load, so a full GB10 never registers them"). On this ring, rank 0 <-> rank 2 and
   rank 1 <-> rank 3 share no RoCE subnet. The later ring fix `38a6af1d3` ("the all-peer start probe runs for direct
   split only, so a ring start on a neighbours-only fabric loads", cuda_ranks.zig probeStart) was host-tested on a fake
   fabric and did not touch warmPeers. Question A: does warmPeers make a ring start impossible on this engine as
   pinned, and if so, what is the least-risk patch shape: (a) warm only the peers the configured reduce mode will
   exchange with (split: every peer; ring: prev and next; gather: none, or prev/next), or (b) warm all but treat a
   refused pair as non-fatal (log and go on)? Which keeps the ibv_reg_mr rationale intact for the pairs that matter?
   Could NCCL's p2p to the rank across the ring succeed through the neighbour's IP forwarding (routable RoCE v2: the
   kernel resolves the gateway MAC, but only if NCCL picks the local device whose netdev the route leaves on, and the
   receiver likewise), or is that a dead end to design around?
2. Other all-pairs assumptions at world 4 in the GLM family or the 105 patches: I see cuda_relay.zig (ring mode sends to
   next/prev only on the main communicator), cuda_sampler.zig:163 (ncclBroadcast from root q), cuda_follow.zig
   (broadcast from 0), dflash_block.zig / cuda_sparse.zig / cuda_forward.zig / cuda_engine_pick.zig (all-gathers),
   cuda_engine.zig:552 (all-reduce), cuda_p2p.zig (world 2 only), tp_watch (TCP lines to NCCL_COMM_ID host:port+1).
   Patches 0033/0063/0065 mention TP3 (comm3, prefill3). Please grep the patches for world == 2 / rank 1 assumptions
   that would misbehave at world 4 (prompt pair 0051, the PDL chain, warm turns 0104, the lane host 0093, the p2p
   gather 0034, the KDA chain) and for anything that posts ncclSend/ncclRecv to a non-neighbour.
3. The NCCL env above: anything wrong for this engine's use of NCCL (one communicator, ncclGroup'ed send/recv in ring
   mode, broadcasts with a non-zero root under NCCL_ALGO=Ring, NCCL graph file with one GPU a node)? Is
   NCCL_IB_SUBNET_AWARE_ROUTING honoured for ncclSend/ncclRecv p2p connections, or only for collectives' channels?
4. Bootstrap: with commit 035a9826d rank 0 makes the NCCL id from NCCL_COMM_ID and ships it over the ranks' TCP lines;
   followers that name NCCL_SOCKET_IFNAME unset NCCL_COMM_ID. If NCCL_COMM_ID names 10.0.0.1 (a /32 on lo, routed), rank
   0's own bootstrap interface is found by subnet match: does NCCL accept lo there? Would spark1's LAN address
   (192.168.1.143) be the safer NCCL_COMM_ID, given followers' NCCL_SOCKET_IFNAME = their LAN netdev? The lines to
   port+1 only need TCP reachability, which both addresses have.
5. Memory at TP=4: each rank holds a quarter of the routed experts (~41 GiB) plus the replicated parts; the KV pool is
   replicated per rank ("the per-token KV cost is the same on every rank"). At TP=2 the recipe needed
   TENSORFOLD_MEMORY_RESERVE_GIB=9.5 because core/kv_pool.zig's planner reserves whole context windows before the
   prompt cache (cache = min(want, left - held)), and crossing the threshold for a third 1M window zeroed the prompt
   cache. At TP=4 the room is ~55 GiB more: what does the planner do with it at --parallel 8 (how many windows, what is
   left for --prompt-cache-gib), and which knob keeps a few GiB of prompt cache: MEMORY_RESERVE_GIB, PROMPT_CACHE_GIB
   (which only caps), or a smaller --context? Name the log lines to read on rank 0.
6. Anything else you would pre-register as a GO/KILL gate for the first TP=4 boot (what to grep in the four ranks'
   logs, what the first greedy smoke should hash-match against: the TP=2 token_sha of the same request is not expected
   to match across world sizes, correct?).

Keep it concrete; say what you are sure of and what you are guessing.
