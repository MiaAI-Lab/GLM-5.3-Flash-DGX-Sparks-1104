# Consult: the engine's one-shot RoCE all-gather on a ring of four (decode on four Sparks)

Same read-only directory: `tf/` (the engine, now with the 107 patches incl. block FP8), `patches/`, plus two files
from the lab's older Python-engine recipe, for design reference only (their code is Python + a C proxy, not ours):
`ref/pairb-0099-roce-ring.patch` (TF_ROCE_RING=1: on a ring of four each rank writes its shard to its two
neighbours, whose proxies relay it to the rank across, half its stripes each way) and `ref/pairb-0098-ring-exchange.patch`.

## Where we are

GLM-5.3-Flash serves at TP=4 on four DGX Sparks cabled as a switchless ring (rank r cabled to r-1 and r+1 only;
ConnectX-7, two PCIe-twin RDMA devices a port). Measured greedy decode at one stream: EXL3 4-bit 56 tok/s prose
(37 ms a round), block-FP8 44.7 (48 ms a round); four streams 115-120 / 83-87. The pair (TP=2, one cable) runs the
engine's two-rank RoCE one-shot all-gather (zig/src/cuda/roce.zig, docs/roce-gather.md: 10.4 us a 4 KiB gather
against NCCL's 12.3; 17.4 vs 51.5 at 64 KiB) for the decode and verify windows' small gathers. At TP=4 `Group.openRoce`
returns at `world != 2` and every gather goes over NCCL ring all-gathers (NCCL_ALGO=Ring, graph files pin each channel
to the cables; the lab's older recipe measured ~109 us for a decode-sized all-gather on this ring). The decode round
has two gathers a layer (prof sections gather_a, gather_b) over 45 layers, plus the sampler's and the drafter's.
I am collecting a TF_GLM_PROF=1 breakdown of the FP8 round now; I expect the NCCL gathers to be a large share.

Upstream's `zig/src/fabric/tp/roce_multi*.zig` (3-8 rank RoCE gathers, a versioned kernel, per-peer queue pairs)
connects over a switchless ring of THREE (7eb8ffde3: each peer's stripes over the HCAs on its /24) but a ring of four
"still refuses and falls back to NCCL" (roce_multi_pairs.zig: a peer out of reach refuses the plan). It is the
fabric/tp harness's transport (TF_COMM_BACKEND), not wired into GLM's tp_group (tp_group.zig: "two ranks add RoCE small
gathers (roce.zig)").

## What I want designed (and checked)

A RoCE one-shot all-gather for GLM's decode-sized gathers on a ring of four, in the engine, so that both neighbours'
shards arrive by direct RDMA writes and the rank across the ring's shard arrives relayed by the two neighbours (half
each, as the lab's 0099 did in its C proxy), with the kernel-side protocol (stage, doorbell, proxy writes, flags, copy
out in rank order, device epoch for graph replay) kept as roce.zig has it. Please answer:

A. Which base to extend: (1) roce.zig's two-rank Gather generalised to a world of 4 with relays (region layout: a
   recv slot per source rank, flags per (source, hca), the proxy also forwarding a neighbour's stripes toward the
   far rank), or (2) tp_group taking fabric/tp's roce_multi with a ring-of-four plan where the far peer's stripes are
   relayed by the neighbours (what roce_multi's kernel expects: a flag per (source, stripe); does its wire allow a
   proxy to re-post a received stripe?), or (3) something cheaper: two two-rank roce.zig Gathers (one per cable) and a
   two-step ring pass (gather with the left neighbour, then with the right, like the lab's 0098 did for the split
   prefill over two-rank NCCL communicators) composed in Group.allGather for world 4. Give the latency you expect for
   a 16 KiB-a-rank gather in each design against NCCL's ~109 us, and the risk order.
B. The correctness traps: slot/epoch accounting with two inbound sources and a relay in flight (a rank's next gather
   can ring before the proxy saw the last: "peers stay within a slot's distance"), the relay's ordering (a relayed
   stripe's flag must land after its data on the far rank's queue pair), both neighbours relaying halves of the same
   shard, failure propagation (a rank that dies mid-relay), and the agreed cap on four ranks.
C. What the GLM side needs: Group.allGather's `fits` rule at world 4, TF_ROCE_HCA naming two ports (one per
   neighbour; today it names the two twins of one port), the GID per device, and whether the decode graph replay
   (device epoch) holds with two sources.
D. The test plan without the hardware (host tests with a fake wire as roce_multi_fake / roce_fake do), then the
   `tf-cuda-test roce-gather`-style device check on the four Sparks before the engine runs it.
E. Effort estimate for each design, and which you would pre-register as the first candidate.
Say what you are sure of and what you are guessing.
