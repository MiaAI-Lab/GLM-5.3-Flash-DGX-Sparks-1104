I would pre-register **(1): extend the engine’s existing one-shot protocol for an explicitly enabled ring of four**, preferably in a sibling module that preserves the TP=2 path. **(3) is the lowest-risk backup.** Design (2) has a suitable kernel and stronger existing transport tests, but requires substantial wire, bootstrap, and engine integration work.

The code findings below are checked against this tree. Latency and effort numbers are estimates. I made no changes and ran only an in-memory routing check, not Zig or CUDA tests.

**A. Base choice and expected latency.** For **16 KiB per rank—64 KiB of output per rank**:

| Design | Expected warmed gather latency | Delivery risk | Implementation + host tests |
|---|---:|---|---:|
| **1. Engine `roce.zig` protocol with relays** | **20–40 µs**, central guess ≈30 µs | Medium | 4–7 person-days |
| **2. `roce_multi` kernel/transport adapted for relays** | **20–40 µs** with the same relay schedule | Highest integration risk | 6–10 person-days |
| **3. Two matched two-rank gathers** | **30–50 µs** | Lowest | 2–4 person-days |
| Existing NCCL reference | ≈109 µs, supplied lab measurement | Baseline | — |

Allow another **1–3 hardware days** for qualification and tuning, assuming normal access to all four machines.

These ranges assume hot proxies, correct port selection, and serialized graph execution. The TP=2 measurements anchor launch/staging costs, but use two twins of one port; they do not establish performance with one selected device per cable. Designs (1) and (2) have essentially the same network work, so I would not predict an inherent latency advantage for either.

For **(1)**, retain the current sequence:

`stage → doorbell → proxy writes → wait for flags → rank-order copy → device epoch`

Extend the receive layout, routing, polling, and proxy state. This is already on GLM’s `Group.allGather` path, including the sampler and drafter calls that use that group. The current dispatch and setup boundaries are in [tp_group.zig](/home/zurih/c1bench/tp4/tf/zig/src/cuda/tp_group.zig:64).

For **(2)**, the kernel can plausibly remain unchanged: it already waits on flags indexed by **original source, slot, and logical stripe**, and copies each source into rank order. See [roce_device.cuh](/home/zurih/c1bench/tp4/tf/zig/src/fabric/tp/roce_device.cuh:180).

**The existing wire API cannot express forwarding.** Its `post(slot, seq, padded)` broadcasts its own send slot, identifies itself as the source, and requires a direct QP to every destination. Its guard also constructs local addresses from `sendAt(slot)`. Supporting relays requires an operation equivalent to:

```text
post_range(original_source, destination_neighbour,
           local_receive_offset, remote_receive_offset,
           length, slot, sequence, logical_stripe)
```

It also needs a relay scheduler and relay completion accounting. Merely allowing an unreachable peer through `plan()` would leave invalid QP accesses or missing flags. The existing refusal is deliberate and tested in [roce_multi_pairs.zig](/home/zurih/c1bench/tp4/tf/zig/src/fabric/tp/roce_multi_pairs.zig:97).

Integration adds another cost: `roce_multi.init` owns a fabric `Bootstrap`, while GLM owns an NCCL `Group`; the fabric TP module itself imports CUDA. Use a narrow adapter or dependency inversion, rather than transplanting the whole harness/session. The versioned mailbox kernel also needs to become an engine build asset.

For **(3)**, use **alternating matchings**, not “every rank calls left, then right”:

| Pass | Concurrent pairs | Bytes contributed per rank | Result |
|---|---|---:|---|
| 1 | `(0,1)` and `(2,3)` | `S` | Ordered pairs `[0,1]` or `[2,3]` |
| 2 | `(0,3)` and `(1,2)` | `2S` | `[0,1,2,3]` everywhere |

With ascending local pair ranks, both second-pass pairs put `[0,1]` before `[2,3]`; **no reorder kernel is needed**. Each rank needs a `2S` temporary buffer and two gather instances. Their endpoint records can be exchanged through the existing four-rank NCCL communicator; separate NCCL pair communicators are not intrinsically necessary.

This reuses the two-rank device protocol but pays for two launches and two host-memory staging/copy cycles. It is a simpler all-gather than the general message-routing machinery in reference 0098.

**B. Correctness protocol.** I would use this layout, with `B` the agreed, page-rounded slot size:

```text
send[2][B]
recv[original_source=4][2][B]
flags[original_source=4][2][logical_half=2]  // 128-byte spacing
metadata beside each flag                  // payload length
ctrl                                      // doorbell, sizes, abort, failure record
device state                              // epoch, poison, arrival counters
```

The important distinction is **logical half versus physical HCA**. With one HCA per neighbour, both halves of a direct neighbour’s shard travel over the same physical QP. The far shard’s two halves arrive over different neighbours.

For rank `r`, let `L=r−1`, `R=r+1`, modulo four:

- Send the **entire local shard to both L and R**.
- Forward half 0 received from L onward to R.
- Forward half 1 received from R onward to L.
- Preserve the original source in the destination slot and flag address.

Thus every rank waits for **six flags: two from each of three original sources**. Each rank sends `2S` of its own data plus `S` of forwarded data. This is the same aggregate payload volume as a conventional ring all-gather; the expected gain comes from fewer serial stages and transport overhead.

With `P=S/16` packs, use `ceil(P/2)` packs in half 0 and the remainder in half 1. At `S=16`, half 1 is empty but **still needs its sequence flag**.

The traps I would explicitly address are:

1. **Separate posting, forwarding, completion, and device progress.** Keep `last_local_posted`, `last_relay_posted[entry]`, and completion state independently. The doorbell can be two operations ahead; recover both sizes from their respective slots. A pending distance greater than two is a poisoned protocol, not something to catch up through.

   Poll relay arrivals even when the local doorbell has not changed. If forwarding runs only upon a new local gather, the first operation can deadlock: the GPU waits for the far shard while the proxy waits for the next doorbell.

2. **Carry reference 0099’s retirement rule.** Before posting local sequence `s`, finish forwarding sequences through `s−1` and reap their signalled completions. Continue servicing relay and CQ progress while waiting.

   This supplies a useful local lifetime proof: a neighbour cannot start `s+1` until it receives this rank’s `s`; withholding `s` until relay reads of `s−1` finish prevents that neighbour from overwriting the parity slot for `s−1`. The guard is visible in [0099](/home/zurih/c1bench/tp4/ref/pairb-0099-roce-ring.patch:424).

   **Posted is not retired.** A non-inline payload remains owned by the NIC until completion; inline metadata can use stack storage after `post_send` returns. See the [rdma-core buffer ownership contract](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_post_send.3).

3. **Order each hop independently.** On the outgoing relay QP, post the received data and then its destination flag. Receiving the first-hop flag establishes readiness to forward; it does not order writes on a different outgoing QP.

   Register the entire region on both devices. Forwarding must use the **outgoing device’s local MR key** and the destination neighbour’s corresponding remote key. Preserve ordered MR registration; relaxed ordering permits network writes to reach memory out of order. See [the registration contract](https://www.man7.org/linux/man-pages/man3/ibv_reg_mr.3.html).

4. **Use immutable half ownership and sequence-specific lengths.** Exactly one neighbour owns each far-half flag. Neither can publish completion for the entire far shard.

   Send `data → total-payload-length → flag` on the first-hop QP, as 0099 does. A relay must not infer an incoming sequence’s size from the latest local doorbell: that may describe the next, differently sized gather. Acquire the flag before reading its associated metadata.

   In particular, copying 0099’s stripe selection with only **one logical stripe** would forward the whole far shard one way. Explicitly retain two logical halves.

5. **Poison and propagate failure.** A missing relay half must leave the gather incomplete. On proxy failure, publish a mapped abort/error word that GPU polling observes promptly; stop future launches through poison and propagate failure through `tp_watch`.

   Today, `Group.abort()` aborts P2P and NCCL but has no RoCE abort hook, and the CUDA RoCE proxy’s `failed` boolean does not directly release GPU waits. That needs attention for the ring: [current abort path](/home/zurih/c1bench/tp4/tf/zig/src/cuda/tp_group.zig:52).

   Treat the entire gathered output as invalid after failure. Do not attempt per-rank NCCL fallback mid-operation. Quiesce QPs and GPU accesses before unregistering memory; retain allocations until process exit if safe teardown cannot be established.

6. **Handle epoch rollover deliberately.** Existing counters are `u32`, with zero also used as an error sentinel. Test the boundary and either fail collectively before rollover or extend the protocol width. Simply skipping zero also affects slot parity.

**C. GLM integration.** Keep routing deterministic from **the common byte count and collectively agreed transport state**:

```text
fits(bytes) = enabled_on_all_four
              && bytes > 0
              && bytes % 16 == 0
              && bytes <= agreed_cap
```

Agree on the minimum proposed cap across all four ranks, the protocol version, ring mode, and topology. Allocate a common layout from that result. Use raw NCCL for bootstrap records and verdicts, and expose the fast path only after every rank succeeds.

I would initially qualify **512 KiB per original shard**, leaving larger operations on NCCL. That is a proposed test ceiling, not an existing TP=4 qualification. There is a documentation discrepancy: **the GLM caller currently defaults to 4 MiB**, while the documentation and device test use 512 KiB; see [roceBytes](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_native.zig:163). Fabric’s versioned kernel/guards also impose a 1 MiB slot ceiling.

For design (3), the common rule must additionally account for doubling in pass two: `S <= cap_first` and `2S <= cap_second` across every participating pair.

Keep misaligned pointers as errors, with failure propagated collectively. A pointer-based fallback on only one rank would mismatch transports.

For configuration:

- In ring mode, `TF_ROCE_HCA` should select **one device on each physical cable**, rather than the two twins of one cable. Validate that distinction and reject ambiguous/truncated selections.
- Match local devices to the appropriate remote endpoints; do not connect list index `h` to remote list index `h` automatically.
- Exchange each device’s actual GID. Detect its RoCE-v2 IPv4 GID independently and support a **per-device override**. The current single `TF_ROCE_GID_INDEX` cannot express different indices on the two selected devices.
- Validate an unambiguous four-cycle. The fabric planner’s `/24` heuristic is useful only when the cable addressing makes it unambiguous.

**Device-epoch graph replay still works.** Additional sources add waits, not additional epochs. Advance once, after every block has finished copying all four shards. Keep epoch and buffers persistent across graph replays, and retain the grid-class arrival-counter discipline for mixed sizes.

The requirement is one ordered gather sequence per transport instance. Multiple served requests are compatible with that when the engine batches/serializes their collectives; overlapping CUDA gather kernels on the same instance need separate treatment.

**D. Tests before hardware, then on the Sparks.**

For host tests, extract injectable wire/progress seams and exercise the real posting and relay scheduler:

- **Routes and bounds:** neighbour-only QPs; both halves cover every far shard exactly once; original-source indexing; correct incoming/outgoing keys; empty halves; odd pack counts; cap edges; rejected topology.
- **Scheduling:** two pending doorbells, changing sizes, delayed proxies, relay progress without new doorbells, delayed completions, queue wraps, and independent completion ordering across QPs.
- **Lifetime:** defer NIC source reads until after `post_send` returns, then aggressively advance other ranks. Verify send and receive slots remain stable until their reads retire.
- **Failure:** stop after data but before flag, after one far half, and before relay completion. Check bounded waits, poison, abort, error identity, and safe shutdown.
- **Dispatch:** one-rank refusal disables RoCE everywhere; unequal caps produce one common threshold; over-cap and non-multiple sizes use NCCL on all ranks; bootstrap stays on NCCL; TP=2 behaviour remains intact.

The current [multi fake wire](/home/zurih/c1bench/tp4/tf/zig/src/fabric/tp/roce_multi_fake.zig:95) copies payloads immediately during posting and restricts sources to send slots. **It needs extension to catch relay lifetime failures.** Add planted controls—wrong source, overlapping halves, premature flags, early reuse—and require the intended test to fail.

I ran a small routing oracle for eight sizes from 16 bytes through 256 KiB. It verified rank-order bytes, both far-half flags, `3S` transmitted per rank, and the two-matching alternative. It establishes routing arithmetic only.

Then extend `tf-cuda-test roce-gather` for four ranks; [the existing runner hardcodes two](/home/zurih/c1bench/tp4/tf/zig/tests/cuda/roce_gather_tests.zig:21):

1. **Require actual ring RoCE activation.** Print peer/device/GID mappings, logical halves, cap, and relay counters. A fallback must fail this qualification test.
2. **Compare bytes against NCCL and an independent rank-pattern oracle.** Include 16, 32, 48 bytes; 4, 8, **16**, 32, 64 KiB; cap and past-cap sizes. Exercise GLM’s in-place contribution form, `mine = all + rank*S`, with guards.
3. **Replay mixed-size graphs with changing input data**, including different grid classes, repeated slot reuse, and intervening NCCL fallbacks. Run long enough to expose stalled proxy/CQ progress.
4. **Inject delays and failures**, including killing each rank during relay. Require bounded failure and no use of gathered output after an error.
5. **Measure both paths on the same build/topology**, in captured graphs and back-to-back calls, after idle gaps, and with dependent GPU work. Record all-rank latency distributions and relay retirement, not just one average.

The new cross-HCA **receive → forward** visibility and ordering are hardware qualification items. Host tests cannot establish them.

**E. First candidate and expected engine benefit.** My first experiment would be **design (1), opt-in at world four**, with a proposed warmed, captured **16 KiB target below 40 µs**, byte equality at every tested size, and bounded injected failures. Design (3) is the fallback if relay qualification takes longer than expected.

The effort estimates include tests and reviewable integration; the main uncertainty for (1) is relay lifetime/failure handling, and for (2) the bootstrap/build/module adaptation. They could roughly double if hardware reveals a visibility or progress issue.

For **90 eligible calls**, replacing 109 µs with 20–40 µs saves approximately **6.2–8.0 ms**. The original 90 calls total 9.81 ms, so that alone would explain about one fifth of a 48 ms round. Additional drafter/verify calls may increase the opportunity.

Re-measure the current engine’s NCCL baseline before converting that saving to tok/s. Also, `gather_b` includes the following `post` and tap work in [the current profiling boundaries](/home/zurih/c1bench/tp4/tf/zig/src/families/glm/cuda_forward.zig:494), so its whole measured duration should not be attributed to communication.