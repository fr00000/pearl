# Direct Miner Kernel Tuning Results

Last updated: 2026-05-18

This file is the quick reference for direct-miner kernel tuning. The full
benchmark notes live in `H100_SXM_VERIFY.md`.

## Current Production Recommendation

Keep the production direct-miner kernel at:

```bash
--m 8192 --n 1048576 --k 32768 \
--max-in-flight 4 \
--enable-b-cache \
--enable-headless-kernel \
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1 \
--kernel-mma-registers 160 \
--kernel-swizzle 8
```

This is the H100-confirmed production setting after the 4 GiB tensor-hash fix,
equal-B-memory K sweep, large-B spare-VRAM confirmation, and stale B-cache
eviction fix. It uses a 32 GiB B tensor and keeps `k=32768`; the five-minute
post-fix confirm measured `21,741,288,261` chance-weighted units/s, **+1.08%**
versus the earlier `8192x262144x32768` production shape.

## 2026-05-18 Persistent Scheduler Probe

Pod artifacts:

```text
/workspace/build-logs/h100-persistent-scheduler-uv-sync-20260518-093836.log
/workspace/sweeps/kernel-h100-20260518-095157/summary.csv
```

Hypothesis: a persistent cluster scheduler might amortize per-CTA prologue and
scheduler overhead in `hopper_mine_ws` by launching roughly one resident CTA
cluster per SM group and walking the logical tile grid inside each cluster.
This followed NVIDIA's documented Hopper persistent/cooperative scheduler model.

Result: rejected.

| Kernel scheduler | Normalized attempts/s | Delta vs production |
|---|---:|---:|
| current single-tile scheduler | 656,523 reference | baseline |
| persistent cluster scheduler | 498,967 | -24.0% |

The persistent prototype built successfully and passed the forced-win pattern
inspector (`PATTERN_COMPATIBLE=true`), so it did not break proof-visible
row/column extraction. It was simply much slower on the production H100 shape,
and the benchmark miner was killed during shutdown before emitting a final line.
Decision: keep the production kernel on the single-tile scheduler. Future big
kernel work should focus on the mainloop/microarchitecture rather than
grid-stride persistent scheduling.

## Metric Rule

Use `normalized_attempt_rate`, not raw outer-tile rate, when comparing kernel
variants at the same `k`.

Raw outer-tile rate counts CTA completions. It is not always equal to mining
lottery tickets because each CTA has a different number of MMA consumer threads
when `tile_m` changes. The comparable rate is:

```text
normalized_attempt_rate = raw_outer_tile_rate * (mma_consumer_threads_per_cta / 256)
```

For the current `tile_m=128` production kernel, raw outer-tile rate and
normalized attempts are equal. For `tile_m=64`, normalized attempts are half the
raw outer-tile rate.

When comparing different `k` values, compare chance-weighted rate instead:

```text
chance_weighted_rate = normalized_attempt_rate * rounded_common_dim
```

For the current fixed row/column pattern, all swept `k` values are multiples of
the rank, so `rounded_common_dim == k`. This matches the protocol difficulty
adjustment factor `h * w * rounded_common_dim`; a lower `k` can produce more
hashes/sec while still being worse for expected coins.

## 2026-05-18 Equal-B K Sweep

Pod artifacts:

```text
/workspace/sweeps/h100-equal-b-k-sweep-20260518-023622
/workspace/sweeps/h100-k32768-confirm-20260518-024145
```

The post-hash-fix `n=524288,k=16384` shape was the best fixed-`k` result, but
it was still worth checking equal-B-memory shapes because the protocol target
scales with `rounded_common_dim`. All cells below keep B at roughly 8 GiB and
use the same kernel tune:

```text
m=8192
max_in_flight=4
B-cache enabled
headless kernel enabled
kernel=128x256x128 stages=3 cluster=2x1 regs=160 swizzle=8
```

Quick sweep:

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `524288x16384` | Decision |
|---|---:|---:|---:|---|
| `8192x1048576x8192` | 2,554,944 | 20,930,101,131 | -2.98% | reject |
| `8192x524288x16384` | 1,304,275 | 21,369,245,583 | baseline | prior production |
| `8192x262144x32768` | 656,621 | 21,516,157,710 | +0.69% | confirm |
| `8192x131072x65536` | 325,571 | 21,336,604,993 | -0.15% | reject |

Four-minute confirmation:

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|
| `8192x524288x16384` | 1,303,676 | 21,359,422,822 | baseline |
| `8192x262144x32768` | 656,810 | 21,522,363,623 | +0.76% |

Decision at the time: promote `8192x262144x32768` for H100/Hopper production.
It was a small but confirmed expected-coin-rate gain and used the same B-cache
footprint as the prior production shape. The larger-B confirmation below
supersedes this with `8192x524288x32768`.

## 2026-05-18 Large-B K32768 Confirmation

Pod artifacts:

```text
/workspace/sweeps/h100-large-b-quick-20260518-163339/summary.csv
/workspace/sweeps/h100-large-b-boundary-20260518-163944/summary.csv
/workspace/sweeps/h100-large-b-confirm-20260518-164232/summary.csv
```

After the `k=32768` shape was selected, the production H100 still had enough
free VRAM to test larger fixed-B tensors. All cells used the same production
kernel tune:

```text
m=8192
k=32768
max_in_flight=4
B-cache enabled
headless kernel enabled
kernel=128x256x128 stages=3 cluster=2x1 regs=160 swizzle=8
```

Quick 85-second sweep:

| Shape | B tensor | Normalized attempts/s | Chance-weighted rate | Decision |
|---|---:|---:|---:|---|
| `8192x262144x32768` | 8 GiB | 655,999 | 21,495,766,659 | baseline |
| `8192x524288x32768` | 16 GiB | 662,347 | 21,703,782,832 | confirm |
| `8192x786432x32768` | 24 GiB | 662,840 | 21,719,931,129 | no material gain over 16 GiB |
| `8192x1048576x32768` | 32 GiB | 548,152 | 17,961,854,260 | reject: OOM/retry churn |

Four-minute paired confirmation:

| Shape | B tensor | Normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|---:|
| `8192x262144x32768` | 8 GiB | 656,403 | 21,509,006,261 | baseline |
| `8192x524288x32768` | 16 GiB | 661,689 | 21,682,231,004 | +0.81% |

Initial decision: promote `8192x524288x32768` for H100 production. The 16 GiB B
tensor captured almost all of the observed larger-N gain with the cache layout
available at the time. The first 32 GiB B run was rejected because template
changes tried to allocate the new 32 GiB `BpEB` while the stale one was still
held by the B-cache, causing repeated OOM handling.

### Stale B-Cache Eviction Fix

Pod artifact:

```text
/workspace/sweeps/h100-large-b-evict-32g-20260518-172031/summary.csv
```

We added a guarded B-cache eviction path for template changes: if a cache miss
is caused by a new hash key while old B-side artifacts exist, the miner
synchronizes CUDA, drops the stale artifacts, calls `torch.cuda.empty_cache()`,
then allocates the new epoch's B-side tensors. This avoids requiring two
32 GiB `BpEB` tensors to coexist across a template change.

Five-minute 32 GiB B confirmation after the fix:

| Shape | B tensor | Normalized attempts/s | Chance-weighted rate | Errors |
|---|---:|---:|---:|---:|
| `8192x1048576x32768` | 32 GiB | 663,491 | 21,741,288,261 | 0 |

The run crossed a template change and logged:

```text
B-cache template change detected; synchronizing before evicting old B-side artifacts
```

Decision: promote `8192x1048576x32768` for H100 production. It is a smaller
increment than the 8 GiB -> 16 GiB move, but it is a measured expected-coin
gain over 16 GiB (`21.741B` vs `21.682B`, about `+0.27%`) and the eviction path
removes the observed OOM failure mode.

Live churn sanity check:

```text
/workspace/sweeps/h100-live-churn-16g-20260518-173709/summary.csv
```

Because the clean 16 GiB confirm had no template invalidation while live 32 GiB
production hit several, we ran a fresh 16 GiB cell under the current network
churn:

| Shape | B tensor | Normalized attempts/s | Chance-weighted rate | Invalidations |
|---|---:|---:|---:|---:|
| `8192x524288x32768` | 16 GiB | 660,646 | 21,648,051,416 | 1 |

That is below the post-fix 32 GiB confirmation and in the same band as the live
32 GiB production intervals, so the 32 GiB default remains preferred.

### Swizzle Recheck on Promoted Shape

Pod artifact:

```text
/workspace/sweeps/h100-n524288-k32768-swizzle-20260518-165912/summary.csv
```

After promoting the first large-B `n=524288,k=32768` shape, we rechecked the
runtime swizzle setting because the prior `swizzle=8` winner was measured on
earlier shapes. The later 32 GiB B confirmation also used `swizzle=8`.

| Swizzle | Normalized attempts/s | Chance-weighted rate | Decision |
|---:|---:|---:|---|
| 4 | 649,692 | 21,289,117,416 | reject |
| 8 | 661,253 | 21,667,952,568 | keep |
| 12 | 659,604 | 21,613,914,274 | close, but lower |
| 16 | 652,576 | 21,383,617,589 | reject |
| 32 | 623,889 | 20,443,579,103 | reject |

Decision: keep `--kernel-swizzle 8` for the promoted production shape.

### Max-In-Flight Recheck on 16 GiB Shape

Pod artifact:

```text
/workspace/sweeps/h100-n524288-k32768-mif-20260518-170948/summary.csv
```

Queue depth was effectively flat on the 16 GiB B shape:

| max_in_flight | Normalized attempts/s | Chance-weighted rate | Decision |
|---:|---:|---:|---|
| 2 | 662,162 | 21,697,711,775 | tied |
| 4 | 662,043 | 21,693,815,814 | keep default |
| 6 | 661,344 | 21,670,935,795 | tied/lower |
| 8 | 661,680 | 21,681,940,651 | tied |

Decision: no default change. The current queue is not the limiter.

## 2026-05-18 H100 Second-Pass Kernel Probes

Pod artifacts:

```text
/workspace/build-logs/h100-kopt-uv-sync-20260518-045331.log
/workspace/build-logs/h100-kopt-uv-sync-20260518-050259.log
/workspace/build-logs/h100-kopt-uv-sync-20260518-051135.log
/workspace/sweeps/kernel-h100-20260518-052956/summary.csv
```

After promoting the `k=32768` shape, we compiled a focused second-pass grid to
test whether the production `128x256x128, stages=3, c2x1, regs=160` kernel was
still leaving easy launch-config wins on the table.

The `tile_m=256` probe family is not viable as a config-only change. PTXAS
reported that the `256x256x128` kernel needs a register target of roughly `154`
or higher, but the `640`-thread CTA launch shape caps the usable register budget
near `96` registers/thread. Keeping this path would require a real live-state or
CTA-layout rewrite before it can compile.

The buildable second-pass cells were then run for quick 60-second checks at the
current production shape:

```text
m=8192 n=262144 k=32768
max_in_flight=4
B-cache enabled
headless kernel enabled
kernel_swizzle=8
```

The sweep used the last steady progress line because timeout shutdown can spend
extra time draining callbacks after `SIGINT`; all cells had `errors=0` and
passed the pattern inspector.

| Variant | Kernel | Normalized attempts/s | Delta vs production | Decision |
|---|---|---:|---:|---|
| `prod_regs160` | `128x256x128 s3 c2x1 regs160` | 656,523 | baseline | keep |
| `k256_s2_c2x1_regs160` | `128x256x256 s2 c2x1 regs160` | 595,217 | -9.34% | reject |
| `k256_s2_c2x1_regs192` | `128x256x256 s2 c2x1 regs192` | 592,011 | -9.83% | reject |
| `prod_s2_c2x1_regs160` | `128x256x128 s2 c2x1 regs160` | 555,949 | -15.32% | reject |
| `prod_s2_c2x1_regs192` | `128x256x128 s2 c2x1 regs192` | 554,100 | -15.60% | reject |
| `prod_s2_c1x1` | `128x256x128 s2 c1x1` | 545,181 | -16.96% | reject |
| `k256_s2_c1x1` | `128x256x256 s2 c1x1` | 502,168 | -23.51% | reject |

Decision: no production change. The current 3-stage `128x256x128 c2x1 regs160
swizzle8` kernel remains the best H100 setting. The result also suggests the
current pipeline is not over-buffered: cutting to two stages or doubling `tile_k`
reduces throughput rather than freeing useful occupancy.

### Low-register production-family probe

Pod artifacts:

```text
/workspace/build-logs/h100-lowregs-uv-sync-20260518-095739.log
/workspace/sweeps/kernel-h100-20260518-101101/summary.csv
```

We also compiled explicit lower MMA register budgets for the exact production
tile family: `regs=96/112/128/144`, plus the existing `regs=160` control. The
goal was to see whether reducing the live accumulator budget unlocked a larger
occupancy win.

All variants built and passed the forced-win pattern inspector. One-minute
production-shape results:

| Variant | Normalized attempts/s | Delta vs `regs160` |
|---|---:|---:|
| `regs160` | 657,829 | baseline |
| `regs144` | 657,321 | -0.08% |
| `regs128` | 656,999 | -0.13% |
| `regs112` | 656,644 | -0.18% |
| `regs96` | 655,421 | -0.37% |

Decision: keep `regs160`. Lowering `warpgroup_reg_alloc` does not produce a
material occupancy win for the current `128x256x128 c2x1` mine kernel, so the
extra compile variants were removed again.

### Tile-M 192 geometry probe

Pod artifact:

```text
/workspace/build-logs/h100-m192-uv-sync-20260518-102114.log
```

We tested the largest remaining config-only tile-M geometry between the known
points:

- `tile_m=64`: buildable, but half the PoW-checking MMA consumers per CTA.
- `tile_m=128`: current production.
- `tile_m=192`: three MMA warpgroups, 384 PoW-checking consumers per CTA.
- `tile_m=256`: previously rejected by PTXAS/register budget.

Result: `tile_m=192` is not viable as a config-only change. Even the lower
`c1x1 regs112/128/144` variants failed PTXAS:

```text
ptxas fatal: (C7602) Insufficient registers (128)
Try to compile with register target of 154 or higher.
```

The 192-row CTA has 512 threads (`384` consumers + `128` producer-warpgroup
threads), so the hardware register cap per thread is too low for the current
live accumulator/transcript state. This reinforces the same boundary as
`tile_m=256`: larger tile-M requires a real mine-only producer/pipeline rewrite
that removes the mostly idle producer warpgroup and/or cuts accumulator live
state. It cannot be unlocked by another `mma_registers` flag.

### Rank-64 protocol/config probe

Pod artifact:

```text
/workspace/sweeps/h100-r64-c1x1-20260518-103145.log
```

We also checked whether reducing `MINER_NOISE_RANK` from `128` to `64` could
recover enough setup/noising cost to beat the current `k=32768` production
shape. Rank 64 passed the forced-win pattern inspector, but protocol sanity
caps the comparable 8 GiB-B shape at `k=16384`.

Quick runtime probe:

| Config | Normalized attempts/s | Chance-weighted/s | Result |
|---|---:|---:|---|
| `rank64, 8192x524288x16384, c1x1` | ~1,078,000 | ~17.66B | reject |
| current production `rank128, 8192x262144x32768, c2x1` | ~656,800 | ~21.52B | keep |

Rank 64 would need a very large additional kernel win just to catch up with the
current production chance-weighted rate. Do not pursue unless a separate R64
kernel architecture emerges.

## 2026-05-18 Producer-Consumer Mine-Only Pipeline Probe

Pod artifacts:

```text
/workspace/build-logs/h100-pc-allpackages-20260518-104347.log
/workspace/build-logs/h100-pc-nom192-allpackages-20260518-105341.log
/workspace/sweeps/h100-pc-prod-tile-20260518-110225.log
```

We tested the remaining architecture-level shortcut from the Hopper/CUTLASS
pipeline research: remove the dedicated producer warpgroup in the mine-only
kernel and make the MMA warpgroups participate as `ProducerConsumer` users of
`PipelineTmaAsync`. The goal was to eliminate the 128-thread producer warpgroup
from headless mining and see whether a self-loading mainloop could unlock larger
tile-M shapes.

The first build with `192x256x128` in the normal matmul grid failed because the
generator also instantiates the full `run_pearl_gemm_` path. That full GEMM path
still keeps the warp-specialized producer warpgroup and hit the known PTXAS
register failure:

```text
ptxas fatal: (C7602) Insufficient registers (128)
Try to compile with register target of 154 or higher.
```

After removing `m192`, the producer-consumer kernel built and passed the
forced-win pattern inspector for the production tile:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

Runtime result:

| Variant | Validation | Normalized attempts/s | Delta vs production | Decision |
|---|---|---:|---:|---|
| `128x256x128 c2x1 regs160` producer-consumer | pattern-compatible, then CUDA launch failure after 80 matmuls | 451,106 | -31.3% | reject |

The known-good production miner was restarted immediately and returned to
`~657k` normalized attempts/s. Decision: revert the producer-consumer kernel.
The lost TMA/WGMMA overlap is larger than the saved producer warpgroup overhead,
so a self-loading mainloop is not a viable shortcut to a mining win.

## 2026-05-18 Nsight Systems Kernel-Time Profile

Pod artifacts:

```text
/workspace/profiles/nsys-prod-k32768-20260518-111154.nsys-rep
/workspace/profiles/nsys-prod-k32768-20260518-111154.sqlite
/workspace/profiles/ncu-hopper-mine-ws-20260518-110746.log
```

Nsight Compute was attempted first, but the pod blocks GPU performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

Nsight Systems did capture a 48.3-second production run at the current H100
setting (`completed=484`, `normalized_attempt_rate=657,327/s`). Kernel-time
summary:

| GPU kernel group | Total GPU time | Share | Avg per launch | Instances |
|---|---:|---:|---:|---:|
| `hopper_mine_ws` | 47.58 s | 97.9% | 98.31 ms | 484 |
| PyTorch int8 random A generation | 0.51 s | 1.0% | 1.03 ms | 492 |
| `MerkleTreeRootsKernel` | 0.24 s | 0.5% | 0.48 ms | 486 |
| `NoisingKernelA` | 0.22 s | 0.5% | 0.46 ms | 484 |
| `NoisingKernelB` | 0.02 s | ~0.0% | 10.8 ms | 2 |
| Noise generation, commitment, root reduction, fills | <0.03 s combined | ~0.1% | tiny | mixed |

Decision: the remaining major target is the `hopper_mine_ws` mainloop itself.
Python overhead, A generation, Merkle hashing, noising, commitment hashing, and
copies are all too small to produce a meaningful coin-rate jump. Future work
should focus on changing the live accumulator/transcript structure inside the
mine kernel, not on surrounding launch or preprocessing work.

## 2026-05-18 Streaming XOR Live-State Probe

Pod artifacts:

```text
/workspace/build-logs/h100-streaming-xor-20260518-114422.log
/workspace/sweeps/h100-streaming-xor-20260518-115242.log
/workspace/build-logs/h100-streaming-xor-m192-20260518-115708.log
```

We tested whether replacing the transcript XOR tree reduction with a four-lane
streaming XOR could reduce live register pressure near the WGMMA accumulator
enough to either speed up the production tile or unlock the larger `tile_m=192`
geometry.

The production tile built and passed the forced-win pattern inspector:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

Runtime result at the current H100 production shape:

| Variant | Validation | Normalized attempts/s | Delta vs production | Decision |
|---|---|---:|---:|---|
| Streaming XOR, `128x256x128 s3 c2x1 regs160` | pattern-compatible | ~643,600 | -2.0% to -2.2% | reject |

The same reducer did not move the `tile_m=192` compile boundary. Both `c1x1`
and `c2x1` `192x256x128 regs160` variants still failed in PTXAS:

```text
ptxas fatal : (C7602) Insufficient registers (128)
Try to compile with register target of 154 or higher.
```

Decision: revert the streaming XOR reducer and the temporary `tile_m=192`
compile variants. The original tree reduction is faster for the production
kernel, and the larger tile still needs a deeper live-state rewrite rather than
a local reduction change.

## 2026-05-18 H100 PoW Hot-Path Micro-Optimizations

Pod artifacts:

```text
/workspace/build-logs/h100-deep-blake3-uv-sync-20260518-072234.log
/workspace/build-logs/h100-deep-blake3-fix-uv-sync-20260518-073608.log
/workspace/sweeps/kernel-h100-20260518-074958/summary.csv
/workspace/build-logs/h100-deep-pow-smem-uv-sync-20260518-075247.log
/workspace/sweeps/kernel-h100-20260518-080540/summary.csv
```

After the launch-shape probes were exhausted, we tested two deeper
`hopper_mine_ws` hot-path changes that do not alter proof semantics:

1. A scheduled single-block keyed BLAKE3 compressor for PoW checks. This avoids
   the generic compressor's mutable `rBlock` copy and per-round permutation
   scratch by using the fixed BLAKE3 message schedule directly.
2. A mine-only shared-memory copy of the 8-word PoW key and 8-word target, so
   each consumer thread reads them from CTA shared memory instead of from the
   tiny global tensors.

Correctness gates:

| Experiment | Correctness result |
|---|---|
| Scheduled BLAKE3 | CUDA helper matched Python `blake3` byte-for-byte for four random 64-byte blocks; forced-win pattern remained compatible |
| Shared PoW key/target | Forced-win pattern remained compatible |

Static resources for the production `regs160` mine kernel stayed in the same
class:

```text
REG:160 STACK:64 SHARED:1024 LOCAL:0 CONSTANT[0]:1136
```

Quick 60-second production-shape results:

| Experiment | Normalized attempts/s | Delta vs `656,523/s` reference | Decision |
|---|---:|---:|---|
| Scheduled BLAKE3 compressor | 656,122 | -0.06% | reject |
| Shared PoW key/target | 656,976 | +0.07% | neutral, reverted |

The scheduled BLAKE3 run reported log "errors" only because timeout interrupted
the direct-miner shutdown drain; there were no CUDA errors or proof-pattern
failures. The shared-memory key/target run had `errors=0`.

Decision: no production change. These micro-optimizations do not move H100
throughput outside short-run noise, which implies the bottleneck is not the
generic BLAKE3 message permutation or tiny PoW key/target global loads. Further
meaningful gains likely require changing the accumulator/WGMMA pipeline or
reducing the live accumulator footprint, not polishing the final PoW compare.

## 2026-05-17 Post-Hash-Fix Wide-n Confirmation

Pod artifacts:

```text
/workspace/sweeps/h100-k16384-posthash-n-sweep-20260517-213911
/workspace/sweeps/h100-k16384-n524288-confirm-clean-20260517-215817
/workspace/sweeps/h100-k16384-n589824-probe-20260517-221018
```

After fixing the `tensor_hash` 4 GiB length ceiling, we re-tested wider
`n` values at the current `k=16384` H100 production kernel settings:

```text
m=8192
k=16384
max_in_flight=4
headless kernel enabled
B-cache enabled
kernel=128x256x128 stages=3 cluster=2x1 regs=160 swizzle=8
```

Use steady-state progress lines for this comparison. The timeout final line on
large `n` runs can include a 30-second callback drain wait after `SIGINT`, which
underreports the actual mining rate while the miner was running.

| Shape | Steady normalized attempts/s | Chance-weighted rate | Delta | Decision |
|---|---:|---:|---:|---|
| `8192x261888x16384` | 1,297,955 | 21,265,694,720 | baseline | prior production |
| `8192x524288x16384` | 1,303,908 | 21,363,228,672 | +0.46% | production |

Decision: promote `n=524288`. It is a modest gain, but it is a direct expected
coin-rate gain at the same `k` and kernel configuration.

Boundary update: the later tensor-hash final-reduction fix lifted the crash at
`n=589824` and `n=655360`. Those wider shapes now run, but quick same-session
benchmarks did not beat `n=524288` at fixed `k=16384`. This fixed-`k` winner
was later superseded by the equal-B `k=32768` promotion above.

Post-fix artifact:

```text
/workspace/sweeps/h100-wide-n-after-hash-reduce-20260518-021013
```

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `n=524288` | Decision |
|---|---:|---:|---:|---|
| `8192x524288x16384` | 1,307,898 | 21,428,592,900 | baseline | production |
| `8192x589824x16384` | 1,305,191 | 21,384,243,439 | -0.21% | valid, not better |
| `8192x655360x16384` | 1,306,144 | 21,399,857,011 | -0.13% | valid, not better |

### 2026-05-18 Production-N M Sweep

Pod artifact:

```text
/workspace/sweeps/h100-m-sweep-after-hash-reduce-20260518-021643
```

After the wider-N fix, we swept `m` at the current production `n=524288`,
`k=16384` shape to see whether larger batches create more mining chances per
second. They do not produce a meaningful gain:

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `m=8192` | Decision |
|---|---:|---:|---:|---|
| `4096x524288x16384` | 1,297,717 | 21,261,801,666 | -0.51% | valid, worse |
| `8192x524288x16384` | 1,304,349 | 21,370,457,895 | baseline | production |
| `12288x524288x16384` | 1,307,660 | 21,424,698,792 | +0.25% | noise-band |
| `16384x524288x16384` | 1,309,901 | 21,461,426,080 | +0.43% | noise-band, more VRAM |

Decision: keep production at `m=8192`. The M dimension is effectively flat in
the current direct-mining regime, so further gains likely require kernel-side
work rather than larger synthetic batches.

### 2026-05-18 Max-In-Flight Sweep

Pod artifact:

```text
/workspace/sweeps/h100-inflight-sweep-20260518-022217
```

At the production shape, queue depth 2 through 6 produced nearly identical
mining throughput:

| max_in_flight | Final normalized attempts/s | Chance-weighted rate | Delta vs 4 | Decision |
|---:|---:|---:|---:|---|
| 2 | 1,307,641 | 21,424,391,256 | +0.24% | low-VRAM option |
| 3 | 1,304,650 | 21,375,388,968 | +0.01% | valid |
| 4 | 1,304,560 | 21,373,905,183 | baseline | production |
| 5 | 1,304,996 | 21,381,046,302 | +0.03% | valid |
| 6 | 1,305,588 | 21,390,752,688 | +0.08% | valid |

Decision: no production change. `max_in_flight=4` remains the default; depth 2
is acceptable when VRAM pressure matters.

### 2026-05-18 Wide-n Register/Swizzle Recheck

Pod artifact:

```text
/workspace/sweeps/h100-n524288-kernel-knobs-20260518-004429
```

After promoting `n=524288`, we rechecked the register cap and scheduler
swizzle knobs on that wider shape:

```text
m=8192
n=524288
k=16384
max_in_flight=4
headless kernel enabled
B-cache enabled
kernel=128x256x128 stages=3 cluster=2x1
```

Quick 75-second cells. Because every cell uses `k=16384`,
`normalized attempts/s` is enough to rank the variants:

| Variant | Normalized attempts/s | Decision |
|---|---:|---|
| `regs160 swizzle8` | 1,309,228 | keep |
| `regs160 swizzle12` | 1,300,985 | reject |
| default registers `swizzle8` | 1,296,702 | reject |
| `regs192 swizzle8` | 1,294,586 | reject |
| `regs160 swizzle16` | 1,290,676 | reject |
| `regs160 swizzle4` | 1,282,027 | reject |

Decision: no production setting change. The previously promoted
`--kernel-mma-registers 160 --kernel-swizzle 8` remains the best measured H100
kernel knob set on the wider `n=524288` production shape.

## 2026-05-15 Chance-Weighted Shape Sweep

Pod artifacts:

```text
/workspace/sweeps/chance-shape-20260515-203520/summary.csv
/workspace/sweeps/chance-shape-highk-20260515-204528/summary.csv
/workspace/sweeps/chance-shape-midk-20260515-204914/summary.csv
/workspace/sweeps/chance-shape-k16384-confirm-*.log
```

All cells used the production kernel config:

```text
max_in_flight=4
headless kernel enabled
B-cache enabled
kernel=128x256x128 stages=3 cluster=2x1
```

The prior production shape was `m=8192 n=524032 k=8192` with a 5-minute
reference of `2,504,835` normalized attempts/s, or `20,519,608,320`
chance-weighted units/s.

| Shape | Normalized attempts/s | Chance-weighted rate | Delta vs prior | Decision |
|---|---:|---:|---:|---|
| `8192x261888x16384` 5-min confirm | 1,292,734 | 21,180,153,856 | +3.22% | production |
| `8192x261888x16384` quick repeat | 1,294,542 | 21,209,776,128 | +3.36% | confirms |
| `8192x349440x12288` quick | 1,716,443 | 21,091,651,584 | +2.79% | backup |
| `8192x209664x20480` quick | 1,022,681 | 20,944,506,880 | +2.07% | reject |
| `8192x130816x32768` quick | 638,090 | 20,908,933,120 | +1.90% | reject |
| `8192x524032x8192` quick | 2,505,167 | 20,522,328,064 | baseline | prior |
| `8192x1048320x4096` quick | 4,574,304 | 18,736,349,184 | -8.69% | reject |
| `8192x2096896x2048` quick | 8,397,809 | 17,198,712,832 | -16.18% | reject |

Very high `k` cells near `49152` and `63488` crashed before producing progress
and are not production candidates.

Conclusion from this sweep: lower `k` is a trap because the protocol target
scales with `rounded_common_dim`. At the time, the best measured shape was
`8192x261888x16384`, which traded about half the normalized attempt rate for
double the target adjustment and netted a confirmed **+3.22% expected mining
chance**. This was superseded by the later post-hash-fix `n=524288` confirmation
above.

### Quick Fixed-k m/n/max-in-flight Check

After selecting `k=16384`, we ran a quick fixed-k sweep to check whether `m`,
`n`, or `max_in_flight` should move around the new winner.

Pod artifacts:

```text
/workspace/sweeps/k16384-mn-mif-quick-20260515-210849/summary.csv
/workspace/sweeps/k16384-combo-quick-20260515-211922/summary.csv
```

All cells used `k=16384`, the production `128x256x128 stages=3 cluster=2x1`
headless kernel, and B-cache. Because `k` was fixed, normalized attempts/s and
chance-weighted rate rank cells the same.

First-pass 45-second results:

| Cell | m | n | max_in_flight | Normalized attempts/s | Notes |
|---|---:|---:|---:|---:|---|
| `mif2` | 8192 | 261888 | 2 | 1,298,558 | first-pass top, not confirmed |
| `n245760` | 8192 | 245760 | 4 | 1,297,163 | noise-level above current |
| `n261888_current` | 8192 | 261888 | 4 | 1,296,366 | current production |
| `n229376` | 8192 | 229376 | 4 | 1,293,775 | slightly lower |
| `m4096` | 4096 | 261888 | 4 | 1,287,998 | lower |
| `m16384` | 16384 | 261888 | 4 | 1,295,296 | not clean; errors/no final |
| `m32768` | 32768 | 261888 | 4 | 1,290,582 | killed/no final |

Combo/repeat 75-second check:

| Cell | m | n | max_in_flight | Normalized attempts/s | Notes |
|---|---:|---:|---:|---:|---|
| `current_mif4` | 8192 | 261888 | 4 | 1,297,360 | repeat winner |
| `current_mif2` | 8192 | 261888 | 2 | 1,295,129 | first-pass `mif=2` did not repeat |
| `n245760_mif2` | 8192 | 245760 | 2 | 1,292,333 | combo lost |

Conclusion at the time: no production change. The quick sweep did not show a
durable gain from smaller `n`, larger/smaller `m`, or changing `max_in_flight`.
The later post-hash-fix sweep superseded this shape-only conclusion and moved
production to `m=8192`, `n=524288`, `k=16384`, `max_in_flight=4`; the later
equal-B K sweep above superseded that again with `n=262144,k=32768`.

## 2026-05-17 H100 Register/Swizzle Tune

Pod artifacts:

```text
/workspace/sweeps/kernel-h100-20260517-175003/summary.csv
/workspace/sweeps/h100-reg-swizzle-20260517-181544/summary.csv
/workspace/sweeps/h100-confirm-regs160-sw8-20260517-183241/summary.csv
```

All cells used the then-current H100 production shape:

```text
m=8192 n=261888 k=16384
max_in_flight=4
headless kernel enabled
B-cache enabled
kernel=128x256x128 stages=3 cluster=2x1
```

The broad current-shape kernel sweep found only one positive launch-flag cell:
`--kernel-mma-registers 160` measured `1,296,388` normalized attempts/s versus
`1,292,831` for the default-register baseline. Nearby stage, cluster,
`tile_k`, and `tile_m` changes were all slower after normalized attempt
accounting.

The focused 120-second register/swizzle sweep then ranked:

| Variant | Normalized attempts/s | Decision |
|---|---:|---|
| `--kernel-mma-registers 160 --kernel-swizzle 8` | 1,299,813 | confirm |
| `--kernel-mma-registers 160` | 1,297,994 | positive, smaller |
| default registers / heuristic swizzle | 1,290,673 | baseline |
| `--kernel-swizzle 8` | 1,290,450 | noise |
| `--kernel-mma-registers 160 --kernel-swizzle 16` | 1,284,186 | reject |
| `--kernel-swizzle 16` | 1,277,303 | reject |
| `--kernel-swizzle 4` | 1,267,994 | reject |

Five-minute paired confirmation:

| Variant | Normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|
| default registers / heuristic swizzle | 1,289,501 | 21,127,184,384 | baseline |
| `--kernel-mma-registers 160 --kernel-swizzle 8` | 1,299,624 | 21,293,039,616 | +0.78% |

Conclusion: promote `--kernel-mma-registers 160 --kernel-swizzle 8` for H100
production. It is not a big win, but it is a measured mining gain on the exact
shape tested there and does not change proof semantics.

### Compiled Register-Grid Follow-Up

Pod artifact:

```text
/workspace/sweeps/h100-compiled-regs-quick-20260517-205142/summary.csv
```

After the `regs160/swizzle8` promotion, we temporarily compiled additional
production-shape register caps (`168`, `176`, `184`) and checked all explicit
caps with `direct-miner-inspect-pattern`. Every tested cap preserved the default
proof row/column pattern, but none beat `160` in a 75-second paired quick sweep
with `regs160` run first and last:

| Variant | Normalized attempts/s | Chance-weighted rate | Decision |
|---|---:|---:|---|
| `regs160_a` | 1,300,275 | 21,303,705,600 | baseline |
| `regs168` | 1,299,820 | 21,296,250,880 | reject |
| `regs160_b` | 1,299,620 | 21,292,974,080 | baseline repeat |
| `regs184` | 1,296,515 | 21,242,101,760 | reject |
| `regs176` | 1,295,573 | 21,226,668,032 | reject |
| `regs224` | 1,291,397 | 21,158,248,448 | reject |
| `regs192` | 1,288,157 | 21,105,164,288 | reject |

Conclusion: no production change. The temporary `168/176/184` compiled variants
do not justify the extra build time or binary size. Keep production on
`--kernel-mma-registers 160 --kernel-swizzle 8`, and keep the default compiled
grid to the previously useful explicit caps.

## 2026-05-15 H100 Sweep

Pod artifact:

```text
/workspace/sweeps/kernel-h100-20260515-151433/summary.csv
```

Shape and mode:

```text
m=8192 n=524032 k=8192
max_in_flight=4
headless kernel enabled
B-cache enabled
```

Top results:

| Variant | Normalized attempts/s | Delta vs default | Decision |
|---|---:|---:|---|
| `128x256x128 s3 c2x1 regs=160` | 2,517,887 | +0.46% | do not promote |
| `128x256x128 s3 c2x1 default regs` | 2,506,384 | baseline | production |
| `128x256x128 s3 c2x1 regs=224` | 2,506,873 | +0.02% | noise |
| `128x256x128 s4 c2x1` | 2,477,460 | -1.15% | reject |
| `128x256x128 s3 c1x2` | 2,404,479 | -4.07% | reject |
| `128x256x64 s4 c2x1` | 2,153,057 | -14.10% | reject |
| `64x256x128 s3 c2x1` | ~1,755,000 | ~-30% | reject after normalization |

Conclusion for this older `k=8192` shape: runtime kernel-config tuning was
effectively exhausted. The later 2026-05-17 current-shape sweep supersedes the
`mma_registers=160` decision for `k=16384`, where `regs=160` plus explicit
`swizzle=8` was confirmed and promoted.

## Next Kernel Work

Further meaningful gains likely require kernel code changes rather than launch
flags. Best next candidates:

- Use opt-in kernel best-hash observability when evaluating future kernel
  changes. This is implemented behind `--enable-kernel-hash-stats` and should be
  paired with `--enable-diagnostics` when JSONL output is needed. Keep it off in
  production because it adds atomics to the PoW hot path.
- Reduce mining epilogue/hash overhead inside the mine kernel.
- Investigate persistent or fused direct-mining kernels to reduce per-launch and
  per-tile setup costs.
- Design a proof-compatible non-256 `tile_n` path only if the proof row/column
  extraction logic is updated and validated first.

### 2026-05-15 Mine-Kernel Profile and MSW Prefilter Rejection

Pod artifacts:

```text
/workspace/sweeps/ncu-k16384-mine-20260515-213359/ncu.log
/workspace/sweeps/nsys-k16384-direct-20260515-213921/
/workspace/sweeps/msw-prefilter-bench-20260515-215246/direct_miner.log
```

Nsight Compute could attach but could not read GPU performance counters in the
pod:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on target device 0
```

Nsight Systems worked when launched directly with `--duration` and
`--trace-fork-before-exec=true`. For the then-current production shape
`8192x261888x16384`, the GPU kernel-time breakdown was:

| Kernel group | GPU kernel time | Instances | Avg duration | Share |
|---|---:|---:|---:|---:|
| `pearl::hopper_mine_ws` | 62.795 s | 1260 | 49.84 ms | 98.1% |
| PyTorch int8 A generation | 0.630 s | 1265 | 0.498 ms | 1.0% |
| A noising | 0.287 s | 1261 | 0.228 ms | 0.4% |
| Merkle roots | 0.253 s | 1264 | 0.200 ms | 0.4% |

This confirms that new gains need to come from `hopper_mine_ws`; surrounding
launch, A-generation, noising, and Merkle work are already too small to matter
much at this shape.

Experiment: add a non-diagnostics fast path that computes only the
most-significant BLAKE3 output word, rejects immediately when it is above the
target MSW, and only falls back to the full uint256 digest comparison when the
MSW is equal.

Result on the H100 pod with the then-current production settings:

| Variant | Normalized attempts/s | Chance-weighted rate | Delta vs 1,292,734/s reference | Decision |
|---|---:|---:|---:|---|
| then-current production reference | 1,292,734 | 21,180,153,856 | baseline | keep |
| MSW prefilter | 1,268,177 | 20,777,816,115 | -1.90% | reject |

Conclusion: the MSW-only helper is semantically valid but slower. It likely
increases code size/register pressure enough to offset the skipped digest-word
stores and comparison loop. The experiment was reverted and the pod was rebuilt
back to the known-good production kernel.

### 2026-05-15 Runtime Swizzle Sweep

Pod artifact:

```text
/workspace/sweeps/swizzle-k16384-20260515-220944/summary.csv
```

The direct miner now exposes the kernel scheduler swizzle for benchmark runs:

```bash
--kernel-swizzle N
--kernel-swizzle-m-major
```

This is a runtime kernel parameter, so no CUDA rebuild is needed. Production
still leaves it unset so `pearl-gemm` uses its L2 heuristic.

Sweep settings:

```text
m=8192 n=261888 k=16384
max_in_flight=4
B-cache enabled
headless kernel enabled
kernel=128x256x128 stages=3 cluster=2x1
```

Results:

| Cell | Axis | Normalized attempts/s | Chance-weighted rate | Decision |
|---|---|---:|---:|---|
| explicit swizzle 8 | N-major | 1,296,932 | 21,248,937,905 | no production change |
| heuristic | N-major | 1,295,396 | 21,223,770,664 | production |
| explicit swizzle 12 | N-major | 1,290,141 | 21,137,664,901 | reject |
| explicit swizzle 16 | M-major | 1,289,590 | 21,128,640,179 | reject |
| explicit swizzle 16 | N-major | 1,281,961 | 21,003,645,252 | reject |
| explicit swizzle 8 | M-major | 1,279,070 | 20,956,282,907 | reject |
| explicit swizzle 4 | N-major | 1,276,675 | 20,917,045,813 | reject |
| explicit swizzle 32 | M-major | 1,274,175 | 20,876,079,951 | reject |
| explicit swizzle 24 | N-major | 1,255,000 | 20,561,926,945 | reject |
| explicit swizzle 32 | N-major | 1,220,842 | 20,002,273,835 | reject |
| explicit swizzle 128 | N-major | 997,351 | 16,340,599,488 | reject |
| explicit swizzle 64 | N-major | 995,131 | 16,304,222,250 | reject |

Conclusion from this swizzle-only run: the existing heuristic was effectively
optimal by itself. The later 2026-05-17 paired sweep found that explicit
`--kernel-swizzle 8` is useful when combined with `--kernel-mma-registers 160`,
so production now passes both together.

### 2026-05-15/16 Tile-Shape Compile Probes

Pod artifacts:

```text
/workspace/sweeps/kernel-compile-probes-20260515/
/workspace/sweeps/bk256-s2-bench-20260516-002726/direct_miner.log
```

Experiment: test whether larger mainloop tiles can create more useful mining
work per CTA or reduce K-loop overhead.

Results:

| Probe | Result | Decision |
|---|---|---|
| `256x256x128 stages=3 cluster=1x1/2x1` | ptxas failed with insufficient registers (`96`, target at least `154`) | reject |
| `192x256x128 stages=3 cluster=1x1/2x1` | ptxas failed with insufficient registers (`128`, target at least `154`) | reject |
| `128x256x256 stages=3 cluster=2x1` | compiled, but launch requested `289 KB` shared memory on an H100 with a `227 KB` opt-in limit | reject |
| `128x256x256 stages=2 cluster=2x1` | pattern-compatible, but benchmarked at `1,195,438` normalized attempts/s and `19,586,051,786` chance-weighted/s | reject |

The `128x256x256 stages=2` forced-win pattern check matched the default proof
rows and columns for all four iterations. It was therefore valid but slower:
about **-7.5%** versus the then-current production reference of `1,292,734`
normalized attempts/s and `21,180,153,856` chance-weighted/s.

Conclusion: wider-M tile shapes are blocked by register pressure with the
current kernel structure, and bK=256 gives back too much pipeline overlap when
reduced to two stages. Keep production on `128x256x128 stages=3 cluster=2x1`.

### 2026-05-16 Direct Transcript Hash Rejection

Pod artifacts:

```text
/workspace/sweeps/direct-hash-bench-20260516-011209/direct_miner.log
/workspace/sweeps/direct-hash-repeat-20260516-011427/direct_miner.log
```

Experiment: add a mine-only transcript accumulator that updates transcript
words directly at each reduction point, instead of using the generic
`TileHashAccumulator` preload/writeback helper. The goal was to remove small
register moves around the 16-word transcript for `bK=R=128`.

Safety checks:

```text
uv run --no-sync pytest miner/direct-miner/tests  # 30 passed
direct-miner-inspect-pattern ... 128x256x128 stages=3 cluster=2x1
PATTERN_COMPATIBLE=true
```

Benchmark results on the then-current production shape:

| Variant | Normalized attempts/s | Chance-weighted rate | Decision |
|---|---:|---:|---|
| direct transcript hash run 1 | 1,297,814 | 21,263,383,372 | reject: noise-level |
| direct transcript hash run 2 | 1,297,011 | 21,250,226,164 | reject: noise-level |

The result is only about `+0.3-0.4%` versus the older `1,292,734` reference and
overlaps prior no-patch short-run noise near `1,297,360`. It is not a durable
improvement, so the code was reverted.

### Rejected: Final WGMMA Wait Elision

Experiment: have `TileHashAccumulator::accumulate()` report whether it already
performed `warpgroup_wait<0>()` on the final `k_block`, then skip the
unconditional post-loop wait in `CollectiveMainloop::mma()` when that happened.

Rationale: for the production `128x256x128` kernel, the final hash accumulation
already waits for the final WGMMA group before reading `tCrC`, so the
post-loop wait looked redundant.

Benchmark on the H100 pod with the production shape and settings:

```text
m=8192 n=524032 k=8192
max_in_flight=4
B-cache enabled
headless kernel enabled
kernel=128x256x128 stages=3 cluster=2x1
```

Results:

| Variant | Normalized attempts/s | Delta vs 2,504,835/s reference | Decision |
|---|---:|---:|---|
| wait elision run 1 | 2,495,833 | -0.36% | reject |
| wait elision run 2 | 2,491,354 | -0.54% | reject |

Conclusion: the final wait is not a useful speed target. Either it was already
hidden by scheduling, or the extra branch/control state costs more than the wait
saves. The experiment was reverted and the pod was rebuilt back to the
known-good production kernel.

## Opt-In Kernel Hash Observability

The direct miner now has an opt-in CUDA-side diagnostics buffer for benchmark
runs:

```bash
--enable-kernel-hash-stats --enable-diagnostics
```

When enabled, each mining kernel call writes a tiny per-slot `uint32` diagnostics
buffer containing:

- number of kernel hash attempts observed by that call
- selected best hash, chosen by lowest most-significant 32-bit word
- tile coordinate and thread index that produced the selected hash

The direct miner decodes this after the CUDA completion event and writes these
JSONL fields:

```text
kernel_hash_attempts
kernel_best_hash_hex
kernel_best_tile_coord
kernel_best_thread_idx
best_observed_hash_log2
margin_log2
```

This is benchmark observability only. It does not change proof generation,
gateway submission, or the host signal winner path. The default remains disabled
so production mining keeps the same hot path and avoids the extra atomics.

Validation on the H100 pod:

- `pearl-gemm` rebuilt successfully with `MAX_JOBS=4`.
- `get_pow_diagnostics_size()` returned `16` `uint32` words.
- A 70-second smoke run with `m=1024 n=8192 k=8192`, B-cache, headless kernel,
  and kernel hash stats produced 84 JSONL records.
- Each smoke record reported `kernel_hash_attempts=65536`, matching
  `outer_tiles_per_matmul=256` × `256` MMA consumer threads.

## 2026-05-15 Disabled-Path Regression Check

The first observability implementation left a runtime `pow_diagnostics !=
nullptr` branch inside the PoW hot path. With kernel hash stats disabled, two
short production runs measured about `2.447M-2.451M` normalized attempts/s,
roughly 2.3% below the pre-observability reference (`2,506,384/s`).

The fix makes PoW diagnostics a compile-time kernel trait:

- `EnablePowDiagnostics=false` is instantiated for production.
- `EnablePowDiagnostics=true` is selected only when a diagnostics buffer is
  passed.
- The diagnostic hash-copy/atomic path is guarded by `if constexpr`, so the
  disabled production kernel does not carry the diagnostic update body.

Validation build:

```bash
MAX_JOBS=4 \
PEARL_GEMM_DISABLE_DEBUG_MODE=TRUE \
PEARL_GEMM_FORCE_BUILD=TRUE \
uv pip install --no-build-isolation -e miner/pearl-gemm
```

`PEARL_GEMM_DISABLE_DEBUG_MODE=TRUE` was used only to shorten the benchmark
build; production and kernel-hash observability both use `EnableDebug=false`.

Results on the H100 pod:

| Check | Result |
|---|---:|
| `get_pow_diagnostics_size()` | 16 `uint32` words |
| hash-observability smoke records | 164 matmul records plus session end |
| smoke `kernel_hash_attempts` | 65,536 per matmul |
| production disabled-stat run | 1,985 matmuls in 103.8s |
| production normalized attempts/s | 2,504,835 |

Conclusion: the compile-time split recovers the disabled-path regression within
measurement noise of the original production reference.

## 2026-05-15 Production-Shape Hash Distribution

After the compile-time diagnostics split, a bounded production-shape
observability run checked whether the kernel is producing the expected lottery
tickets and hash distribution:

```bash
timeout -s INT 240s uv run --no-sync direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --enable-diagnostics \
  --enable-kernel-hash-stats \
  --metrics-output /workspace/kernel-hash-prod-constexpr.jsonl \
  --phase-tag powdiag-prod-shape \
  --kernel-tile-m 128 --kernel-tile-n 256 --kernel-tile-k 128 \
  --kernel-stages 3 --kernel-cluster-m 2 --kernel-cluster-n 1
```

The stats path is intentionally slow because it adds atomics to the PoW hot
path. The run completed 423 matmuls in 235.0s, so its throughput is not a
production benchmark.

Observed distribution:

| Metric | Value |
|---|---:|
| matmul records | 423 |
| kernel attempts per matmul | 33,538,048 |
| attempts formula | `131,008 CTAs * 256 MMA threads` |
| `log2(attempts_per_matmul)` | 24.999 |
| mean best hash log2 | 230.172 |
| expected mean best hash log2 | 230.168 |
| median best hash log2 | 230.502 |
| best hash log2 over run | 223.427 |
| expected best hash log2 over run | 221.443 |
| best margin over target | 16.410 log2 |

Conclusion: production-shape kernel hash behavior is statistically sane. The
attempt count exactly matches the expected CTA/thread geometry, and the mean
best observed hash is essentially equal to the random-hash expectation. Future
kernel work can focus on speed rather than a suspected lottery-ticket
correctness issue.

## 2026-05-18 Narrow Proof-Pattern Sweep

We tested whether a narrower `tile_n=128` proof pattern could beat the default
`128x256` production kernel once scored by the full protocol-weighted chance
rate. This matters because raw CTA/s is misleading when the proof pattern
changes: `tile_n=128` produces a `cols_pattern` of 32 columns instead of the
default 64, so each CTA has half the proof width and must run enough faster to
compensate.

Build notes:

- Added research-only `128x128`, `192x128`, and `256x128` headless variants.
- `320x128` and `384x128` were pruned after the H100 build hit the
  `KTraits::kNumWarps` static assertion in `pearl_gemm_kernel.h`; those shapes
  are outside the supported warp-count set.
- The reduced grid built successfully on the H100 pod.

Sweep location:

```text
/workspace/sweeps/h100-narrow-mtile-pruned/20260518-142955/summary.csv
```

Same-run comparison on the H100 pod:

| Variant | Proof width | Protocol-weighted attempts/s | Delta vs baseline |
|---|---:|---:|---:|
| `128x256 c2x1 regs160` production baseline | 128 | 2.748T | reference |
| `256x128 c1x1 regs96` | 64 | 2.582T | -6.1% |
| `256x128 c2x1 regs96` | 64 | 2.609T | -5.1% |
| `256x128 c1x1 regs112` | 64 | 2.587T | -5.9% |
| `256x128 c2x1 regs112` | 64 | 2.627T | -4.4% |

Earlier quick cells in the same branch also tested `128x128` and `192x128`;
both were further behind once normalized by proof width. The best narrow
candidate was `256x128 c2x1 regs112`, but it still lost to the production
`128x256 c2x1 regs160` kernel by about 4.4%.

Conclusion: do not switch production to a `tile_n=128` proof pattern. The
narrow pattern raises raw matmul/CTA cadence, but the lost 32 columns of
proof width more than cancel that gain. Keep the production settings at
`tile_m=128`, `tile_n=256`, `tile_k=128`, `cluster=2x1`, `stages=3`,
`mma_registers=160`.

## 2026-05-18 Headless-Only Wide-M Compile Probe

We split the build/dispatch grid so `mine_only` kernels can compile extra
headless-only configs without also instantiating the ordinary GEMM/denoising
template path. This lets future research probe mining-only shapes that are not
valid or practical for full GEMM.

Build change:

- Ordinary GEMM dispatch still uses `MATMUL_CONFIG_SWITCH`.
- Headless mining dispatch now uses `MINE_CONFIG_SWITCH`.
- Normal matmul configs remain visible to both paths.
- Extra `mine_only_matmul_kernels` generate separate `mine_*.cu` files and are
  visible only to `mine_only` dispatch.

Wide-M probes attempted on the H100 pod:

| Variant family | Result |
|---|---|
| `192x256x128`, `stages=3`, `cluster=1x1/2x1`, `regs=160/192` | PTXAS failed: `Insufficient registers (128)`, target needs `154+` |
| `256x256x128`, `stages=3`, `cluster=1x1/2x1`, `regs=160` | PTXAS failed: `Insufficient registers (96)`, target needs `154+` |

Build log:

```text
/workspace/build-logs/h100-mine-only-wideM-parallel-20260518-144613.log
```

After pruning those failing probe configs from the default grid, the H100 clone
rebuilt successfully with the new split dispatch infrastructure:

```text
/workspace/build-logs/h100-mine-switch-pruned-parallel-20260518-145323.log
```

Runtime smoke also passed through the headless mine switch on the benchmark
clone before the intentional `timeout` stopped it:

```text
/workspace/build-logs/h100-mine-switch-smoke-20260518-150124.log
```

Conclusion: the previous wide-M compile blocker is not only in the full
GEMM/denoise path. The headless `hopper_mine_ws` kernel itself still carries
enough live state that `192x256` cannot fit under the 128-register/thread cap
of a 512-thread CTA, and `256x256` is even further away under the 96-register
cap of a 640-thread CTA.

Do not compile wide-M probes by default. The next viable route is a real
headless live-state reduction: reduce the mining kernel's per-thread state from
about 154+ required registers to at most 128 for `192x256`. If that succeeds,
`192x256` can be re-enabled through the new headless-only grid and benchmarked
without touching the ordinary GEMM path.

### Scalar XOR Register Probe

We tested whether the XOR tree in `pow_utils.hpp` was the live-register cliff.
The probe replaced the wide-M mine-only hash reduction with a scalar XOR loop
and re-enabled `192x256x128` at `mma_registers=128`, which is the maximum
register target available to a 512-thread CTA.

Build log:

```text
/workspace/build-logs/h100-scalar-xor-wideM-parallel-20260518-150527.log
```

Result: both `192x256` variants still failed PTXAS with
`Insufficient registers (128)` and the same `Try to compile with register
target of 154 or higher` message.

Conclusion: the XOR tree temporaries are not the controlling register cliff.
The 154+ register requirement is dominated by the WGMMA accumulator/transcript
state around `tCrC`. Unlocking `192x256` needs a larger accumulator redesign,
not a local hash-reduction tweak.

After pruning the scalar probe, the final clean branch rebuilt successfully:

```text
/workspace/build-logs/h100-final-pruned-mine-switch-parallel-20260518-151145.log
```

### Shared Transcript and One-Producer Wide-M Probes

We made two deeper attempts to unlock `192x256` for the headless mine-only
kernel after the scalar XOR test showed the hash reduction itself was not the
register cliff.

First, the shared-transcript probe moved the 16-word BLAKE3 transcript scratch
out of per-thread registers for wide-M mine-only kernels. This still failed at
the same PTXAS point:

```text
/workspace/build-logs/h100-shared-transcript-wideM-parallel-20260518-155553.log
```

Result: `192x256x128`, `stages=3`, `cluster=1x1/2x1`, `regs=128` both failed
with `Insufficient registers (128)` and `Try to compile with register target
of 154 or higher`.

Second, the one-producer-warp probe tried to reduce CTA size from 512 threads
to 416 threads by using a single 32-thread producer warp instead of a full
128-thread producer warpgroup. The goal was to raise the effective PTXAS
register cap enough for the 154+ register request.

Build logs:

```text
/workspace/build-logs/h100-one-producer-wideM-parallel-20260518-160542.log
/workspace/build-logs/h100-one-producer-wideM-regs160-parallel-20260518-161119.log
/workspace/build-logs/h100-one-producer-no-setmax-wideM-parallel-20260518-161703.log
```

Findings:

- `mma_registers=154` is not a legal `setmaxnreg` immediate.
- `mma_registers=160` still failed with `Insufficient registers (128)`.
- Skipping dynamic `warpgroup_reg_alloc`/`warpgroup_reg_dealloc` for the
  one-producer path still failed with the same 128-register cap.

Conclusion: a simple partial-producer thread remap does not unlock wide-M. The
compiler/hardware allocation model still behaves like the consumer threads are
capped at 128 registers for this 192-wide CTA, likely because the launch or
warpgroup allocation rounds the 13-warp CTA into the same limiting bucket. The
probe was pruned from the default grid so the branch remains buildable.

The practical research conclusion is now sharper: `192x256` needs a true
live-state reduction in the MMA/mining path, not movement of hash scratch state
or a smaller producer role. The dominant state is the WGMMA accumulator plus
the transcript extraction state around `tCrC`.

After pruning the one-producer probe, the H100 benchmark clone rebuilt cleanly:

```text
/workspace/build-logs/h100-pruned-one-producer-final-parallel-20260518-162706.log
```
