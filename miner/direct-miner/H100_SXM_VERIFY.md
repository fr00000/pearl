# H100 SXM Quick Verification

Does the H200 stress sweep winner (`stress_n256` = 8192 × 262144 × 8192) carry over to H100 SXM, or does H100's narrower HBM bandwidth shift the optimum?

## Configuration

- Hardware: 1× NVIDIA H100 80GB HBM3, driver 580.126.09, compute 9.0
- Date: 2026-05-14 ~15:49–16:07 UTC
- Duration per cell: 5 min (293 s effective after warmup)
- Production mode: `--enable-b-cache --max-in-flight 4`, diagnostics off
- Stack: pearld synced (blocks 52175), oyster + pearl-gateway running

## Current production winner

As of the 2026-05-18 large-B K32768 confirmation, the recommended H100
direct-miner command is:

```bash
uv run direct-miner \
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

Confirmed five-minute rate:
**663,491 normalized attempts/s per GPU**, or **21,741,288,261
chance-weighted units/s**. Because the protocol difficulty target scales by
`h * w * rounded_common_dim`, the coin-rate comparison across different `k`
values is `normalized_attempt_rate * k`; this setting is **+1.08%** versus the
earlier `8192 x 262144 x 32768` production shape and **+0.27%** versus the
intermediate 16 GiB B setting. The H100 startup script uses this shape and
kernel tune by default.

### 2026-05-18 large-B K32768 confirmation

Pod artifacts:

```text
/workspace/sweeps/h100-large-b-quick-20260518-163339/summary.csv
/workspace/sweeps/h100-large-b-boundary-20260518-163944/summary.csv
/workspace/sweeps/h100-large-b-confirm-20260518-164232/summary.csv
```

Quick spare-VRAM sweep:

| Shape | B tensor | Final normalized attempts/s | Chance-weighted rate | Decision |
|---|---:|---:|---:|---|
| `8192 x 262144 x 32768` | 8 GiB | 655,999 | 21,495,766,659 | baseline |
| `8192 x 524288 x 32768` | 16 GiB | 662,347 | 21,703,782,832 | confirm |
| `8192 x 786432 x 32768` | 24 GiB | 662,840 | 21,719,931,129 | no material gain over 16 GiB |
| `8192 x 1048576 x 32768` | 32 GiB | 548,152 | 17,961,854,260 | reject: OOM/retry churn |

Four-minute confirmation:

| Shape | B tensor | Final normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|---:|
| `8192 x 262144 x 32768` | 8 GiB | 656,403 | 21,509,006,261 | baseline |
| `8192 x 524288 x 32768` | 16 GiB | 661,689 | 21,682,231,004 | +0.81% |

Initial decision: promote `8192 x 524288 x 32768`. The 16 GiB B tensor
captured the larger-N gain without the 24 GiB/32 GiB memory pressure seen in
the first quick sweep. That was superseded by the stale-cache eviction fix
below.

Stale B-cache eviction fix:

```text
/workspace/sweeps/h100-large-b-evict-32g-20260518-172031/summary.csv
```

The initial 32 GiB B failure happened on template change: the old cached
32 GiB `BpEB` was still held while the miner attempted to allocate the
replacement. The fix synchronizes CUDA and evicts stale B-side artifacts before
allocating the new template's B-side tensors.

Post-fix confirmation:

| Shape | B tensor | Final normalized attempts/s | Chance-weighted rate | Errors |
|---|---:|---:|---:|---:|
| `8192 x 1048576 x 32768` | 32 GiB | 663,491 | 21,741,288,261 | 0 |

Decision: promote `8192 x 1048576 x 32768`. The run crossed a template change,
logged the stale-cache eviction, and avoided the previous OOM churn.

B-cache buffer reuse follow-up:

```text
/workspace/logs/direct-miner-h100-prod-n1048576-k32768-reuse-safe-20260518-182528.log
```

After validating the eviction path, we tightened template changes further:
after synchronizing CUDA, the miner now reuses the stale B-side tensors in
place for the next template instead of freeing and reallocating them. This
preserves the safety boundary while avoiding allocator churn on the 32 GiB
`BpEB` buffer. `commitment_hash_B` remains freshly allocated per template
because the async proof callback can hold the previous template's commitment
tensor.

| Shape | B tensor | Normalized attempts/s at 1000 completions | Chance-weighted rate | Template reuses |
|---|---:|---:|---:|---:|
| `8192 x 1048576 x 32768` | 32 GiB | 662,150 | 21,697,347,094 | 2 |

Decision: keep the reuse path. It is a stability/allocator-pressure improvement
rather than a measurable steady-state throughput gain.

Live churn sanity check:

```text
/workspace/sweeps/h100-live-churn-16g-20260518-173709/summary.csv
```

| Shape | B tensor | Final normalized attempts/s | Chance-weighted rate | Invalidations |
|---|---:|---:|---:|---:|
| `8192 x 524288 x 32768` | 16 GiB | 660,646 | 21,648,051,416 | 1 |

The live 16 GiB comparison stayed below the post-fix 32 GiB confirmation and
did not disprove the 32 GiB default under template churn.

Swizzle recheck on the promoted shape:

```text
/workspace/sweeps/h100-n524288-k32768-swizzle-20260518-165912/summary.csv
```

| Swizzle | Final normalized attempts/s | Chance-weighted rate | Decision |
|---:|---:|---:|---|
| 4 | 649,692 | 21,289,117,416 | reject |
| 8 | 661,253 | 21,667,952,568 | keep |
| 12 | 659,604 | 21,613,914,274 | close, but lower |
| 16 | 652,576 | 21,383,617,589 | reject |
| 32 | 623,889 | 20,443,579,103 | reject |

Max-in-flight recheck on the intermediate 16 GiB B shape:

```text
/workspace/sweeps/h100-n524288-k32768-mif-20260518-170948/summary.csv
```

| max_in_flight | Final normalized attempts/s | Chance-weighted rate | Decision |
|---:|---:|---:|---|
| 2 | 662,162 | 21,697,711,775 | tied |
| 4 | 662,043 | 21,693,815,814 | keep default |
| 6 | 661,344 | 21,670,935,795 | tied/lower |
| 8 | 661,680 | 21,681,940,651 | tied |

### 2026-05-18 equal-B K confirmation

Pod artifacts:

```text
/workspace/sweeps/h100-equal-b-k-sweep-20260518-023622
/workspace/sweeps/h100-k32768-confirm-20260518-024145
```

Quick equal-B sweep:

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `524288x16384` |
|---|---:|---:|---:|
| `8192 x 1048576 x 8192` | 2,554,944 | 20,930,101,131 | -2.98% |
| `8192 x 524288 x 16384` | 1,304,275 | 21,369,245,583 | baseline |
| `8192 x 262144 x 32768` | 656,621 | 21,516,157,710 | +0.69% |
| `8192 x 131072 x 65536` | 325,571 | 21,336,604,993 | -0.15% |

Four-minute confirmation:

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|
| `8192 x 524288 x 16384` | 1,303,676 | 21,359,422,822 | baseline |
| `8192 x 262144 x 32768` | 656,810 | 21,522,363,623 | +0.76% |

Decision: promote `8192 x 262144 x 32768`. It keeps the same 8 GiB B tensor
footprint as the previous production shape, while the A-slot pool grows from
about 1.58 GiB to about 2.36 GiB at `max_in_flight=4`.

### 2026-05-18 persistent scheduler probe

Pod artifacts:

```text
/workspace/build-logs/h100-persistent-scheduler-uv-sync-20260518-093836.log
/workspace/sweeps/kernel-h100-20260518-095157/summary.csv
```

We tested a larger scheduling rewrite: replace the mine-only
`SingleTileScheduler` with a cluster-aware persistent scheduler so each
resident cluster walks the swizzled logical tile grid by grid-stride iteration.
The goal was to reduce per-CTA prologue/scheduler overhead in `hopper_mine_ws`.

Validation:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

Benchmark result on the production shape:

| Scheduler | Normalized attempts/s | Delta |
|---|---:|---:|
| single-tile production reference | 656,523 | baseline |
| persistent cluster scheduler | 498,967 | -24.0% |

The persistent scheduler was correct but slower, likely because the current
Hopper mine-only pipeline prefers one logical tile per CTA cluster and loses
more from reduced hardware work distribution than it gains from amortized
prologue. The experiment was reverted. Keep the production launch on the
single-tile scheduler.

### 2026-05-18 second-pass kernel probes

Pod artifacts:

```text
/workspace/build-logs/h100-kopt-uv-sync-20260518-045331.log
/workspace/build-logs/h100-kopt-uv-sync-20260518-050259.log
/workspace/build-logs/h100-kopt-uv-sync-20260518-051135.log
/workspace/sweeps/kernel-h100-20260518-052956/summary.csv
```

After selecting `8192 x 262144 x 32768`, we checked a focused second-pass H100
kernel grid. The larger `tile_m=256` family was rejected at build time: PTXAS
needs roughly `154` registers/thread for `256x256x128`, while the `640`-thread
CTA shape caps the usable register budget near `96`. That route needs a deeper
kernel rewrite, not another launch flag.

The buildable quick cells all lost to production:

| Variant | Kernel | Normalized attempts/s | Delta vs production |
|---|---|---:|---:|
| `prod_regs160` | `128x256x128 s3 c2x1 regs160 sw8` | 656,523 | baseline |
| `k256_s2_c2x1_regs160` | `128x256x256 s2 c2x1 regs160 sw8` | 595,217 | -9.34% |
| `k256_s2_c2x1_regs192` | `128x256x256 s2 c2x1 regs192 sw8` | 592,011 | -9.83% |
| `prod_s2_c2x1_regs160` | `128x256x128 s2 c2x1 regs160 sw8` | 555,949 | -15.32% |
| `prod_s2_c2x1_regs192` | `128x256x128 s2 c2x1 regs192 sw8` | 554,100 | -15.60% |
| `prod_s2_c1x1` | `128x256x128 s2 c1x1 sw8` | 545,181 | -16.96% |
| `k256_s2_c1x1` | `128x256x256 s2 c1x1 sw8` | 502,168 | -23.51% |

Decision: keep the current production kernel. The simple config space around
pipeline stages, `tile_k`, and register cap is exhausted for this shape.

### 2026-05-18 low-register production-family probe

Pod artifacts:

```text
/workspace/build-logs/h100-lowregs-uv-sync-20260518-095739.log
/workspace/sweeps/kernel-h100-20260518-101101/summary.csv
```

We compiled and swept lower explicit `warpgroup_reg_alloc` values for the exact
production tile family (`128x256x128 s3 c2x1`) to check whether register-pressure
relief could produce an occupancy-level gain.

| Variant | Normalized attempts/s | Delta vs `regs160` |
|---|---:|---:|
| `regs160` | 657,829 | baseline |
| `regs144` | 657,321 | -0.08% |
| `regs128` | 656,999 | -0.13% |
| `regs112` | 656,644 | -0.18% |
| `regs96` | 655,421 | -0.37% |

All variants passed the forced-win pattern inspector. Decision: keep `regs160`
and remove the extra compile variants; the current production kernel does not
gain meaningful occupancy from a lower explicit MMA register budget.

### 2026-05-18 tile-M 192 and rank-64 boundary probes

Pod artifacts:

```text
/workspace/build-logs/h100-m192-uv-sync-20260518-102114.log
/workspace/sweeps/h100-r64-c1x1-20260518-103145.log
```

The remaining config-only tile-M geometry, `192x256x128`, failed at build time:

```text
ptxas fatal: (C7602) Insufficient registers (128)
Try to compile with register target of 154 or higher.
```

This happened even for `c1x1 regs112/128/144`. The 192-row tile launches 512
threads per CTA (`384` MMA consumers plus the current 128-thread producer
warpgroup), so the register cap is below the live-state target. Like the earlier
`tile_m=256` failure, this needs a real mine-only pipeline rewrite, not another
launch-config knob.

Rank 64 also passed pattern inspection, but a quick `8192 x 524288 x 16384`
runtime probe with the already-compiled `R64 c1x1` kernel reached only about
`17.66B` chance-weighted/s versus the current production `~21.52B/s`. Rank 64
does not look competitive without a separate large kernel win.

### 2026-05-18 producer-consumer mine-only pipeline probe

Pod artifacts:

```text
/workspace/build-logs/h100-pc-allpackages-20260518-104347.log
/workspace/build-logs/h100-pc-nom192-allpackages-20260518-105341.log
/workspace/sweeps/h100-pc-prod-tile-20260518-110225.log
```

We tried a larger mine-only kernel rewrite: route headless mining through a
producer-consumer TMA mainloop where the MMA warpgroups issue their own loads
instead of reserving a separate producer warpgroup. This was meant to test
whether the 128 producer threads are the real ceiling for larger tile-M shapes.

The first build with `192x256x128` in the normal matmul grid failed because the
build generator also instantiates the full `run_pearl_gemm_` path, which still
uses the warp-specialized producer warpgroup and hit the known PTXAS
`register target of 154 or higher` failure. After removing `m192`, the
producer-consumer production tile built and passed the forced-win pattern
inspector:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

Benchmark result:

| Variant | Validation | Normalized attempts/s | Delta |
|---|---|---:|---:|
| `128x256x128 s3 c2x1 regs160` producer-consumer | pattern-compatible, then CUDA launch failure after 80 matmuls | 451,106 | -31.3% |

Decision: reject and revert. Losing warp-specialized overlap costs far more
than the saved producer warpgroup. The known-good production miner was restarted
immediately and returned to `~657k` normalized attempts/s.

### 2026-05-18 Nsight Systems kernel-time profile

Pod artifacts:

```text
/workspace/profiles/nsys-prod-k32768-20260518-111154.nsys-rep
/workspace/profiles/nsys-prod-k32768-20260518-111154.sqlite
/workspace/profiles/ncu-hopper-mine-ws-20260518-110746.log
```

Nsight Compute could not be used for stall counters because the pod blocks GPU
performance-counter access (`ERR_NVGPUCTRPERM`). Nsight Systems did capture a
48.3-second production run at the current H100 setting:

```text
completed=484
normalized_attempt_rate=657,327/s
chance_weighted_rate=21.54B/s
```

GPU kernel time was overwhelmingly concentrated in the headless mine kernel:

| Kernel group | Share of GPU kernel time | Avg launch time |
|---|---:|---:|
| `hopper_mine_ws` | 97.9% | 98.31 ms |
| PyTorch int8 random A generation | 1.0% | 1.03 ms |
| `MerkleTreeRootsKernel` | 0.5% | 0.48 ms |
| `NoisingKernelA` | 0.5% | 0.46 ms |
| Everything else | ~0.1% | tiny |

Conclusion: the major optimization surface is now only the `hopper_mine_ws`
mainloop/transcript machinery. Optimizing Python, noising, Merkle roots, CUDA
copies, or gateway-side launch plumbing cannot produce a large mining gain at
this point.

### 2026-05-18 streaming XOR live-state probe

Pod artifacts:

```text
/workspace/build-logs/h100-streaming-xor-20260518-114422.log
/workspace/sweeps/h100-streaming-xor-20260518-115242.log
/workspace/build-logs/h100-streaming-xor-m192-20260518-115708.log
```

We tried to reduce accumulator-side live state by changing the transcript XOR
reduction from the original compile-time tree to a four-lane streaming XOR. The
production tile built and stayed proof-pattern compatible:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

But the steady production-shape rate fell to about `643.6k` normalized
attempts/s, roughly `-2%` versus the current `~656k-658k` production reference.
Using the same reducer to probe `tile_m=192` did not help either; both
`192x256x128 c1x1 regs160` and `192x256x128 c2x1 regs160` still failed at
compile time:

```text
ptxas fatal : (C7602) Insufficient registers (128)
Try to compile with register target of 154 or higher.
```

Decision: reject and revert. The tree reducer remains the production choice,
and `tile_m=192` still needs a deeper rewrite of the live transcript or CTA
layout before it can compile.

### 2026-05-18 PoW hot-path micro-optimizations

Pod artifacts:

```text
/workspace/build-logs/h100-deep-blake3-uv-sync-20260518-072234.log
/workspace/build-logs/h100-deep-blake3-fix-uv-sync-20260518-073608.log
/workspace/sweeps/kernel-h100-20260518-074958/summary.csv
/workspace/build-logs/h100-deep-pow-smem-uv-sync-20260518-075247.log
/workspace/sweeps/kernel-h100-20260518-080540/summary.csv
```

Two deeper `hopper_mine_ws` micro-optimizations were tested after the
second-pass launch grid:

| Experiment | Correctness | Normalized attempts/s | Delta vs `656,523/s` reference | Decision |
|---|---|---:|---:|---|
| Scheduled single-block keyed BLAKE3 PoW compressor | Python `blake3` byte-match plus pattern-compatible | 656,122 | -0.06% | reject |
| Shared-memory PoW key/target staging | pattern-compatible | 656,976 | +0.07% | neutral, reverted |

Both stayed in the same static resource class for the explicit production
`regs160` kernel:

```text
REG:160 STACK:64 SHARED:1024 LOCAL:0 CONSTANT[0]:1136
```

Decision: no production change. The final PoW BLAKE3 scheduling and tiny
key/target global loads are not measurable H100 bottlenecks at the current
shape. Future work should target the accumulator/WGMMA mainloop or a true
mine-only pipeline rewrite.

### 2026-05-17 register/swizzle confirmation

Pod artifacts:

```text
/workspace/sweeps/kernel-h100-20260517-175003/summary.csv
/workspace/sweeps/h100-reg-swizzle-20260517-181544/summary.csv
/workspace/sweeps/h100-confirm-regs160-sw8-20260517-183241/summary.csv
```

Five-minute paired result on the then-current same-k production shape:

| Variant | Normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|
| default registers / heuristic swizzle | 1,289,501 | 21,127,184,384 | baseline |
| `--kernel-mma-registers 160 --kernel-swizzle 8` | 1,299,624 | 21,293,039,616 | +0.78% |

Decision: promote the paired H100 tune. `--kernel-swizzle 8` alone was not a
durable win, and nearby stage/cluster/tile changes were slower after normalized
attempt accounting.

Follow-up compiled-register test:
`/workspace/sweeps/h100-compiled-regs-quick-20260517-205142/summary.csv`.
Temporary `168`, `176`, and `184` cap instantiations were pattern-compatible
but did not beat the `160` baseline repeat (`regs160_a=1,300,275`,
`regs160_b=1,299,620`; best non-160 was `regs168=1,299,820`). Do not expand
the production compiled grid for these caps.

### 2026-05-17 post-hash-fix wide-n confirmation

Pod artifacts:

```text
/workspace/sweeps/h100-k16384-posthash-n-sweep-20260517-213911
/workspace/sweeps/h100-k16384-n524288-confirm-clean-20260517-215817
/workspace/sweeps/h100-k16384-n589824-probe-20260517-221018
```

After the `tensor_hash` 4 GiB length fix, exact `n=524288` became usable at
`k=16384`. A clean paired H100 run with the production kernel flags showed:

| Shape | Steady normalized attempts/s | Chance-weighted rate | Delta |
|---|---:|---:|---:|
| `8192 x 261888 x 16384` | 1,297,955 | 21,265,694,720 | baseline |
| `8192 x 524288 x 16384` | 1,303,908 | 21,363,228,672 | +0.46% |

Use the steady-state progress lines for this comparison. On wide `n` runs, the
timeout final line can include a 30-second callback drain after `SIGINT`, which
underreports the mining rate while the miner was actually running.

Decision: promote `n=524288` for H100 production. It is a small but measured
expected coin-rate gain at the same `k`, with the same proof pattern and kernel
configuration.

Boundary update: the later tensor-hash final-reduction fix lifted the crash at
`n=589824` and `n=655360`. The failure was the final `tensor_hash` root
reducer, not B-noising. Both wider shapes now run, but quick same-session
benchmarks stayed slightly below `n=524288` at fixed `k=16384`. The later
equal-B K sweep superseded this fixed-`k` winner with `n=262144,k=32768`.

Post-fix artifact:

```text
/workspace/sweeps/h100-wide-n-after-hash-reduce-20260518-021013
```

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `n=524288` |
|---|---:|---:|---:|
| `8192 x 524288 x 16384` | 1,307,898 | 21,428,592,900 | baseline |
| `8192 x 589824 x 16384` | 1,305,191 | 21,384,243,439 | -0.21% |
| `8192 x 655360 x 16384` | 1,306,144 | 21,399,857,011 | -0.13% |

We also swept `m` at the current production `n=524288,k=16384` point:

```text
/workspace/sweeps/h100-m-sweep-after-hash-reduce-20260518-021643
```

| Shape | Final normalized attempts/s | Chance-weighted rate | Delta vs `m=8192` |
|---|---:|---:|---:|
| `4096 x 524288 x 16384` | 1,297,717 | 21,261,801,666 | -0.51% |
| `8192 x 524288 x 16384` | 1,304,349 | 21,370,457,895 | baseline |
| `12288 x 524288 x 16384` | 1,307,660 | 21,424,698,792 | +0.25% |
| `16384 x 524288 x 16384` | 1,309,901 | 21,461,426,080 | +0.43% |

The larger-M cells are valid but the gain is inside short-run noise and uses
more A-slot VRAM, so keep production at `m=8192`.

Finally, we swept queue depth at the production shape:

```text
/workspace/sweeps/h100-inflight-sweep-20260518-022217
```

| max_in_flight | Final normalized attempts/s | Chance-weighted rate | Delta vs 4 |
|---:|---:|---:|---:|
| 2 | 1,307,641 | 21,424,391,256 | +0.24% |
| 3 | 1,304,650 | 21,375,388,968 | +0.01% |
| 4 | 1,304,560 | 21,373,905,183 | baseline |
| 5 | 1,304,996 | 21,381,046,302 | +0.03% |
| 6 | 1,305,588 | 21,390,752,688 | +0.08% |

Depth 2 through 6 are effectively tied. Keep production at `max_in_flight=4`;
use depth 2 only as a low-VRAM mode.

## Results

| Cell | m | n | k | outer tiles/mm | mm/s | tile rate (tiles/s) | cache h/m/inv | errors |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| prior | 4096 | 32768 | 8192 | 4 096 | 468.9 | 1,920,472 | 137 399 / 2 / 1 | 0 |
| stress_n128 | 8192 | 131072 | 8192 | 32 768 | 63.2 | 2,069,611 | 18 501 / 4 / 3 | 0 |
| **stress_n256** | **8192** | **262144** | **8192** | **65 536** | **32.1** | **2,101,356** | 9 402 / 5 / 4 | **0** |

Ranking on H100 SXM (best first): **stress_n256 > stress_n128 > prior**.

- stress_n256 vs prior:    **+9.4 %** tile-rate gain
- stress_n128 vs prior:    +7.8 % tile-rate gain
- stress_n256 vs n128:     +1.5 % (within 5-min noise)

Memory headroom on H100 80GB at stress_n256: GPU stayed well below cap during the run; no OOM, no allocation failures. Same shape produced no problems on this hardware.

## Cross-platform comparison

| Hardware | Best shape | Tile rate | Notes |
|---|---|---:|---|
| H200 SXM | stress_n256 | 2,123,968 | from prior sweep on the H200 box |
| H100 SXM | stress_n256 | 2,101,356 | this run |

**H200 / H100 ratio at best shape: 1.011×** — H200 is only ~1 % faster at the optimal shape.

If H100 SXM is at $2.99/hr and H200 SXM is at $3.99/hr, $/tile favours H100 SXM by ~32 %:
- H100 SXM: $2.99 ÷ 2.101 Mtiles/s = $1.42 / Mtiles per hour
- H200 SXM: $3.99 ÷ 2.124 Mtiles/s = $1.88 / Mtiles per hour
- H100 SXM is **~25 % cheaper per tile** at the optimal shape.

## Does wider-n transfer?

**Yes.** The "wider-n is the cheap axis" pattern observed across the H100 shape sweep and the H200 stress sweep holds on H100 SXM too. stress_n256 wins on H100 SXM by the same +9 % margin as the prior shape sweep would predict.

Why: at these shapes, the kernel is no longer matmul-rate-bound (32 mm/s × 65 536 tiles/mm = 2.1 M tiles/s, vs 1577 mm/s × 1024 = 1.6 M tiles/s for the original baseline). With each matmul producing ~64× more outer tiles than baseline, per-call overhead is amortized across so many tiles that bandwidth per tile dominates — and that scales reasonably similarly on H100 and H200 at these "shape-saturated" working sets.

The H200's wider 4.8 TB/s HBM bus (vs H100's 3.4 TB/s) doesn't separate the two hardware at stress_n256 because both are evidently bound by something other than raw HBM bandwidth in this regime — likely tensor-core throughput or commitment/Blake hashing serialised on the host side.

## Historical production config for H100 SXM

- Shape: `--m 8192 --n 262144 --k 8192`
- max_in_flight: 4
- `--enable-b-cache`
- Diagnostics off (default after the recent flag rename)
- Expected per-GPU tile rate: **~2.1 M tiles/s** sustained

This was the best setting before the dedicated headless kernel, wider
`n=524032` shape, and `128x256x128, c2x1` cluster tuning. Keep it only as a
conservative fallback.

## Caveats

- 5-min samples have ~±2-3 % variance. The 1.5 % spread between stress_n128 and stress_n256 is at the edge of resolvable. Either is a defensible operational choice if stress_n256 ever runs into a memory wall (e.g., when more processes share the GPU).
- This was on a single H100 SXM with no other GPU contention. Multi-tenant pods may differ.
- pearld was fully synced; mining was against live templates (gateway saw a single template through cell 1, four rotations through cells 2 and 3).

## Addendum: kernel hard ceiling

After the 3 cells above, we tested whether pushing further to n=524 288 (`stress_n512`, 8192 × 524288 × 8192, mif=4) yielded more. It does not:

```
CUDA error: invalid configuration argument at
miner/pearl-gemm/csrc/tensor_hash/tensor_hash_host.hpp:72
```

The failing launch is the `MerkleTreeRootsKernel` inside `tensor_hash`. At n × k = 524288 × 8192 = 4 GB the kernel's computed grid shape exceeds CUDA's max grid dimension (likely 2³¹−1 in x). Memory was *not* the constraint — the launch never succeeded, so allocation was never attempted. The miner ran for 5 min with `completed=0` and `errors=1749`.

This is a kernel-side ceiling, not a hardware one. `stress_n1024` (n=1 048 576) would fail identically and was skipped. To exceed the stress_n256 throughput would require modifying `MerkleTreeRootsKernel::get_grid_shape` to chunk larger inputs across multiple launches or use a different block sizing — an opt/direct-mining change that's out of scope here.

**Practical conclusion at the time:** exact 4 GiB B-side tensor hashes fail on the current `tensor_hash` API. Larger `n` is blocked at the `uint32_t data_size`/launch-config boundary, not by H100 memory capacity.

## Addendum: headless mining op and freed-memory sweep

On 2026-05-15 we built `opt/headless-mine-op` (`cdc12c1e`), a dedicated `headless_mine` CUDA op for direct mining. This path avoids allocating or writing the large `C` output tensor while preserving the noising, transcript extraction, PoW signal, and gateway proof/submission path.

### 5-minute production-shape benchmark

Configuration:

- Hardware: 1x NVIDIA H100 80GB HBM3
- Shape: `m=8192, n=262144, k=8192`
- Flags: `--max-in-flight 4 --enable-b-cache --enable-headless-kernel`
- Diagnostics: off
- Effective duration: 293.4 s

| Path | m | n | k | outer tiles/mm | mm/s | tile rate (tiles/s) | cache h/m/inv | errors |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Original non-headless | 8192 | 262144 | 8192 | 65 536 | 32.1 | 2 101 356 | 9 402 / 5 / 4 | 0 |
| Headless-lite | 8192 | 262144 | 8192 | 65 536 | 34.7 | 2 273 295 | n/a | 0 |
| Dedicated `headless_mine` | 8192 | 262144 | 8192 | 65 536 | 34.9 | 2 284 171 | 10 224 / 3 / 2 | 0 |

Result:

- Dedicated `headless_mine` is **+8.7%** vs the original non-headless path.
- It is only **+0.5%** vs headless-lite, so the throughput ceiling is essentially unchanged.
- The real win is memory: observed steady VRAM during the 5-minute run was about **5.7 GiB**, with the quick-sweep poll seeing a transient peak around **7.8 GiB**. This is much lower than carrying `C` at this shape.

### 60-second freed-memory sweep

We then spent the freed memory on larger synthetic shapes. Each cell used:

```bash
uv run direct-miner \
  --m <m> --n <n> --k 8192 \
  --max-in-flight <mif> \
  --enable-b-cache \
  --enable-headless-kernel \
  --log-interval 100
```

| Cell | m | n | k | max_in_flight | mm/s | tile rate (tiles/s) | peak VRAM MiB | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base_n256 | 8192 | 262144 | 8192 | 4 | 34.9 | 2 286 220 | 7 837 | clean |
| n320 | 8192 | 327680 | 8192 | 4 | 27.9 | 2 283 071 | 6 803 | clean |
| n384 | 8192 | 393216 | 8192 | 4 | 23.2 | 2 285 138 | 7 915 | clean |
| n448 | 8192 | 458752 | 8192 | 4 | 20.0 | 2 295 710 | 9 027 | clean |
| n480 | 8192 | 491520 | 8192 | 4 | 18.7 | 2 297 868 | 9 583 | clean |
| **n512minus** | **8192** | **524032** | **8192** | **4** | **17.6** | **2 300 118** | **10 137** | **clean** |
| m16_n256 | 16384 | 262144 | 8192 | 4 | 17.4 | 2 282 073 | 6 219 | clean |
| m32_n256 | 32768 | 262144 | 8192 | 4 | n/a | n/a | 9 429 | killed during short drain |
| n512minus_mif8 | 8192 | 524032 | 8192 | 8 | live ~17.5 | live ~2.29M | 11 197 | killed during short drain |
| m16_n512minus | 16384 | 524032 | 8192 | 4 | final not representative | live ~2.30M | 10 685 | slot drain timeout |

Notes:

- `n=524032` works because `n * k = 4 292 870 144` bytes, just below the 4 GiB `uint32_t` boundary. Exact `n=524288` still fails.
- Wider `n` is the only useful way to spend the freed memory. Bigger `m` was flat or worse in quick tests.
- The best clean 60-second cell was `8192 x 524032 x 8192`, but it was only **+0.6%** over `8192 x 262144 x 8192`.
- The extra memory therefore buys a small throughput gain, not a new regime. The practical production choice is:

```bash
uv run direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --log-interval 100
```

This is the best measured memory-heavy shape, but the margin is small enough that `8192 x 262144 x 8192` remains a reasonable conservative fallback if `n=524032` shows any long-run instability.

## Addendum: mine-only A-noising specialization

On 2026-05-15 we tested the next conservative kernel change: make the A-side noising kernel skip `AxEBL` work when invoked through `headless_mine`. In mine-only mode, `AxEBL` is not consumed by the dedicated mining matmul path, so the specialization avoids:

- EBL TMA loads in `NoisingKernelA`
- `A * EBL` WGMMA work
- `AxEBL` TMA stores

The normal non-mine path still instantiates and dispatches the original `ComputeAxEBL=true` kernel. The new specialization was built on the H100 pod with:

```bash
MAX_JOBS=8 PEARL_GEMM_FORCE_BUILD=TRUE uv sync --all-packages \
  --reinstall-package pearl-gemm-build-utils \
  --reinstall-package pearl-gemm
```

### 60-second verification

Each cell used production mode:

```bash
uv run direct-miner \
  --m <m> --n <n> --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --log-interval 100
```

| Cell | m | n | k | mm/s | tile rate (tiles/s) | cache h/m/inv | result |
|---|---:|---:|---:|---:|---:|---|---|
| prior base_n256 | 8192 | 262144 | 8192 | 34.9 | 2 286 220 | n/a | clean |
| mine-only A skip, n256 | 8192 | 262144 | 8192 | 34.9 | 2 286 404 | 2379 / 1 / 0 | clean |
| prior n512minus | 8192 | 524032 | 8192 | 17.6 | 2 300 118 | n/a | clean |
| mine-only A skip, n512minus | 8192 | 524032 | 8192 | 17.6 | 2 300 330 | 1191 / 2 / 1 | clean |

Result: **flat**. The measured deltas are ~+0.01%, far inside one-minute noise.

Conclusion: `AxEBL` work is real wasted work in the headless path, but it is not the current throughput limiter at the saturated direct-mining shapes. Keep the specialization if we want the code path to be semantically tighter, but do not expect a measurable production gain from this alone. The next meaningful kernel work should target either the actual mine matmul/inner-hash path or the `tensor_hash` grid-size ceiling that blocks exact `n=524288` and larger B tensors.

## Addendum: mine-kernel profile and cluster autotune

On 2026-05-15 we profiled the current headless path with Nsight Systems on the H100 pod:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --delay=12 --duration=25 \
  -o /workspace/nsys_headless_n256 \
  uv run direct-miner \
    --m 8192 --n 262144 --k 8192 \
    --max-in-flight 4 \
    --enable-b-cache \
    --enable-headless-kernel
```

Kernel time split from `nsys stats --report cuda_gpu_kern_sum`:

| Kernel | GPU time share | Avg time |
|---|---:|---:|
| `hopper_mine_ws` | 98.2% | 28.34 ms |
| A random fill | 0.8% | 0.22 ms |
| A `MerkleTreeRootsKernel` | 0.6% | 0.16 ms |
| `NoisingKernelA` | 0.4% | 0.11 ms |
| everything else | <0.2% | n/a |

Conclusion: the current bottleneck is overwhelmingly the mine matmul/inner-hash kernel. CUDA launch overhead and A-side prep are not material.

### Fixed-tile cluster/stage sweep

We then compiled a focused autotune grid that preserves the `128x256x128` tile shape and only varies pipeline stages plus cluster shape. This keeps the proof row/column extraction pattern aligned with the default mining configuration.

Each short cell used:

```bash
uv run direct-miner \
  --m 8192 --n 262144 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-stages <stages> \
  --kernel-cluster-m <cM> \
  --kernel-cluster-n <cN> \
  --log-interval 200
```

| Variant | Stages | Cluster | tile rate (tiles/s) | Result |
|---|---:|---:|---:|---|
| baseline | 3 | 1x1 | 2 282 247 | clean |
| s4_c1x1 | 4 | 1x1 | 2 184 087 | slower |
| s5_c1x1 | 5 | 1x1 | 0 | invalid: 241 KiB smem > 227 KiB limit |
| **s3_c2x1** | **3** | **2x1** | **2 488 762** | **winner** |
| s3_c1x2 | 3 | 1x2 | 2 381 428 | faster than baseline |
| s3_c2x2 | 3 | 2x2 | 2 391 889 | faster than baseline |
| s4_c2x1 | 4 | 2x1 | 2 473 038 | near winner |
| s4_c1x2 | 4 | 1x2 | 2 377 482 | faster than baseline |
| s4_c2x2 | 4 | 2x2 | 2 390 194 | faster than baseline |

The best fixed-tile kernel is:

```bash
--kernel-stages 3 --kernel-cluster-m 2 --kernel-cluster-n 1
```

### Winner validation on memory-heavy production shape

One-minute validation on the previous best shape:

| Shape | Kernel | tile rate (tiles/s) | cache h/m/inv | Result |
|---|---|---:|---|---|
| 8192 x 524032 x 8192 | baseline 1x1 stage 3 | 2 300 118 | n/a | prior best |
| 8192 x 524032 x 8192 | 2x1 stage 3 | 2 505 314 | 1314 / 1 / 0 | clean |

Longer confirmation:

| Shape | Kernel | Duration | mm/s | tile rate (tiles/s) | cache h/m/inv | Result |
|---|---|---:|---:|---:|---|---|
| 8192 x 524032 x 8192 | 2x1 stage 3 | 323.6 s | 19.1 | 2 503 688 | 6182 / 4 / 3 | clean |

This is **+8.8%** over the prior `n512minus` best and **+19.1%** over the original non-headless `n256` baseline.

Current recommended production command:

```bash
uv run direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-stages 3 \
  --kernel-cluster-m 2 \
  --kernel-cluster-n 1 \
  --log-interval 100
```

Next kernel work:

- Profile `hopper_mine_ws` with Nsight Compute if available; Nsight Systems only says the mine kernel dominates, not which instructions or memory paths inside it dominate.
- Test a second, proof-pattern-validated tile-shape grid only after confirming the extracted row/column pattern still matches the mining configuration, or after teaching the miner to derive the job key/target from the actual variant pattern.
- Separately fix the `tensor_hash` 4 GiB grid/`uint32_t` ceiling if we want to explore `n >= 524288`.

## Addendum: tile-shape probe grid

On 2026-05-15 we added a small proof-pattern inspector and a first tile-shape probe grid. The inspector forces an easy win with a max target, reads the `HostSignalHeader`, and checks that the extracted proof rows/columns still match the default mining configuration:

- rows: `[0, 8]`
- cols: `[0, 1, 8, 9, ..., 248, 249]`

Build pruning:

- `256x256x128` failed compile with `ptxas fatal (C7602): Insufficient registers (96)` and requested a register target of 154.
- `256x256x64` failed the same way.
- Both `256x256` probes were removed from the compiled grid. Retesting that family requires launch-bound/register-allocation work first.

All remaining compiled candidates were pattern-compatible:

| Tile | Cluster | Stages | Pattern result |
|---|---|---:|---|
| 128x256x128 | 2x1 | 3 | compatible |
| 64x256x128 | 1x1 | 3 | compatible |
| 64x256x128 | 2x1 | 3 | compatible |
| 128x256x64 | 1x1 | 3 | compatible |
| 128x256x64 | 2x1 | 3 | compatible |
| 128x256x64 | 2x1 | 4 | compatible |

Important accounting note: raw `tile_rate` is not directly comparable when `tile_m` changes. In `KernelTraits`, `kNumMmaThreads = (tile_m / 64) * 128`, and each MMA consumer thread checks the PoW target once per CTA. Therefore a `64x256` CTA has half as many PoW checks as a `128x256` CTA. The apples-to-apples normalized attempt rate is:

```text
normalized_attempt_rate = raw_tile_rate * (tile_m / 128)
```

### Quick production-shape benchmark

Each cell used:

```bash
uv run direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-tile-m <tile_m> \
  --kernel-tile-n 256 \
  --kernel-tile-k <tile_k> \
  --kernel-stages <stages> \
  --kernel-cluster-m <cM> \
  --kernel-cluster-n 1 \
  --log-interval 100
```

| Variant | Raw tile rate (tiles/s) | Normalized attempt rate | Result |
|---|---:|---:|---|
| **128x256x128, c2x1, s3** | **2 506 544** | **2 506 544** | current winner |
| 64x256x128, c1x1, s3 | 2 958 630 | 1 479 315 | worse after normalization |
| 64x256x128, c2x1, s3 | ~3 513 599 | ~1 756 800 | worse after normalization; timeout killed during drain |
| 128x256x64, c1x1, s3 | 2 080 652 | 2 080 652 | worse |
| 128x256x64, c2x1, s3 | 2 084 419 | 2 084 419 | worse |
| 128x256x64, c2x1, s4 | 2 153 039 | 2 153 039 | worse |

Conclusion: keep the current production kernel:

```bash
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1
```

The raw `64x256x128` numbers are tempting, but they do not represent more PoW checks per second after accounting for the smaller M tile. The next real optimization target is inside `hopper_mine_ws`, not this first tile-shape grid.

## Addendum: NCU attempt and static resource analysis

On 2026-05-15 we tried to profile the current winner with Nsight Compute:

```bash
/usr/local/cuda/bin/ncu \
  --target-processes application-only \
  --kernel-name regex:hopper_mine_ws \
  --launch-skip 2 \
  --launch-count 1 \
  --kill yes \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  -o /workspace/ncu_hopper_mine_direct_<ts> \
  /root/pearl/.venv/bin/direct-miner ...
```

The pod host blocks NVIDIA performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

So NCU cannot provide stall, occupancy, scheduler, or speed-of-light counters from this container unless the host enables unrestricted profiling counters.

As a fallback, we dumped static resource usage from the compiled extension:

```bash
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  /root/pearl/miner/pearl-gemm/src/pearl_gemm_cuda.cpython-312-x86_64-linux-gnu.so
```

Relevant `hopper_mine_ws` resource usage:

| Variant | Registers/thread | Stack | Static shared | Constant |
|---|---:|---:|---:|---:|
| 128x256x128, R128 | 168 | 64 | 1024 | 1136 |
| 128x256x64, R128 | 168 | 64 | 1024 | 1136 |
| 64x256x128, R128 | 255 | 64 | 1024 | 1136 |

Interpretation:

- Current winner launches `tile_m=128`, so `kNumMmaThreads = 256` and total block threads are `256 consumers + 128 producer-warpgroup threads = 384`.
- `384 * 168 = 64 512` registers, essentially one full H100 SM register file. This explains why the kernel behaves as one CTA per SM and why larger `256x256` tiles fail to compile.
- `64x256x128` uses fewer consumer threads but jumps to 255 registers/thread, landing at `256 * 255 = 65 280` registers per CTA. That also fills an SM while doing only half the normalized PoW checks per CTA.
- `128x256x64` does not reduce register pressure, so its extra K-loop overhead simply makes it slower.

### Next kernel target

The next plausible kernel optimization is to remove the mostly idle producer warpgroup from the mine-only launch shape.

Today the kernel launches a full 128-thread producer warpgroup even though the TMA path effectively uses one producer warp. The existing named-barrier accounting already often uses `kNumMmaThreads + cutlass::NumThreadsPerWarp`, which suggests only 32 producer threads need to participate in some synchronization points.

Candidate experiment:

- keep consumer WGMMA warpgroups aligned at thread IDs `0..255`
- move the producer warp to thread IDs `256..287`
- change `kNumThreads` from `kNumMmaThreads + 128` to `kNumMmaThreads + 32` for `MineOnly`
- make producer role selection explicit instead of using `warp_group_idx == 0`
- set `consumer_tix = threadIdx.x` for consumer threads
- preserve existing full-producer-warpgroup path for normal non-mine GEMM

For the current 128 tile this would reduce launched threads from 384 to 288. It will not magically halve registers, but it could reduce scheduler pressure, reduce dead producer-thread overhead, and create room for a later `tile_m=192/256` experiment if register allocation also improves.

This is a real kernel surgery item, not a CLI/autotune item. It should be implemented behind a separate mine-only specialization and benchmarked against the current `128x256x128, c2x1, s3` winner.

### One-producer-warp experiment result

We tried the candidate one-producer-warp layout:

- `KernelTraits::kNumThreads = kNumMmaThreads + 32` for `MineOnly`
- consumer threads moved to `threadIdx.x = 0..255`
- producer warp moved to `threadIdx.x = 256..287`
- `consumer_tix = threadIdx.x`

The extension compiled successfully, and static resources per thread were unchanged:

| Variant | Registers/thread | Stack | Static shared | Constant |
|---|---:|---:|---:|---:|
| 128x256x128, R128 | 168 | 64 | 1024 | 1136 |

But the forced-win pattern inspector hung and had to be killed:

```bash
timeout --signal=SIGINT --kill-after=10 20 \
  uv run direct-miner-inspect-pattern \
    --tile-m 128 --tile-n 256 --tile-k 128 \
    --cluster-m 2 --cluster-n 1 --stages 3 --iterations 1
```

Result:

```text
Killed
STATUS=137
```

No `HostSignalHeader` was produced. The most likely cause is that `PipelineTmaAsync` / TMA barrier signaling assumes producer participation from a normal warpgroup shape even though only one producer warp issues copies. The experiment was rolled back to the known-good full producer warpgroup layout.

Conclusion: do not use the one-producer-warp layout. A future version would need a purpose-built mine-only pipeline or a deeper CUTLASS pipeline rewrite, not just a thread remap.

## Register-allocation and 64-row tile resweep

On 2026-05-15 we added an opt-in `mma_registers` kernel-dispatch knob and
compiled explicit register-allocation variants for:

- `128x256x128, c2x1, s3` with `mma_registers={160,192,224}`
- `64x256x128, c1x1/c2x1, s3` with `mma_registers={128,160,192,224}`

The default `mma_registers=0` path preserves the existing kernel heuristic.

Forced-win pattern inspection passed for every tested variant. The emitted
proof row and column patterns matched the default mining configuration:

```text
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
PATTERN_COMPATIBLE=true
```

One-minute production-shape sweep:

Important correction: the direct-miner `tile_rate` log field is raw CTA /
outer-tile rate. When `tile_m` changes, it is not an apples-to-apples count of
lottery tickets. `tile_m=64` has one MMA warpgroup and 128 MMA consumer
threads; `tile_m=128` has two MMA warpgroups and 256 MMA consumer threads.
Each MMA consumer thread performs one PoW check per CTA, so `64x256` raw
outer-tile rates must be multiplied by `64 / 128 = 0.5` before comparing them
to `128x256` variants.

| Variant | Source | Raw outer-tiles/s | Normalized 128-equivalent attempts/s |
|---|---:|---:|---:|
| 64x256x128, c2x1, default regs | final | 3 520 483 | 1 760 242 |
| 64x256x128, c2x1, regs=224 | final | 3 520 104 | 1 760 052 |
| 64x256x128, c2x1, regs=192 | last steady log | 3 517 276 | 1 758 638 |
| 64x256x128, c2x1, regs=128 | last steady log | 3 517 018 | 1 758 509 |
| 64x256x128, c2x1, regs=160 | last steady log | 3 496 877 | 1 748 439 |
| 64x256x128, c1x1, default/explicit regs | mixed | ~2 944 000 - 2 959 000 | ~1 472 000 - 1 479 500 |
| 128x256x128, c2x1, regs=160 | final | 2 510 826 | 2 510 826 |
| 128x256x128, c2x1, default regs | final | 2 502 826 | 2 502 826 |

Five-minute confirmation for the raw-fast but normalized-worse 64-row variant:

```text
Variant: 64x256x128, c2x1, s3, default regs
Completed: 4344 matmuls
Elapsed: 324.0 s
Completion rate: 13.4 matmuls/s
Tile rate: 3 512 843 tiles/s
B-cache: hits=4341 misses=4 invalidations=3
```

This does **not** supersede the earlier tile-shape conclusion. With the
production `n=524032` shape, direct-miner raw outer-tile accounting, headless
kernel, and B-cache enabled, `64x256x128, c2x1, s3` looked about:

```text
3 512 843 / 2 503 688 = 1.403x
```

or roughly **+40%** on raw CTA rate, but after normalizing for the half-sized
MMA consumer thread count:

```text
(3 512 843 * 0.5) / 2 503 688 = 0.701x
```

So `64x256x128` produces about **30% fewer PoW attempts/s** than the
`128x256x128, c2x1, s3` production kernel. The 64-row result is retained here
as a case study in why raw direct-miner `tile_rate` must be normalized when
changing `tile_m`.

Recommended production kernel flags:

```bash
--enable-headless-kernel \
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1
```

## Five-minute focused kernel sweep

On 2026-05-15 we ran a 5-minute-per-cell sweep around the current production
winner on the H100 pod:

```text
m=8192 n=524032 k=8192
max_in_flight=4
headless kernel enabled
B-cache enabled
```

The sweep logs and CSV were written on the pod under:

```text
/workspace/sweeps/kernel-h100-20260515-151433/
```

The miner now reports two rates:

- `raw_outer_tile_rate`: CTA/outer-tile completions per second.
- `normalized_attempt_rate`: protocol-comparable 128-equivalent PoW attempts/s.

For `tile_m=128`, these are equal. For `tile_m=64`, normalized attempts are
half the raw outer-tile rate because the CTA has only 128 MMA consumer threads
instead of 256.

| Variant | Normalized attempts/s | Delta vs prod default | Notes |
|---|---:|---:|---|
| `128x256x128 s3 c2x1 regs=160` | 2,517,887 | +0.46% | best measured, too small to promote |
| `128x256x128 s3 c2x1 default regs` | 2,506,384 | baseline | current production |
| `128x256x128 s3 c2x1 regs=224` | 2,506,873 | +0.02% | noise-level |
| `128x256x128 s3 c2x1 regs=192` | 2,504,536 | -0.07% | noise-level |
| `128x256x128 s4 c2x1` | 2,477,460 | -1.15% | deeper pipeline loses |
| `128x256x128 s3 c1x2` | 2,404,479 | -4.07% | N-cluster loses |
| `128x256x128 s4 c2x2` | 2,393,608 | -4.50% | N-cluster loses |
| `128x256x128 s3 c2x2` | 2,391,764 | -4.57% | N-cluster loses |
| `128x256x128 s4 c1x2` | 2,391,065 | -4.60% | N-cluster loses |
| `128x256x128 s3 c1x1` | 2,302,027 | -8.15% | confirms `c2x1` is real |
| `128x256x128 s4 c1x1` | 2,201,806 | -12.15% | loses |
| `128x256x64 s4 c2x1` | 2,153,057 | -14.10% | shorter K tile loses |
| `128x256x64 s3 c1x1` | 2,084,854 | -16.82% | shorter K tile loses |
| `128x256x64 s3 c2x1` | 2,083,953 | -16.86% | shorter K tile loses |

Skipped/guard cells:

- `128x256x128 s5 c1x1` failed the pattern-inspector launch because the
  kernel requested 246,784 bytes of shared memory, above the device limit
  reported to that launch path.
- `64x256x128 s3 c1x1/c2x1` passed pattern inspection but did not emit FINAL
  lines before timeout cleanup. Their steady-state logs still showed the
  normalized-accounting issue clearly: `c2x1` ran about **3.51M raw outer
  tiles/s** but only **1.76M normalized attempts/s**, roughly **30% below**
  the 128-row production kernel.

Conclusion:

- Keep production on `128x256x128, stages=3, cluster=2x1`.
- Historical note: on this older `k=8192` shape, `--kernel-mma-registers 160`
  alone was not promoted. The 2026-05-17 current-shape confirmation supersedes
  that decision for `k=16384` by pairing `regs=160` with `swizzle=8`.
- Do not pursue `tile_k=64`, `cluster_n=2`, or `stages=4` for this production
  shape.
- Do not sweep `tile_n=128/512` as a simple runtime setting. The default proof
  column pattern reaches columns 248/249, so non-256 N tiles need a separate
  proof-pattern design before they can be production candidates.

## Addendum: opt-in kernel hash observability

The next kernel-support change is benchmark-only observability for the PoW hash
path. It adds an optional per-slot `PowDiagnostics` buffer to `noisy_gemm` and
`headless_mine`. When the pointer is null, the kernel takes the existing
production path. When enabled, each PoW check increments an attempt counter and
the kernel stores the best observed hash prefix for the completed call.

Direct miner flag:

```bash
--enable-kernel-hash-stats
```

Recommended benchmark usage:

```bash
--enable-kernel-hash-stats \
--enable-diagnostics \
--metrics-output /workspace/kernel-hash-smoke.jsonl
```

New JSONL fields:

```text
kernel_hash_attempts
kernel_best_hash_hex
kernel_best_tile_coord
kernel_best_thread_idx
best_observed_hash_log2
margin_log2
```

This is intentionally not a production default. The diagnostic path adds
atomics to the PoW hot path and is meant to answer, "is this kernel producing a
reasonable hash distribution?" without waiting for a rare network win. It does
not alter proof generation, gateway submission, or the host-signal winner path.

Validation:

- Built and installed `pearl-gemm` on the H100 pod after adding the diagnostics
  pointer through `PearlAPIParams`, `CollectiveMainloop`, and both GEMM entry
  points.
- Verified `get_pow_diagnostics_size() == 16`.
- Ran a 70-second smoke with `m=1024 n=8192 k=8192`, B-cache, headless kernel,
  diagnostics, and kernel hash stats. The run completed 84 matmuls and wrote 84
  JSONL records.
- Each record reported `kernel_hash_attempts=65536`, matching the expected
  `256` outer tiles × `256` MMA consumer threads for that smoke shape.

Follow-up disabled-path check:

- The initial observability version measured `2.447M-2.451M` normalized
  attempts/s with kernel stats disabled, about 2.3% below the pre-observability
  `2,506,384/s` reference.
- Moved diagnostics selection into the kernel template as
  `EnablePowDiagnostics`; production instantiates `false`, and the diagnostics
  write path is guarded by `if constexpr`.
- Rebuilt on the H100 pod with:

```bash
MAX_JOBS=4 \
PEARL_GEMM_DISABLE_DEBUG_MODE=TRUE \
PEARL_GEMM_FORCE_BUILD=TRUE \
uv pip install --no-build-isolation -e miner/pearl-gemm
```

- Re-ran the hash smoke with a clean SIGINT drain. It wrote 164 matmul records
  plus session end, each with `kernel_hash_attempts=65536`.
- Re-ran production mode with kernel hash stats and JSON diagnostics disabled:
  `m=8192 n=524032 k=8192`, B-cache, headless, `max_in_flight=4`,
  `128x256x128`, `stages=3`, `cluster=2x1`.
- Final production-disabled result: 1,985 completed matmuls in 103.8s,
  `normalized_attempt_rate=2,504,835/s`.

Conclusion: compile-time diagnostics recover the disabled-path regression to
within measurement noise of the original production reference.

Production-shape hash distribution check:

- Ran a bounded 240-second observability sample with the production shape and
  kernel settings: `m=8192 n=524032 k=8192`, B-cache, headless,
  `max_in_flight=4`, `128x256x128`, `stages=3`, `cluster=2x1`.
- The stats path is intentionally slow because it adds atomics to the PoW hot
  path; this run is a distribution check, not a throughput benchmark.
- Completed 423 matmul records in 235.0s.
- Every record reported `kernel_hash_attempts=33,538,048`, equal to
  `131,008` CTAs × `256` MMA consumer threads.
- Mean best hash log2 was `230.172`; random-hash expectation for
  `33,538,048` attempts is `230.168`.
- Best hash log2 over the run was `223.427`; expected run-best across all
  sampled attempts is about `221.443`, comfortably plausible for this sample
  size.
- Best observed margin over target was `16.410 log2`.

## Addendum: tensor_hash 4 GiB ceiling fix

On 2026-05-17 we fixed the `tensor_hash` byte-length ceiling that blocked exact
4 GiB B tensors. The root cause was a 32-bit `data_size` argument in the
tensor-hash host/device launch path. A tensor with exactly `2^32` bytes
truncated to zero before the SM90 Merkle roots kernel computed its grid shape.

Patch summary:

- Carry `data_size` / `data_len` as `uint64_t` through
  `tensor_hash_decl.hpp`, `tensor_hash_host.hpp`, and
  `merkle_tree_roots_kernel.hpp`.
- Preserve the existing 32-bit root-count interface for the later reduction
  kernels, but add an explicit host-side guard if `num_blocks` exceeds
  `uint32_t`.
- Promote BLAKE3 chunk-counter arithmetic to `uint64_t` so large tensor chunk
  indices do not overflow before being written into the existing 64-bit
  BLAKE3 counter field.

Validation on the H100 pod:

```text
524288 x 8192 uint8 tensor
num_bytes = 4,294,967,296 = 2^32
scratchpad = 1,048,576 bytes
tensor_hash runtime = 5.37 ms
result: clean

524289 x 8192 uint8 tensor
num_bytes = 4,294,975,488 = 2^32 + 8192
scratchpad = 1,048,608 bytes
tensor_hash runtime = 5.63 ms
result: clean
```

Boundary shape checks:

| Shape | Normalized attempts/s | Chance-weighted/s | Result |
|---|---:|---:|---|
| `8192 x 524288 x 8192` | 2,501,449 | 20.492B | exact 4 GiB B hash works, not a new winner |
| `8192 x 262144 x 16384` | 1,294,259 | 21.205B | exact 4 GiB B hash works, slightly below current winner |

Conclusion at that point: the ceiling was fixed and exact 4 GiB B tensors were
usable, but the exact boundary shapes were within measurement noise of the prior
near-boundary cells at `k=8192`. The later `k=16384` wide-`n` confirmation above
promotes `8192 x 524288 x 16384` for H100 production.

## Addendum: post hash-fix H100 profile

After the 4 GiB hash fix, Nsight Compute was retried with
`--target-processes all` and a single `hopper_mine_ws` launch target. The pod
host still blocks hardware performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

The fallback Nsight Systems run used the then-current production H100 shape:

```bash
direct-miner \
  --m 8192 --n 261888 --k 16384 \
  --max-in-flight 4 --enable-b-cache --enable-headless-kernel \
  --kernel-tile-m 128 --kernel-tile-n 256 --kernel-tile-k 128 \
  --kernel-stages 3 --kernel-cluster-m 2 --kernel-cluster-n 1
```

Run result:

```text
completed=1638 elapsed=82.9s
completion_rate=19.8/s
normalized_attempt_rate=1,294,144/s
chance_weighted_rate=21,203,256,794/s
```

Nsight Systems kernel-time summary:

| Kernel | Time share | Avg time | Instances |
|---|---:|---:|---:|
| `hopper_mine_ws` | 98.0% | 49.898 ms | 1,638 |
| A random/fill kernels | 1.0% | 0.483 ms | 1,641 |
| `MerkleTreeRootsKernel` | 0.5% | 0.266 ms | 1,640 |
| `NoisingKernelA` | 0.4% | 0.225 ms | 1,638 |
| other hash/noise helpers | <0.1% each | | |

Static resource usage from the same compiled extension still reports the
production `hopper_mine_ws` path at:

```text
REG:168 STACK:64 SHARED:1024 LOCAL:0
```

Conclusion: after B-cache and headless mode, the meaningful H100 kernel lever
is inside `hopper_mine_ws`. The surrounding hash/noise setup is now less than
2% of GPU kernel time in steady state, and the current register allocation
remains too high for a simple two-CTA-per-SM launch-bounds fix.

### Mine-only pipeline cleanup checks

Two small `hopper_mine_ws` cleanups were tested after the profile:

1. Move `load_tail()` from after each logical work tile to producer exit.
2. Skip the `MmaComplete` named-barrier arrive when `MineOnly=true`.

Both compiled in parallel and passed the forced-win pattern inspector:

```text
PATTERN_COMPATIBLE=true
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
```

Quick production-shape results:

| Experiment | Normalized attempts/s | Chance-weighted/s | Decision |
|---|---:|---:|---|
| post-fix profile baseline | 1,294,144 | 21.203B | reference |
| defer producer tail | 1,294,987 | 21.217B | neutral, rolled back |
| skip `MmaComplete` arrive | 1,267,775 | 20.771B | slower, rolled back |

The tail deferral is neutral because the current `SingleTileScheduler` gives
each CTA exactly one logical work tile, so there is no next tile to overlap
within the same CTA. The `MmaComplete` skip looked dead on paper but benchmarked
slower, likely from schedule perturbation. Neither change should be promoted.

Conclusion: production-shape kernel hash behavior is statistically sane. The
kernel is producing the expected number of lottery tickets, and their quality
matches the random-hash expectation closely enough that future kernel work
should focus on speed, not lottery-ticket correctness.
