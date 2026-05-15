# H200 SXM Sweep Results

## Hardware
- **4× NVIDIA H200** (143,771 MiB HBM3e each — listed as "H200" not "H200 SXM" in `nvidia-smi`, but power/memory profile consistent with SXM)
- Compute capability: sm_90 (Hopper)
- Driver: 550.144.03
- nvcc: CUDA 12.8 (release V12.8.93)
- PyTorch: 2.11.0+cu129 (venv-bundled libs override system CUDA 12.8 for runtime)
- Host: RunPod, Ubuntu 24.04.4, 192 cores, 2 TB RAM

## Sweep design
- **5 shapes × 3 max_in_flight values = 15 cells**, 5 minutes each
- Distributed round-robin across all 4 GPUs (initial 11 cells in ~20 min; 4 missing cells re-run in parallel, ~5 min)
- Production mode throughout (`--enable-b-cache`, diagnostics OFF)
- All 4 miners share one pearl-gateway socket — matches prod conditions
- All 15 cells completed with zero errors

## Results (sorted by tile rate, all 15 cells)

| Shape          | m     | n     | k    | mif | mm/s    | tile/s        | err |
|----------------|-------|-------|------|-----|---------|---------------|-----|
| **s_xl**       | 8192  | 65536 | 8192 | **8** | 124.5 | **2,039,397** | 0   |
| s_xl           | 8192  | 65536 | 8192 | 4   | 124.3   | 2,036,430     | 0   |
| s_xl           | 8192  | 65536 | 8192 | 2   | 124.3   | 2,036,040     | 0   |
| s_xxl          | 16384 | 32768 | 8192 | 4   | 122.2   | 2,001,835     | 0   |
| s_xxl          | 16384 | 32768 | 8192 | 2   | 121.9   | 1,997,809     | 0   |
| s_max_h100     | 8192  | 32768 | 8192 | 8   | 243.0   | 1,990,747     | 0   |
| s_max_h100     | 8192  | 32768 | 8192 | 2   | 242.7   | 1,987,927     | 0   |
| s_xxl          | 16384 | 32768 | 8192 | 8   | 120.4   | 1,972,413     | 0   |
| s_max_h100     | 8192  | 32768 | 8192 | 4   | 238.7   | 1,955,570     | 0   |
| s_winner_h100  | 4096  | 32768 | 8192 | 4   | 475.3   | 1,946,883     | 0   |
| s_winner_h100  | 4096  | 32768 | 8192 | 8   | 471.1   | 1,929,565     | 0   |
| s_winner_h100  | 4096  | 32768 | 8192 | 2   | 467.1   | 1,913,257     | 0   |
| s_baseline     | 4096  | 8192  | 8192 | 8   | 1612.3  | 1,651,023     | 0   |
| s_baseline     | 4096  | 8192  | 8192 | 4   | 1589.8  | 1,627,947     | 0   |
| s_baseline     | 4096  | 8192  | 8192 | 2   | 1526.1  | 1,562,769     | 0   |

## Best max_in_flight per shape

| Shape          | best mif | tile/s        | notes                                                |
|----------------|----------|---------------|------------------------------------------------------|
| s_xl           | 8        | 2,039,397     | mif=2/4/8 all within 0.2% — pipeline depth saturated |
| s_xxl          | 4        | 2,001,835     | mif=8 worse than mif=4 — diminishing returns         |
| s_max_h100     | 8        | 1,990,747     | mif=2 essentially tied; mif=4 oddly lower (noise)    |
| s_winner_h100  | 4        | 1,946,883     | mif=4 narrowly beats mif=8                           |
| s_baseline     | 8        | 1,651,023     | Small shape — mif monotonically helps                |

For the winning s_xl shape, mif=2, 4, and 8 are statistically tied (≤0.2% spread). Picking mif=8
is essentially arbitrary; mif=4 would use less memory and produce indistinguishable rates.

## Cross-platform comparison

| Reference                                   | tile/s        | Notes                          |
|---------------------------------------------|---------------|--------------------------------|
| H100 PCIe winner (4096×32768×8192, mif=4)   | 1,935,876     | Prior production               |
| H200 same shape, same mif (this sweep)      | 1,946,883     | **1.01×** — essentially tied   |
| H200 winner (s_xl 8192×65536×8192, mif=8)   | 2,039,397     | **1.05×** vs H100 winner       |

H200's compute throughput and bandwidth per GPU are comparable to H100 at shapes that fit on
both. The advantage shows up only at the wider shape (`n=65536`) that doesn't fit in H100's 80 GB
budget. The bump is modest (~5%) — the kernel is not memory-bound enough to fully exploit the
extra HBM3e bandwidth, and tile rate appears to plateau around 2M tiles/s on this generation of
Hopper hardware regardless of which "big" shape we pick.

## Memory headroom
- s_xl winner (8192×65536×8192, mif=8): peak GPU memory ~30–40 GB
- Headroom on 143 GB H200: ~100+ GB unused

Larger candidates (e.g. n=131072) might fit but were not tested — the tile rate plateaued, so
further memory expansion is unlikely to help.

## Errors / anomalies
- **Zero CUDA / OOM / sanity-check failures** across all 15 cells
- All cells hit B-cache hit rate ≈ 100% (1 miss = first iteration, all subsequent hit)
- One benign `terminate called without an active exception` at every cell's shutdown (C++
  destructor ordering; no effect on results)
- An initial bash arithmetic bug in `sweep_h200.sh` made 4 cells skip on the first pass — the
  bug is now fixed (`grep -c "..." || echo "0"` produced multi-line "0\n0" when no matches; now
  uses `grep -c "..." 2>/dev/null; true` with `${var:-0}` fallback). All 15 cells now collected.

## Recommendation — production config for 4× H200

```
m            = 8192
n            = 65536
k            = 8192
max_in_flight = 8
flags        = --enable-b-cache
```

- Per-GPU rate: **124.5 mm/s, 2,039,397 tiles/s**
- 4-GPU projected aggregate (linear scaling): **498 mm/s, 8,157,588 tiles/s**
- Verify the aggregate at launch — if it falls short of ~8M tiles/s, pearl-gateway is the
  bottleneck and we should investigate

Note: mif=4 produces statistically equivalent throughput (2,036,430 t/s) at lower memory cost
(~half the in-flight buffers). Either is fine; mif=8 is what the launcher is configured for.

## Caveats
- 5-min runs are statistically noisy at the few-percent level; cells within ±2% are tied
- Tile rates may shift slightly under sustained production load (4 miners share one pearl-gateway),
  but the sweep already ran 4-up so contention is partially baked in
- Block production is governed by network share and target difficulty — high tile rate doesn't
  guarantee blocks won

---

# Stress Sweep Addendum

## Motivation
The initial H200 sweep showed tile rate plateauing at ~2M tiles/sec/GPU across all five shapes. This addendum pushes the boundaries — `n` up to 262144, `m` up to 32768, and `max_in_flight` up to 16 — to determine whether the plateau is a hard kernel-throughput limit or whether further gains are available with bigger workloads.

## Method
- 8 stress cells × 5 min each, parallel 4-up across all 4 GPUs (2 batches)
- Wall time: ~12 minutes
- Production mode (`--enable-b-cache`, diagnostics off)
- All cells share one pearl-gateway socket — matches prod conditions and initial-sweep methodology
- Memory poller (`nvidia-smi`, 5-second cadence) captured peak GPU memory per cell

## Pre-flight outcomes
- All 3 highest-risk cells passed 30-second pre-flight (stress_n256, stress_m32, mif16_max). No cells excluded.

## Cells (sorted by tile rate)

| Cell              | m     | n      | k    | mif | tiles/mm | mm/s   | tiles/s     | peak    | err |
|-------------------|-------|--------|------|-----|----------|--------|-------------|---------|-----|
| **stress_n256**   | 8192  | 262144 | 8192 | 4   | 65536    | 32.4   | **2,123,968** | 24.2 GB | 0   |
| stress_n128       | 8192  | 131072 | 8192 | 4   | 32768    | 64.4   | 2,108,646   | 12.9 GB | 0   |
| stress_max        | 16384 | 131072 | 8192 | 4   | 65536    | 31.9   | 2,088,658   | 20.9 GB | 0   |
| stress_n128_mif8  | 8192  | 131072 | 8192 | 8   | 32768    | 63.6   | 2,085,186   | 21.1 GB | 0   |
| stress_both       | 16384 | 65536  | 8192 | 4   | 32768    | 63.4   | 2,075,854   | 11.3 GB | 0   |
| mif16_xl          | 8192  | 65536  | 8192 | 16  | 16384    | 126.0  | 2,063,717   | 20.6 GB | 0   |
| mif16_max         | 16384 | 65536  | 8192 | 16  | 32768    | 62.5   | 2,047,023*  | 38.7 GB | 0   |
| stress_m32        | 32768 | 16384  | 8192 | 4   | 16384    | 112.5  | 1,843,139   | 7.2 GB  | 0   |

\* `mif16_max` data is from its last in-flight log line, not the FINAL line: SIGINT from the timeout interrupted the drain `synchronize()` call before FINAL printed. The run itself was clean (18,200 matmuls completed at a steady 2,047k tile/s in the 5 minutes before SIGINT) and the rate is mid-run-average, comparable to other cells.

## mif scaling at fixed shape (s_xl: 8192×65536×8192)

| mif | tile/s        | Δ vs mif=8 |
|-----|---------------|------------|
| 2   | 2,036,040     | -0.2%      |
| 4   | 2,036,430     | -0.1%      |
| 8   | 2,039,397     | (baseline) |
| 16  | 2,063,717     | +1.2%      |

**Pipeline saturated.** mif=16 is within run-to-run noise (±2%) of mif=8. Doubling pipeline depth past 8 does not help on this kernel.

## Cross-cell pattern
All "wider-n" cells (stress_n128 / stress_n128_mif8 / stress_n256 / stress_max / stress_both) cluster between **2.07M and 2.13M tiles/s** — a 3% spread. The +4.1% over the prior winner is consistent across this group, not a single outlier. So the gain is real but small.

The exception is **stress_m32** (large m=32768, narrow n=16384) at 1.84M — ~10% worse than every wider-n cell. Large m + narrow n is not a productive combination for this kernel.

## Peak memory observations
- Largest cell (mif16_max): 38.7 GB / 143 GB → 104 GB headroom
- Predicted ~112 GB for mif16_max in the spec was 3× too high; actual is 39 GB
- Memory is not a binding constraint at any tested shape
- 100+ GB of HBM3e per GPU is unused at every tested workload

## Verdict
**Marginal gain (+4.1%).** stress_n256 beats prior winner. The gain is real (multiple wider-n cells cluster at 2.08–2.12M) but small. Cost of adoption is trivial: same m and k, same `max_in_flight`, 4× wider n, 7 GB extra memory.

`max_in_flight=16` brings nothing. Pipeline depth past 8 is wasted.

## Updated production config

```
m            = 8192
n            = 262144     ← was 65536
k            = 8192
max_in_flight = 4         ← was 8 (statistically indistinguishable from 4)
flags        = --enable-b-cache
```

- Per-GPU rate: **32.4 mm/s, 2,123,968 tiles/s** (+4.1% over prior)
- 4-GPU projected aggregate: **129.6 mm/s, 8,495,872 tiles/s**
- Peak memory: ~24 GB per GPU (well within 143 GB)

Note: matmul rate (mm/s) is 4× lower than at the prior winner shape because each matmul covers 4× more tiles. The optimization target is tile rate; matmul rate is a derived counter.

## Caveats
- 5-min runs have ±2-3% variance; cells within 3% of each other are statistically tied
- The 4% bump is at the edge of where I'd want a longer (30-min) verification before adopting in production — recommend a brief replicated re-run at this shape before declaring it the new baseline
- The kernel is compute-bound at ~2M tiles/sec/GPU on H200 — further bandwidth or memory expansion is unlikely to help. Future gains will come from kernel work, not shape tuning.

---

## Post-headless/kernel-sweep launcher note

The production launcher now defaults to the latest measured Hopper direct-miner
winner from `H100_SXM_VERIFY.md`:

```bash
--m 8192 --n 524032 --k 8192 \
--max-in-flight 4 \
--enable-b-cache \
--enable-headless-kernel \
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1
```

That configuration sustained **2,503,688 raw outer-tiles/s** on H100 SXM and is
the best measured normalized PoW-attempt setting. It has not yet been
separately re-swept on H200 after the headless/tile-kernel changes, but it is
the best measured direct-miner setting and is safe to override via
`SHAPE_*`, `MAX_IN_FLIGHT`, and `KERNEL_*` environment variables if H200 needs a
local confirmation sweep.
