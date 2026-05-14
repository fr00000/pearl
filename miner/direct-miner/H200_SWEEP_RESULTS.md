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
