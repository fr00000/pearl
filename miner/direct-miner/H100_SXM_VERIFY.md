# H100 SXM Quick Verification

Does the H200 stress sweep winner (`stress_n256` = 8192 × 262144 × 8192) carry over to H100 SXM, or does H100's narrower HBM bandwidth shift the optimum?

## Configuration

- Hardware: 1× NVIDIA H100 80GB HBM3, driver 580.126.09, compute 9.0
- Date: 2026-05-14 ~15:49–16:07 UTC
- Duration per cell: 5 min (293 s effective after warmup)
- Production mode: `--enable-b-cache --max-in-flight 4`, diagnostics off
- Stack: pearld synced (blocks 52175), oyster + pearl-gateway running

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

## Production config for H100 SXM (if used)

- Shape: `--m 8192 --n 262144 --k 8192`
- max_in_flight: 4
- `--enable-b-cache`
- Diagnostics off (default after the recent flag rename)
- Expected per-GPU tile rate: **~2.1 M tiles/s** sustained

## Caveats

- 5-min samples have ~±2-3 % variance. The 1.5 % spread between stress_n128 and stress_n256 is at the edge of resolvable. Either is a defensible operational choice if stress_n256 ever runs into a memory wall (e.g., when more processes share the GPU).
- This was on a single H100 SXM with no other GPU contention. Multi-tenant pods may differ.
- pearld was fully synced; mining was against live templates (gateway saw a single template through cell 1, four rotations through cells 2 and 3).
