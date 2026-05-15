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
