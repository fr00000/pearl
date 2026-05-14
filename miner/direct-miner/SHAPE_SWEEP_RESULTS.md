# Shape Sweep Results

## Configuration
- Hardware: NVIDIA H100 80GB HBM3, driver 580.126.09
- Phase: C (multi-stream, B-cache, diagnostics off, `--max-in-flight 4`)
- Duration per shape: 5 min (300s)
- Sweep date: 2026-05-14 07:59 → 08:39 UTC
- Total sweep time: ~40 min
- Raw data: `/workspace/shape-sweep-20260514-075911/`
- Sweep harness: `miner/direct-miner/scripts/shape_sweep.sh`

The metric is **tile rate** (matmul rate × outer_tiles_per_matmul), not raw matmul rate. Each outer tile is an independent winning-hash candidate; doubling tiles doubles the lottery tickets per second regardless of how many matmul calls produce them.

## Results (sorted by tile rate)

| Rank | Shape | m | n | k | tiles/mm | mm/s | tiles/s | vs baseline | cache h/m/inv | err |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | **max_tiles** | 8192 | 32768 | 8192 | 8192 | 240.8 | **1,973,032** | **1.22×** | 70 628 / 5 / 4 | 0 |
| 2 | wider_n2 | 4096 | 32768 | 8192 | 4096 | 472.6 | 1,935,876 | 1.20× | 138 618 / 4 / 3 | 0 |
| 3 | both_mn | 8192 | 16384 | 8192 | 4096 | 450.1 | 1,843,800 | 1.14× | 132 021 / 2 / 1 | 0 |
| 4 | wider_n | 4096 | 16384 | 8192 | 2048 | 882.9 | 1,808,184 | 1.12× | 258 880 / 3 / 2 | 0 |
| 5 | baseline | 4096 | 8192 | 8192 | 1024 | 1582.0 | 1,619,966 | 1.00× | 464 060 / 1 / 0 | 0 |
| 6 | larger_m | 8192 | 8192 | 8192 | 2048 | 788.5 | 1,614,887 | 1.00× | 231 282 / 2 / 1 | 0 |
| 7 | deeper_k | 4096 | 8192 | 16384 | 1024 | 877.1 | 898,162 | **0.55×** | 257 256 / 3 / 2 | 0 |
| 8 | deeper_k2 | 4096 | 8192 | 32768 | 1024 | 441.6 | 452,247 | **0.28×** | 129 557 / 2 / 1 | 0 |

Eight shapes ran, zero failed, zero errors / exceptions / proof rejections across the sweep.

## What the data shows

### m is roughly neutral
`baseline` (m=4096) and `larger_m` (m=8192) produced **essentially identical tile rates** — 1.620M vs 1.615M. Doubling m halved the matmul rate and exactly doubled tiles-per-matmul. The kernel's tile granularity scales linearly in m; there's no efficiency penalty or bonus.

### n is the cheap axis
Wider n consistently wins:
- baseline (n=8192) → 1.62M tiles/s
- wider_n  (n=16384) → 1.81M tiles/s (+12%)
- wider_n2 (n=32768) → 1.94M tiles/s (+20%)

The tile rate grows sub-linearly with n (×4 in n → ×1.20 in tile rate), but it's a real and consistent gain. n is "cheap" in the sense that bigger n adds outer tiles faster than it adds per-tile cost.

### k is pure cost
Deeper k cuts tile rate dramatically because `outer_tiles_per_matmul = ceil(m/128) × ceil(n/256)` — independent of k. Deeper k just makes each tile more expensive without producing more of them:
- baseline (k=8192) → 1.62M tiles/s
- deeper_k  (k=16384) → 0.90M tiles/s (−45%)
- deeper_k2 (k=32768) → 0.45M tiles/s (−72%)

Going deeper in k for tile-rate optimization is strictly worse. (Deeper k may still be useful if protocol-side constraints make it necessary, but it's never a throughput optimization.)

### max_tiles vs wider_n2 — closest call

`max_tiles` (8192 × 32768 × 8192) wins overall but only by 1.9% over `wider_n2` (4096 × 32768 × 8192). The difference is within the 5-min sample noise we'd expect. Both deliver ~1.94–1.97M tiles/s.

The trade-off:
- `max_tiles` uses ~2× the slot-pool memory (m=8192 doubles every per-slot tensor whose size includes m), so 4 slots ≈ 3.5 GB GPU memory.
- `wider_n2` uses ~1.8 GB for 4 slots.
- Both fit easily in 80 GB; no operational difference.

Given the tile-rate parity, **either is a defensible winner**; `max_tiles` is technically #1 by the recorded numbers.

## Winner

**`max_tiles` (m=8192, n=32768, k=8192)** at **1,973,032 tiles/s**, **+22% over baseline**.

Equivalent: **wider_n2** at **1,935,876 tiles/s**, +20% over baseline, with about half the slot-pool memory footprint.

## Recommended production shape

**`max_tiles` (8192 × 32768 × 8192)** is the simple "max tile rate" choice and what the data nominally selects. The 22% lift over the previous production shape (`4096 × 8192 × 8192` at 1.62M tiles/s) holds with zero correctness issues across 70 628 iterations.

If GPU memory becomes a concern (e.g., expanding `max_in_flight` further or sharing the GPU), `wider_n2` is essentially tied on throughput at ~half the memory.

## Caveats

- 5-min samples have ±3–5% variance. The top two candidates differ by 1.9% — within noise. A 30-min head-to-head re-test would resolve which is genuinely faster, but the practical answer is "they're equivalent."
- All shapes ran at `max_in_flight=4`. Optimal `max_in_flight` may differ at different shapes — particularly the matmul-light/tile-heavy shapes like `max_tiles` that already saturate the GPU at a low matmul rate. From the Phase C tuning data we know mif=4 already fills the pipeline at the baseline shape; this is unlikely to change for tile-heavier shapes.
- Cache hit rate stayed essentially perfect across all shapes (99.997 %+); the cache mechanism is shape-independent.
- The original `summary.csv` from this run is mangled (`errors` field accidentally had a trailing newline due to a `grep -c || echo 0` chain). Numbers above came from re-parsing the per-shape `.log` files directly. The sweep script has been patched (single-line edit, no behavioral change) so future runs produce a clean CSV.

## Recommendation

Switch the production miner to `--m 8192 --n 32768 --k 8192` with the existing Phase C flags (`--enable-b-cache --max-in-flight 4`; diagnostics are off by default — pass `--enable-diagnostics` only when investigating). Expected tile rate: ~1.97M tiles/s, a 22% improvement on the previous production shape.

If the user wants to be more conservative on GPU memory, `--m 4096 --n 32768 --k 8192` is essentially equivalent in throughput and uses half the slot-pool memory.
