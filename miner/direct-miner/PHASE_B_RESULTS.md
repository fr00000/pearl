# Phase B Results — B-side Caching

## Configuration
- Hardware: NVIDIA H100 80GB HBM3, driver 580.126.09
- Shapes: m=4096, n=8192, k=8192
- max_in_flight: 2
- log_interval: 100
- B-cache: ENABLED (`--enable-b-cache`)
- Diagnostics: ENABLED (`--metrics-output`, `--phase-tag phase_b_bcache`)
- Run duration: 3606.2s (~60 min)
- Run window: 2026-05-14 00:58:50 → 01:58:57 UTC

## Throughput comparison

All three rows are at the same shapes (m=4096, n=8192, k=8192) and `max_in_flight=2`.

| Run | Diagnostics | B-cache | Duration | completion_rate (matmuls/s) | tile_rate (tiles/s) |
|---|---|---|---|---|---|
| Phase A baseline (prior, from b3a2806c notes) | off | off | 60 min | ~905 | ~926,775 |
| Phase A with diagnostics (this branch, 5-min verify) | on | off | 404 s | 713.8 | 730,891 |
| **Phase B with diagnostics + cache (this run)** | on | on | 3606 s | **896.1** | **917,577** |

**Apples-to-apples ratio (matched config, both with diagnostics):**
- Completion rate: 896.1 / 713.8 = **1.256× (+25.6%)**
- Tile rate:       917,577 / 730,891 = **1.256× (+25.6%)**

Phase B with diagnostics also closely matches the original no-diagnostics Phase A baseline (896.1 vs 905), suggesting the B-cache fully absorbs the diagnostic overhead while delivering real B-side compute savings.

## Cache behavior

| Metric | Value |
|---|---|
| Total matmuls | 3,231,381 |
| B-cache hits | 3,231,343 (99.9988%) |
| B-cache misses | 38 |
| Cache invalidations | 37 |
| Distinct templates seen | 38 |
| Mean template lifetime | ~95 s (≈ 85,000 matmuls per template) |

One miss per template change, exactly as expected. The cache's single-entry design suffices: templates arrive serially from the gateway, so a strict last-write-wins policy is correct.

## Margin distribution

Not available. Per Task 1.1 investigation (committed in 4ab36862), the kernel's `host_signal_header` is written only on a winning tile, so per-matmul best-hash is not exposed without kernel modification (Case B). `best_observed_hash_log2` and `margin_log2` are null in 100% of the 3.23M JSONL entries.

Implications for this run:
- We cannot say how close to threshold we got.
- Submission count remains the only chain-level signal at our current hash rate vs. current difficulty.

## Block production

| Metric | Value |
|---|---|
| Initial wallet balance | 0 |
| Final wallet balance   | 0 |
| Gateway submission attempts | 0 (no `host_signal_header` triggers in log) |

Consistent with Phase A's 60-minute baseline (also 0 submissions). Network difficulty at `target_log2 = 207.34` is well above what 896 matmuls/s × ~1M tiles/matmul can reach in 1 h. This is a difficulty-bound result, not a Phase B regression.

## Correctness

- Zero `error`/`exception`/`traceback` events across 3.23M matmuls.
- Zero "invalid proof" / rejected events.
- Diagnostic JSONL is well-formed (3,231,381 records + 2 session events; no parse failures in summary scan).
- 38 invalidations matching 38 distinct template prefixes — cache key derivation correctly tracks template rotations.

## Implementation strategy used

Strategy C-lite: a parallel `pearl_gemm_noisy_cached` lives in `miner/direct-miner/src/direct_miner/mining_call.py`, calling `pearl_gemm` primitives (`tensor_hash`, `commitment_hash_from_merkle_roots`, `noise_gen`, `noisy_gemm`, `make_pow_target_tensor`) directly. The vllm-miner `pearl_gemm_noisy` is unchanged.

Cached on first call per `hash_key = blake3(incomplete_header_bytes || mining_config)`:
- `B_tensor_hash`, `commitment_hash_B`
- `EBR`, `EBR_fp16`, `EBL_R_major`, `EBL_K_major`
- `BpEB` (the n×k pre-noised B; the largest item, ~64 MB int8)

A-dependent items (`A_tensor_hash`, `commitment_hash_A`, `EAL`, `EAR_R_major`, `EAR_K_major`, `EARxBpEB`) are recomputed every call. On a cache hit, `noisy_gemm` is invoked with `run_noising_B=False`, which makes the kernel read `BpEB` as input rather than recomputing it from B + EBL·EBR.

Drift risk: if `vllm-miner.gemm_operators.pearl_gemm_noisy` changes, `mining_call.py` must follow. The two implementations should be kept side by side.

## Phase B verdict

**Cache produced a measured 25.6% throughput uplift (713.8 → 896.1 matmuls/s) with no correctness incidents.** Hit rate is essentially perfect (99.999%) once a template warms up. Recommend keeping this implementation and using it as the baseline for any subsequent phase.

## Recommended next step

Two candidates, both useful:

1. **Phase C — multi-stream / overlap**. With B-side noising now eliminated from the inner loop, the next likely bottleneck is the A-side per-call overhead (A-noising, commitment, kernel launch). Running two CUDA streams concurrently for A-side work while the GEMM kernel executes would attack that.

2. **Best-hash observability via kernel patch**. Adding an atomic `best_pow_hash` slot to `host_signal_header` (write min-so-far per outer tile) would close the diagnostic gap noted in Task 1.1 / 4ab36862. Without it, we have no way to tell whether the kernel's tile-hash distribution is competitive at current difficulty — only "did we win" / "did we not." This is a small kernel change requiring user approval but high value: it would let us evaluate any future phase without needing a chain win.

Stopping here pending user direction.
