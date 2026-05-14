# Phase C Results — Multi-Stream A-side Overlap

## Configuration
- Hardware: NVIDIA H100 80GB HBM3, driver 580.126.09
- Shapes: m=4096, n=8192, k=8192
- max_in_flight: 2
- B-cache: ENABLED
- Multi-stream: ENABLED (stream_main + stream_prep, both non-default)
- Diagnostics: ENABLED (`MINER_DEBUG=true`)
- Run duration: 3605.8s (~60 min)
- Run window: 2026-05-14 03:44:08 → 04:44:15 UTC

## Throughput comparison

| Run | Diag | Cache | Multi-stream | matmul/s | tile/s | Δ vs prev |
|---|---|---|---|---:|---:|---:|
| Phase A bare (b3a2806c notes) | off | off | off | ~905 | ~926,775 | — |
| Phase A + diagnostics (5-min verify) | on | off | off | 713.8 | 730,891 | — |
| Phase B + diag + cache (60-min) | on | on | off | 896.1 | 917,577 | **1.256× vs A+diag** |
| **Phase C + diag + cache + ms (60-min)** | on | on | on | **970.3** | **993,610** | **1.083× vs B** |

- Phase C / Phase B: 970.3 / 896.1 = **1.083×** (+8.3%)
- Phase C / Phase A bare: 970.3 / 905 = 1.072×
- Phase C / Phase A + diagnostics: 970.3 / 713.8 = **1.360×** (cumulative win from B+C)

The 5-minute smoke test caught a higher instant rate (**1008 matmul/s, +12.5%**) than the 1-hour sustained number. The smoke window happened to see only 1 cache invalidation (a single template throughout), whereas the 1-hour run hit 41 invalidations. Each invalidation forces the rare cache-miss path (B-side compute on stream_main with a cross-stream barrier) — small individual hit, but they accumulate. The sustained 1h number (970.3) is the honest figure.

This is below the profiling-report prediction of 15–20%. See the analysis below.

## Cache behavior

| Metric | Value |
|---|---:|
| Total matmuls | 3,498,777 |
| B-cache hits | 3,498,735 (99.9988%) |
| B-cache misses | 42 |
| Cache invalidations | 41 |
| Distinct templates seen | 42 |
| Mean template lifetime | ~86 s |

Identical pattern to Phase B (one miss per template change). The slot pool and multi-stream design preserves cache correctness — no extra invalidations introduced by stream concurrency.

## Correctness

| Check | Result |
|---|---|
| `Traceback` / `Exception` / `ERROR` | 0 |
| `CUDA error` / `OOM` / `Killed` | 0 |
| `invalid proof` / `rejected` | 0 |
| Cross-iteration slot contamination check | not directly observable — but proof verification under `MINER_DEBUG=true` reported 0 issues across 3.5M iterations |
| Cache integrity | preserved (one miss per template rotation, matches Phase B) |

Phase C was the first change to break default-stream ordering, the highest-risk change in the roadmap. 3.5 M iterations across the 1-hour run, plus 454 k smoke iterations, all completed without a single error.

## Block production

| Metric | Value |
|---|---:|
| Initial wallet balance | 0 |
| Final wallet balance | 0 |
| Gateway submission attempts | 0 |

Consistent with prior phases at `target_log2 ≈ 207.3`. Difficulty-bound; not a Phase C regression.

## Why 8.3% and not 15–20%?

The profiling report predicted that up to 222 µs of A-side prep could overlap with the 508 µs main kernel. The actual measured overlap is smaller than that. Specifically:

1. **`max_in_flight=2` caps overlap window.** With at most two iterations in flight, the prep stream serializes against itself: iter N+1's prep cannot start until iter N's prep finishes. So the overlap is one-iter-deep, not arbitrarily pipelined. To get the full ceiling, `max_in_flight=4` or higher is needed (each iter's prep runs concurrently with N+1's main kernel and N+2's pending start).

2. **A-side noising still happens inside `noisy_gemm`.** From nsys: `NoisingKernelA` is 9.4% of GPU time as a separate kernel launch on stream_main. This A-side work has to wait for the cross-stream barrier (`commitment_hash_A` ready) before launching, so it's not overlappable.

3. **Cache-miss path stalls overlap.** On a cache miss, B-side compute runs on stream_main with `stream_prep` waiting for `B_tensor_hash` via cross-stream sync. This serializes both streams for that iteration. Happens once per template (~88 s in this run) — small impact individually but consistent.

4. **Diagnostics overhead is a fixed CPU tax per matmul.** At 970 mm/s, JSON serialization + disk flushes consume more wall-clock per second than at 896 mm/s. The Phase B → Phase C improvement is partially absorbed by this. With `--no-diagnostics` (added but not used in this run), the same Phase C code should clear ~1100–1200 mm/s.

## Recommended next steps

1. **Run with `--no-diagnostics` to recover the instrumentation tax.** The `--no-diagnostics` flag was added but deliberately not enabled in this verification run — diagnostics are how we catch correctness bugs and this was the first run of new stream-ordering code. Now that Phase C is validated, a follow-up production run with diagnostics off should add another ~15–20% throughput on top of the 970 mm/s.

2. **Bump `max_in_flight` to 4 then 8.** Phase C with max_in_flight=2 only allows one-deep prep overlap. The slot pool already supports arbitrary num_slots. Higher in-flight depth gets us closer to the 1100–1200 mm/s ceiling implied by the profiling data, at the cost of (num_slots × ~190 MB) GPU memory.

3. **Investigate NoisingKernelA fusion or split.** 9.4% of GPU time spent in a separate launch that depends on commitment_hash_A is a candidate for either (a) absorbing into the prep stream by passing pre-allocated buffers and a different barrier strategy, or (b) merging into the main `hopper_gemm_ws` kernel as a fused noising-A epilogue. Both are non-trivial — only worth it if the multi-GPU path doesn't apply.

4. **Multi-GPU scaling.** If single-GPU throughput is near its ceiling, the next 5–10× comes from running this miner on N GPUs in parallel. Out of scope for this branch but a natural next phase.

## Verdict

**Phase C: validated win.** +8.3% sustained throughput over Phase B, zero correctness incidents across 3.5 M iterations. Below the most-optimistic profiling prediction but above the "worth keeping" threshold (≥5%). The architecture (two non-default streams, per-iter slot pool, cross-stream event barriers, optional external-event tracker) is now in place and unlocks the larger wins from `--no-diagnostics` and higher `max_in_flight` that follow.
