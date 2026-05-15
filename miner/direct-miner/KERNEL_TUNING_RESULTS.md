# Direct Miner Kernel Tuning Results

Last updated: 2026-05-15

This file is the quick reference for direct-miner kernel tuning. The full
benchmark notes live in `H100_SXM_VERIFY.md`.

## Current Production Recommendation

Keep the production direct-miner kernel at:

```bash
--m 8192 --n 261888 --k 16384 \
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

Do not pass `--kernel-mma-registers` by default.

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

Conclusion: lower `k` is a trap because the protocol target scales with
`rounded_common_dim`. The best measured shape is `8192x261888x16384`, which
trades about half the normalized attempt rate for double the target adjustment
and nets a confirmed **+3.22% expected mining chance**.

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

Conclusion: no production change. The quick sweep did not show a durable gain
from smaller `n`, larger/smaller `m`, or changing `max_in_flight`. Keep
`m=8192`, `n=261888`, `k=16384`, `max_in_flight=4`.

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

Conclusion: runtime kernel-config tuning is effectively exhausted around this
shape. The only measured improvement, `mma_registers=160`, is below 1% and not
worth adding production config surface.

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
