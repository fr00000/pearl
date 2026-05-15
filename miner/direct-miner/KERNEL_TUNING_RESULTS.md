# Direct Miner Kernel Tuning Results

Last updated: 2026-05-15

This file is the quick reference for direct-miner kernel tuning. The full
benchmark notes live in `H100_SXM_VERIFY.md`.

## Current Production Recommendation

Keep the production direct-miner kernel at:

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

Do not pass `--kernel-mma-registers` by default.

## Metric Rule

Use `normalized_attempt_rate`, not raw outer-tile rate, for kernel decisions.

Raw outer-tile rate counts CTA completions. It is not always equal to mining
lottery tickets because each CTA has a different number of MMA consumer threads
when `tile_m` changes. The comparable rate is:

```text
normalized_attempt_rate = raw_outer_tile_rate * (mma_consumer_threads_per_cta / 256)
```

For the current `tile_m=128` production kernel, raw outer-tile rate and
normalized attempts are equal. For `tile_m=64`, normalized attempts are half the
raw outer-tile rate.

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
