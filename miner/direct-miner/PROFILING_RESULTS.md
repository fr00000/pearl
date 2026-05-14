# Kernel Profiling Results

## Setup

- Hardware: NVIDIA H100 80GB HBM3, driver 580.126.09
- CUDA: 12.8
- Shape: m=4096, n=8192, k=8192
- max_in_flight: 2
- Branch: `opt/direct-mining`, after Phase B (commit `aecd1fc6`)
- Profiling tools availability:
  - **nsys**: installed via `apt install cuda-nsight-systems-12-8`, version 2024.6.2. **Worked.**
  - **ncu**: pre-installed, version 2025.1.1.0. **Blocked** — `ERR_NVGPUCTRPERM`: the kernel module `nvidia` is built with `NVreg_RestrictProfilingToAdminUsers=1` (the RunPod default). Fixing it requires editing `/etc/modprobe.d/` and rebooting the host — not feasible in-session. No SM/memory/occupancy data could be collected.
  - **Python CUDA-event timing**: bespoke script `direct_miner.profile_run`. **Worked.**

The data below combines nsys (kernel-level breakdown over 60 s of steady-state production) and per-function CUDA events (30 iterations each, cache-on and cache-off).

## 1. Top kernels by GPU time — nsys, 60 s, B-cache ON

| % | Total time | Instances | Avg | Kernel |
|---:|---:|---:|---:|---|
| **61.9%** | 19.19 s | 45,120 | 425 µs | **`hopper_gemm_ws` (noisy_gemm main kernel)** |
| 12.3% | 3.82 s | 45,121 | 84.6 µs | `random_from_to_kernel` (int8 RNG for synthetic A) |
| 9.4% | 2.92 s | 45,120 | 64.8 µs | `NoisingKernelA` |
| 8.2% | 2.55 s | 45,121 | 56.5 µs | `direct_copy_kernel_cuda` (A.to(uint8) for tensor_hash) |
| 5.1% | 1.59 s | 45,121 | 35.2 µs | `MerkleTreeRootsKernel` (A-side tensor_hash) |
| 1.2% | 0.36 s | 45,121 | 7.9 µs | `ComputeBlakeMTKernel` (A-side inner blake) |
| 0.9% | 0.27 s | 45,120 | 5.9 µs | `NoiseGenerationKernel` (A-only path) |
| 0.4% | 0.13 s | 45,121 | 2.9 µs | `CommitmentHashFromMerkleRootsKernel` |
| 0.2% | 0.07 s | 45,121 | 1.6 µs | uniform RNG (A_scales) |
| 0.2% | 0.06 s | 45,121 | 1.2 µs | `Mul` epilogue |
| 0.000% | 82 µs | **1** | 82 µs | **`NoisingKernelB`** — fired once for the single B-cache miss |

Total profiled GPU time across all kernels: ~30.95 s of 60 s wall, i.e. GPU was actively executing kernels ~52% of wall. The remaining ~48% is gaps between kernels (CPU launch latency + waiting on host-signal-sync poll). Matches `cudaStreamSynchronize` taking 82.3% of CUDA API time (much of that is overlapped with kernel work via `max_in_flight=2`).

**Confirmation that the B-cache works as designed: `NoisingKernelB` ran exactly once, despite 45,120 matmuls.**

## 2. noisy_gemm internal metrics — ncu

Blocked by `ERR_NVGPUCTRPERM`. We have no SM-active %, no compute / memory / DRAM throughput, no achieved occupancy, no stall-reason breakdown. To unblock this would require host-level reboot with `nvidia.NVreg_RestrictProfilingToAdminUsers=0` (not in scope).

What we *can* infer indirectly from nsys + per-function timing:

- Each `hopper_gemm_ws` invocation is 425 µs (median 425 µs, std-dev only 2.8 µs — extremely consistent, so the kernel is steady-state shape-bound, not data-bound).
- The kernel is doing m × n × k × 2 = 549 GMACs per call. 549 GMACs / 425 µs = 1.29 PMAC/s = 2.58 PINT8-OPS/s. H100 INT8 tensor-core peak is ~1979 TOPS dense / 3958 TOPS sparse. So the GEMM portion alone reaches **~65% of dense INT8 peak** (or ~33% of sparse peak). Without ncu we can't separate noising work inside the main kernel from the GEMM proper, but a 65% INT8-tensor-core utilization is high for a fused kernel that also does noising + Blake hashing + reduction in the epilogue.

(Caveat: the 425 µs `hopper_gemm_ws` time includes the integrated noising-A invocation inside the same kernel, which is why a separate `NoisingKernelA` symbol also appears. The two are not double-counted — `NoisingKernelA` is a different launched kernel, not part of the 425 µs.)

## 3. Per-function CUDA-event timing (30 iterations each)

### Cache ON (steady-state Phase B)

| Phase | Avg µs | % of total |
|---|---:|---:|
| **noisy_gemm** | 508.5 | **58.2%** |
| A_tensor_hash | 129.2 | 14.8% |
| preamble_buffers | 97.9 | 11.2% |
| make_synthetic_a | 93.6 | 10.7% |
| hash_key_derive | 16.5 | 1.9% |
| noise_gen | 9.8 | 1.1% |
| commitment_hash | 6.7 | 0.8% |
| get_mining_job | 2.9 | 0.3% |
| schedule_status_check | 2.9 | 0.3% |
| setup_alloc | 2.9 | 0.3% |
| post_cache_populate | 2.9 | 0.3% |
| **TOTAL** | **874** | 100% |

### Cache OFF (Phase A path)

| Phase | Avg µs | % of total | Δ vs cache-on |
|---|---:|---:|---:|
| **noisy_gemm** | 593.7 | 52.3% | +85.2 µs (B-noising inside the kernel) |
| **B_tensor_hash** | 189.2 | 16.7% | +189.2 µs (saved entirely by cache) |
| A_tensor_hash | 114.0 | 10.0% | ≈ (noise; A-only) |
| preamble_buffers | 97.5 | 8.6% | ≈ |
| make_synthetic_a | 93.1 | 8.2% | ≈ |
| hash_key_derive | 16.1 | 1.4% | ≈ |
| noise_gen | 12.6 | 1.1% | +2.8 µs (B-side noise factors) |
| commitment_hash | 6.9 | 0.6% | ≈ |
| (others) | < 3 each | < 1% each | ≈ |
| **TOTAL** | **1135** | 100% | **+261 µs / +30%** |

### Comparison: B-cache hit vs miss

- Miss path:  1135 µs/iter → 881 matmuls/s ceiling
- Hit  path:   874 µs/iter → 1144 matmuls/s ceiling
- Cache saves **261 µs/iter (23%)**, attributable to:
  - `B_tensor_hash` eliminated entirely → 189 µs
  - B-side `noise_gen` skipped → ~3 µs
  - B-noising inside `noisy_gemm` skipped (kernel takes 85 µs less) → 85 µs

Predicted Phase B / Phase A throughput ratio: 1144 / 881 = **1.30×**. Measured (60-min prod): **1.256×**. The gap is the small extra cost of dispatch overhead being a larger share at higher throughput. Consistent.

## 4. Where 87% of theoretical peak is going

Reference: H100 INT8 dense peak for our shape is ~1979 TOPS → ~3608 matmul/s if every cycle were pure GEMM. Measured: ~896 matmul/s = 25%, *not* 13% as the original spec estimated. (The 13% / 7000-matmul/s figure assumed structured-sparse peak; the dense reality is ~3608.)

Per-iteration cost decomposition at cache-on steady state, with per-iter wall ≈ 1116 µs (1 / 896 s ≈ 1116 µs):

| Category | µs/iter | % of wall |
|---|---:|---:|
| `noisy_gemm` kernel (matmul + A-noising + Blake epilogue) | 508 | 46% |
| `make_synthetic_a` (curand int8 RNG of m×k = 32 M elems) | 94 | 8% |
| A-side `tensor_hash` (`A.to(uint8)` + MerkleTreeRoots) | 129 | 12% |
| Preamble buffer allocations + pinned-buffer acquire | 98 | 9% |
| `hash_key_derive` (CPU blake3 + small H2D copy) | 16 | 1% |
| Everything else (gateway query, noise_gen, commit_hash, post-launch bookkeeping, callback scheduling) | 29 | 3% |
| **Sum of timed phases** | **874** | **78%** |
| Unaccounted (CPU↔GPU scheduling, kernel launch gaps, max_in_flight=2 overlap noise) | ~242 | ~22% |

So the breakdown of where time goes, in plain English:

1. **The kernel itself is 46% of wall.** Inside that, the dense INT8 GEMM is doing real work at ~65% of H100 peak — already efficient.
2. **A-side preparation eats 30% of wall** — RNG (8%) + tensor_hash (12%) + allocations (9%) + small misc (1%). All of this is recomputed per iteration because A changes per call.
3. **The remaining ~22%** is gaps the timer can't see: CPU launch latency between kernels, `cudaStreamSynchronize` blocking on the in-flight event, and the cost of `max_in_flight=2` not perfectly overlapping CPU prep with GPU execution.

This is the data the original goal asked for: **the 75% gap from peak is not bandwidth-bound matmul, it is per-call A-side prep work and kernel-launch gaps.**

## 5. Verdict

**Phase C should target A-side overlap, not the kernel.**

Specific recommendations, with the numbers each one would attack:

1. **Multi-stream A-side prep (highest leverage, no kernel changes).**
   Generate A_next, hash it, and run noising_A on a second CUDA stream while the current iteration's `hopper_gemm_ws` is executing. Up to 222 µs of A-side prep (`make_synthetic_a` 94 µs + A-side `tensor_hash` 129 µs) can overlap with the 508 µs main kernel. Best-case throughput improvement: an additional ~25% (from 896 → ~1120 matmul/s) at the same shapes.
   Caveat: A-side `noisy_gemm` (the integrated NoisingKernelA inside the main kernel — 9.4% of GPU per nsys) still has to happen before the main mainloop tile starts; not all 222 µs is overlappable. Realistic gain: 15–20%.

2. **Persistent tensor pool for A-side per-call allocations (smaller, simpler).**
   The 98 µs `preamble_buffers` is mostly `torch.empty` for `ApEA` (32 MB int8) and `A_E_BL` (1 MB fp16) plus pinned-buffer acquire. PyTorch's caching allocator already amortizes most of this, so the gain is likely <5%, but a fixed pre-allocated pool keyed by shape would eliminate the residual.

3. **Lower-precision A RNG or torch-internal RNG path (small win).**
   `random_from_to_kernel` for a 32 M-element int8 tensor takes 84 µs (12.3% of GPU time in nsys). If `make_synthetic_a` could draw from a smaller bit-source and replicate (curand on m×k/4 then bit-expand), this could drop to ~25 µs. ~10% throughput improvement.

4. **Skipped — kernel-internal optimization.**
   Without ncu data we can't tell whether `hopper_gemm_ws` is memory- or compute-bound. At 65% of dense INT8 peak it is unlikely to have a large improvement available without a significant kernel rewrite. Do not pursue without first acquiring ncu access on a privileged host.

5. **Skipped — Multi-GPU / multi-process.**
   Out of scope. Single-GPU optimization ceiling has not been reached yet.

**Recommended single Phase C scope: implement Strategy 1 (multi-stream A overlap) and measure.** This is the only change with a ≥15% expected uplift on data we have. Combine with Strategy 3 only if Strategy 1 doesn't deliver — they share the same target (A-side prep).

## 6. Caveats

- ncu data unavailable. The above analysis treats `noisy_gemm` as a single 508 µs cost and cannot tell us whether multi-stream concurrency is even possible (i.e., whether the kernel already saturates the SMs and a second stream would compete for the same resources rather than overlap). If a privileged ncu run becomes possible, the first thing to check is achieved occupancy: > 70% means multi-stream will compete, < 50% means it will help.
- 30-iteration python-timing samples have noticeable noise (±5%). The 60-second nsys window is the more reliable kernel-level data; python timing is best used to *attribute* per-iteration GPU time to Python phases, not to compare runs to each other.
- All numbers are for `max_in_flight=2`. With `max_in_flight=1` the python-event timing should be the dominant view since there's no concurrency to hide; with `max_in_flight>2` the kernel-only view from nsys is more representative.
