# H100 SXM Quick Verification

Does the H200 stress sweep winner (`stress_n256` = 8192 × 262144 × 8192) carry over to H100 SXM, or does H100's narrower HBM bandwidth shift the optimum?

## Configuration

- Hardware: 1× NVIDIA H100 80GB HBM3, driver 580.126.09, compute 9.0
- Date: 2026-05-14 ~15:49–16:07 UTC
- Duration per cell: 5 min (293 s effective after warmup)
- Production mode: `--enable-b-cache --max-in-flight 4`, diagnostics off
- Stack: pearld synced (blocks 52175), oyster + pearl-gateway running

## Current production winner

As of the 2026-05-15 headless/kernel sweep, the recommended H100/Hopper
direct-miner command is:

```bash
uv run direct-miner \
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

Confirmed 5-minute rate: **2,503,688 raw outer-tiles/s per GPU**, which is
also **2,503,688 normalized 128-tile-equivalent attempts/s**. The startup
scripts use this by default.

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

## Historical production config for H100 SXM

- Shape: `--m 8192 --n 262144 --k 8192`
- max_in_flight: 4
- `--enable-b-cache`
- Diagnostics off (default after the recent flag rename)
- Expected per-GPU tile rate: **~2.1 M tiles/s** sustained

This was the best setting before the dedicated headless kernel, wider
`n=524032` shape, and `128x256x128, c2x1` cluster tuning. Keep it only as a
conservative fallback.

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

## Addendum: mine-only A-noising specialization

On 2026-05-15 we tested the next conservative kernel change: make the A-side noising kernel skip `AxEBL` work when invoked through `headless_mine`. In mine-only mode, `AxEBL` is not consumed by the dedicated mining matmul path, so the specialization avoids:

- EBL TMA loads in `NoisingKernelA`
- `A * EBL` WGMMA work
- `AxEBL` TMA stores

The normal non-mine path still instantiates and dispatches the original `ComputeAxEBL=true` kernel. The new specialization was built on the H100 pod with:

```bash
MAX_JOBS=8 PEARL_GEMM_FORCE_BUILD=TRUE uv sync --all-packages \
  --reinstall-package pearl-gemm-build-utils \
  --reinstall-package pearl-gemm
```

### 60-second verification

Each cell used production mode:

```bash
uv run direct-miner \
  --m <m> --n <n> --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --log-interval 100
```

| Cell | m | n | k | mm/s | tile rate (tiles/s) | cache h/m/inv | result |
|---|---:|---:|---:|---:|---:|---|---|
| prior base_n256 | 8192 | 262144 | 8192 | 34.9 | 2 286 220 | n/a | clean |
| mine-only A skip, n256 | 8192 | 262144 | 8192 | 34.9 | 2 286 404 | 2379 / 1 / 0 | clean |
| prior n512minus | 8192 | 524032 | 8192 | 17.6 | 2 300 118 | n/a | clean |
| mine-only A skip, n512minus | 8192 | 524032 | 8192 | 17.6 | 2 300 330 | 1191 / 2 / 1 | clean |

Result: **flat**. The measured deltas are ~+0.01%, far inside one-minute noise.

Conclusion: `AxEBL` work is real wasted work in the headless path, but it is not the current throughput limiter at the saturated direct-mining shapes. Keep the specialization if we want the code path to be semantically tighter, but do not expect a measurable production gain from this alone. The next meaningful kernel work should target either the actual mine matmul/inner-hash path or the `tensor_hash` grid-size ceiling that blocks exact `n=524288` and larger B tensors.

## Addendum: mine-kernel profile and cluster autotune

On 2026-05-15 we profiled the current headless path with Nsight Systems on the H100 pod:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --delay=12 --duration=25 \
  -o /workspace/nsys_headless_n256 \
  uv run direct-miner \
    --m 8192 --n 262144 --k 8192 \
    --max-in-flight 4 \
    --enable-b-cache \
    --enable-headless-kernel
```

Kernel time split from `nsys stats --report cuda_gpu_kern_sum`:

| Kernel | GPU time share | Avg time |
|---|---:|---:|
| `hopper_mine_ws` | 98.2% | 28.34 ms |
| A random fill | 0.8% | 0.22 ms |
| A `MerkleTreeRootsKernel` | 0.6% | 0.16 ms |
| `NoisingKernelA` | 0.4% | 0.11 ms |
| everything else | <0.2% | n/a |

Conclusion: the current bottleneck is overwhelmingly the mine matmul/inner-hash kernel. CUDA launch overhead and A-side prep are not material.

### Fixed-tile cluster/stage sweep

We then compiled a focused autotune grid that preserves the `128x256x128` tile shape and only varies pipeline stages plus cluster shape. This keeps the proof row/column extraction pattern aligned with the default mining configuration.

Each short cell used:

```bash
uv run direct-miner \
  --m 8192 --n 262144 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-stages <stages> \
  --kernel-cluster-m <cM> \
  --kernel-cluster-n <cN> \
  --log-interval 200
```

| Variant | Stages | Cluster | tile rate (tiles/s) | Result |
|---|---:|---:|---:|---|
| baseline | 3 | 1x1 | 2 282 247 | clean |
| s4_c1x1 | 4 | 1x1 | 2 184 087 | slower |
| s5_c1x1 | 5 | 1x1 | 0 | invalid: 241 KiB smem > 227 KiB limit |
| **s3_c2x1** | **3** | **2x1** | **2 488 762** | **winner** |
| s3_c1x2 | 3 | 1x2 | 2 381 428 | faster than baseline |
| s3_c2x2 | 3 | 2x2 | 2 391 889 | faster than baseline |
| s4_c2x1 | 4 | 2x1 | 2 473 038 | near winner |
| s4_c1x2 | 4 | 1x2 | 2 377 482 | faster than baseline |
| s4_c2x2 | 4 | 2x2 | 2 390 194 | faster than baseline |

The best fixed-tile kernel is:

```bash
--kernel-stages 3 --kernel-cluster-m 2 --kernel-cluster-n 1
```

### Winner validation on memory-heavy production shape

One-minute validation on the previous best shape:

| Shape | Kernel | tile rate (tiles/s) | cache h/m/inv | Result |
|---|---|---:|---|---|
| 8192 x 524032 x 8192 | baseline 1x1 stage 3 | 2 300 118 | n/a | prior best |
| 8192 x 524032 x 8192 | 2x1 stage 3 | 2 505 314 | 1314 / 1 / 0 | clean |

Longer confirmation:

| Shape | Kernel | Duration | mm/s | tile rate (tiles/s) | cache h/m/inv | Result |
|---|---|---:|---:|---:|---|---|
| 8192 x 524032 x 8192 | 2x1 stage 3 | 323.6 s | 19.1 | 2 503 688 | 6182 / 4 / 3 | clean |

This is **+8.8%** over the prior `n512minus` best and **+19.1%** over the original non-headless `n256` baseline.

Current recommended production command:

```bash
uv run direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-stages 3 \
  --kernel-cluster-m 2 \
  --kernel-cluster-n 1 \
  --log-interval 100
```

Next kernel work:

- Profile `hopper_mine_ws` with Nsight Compute if available; Nsight Systems only says the mine kernel dominates, not which instructions or memory paths inside it dominate.
- Test a second, proof-pattern-validated tile-shape grid only after confirming the extracted row/column pattern still matches the mining configuration, or after teaching the miner to derive the job key/target from the actual variant pattern.
- Separately fix the `tensor_hash` 4 GiB grid/`uint32_t` ceiling if we want to explore `n >= 524288`.

## Addendum: tile-shape probe grid

On 2026-05-15 we added a small proof-pattern inspector and a first tile-shape probe grid. The inspector forces an easy win with a max target, reads the `HostSignalHeader`, and checks that the extracted proof rows/columns still match the default mining configuration:

- rows: `[0, 8]`
- cols: `[0, 1, 8, 9, ..., 248, 249]`

Build pruning:

- `256x256x128` failed compile with `ptxas fatal (C7602): Insufficient registers (96)` and requested a register target of 154.
- `256x256x64` failed the same way.
- Both `256x256` probes were removed from the compiled grid. Retesting that family requires launch-bound/register-allocation work first.

All remaining compiled candidates were pattern-compatible:

| Tile | Cluster | Stages | Pattern result |
|---|---|---:|---|
| 128x256x128 | 2x1 | 3 | compatible |
| 64x256x128 | 1x1 | 3 | compatible |
| 64x256x128 | 2x1 | 3 | compatible |
| 128x256x64 | 1x1 | 3 | compatible |
| 128x256x64 | 2x1 | 3 | compatible |
| 128x256x64 | 2x1 | 4 | compatible |

Important accounting note: raw `tile_rate` is not directly comparable when `tile_m` changes. In `KernelTraits`, `kNumMmaThreads = (tile_m / 64) * 128`, and each MMA consumer thread checks the PoW target once per CTA. Therefore a `64x256` CTA has half as many PoW checks as a `128x256` CTA. The apples-to-apples normalized attempt rate is:

```text
normalized_attempt_rate = raw_tile_rate * (tile_m / 128)
```

### Quick production-shape benchmark

Each cell used:

```bash
uv run direct-miner \
  --m 8192 --n 524032 --k 8192 \
  --max-in-flight 4 \
  --enable-b-cache \
  --enable-headless-kernel \
  --kernel-tile-m <tile_m> \
  --kernel-tile-n 256 \
  --kernel-tile-k <tile_k> \
  --kernel-stages <stages> \
  --kernel-cluster-m <cM> \
  --kernel-cluster-n 1 \
  --log-interval 100
```

| Variant | Raw tile rate (tiles/s) | Normalized attempt rate | Result |
|---|---:|---:|---|
| **128x256x128, c2x1, s3** | **2 506 544** | **2 506 544** | current winner |
| 64x256x128, c1x1, s3 | 2 958 630 | 1 479 315 | worse after normalization |
| 64x256x128, c2x1, s3 | ~3 513 599 | ~1 756 800 | worse after normalization; timeout killed during drain |
| 128x256x64, c1x1, s3 | 2 080 652 | 2 080 652 | worse |
| 128x256x64, c2x1, s3 | 2 084 419 | 2 084 419 | worse |
| 128x256x64, c2x1, s4 | 2 153 039 | 2 153 039 | worse |

Conclusion: keep the current production kernel:

```bash
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1
```

The raw `64x256x128` numbers are tempting, but they do not represent more PoW checks per second after accounting for the smaller M tile. The next real optimization target is inside `hopper_mine_ws`, not this first tile-shape grid.

## Addendum: NCU attempt and static resource analysis

On 2026-05-15 we tried to profile the current winner with Nsight Compute:

```bash
/usr/local/cuda/bin/ncu \
  --target-processes application-only \
  --kernel-name regex:hopper_mine_ws \
  --launch-skip 2 \
  --launch-count 1 \
  --kill yes \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  -o /workspace/ncu_hopper_mine_direct_<ts> \
  /root/pearl/.venv/bin/direct-miner ...
```

The pod host blocks NVIDIA performance counters:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

So NCU cannot provide stall, occupancy, scheduler, or speed-of-light counters from this container unless the host enables unrestricted profiling counters.

As a fallback, we dumped static resource usage from the compiled extension:

```bash
/usr/local/cuda/bin/cuobjdump --dump-resource-usage \
  /root/pearl/miner/pearl-gemm/src/pearl_gemm_cuda.cpython-312-x86_64-linux-gnu.so
```

Relevant `hopper_mine_ws` resource usage:

| Variant | Registers/thread | Stack | Static shared | Constant |
|---|---:|---:|---:|---:|
| 128x256x128, R128 | 168 | 64 | 1024 | 1136 |
| 128x256x64, R128 | 168 | 64 | 1024 | 1136 |
| 64x256x128, R128 | 255 | 64 | 1024 | 1136 |

Interpretation:

- Current winner launches `tile_m=128`, so `kNumMmaThreads = 256` and total block threads are `256 consumers + 128 producer-warpgroup threads = 384`.
- `384 * 168 = 64 512` registers, essentially one full H100 SM register file. This explains why the kernel behaves as one CTA per SM and why larger `256x256` tiles fail to compile.
- `64x256x128` uses fewer consumer threads but jumps to 255 registers/thread, landing at `256 * 255 = 65 280` registers per CTA. That also fills an SM while doing only half the normalized PoW checks per CTA.
- `128x256x64` does not reduce register pressure, so its extra K-loop overhead simply makes it slower.

### Next kernel target

The next plausible kernel optimization is to remove the mostly idle producer warpgroup from the mine-only launch shape.

Today the kernel launches a full 128-thread producer warpgroup even though the TMA path effectively uses one producer warp. The existing named-barrier accounting already often uses `kNumMmaThreads + cutlass::NumThreadsPerWarp`, which suggests only 32 producer threads need to participate in some synchronization points.

Candidate experiment:

- keep consumer WGMMA warpgroups aligned at thread IDs `0..255`
- move the producer warp to thread IDs `256..287`
- change `kNumThreads` from `kNumMmaThreads + 128` to `kNumMmaThreads + 32` for `MineOnly`
- make producer role selection explicit instead of using `warp_group_idx == 0`
- set `consumer_tix = threadIdx.x` for consumer threads
- preserve existing full-producer-warpgroup path for normal non-mine GEMM

For the current 128 tile this would reduce launched threads from 384 to 288. It will not magically halve registers, but it could reduce scheduler pressure, reduce dead producer-thread overhead, and create room for a later `tile_m=192/256` experiment if register allocation also improves.

This is a real kernel surgery item, not a CLI/autotune item. It should be implemented behind a separate mine-only specialization and benchmarked against the current `128x256x128, c2x1, s3` winner.

### One-producer-warp experiment result

We tried the candidate one-producer-warp layout:

- `KernelTraits::kNumThreads = kNumMmaThreads + 32` for `MineOnly`
- consumer threads moved to `threadIdx.x = 0..255`
- producer warp moved to `threadIdx.x = 256..287`
- `consumer_tix = threadIdx.x`

The extension compiled successfully, and static resources per thread were unchanged:

| Variant | Registers/thread | Stack | Static shared | Constant |
|---|---:|---:|---:|---:|
| 128x256x128, R128 | 168 | 64 | 1024 | 1136 |

But the forced-win pattern inspector hung and had to be killed:

```bash
timeout --signal=SIGINT --kill-after=10 20 \
  uv run direct-miner-inspect-pattern \
    --tile-m 128 --tile-n 256 --tile-k 128 \
    --cluster-m 2 --cluster-n 1 --stages 3 --iterations 1
```

Result:

```text
Killed
STATUS=137
```

No `HostSignalHeader` was produced. The most likely cause is that `PipelineTmaAsync` / TMA barrier signaling assumes producer participation from a normal warpgroup shape even though only one producer warp issues copies. The experiment was rolled back to the known-good full producer warpgroup layout.

Conclusion: do not use the one-producer-warp layout. A future version would need a purpose-built mine-only pipeline or a deeper CUTLASS pipeline rewrite, not just a thread remap.

## Register-allocation and 64-row tile resweep

On 2026-05-15 we added an opt-in `mma_registers` kernel-dispatch knob and
compiled explicit register-allocation variants for:

- `128x256x128, c2x1, s3` with `mma_registers={160,192,224}`
- `64x256x128, c1x1/c2x1, s3` with `mma_registers={128,160,192,224}`

The default `mma_registers=0` path preserves the existing kernel heuristic.

Forced-win pattern inspection passed for every tested variant. The emitted
proof row and column patterns matched the default mining configuration:

```text
rows=[0, 8]
cols=[0, 1, 8, 9, ..., 248, 249]
PATTERN_COMPATIBLE=true
```

One-minute production-shape sweep:

Important correction: the direct-miner `tile_rate` log field is raw CTA /
outer-tile rate. When `tile_m` changes, it is not an apples-to-apples count of
lottery tickets. `tile_m=64` has one MMA warpgroup and 128 MMA consumer
threads; `tile_m=128` has two MMA warpgroups and 256 MMA consumer threads.
Each MMA consumer thread performs one PoW check per CTA, so `64x256` raw
outer-tile rates must be multiplied by `64 / 128 = 0.5` before comparing them
to `128x256` variants.

| Variant | Source | Raw outer-tiles/s | Normalized 128-equivalent attempts/s |
|---|---:|---:|---:|
| 64x256x128, c2x1, default regs | final | 3 520 483 | 1 760 242 |
| 64x256x128, c2x1, regs=224 | final | 3 520 104 | 1 760 052 |
| 64x256x128, c2x1, regs=192 | last steady log | 3 517 276 | 1 758 638 |
| 64x256x128, c2x1, regs=128 | last steady log | 3 517 018 | 1 758 509 |
| 64x256x128, c2x1, regs=160 | last steady log | 3 496 877 | 1 748 439 |
| 64x256x128, c1x1, default/explicit regs | mixed | ~2 944 000 - 2 959 000 | ~1 472 000 - 1 479 500 |
| 128x256x128, c2x1, regs=160 | final | 2 510 826 | 2 510 826 |
| 128x256x128, c2x1, default regs | final | 2 502 826 | 2 502 826 |

Five-minute confirmation for the raw-fast but normalized-worse 64-row variant:

```text
Variant: 64x256x128, c2x1, s3, default regs
Completed: 4344 matmuls
Elapsed: 324.0 s
Completion rate: 13.4 matmuls/s
Tile rate: 3 512 843 tiles/s
B-cache: hits=4341 misses=4 invalidations=3
```

This does **not** supersede the earlier tile-shape conclusion. With the
production `n=524032` shape, direct-miner raw outer-tile accounting, headless
kernel, and B-cache enabled, `64x256x128, c2x1, s3` looked about:

```text
3 512 843 / 2 503 688 = 1.403x
```

or roughly **+40%** on raw CTA rate, but after normalizing for the half-sized
MMA consumer thread count:

```text
(3 512 843 * 0.5) / 2 503 688 = 0.701x
```

So `64x256x128` produces about **30% fewer PoW attempts/s** than the
`128x256x128, c2x1, s3` production kernel. The 64-row result is retained here
as a case study in why raw direct-miner `tile_rate` must be normalized when
changing `tile_m`.

Recommended production kernel flags:

```bash
--enable-headless-kernel \
--kernel-tile-m 128 \
--kernel-tile-n 256 \
--kernel-tile-k 128 \
--kernel-stages 3 \
--kernel-cluster-m 2 \
--kernel-cluster-n 1
```

Do not pass `--kernel-mma-registers` for the production default yet. The
`regs=160` 128-row variant was only a tiny one-minute improvement over default
and has not been five-minute confirmed.
