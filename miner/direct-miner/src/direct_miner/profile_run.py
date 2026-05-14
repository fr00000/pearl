"""Per-function CUDA-event timing for direct-miner.

Runs a small number of mining iterations synchronously, timing each
phase inside pearl_gemm_noisy_cached via torch.cuda.Event pairs. Outputs
a per-phase breakdown showing where GPU time goes.

Lower fidelity than nsys but always works (no GPU counter permissions
needed) and isolates Python-side allocations and CPU work that nsys's
kernel-only summary doesn't show.

Run as:
    uv run python -m direct_miner.profile_run [--iterations N] [--no-cache]
"""

import argparse
import logging
import sys
import time
from typing import Optional

import torch
from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from pearl_gemm import (
    commitment_hash_from_merkle_roots,
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
    make_pow_target_tensor,
    noise_gen,
    noisy_gemm,
    tensor_hash,
)
from vllm_miner.callbacks import StatusCheckCallback
from vllm_miner.mining_state import (
    init_async_manager,
    init_pinned_pool,
    get_async_manager,
    get_pinned_pool,
)

from direct_miner.b_cache import BSideArtifacts, BSideCache
from direct_miner.synthetic_data import FixedBPool, make_synthetic_a


logger = logging.getLogger(__name__)


class CudaTimer:
    """Pair GPU operations with torch.cuda.Event objects.

    `start(label)` returns a handle; `end(handle)` records the end event.
    Call `synchronize_and_compute()` once after all iterations to wait
    and accumulate totals.
    """

    def __init__(self):
        self.entries: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.totals_ms: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def start(self, label: str):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        return label, start_evt, end_evt

    def end(self, handle) -> None:
        label, start_evt, end_evt = handle
        end_evt.record()
        self.entries.append((label, start_evt, end_evt))

    def synchronize_and_compute(self) -> None:
        torch.cuda.synchronize()
        for label, start_evt, end_evt in self.entries:
            elapsed_ms = start_evt.elapsed_time(end_evt)
            self.totals_ms[label] = self.totals_ms.get(label, 0.0) + elapsed_ms
            self.counts[label] = self.counts.get(label, 0) + 1
        self.entries.clear()

    def report(self) -> None:
        if not self.totals_ms:
            print("No timing data collected.")
            return
        total = sum(self.totals_ms.values())
        print(
            f"\n{'Phase':<28} {'Calls':>6} {'Total (ms)':>12} "
            f"{'Avg (us)':>10} {'% of total':>10}"
        )
        print("-" * 72)
        for label, total_ms in sorted(self.totals_ms.items(), key=lambda x: -x[1]):
            count = self.counts[label]
            avg_us = total_ms * 1000 / count
            pct = total_ms / total * 100 if total > 0 else 0
            print(f"{label:<28} {count:>6} {total_ms:>12.2f} "
                  f"{avg_us:>10.1f} {pct:>9.1f}%")
        print("-" * 72)
        print(f"{'TOTAL':<28} {'':>6} {total:>12.2f}")


def profile_one_iteration(
    timer: CudaTimer,
    A: torch.Tensor,
    A_scales: torch.Tensor,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    matmul_config,
    settings,
    b_cache: Optional[BSideCache],
    out_dtype: torch.dtype = torch.bfloat16,
) -> bool:
    m, k = A.shape
    n = B.shape[0]
    r = settings.noise_rank
    device = A.device

    h = timer.start("setup_alloc")
    C = torch.empty((m, n), dtype=out_dtype, device=device)
    matrix_bytes = max(m * k, n * k)
    tensor_hash_scratchpad = torch.empty(
        get_required_scratchpad_bytes(matrix_bytes),
        dtype=torch.uint8, device=device,
    )
    timer.end(h)

    h = timer.start("get_mining_job")
    mining_job = get_async_manager().get_mining_job()
    mining_config = matmul_config.mining_config
    adjusted_target = mining_job.adjust_target(mining_config=mining_config)
    timer.end(h)

    h = timer.start("hash_key_derive")
    hash_key = CommitmentHasher.get_key(
        mining_job.incomplete_header_bytes, mining_config
    )
    key_tensor = torch.frombuffer(
        bytearray(hash_key), dtype=torch.uint8
    ).to(device)
    timer.end(h)

    h = timer.start("A_tensor_hash")
    A_tensor_hash = torch.empty(32, device=device, dtype=torch.uint8)
    tensor_hash(
        A.to(torch.uint8), key_tensor, A_tensor_hash, tensor_hash_scratchpad
    )
    timer.end(h)

    cached: Optional[BSideArtifacts] = (
        b_cache.get(hash_key) if b_cache is not None else None
    )
    b_cache_hit = cached is not None

    if cached is None:
        h = timer.start("B_tensor_hash")
        B_tensor_hash = torch.empty(32, device=device, dtype=torch.uint8)
        tensor_hash(
            B.to(torch.uint8), key_tensor, B_tensor_hash,
            tensor_hash_scratchpad,
        )
        timer.end(h)
    else:
        B_tensor_hash = cached.B_tensor_hash

    h = timer.start("commitment_hash")
    commitment_hash_A_tensor = torch.empty(32, device=device, dtype=torch.uint8)
    if cached is not None:
        commitment_hash_B_throwaway = torch.empty(
            32, device=device, dtype=torch.uint8
        )
        commitment_hash_from_merkle_roots(
            A_tensor_hash, B_tensor_hash, key_tensor,
            commitment_hash_A_tensor, commitment_hash_B_throwaway,
        )
        commitment_hash_B_tensor = cached.commitment_hash_B
    else:
        commitment_hash_B_tensor = torch.empty(
            32, device=device, dtype=torch.uint8
        )
        commitment_hash_from_merkle_roots(
            A_tensor_hash, B_tensor_hash, key_tensor,
            commitment_hash_A_tensor, commitment_hash_B_tensor,
        )
    timer.end(h)

    h = timer.start("noise_gen")
    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    if cached is not None:
        noise_gen(
            R=r, EAL=EAL, EAL_fp16=EAL_fp16,
            EAR_R_major=EAR_R_major, EAR_K_major=EAR_K_major,
            key_A=commitment_hash_A_tensor,
        )
        EBR = cached.EBR
        EBR_fp16 = cached.EBR_fp16
        EBL_R_major = cached.EBL_R_major
        EBL_K_major = cached.EBL_K_major
    else:
        EBR = torch.empty((n, r), dtype=torch.int8, device=device)
        EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
        EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
        EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
        noise_gen(
            R=r, EAL=EAL, EAL_fp16=EAL_fp16,
            EAR_R_major=EAR_R_major, EAR_K_major=EAR_K_major,
            EBL_R_major=EBL_R_major, EBL_K_major=EBL_K_major,
            EBR=EBR, EBR_fp16=EBR_fp16,
            key_A=commitment_hash_A_tensor,
            key_B=commitment_hash_B_tensor,
        )
    timer.end(h)

    h = timer.start("preamble_buffers")
    if cached is None:
        BpEB = torch.empty((n, k), dtype=torch.int8, device=device)
    else:
        BpEB = cached.BpEB
    EARxBpEB = torch.empty((n, r), dtype=torch.float16, device=device)
    ApEA = torch.empty((m, k), dtype=torch.int8, device=device)
    A_E_BL = torch.empty((m, r), dtype=torch.float16, device=device)
    host_signal_sync_size = get_host_signal_sync_size()
    host_signal_sync = torch.zeros(
        (host_signal_sync_size,), dtype=torch.int8, device=device
    )
    host_signal_header_pinned = get_pinned_pool().acquire()
    pow_target_tensor = make_pow_target_tensor(adjusted_target)
    timer.end(h)

    run_noising_B = (cached is None)

    h = timer.start("noisy_gemm")
    noisy_gemm(
        A=A, B=B,
        EAL=EAL, EAL_fp16=EAL_fp16,
        EBR=EBR, EBR_fp16=EBR_fp16,
        EAR_R_major=EAR_R_major, EBL_R_major=EBL_R_major,
        EAR_K_major=EAR_K_major, EBL_K_major=EBL_K_major,
        AxEBL_fp16=A_E_BL, EARxBpEB_fp16=EARxBpEB,
        ApEA=ApEA, BpEB=BpEB,
        A_scales=A_scales, B_scales=B_scales,
        C=C,
        host_signal_header_pinned=host_signal_header_pinned,
        host_signal_sync=host_signal_sync,
        pow_target=pow_target_tensor,
        pow_key=commitment_hash_A_tensor.view(torch.uint32),
        tile_size_m=settings.tile_size_m,
        tile_size_n=settings.tile_size_n,
        tile_size_k=settings.tile_size_k,
        run_noising_A=True,
        run_noising_B=run_noising_B,
        skip_reduction=False,
        skip_denoising=False,
    )
    timer.end(h)

    h = timer.start("post_cache_populate")
    if b_cache is not None and cached is None:
        b_cache.put(
            hash_key,
            BSideArtifacts(
                B_tensor_hash=B_tensor_hash,
                commitment_hash_B=commitment_hash_B_tensor,
                EBR=EBR, EBR_fp16=EBR_fp16,
                EBL_R_major=EBL_R_major, EBL_K_major=EBL_K_major,
                BpEB=BpEB,
            ),
        )
    timer.end(h)

    h = timer.start("schedule_status_check")
    cuda_event = torch.cuda.Event()
    cuda_event.record()
    callback = StatusCheckCallback(
        host_signal_header_pinned=host_signal_header_pinned,
        commitment_hash_A_tensor=commitment_hash_A_tensor,
        commitment_hash_B_tensor=commitment_hash_B_tensor,
        A=A, B=B, mining_job=mining_job,
    )
    get_async_manager().schedule_status_check(cuda_event, callback)
    timer.end(h)

    return b_cache_hit


def profile_make_synthetic_a(
    timer: CudaTimer, m: int, k: int, generator: Optional[torch.Generator]
) -> tuple[torch.Tensor, torch.Tensor]:
    h = timer.start("make_synthetic_a")
    A, A_scales = make_synthetic_a(m=m, k=k, device="cuda", generator=generator)
    timer.end(h)
    return A, A_scales


def main():
    parser = argparse.ArgumentParser(
        description="Profile direct-miner per-function timing"
    )
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="number of mining iterations to profile (default 20)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="warmup iterations before timing (default 5)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="profile without B-cache (Phase A baseline path)"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("Initializing async manager and pinned pool...")
    init_async_manager()
    init_pinned_pool(get_async_manager()._conf.pinned_pool_size)
    time.sleep(1.0)

    try:
        get_async_manager().get_mining_job()
        print("Gateway connection OK")
    except Exception as e:
        print(f"Gateway connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Allocating B matrix ({args.n}x{args.k} int8)...")
    b_pool = FixedBPool(n=args.n, k=args.k, device="cuda", seed=args.seed)
    torch.cuda.synchronize()

    settings = get_async_manager()._conf
    matmul_config = GPUMatmulConfigFactory.create(
        k=args.k, noise_rank=settings.noise_rank
    )

    b_cache = None if args.no_cache else BSideCache()
    if b_cache is not None:
        print("B-cache ENABLED")
    else:
        print("B-cache DISABLED (Phase A path)")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed)

    print(f"\nWarming up ({args.warmup} iterations)...")
    discard = CudaTimer()
    for _ in range(args.warmup):
        A, A_scales = make_synthetic_a(
            m=args.m, k=args.k, device="cuda", generator=generator
        )
        profile_one_iteration(
            discard, A, A_scales, b_pool.B, b_pool.B_scales,
            matmul_config, settings, b_cache,
        )
    torch.cuda.synchronize()
    discard.entries.clear()

    print(f"\nProfiling {args.iterations} iterations...")
    timer = CudaTimer()
    cache_hits = 0
    wall_start = time.time()

    for _ in range(args.iterations):
        A, A_scales = profile_make_synthetic_a(timer, args.m, args.k, generator)
        hit = profile_one_iteration(
            timer, A, A_scales, b_pool.B, b_pool.B_scales,
            matmul_config, settings, b_cache,
        )
        if hit:
            cache_hits += 1

    timer.synchronize_and_compute()
    wall_elapsed = time.time() - wall_start

    print(f"\n=== Profile Results ===")
    print(f"Iterations: {args.iterations}")
    print(f"Cache hits: {cache_hits} / {args.iterations}")
    print(f"Wall time: {wall_elapsed:.2f}s")
    print(f"Effective rate: {args.iterations / wall_elapsed:.1f} matmuls/sec")

    timer.report()

    total = sum(timer.totals_ms.values())
    noisy_gemm_pct = (
        timer.totals_ms.get("noisy_gemm", 0) / total * 100 if total else 0
    )
    print("\n=== Interpretation ===")
    if noisy_gemm_pct > 80:
        print(f"noisy_gemm dominates ({noisy_gemm_pct:.0f}% of GPU time).")
        print("Per-call CPU/setup work is small relative to kernel.")
        print("Optimization focus: inside the kernel.")
    elif noisy_gemm_pct > 50:
        print(f"noisy_gemm is majority ({noisy_gemm_pct:.0f}% of GPU time)")
        print("but per-call overhead is meaningful.")
        print("Optimization focus: both kernel and Python-side work.")
    else:
        print(f"noisy_gemm is only {noisy_gemm_pct:.0f}% of GPU time.")
        print("Per-call overhead is the bigger optimization target.")
        print("Look at: tensor_hash, noise_gen, allocations.")


if __name__ == "__main__":
    main()
