"""Phase B mining call with B-side artifact caching.

Mirrors the body of vllm_miner.gemm_operators.pearl_gemm_noisy but adds
hooks for skipping B-side recomputation when the template hasn't changed.

Strategy C-lite: this is a parallel implementation rather than a
modification of pearl_gemm_noisy in vllm-miner. The duplication carries
a drift hazard — if pearl_gemm_noisy changes, this file must follow.

Lower-level building blocks (tensor_hash, commitment_hash_from_merkle_roots,
noise_gen, noisy_gemm, make_pow_target_tensor) are imported from pearl_gemm
directly; we do NOT call vllm-miner's pearl_gemm_noisy here.
"""

import logging
from typing import Optional

import torch
from miner_base.commitment_hash import CommitmentHasher
from pearl_gateway.comm.dataclasses import MiningJob
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
from vllm_miner.mining_state import get_async_manager, get_pinned_pool

from .b_cache import BSideArtifacts, BSideCache


logger = logging.getLogger(__name__)


def pearl_gemm_noisy_cached(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    matmul_config,  # GPUMatmulConfigFactory.create(...) result
    settings,  # MinerSettings (tile sizes, noise rank)
    b_cache: Optional[BSideCache] = None,
    submit_block: bool = True,
) -> tuple[torch.Tensor, bool]:
    """Mining call with optional B-side caching.

    Returns (C, b_cache_hit) where b_cache_hit is True iff cached B-side
    artifacts were used (False on cache miss or when b_cache is None).
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    m, k = A.shape
    n = B.shape[0]
    r = settings.noise_rank
    device = A.device

    C = torch.empty((m, n), dtype=out_dtype, device=device)

    matrix_bytes = max(m * k, n * k)
    tensor_hash_scratchpad = torch.empty(
        get_required_scratchpad_bytes(matrix_bytes),
        dtype=torch.uint8,
        device=device,
    )

    mining_job: MiningJob = get_async_manager().get_mining_job()
    mining_config = matmul_config.mining_config
    adjusted_target = mining_job.adjust_target(mining_config=mining_config)

    hash_key = CommitmentHasher.get_key(
        mining_job.incomplete_header_bytes, mining_config
    )
    key_tensor = torch.frombuffer(
        bytearray(hash_key), dtype=torch.uint8
    ).to(device)

    # ---- A-side (always per-call) ----
    A_tensor_hash = torch.empty(32, device=device, dtype=torch.uint8)
    tensor_hash(
        A.to(torch.uint8),
        key_tensor,
        A_tensor_hash,
        tensor_hash_scratchpad,
    )

    # ---- B-side (cacheable) ----
    cached: Optional[BSideArtifacts] = (
        b_cache.get(hash_key) if b_cache is not None else None
    )
    b_cache_hit = cached is not None

    if cached is not None:
        B_tensor_hash = cached.B_tensor_hash
        commitment_hash_B_tensor = cached.commitment_hash_B
        EBR = cached.EBR
        EBR_fp16 = cached.EBR_fp16
        EBL_R_major = cached.EBL_R_major
        EBL_K_major = cached.EBL_K_major
        BpEB = cached.BpEB
    else:
        B_tensor_hash = torch.empty(32, device=device, dtype=torch.uint8)
        tensor_hash(
            B.to(torch.uint8),
            key_tensor,
            B_tensor_hash,
            tensor_hash_scratchpad,
        )

    # commitment_hash_from_merkle_roots writes BOTH A and B commitment
    # hashes. On a cache hit we still want commitment_hash_A but already
    # have commitment_hash_B; we pass a throwaway tensor for B and ignore
    # the write. The cost is one blake3 of 64 bytes — negligible.
    commitment_hash_A_tensor = torch.empty(32, device=device, dtype=torch.uint8)
    if cached is not None:
        commitment_hash_B_throwaway = torch.empty(
            32, device=device, dtype=torch.uint8
        )
        commitment_hash_from_merkle_roots(
            A_tensor_hash,
            B_tensor_hash,
            key_tensor,
            commitment_hash_A_tensor,
            commitment_hash_B_throwaway,
        )
        del commitment_hash_B_throwaway
    else:
        commitment_hash_B_tensor = torch.empty(
            32, device=device, dtype=torch.uint8
        )
        commitment_hash_from_merkle_roots(
            A_tensor_hash,
            B_tensor_hash,
            key_tensor,
            commitment_hash_A_tensor,
            commitment_hash_B_tensor,
        )

    # ---- Noise factors ----
    # A-side: always generated.
    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)

    if cached is not None:
        # Skip B-side noise outputs and key_B per noise_gen docstring:
        # "To not involve noise matrices, just skip that argument."
        noise_gen(
            R=r,
            EAL=EAL,
            EAL_fp16=EAL_fp16,
            EAR_R_major=EAR_R_major,
            EAR_K_major=EAR_K_major,
            key_A=commitment_hash_A_tensor,
        )
    else:
        EBR = torch.empty((n, r), dtype=torch.int8, device=device)
        EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
        EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
        EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
        noise_gen(
            R=r,
            EAL=EAL,
            EAL_fp16=EAL_fp16,
            EAR_R_major=EAR_R_major,
            EAR_K_major=EAR_K_major,
            EBL_R_major=EBL_R_major,
            EBL_K_major=EBL_K_major,
            EBR=EBR,
            EBR_fp16=EBR_fp16,
            key_A=commitment_hash_A_tensor,
            key_B=commitment_hash_B_tensor,
        )

    # ---- BpEB / EARxBpEB / kernel output buffers ----
    if cached is None:
        # Cache miss: BpEB is filled by noisy_gemm's B-noising path,
        # so allocate it as an OUTPUT buffer here.
        BpEB = torch.empty((n, k), dtype=torch.int8, device=device)

    EARxBpEB = torch.empty((n, r), dtype=torch.float16, device=device)
    ApEA = torch.empty((m, k), dtype=torch.int8, device=device)
    A_E_BL = torch.empty((m, r), dtype=torch.float16, device=device)

    host_signal_sync_size = get_host_signal_sync_size()
    host_signal_sync = torch.zeros(
        (host_signal_sync_size,), dtype=torch.int8, device=device
    )
    host_signal_header_pinned = get_pinned_pool().acquire()

    pow_target_tensor = make_pow_target_tensor(adjusted_target)

    # Critical: on a cache hit, run_noising_B=False uses the cached BpEB
    # as INPUT instead of recomputing it from B+EBR+EBL. The kernel still
    # needs EBR/EBL/EBR_fp16 for the denoising epilogue, hence we pass
    # the cached versions.
    run_noising_B = (cached is None)

    noisy_gemm(
        A=A,
        B=B,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        EAR_R_major=EAR_R_major,
        EBL_R_major=EBL_R_major,
        EAR_K_major=EAR_K_major,
        EBL_K_major=EBL_K_major,
        AxEBL_fp16=A_E_BL,
        EARxBpEB_fp16=EARxBpEB,
        ApEA=ApEA,
        BpEB=BpEB,
        A_scales=A_scales,
        B_scales=B_scales,
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

    # ---- Post-launch: cache populate, schedule status check ----
    if b_cache is not None and cached is None:
        # Populate cache with what we just computed.
        b_cache.put(
            hash_key,
            BSideArtifacts(
                B_tensor_hash=B_tensor_hash,
                commitment_hash_B=commitment_hash_B_tensor,
                EBR=EBR,
                EBR_fp16=EBR_fp16,
                EBL_R_major=EBL_R_major,
                EBL_K_major=EBL_K_major,
                BpEB=BpEB,
            ),
        )

    if submit_block:
        cuda_event = torch.cuda.Event()
        cuda_event.record()
        callback = StatusCheckCallback(
            host_signal_header_pinned=host_signal_header_pinned,
            commitment_hash_A_tensor=commitment_hash_A_tensor,
            commitment_hash_B_tensor=commitment_hash_B_tensor,
            A=A,
            B=B,
            mining_job=mining_job,
        )
        get_async_manager().schedule_status_check(cuda_event, callback)
        host_signal_header_pinned = None  # owned by callback
    else:
        get_pinned_pool().release(host_signal_header_pinned)
        del host_signal_header_pinned

    return C, b_cache_hit
