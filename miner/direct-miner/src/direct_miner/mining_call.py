"""Phase C mining call: B-side cache + multi-stream A-side overlap.

A-side prep (in-place A generation, A tensor_hash, commitment_hash,
A-side noise_gen) runs on stream_prep so it overlaps with the
previous iteration's main kernel on stream_main.

Cross-stream barrier:
    prep_done_event (recorded on stream_prep) →
    stream_main.wait_event(prep_done_event) before noisy_gemm

Cache-miss path: B-side compute runs on stream_main, with stream_prep
waiting via b_hash_done for B_tensor_hash before commitment_hash. This
adds a sync point but only fires once per template (~95 s).

Strategy C-lite caveat unchanged: this is a parallel implementation
of vllm_miner.gemm_operators.pearl_gemm_noisy and must be kept in
sync if the protocol changes.

Slot-pool tensors are owned by the caller's ASlotPool. This function
WRITES into slot.A, slot.A_scales (via make_synthetic_a_into),
slot.A_tensor_hash, slot.commitment_hash_A, slot.EAL, slot.EAR_*,
slot.host_signal_sync, slot.C; READS from slot.tensor_hash_scratchpad.

Returns (slot.C, b_cache_hit, completion_event). The completion_event
is recorded on stream_main immediately after noisy_gemm; pass it to
CompletionTracker.record_launch(event=...) so the callback fires when
the main kernel actually finishes, not when this function returns.
"""

import logging
from typing import Callable, Optional

import torch
from miner_base.commitment_hash import CommitmentHasher
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gemm import (
    commitment_hash_from_merkle_roots,
    make_pow_target_tensor,
    noise_gen,
    noisy_gemm,
    tensor_hash,
)
from vllm_miner.callbacks import StatusCheckCallback
from vllm_miner.mining_state import get_async_manager, get_pinned_pool

from .a_slot_pool import ASlot
from .b_cache import BSideArtifacts, BSideCache
from .synthetic_data import make_synthetic_a_into


logger = logging.getLogger(__name__)


class UnsafeSlotReleaseError(RuntimeError):
    """Raised when cleanup cannot prove GPU work is complete, so the A
    slot must not be released. Fatal for the miner process — the run
    loop catches this exception specifically and exits non-zero so a
    supervisor / bootstrap re-run can restart cleanly.

    Distinct from generic RuntimeError so the broad except-Exception
    handler in _mine_one_iteration can let this one escape without
    swallowing routine errors (e.g., transient kernel failures).
    """
    pass


def _cleanup_unscheduled_slot(
    *,
    gpu_work_queued: bool,
    completion_event,
    host_signal_header_pinned,
    release_pinned_header: Callable[[object], None],
    on_callback_done: Optional[Callable[[], None]],
    device=None,
) -> None:
    """Outer-finally cleanup for paths that did not transfer slot
    ownership to a scheduled callback.

    GPU-work scope: stream_prep starts writing slot tensors (slot.A,
    slot.A_tensor_hash, slot.commitment_hash_A, ...) BEFORE the main
    noisy_gemm kernel launches. Use "any slot tensor enqueued on a
    stream" as the gate, NOT "main kernel launched" — the v4 mistake
    missed the stream_prep window. gpu_work_queued must be set True
    before the first stream context that touches a slot tensor.

    Order matters:
      1. If GPU work was queued, synchronize before touching any
         slot tensors. Two paths:
           - completion_event exists (main kernel was launched and
             recorded): try event.synchronize() first. Cheaper than
             a device-wide sync; covers both streams because
             stream_main waited on prep_done_event from stream_prep.
           - completion_event is None (pre-main failure): no event
             to sync on, fall through to torch.cuda.synchronize().
         Final fallback in either path: torch.cuda.synchronize(),
         which waits for all kernels on all streams of this device.
         If all sync attempts fail, raise — we cannot prove the GPU
         is idle, so releasing the slot would risk corruption.
      2. Only after sync succeeds: release the pinned header (live
         as the kernel's output buffer; same safety argument).
      3. Only after the pinned header is back in the pool: release
         the A slot via on_callback_done.

    On sync failure, both pinned header and slot are deliberately
    leaked. The miner will eventually hang on a future acquire()
    (default timeout=None, fail-closed). That's the intended visible
    failure — operator restart via bootstrap re-run is the recovery
    path. A prematurely released slot can corrupt a winning proof;
    a leaked slot just costs throughput.
    """
    if gpu_work_queued:
        sync_ok = False

        # Prefer event-specific sync when we have one — cheaper than
        # a device-wide sync, and waits only on the recorded point.
        if completion_event is not None:
            try:
                completion_event.synchronize()
                sync_ok = True
            except Exception:
                logger.exception(
                    "completion_event.synchronize() failed; "
                    "falling back to torch.cuda.synchronize()"
                )

        # Fallback (also the primary path when no completion event
        # exists yet — pre-main-kernel failure). Waits for all
        # kernels on all streams of this device, covering both
        # stream_prep and stream_main regardless of which streams
        # queued work. Target slot.A.device specifically when known;
        # syncing the wrong device under a future multi-GPU mode
        # would silently leave work in flight.
        if not sync_ok:
            try:
                import torch
                if device is not None:
                    torch.cuda.synchronize(device=device)
                else:
                    torch.cuda.synchronize()
                sync_ok = True
            except Exception:
                logger.exception(
                    "torch.cuda.synchronize() failed; "
                    "refusing to release A slot"
                )

        if not sync_ok:
            raise UnsafeSlotReleaseError(
                "Failed to synchronize CUDA work before slot release; "
                "refusing to release A slot to avoid tensor corruption. "
                "Miner must be restarted via the bootstrap script."
            )

    if host_signal_header_pinned is not None:
        try:
            release_pinned_header(host_signal_header_pinned)
        except Exception:
            logger.exception(
                "Failed to release pinned header in outer cleanup"
            )

    if on_callback_done is not None:
        try:
            on_callback_done()
        except Exception:
            logger.exception("Slot release in outer cleanup failed")


class _SlotReleasingCallback:
    """Wraps StatusCheckCallback so the A-slot is released even if the
    callback raises. The slot must stay owned until any win-path work
    (proof construction, .cpu() copies, submission) is complete; this
    wrapper invokes the release callable in a finally block.
    """

    __slots__ = ("_inner", "_release_slot", "_released")

    def __init__(
        self,
        inner: StatusCheckCallback,
        release_slot: Callable[[], None],
    ):
        self._inner = inner
        self._release_slot = release_slot
        self._released = False

    def __call__(self, handle_submit_block):
        try:
            return self._inner(handle_submit_block)
        finally:
            # Release exactly once even if __call__ is invoked twice.
            if not self._released:
                try:
                    self._release_slot()
                except Exception:
                    logger.exception("Slot release failed")
                self._released = True


def pearl_gemm_noisy_phase_c(
    slot: ASlot,
    B: torch.Tensor,
    B_scales: torch.Tensor,
    matmul_config,
    settings,
    b_cache: Optional[BSideCache],
    stream_main: torch.cuda.Stream,
    stream_prep: torch.cuda.Stream,
    a_generator: Optional[torch.Generator] = None,
    submit_block: bool = True,
    on_callback_done: Optional[Callable[[], None]] = None,
) -> tuple[torch.Tensor, bool, torch.cuda.Event]:
    """Phase C multi-stream cached call.

    on_callback_done: if provided, MUST be called exactly once per call
    to this function. It is invoked from the async callback's finally
    block after any win-path work completes. If we fail before
    scheduling the callback (early error, submit_block=False), this
    function invokes it directly via the outer finally to release the
    slot — exactly one release path runs.
    """
    # Tracks whether ownership of on_callback_done has been
    # transferred to a scheduled callback (or whether we still
    # owe the release on the error path).
    scheduled_or_owned = False

    # Tracks whether ANY GPU work that touches slot tensors has been
    # enqueued. stream_prep writes slot.A / slot.A_tensor_hash /
    # slot.commitment_hash_A etc. WELL BEFORE the main noisy_gemm
    # kernel launches, so "kernel launched" is too narrow — the
    # cleanup helper must synchronize on the wider GPU-work scope
    # whenever this flag is True.
    gpu_work_queued = False

    # Acquired only if we enter the GPU-work path. Lives in the outer
    # scope so the finally block can release it on failure paths.
    host_signal_header_pinned = None

    # completion_event is created inside the try block. Bind in outer
    # scope so the finally can reference it; it stays None until the
    # kernel actually launches.
    completion_event = None

    try:
        # PREFLIGHT: verify async event processing is enabled BEFORE
        # any GPU work or pinned-header acquisition. Inside the try
        # so the outer finally still releases the slot (via
        # on_callback_done) on this failure path — otherwise preflight
        # raises would leak the caller's already-acquired slot.
        if submit_block:
            manager = get_async_manager()
            if not manager._conf.enable_async_cuda_event_processing:
                raise RuntimeError(
                    "Direct miner requires async CUDA event processing "
                    "(enable_async_cuda_event_processing=True). The slot "
                    "lifetime contract depends on the async callback "
                    "firing to release slot ownership."
                )

        m, k = slot.A.shape
        n = B.shape[0]
        r = settings.noise_rank
        device = slot.A.device

        mining_job: MiningJob = get_async_manager().get_mining_job()
        mining_config = matmul_config.mining_config
        adjusted_target = mining_job.adjust_target(mining_config=mining_config)

        hash_key = CommitmentHasher.get_key(
            mining_job.incomplete_header_bytes, mining_config
        )

        cached: Optional[BSideArtifacts] = (
            b_cache.get(hash_key) if b_cache is not None else None
        )
        b_cache_hit = cached is not None

        # ===== A-side prep on stream_prep =====
        # The H2D copy of key_tensor happens inside stream_prep so the
        # tensor_hash on stream_prep sees the populated buffer without an
        # implicit-default-stream sync. On cache miss, stream_main also
        # needs key_tensor for B_tensor_hash; we record an event after the
        # copy and have stream_main wait on it below.
        #
        # From this point on, stream_prep is queuing writes to slot
        # tensors. Any failure before the main completion event is
        # recorded must still synchronize this work before releasing
        # the slot — flip the gate now, not after noisy_gemm.
        gpu_work_queued = True
        with torch.cuda.stream(stream_prep):
            key_tensor = torch.frombuffer(
                bytearray(hash_key), dtype=torch.uint8
            ).to(device, non_blocking=True)

            # Record event after key_tensor copy so stream_main (cache-miss
            # path) can sync against it without serialising on stream_prep
            # work that comes after.
            key_ready_event = torch.cuda.Event()
            key_ready_event.record(stream_prep)

            make_synthetic_a_into(
                slot.A, slot.A_scales, generator=a_generator
            )

            # A.view(uint8) is byte-identical to .to(uint8) (-64→192 etc),
            # confirmed before writing this code. Skipping the copy saves
            # the 56 µs direct_copy_kernel_cuda that showed up at 8.2%
            # of GPU time in the Phase B nsys profile.
            tensor_hash(
                slot.A.view(torch.uint8),
                key_tensor,
                slot.A_tensor_hash,
                slot.tensor_hash_scratchpad,
            )

        # ===== B-side: cache lookup or compute on stream_main =====
        if cached is not None:
            B_tensor_hash = cached.B_tensor_hash
            commitment_hash_B_tensor = cached.commitment_hash_B
            EBR = cached.EBR
            EBR_fp16 = cached.EBR_fp16
            EBL_R_major = cached.EBL_R_major
            EBL_K_major = cached.EBL_K_major
            BpEB = cached.BpEB
        else:
            # Cache miss — compute B-side on stream_main. stream_main needs
            # key_tensor first.
            stream_main.wait_event(key_ready_event)
            with torch.cuda.stream(stream_main):
                B_tensor_hash = torch.empty(
                    32, dtype=torch.uint8, device=device
                )
                tensor_hash(
                    B.view(torch.uint8),
                    key_tensor,
                    B_tensor_hash,
                    slot.tensor_hash_scratchpad,
                )
                commitment_hash_B_tensor = torch.empty(
                    32, dtype=torch.uint8, device=device
                )
                EBR = torch.empty((n, r), dtype=torch.int8, device=device)
                EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
                EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
                EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
                BpEB = torch.empty((n, k), dtype=torch.int8, device=device)

            # stream_prep needs B_tensor_hash before commitment_hash below.
            b_hash_done = torch.cuda.Event()
            b_hash_done.record(stream_main)
            stream_prep.wait_event(b_hash_done)

        # commitment_hash + A-side noise on stream_prep
        with torch.cuda.stream(stream_prep):
            if cached is not None:
                # commitment_hash_from_merkle_roots always writes both A and
                # B. On a cache hit we already have commitment_hash_B; use a
                # throwaway for the B output. Cost: blake3 of 64 bytes,
                # negligible.
                commitment_hash_B_throwaway = torch.empty(
                    32, dtype=torch.uint8, device=device
                )
                commitment_hash_from_merkle_roots(
                    slot.A_tensor_hash,
                    B_tensor_hash,
                    key_tensor,
                    slot.commitment_hash_A,
                    commitment_hash_B_throwaway,
                )
                del commitment_hash_B_throwaway

                noise_gen(
                    R=r,
                    EAL=slot.EAL,
                    EAL_fp16=slot.EAL_fp16,
                    EAR_R_major=slot.EAR_R_major,
                    EAR_K_major=slot.EAR_K_major,
                    key_A=slot.commitment_hash_A,
                )
            else:
                commitment_hash_from_merkle_roots(
                    slot.A_tensor_hash,
                    B_tensor_hash,
                    key_tensor,
                    slot.commitment_hash_A,
                    commitment_hash_B_tensor,
                )

                # Cache miss path: B-side noise factors generated together
                # with A-side. stream_prep is already in front of
                # B_tensor_hash via b_hash_done; commitment_hash_B is also
                # ready by the time noise_gen reads it (same stream).
                noise_gen(
                    R=r,
                    EAL=slot.EAL,
                    EAL_fp16=slot.EAL_fp16,
                    EAR_R_major=slot.EAR_R_major,
                    EAR_K_major=slot.EAR_K_major,
                    EBL_R_major=EBL_R_major,
                    EBL_K_major=EBL_K_major,
                    EBR=EBR,
                    EBR_fp16=EBR_fp16,
                    key_A=slot.commitment_hash_A,
                    key_B=commitment_hash_B_tensor,
                )

        prep_done_event = torch.cuda.Event()
        prep_done_event.record(stream_prep)

        # ===== Main kernel on stream_main, waits for prep_done =====
        stream_main.wait_event(prep_done_event)

        host_signal_header_pinned = get_pinned_pool().acquire()
        pow_target_tensor = make_pow_target_tensor(adjusted_target)
        run_noising_B = (cached is None)

        with torch.cuda.stream(stream_main):
            # Slot's host_signal_sync is reused across iterations; reset
            # before the kernel reads it.
            slot.host_signal_sync.zero_()

            noisy_gemm(
                A=slot.A,
                B=B,
                EAL=slot.EAL,
                EAL_fp16=slot.EAL_fp16,
                EBR=EBR,
                EBR_fp16=EBR_fp16,
                EAR_R_major=slot.EAR_R_major,
                EBL_R_major=EBL_R_major,
                EAR_K_major=slot.EAR_K_major,
                EBL_K_major=EBL_K_major,
                AxEBL_fp16=slot.A_E_BL,
                EARxBpEB_fp16=slot.EARxBpEB,
                ApEA=slot.ApEA,
                BpEB=BpEB,
                A_scales=slot.A_scales,
                B_scales=B_scales,
                C=slot.C,
                host_signal_header_pinned=host_signal_header_pinned,
                host_signal_sync=slot.host_signal_sync,
                pow_target=pow_target_tensor,
                pow_key=slot.commitment_hash_A.view(torch.uint32),
                tile_size_m=settings.tile_size_m,
                tile_size_n=settings.tile_size_n,
                tile_size_k=settings.tile_size_k,
                run_noising_A=True,
                run_noising_B=run_noising_B,
                skip_reduction=False,
                skip_denoising=False,
            )

        completion_event = torch.cuda.Event()
        completion_event.record(stream_main)

        # Post-launch: populate cache and schedule status-check callback.
        if b_cache is not None and cached is None:
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
            inner_cb = StatusCheckCallback(
                host_signal_header_pinned=host_signal_header_pinned,
                commitment_hash_A_tensor=slot.commitment_hash_A,
                commitment_hash_B_tensor=commitment_hash_B_tensor,
                A=slot.A,
                B=B,
                mining_job=mining_job,
            )
            wrapped_cb = (
                _SlotReleasingCallback(inner_cb, on_callback_done)
                if on_callback_done is not None
                else inner_cb
            )

            # Async-enabled check already happened in preflight at
            # function entry. schedule_status_check could still raise
            # for other internal reasons; outer finally handles cleanup
            # via the gpu_work_queued + scheduled_or_owned flags.
            get_async_manager().schedule_status_check(completion_event, wrapped_cb)
            host_signal_header_pinned = None  # owned by callback
            # Ownership of on_callback_done has been transferred to the
            # callback wrapper; outer finally must NOT release.
            scheduled_or_owned = True
        else:
            get_pinned_pool().release(host_signal_header_pinned)
            host_signal_header_pinned = None
            # No callback scheduled — outer finally releases the slot.

        return slot.C, b_cache_hit, completion_event

    finally:
        if not scheduled_or_owned:
            # If sync fails inside the helper, it raises and we
            # deliberately do NOT release the slot (or pinned header).
            # Leak is the correct behavior — see helper docstring.
            _cleanup_unscheduled_slot(
                gpu_work_queued=gpu_work_queued,
                completion_event=completion_event,
                host_signal_header_pinned=host_signal_header_pinned,
                release_pinned_header=lambda h: get_pinned_pool().release(h),
                on_callback_done=on_callback_done,
                device=slot.A.device,
            )


# Legacy single-stream cached call retained for profile_run.py compatibility.
# profile_run uses synchronous CUDA-event timing where multi-stream would
# muddy the per-phase breakdown, so it keeps the simpler Phase B body.
def pearl_gemm_noisy_cached(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    out_dtype: torch.dtype,
    matmul_config,
    settings,
    b_cache: Optional[BSideCache] = None,
    submit_block: bool = True,
) -> tuple[torch.Tensor, bool]:
    """Phase B single-stream cached call. Kept for profile_run.py."""
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    from pearl_gemm import get_host_signal_sync_size, get_required_scratchpad_bytes

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

    A_tensor_hash = torch.empty(32, device=device, dtype=torch.uint8)
    tensor_hash(
        A.view(torch.uint8),
        key_tensor,
        A_tensor_hash,
        tensor_hash_scratchpad,
    )

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
            B.view(torch.uint8),
            key_tensor,
            B_tensor_hash,
            tensor_hash_scratchpad,
        )

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

    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)

    if cached is not None:
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

    if cached is None:
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

    if b_cache is not None and cached is None:
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
        host_signal_header_pinned = None
    else:
        get_pinned_pool().release(host_signal_header_pinned)
        del host_signal_header_pinned

    return C, b_cache_hit
