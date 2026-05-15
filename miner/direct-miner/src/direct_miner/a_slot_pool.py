"""Per-iteration A-side tensor pool for Phase C multi-stream mining.

Each in-flight iteration gets its own slot of A-side intermediates so
iter N's tensors aren't overwritten while iter N+1 prepares on the
other stream. Slot count = max_in_flight; CompletionTracker bounds
launched work, and per-slot Events ensure async callbacks have
released their tensor references before slot reuse.

Tensors are allocated once at startup and reused across the session.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import torch


logger = logging.getLogger(__name__)


class SlotAcquireTimeout(RuntimeError):
    """Raised when ASlotPool.acquire() times out waiting for a slot's
    previous async callback to release it. Indicates a stuck callback;
    the safe response is to stop the miner, not to proceed and risk
    reading slot.A while the next iteration writes it.
    """
    pass


@dataclass
class ASlot:
    """A-side intermediate tensors owned by one in-flight slot."""

    A: torch.Tensor                       # (m, k) int8 — overwritten per iter
    A_scales: torch.Tensor                # (m,) fp32 — overwritten per iter

    A_tensor_hash: torch.Tensor           # (32,) uint8
    commitment_hash_A: torch.Tensor       # (32,) uint8

    EAL: torch.Tensor                     # (m, r) int8
    EAL_fp16: torch.Tensor                # (m, r) fp16
    EAR_R_major: torch.Tensor             # (k, r) int8
    EAR_K_major: torch.Tensor             # (r, k) int8

    EARxBpEB: torch.Tensor                # (n, r) fp16
    ApEA: torch.Tensor                    # (m, k) int8
    A_E_BL: torch.Tensor                  # (m, r) fp16

    C: Optional[torch.Tensor]             # (m, n) bf16 output; None for headless
    host_signal_sync: torch.Tensor        # (host_signal_sync_size,) int8
    tensor_hash_scratchpad: torch.Tensor  # uint8


class ASlotPool:
    """Round-robin allocator over num_slots pre-built ASlot objects.

    Slot reuse is gated on two conditions:
    1. CompletionTracker has reaped the previous CUDA event (kernel done)
    2. Any async callback holding refs to this slot has called release()

    Without (2), the win-path StatusCheckCallback could read slot.A
    while the next iteration's kernel is mid-write — corrupt proof
    submission. release() is invoked by the _SlotReleasingCallback
    wrapper in mining_call.py.
    """

    def __init__(
        self,
        num_slots: int,
        m: int,
        n: int,
        k: int,
        noise_rank: int,
        host_signal_sync_size: int,
        scratchpad_bytes: int,
        device: torch.device | str = "cuda",
        out_dtype: torch.dtype = torch.bfloat16,
        allocate_c: bool = True,
    ):
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.num_slots = num_slots
        self._next_slot = 0

        self.slots: List[ASlot] = []
        for _ in range(num_slots):
            slot = ASlot(
                A=torch.empty((m, k), dtype=torch.int8, device=device),
                A_scales=torch.empty(m, dtype=torch.float32, device=device),
                A_tensor_hash=torch.empty(32, dtype=torch.uint8, device=device),
                commitment_hash_A=torch.empty(32, dtype=torch.uint8, device=device),
                EAL=torch.empty((m, noise_rank), dtype=torch.int8, device=device),
                EAL_fp16=torch.empty((m, noise_rank), dtype=torch.float16, device=device),
                EAR_R_major=torch.empty((k, noise_rank), dtype=torch.int8, device=device),
                EAR_K_major=torch.empty((noise_rank, k), dtype=torch.int8, device=device),
                EARxBpEB=torch.empty((n, noise_rank), dtype=torch.float16, device=device),
                ApEA=torch.empty((m, k), dtype=torch.int8, device=device),
                A_E_BL=torch.empty((m, noise_rank), dtype=torch.float16, device=device),
                C=(
                    torch.empty((m, n), dtype=out_dtype, device=device)
                    if allocate_c
                    else None
                ),
                host_signal_sync=torch.zeros(
                    (host_signal_sync_size,), dtype=torch.int8, device=device
                ),
                tensor_hash_scratchpad=torch.empty(
                    scratchpad_bytes, dtype=torch.uint8, device=device
                ),
            )
            self.slots.append(slot)

        # Per-slot lifetime gate. Initially set (slots are free); cleared
        # by acquire(), set by release(). Release is the responsibility
        # of whoever last touched the slot's tensors — typically the
        # _SlotReleasingCallback wrapper in mining_call.py.
        self._callback_done: List[threading.Event] = [
            threading.Event() for _ in range(num_slots)
        ]
        for ev in self._callback_done:
            ev.set()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        per_slot_bytes = (
            m * k * 1                              # A int8
            + m * 4                                 # A_scales fp32
            + 32 + 32                               # hashes
            + m * noise_rank * 1                    # EAL int8
            + m * noise_rank * 2                    # EAL_fp16
            + k * noise_rank * 2                    # EAR_R_major + K_major int8
            + n * noise_rank * 2                    # EARxBpEB fp16
            + m * k * 1                             # ApEA int8
            + m * noise_rank * 2                    # A_E_BL fp16
            + (m * n * 2 if allocate_c else 0)      # C bf16
            + host_signal_sync_size                 # sync int8
            + scratchpad_bytes                      # scratchpad uint8
        )
        total_mb = per_slot_bytes * num_slots / 1024 / 1024
        logger.info(
            f"A-slot pool: {num_slots} slots × "
            f"{per_slot_bytes / 1024 / 1024:.1f} MB = {total_mb:.1f} MB"
        )

    def acquire(self, timeout: float | None = None) -> tuple[int, ASlot]:
        """Return (slot_idx, slot). Blocks until the slot's previous
        async callback (if any) has released it.

        timeout: max seconds to wait. Default None means block
        indefinitely — the right behavior in production, where a stuck
        callback indicates a real bug we'd rather surface as a hang
        (visible to operators) than silently risk tensor corruption.

        Tests and supervised runs may pass a finite timeout; on
        timeout, raises SlotAcquireTimeout. NEVER force-releases the
        slot — the caller is responsible for choosing how to recover
        (typically: drain, exit cleanly, let the bootstrap script's
        idempotent re-run path bring the miner back up).
        """
        slot_idx = self._next_slot
        released = self._callback_done[slot_idx].wait(timeout=timeout)
        if not released:
            raise SlotAcquireTimeout(
                f"Slot {slot_idx} acquire timed out after {timeout}s — "
                "callback from previous iteration has not completed. "
                "This indicates a stuck async callback; the miner should "
                "exit rather than risk reading slot tensors mid-write."
            )
        self._callback_done[slot_idx].clear()
        slot = self.slots[slot_idx]
        self._next_slot = (self._next_slot + 1) % self.num_slots
        return slot_idx, slot

    def release(self, slot_idx: int) -> None:
        """Mark slot as no longer in use by async callbacks.

        Must be called exactly once per acquire(), typically via the
        _SlotReleasingCallback wrapper's finally block in
        mining_call.py. Safe to call on an already-released slot
        (Event.set() is idempotent).
        """
        if 0 <= slot_idx < self.num_slots:
            self._callback_done[slot_idx].set()
        else:
            logger.error(f"release() called with invalid slot_idx={slot_idx}")

    def wait_all_released(self, timeout: float = 30.0) -> bool:
        """Block until every slot has been released or timeout expires.

        Returns True if all slots released within the timeout, False if
        any slot was still held when the deadline passed. Used by the
        shutdown drain path so pending callbacks can finish their
        win-path work before we tear down the process.
        """
        deadline = time.monotonic() + timeout
        for slot_idx, ev in enumerate(self._callback_done):
            remaining = max(0.0, deadline - time.monotonic())
            if not ev.wait(timeout=remaining):
                logger.warning(
                    f"Slot {slot_idx} callback did not complete within "
                    f"{timeout}s shutdown deadline"
                )
                return False
        return True
