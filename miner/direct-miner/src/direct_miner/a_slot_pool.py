"""Per-iteration A-side tensor pool for Phase C multi-stream mining.

Each in-flight iteration gets its own slot of A-side intermediates so
iter N's tensors aren't overwritten while iter N+1 prepares on the
other stream. Slot count = max_in_flight; CompletionTracker enforces
the cap, so round-robin slot pickup is always safe.

Tensors are allocated once at startup and reused across the session.
"""

import logging
from dataclasses import dataclass
from typing import List

import torch


logger = logging.getLogger(__name__)


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

    C: torch.Tensor                       # (m, n) bf16 output
    host_signal_sync: torch.Tensor        # (host_signal_sync_size,) int8
    tensor_hash_scratchpad: torch.Tensor  # uint8


class ASlotPool:
    """Round-robin allocator over num_slots pre-built ASlot objects."""

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
                C=torch.empty((m, n), dtype=out_dtype, device=device),
                host_signal_sync=torch.zeros(
                    (host_signal_sync_size,), dtype=torch.int8, device=device
                ),
                tensor_hash_scratchpad=torch.empty(
                    scratchpad_bytes, dtype=torch.uint8, device=device
                ),
            )
            self.slots.append(slot)

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
            + m * n * 2                             # C bf16
            + host_signal_sync_size                 # sync int8
            + scratchpad_bytes                      # scratchpad uint8
        )
        total_mb = per_slot_bytes * num_slots / 1024 / 1024
        logger.info(
            f"A-slot pool: {num_slots} slots × "
            f"{per_slot_bytes / 1024 / 1024:.1f} MB = {total_mb:.1f} MB"
        )

    def acquire(self) -> tuple[int, ASlot]:
        """Return (slot_idx, slot). Caller resets per-iter state as needed."""
        slot_idx = self._next_slot
        slot = self.slots[slot_idx]
        self._next_slot = (self._next_slot + 1) % self.num_slots
        return slot_idx, slot
