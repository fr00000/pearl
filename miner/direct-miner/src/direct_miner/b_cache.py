"""B-side artifact cache for Phase B mining optimization.

Caches per-template B-derived intermediate tensors so they are computed
once per template and reused across mining iterations within the same
template epoch.

Cache key: hash_key = blake3(incomplete_header_bytes || mining_config.to_bytes())
            (defined by miner_base.commitment_hash.CommitmentHasher.get_key)
Invalidation: triggered by hash_key change (new block template).

What gets cached (per-template, all confirmed via source review):
    B_tensor_hash         depends on (B, hash_key)
    commitment_hash_B     = blake3(hash_key + B_merkle_root)
    EBR, EBR_fp16         from noise_gen(key_B=commitment_hash_B)
    EBL_R_major, EBL_K_major  from noise_gen(key_B=commitment_hash_B)
    BpEB                  noisy_gemm B-noising output, depends only on
                          B + EBL + EBR

What is NEVER cached (per-call, depends on per-iteration A):
    A_tensor_hash, commitment_hash_A
    EAL, EAL_fp16
    EAR_R_major, EAR_K_major
    EARxBpEB
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch


logger = logging.getLogger(__name__)


@dataclass
class BSideArtifacts:
    """All B-side artifacts derived from one (B, hash_key) pair.

    All tensors live on CUDA. Holding a BSideArtifacts keeps these
    GPU allocations alive — the cache replaces the entry on template
    change which lets the prior allocations free.
    """
    B_tensor_hash: torch.Tensor          # (32,) uint8
    commitment_hash_B: torch.Tensor      # (32,) uint8
    EBR: torch.Tensor                    # (n, r) int8
    EBR_fp16: torch.Tensor               # (n, r) float16
    EBL_R_major: torch.Tensor            # (k, r) int8
    EBL_K_major: torch.Tensor            # (r, k) int8
    BpEB: torch.Tensor                   # (n, k) int8 — largest item


class BSideCache:
    """Single-entry cache keyed by hash_key.

    Phase B keeps it simple: one cached entry at a time. On template
    change, the entry is replaced and the prior tensors become eligible
    for GC (as long as no in-flight kernel still references them).
    """

    def __init__(self):
        self._cached_hash_key: Optional[bytes] = None
        self._artifacts: Optional[BSideArtifacts] = None
        self._lock = threading.Lock()

        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def get(self, hash_key: bytes) -> Optional[BSideArtifacts]:
        """Return cached artifacts if hash_key matches, else None.
        Counts hit/miss as a side effect."""
        with self._lock:
            if (
                self._cached_hash_key is not None
                and self._artifacts is not None
                and self._cached_hash_key == hash_key
            ):
                self.hits += 1
                return self._artifacts
            self.misses += 1
            return None

    def put(self, hash_key: bytes, artifacts: BSideArtifacts) -> None:
        """Store artifacts under hash_key, replacing any existing entry."""
        with self._lock:
            if (
                self._cached_hash_key is not None
                and self._cached_hash_key != hash_key
            ):
                self.invalidations += 1
                logger.debug(
                    "B-cache invalidating: %s -> %s",
                    self._cached_hash_key[:8].hex(),
                    hash_key[:8].hex(),
                )
            self._cached_hash_key = hash_key
            self._artifacts = artifacts

    def current_hash_key_prefix(self) -> Optional[str]:
        with self._lock:
            return (
                self._cached_hash_key[:8].hex()
                if self._cached_hash_key
                else None
            )

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "invalidations": self.invalidations,
                "current_key_prefix": (
                    self._cached_hash_key[:8].hex()
                    if self._cached_hash_key
                    else None
                ),
            }
