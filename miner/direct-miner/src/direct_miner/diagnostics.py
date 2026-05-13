"""Diagnostic metrics for direct mining.

Captures per-matmul template/target/win data after CUDA event completion.
Writes append-only JSONL for offline analysis.

Designed to run continuously through Phase B and beyond.

NOTE on best-hash observability (Case B):
The kernel does not expose per-call best-tile-hash. host_signal_header is
written ONLY on a winning tile (status flips kSignalIdle->kSignalTriggered)
and even then captures tile/thread coordinates and the target, not the
hash value itself. So `best_observed_hash_log2` and `margin_log2` are
always null. We instead record `block_found` — the actual win signal,
read from header.status in the on-complete callback.
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class MatmulMetrics:
    """Per-matmul observation, recorded AFTER GPU completion."""
    ts: float
    matmul_index: int
    phase: str
    template_height: int | None
    template_hash_prefix: str | None
    target_log2: float | None
    best_observed_hash_log2: float | None  # always null in Case B
    margin_log2: float | None              # always null in Case B
    block_found: bool | None
    b_cache_hit: bool | None


class DiagnosticsCollector:
    """Writes per-matmul metrics to JSONL.

    Append-only, never overwrites. Use `phase_tag` to distinguish runs
    sharing the same output file.
    """

    def __init__(
        self,
        output_path: str = "/workspace/direct-miner-metrics.jsonl",
        flush_every_n: int = 100,
        phase_tag: str = "phase_a",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.output_path, "a", buffering=1)
        self.flush_every_n = flush_every_n
        self.phase_tag = phase_tag
        self._buffer: list[MatmulMetrics] = []
        self._matmul_index = 0

        self._best_margin_seen_log2: float | None = None
        self._total_matmuls_recorded = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._blocks_found = 0

        self._fh.write(json.dumps({
            "ts": time.time(),
            "event": "session_start",
            "phase": phase_tag,
            "output_path": str(self.output_path),
        }) + "\n")
        self._fh.flush()

        logger.info(
            f"Diagnostics writing to {self.output_path} "
            f"(phase={phase_tag}, flush every {flush_every_n})"
        )

    def record(
        self,
        template_height: int | None,
        template_hash_prefix: str | None,
        target_log2: float | None,
        best_observed_hash_log2: float | None,
        block_found: bool | None = None,
        b_cache_hit: bool | None = None,
    ) -> None:
        """Record one matmul's diagnostic data."""
        self._matmul_index += 1

        margin_log2: float | None = None
        if target_log2 is not None and best_observed_hash_log2 is not None:
            margin_log2 = best_observed_hash_log2 - target_log2
            if (
                margin_log2 >= 0
                and (
                    self._best_margin_seen_log2 is None
                    or margin_log2 < self._best_margin_seen_log2
                )
            ):
                self._best_margin_seen_log2 = margin_log2

        if b_cache_hit is True:
            self._cache_hits += 1
        elif b_cache_hit is False:
            self._cache_misses += 1

        if block_found:
            self._blocks_found += 1

        m = MatmulMetrics(
            ts=time.time(),
            matmul_index=self._matmul_index,
            phase=self.phase_tag,
            template_height=template_height,
            template_hash_prefix=template_hash_prefix,
            target_log2=target_log2,
            best_observed_hash_log2=best_observed_hash_log2,
            margin_log2=margin_log2,
            block_found=block_found,
            b_cache_hit=b_cache_hit,
        )
        self._buffer.append(m)
        self._total_matmuls_recorded += 1

        if len(self._buffer) >= self.flush_every_n:
            self._flush()

    def _flush(self) -> None:
        for m in self._buffer:
            self._fh.write(json.dumps(asdict(m)) + "\n")
        self._fh.flush()
        self._buffer.clear()

    def best_margin_log2(self) -> float | None:
        return self._best_margin_seen_log2

    def total_recorded(self) -> int:
        return self._total_matmuls_recorded

    def cache_stats(self) -> tuple[int, int]:
        """(hits, misses) for B-side cache (Phase B only)."""
        return self._cache_hits, self._cache_misses

    def blocks_found(self) -> int:
        return self._blocks_found

    def close(self) -> None:
        self._flush()
        self._fh.write(json.dumps({
            "ts": time.time(),
            "event": "session_end",
            "phase": self.phase_tag,
            "total_matmuls": self._total_matmuls_recorded,
            "best_margin_log2_seen": self._best_margin_seen_log2,
            "blocks_found": self._blocks_found,
            "b_cache_hits": self._cache_hits,
            "b_cache_misses": self._cache_misses,
        }) + "\n")
        self._fh.flush()
        self._fh.close()


def hash256_to_log2(hash_bytes: bytes) -> float:
    """Convert a 256-bit hash to log2 for margin math.

    Pearl uses little-endian 256-bit integers (see proof.rs).
    Returns -inf for zero hash (defensively handled).
    """
    if len(hash_bytes) != 32:
        raise ValueError(f"Expected 32-byte hash, got {len(hash_bytes)} bytes")
    n = int.from_bytes(hash_bytes, byteorder="little")
    if n == 0:
        return float("-inf")
    return math.log2(n)


def target_to_log2(target_int: int) -> float:
    """Convert target integer to log2."""
    if target_int <= 0:
        return float("-inf")
    return math.log2(target_int)
