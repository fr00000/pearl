"""Throughput accounting helpers for direct mining.

The direct miner's kernel grid is expressed as outer CTAs. That is not always
the same as protocol-comparable PoW attempts: each MMA consumer thread checks
one BLAKE3 transcript, and the number of consumer threads changes with
``tile_m``.
"""

from __future__ import annotations

import math


BASELINE_CONSUMER_THREADS = 256


def outer_tiles_per_matmul(
    *, m: int, n: int, tile_m: int, tile_n: int
) -> int:
    """Return raw outer CTA tiles launched per matmul."""
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError("tile dimensions must be positive")
    return math.ceil(m / tile_m) * math.ceil(n / tile_n)


def mma_consumer_threads_per_cta(*, tile_m: int) -> int:
    """Return the number of per-CTA MMA threads that check PoW.

    This mirrors KernelTraits:
      kNumMmaWarpgroups = tile_m / 64
      kNumMmaThreads = kNumMmaWarpgroups * 128
    """
    if tile_m <= 0 or tile_m % 64 != 0:
        raise ValueError("tile_m must be a positive multiple of 64")
    return (tile_m // 64) * 128


def normalized_attempt_scale(*, tile_m: int) -> float:
    """Return raw outer-tile to 128x256-equivalent attempt scale."""
    return mma_consumer_threads_per_cta(tile_m=tile_m) / BASELINE_CONSUMER_THREADS


def normalized_attempts_per_matmul(
    *, m: int, n: int, tile_m: int, tile_n: int
) -> float:
    """Return comparable PoW attempts per matmul.

    A 128x256 kernel has 256 MMA consumer threads per CTA and is the historical
    baseline. A 64x256 kernel has twice as many CTAs for the same M extent, but
    each CTA has half as many consumer threads, so its raw outer-tile rate must
    be scaled by 0.5 before comparing it.
    """
    return outer_tiles_per_matmul(
        m=m, n=n, tile_m=tile_m, tile_n=tile_n
    ) * normalized_attempt_scale(tile_m=tile_m)


def rounded_common_dim(*, k: int, rank: int) -> int:
    """Return the common dimension used by the protocol difficulty scaling."""
    if k <= 0 or rank <= 0:
        raise ValueError("k and rank must be positive")
    return k - (k % rank)


def chance_weighted_attempts_per_matmul(
    *, m: int, n: int, k: int, rank: int, tile_m: int, tile_n: int
) -> float:
    """Return a shape-comparable expected mining-chance score per matmul.

    The protocol multiplies the base target by h * w * rounded_common_dim.
    For a fixed row/column pattern, h*w is constant, so comparing different
    ``k`` values requires multiplying normalized attempts by rounded_common_dim.
    """
    return normalized_attempts_per_matmul(
        m=m, n=n, tile_m=tile_m, tile_n=tile_n
    ) * rounded_common_dim(k=k, rank=rank)


def protocol_weighted_attempts_per_matmul(
    *,
    m: int,
    n: int,
    k: int,
    rank: int,
    tile_m: int,
    tile_n: int,
    hash_tile_elements: int,
) -> float:
    """Return expected-work units including the proof pattern size.

    ``chance_weighted_attempts_per_matmul`` intentionally omits ``h*w`` because
    all production comparisons used the same row/column pattern. Experiments
    with narrower or wider proof patterns need the full protocol adjustment:
    normalized attempts * h * w * rounded_common_dim.
    """
    if hash_tile_elements <= 0:
        raise ValueError("hash_tile_elements must be positive")
    return (
        chance_weighted_attempts_per_matmul(
            m=m, n=n, k=k, rank=rank, tile_m=tile_m, tile_n=tile_n
        )
        * hash_tile_elements
    )
