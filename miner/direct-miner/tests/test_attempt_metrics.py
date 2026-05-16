import pytest

from direct_miner.attempt_metrics import (
    chance_weighted_attempts_per_matmul,
    mma_consumer_threads_per_cta,
    normalized_attempt_scale,
    normalized_attempts_per_matmul,
    outer_tiles_per_matmul,
    rounded_common_dim,
)


def test_outer_tiles_per_matmul_uses_kernel_tile_shape():
    assert outer_tiles_per_matmul(
        m=8192, n=524032, tile_m=128, tile_n=256
    ) == 131_008
    assert outer_tiles_per_matmul(
        m=8192, n=524032, tile_m=64, tile_n=256
    ) == 262_016


def test_normalized_attempts_prevent_tile_m_64_overcount():
    assert mma_consumer_threads_per_cta(tile_m=128) == 256
    assert mma_consumer_threads_per_cta(tile_m=64) == 128
    assert normalized_attempt_scale(tile_m=128) == 1.0
    assert normalized_attempt_scale(tile_m=64) == 0.5
    assert normalized_attempts_per_matmul(
        m=8192, n=524032, tile_m=128, tile_n=256
    ) == normalized_attempts_per_matmul(
        m=8192, n=524032, tile_m=64, tile_n=256
    )


def test_invalid_tile_sizes_raise():
    with pytest.raises(ValueError):
        outer_tiles_per_matmul(m=1, n=1, tile_m=0, tile_n=256)
    with pytest.raises(ValueError):
        mma_consumer_threads_per_cta(tile_m=96)


def test_chance_weighted_attempts_scale_by_rounded_common_dim():
    current_per_matmul = chance_weighted_attempts_per_matmul(
        m=8192, n=524032, k=8192, rank=128, tile_m=128, tile_n=256
    )
    high_k_per_matmul = chance_weighted_attempts_per_matmul(
        m=8192, n=261888, k=16384, rank=128, tile_m=128, tile_n=256
    )

    assert rounded_common_dim(k=8192, rank=128) == 8192
    assert rounded_common_dim(k=8193, rank=128) == 8192
    assert current_per_matmul == 131_008 * 8192
    assert high_k_per_matmul == 65_472 * 16_384
