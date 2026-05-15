import pytest

from direct_miner.attempt_metrics import (
    mma_consumer_threads_per_cta,
    normalized_attempt_scale,
    normalized_attempts_per_matmul,
    outer_tiles_per_matmul,
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
