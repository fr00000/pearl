import math

import torch

from direct_miner.diagnostics import decode_pow_diagnostics


def test_decode_pow_diagnostics_reads_full_hash_and_coordinates():
    t = torch.zeros(16, dtype=torch.uint32)
    t[0] = 123
    t[1] = 0x01020304
    words = [
        0x89ABCDEF,
        0x01234567,
        0x11111111,
        0x22222222,
        0x33333333,
        0x44444444,
        0x55555555,
        0x01020304,
    ]
    for i, word in enumerate(words):
        t[4 + i] = word
    t[12] = 7
    t[13] = 8
    t[14] = 0
    t[15] = 42

    decoded = decode_pow_diagnostics(t)

    expected_int = sum(word << (32 * i) for i, word in enumerate(words))
    assert decoded["kernel_hash_attempts"] == 123
    assert decoded["kernel_best_hash_hex"] == expected_int.to_bytes(
        32, byteorder="little"
    ).hex()
    assert decoded["kernel_best_hash_log2"] == math.log2(expected_int)
    assert decoded["kernel_best_tile_coord"] == [7, 8, 0]
    assert decoded["kernel_best_thread_idx"] == 42


def test_decode_pow_diagnostics_handles_empty_buffer():
    t = torch.zeros(16, dtype=torch.uint32)

    decoded = decode_pow_diagnostics(t)

    assert decoded["kernel_hash_attempts"] == 0
    assert decoded["kernel_best_hash_hex"] is None
    assert decoded["kernel_best_hash_log2"] is None
    assert decoded["kernel_best_tile_coord"] is None
    assert decoded["kernel_best_thread_idx"] is None
