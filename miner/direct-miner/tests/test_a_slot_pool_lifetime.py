"""Test that ASlotPool acquire() blocks until release() is called.

Verifies the fix for the slot-reuse-vs-callback race that would
corrupt winning proofs. CPU-only — exercises the lifecycle gate,
not the CUDA tensor work.
"""

import threading
import time

import pytest

from direct_miner.a_slot_pool import ASlotPool


@pytest.fixture
def pool():
    return ASlotPool(
        num_slots=2,
        m=128,
        n=128,
        k=128,
        noise_rank=32,
        host_signal_sync_size=256,
        scratchpad_bytes=1024,
        device="cpu",
    )


def test_acquire_blocks_until_release(pool):
    """Slot reuse must block until the prior callback called release()."""
    # First acquire of slot 0 — clears the gate.
    idx_a, _ = pool.acquire()
    assert idx_a == 0

    # Slot 1 acquires fine (different slot).
    idx_b, _ = pool.acquire()
    assert idx_b == 1

    # Third acquire (round-robin returns to slot 0) must BLOCK.
    acquired_event = threading.Event()

    def try_acquire():
        pool.acquire(timeout=2.0)
        acquired_event.set()

    t = threading.Thread(target=try_acquire, daemon=True)
    t.start()

    # Give it 200ms to (incorrectly) return; should still be blocked.
    assert not acquired_event.wait(0.2), (
        "acquire() returned before release() — race fix is broken"
    )

    # Now release slot 0; the blocked thread should proceed.
    pool.release(0)
    assert acquired_event.wait(1.0), "release() did not unblock acquire()"
    t.join(timeout=1.0)


def test_release_invalid_slot_is_noop(pool):
    """release() with an out-of-range idx logs but doesn't crash."""
    pool.release(-1)
    pool.release(99)
    # Pool still functional after garbage releases.
    idx, slot = pool.acquire()
    assert slot is not None
    assert idx == 0


def test_double_release_is_safe(pool):
    """Calling release() twice for the same slot is idempotent."""
    idx, _ = pool.acquire()
    pool.release(idx)
    pool.release(idx)  # Event.set() second time is a no-op.
    # Subsequent acquire still works.
    idx2, _ = pool.acquire()
    assert idx2 is not None


def test_timeout_unblocks_with_warning(pool):
    """If both slots are held, the third acquire times out and proceeds."""
    pool.acquire()  # holds slot 0
    pool.acquire()  # holds slot 1
    # Third acquire (round-robin slot 0 again) should time out and
    # proceed defensively rather than hang.
    start = time.monotonic()
    idx, slot = pool.acquire(timeout=0.5)
    elapsed = time.monotonic() - start
    assert 0.4 < elapsed < 1.5, f"timeout was {elapsed:.2f}s, expected ~0.5s"
    assert slot is not None


def test_wait_all_released_returns_true_when_idle():
    """Fresh pool starts with all slots released — wait_all returns immediately."""
    p = ASlotPool(
        num_slots=4, m=64, n=64, k=64, noise_rank=16,
        host_signal_sync_size=128, scratchpad_bytes=512, device="cpu",
    )
    start = time.monotonic()
    assert p.wait_all_released(timeout=1.0) is True
    assert time.monotonic() - start < 0.1


def test_wait_all_released_returns_false_on_held_slot():
    """If a slot is held without release, wait_all reports failure."""
    p = ASlotPool(
        num_slots=2, m=64, n=64, k=64, noise_rank=16,
        host_signal_sync_size=128, scratchpad_bytes=512, device="cpu",
    )
    p.acquire()  # holds slot 0
    start = time.monotonic()
    assert p.wait_all_released(timeout=0.3) is False
    elapsed = time.monotonic() - start
    assert 0.2 < elapsed < 0.8
