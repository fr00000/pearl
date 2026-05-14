"""Test ASlotPool slot-lifetime contract (v2: fail-closed).

v1 used acquire(timeout=30.0) and force-released on expiry, reopening
the corruption window the fix was meant to close. v2 raises
SlotAcquireTimeout and never mutates state on timeout. Default
timeout=None blocks indefinitely so production miners surface stuck
callbacks as visible hangs (operator-recoverable) rather than silent
risk.
"""

import threading
import time

import pytest

from direct_miner.a_slot_pool import ASlotPool, SlotAcquireTimeout


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
    idx_a, _ = pool.acquire()
    assert idx_a == 0

    idx_b, _ = pool.acquire()
    assert idx_b == 1

    # Third acquire (round-robin to slot 0) must BLOCK.
    acquired_event = threading.Event()

    def try_acquire():
        try:
            pool.acquire(timeout=2.0)
            acquired_event.set()
        except SlotAcquireTimeout:
            pass

    t = threading.Thread(target=try_acquire, daemon=True)
    t.start()

    assert not acquired_event.wait(0.2), (
        "acquire() returned before release() — race fix is broken"
    )

    pool.release(0)
    assert acquired_event.wait(1.0), "release() did not unblock acquire()"
    t.join(timeout=1.0)


def test_release_invalid_slot_is_noop(pool):
    """release() with an out-of-range idx logs but doesn't crash."""
    pool.release(-1)
    pool.release(99)
    idx, slot = pool.acquire()
    assert slot is not None
    assert idx == 0


def test_double_release_is_safe(pool):
    """Calling release() twice for the same slot is idempotent."""
    idx, _ = pool.acquire()
    pool.release(idx)
    pool.release(idx)
    idx2, _ = pool.acquire()
    assert idx2 is not None


def test_acquire_raises_on_timeout(pool):
    """When both slots are held, acquire(timeout=...) raises rather
    than force-releasing. The v1 'proceed anyway' behavior reopened
    the corruption window; v2 fails closed."""
    pool.acquire()  # holds slot 0
    pool.acquire()  # holds slot 1

    start = time.monotonic()
    with pytest.raises(SlotAcquireTimeout, match="Slot 0 acquire timed out"):
        pool.acquire(timeout=0.5)
    elapsed = time.monotonic() - start
    assert 0.4 < elapsed < 1.5, f"timeout was {elapsed:.2f}s, expected ~0.5s"


def test_timeout_does_not_mark_slot_released(pool):
    """After acquire() times out, the slot's Event is still cleared.
    A subsequent release() must unblock future acquires; the timeout
    must NOT have set the Event itself (that was the v1 bug)."""
    pool.acquire()  # holds slot 0
    pool.acquire()  # holds slot 1

    with pytest.raises(SlotAcquireTimeout):
        pool.acquire(timeout=0.2)

    # Now release slot 0; future acquire should succeed and return slot 0
    # (round-robin _next_slot did not advance on the failure path).
    pool.release(0)
    idx, _ = pool.acquire(timeout=1.0)
    assert idx == 0, (
        "Expected slot 0 (timeout should not have advanced _next_slot); "
        f"got slot {idx}"
    )


def test_acquire_with_no_timeout_blocks_indefinitely(pool):
    """The default timeout=None must block forever, not silently succeed.
    Test via short-lived thread: it should still be alive after 500ms."""
    pool.acquire()  # holds slot 0
    pool.acquire()  # holds slot 1

    finished = threading.Event()

    def try_acquire_no_timeout():
        pool.acquire()  # default timeout=None — must block forever
        finished.set()

    t = threading.Thread(target=try_acquire_no_timeout, daemon=True)
    t.start()

    assert not finished.wait(0.5), (
        "acquire() with default timeout=None should block indefinitely; "
        "it returned without anyone calling release()"
    )

    # Unblock for clean test exit.
    pool.release(0)
    assert finished.wait(1.0)
    t.join(timeout=1.0)


def test_wait_all_released_returns_true_when_idle():
    p = ASlotPool(
        num_slots=4, m=64, n=64, k=64, noise_rank=16,
        host_signal_sync_size=128, scratchpad_bytes=512, device="cpu",
    )
    start = time.monotonic()
    assert p.wait_all_released(timeout=1.0) is True
    assert time.monotonic() - start < 0.1


def test_wait_all_released_returns_false_on_held_slot():
    p = ASlotPool(
        num_slots=2, m=64, n=64, k=64, noise_rank=16,
        host_signal_sync_size=128, scratchpad_bytes=512, device="cpu",
    )
    p.acquire()  # holds slot 0
    start = time.monotonic()
    assert p.wait_all_released(timeout=0.3) is False
    elapsed = time.monotonic() - start
    assert 0.2 < elapsed < 0.8
