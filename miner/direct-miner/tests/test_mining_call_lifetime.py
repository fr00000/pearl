"""Tests for pearl_gemm_noisy_phase_c slot-lifetime guarantees.

The contract that v4 finalizes:

  An acquired A slot is released exactly once, and only after it is
  safe to reuse.

"Safe to reuse" = the GPU is no longer reading from or writing to
slot.A. Concretely: either the async callback fired (success path) or
the outer cleanup helper synchronized the completion event before
calling on_callback_done.

v4 tests:
  - Preflight failure releases the caller's slot (was leaked in v3).
  - Cleanup synchronizes BEFORE calling on_callback_done.
  - Sync failure does NOT call on_callback_done (slot stays held).
  - Pinned header is released after sync, before slot release.

A separate preflight integration check confirms that no GPU work
runs when async is disabled — verifies the preflight is at function
entry, not after the kernel launch.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _cleanup_unscheduled_slot: directly testable; no CUDA needed.
# ---------------------------------------------------------------------------


def test_cleanup_synchronizes_before_slot_release():
    """When the kernel was launched and scheduling didn't complete,
    cleanup must call completion_event.synchronize() BEFORE invoking
    on_callback_done. The order is what makes slot release safe."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    call_order = []

    fake_event = MagicMock()
    fake_event.synchronize.side_effect = lambda: call_order.append("sync")

    release_pinned = MagicMock(side_effect=lambda h: call_order.append("pinned"))
    on_callback_done = MagicMock(side_effect=lambda: call_order.append("slot"))

    _cleanup_unscheduled_slot(
        kernel_launched=True,
        completion_event=fake_event,
        host_signal_header_pinned=object(),  # sentinel
        release_pinned_header=release_pinned,
        on_callback_done=on_callback_done,
    )

    assert call_order == ["sync", "pinned", "slot"], (
        f"Cleanup order wrong: {call_order}. sync must precede pinned "
        "release, which must precede slot release."
    )


def test_cleanup_sync_failure_does_not_release_slot():
    """If completion_event.synchronize() AND torch.cuda.synchronize()
    both fail, cleanup must raise without calling on_callback_done.
    A prematurely released slot can corrupt a winning proof; a
    leaked slot just costs throughput."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    fake_event = MagicMock()
    fake_event.synchronize.side_effect = RuntimeError("CUDA event sync failed")

    release_pinned = MagicMock()
    on_callback_done = MagicMock()

    # Patch torch.cuda.synchronize to also fail.
    with patch("torch.cuda.synchronize", side_effect=RuntimeError("driver dead")):
        with pytest.raises(RuntimeError, match="refusing to release A slot"):
            _cleanup_unscheduled_slot(
                kernel_launched=True,
                completion_event=fake_event,
                host_signal_header_pinned=object(),
                release_pinned_header=release_pinned,
                on_callback_done=on_callback_done,
            )

    on_callback_done.assert_not_called()
    release_pinned.assert_not_called()


def test_cleanup_pinned_released_after_sync_success():
    """When sync succeeds, the pinned header is released. Sequence is
    sync -> pinned release -> slot release."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    pinned_sentinel = object()
    fake_event = MagicMock()  # synchronize() returns None on success
    release_pinned = MagicMock()
    on_callback_done = MagicMock()

    _cleanup_unscheduled_slot(
        kernel_launched=True,
        completion_event=fake_event,
        host_signal_header_pinned=pinned_sentinel,
        release_pinned_header=release_pinned,
        on_callback_done=on_callback_done,
    )

    fake_event.synchronize.assert_called_once_with()
    release_pinned.assert_called_once_with(pinned_sentinel)
    on_callback_done.assert_called_once_with()


def test_cleanup_falls_back_to_torch_cuda_synchronize():
    """If the event-specific synchronize fails, the helper tries
    torch.cuda.synchronize() as a fallback before giving up. Only
    when both fail do we raise and refuse to release."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    fake_event = MagicMock()
    fake_event.synchronize.side_effect = RuntimeError("event sync borked")

    release_pinned = MagicMock()
    on_callback_done = MagicMock()

    # Fallback succeeds.
    with patch("torch.cuda.synchronize") as fallback_sync:
        _cleanup_unscheduled_slot(
            kernel_launched=True,
            completion_event=fake_event,
            host_signal_header_pinned=object(),
            release_pinned_header=release_pinned,
            on_callback_done=on_callback_done,
        )
        fallback_sync.assert_called_once_with()

    # Sync succeeded via fallback → release proceeds normally.
    release_pinned.assert_called_once()
    on_callback_done.assert_called_once_with()


def test_cleanup_no_sync_when_kernel_not_launched():
    """If the kernel never launched, no synchronization is needed.
    Pinned header release and slot release still fire."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    fake_event = MagicMock()
    release_pinned = MagicMock()
    on_callback_done = MagicMock()

    _cleanup_unscheduled_slot(
        kernel_launched=False,
        completion_event=fake_event,
        host_signal_header_pinned=object(),
        release_pinned_header=release_pinned,
        on_callback_done=on_callback_done,
    )

    fake_event.synchronize.assert_not_called()
    release_pinned.assert_called_once()
    on_callback_done.assert_called_once()


def test_cleanup_handles_none_pinned_and_none_callback():
    """If neither pinned header nor callback was set, cleanup is a
    no-op (apart from sync). Defensive against partial initialization."""

    from direct_miner.mining_call import _cleanup_unscheduled_slot

    release_pinned = MagicMock()

    _cleanup_unscheduled_slot(
        kernel_launched=False,
        completion_event=None,
        host_signal_header_pinned=None,
        release_pinned_header=release_pinned,
        on_callback_done=None,
    )

    release_pinned.assert_not_called()


# ---------------------------------------------------------------------------
# Preflight: full pearl_gemm_noisy_phase_c entry path.
# ---------------------------------------------------------------------------


class _FakeManager:
    """Minimal stand-in for AsyncLoopManager exposing only the conf flag."""

    def __init__(self, enable_async: bool):
        self._conf = MagicMock()
        self._conf.enable_async_cuda_event_processing = enable_async


def test_preflight_releases_slot_when_async_disabled():
    """The v4 fix: when async is disabled, the preflight raises
    AND the caller's on_callback_done is invoked exactly once via
    the outer finally — slot is not leaked."""

    from direct_miner import mining_call

    fake_manager = _FakeManager(enable_async=False)
    on_callback_done = MagicMock()
    noisy_gemm_called = MagicMock()

    with patch.object(mining_call, "get_async_manager", return_value=fake_manager), \
         patch.object(mining_call, "noisy_gemm", noisy_gemm_called):
        with pytest.raises(RuntimeError, match="async CUDA event processing"):
            mining_call.pearl_gemm_noisy_phase_c(
                slot=MagicMock(),
                B=MagicMock(),
                B_scales=MagicMock(),
                matmul_config=MagicMock(),
                settings=MagicMock(),
                b_cache=None,
                stream_main=MagicMock(),
                stream_prep=MagicMock(),
                a_generator=None,
                submit_block=True,
                on_callback_done=on_callback_done,
            )

    # The whole point of preflight: no GPU work ran.
    noisy_gemm_called.assert_not_called()
    # The v4 fix: slot is released, not leaked.
    on_callback_done.assert_called_once()


def test_preflight_skipped_when_submit_block_false():
    """submit_block=False path doesn't need the async manager;
    preflight should skip the check."""

    from direct_miner import mining_call

    fake_manager = _FakeManager(enable_async=False)

    with patch.object(mining_call, "get_async_manager", return_value=fake_manager):
        try:
            mining_call.pearl_gemm_noisy_phase_c(
                slot=MagicMock(),
                B=MagicMock(),
                B_scales=MagicMock(),
                matmul_config=MagicMock(),
                settings=MagicMock(),
                b_cache=None,
                stream_main=MagicMock(),
                stream_prep=MagicMock(),
                a_generator=None,
                submit_block=False,
                on_callback_done=lambda: None,
            )
        except RuntimeError as e:
            assert "async CUDA event processing" not in str(e), (
                "Preflight should not fire when submit_block=False"
            )
        except Exception:
            # Any other exception is fine — we only verify preflight didn't fire.
            pass
