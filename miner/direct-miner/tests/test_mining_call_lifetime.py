"""Tests for pearl_gemm_noisy_phase_c slot-lifetime guarantees.

v3 closes two remaining holes from v2 (6ca474a):
1. Preflight: async-enabled check must happen BEFORE any GPU work
   so the failure path has no in-flight kernel writing to slot.A.
2. Post-launch guard: if scheduling fails after kernel launch, the
   completion event must be synchronized before slot release.

The preflight is unit-testable without CUDA by mocking
get_async_manager and asserting pearl_gemm.noisy_gemm was never
invoked. The post-launch synchronize contract requires a CUDA
runtime or extensive mocking; documented as an integration-test
stub below with the rationale.
"""

from unittest.mock import MagicMock, patch

import pytest


class _FakeManager:
    """Minimal stand-in for AsyncLoopManager exposing only the conf flag."""

    def __init__(self, enable_async: bool):
        self._conf = MagicMock()
        self._conf.enable_async_cuda_event_processing = enable_async

    def schedule_status_check(self, event, callback):
        raise AssertionError(
            "schedule_status_check must not be called when async is disabled"
        )


def test_preflight_raises_before_gpu_work_when_async_disabled():
    """When enable_async_cuda_event_processing is False, the preflight
    in pearl_gemm_noisy_phase_c must raise RuntimeError BEFORE any
    GPU work (pinned-header acquisition, A-generation, noisy_gemm
    launch). Verified by patching get_async_manager and noisy_gemm:
    if noisy_gemm is called, the preflight ran too late."""

    from direct_miner import mining_call

    fake_manager = _FakeManager(enable_async=False)
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
                on_callback_done=lambda: None,
            )

    # The whole point of preflight: no GPU-side function ran.
    noisy_gemm_called.assert_not_called()


def test_preflight_passes_when_async_enabled():
    """When async is enabled, preflight does not raise. (Function will
    fail later for other reasons in this test since we don't provide
    real CUDA context, but we shouldn't see the preflight RuntimeError.)"""

    from direct_miner import mining_call

    fake_manager = _FakeManager(enable_async=True)

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
                submit_block=True,
                on_callback_done=lambda: None,
            )
        except RuntimeError as e:
            assert "async CUDA event processing" not in str(e), (
                f"Preflight raised when it shouldn't have: {e}"
            )
        except Exception:
            # Any other exception is fine — we only check preflight didn't fire.
            pass


def test_preflight_skipped_when_submit_block_false():
    """submit_block=False path doesn't need the async manager at all;
    preflight should skip the check."""

    from direct_miner import mining_call

    fake_manager = _FakeManager(enable_async=False)  # disabled

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
                submit_block=False,  # no callback needed
                on_callback_done=lambda: None,
            )
        except RuntimeError as e:
            assert "async CUDA event processing" not in str(e), (
                "Preflight should not fire when submit_block=False"
            )
        except Exception:
            pass


def test_post_launch_synchronize_on_scheduling_failure():
    """Documented as integration test — requires CUDA runtime.

    Verifies that if schedule_status_check raises AFTER kernel launch,
    the outer finally calls completion_event.synchronize() before
    releasing the slot, and releases the pinned header that would
    otherwise leak.

    The contract is verifiable by inspection of mining_call.py:
    - kernel_launched is set to True only after completion_event.record()
    - outer finally checks kernel_launched and synchronizes before release
    - outer finally releases host_signal_header_pinned if still owned
    - on_callback_done() fires last, ensuring GPU is idle and pinned
      header is back in the pool before the slot becomes acquirable
    """
    pytest.skip(
        "Integration test — requires CUDA runtime. Contract verified "
        "by code review of the outer finally in pearl_gemm_noisy_phase_c."
    )
