"""Tracks actual GPU completion via CUDA events.

CRITICAL: PyTorch CUDA operations are asynchronous. Calling pearl_gemm_noisy
queues work to a stream and returns immediately — the GPU has not yet
completed (or possibly even started) the work. Incrementing a counter at
the call site measures Python's launch rate, not GPU throughput.

This module uses torch.cuda.Event to record actual completion times,
and bounds in-flight work to prevent the Python launch loop from
outpacing GPU drain (which causes OOM or stream backpressure stalls).

Optionally invokes an on_complete callback per finished launch with
metadata supplied at record_launch — fires AFTER the CUDA event has
completed, so any diagnostic reads see actual GPU output.
"""

import logging
from collections import deque
from typing import Any, Callable, Optional

import torch


logger = logging.getLogger(__name__)


class CompletionTracker:
    """Tracks asynchronous CUDA work completion using events.

    Usage pattern:
        tracker = CompletionTracker(max_in_flight=2, on_complete=cb)
        for _ in range(N):
            tracker.wait_for_slot()
            tracker.reap_completed()
            A, A_scales = make_a(...)
            pearl_gemm_noisy(A, ...)
            tracker.record_launch({"meta": ...}, A, A_scales)
        tracker.drain()
    """

    def __init__(
        self,
        max_in_flight: int,
        on_complete: Optional[Callable[[dict], None]] = None,
    ):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self.max_in_flight = max_in_flight
        # Each entry: (event, metadata_dict, *tensor_refs)
        self.in_flight: deque[tuple[Any, ...]] = deque()
        self.completed_count: int = 0
        self._on_complete = on_complete

    def record_launch(
        self,
        metadata: dict | None,
        *tensor_refs: torch.Tensor,
        event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Record event after kernel launch.

        metadata is opaque to the tracker; passed to on_complete callback
        when the event finishes. tensor_refs are stored to keep tensors
        alive until the GPU is done with them.

        event: if provided, used as the completion event (must already
        be recorded on the appropriate stream). If None, the tracker
        creates and records its own event on the current stream — fine
        for single-default-stream callers. Phase C callers pass an
        event recorded on stream_main so the callback fires when the
        main kernel actually completes, not when this method is called
        from outside any stream context.
        """
        if event is None:
            event = torch.cuda.Event()
            event.record()
        self.in_flight.append((event, metadata or {}, *tensor_refs))

    def reap_completed(self) -> int:
        """Pop completed entries; invoke callback for each. Returns count freed."""
        freed = 0
        while self.in_flight and self.in_flight[0][0].query():
            entry = self.in_flight.popleft()
            _event, metadata, *_refs = entry
            self.completed_count += 1
            freed += 1
            if self._on_complete is not None and metadata:
                try:
                    self._on_complete(metadata)
                except Exception:
                    logger.exception("on_complete callback failed")
        return freed

    def wait_for_slot(self) -> None:
        """Block until in_flight count drops below max_in_flight."""
        while len(self.in_flight) >= self.max_in_flight:
            if self.reap_completed() > 0:
                continue
            self.in_flight[0][0].synchronize()

    def drain(self) -> None:
        """Wait for all in-flight work to complete (shutdown path)."""
        while self.in_flight:
            self.in_flight[0][0].synchronize()
            self.reap_completed()

    def current_in_flight(self) -> int:
        return len(self.in_flight)
