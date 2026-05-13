"""Tracks actual GPU completion via CUDA events.

CRITICAL: PyTorch CUDA operations are asynchronous. Calling pearl_gemm_noisy
queues work to a stream and returns immediately — the GPU has not yet
completed (or possibly even started) the work. Incrementing a counter at
the call site measures Python's launch rate, not GPU throughput.

This module uses torch.cuda.Event to record actual completion times,
and bounds in-flight work to prevent the Python launch loop from
outpacing GPU drain (which causes OOM or stream backpressure stalls).
"""

from collections import deque
from typing import Any

import torch


class CompletionTracker:
    """Tracks asynchronous CUDA work completion using events.

    Usage pattern:
        tracker = CompletionTracker(max_in_flight=2)
        for _ in range(N):
            tracker.wait_for_slot()       # block if at max
            tracker.reap_completed()      # update counters for finished
            A, A_scales = make_a(...)
            pearl_gemm_noisy(A, ...)      # async launch
            tracker.record_launch(A, A_scales)  # event + ref hold
        tracker.drain()  # on shutdown
    """

    def __init__(self, max_in_flight: int):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        self.max_in_flight = max_in_flight
        # Each entry: (cuda_event, *tensor_refs_to_hold_alive)
        # Holding tensor refs prevents GC from freeing tensors the GPU is using
        self.in_flight: deque[tuple[Any, ...]] = deque()
        self.completed_count: int = 0

    def record_launch(self, *tensor_refs: torch.Tensor) -> None:
        """Record event after the latest kernel launch.

        tensor_refs are stored in the queue and held alive until the
        event completes. Without this, Python GC could free tensors
        the GPU is still actively using.
        """
        event = torch.cuda.Event()
        event.record()
        self.in_flight.append((event, *tensor_refs))

    def reap_completed(self) -> int:
        """Pop completed entries. Returns count freed in this call."""
        freed = 0
        while self.in_flight and self.in_flight[0][0].query():
            # event.query() is non-blocking; True means all preceding
            # work in the stream where the event was recorded is done
            self.in_flight.popleft()
            self.completed_count += 1
            freed += 1
        return freed

    def wait_for_slot(self) -> None:
        """Block until in_flight count drops below max_in_flight.

        First tries non-blocking reap; if still at cap, synchronizes
        on the oldest event (most likely to complete soonest).
        """
        while len(self.in_flight) >= self.max_in_flight:
            if self.reap_completed() > 0:
                continue
            # Nothing finished yet; block on oldest
            self.in_flight[0][0].synchronize()

    def drain(self) -> None:
        """Wait for all in-flight work to complete (shutdown path)."""
        while self.in_flight:
            self.in_flight[0][0].synchronize()
            self.reap_completed()

    def current_in_flight(self) -> int:
        return len(self.in_flight)
