"""Verify CompletionTracker correctly tracks async CUDA work."""

import time

import torch

from direct_miner.completion_tracker import CompletionTracker


def test_basic_completion():
    """Launch 3 matmuls, drain, verify count."""
    tracker = CompletionTracker(max_in_flight=4)

    for _ in range(3):
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        _c = torch.matmul(a, b)
        tracker.record_launch(a, b)

    assert tracker.current_in_flight() == 3, (
        f"Expected 3 in flight, got {tracker.current_in_flight()}"
    )

    tracker.drain()

    assert tracker.completed_count == 3, (
        f"Expected 3 completed, got {tracker.completed_count}"
    )
    assert tracker.current_in_flight() == 0
    print("test_basic_completion: PASS")


def test_wait_for_slot_blocks():
    """Verify wait_for_slot blocks at max_in_flight."""
    tracker = CompletionTracker(max_in_flight=2)

    for _ in range(2):
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        _c = torch.matmul(a, b)
        tracker.record_launch(a, b)

    assert tracker.current_in_flight() == 2

    start = time.time()
    tracker.wait_for_slot()
    elapsed = time.time() - start

    assert tracker.current_in_flight() < 2, (
        f"After wait_for_slot expected <2, got {tracker.current_in_flight()}"
    )

    tracker.drain()
    assert tracker.completed_count == 2
    print(f"test_wait_for_slot_blocks: PASS (waited {elapsed*1000:.1f}ms)")


def test_tensor_refs_kept_alive():
    """Verify tensor refs in tracker prevent GC."""
    import gc
    tracker = CompletionTracker(max_in_flight=4)

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    initial_id = id(a)

    _c = torch.matmul(a, b)
    tracker.record_launch(a, b)

    del a, b
    gc.collect()

    assert tracker.current_in_flight() == 1
    entry = tracker.in_flight[0]
    assert id(entry[1]) == initial_id, "Tracker dropped tensor ref prematurely"

    tracker.drain()
    print("test_tensor_refs_kept_alive: PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        exit(0)

    test_basic_completion()
    test_wait_for_slot_blocks()
    test_tensor_refs_kept_alive()
    print("\nAll tests passed.")
