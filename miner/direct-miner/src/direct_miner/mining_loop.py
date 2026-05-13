"""Phase A mining loop: minimal, single-stream, no caching."""

import logging
import math
import time
from typing import Optional

import torch

from vllm_miner.gemm_operators import pearl_gemm_noisy
from vllm_miner.mining_state import (
    init_async_manager,
    init_pinned_pool,
    get_async_manager,
)

from .config import MinerConfig
from .completion_tracker import CompletionTracker
from .synthetic_data import FixedBPool, make_synthetic_a


logger = logging.getLogger(__name__)

# Outer tile sizes from the compiled kernel (kernel_traits.hpp).
# Used only for computing reported tile rate; don't hardcode anywhere else.
KERNEL_TILE_SIZE_M = 128
KERNEL_TILE_SIZE_N = 256


class DirectMiner:
    """Phase A synthetic miner: minimal viable implementation.

    Connects to existing pearl-gateway via UDS. Generates fresh A per
    iteration, reuses static B for the session, calls pearl_gemm_noisy,
    tracks completion via CUDA events.

    No caching of B-derived artifacts. No multi-stream. No autotuning.
    Pure baseline to answer: does synthetic mining produce blocks?
    """

    def __init__(self, config: MinerConfig):
        config.shapes.validate()
        self.config = config
        self.b_pool: Optional[FixedBPool] = None
        self.tracker = CompletionTracker(max_in_flight=config.max_in_flight)

        # Pre-compute outer tiles per matmul for reporting
        self._outer_tiles_per_matmul = (
            math.ceil(config.shapes.m / KERNEL_TILE_SIZE_M) *
            math.ceil(config.shapes.n / KERNEL_TILE_SIZE_N)
        )

        # Metrics
        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_completed_count: int = 0
        self._last_launch_count: int = 0
        self._launch_count: int = 0

    def initialize(self) -> None:
        """Connect to gateway, allocate B."""
        logger.info("Initializing direct miner...")

        # Same init the vLLM plugin uses; opens UDS to gateway
        init_async_manager()
        init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

        # Allow connection to settle
        time.sleep(1.0)

        # Verify gateway is responsive
        try:
            get_async_manager().get_mining_job()
            logger.info("Connected to gateway, mining template available")
        except Exception as e:
            logger.error(f"Failed to get mining job from gateway: {e}")
            logger.error(
                f"Is pearl-gateway running on {self.config.gateway_socket_path}?"
            )
            raise

        # Generate session B (forever-cacheable for this session)
        b_size_mb = self.config.shapes.n * self.config.shapes.k / 1024**2
        logger.info(
            f"Generating fixed B: shape=({self.config.shapes.n}, "
            f"{self.config.shapes.k}), size={b_size_mb:.1f}MB"
        )
        self.b_pool = FixedBPool(
            n=self.config.shapes.n,
            k=self.config.shapes.k,
            device="cuda",
            seed=self.config.seed,
        )
        torch.cuda.synchronize()

        logger.info(
            f"Direct miner initialized. "
            f"outer_tiles_per_matmul={self._outer_tiles_per_matmul} "
            f"max_in_flight={self.config.max_in_flight}"
        )

    def run(self) -> None:
        """Mining loop until interrupted."""
        if self.b_pool is None:
            raise RuntimeError("Must call initialize() before run()")

        logger.info(
            f"Starting mining loop: m={self.config.shapes.m} "
            f"n={self.config.shapes.n} k={self.config.shapes.k}"
        )

        self._start_time = time.time()
        self._last_log_time = self._start_time

        if self.config.seed is not None:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(self.config.seed + 1)
        else:
            generator = None

        try:
            while True:
                self._mine_one_iteration(generator)
        except KeyboardInterrupt:
            logger.info("Interrupted; draining in-flight work...")
            self.tracker.drain()
            self._log_final_stats()

    def _mine_one_iteration(self, generator: Optional[torch.Generator]) -> None:
        """One mining matmul, with backpressure."""

        # Backpressure: block until there's a free slot
        self.tracker.wait_for_slot()

        # Reap any completed work (updates completed_count, frees refs)
        self.tracker.reap_completed()

        # Fresh A every iteration — required for protocol correctness
        # because noise factors derive from commitment_hash_A which
        # depends on A's bytes
        A, A_scales = make_synthetic_a(
            m=self.config.shapes.m,
            k=self.config.shapes.k,
            device="cuda",
            generator=generator,
        )

        try:
            pearl_gemm_noisy(
                A,
                self.b_pool.B,
                scale_a=A_scales,
                scale_b=self.b_pool.B_scales,
                out_dtype=torch.bfloat16,
                layer=None,
                submit_block=True,
            )
        except Exception as e:
            logger.error(f"pearl_gemm_noisy failed: {e}", exc_info=True)
            time.sleep(0.5)
            return

        # Record event AFTER launch; hold A and A_scales refs alive
        # until GPU finishes (pearl_gemm_noisy internally keeps refs
        # for its callback, but we need our own event for OUR completion
        # tracking — separate concerns)
        self.tracker.record_launch(A, A_scales)
        self._launch_count += 1

        # Log based on COMPLETED count, not launched count
        completed_delta = (
            self.tracker.completed_count - self._last_completed_count
        )
        if completed_delta >= self.config.log_interval:
            self._log_progress()

    def _log_progress(self) -> None:
        """Report completion rate (the truth) and launch rate (for diagnosis)."""
        now = time.time()
        elapsed_total = now - self._start_time
        elapsed_interval = now - self._last_log_time

        completed = self.tracker.completed_count
        completed_delta = completed - self._last_completed_count
        launched_delta = self._launch_count - self._last_launch_count
        in_flight = self.tracker.current_in_flight()

        instant_completion_rate = (
            completed_delta / elapsed_interval if elapsed_interval > 0 else 0
        )
        cumulative_completion_rate = (
            completed / elapsed_total if elapsed_total > 0 else 0
        )
        instant_launch_rate = (
            launched_delta / elapsed_interval if elapsed_interval > 0 else 0
        )

        # Tile rate (estimate of mining lottery tickets/sec)
        instant_tile_rate = instant_completion_rate * self._outer_tiles_per_matmul
        cumulative_tile_rate = (
            cumulative_completion_rate * self._outer_tiles_per_matmul
        )

        logger.info(
            f"[DIRECT MINER] completed={completed} "
            f"({cumulative_completion_rate:.1f}/s avg, "
            f"{instant_completion_rate:.1f}/s now) "
            f"tiles=({cumulative_tile_rate:.0f}/s avg, "
            f"{instant_tile_rate:.0f}/s now) "
            f"launched={self._launch_count} "
            f"(launch_rate={instant_launch_rate:.1f}/s) "
            f"in_flight={in_flight}"
        )

        self._last_log_time = now
        self._last_completed_count = completed
        self._last_launch_count = self._launch_count

    def _log_final_stats(self) -> None:
        elapsed = time.time() - self._start_time
        completed = self.tracker.completed_count
        rate = completed / elapsed if elapsed > 0 else 0
        tile_rate = rate * self._outer_tiles_per_matmul
        logger.info(
            f"[DIRECT MINER] FINAL: completed={completed} "
            f"launched={self._launch_count} elapsed={elapsed:.1f}s "
            f"completion_rate={rate:.1f}/s "
            f"tile_rate={tile_rate:.0f}/s"
        )
