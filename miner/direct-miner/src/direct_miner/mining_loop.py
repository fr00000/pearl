"""Direct synthetic mining loop with diagnostics + optional Phase B B-cache."""

import logging
import math
import time
from typing import Optional

import torch

from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from vllm_miner.gemm_operators import pearl_gemm_noisy
from vllm_miner.mining_state import (
    init_async_manager,
    init_pinned_pool,
    get_async_manager,
)

from .config import MinerConfig
from .completion_tracker import CompletionTracker
from .diagnostics import DiagnosticsCollector, target_to_log2
from .synthetic_data import FixedBPool, make_synthetic_a


logger = logging.getLogger(__name__)

# Outer tile sizes from the compiled kernel (kernel_traits.hpp).
KERNEL_TILE_SIZE_M = 128
KERNEL_TILE_SIZE_N = 256


class DirectMiner:
    """Direct synthetic miner with diagnostics.

    Connects to existing pearl-gateway via UDS. Generates fresh A per
    iteration, reuses static B for the session, calls pearl_gemm_noisy,
    tracks completion via CUDA events, emits per-matmul diagnostic
    JSONL via the tracker callback.

    Phase A: no caching of B-derived artifacts.
    Phase B: optional B-side artifact caching (config.enable_b_cache).
    """

    def __init__(self, config: MinerConfig):
        config.shapes.validate()
        self.config = config
        self.b_pool: Optional[FixedBPool] = None

        self.diagnostics = DiagnosticsCollector(
            output_path=config.metrics_output_path,
            flush_every_n=100,
            phase_tag=config.phase_tag,
        )

        self.tracker = CompletionTracker(
            max_in_flight=config.max_in_flight,
            on_complete=self._on_matmul_complete,
        )

        # Phase B B-side cache (None unless enabled). Implementation lands
        # in Part 2; for now, the field is reserved so config.enable_b_cache
        # produces a clear "not yet implemented" message.
        self.b_cache = None
        if config.enable_b_cache:
            try:
                from .b_cache import BSideCache
                self.b_cache = BSideCache()
                logger.info("B-side cache ENABLED (Phase B)")
            except ImportError:
                logger.warning(
                    "enable_b_cache=True but b_cache module not available; "
                    "running without cache."
                )

        # Tile-rate accounting
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

        # Cached matmul_config for the configured k (it depends only on k
        # and noise_rank, both fixed for a session).
        self._matmul_config = None  # built lazily in initialize()

        # Per-template diagnostic metadata cache. Avoids paying a blake3
        # hash + adjust_target on every iteration. Invalidated when the
        # gateway returns a new incomplete_header_bytes (= template change).
        self._meta_cache_header_bytes: bytes | None = None
        self._meta_cache_hash_key: bytes | None = None
        self._meta_cache_target_log2: float | None = None

    def initialize(self) -> None:
        """Connect to gateway, allocate B."""
        logger.info("Initializing direct miner...")

        init_async_manager()
        init_pinned_pool(get_async_manager()._conf.pinned_pool_size)

        time.sleep(1.0)

        try:
            get_async_manager().get_mining_job()
            logger.info("Connected to gateway, mining template available")
        except Exception as e:
            logger.error(f"Failed to get mining job from gateway: {e}")
            logger.error(
                f"Is pearl-gateway running on {self.config.gateway_socket_path}?"
            )
            raise

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

        # Build matmul_config once — k and noise_rank are session-stable.
        noise_rank = get_async_manager()._conf.noise_rank
        self._matmul_config = GPUMatmulConfigFactory.create(
            k=self.config.shapes.k, noise_rank=noise_rank
        )

        logger.info(
            f"Direct miner initialized. "
            f"outer_tiles_per_matmul={self._outer_tiles_per_matmul} "
            f"max_in_flight={self.config.max_in_flight}"
        )

    def run(self) -> None:
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
            self.diagnostics.close()

    def _capture_template_metadata(self) -> dict:
        """Read current template state. Called BEFORE the launch so the
        recorded template matches what the kernel actually consumed.

        Caches hash_key and target_log2 by incomplete_header_bytes —
        these change only on template rotation (every ~minutes), so
        per-iteration cost is identity comparison + dict build.
        """
        mining_job = get_async_manager().get_mining_job()
        mining_config = self._matmul_config.mining_config
        header_bytes = mining_job.incomplete_header_bytes

        # The kernel's per-call MiningJob fetch and our cache invalidation
        # use the same source; a header byte change means a new template.
        # Identity check first (same object across calls is common); fall
        # back to value compare to be safe across deserialisation.
        if (
            self._meta_cache_header_bytes is not header_bytes
            and self._meta_cache_header_bytes != header_bytes
        ):
            self._meta_cache_hash_key = CommitmentHasher.get_key(
                header_bytes, mining_config
            )
            target = mining_job.adjust_target(mining_config=mining_config)
            self._meta_cache_target_log2 = target_to_log2(int(target))
            self._meta_cache_header_bytes = header_bytes

        # MiningJob carries no height attribute (see dataclasses.py:142):
        # only incomplete_header_bytes + target. template_hash_prefix
        # gives us the per-template uniqueness we actually need.
        return {
            "template_height": None,
            "template_hash_prefix": self._meta_cache_hash_key[:8].hex(),
            "hash_key": self._meta_cache_hash_key,
            "target_log2": self._meta_cache_target_log2,
        }

    def _on_matmul_complete(self, metadata: dict) -> None:
        """Fired by CompletionTracker after the CUDA event completes.

        block_found is null in Phase A: the per-call host_signal_header
        pinned buffer is acquired and consumed inside pearl_gemm_noisy /
        StatusCheckCallback; we don't have a reference to read its
        triggered status from here without modifying gemm_operators.
        Wins surface via wallet balance + gateway logs.
        """
        self.diagnostics.record(
            template_height=metadata.get("template_height"),
            template_hash_prefix=metadata.get("template_hash_prefix"),
            target_log2=metadata.get("target_log2"),
            best_observed_hash_log2=None,  # Case B: not exposed by kernel
            block_found=None,
            b_cache_hit=metadata.get("b_cache_hit"),
        )

    def _mine_one_iteration(self, generator: Optional[torch.Generator]) -> None:
        self.tracker.wait_for_slot()
        self.tracker.reap_completed()

        # Capture template state for THIS launch (matches what the kernel
        # reads inside pearl_gemm_noisy when it calls get_mining_job()).
        meta = self._capture_template_metadata()

        # Periodic template-freshness log
        if self._launch_count > 0 and self._launch_count % 500 == 0:
            target_log2 = meta.get("target_log2")
            target_str = f"{target_log2:.2f}" if target_log2 is not None else "n/a"
            logger.info(
                f"[TEMPLATE CHECK] launch={self._launch_count} "
                f"template_height={meta.get('template_height')} "
                f"template_prefix={meta.get('template_hash_prefix')} "
                f"target_log2={target_str}"
            )

        A, A_scales = make_synthetic_a(
            m=self.config.shapes.m,
            k=self.config.shapes.k,
            device="cuda",
            generator=generator,
        )

        # Phase B b_cache_hit tracking. None when cache is disabled (Phase A).
        # Set to True/False per call when cache is enabled (Phase B).
        meta["b_cache_hit"] = None
        if self.b_cache is not None:
            meta["b_cache_hit"] = self.b_cache.get(meta["hash_key"]) is not None
            # Population/use of the cached value is Part 2's wiring; this
            # branch is the diagnostic-only stub that goes live there.

        # Drop the full hash_key before passing through the tracker; the
        # callback only needs the public-facing prefix and other primitives.
        callback_meta = {k: v for k, v in meta.items() if k != "hash_key"}

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

        self.tracker.record_launch(callback_meta, A, A_scales)
        self._launch_count += 1

        completed_delta = (
            self.tracker.completed_count - self._last_completed_count
        )
        if completed_delta >= self.config.log_interval:
            self._log_progress()

    def _log_progress(self) -> None:
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

        # Diagnostics summary (best margin is null in Case B but we log
        # whatever we have)
        best_margin = self.diagnostics.best_margin_log2()
        if best_margin is not None:
            logger.info(
                f"[DIAGNOSTICS] best_margin_log2={best_margin:.2f} "
                f"(0 = win threshold; smaller positive = closer)"
            )

        hits, misses = self.diagnostics.cache_stats()
        if hits + misses > 0:
            hit_rate = hits / (hits + misses) * 100
            logger.info(
                f"[BCACHE] hit_rate={hit_rate:.1f}% "
                f"({hits} hits, {misses} misses)"
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
        if self.b_cache is not None:
            hits, misses = self.diagnostics.cache_stats()
            invalidations = getattr(self.b_cache, "invalidations", 0)
            logger.info(
                f"[BCACHE FINAL] hits={hits} misses={misses} "
                f"invalidations={invalidations}"
            )
