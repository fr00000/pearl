"""Direct synthetic mining loop with diagnostics, B-cache, and Phase C
multi-stream A-side overlap."""

import logging
import math
import time
from typing import Optional

import torch

from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from pearl_gemm import (
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
)
from vllm_miner.mining_state import (
    init_async_manager,
    init_pinned_pool,
    get_async_manager,
)

from .a_slot_pool import ASlotPool
from .b_cache import BSideCache
from .completion_tracker import CompletionTracker
from .config import MinerConfig
from .diagnostics import DiagnosticsCollector, target_to_log2
from .mining_call import pearl_gemm_noisy_phase_c
from .synthetic_data import FixedBPool


logger = logging.getLogger(__name__)

# Outer tile sizes from the compiled kernel (kernel_traits.hpp).
KERNEL_TILE_SIZE_M = 128
KERNEL_TILE_SIZE_N = 256


class DirectMiner:
    """Direct synthetic miner.

    Phase A: single stream, no caching.
    Phase B: single stream + B-side caching (config.enable_b_cache).
    Phase C: two streams (main + prep), B-cache, A-slot pool. The
    A-side prep on stream_prep overlaps with the previous iteration's
    main kernel on stream_main.
    """

    def __init__(self, config: MinerConfig):
        config.shapes.validate()
        self.config = config
        self.b_pool: Optional[FixedBPool] = None

        # Diagnostics OFF by default (production); opt in with
        # --enable-diagnostics during verification or investigation.
        self.diagnostics: Optional[DiagnosticsCollector] = None
        if config.enable_diagnostics:
            self.diagnostics = DiagnosticsCollector(
                output_path=config.metrics_output_path,
                flush_every_n=100,
                phase_tag=config.phase_tag,
            )
            on_complete = self._on_matmul_complete
        else:
            on_complete = None

        self.tracker = CompletionTracker(
            max_in_flight=config.max_in_flight,
            on_complete=on_complete,
        )

        self.b_cache: BSideCache | None = None
        if config.enable_b_cache:
            self.b_cache = BSideCache()
            logger.info("B-side cache ENABLED (Phase B)")

        # Phase C streams + slot pool — initialised in initialize()
        # once shape/settings are known.
        self.stream_main: Optional[torch.cuda.Stream] = None
        self.stream_prep: Optional[torch.cuda.Stream] = None
        self.a_pool: Optional[ASlotPool] = None

        self._outer_tiles_per_matmul = (
            math.ceil(config.shapes.m / KERNEL_TILE_SIZE_M) *
            math.ceil(config.shapes.n / KERNEL_TILE_SIZE_N)
        )

        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_completed_count: int = 0
        self._last_launch_count: int = 0
        self._launch_count: int = 0

        self._matmul_config = None  # built in initialize()

        # Per-template diagnostic metadata cache.
        self._meta_cache_header_bytes: bytes | None = None
        self._meta_cache_hash_key: bytes | None = None
        self._meta_cache_target_log2: float | None = None

    def initialize(self) -> None:
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

        settings = get_async_manager()._conf
        self._matmul_config = GPUMatmulConfigFactory.create(
            k=self.config.shapes.k, noise_rank=settings.noise_rank
        )

        # Phase C: two non-default streams.
        # Both non-default to avoid legacy-default-stream implicit
        # synchronization with each other.
        self.stream_main = torch.cuda.Stream()
        self.stream_prep = torch.cuda.Stream()
        logger.info(
            f"Phase C streams: main={self.stream_main} "
            f"prep={self.stream_prep}"
        )

        scratchpad_bytes = get_required_scratchpad_bytes(
            max(self.config.shapes.m * self.config.shapes.k,
                self.config.shapes.n * self.config.shapes.k)
        )
        host_signal_sync_size = get_host_signal_sync_size()
        self.a_pool = ASlotPool(
            num_slots=self.config.max_in_flight,
            m=self.config.shapes.m,
            n=self.config.shapes.n,
            k=self.config.shapes.k,
            noise_rank=settings.noise_rank,
            host_signal_sync_size=host_signal_sync_size,
            scratchpad_bytes=scratchpad_bytes,
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
            if self.diagnostics is not None:
                self.diagnostics.close()

    def _capture_template_metadata(self) -> dict:
        mining_job = get_async_manager().get_mining_job()
        mining_config = self._matmul_config.mining_config
        header_bytes = mining_job.incomplete_header_bytes

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

        return {
            "template_height": None,
            "template_hash_prefix": self._meta_cache_hash_key[:8].hex(),
            "target_log2": self._meta_cache_target_log2,
        }

    def _on_matmul_complete(self, metadata: dict) -> None:
        if self.diagnostics is None:
            return
        self.diagnostics.record(
            template_height=metadata.get("template_height"),
            template_hash_prefix=metadata.get("template_hash_prefix"),
            target_log2=metadata.get("target_log2"),
            best_observed_hash_log2=None,
            block_found=None,
            b_cache_hit=metadata.get("b_cache_hit"),
        )

    def _mine_one_iteration(self, generator: Optional[torch.Generator]) -> None:
        self.tracker.wait_for_slot()
        self.tracker.reap_completed()

        meta = self._capture_template_metadata()

        if self._launch_count > 0 and self._launch_count % 500 == 0:
            target_log2 = meta.get("target_log2")
            target_str = f"{target_log2:.2f}" if target_log2 is not None else "n/a"
            logger.info(
                f"[TEMPLATE CHECK] launch={self._launch_count} "
                f"template_height={meta.get('template_height')} "
                f"template_prefix={meta.get('template_hash_prefix')} "
                f"target_log2={target_str}"
            )

        slot_idx, slot = self.a_pool.acquire()

        callback_meta = dict(meta)
        callback_meta["b_cache_hit"] = None
        callback_meta["slot_idx"] = slot_idx

        try:
            _C, b_cache_hit, completion_event = pearl_gemm_noisy_phase_c(
                slot=slot,
                B=self.b_pool.B,
                B_scales=self.b_pool.B_scales,
                matmul_config=self._matmul_config,
                settings=get_async_manager()._conf,
                b_cache=self.b_cache,
                stream_main=self.stream_main,
                stream_prep=self.stream_prep,
                a_generator=generator,
                submit_block=True,
            )
        except Exception as e:
            logger.error(
                f"pearl_gemm_noisy_phase_c failed: {e}", exc_info=True
            )
            time.sleep(0.5)
            return

        if self.b_cache is not None:
            callback_meta["b_cache_hit"] = b_cache_hit

        # The slot's tensors are owned by ASlotPool and protected from
        # reuse by max_in_flight; we pass no per-iteration tensor refs
        # to the tracker. Use the stream_main-recorded completion_event
        # so the on_complete callback fires when the kernel actually
        # finishes (not when this function returned on the host).
        self.tracker.record_launch(callback_meta, event=completion_event)
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

        if self.diagnostics is not None:
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
            if self.diagnostics is not None:
                hits, misses = self.diagnostics.cache_stats()
            else:
                stats = self.b_cache.stats()
                hits, misses = stats["hits"], stats["misses"]
            invalidations = getattr(self.b_cache, "invalidations", 0)
            logger.info(
                f"[BCACHE FINAL] hits={hits} misses={misses} "
                f"invalidations={invalidations}"
            )
