"""Direct synthetic mining loop with diagnostics, B-cache, and Phase C
multi-stream A-side overlap."""

import logging
import time
from typing import Optional

import torch

from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from pearl_gemm import (
    get_host_signal_sync_size,
    get_pow_diagnostics_size,
    get_required_scratchpad_bytes,
)
from vllm_miner.mining_state import (
    init_async_manager,
    init_pinned_pool,
    get_async_manager,
)

from .a_slot_pool import ASlotPool, SlotAcquireTimeout
from .attempt_metrics import (
    chance_weighted_attempts_per_matmul,
    mma_consumer_threads_per_cta,
    normalized_attempts_per_matmul,
    normalized_attempt_scale,
    outer_tiles_per_matmul,
    rounded_common_dim,
)
from .b_cache import BSideCache
from .completion_tracker import CompletionTracker
from .config import MinerConfig
from .diagnostics import (
    DiagnosticsCollector,
    decode_pow_diagnostics,
    target_to_log2,
)
from .mining_call import UnsafeSlotReleaseError, pearl_gemm_noisy_phase_c
from .synthetic_data import FixedBPool


logger = logging.getLogger(__name__)

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
        if config.enable_diagnostics or config.enable_kernel_hash_stats:
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

        self._outer_tiles_per_matmul = outer_tiles_per_matmul(
            m=config.shapes.m,
            n=config.shapes.n,
            tile_m=config.kernel_tile_size_m,
            tile_n=config.kernel_tile_size_n,
        )
        self._attempt_scale = normalized_attempt_scale(
            tile_m=config.kernel_tile_size_m
        )
        self._mma_threads_per_cta = mma_consumer_threads_per_cta(
            tile_m=config.kernel_tile_size_m
        )
        self._normalized_attempts_per_matmul = (
            normalized_attempts_per_matmul(
                m=config.shapes.m,
                n=config.shapes.n,
                tile_m=config.kernel_tile_size_m,
                tile_n=config.kernel_tile_size_n,
            )
        )
        self._rounded_common_dim = 0
        self._chance_weighted_attempts_per_matmul = 0.0

        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_completed_count: int = 0
        self._last_launch_count: int = 0
        self._launch_count: int = 0

        self._matmul_config = None  # built in initialize()
        self._kernel_best_margin_log2: float | None = None
        self._kernel_hash_records: int = 0

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
        self._rounded_common_dim = rounded_common_dim(
            k=self.config.shapes.k,
            rank=settings.noise_rank,
        )
        self._chance_weighted_attempts_per_matmul = (
            chance_weighted_attempts_per_matmul(
                m=self.config.shapes.m,
                n=self.config.shapes.n,
                k=self.config.shapes.k,
                rank=settings.noise_rank,
                tile_m=self.config.kernel_tile_size_m,
                tile_n=self.config.kernel_tile_size_n,
            )
        )
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
        pow_diagnostics_size = (
            get_pow_diagnostics_size()
            if self.config.enable_kernel_hash_stats
            else None
        )
        self.a_pool = ASlotPool(
            num_slots=self.config.max_in_flight,
            m=self.config.shapes.m,
            n=self.config.shapes.n,
            k=self.config.shapes.k,
            noise_rank=settings.noise_rank,
            host_signal_sync_size=host_signal_sync_size,
            scratchpad_bytes=scratchpad_bytes,
            pow_diagnostics_size=pow_diagnostics_size,
            allocate_c=not self.config.enable_headless_kernel,
        )

        logger.info(
            f"Direct miner initialized. "
            f"outer_tiles_per_matmul={self._outer_tiles_per_matmul} "
            f"mma_threads_per_cta={self._mma_threads_per_cta} "
            f"attempt_scale={self._attempt_scale:.3f} "
            f"normalized_attempts_per_matmul="
            f"{self._normalized_attempts_per_matmul:.1f} "
            f"rounded_common_dim={self._rounded_common_dim} "
            f"chance_weighted_attempts_per_matmul="
            f"{self._chance_weighted_attempts_per_matmul:.1f} "
            f"max_in_flight={self.config.max_in_flight} "
            f"headless_kernel={self.config.enable_headless_kernel} "
            f"kernel_hash_stats={self.config.enable_kernel_hash_stats} "
            f"kernel={self.config.kernel_tile_size_m}x"
            f"{self.config.kernel_tile_size_n}x"
            f"{self.config.kernel_tile_size_k} "
            f"cluster={self.config.kernel_cluster_size_m}x"
            f"{self.config.kernel_cluster_size_n} "
            f"stages={self.config.kernel_pipeline_stages or 'default'} "
            f"mma_registers={self.config.kernel_mma_registers or 'default'}"
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
            self._shutdown_drain()
            self._log_final_stats()
            if self.diagnostics is not None:
                self.diagnostics.close()
        except SlotAcquireTimeout:
            logger.critical(
                "Mining loop exiting due to slot lifetime failure. "
                "Draining and shutting down — bootstrap re-run will "
                "bring the miner back up."
            )
            self._shutdown_drain()
            self._log_final_stats()
            if self.diagnostics is not None:
                self.diagnostics.close()
            # Re-raise so the process exits with a non-zero status that
            # operators (and monitor cron) can detect as a hard failure.
            raise
        except UnsafeSlotReleaseError:
            logger.critical(
                "Mining loop exiting due to unsafe slot cleanup failure. "
                "Refusing to continue because slot ownership cannot be "
                "proven safe."
            )
            # Deliberately NOT calling self._shutdown_drain() here:
            # the cleanup failure means CUDA sync is already unhealthy,
            # so draining (which itself relies on sync) could hang or
            # hit the same broken path. The priority is to exit so the
            # supervisor / bootstrap re-run can restart cleanly.
            self._log_final_stats()
            if self.diagnostics is not None:
                self.diagnostics.close()
            raise

    def _shutdown_drain(self) -> None:
        """Best-effort drain on shutdown. Order matters:
        1. Caller has already stopped feeding new launches.
        2. Drain CompletionTracker — wait for queued kernels to finish.
        3. Wait for async callbacks to release their slots.

        Step 3 may time out if a callback is the reason we're shutting
        down; we log and exit anyway.
        """
        logger.info("Draining CompletionTracker (waiting for in-flight kernels)...")
        try:
            self.tracker.drain()
        except Exception:
            logger.exception("Tracker drain failed")

        logger.info("Waiting for async callbacks to release slots...")
        all_released = self.a_pool.wait_all_released(timeout=30.0)
        if not all_released:
            logger.warning(
                "Not all slots released within 30s drain deadline; "
                "some callbacks may have been stuck. Exiting anyway."
            )

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
        kernel_stats = None
        pow_diagnostics = metadata.get("pow_diagnostics")
        if pow_diagnostics is not None:
            kernel_stats = decode_pow_diagnostics(pow_diagnostics)
            self._kernel_hash_records += 1
            target_log2 = metadata.get("target_log2")
            best_log2 = kernel_stats.get("kernel_best_hash_log2")
            if target_log2 is not None and best_log2 is not None:
                margin = best_log2 - target_log2
                if (
                    margin >= 0
                    and (
                        self._kernel_best_margin_log2 is None
                        or margin < self._kernel_best_margin_log2
                    )
                ):
                    self._kernel_best_margin_log2 = margin

        if self.diagnostics is None:
            return
        self.diagnostics.record(
            template_height=metadata.get("template_height"),
            template_hash_prefix=metadata.get("template_hash_prefix"),
            target_log2=metadata.get("target_log2"),
            best_observed_hash_log2=(
                kernel_stats.get("kernel_best_hash_log2")
                if kernel_stats is not None
                else None
            ),
            kernel_hash_attempts=(
                kernel_stats.get("kernel_hash_attempts")
                if kernel_stats is not None
                else None
            ),
            kernel_best_hash_hex=(
                kernel_stats.get("kernel_best_hash_hex")
                if kernel_stats is not None
                else None
            ),
            kernel_best_tile_coord=(
                kernel_stats.get("kernel_best_tile_coord")
                if kernel_stats is not None
                else None
            ),
            kernel_best_thread_idx=(
                kernel_stats.get("kernel_best_thread_idx")
                if kernel_stats is not None
                else None
            ),
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

        try:
            slot_idx, slot = self.a_pool.acquire()
        except SlotAcquireTimeout as e:
            # Stuck callback — fail closed. Continuing to mine would
            # risk reading slot.A while the next kernel writes it.
            # Surface via exception; run() will drain and exit.
            logger.critical(
                f"A-slot lifetime failure: {e}. Stopping miner — "
                "bootstrap script re-run will bring it back up cleanly."
            )
            raise

        callback_meta = dict(meta)
        callback_meta["b_cache_hit"] = None
        callback_meta["slot_idx"] = slot_idx
        callback_meta["pow_diagnostics"] = slot.pow_diagnostics

        # Bind slot_idx into a release callable. Default-arg captures
        # by value so each lambda owns its own slot_idx — closing over
        # the live `slot_idx` variable would race the next iteration.
        release_this_slot = lambda idx=slot_idx: self.a_pool.release(idx)

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
                on_callback_done=release_this_slot,
                mine_only=self.config.enable_headless_kernel,
                kernel_tile_size_m=self.config.kernel_tile_size_m,
                kernel_tile_size_n=self.config.kernel_tile_size_n,
                kernel_tile_size_k=self.config.kernel_tile_size_k,
                kernel_cluster_size_m=self.config.kernel_cluster_size_m,
                kernel_cluster_size_n=self.config.kernel_cluster_size_n,
                kernel_pipeline_stages=self.config.kernel_pipeline_stages,
                kernel_mma_registers=self.config.kernel_mma_registers,
                pow_diagnostics=slot.pow_diagnostics,
            )
        except UnsafeSlotReleaseError:
            # Cleanup couldn't prove the GPU is idle, so the slot was
            # deliberately not released. Continuing to mine would risk
            # proof corruption — surface as a fatal exit so a
            # supervisor / bootstrap re-run can restart cleanly.
            # Do NOT swallow into the generic handler below.
            logger.critical(
                "Fatal A-slot cleanup failure; exiting so supervisor "
                "can restart.",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(
                f"pearl_gemm_noisy_phase_c failed: {e}", exc_info=True
            )
            # pearl_gemm_noisy_phase_c's outer finally already released
            # the slot on its error path; do not double-release.
            time.sleep(0.5)
            return

        if self.b_cache is not None:
            callback_meta["b_cache_hit"] = b_cache_hit

        # Slot lifetime is now tracked by ASlotPool via the per-slot
        # Event (release fires from the callback's finally block). The
        # CUDA event below tracks GPU completion for throughput and
        # launch backpressure — both gates must clear before slot reuse.
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

        instant_outer_tile_rate = (
            instant_completion_rate * self._outer_tiles_per_matmul
        )
        cumulative_outer_tile_rate = (
            cumulative_completion_rate * self._outer_tiles_per_matmul
        )
        instant_attempt_rate = (
            instant_completion_rate * self._normalized_attempts_per_matmul
        )
        cumulative_attempt_rate = (
            cumulative_completion_rate * self._normalized_attempts_per_matmul
        )
        instant_chance_weighted_rate = (
            instant_completion_rate * self._chance_weighted_attempts_per_matmul
        )
        cumulative_chance_weighted_rate = (
            cumulative_completion_rate
            * self._chance_weighted_attempts_per_matmul
        )

        logger.info(
            f"[DIRECT MINER] completed={completed} "
            f"({cumulative_completion_rate:.1f}/s avg, "
            f"{instant_completion_rate:.1f}/s now) "
            f"outer_tiles=({cumulative_outer_tile_rate:.0f}/s avg, "
            f"{instant_outer_tile_rate:.0f}/s now raw) "
            f"attempts=({cumulative_attempt_rate:.0f}/s avg, "
            f"{instant_attempt_rate:.0f}/s now 128eq) "
            f"chance_weighted=({cumulative_chance_weighted_rate:.0f}/s avg, "
            f"{instant_chance_weighted_rate:.0f}/s now) "
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

        if self._kernel_best_margin_log2 is not None:
            logger.info(
                f"[KERNEL HASH] best_margin_log2="
                f"{self._kernel_best_margin_log2:.2f} "
                f"records={self._kernel_hash_records}"
            )

        self._last_log_time = now
        self._last_completed_count = completed
        self._last_launch_count = self._launch_count

    def _log_final_stats(self) -> None:
        elapsed = time.time() - self._start_time
        completed = self.tracker.completed_count
        rate = completed / elapsed if elapsed > 0 else 0
        raw_outer_tile_rate = rate * self._outer_tiles_per_matmul
        normalized_attempt_rate = rate * self._normalized_attempts_per_matmul
        chance_weighted_rate = (
            rate * self._chance_weighted_attempts_per_matmul
        )
        logger.info(
            f"[DIRECT MINER] FINAL: completed={completed} "
            f"launched={self._launch_count} elapsed={elapsed:.1f}s "
            f"completion_rate={rate:.1f}/s "
            f"raw_outer_tile_rate={raw_outer_tile_rate:.0f}/s "
            f"normalized_attempt_rate={normalized_attempt_rate:.0f}/s "
            f"chance_weighted_rate={chance_weighted_rate:.0f}/s"
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
