"""Configuration for direct synthetic mining."""

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class MiningShapes:
    """Synthetic matmul dimensions.

    Must satisfy protocol constraints from sanity_checks.rs:
    - k must be multiple of 64, in [1024, 65536]
    - k must be >= 16*r and <= 4*r^2 (r is noise rank, typically 128)
    - m <= 2^24, n <= 2^24
    - (h+w)*dot_product_length <= 2^22 (4 MiB worker input cap)

    Defaults tuned for H100 80GB with compiled kernel grid
    (128x256x128 matmul tiles, R=64/128 noise rank).
    """
    m: int = 4096
    n: int = 8192
    k: int = 8192

    def validate(self) -> None:
        assert self.k % 64 == 0, f"k must be multiple of 64, got {self.k}"
        assert 1024 <= self.k <= (1 << 16), f"k must be in [1024, 65536], got {self.k}"
        assert self.m <= (1 << 24), f"m must be <= 2^24, got {self.m}"
        assert self.n <= (1 << 24), f"n must be <= 2^24, got {self.n}"
        assert self.m >= 128, f"m must be >= 128 for tiling, got {self.m}"
        assert self.n >= 256, f"n must be >= 256 for tiling, got {self.n}"


DEFAULT_SHAPES: Final = MiningShapes()


@dataclass(frozen=True)
class MinerConfig:
    """Top-level miner configuration.

    Phase A keeps things deliberately simple: single CUDA stream,
    low max_in_flight, no caching, no autotuning.
    """
    shapes: MiningShapes = field(default_factory=MiningShapes)

    # Bound concurrent in-flight CUDA work. Phase A uses a low default
    # to keep memory pressure low and behavior simple. Tune in Phase B
    # only after baseline works.
    max_in_flight: int = 2

    # Random seed for deterministic testing. None for production.
    seed: int | None = None

    # Log progress every N completed (not launched) matmuls
    log_interval: int = 50

    # Gateway UDS socket path
    gateway_socket_path: str = "/tmp/pearlgw.sock"

    # Path for structured per-matmul metrics
    metrics_output_path: str = "/workspace/direct-miner-metrics.jsonl"

    # Tag identifying which phase this run is testing (for filtering JSONL)
    phase_tag: str = "phase_a"

    # Phase B: enable B-side caching across iterations within a template
    enable_b_cache: bool = False

    # Diagnostics OFF by default — production is the default path.
    # Pass --enable-diagnostics to opt into per-matmul JSONL writes
    # and live cache-stat logging. Cost: ~20% throughput tax (JSON
    # serialise + flush every 100 records). Useful when validating
    # new code paths, debugging proof rejections, or investigating
    # cache behaviour.
    enable_diagnostics: bool = False

    # Headless mining kernel path. When enabled, the main GEMM still
    # computes the mining transcript and writes HostSignalHeader on a
    # win, but skips denoising, output scaling, and C stores.
    enable_headless_kernel: bool = False

    # Benchmark-only kernel hash observability. When enabled, the mining
    # kernel updates a tiny diagnostics buffer with per-call attempt count
    # and best observed hash. Disabled by default because it adds atomics
    # to the PoW check path.
    enable_kernel_hash_stats: bool = False

    # Main mining kernel launch parameters. These are separate from
    # MiningConfiguration: changing cluster/stage values should not
    # change proof semantics, while changing tile sizes must be treated
    # carefully because it can change the extracted row/column pattern.
    kernel_tile_size_m: int = 128
    kernel_tile_size_n: int = 256
    kernel_tile_size_k: int = 128
    kernel_cluster_size_m: int = 1
    kernel_cluster_size_n: int = 1
    kernel_pipeline_stages: int | None = None
    kernel_mma_registers: int | None = None


DEFAULT_CONFIG: Final = MinerConfig()
