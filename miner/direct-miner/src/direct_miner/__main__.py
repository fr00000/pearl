"""CLI entry point: `direct-miner` or `python -m direct_miner`."""

import argparse
import logging
import sys

from .config import MinerConfig, MiningShapes
from .mining_loop import DirectMiner


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pearl direct synthetic miner (Phase A baseline)"
    )
    parser.add_argument("--m", type=int, default=4096, help="A batch dim")
    parser.add_argument("--n", type=int, default=8192, help="B output dim")
    parser.add_argument("--k", type=int, default=8192, help="inner dim")
    parser.add_argument(
        "--max-in-flight", type=int, default=2,
        help="concurrent in-flight matmul cap (default 2)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="log every N completed matmuls"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--metrics-output", default="/workspace/direct-miner-metrics.jsonl",
        help="JSONL output path for per-matmul diagnostics"
    )
    parser.add_argument(
        "--phase-tag", default="phase_a",
        help="phase tag for filtering JSONL across runs"
    )
    parser.add_argument(
        "--enable-b-cache", action="store_true",
        help="enable Phase B B-side artifact caching"
    )
    parser.add_argument(
        "--enable-diagnostics", action="store_true",
        help="enable per-matmul diagnostic JSONL writes and live "
             "cache-stat logging. Off by default (production mode). "
             "Costs ~20%% throughput; use when validating new code "
             "paths or investigating cache/proof behaviour."
    )
    parser.add_argument(
        "--enable-headless-kernel", action="store_true",
        help="enable the mine-only GEMM kernel path. This skips "
             "denoising, output scaling, and C stores while preserving "
             "the PoW signal/proof path."
    )
    parser.add_argument(
        "--enable-kernel-hash-stats", action="store_true",
        help="enable benchmark-only kernel best-hash diagnostics. "
             "Adds atomics to the PoW path, so leave off for production."
    )
    parser.add_argument(
        "--kernel-tile-m", type=int, default=128,
        help="main mining kernel tile M dimension"
    )
    parser.add_argument(
        "--kernel-tile-n", type=int, default=256,
        help="main mining kernel tile N dimension"
    )
    parser.add_argument(
        "--kernel-tile-k", type=int, default=128,
        help="main mining kernel tile K dimension"
    )
    parser.add_argument(
        "--kernel-cluster-m", type=int, default=1,
        help="main mining kernel cluster M dimension"
    )
    parser.add_argument(
        "--kernel-cluster-n", type=int, default=1,
        help="main mining kernel cluster N dimension"
    )
    parser.add_argument(
        "--kernel-stages", type=int, default=None,
        help="main mining kernel pipeline stages"
    )
    parser.add_argument(
        "--kernel-mma-registers", type=int, default=None,
        help="explicit MMA warpgroup register allocation; default uses "
             "the kernel heuristic"
    )
    parser.add_argument(
        "--kernel-swizzle", type=int, default=None,
        help="override the mining kernel CTA scheduler swizzle; default "
             "uses the pearl-gemm L2 heuristic"
    )
    parser.add_argument(
        "--kernel-swizzle-m-major", action="store_true",
        help="schedule swizzle groups in M-major order instead of the "
             "default N-major order"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = MinerConfig(
        shapes=MiningShapes(m=args.m, n=args.n, k=args.k),
        max_in_flight=args.max_in_flight,
        seed=args.seed,
        log_interval=args.log_interval,
        metrics_output_path=args.metrics_output,
        phase_tag=args.phase_tag,
        enable_b_cache=args.enable_b_cache,
        enable_diagnostics=args.enable_diagnostics,
        enable_headless_kernel=args.enable_headless_kernel,
        enable_kernel_hash_stats=args.enable_kernel_hash_stats,
        kernel_tile_size_m=args.kernel_tile_m,
        kernel_tile_size_n=args.kernel_tile_n,
        kernel_tile_size_k=args.kernel_tile_k,
        kernel_cluster_size_m=args.kernel_cluster_m,
        kernel_cluster_size_n=args.kernel_cluster_n,
        kernel_pipeline_stages=args.kernel_stages,
        kernel_mma_registers=args.kernel_mma_registers,
        kernel_swizzle=args.kernel_swizzle,
        kernel_swizzle_n_maj=not args.kernel_swizzle_m_major,
    )

    miner = DirectMiner(config)
    miner.initialize()
    miner.run()


if __name__ == "__main__":
    main()
