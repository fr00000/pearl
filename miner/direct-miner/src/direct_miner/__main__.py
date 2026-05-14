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
        "--no-diagnostics", action="store_true",
        help="disable diagnostic JSONL writes for max throughput "
             "(use only for validated production runs; not for "
             "Phase C correctness verification)"
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
        disable_diagnostics=args.no_diagnostics,
    )

    miner = DirectMiner(config)
    miner.initialize()
    miner.run()


if __name__ == "__main__":
    main()
