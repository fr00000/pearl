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
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = MinerConfig(
        shapes=MiningShapes(m=args.m, n=args.n, k=args.k),
        max_in_flight=args.max_in_flight,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    miner = DirectMiner(config)
    miner.initialize()
    miner.run()


if __name__ == "__main__":
    main()
