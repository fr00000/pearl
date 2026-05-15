"""Inspect proof row/column patterns for compiled mine-kernel variants.

This is a safety tool for tile-shape autotuning. Cluster/stage changes
preserve the proof pattern, but changing the tile shape can change the
thread-local rows/columns written to HostSignalHeader. If those differ
from the mining configuration used to derive the job key and target, a
fast kernel can produce invalid submissions.
"""

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from miner_base.settings import MinerSettings
from pearl_gemm import (
    HostSignalStatus,
    extract_indices,
    get_host_signal_header,
    get_host_signal_header_size,
    get_host_signal_sync_size,
    headless_mine,
    noise_gen,
)


def _normalize(indices: list[int]) -> list[int]:
    if not indices:
        return []
    base = min(indices)
    return [idx - base for idx in indices]


@dataclass(frozen=True)
class PatternResult:
    tile_m: int
    tile_n: int
    tile_k: int
    cluster_m: int
    cluster_n: int
    stages: int | None
    mma_registers: int | None
    iteration: int
    status: str
    tile_coord: tuple[int, int, int]
    thread_idx: tuple[int, int, int]
    num_registers_per_thread: int
    row_pattern: list[int]
    col_pattern: list[int]
    row_matches_default: bool
    col_matches_default: bool


def inspect_once(args, iteration: int) -> PatternResult:
    device = torch.device(args.device)
    r = args.rank

    # Use enough matrix extent to cover cluster padding and proof bounds
    # while keeping the inspector cheap.
    m = args.m or args.tile_m * max(1, args.cluster_m)
    n = args.n or args.tile_n * max(1, args.cluster_n)
    k = args.k

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + iteration)

    A = torch.randint(
        -64, 64, (m, k), dtype=torch.int8, device=device, generator=generator
    )
    B = torch.randint(
        -64, 64, (n, k), dtype=torch.int8, device=device, generator=generator
    )

    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EBR = torch.empty((n, r), dtype=torch.int8, device=device)
    EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)

    key_A = torch.randint(
        0, 256, (32,), dtype=torch.uint8, device=device, generator=generator
    )
    key_B = torch.randint(
        0, 256, (32,), dtype=torch.uint8, device=device, generator=generator
    )
    noise_gen(
        R=r,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        key_A=key_A,
        key_B=key_B,
    )

    AxEBL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EARxBpEB_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)
    ApEA = torch.empty((m, k), dtype=torch.int8, device=device)
    BpEB = torch.empty((n, k), dtype=torch.int8, device=device)
    host_signal_header = torch.zeros(
        (get_host_signal_header_size(),), dtype=torch.int8, pin_memory=True
    )
    host_signal_sync = torch.zeros(
        (get_host_signal_sync_size(),), dtype=torch.int8, device=device
    )
    pow_target = torch.full((8,), 0xFFFFFFFF, dtype=torch.uint32, device=device)

    headless_mine(
        A=A,
        B=B,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        EAR_R_major=EAR_R_major,
        EBL_R_major=EBL_R_major,
        EAR_K_major=EAR_K_major,
        EBL_K_major=EBL_K_major,
        AxEBL_fp16=AxEBL_fp16,
        EARxBpEB_fp16=EARxBpEB_fp16,
        ApEA=ApEA,
        BpEB=BpEB,
        host_signal_header_pinned=host_signal_header,
        host_signal_sync=host_signal_sync,
        pow_target=pow_target,
        pow_key=key_A.view(torch.uint32),
        tile_size_m=args.tile_m,
        tile_size_n=args.tile_n,
        tile_size_k=args.tile_k,
        cluster_size_m=args.cluster_m,
        cluster_size_n=args.cluster_n,
        pipeline_stages=args.stages,
        mma_registers=args.mma_registers,
        run_noising_A=True,
        run_noising_B=True,
    )
    torch.cuda.synchronize(device=device)

    header = get_host_signal_header(host_signal_header)
    settings = MinerSettings()

    row_pattern: list[int] = []
    col_pattern: list[int] = []
    if header.status == HostSignalStatus.kSignalTriggered:
        indices = extract_indices(header)
        row_pattern = _normalize(indices.A_row_indices)
        col_pattern = _normalize(indices.B_column_indices)

    return PatternResult(
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
        stages=args.stages,
        mma_registers=args.mma_registers,
        iteration=iteration,
        status=str(header.status).split(".")[-1],
        tile_coord=tuple(int(v) for v in header.tileCoord),
        thread_idx=tuple(int(v) for v in header.threadIdx),
        num_registers_per_thread=int(header.num_registers_per_thread),
        row_pattern=row_pattern,
        col_pattern=col_pattern,
        row_matches_default=(row_pattern == settings.rows_pattern),
        col_matches_default=(col_pattern == settings.cols_pattern),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect proof patterns for a headless mine-kernel variant"
    )
    parser.add_argument("--tile-m", type=int, required=True)
    parser.add_argument("--tile-n", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument("--mma-registers", type=int, default=None)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [inspect_once(args, i) for i in range(args.iterations)]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        print(
            f"iter={result.iteration} status={result.status} "
            f"tile={result.tile_m}x{result.tile_n}x{result.tile_k} "
            f"cluster={result.cluster_m}x{result.cluster_n} "
            f"stages={result.stages or 'default'} "
            f"mma_registers={result.mma_registers or 'default'} "
            f"tileCoord={result.tile_coord} threadIdx={result.thread_idx} "
            f"regs={result.num_registers_per_thread}"
        )
        print(
            f"  rows={result.row_pattern} "
            f"matches_default={result.row_matches_default}"
        )
        print(
            f"  cols={result.col_pattern} "
            f"matches_default={result.col_matches_default}"
        )

    all_match = all(
        result.row_matches_default and result.col_matches_default
        for result in results
    )
    print(f"PATTERN_COMPATIBLE={str(all_match).lower()}")


if __name__ == "__main__":
    main()
