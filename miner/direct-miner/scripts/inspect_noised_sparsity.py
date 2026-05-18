#!/usr/bin/env python3
"""Check whether Pearl noising preserves synthetic 2:4 sparsity.

The direct miner controls raw synthetic A and B, so a tempting moonshot is to
make them 2:4 sparse and use sparse Tensor Cores. Pearl mining, however, runs
the matmul over noised operands:

    ApEA = A + EAL * EAR
    BpEB = B + EBL * EBR

This tool measures whether those actual operands still satisfy contiguous-K
2:4 sparsity after the production noising path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from pearl_gemm import (
    get_host_signal_header_size,
    get_host_signal_sync_size,
    headless_mine,
    noise_gen,
)


@dataclass(frozen=True)
class SparsityStats:
    name: str
    shape: tuple[int, int]
    zero_fraction: float
    two_of_four_at_most_pct: float
    two_of_four_exact_pct: float
    mean_nonzeros_per_quad: float


def make_2of4_int8(rows: int, cols: int, *, device: torch.device, seed: int) -> torch.Tensor:
    if cols % 4 != 0:
        raise ValueError(f"cols must be divisible by 4, got {cols}")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tensor = torch.zeros((rows, cols), dtype=torch.int8, device=device)
    values = torch.randint(-64, 64, (rows, cols // 2), dtype=torch.int8, device=device, generator=generator)
    # Avoid accidental zeros in the nonzero lanes so the raw matrix is exactly 2:4.
    values = torch.where(values == 0, torch.ones_like(values), values)
    tensor[:, 0::4] = values[:, 0::2]
    tensor[:, 2::4] = values[:, 1::2]
    return tensor


def sparsity_stats(name: str, tensor: torch.Tensor) -> SparsityStats:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    if tensor.shape[1] % 4 != 0:
        raise ValueError(f"{name} K dimension must be divisible by 4")

    quads = tensor.reshape(tensor.shape[0], tensor.shape[1] // 4, 4)
    nonzeros = (quads != 0).sum(dim=2)
    total_groups = nonzeros.numel()
    at_most = (nonzeros <= 2).sum().item()
    exact = (nonzeros == 2).sum().item()
    zeros = (tensor == 0).sum().item()
    total = tensor.numel()
    return SparsityStats(
        name=name,
        shape=(int(tensor.shape[0]), int(tensor.shape[1])),
        zero_fraction=zeros / total,
        two_of_four_at_most_pct=100.0 * at_most / total_groups,
        two_of_four_exact_pct=100.0 * exact / total_groups,
        mean_nonzeros_per_quad=float(nonzeros.float().mean().item()),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    A = make_2of4_int8(args.m, args.k, device=device, seed=args.seed)
    B = make_2of4_int8(args.n, args.k, device=device, seed=args.seed + 1)

    r = args.rank
    EAL = torch.empty((args.m, r), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((args.m, r), dtype=torch.float16, device=device)
    EBR = torch.empty((args.n, r), dtype=torch.int8, device=device)
    EBR_fp16 = torch.empty((args.n, r), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((args.k, r), dtype=torch.int8, device=device)
    EBL_R_major = torch.empty((args.k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, args.k), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty((r, args.k), dtype=torch.int8, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2)
    key_A = torch.randint(0, 256, (32,), dtype=torch.uint8, device=device, generator=generator)
    key_B = torch.randint(0, 256, (32,), dtype=torch.uint8, device=device, generator=generator)

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

    AxEBL_fp16 = torch.empty((args.m, r), dtype=torch.float16, device=device)
    EARxBpEB_fp16 = torch.empty((args.n, r), dtype=torch.float16, device=device)
    ApEA = torch.empty((args.m, args.k), dtype=torch.int8, device=device)
    BpEB = torch.empty((args.n, args.k), dtype=torch.int8, device=device)
    host_signal_header = torch.zeros((get_host_signal_header_size(),), dtype=torch.int8, pin_memory=True)
    host_signal_sync = torch.zeros((get_host_signal_sync_size(),), dtype=torch.int8, device=device)
    pow_target = torch.zeros((8,), dtype=torch.uint32, device=device)

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

    stats = [
        sparsity_stats("raw_A", A),
        sparsity_stats("raw_B", B),
        sparsity_stats("noised_ApEA", ApEA),
        sparsity_stats("noised_BpEB", BpEB),
    ]
    return {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "rank": args.rank,
        "seed": args.seed,
        "stats": [asdict(item) for item in stats],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=256)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--cluster-m", type=int, default=1)
    parser.add_argument("--cluster-n", type=int, default=1)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--mma-registers", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
