"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.
"""

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Build matmul kernels.
_matmul_kernels = []
_matmul_kernel_keys = set()
_mine_only_matmul_kernels = []
_mine_only_matmul_kernel_keys = set()


def _add_matmul_kernel(
    *,
    tile_size_m: int = 128,
    tile_size_n: int = 256,
    tile_size_k: int = 128,
    R: int = 128,
    pipeline_stages: int = 3,
    cM: int = 1,
    cN: int = 1,
    mma_registers: int = 0,
) -> None:
    key = (
        tile_size_m,
        tile_size_n,
        tile_size_k,
        R,
        pipeline_stages,
        cM,
        cN,
        mma_registers,
    )
    if key in _matmul_kernel_keys:
        return
    _matmul_kernel_keys.add(key)
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=tile_size_m,
            tile_size_n=tile_size_n,
            tile_size_k=tile_size_k,
            R=R,
            pipeline_stages=pipeline_stages,
            cM=cM,
            cN=cN,
            mma_registers=mma_registers,
        )
    )


def _add_mine_only_matmul_kernel(
    *,
    tile_size_m: int = 128,
    tile_size_n: int = 256,
    tile_size_k: int = 128,
    R: int = 128,
    pipeline_stages: int = 3,
    cM: int = 1,
    cN: int = 1,
    mma_registers: int = 0,
) -> None:
    key = (
        tile_size_m,
        tile_size_n,
        tile_size_k,
        R,
        pipeline_stages,
        cM,
        cN,
        mma_registers,
    )
    if key in _matmul_kernel_keys or key in _mine_only_matmul_kernel_keys:
        return
    _mine_only_matmul_kernel_keys.add(key)
    _mine_only_matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=tile_size_m,
            tile_size_n=tile_size_n,
            tile_size_k=tile_size_k,
            R=R,
            pipeline_stages=pipeline_stages,
            cM=cM,
            cN=cN,
            mma_registers=mma_registers,
        )
    )


# 128x256x128, R=64/128, stage=3
for R in [64, 128]:
    _add_matmul_kernel(R=R)

# Focused headless-mining autotune variants. These preserve the
# 128x256x128 tile shape so the proof row/column pattern remains
# compatible with the default mining configuration; only runtime
# scheduling/pipeline choices change.
for pipeline_stages, cM, cN in [
    (4, 1, 1),
    (5, 1, 1),
    (3, 2, 1),
    (3, 1, 2),
    (3, 2, 2),
    (4, 2, 1),
    (4, 1, 2),
    (4, 2, 2),
]:
    _add_matmul_kernel(
        R=128,
        pipeline_stages=pipeline_stages,
        cM=cM,
        cN=cN,
    )

# First tile-shape probe grid. These are not automatically safe for
# production mining; validate the extracted proof pattern with
# direct-miner-inspect-pattern before benchmarking/submitting.
for tile_size_m, tile_size_n, tile_size_k, pipeline_stages, cM, cN in [
    (64, 256, 128, 3, 1, 1),
    (64, 256, 128, 3, 2, 1),
    (128, 256, 64, 3, 1, 1),
    (128, 256, 64, 3, 2, 1),
    (128, 256, 64, 4, 2, 1),
]:
    _add_matmul_kernel(
        tile_size_m=tile_size_m,
        tile_size_n=tile_size_n,
        tile_size_k=tile_size_k,
        R=128,
        pipeline_stages=pipeline_stages,
        cM=cM,
        cN=cN,
    )

# Register-allocation sweep for the current production winner and the
# strongest 64x256x128 probe family. mma_registers=0 kernels above keep the
# existing default heuristic, so we only add explicit non-default values.
for mma_registers in [160, 192, 224]:
    _add_matmul_kernel(
        tile_size_m=128,
        tile_size_n=256,
        tile_size_k=128,
        R=128,
        pipeline_stages=3,
        cM=2,
        cN=1,
        mma_registers=mma_registers,
    )

for cM, cN in [(1, 1), (2, 1)]:
    for mma_registers in [128, 160, 192, 224]:
        _add_matmul_kernel(
            tile_size_m=64,
            tile_size_n=256,
            tile_size_k=128,
            R=128,
            pipeline_stages=3,
            cM=cM,
            cN=cN,
            mma_registers=mma_registers,
        )

# Second H100 mining grid. These variants preserve the default 256-column proof
# pattern while changing the CTA work shape enough to test genuine kernel-side
# ceilings at the current production shape (8192x262144x32768).
#
# - 128x256x128 stages=2 checks whether the production kernel is over-buffered
#   now that the main path is mine-only and writes no C tile.
# - 128x256x256 stages=2 halves the number of mainloop K tiles and transcript
#   writeback points at the same problem K, trading more shared memory per stage
#   for less loop/control overhead.
# - 256x256x128 was considered but is intentionally omitted: the H100 pod
#   build showed the current transcript path needs ~154 registers/thread for
#   that shape, while a 640-thread CTA is capped near 96 by the SM register
#   budget. That path needs a deeper live-state rewrite before it is feasible.
for pipeline_stages, cM, cN, mma_registers in [
    (2, 1, 1, 0),
    (2, 2, 1, 160),
    (2, 2, 1, 192),
]:
    _add_matmul_kernel(
        tile_size_m=128,
        tile_size_n=256,
        tile_size_k=128,
        R=128,
        pipeline_stages=pipeline_stages,
        cM=cM,
        cN=cN,
        mma_registers=mma_registers,
    )

for cM, cN, mma_registers in [
    (1, 1, 0),
    (2, 1, 160),
    (2, 1, 192),
]:
    _add_matmul_kernel(
        tile_size_m=128,
        tile_size_n=256,
        tile_size_k=256,
        R=128,
        pipeline_stages=2,
        cM=cM,
        cN=cN,
        mma_registers=mma_registers,
    )

# Noising A: 64x64, fp16/int32
_noising_a_kernels = [
    NoisingAKernelConfig(
        tile_size_m=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        AxEBL_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

# Noising B: 64x64, fp16/int32
_noising_b_kernels = [
    NoisingBKernelConfig(
        tile_size_n=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        EARxBpEB_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    mine_only_matmul_kernels=_mine_only_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
