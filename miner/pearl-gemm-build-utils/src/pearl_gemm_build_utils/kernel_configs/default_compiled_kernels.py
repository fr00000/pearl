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
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
