"""
Kernel compilation grid schema.

Defines the complete set of kernels to compile for a given configuration.
Used by:
- default_compiled_kernels.py and release_kernels.py (direct instantiation)
- setup.py (loaded and filtered by R values)
"""

from pydantic import BaseModel, Field

from pearl_gemm_build_utils.kernel_configs.gemm import MatmulKernelConfig
from pearl_gemm_build_utils.kernel_configs.noising import NoisingAKernelConfig, NoisingBKernelConfig


class KernelCompilationGrid(BaseModel):
    """
    Complete set of kernels to compile.

    This schema represents all the kernel configurations that will be
    compiled into the library for a given build configuration.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    matmul_kernels: list[MatmulKernelConfig] = Field(
        default_factory=list, description="GEMM/matmul kernel configurations to compile"
    )

    mine_only_matmul_kernels: list[MatmulKernelConfig] = Field(
        default_factory=list,
        description=(
            "Headless mining-only matmul configurations to compile. These are "
            "visible only to mine_only dispatch and do not instantiate the full "
            "GEMM/denoising template path."
        ),
    )

    noising_a_kernels: list[NoisingAKernelConfig] = Field(
        default_factory=list, description="Noising A kernel configurations to compile"
    )

    noising_b_kernels: list[NoisingBKernelConfig] = Field(
        default_factory=list, description="Noising B kernel configurations to compile"
    )
