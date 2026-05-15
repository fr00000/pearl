"""
GEMM kernel configuration schemas.

Defines configurations for:
- MatmulKernelConfig: Build-time configuration for codegen
"""

from pydantic import BaseModel, Field


class MatmulKernelConfig(BaseModel):
    """
    Build-time configuration for GEMM kernel codegen.

    Used by setup.py and pearl-gemm-build-utils to generate kernel instantiations.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Tile dimensions
    tile_size_m: int = Field(ge=16, le=512)
    tile_size_n: int = Field(ge=16, le=512)
    tile_size_k: int = Field(ge=32, le=512)

    # Low-rank dimension
    R: int = Field(ge=16, le=256, description="Low-rank dimension for noise matrices")

    # Pipeline configuration
    pipeline_stages: int = Field(ge=1, le=8, description="Number of pipeline stages")

    # Cluster configuration
    cM: int = Field(default=1, ge=1, le=2, description="Cluster size in M dimension")
    cN: int = Field(default=1, ge=1, le=2, description="Cluster size in N dimension")

    # Warpgroup register allocation. 0 preserves the kernel's current default
    # heuristic; non-zero values instantiate explicit warpgroup_reg_alloc<N>
    # variants for tuning.
    mma_registers: int = Field(
        default=0,
        ge=0,
        le=256,
        description="MMA warpgroup register allocation; 0 uses kernel default",
    )
