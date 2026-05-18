from pathlib import Path

from pearl_gemm_build_utils.kernel_configs import (
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

cutlass_types_dict = {
    "fp16": "cutlass::half_t",
    "int32": "int",
}

torch_types_dict = {
    "fp16": "torch::kFloat16",
    "int32": "torch::kInt32",
}


def write_matmul_switch(filename: str | Path, kernel_configs: list[MatmulKernelConfig]):
    with open(filename, "w") as fh:
        fh.write(
            "/* Do not edit -- programmatically generated from build_utils/write_static_switches.py. */\n"
            "#define MATMUL_CONFIG_SWITCH(BM_VAR, BN_VAR, BK_VAR, R_VAR, "
            "STAGES_VAR, cM_VAR, cN_VAR, MMA_REGS_VAR, ...) \\\n"
        )
        fh.write("\tdo { \\\n")
        for config in kernel_configs:
            fh.write(
                f"\t\tMATMUL_CONFIG_OPTION(BM_VAR, BN_VAR, BK_VAR, R_VAR, "
                f"STAGES_VAR, cM_VAR, cN_VAR, MMA_REGS_VAR, "
                f"{config.tile_size_m}, {config.tile_size_n}, "
                f"{config.tile_size_k}, {config.R}, {config.pipeline_stages}, "
                f"{config.cM}, {config.cN}, {config.mma_registers}, "
                f"__VA_ARGS__); \\\n"
            )
        fh.write("\t} while(0)\n")


def write_mine_switch(filename: str | Path, kernel_configs: list[MatmulKernelConfig]):
    with open(filename, "w") as fh:
        fh.write(
            "/* Do not edit -- programmatically generated from build_utils/write_static_switches.py. */\n"
            "#define MINE_CONFIG_SWITCH(BM_VAR, BN_VAR, BK_VAR, R_VAR, "
            "STAGES_VAR, cM_VAR, cN_VAR, MMA_REGS_VAR, ...) \\\n"
        )
        fh.write("\tdo { \\\n")
        for config in kernel_configs:
            fh.write(
                f"\t\tMATMUL_CONFIG_OPTION(BM_VAR, BN_VAR, BK_VAR, R_VAR, "
                f"STAGES_VAR, cM_VAR, cN_VAR, MMA_REGS_VAR, "
                f"{config.tile_size_m}, {config.tile_size_n}, "
                f"{config.tile_size_k}, {config.R}, {config.pipeline_stages}, "
                f"{config.cM}, {config.cN}, {config.mma_registers}, "
                f"__VA_ARGS__); \\\n"
            )
        fh.write("\t} while(0)\n")


def write_noising_a_switch(filename: str | Path, kernel_configs: list[NoisingAKernelConfig]):
    with open(filename, "w") as fh:
        fh.write(
            "/* Do not edit -- programmatically generated from build_utils/write_static_switches.py. */\n"
            "#define NOISING_A_CONFIG_SWITCH(BM_VAR, BK_VAR, R_VAR, STAGES_VAR, AxEBL_TYPE_VAR, ...) \\\n"
        )
        fh.write("\tdo { \\\n")
        for config in kernel_configs:
            AxEBL_type_cutlass = cutlass_types_dict[config.AxEBL_type]
            AxEBL_type_torch = torch_types_dict[config.AxEBL_type]
            fh.write(
                f"\t\tNOISING_A_CONFIG_OPTION(BM_VAR, BK_VAR, R_VAR, STAGES_VAR, "
                f"AxEBL_TYPE_VAR, {config.tile_size_m}, "
                f"{config.tile_size_k}, {config.R}, {config.pipeline_stages}, "
                f"{AxEBL_type_torch}, {AxEBL_type_cutlass}, __VA_ARGS__); \\\n"
            )
        fh.write("\t} while(0)\n")


def write_noising_b_switch(filename: str | Path, kernel_configs: list[NoisingBKernelConfig]):
    with open(filename, "w") as fh:
        fh.write(
            "/* Do not edit -- programmatically generated from build_utils/write_static_switches.py. */\n"
            "#define NOISING_B_CONFIG_SWITCH(BN_VAR, BK_VAR, R_VAR, STAGES_VAR, "
            "EARxBpEB_TYPE_VAR, ...) \\\n"
        )
        fh.write("\tdo { \\\n")
        for config in kernel_configs:
            EARxBpEB_type_cutlass = cutlass_types_dict[config.EARxBpEB_type]
            EARxBpEB_type_torch = torch_types_dict[config.EARxBpEB_type]
            fh.write(
                f"\t\tNOISING_B_CONFIG_OPTION(BN_VAR, BK_VAR, R_VAR, STAGES_VAR, "
                f"EARxBpEB_TYPE_VAR, {config.tile_size_n}, {config.tile_size_k}, "
                f"{config.R}, {config.pipeline_stages}, "
                f"{EARxBpEB_type_torch}, {EARxBpEB_type_cutlass}, __VA_ARGS__); \\\n"
            )
        fh.write("\t} while(0)\n")
