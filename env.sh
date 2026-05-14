# Source this file from any shell that needs the Pearl miner environment.
# Sets LD_LIBRARY_PATH so the venv's bundled CUDA 12.9 libs are resolved
# before the system libs (system has CUDA 12.8; cusparse expects 12.9).

PEARL_ROOT="${PEARL_ROOT:-/root/pearl}"
PEARL_VENV="${PEARL_ROOT}/.venv"

if [ -d "${PEARL_VENV}/lib/python3.12/site-packages/nvidia" ]; then
    NVIDIA_LIBS=$(find "${PEARL_VENV}/lib/python3.12/site-packages/nvidia" -name "lib" -type d | tr '\n' ':')
    TORCH_LIB="${PEARL_VENV}/lib/python3.12/site-packages/torch/lib"
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

export PATH="${PEARL_ROOT}/bin:/usr/local/go/bin:${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"
export GOTOOLCHAIN=auto
