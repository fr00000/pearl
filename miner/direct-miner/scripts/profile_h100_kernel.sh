#!/usr/bin/env bash
# Profile the H100 direct-miner production kernel in a controlled short run.
#
# This intentionally refuses to run while a direct-miner process is active.
# Stop the miner first, run this script, then restart production mining.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/pearl}"
if [[ -f "${REPO_DIR}/env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_DIR}/env.sh"
fi

if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway is not running. Start it and retry."
    exit 1
fi

if pgrep -f "/\.venv/bin/direct-miner|python -m direct_miner" > /dev/null; then
    echo "ERROR: direct-miner already running. Stop it before profiling."
    exit 1
fi

SHAPE_M="${SHAPE_M:-8192}"
SHAPE_N="${SHAPE_N:-262144}"
SHAPE_K="${SHAPE_K:-32768}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"
GPU_INDEX="${GPU_INDEX:-0}"
DURATION_S="${DURATION_S:-75}"
KERNEL_TILE_M="${KERNEL_TILE_M:-128}"
KERNEL_TILE_N="${KERNEL_TILE_N:-256}"
KERNEL_TILE_K="${KERNEL_TILE_K:-128}"
KERNEL_STAGES="${KERNEL_STAGES:-3}"
KERNEL_CLUSTER_M="${KERNEL_CLUSTER_M:-2}"
KERNEL_CLUSTER_N="${KERNEL_CLUSTER_N:-1}"
KERNEL_MMA_REGISTERS="${KERNEL_MMA_REGISTERS:-160}"
KERNEL_SWIZZLE="${KERNEL_SWIZZLE:-8}"

OUTDIR="${OUTDIR:-/workspace/profiles/h100-kernel-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

COMMON_MINER_ARGS=(
    --m "$SHAPE_M"
    --n "$SHAPE_N"
    --k "$SHAPE_K"
    --max-in-flight "$MAX_IN_FLIGHT"
    --enable-b-cache
    --enable-headless-kernel
    --kernel-tile-m "$KERNEL_TILE_M"
    --kernel-tile-n "$KERNEL_TILE_N"
    --kernel-tile-k "$KERNEL_TILE_K"
    --kernel-stages "$KERNEL_STAGES"
    --kernel-cluster-m "$KERNEL_CLUSTER_M"
    --kernel-cluster-n "$KERNEL_CLUSTER_N"
    --kernel-mma-registers "$KERNEL_MMA_REGISTERS"
    --kernel-swizzle "$KERNEL_SWIZZLE"
    --log-interval 100
)

echo "Profile output: $OUTDIR"
echo "Shape: m=$SHAPE_M n=$SHAPE_N k=$SHAPE_K mif=$MAX_IN_FLIGHT"
echo "Kernel: ${KERNEL_TILE_M}x${KERNEL_TILE_N}x${KERNEL_TILE_K} stages=${KERNEL_STAGES} cluster=${KERNEL_CLUSTER_M}x${KERNEL_CLUSTER_N} regs=${KERNEL_MMA_REGISTERS} swizzle=${KERNEL_SWIZZLE}"
echo ""

if command -v ncu > /dev/null; then
    echo "Running Nsight Compute single-kernel probe..."
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    MINER_DEBUG=true \
    ncu --target-processes all \
        --kernel-name-base demangled \
        --kernel-name 'regex:.*hopper_mine_ws.*' \
        --launch-skip 5 \
        --launch-count 1 \
        --set full \
        --section LaunchStats \
        --section Occupancy \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        -o "$OUTDIR/ncu_hopper_mine" \
        -- uv run direct-miner "${COMMON_MINER_ARGS[@]}" \
            --phase-tag h100_ncu_profile \
        > "$OUTDIR/ncu.log" 2>&1
    ncu_exit=$?
    echo "ncu exit code: $ncu_exit"
    if [[ $ncu_exit -ne 0 ]]; then
        echo "ncu did not complete. This is expected on many RunPod hosts when GPU counters are restricted."
    fi
else
    echo "ncu not found; skipping Nsight Compute."
fi

echo ""
echo "Running Nsight Systems ${DURATION_S}s profile..."
if command -v nsys > /dev/null; then
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    MINER_DEBUG=true \
    timeout --signal=SIGINT --kill-after=30 "$DURATION_S" \
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        -o "$OUTDIR/nsys_h100_direct" \
        uv run direct-miner "${COMMON_MINER_ARGS[@]}" \
            --phase-tag h100_nsys_profile \
        > "$OUTDIR/nsys_direct_miner.log" 2>&1
    nsys_exit=$?
    echo "nsys profile exit code: $nsys_exit"
    if [[ -f "$OUTDIR/nsys_h100_direct.nsys-rep" ]]; then
        nsys stats --report cuda_gpu_kern_sum \
            "$OUTDIR/nsys_h100_direct.nsys-rep" \
            > "$OUTDIR/nsys_cuda_gpu_kern_sum.txt" 2>&1 || true
    fi
else
    echo "nsys not found; skipping Nsight Systems."
fi

echo ""
echo "Latest miner line:"
grep "DIRECT MINER" "$OUTDIR"/nsys_direct_miner.log 2>/dev/null | tail -1 || true
echo ""
echo "Top nsys kernels:"
sed -n '1,40p' "$OUTDIR/nsys_cuda_gpu_kern_sum.txt" 2>/dev/null || true
