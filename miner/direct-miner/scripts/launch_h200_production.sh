#!/usr/bin/env bash
# Production launch on Hopper GPUs with the latest direct-miner winner config.
# Override SHAPE_*/MAX_IN_FLIGHT/KERNEL_* via env vars for a local sweep.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/pearl}"
source "${REPO_DIR}/env.sh"

# === Latest measured direct-mining defaults; see H100_SXM_VERIFY.md ===
SHAPE_M="${SHAPE_M:-8192}"
SHAPE_N="${SHAPE_N:-524032}"
SHAPE_K="${SHAPE_K:-8192}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"
KERNEL_TILE_M="${KERNEL_TILE_M:-64}"
KERNEL_TILE_N="${KERNEL_TILE_N:-256}"
KERNEL_TILE_K="${KERNEL_TILE_K:-128}"
KERNEL_STAGES="${KERNEL_STAGES:-3}"
KERNEL_CLUSTER_M="${KERNEL_CLUSTER_M:-2}"
KERNEL_CLUSTER_N="${KERNEL_CLUSTER_N:-1}"
# ===========================================

LOG_DIR="/workspace/production-logs"
PIDFILE_DIR="/tmp/direct-miner-pids"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$LOG_DIR" "$PIDFILE_DIR"

# Pre-flight
if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway not running"
    exit 1
fi
if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    echo "ERROR: direct-miner already running"
    exit 1
fi

GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
echo "Detected GPUs: $GPUS"
echo "Config: m=$SHAPE_M n=$SHAPE_N k=$SHAPE_K mif=$MAX_IN_FLIGHT"
echo "Kernel: ${KERNEL_TILE_M}x${KERNEL_TILE_N}x${KERNEL_TILE_K} stages=${KERNEL_STAGES} cluster=${KERNEL_CLUSTER_M}x${KERNEL_CLUSTER_N} headless"
echo ""

for gpu_idx in $GPUS; do
    log="$LOG_DIR/gpu${gpu_idx}-${TIMESTAMP}.log"
    pidfile="$PIDFILE_DIR/${gpu_idx}.pid"

    CUDA_VISIBLE_DEVICES="$gpu_idx" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    MINER_DEBUG=true \
    nohup uv run direct-miner \
        --m "$SHAPE_M" --n "$SHAPE_N" --k "$SHAPE_K" \
        --max-in-flight "$MAX_IN_FLIGHT" \
        --enable-b-cache \
        --enable-headless-kernel \
        --kernel-tile-m "$KERNEL_TILE_M" \
        --kernel-tile-n "$KERNEL_TILE_N" \
        --kernel-tile-k "$KERNEL_TILE_K" \
        --kernel-stages "$KERNEL_STAGES" \
        --kernel-cluster-m "$KERNEL_CLUSTER_M" \
        --kernel-cluster-n "$KERNEL_CLUSTER_N" \
        --log-interval 500 \
        --phase-tag "h200_prod_gpu${gpu_idx}" \
        > "$log" 2>&1 &
    pid=$!
    disown $pid 2>/dev/null || true
    echo "$pid" > "$pidfile"
    echo "GPU $gpu_idx: PID $pid -> $log"

    sleep 4

    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: GPU $gpu_idx miner died within 4s. Tail:"
        tail -30 "$log"
        exit 1
    fi
done

echo ""
echo "All $(echo $GPUS | wc -w) instances launched."
echo "Logs in: $LOG_DIR/"
echo "PID files in: $PIDFILE_DIR/"
echo ""
echo "Monitor with:"
echo "  for f in $LOG_DIR/gpu*-${TIMESTAMP}.log; do echo \"--- \$f ---\"; tail -3 \"\$f\" | grep DIRECT; done"
