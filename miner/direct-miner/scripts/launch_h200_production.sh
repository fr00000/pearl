#!/usr/bin/env bash
# Production launch on 4× H200 with sweep-identified winner config.
# Edit SHAPE_M/N/K and MAX_IN_FLIGHT below from sweep results.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/pearl}"
source "${REPO_DIR}/env.sh"

# === EDIT THESE FROM H200 SWEEP RESULTS ===
SHAPE_M="${SHAPE_M:-8192}"
SHAPE_N="${SHAPE_N:-65536}"
SHAPE_K="${SHAPE_K:-8192}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-8}"
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
