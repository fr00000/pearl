#!/usr/bin/env bash
# H200 stress sweep — parallel 4-up across GPUs.
#
# Runs 8 stress cells (or fewer if pre-flight excluded any), 4 at a time
# in batches across all available GPUs. Production mode throughout
# (--enable-b-cache, diagnostics off). Matches H200_SWEEP_RESULTS.md
# methodology (4-up sharing one gateway) so results are comparable.
#
# Tests whether the ~2M tiles/sec/GPU plateau seen in the initial sweep
# is a hard kernel limit or whether bigger shapes (n up to 262144,
# m up to 32768) and deeper pipeline (mif=16) unlock further gains.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/pearl}"
source "${REPO_DIR}/env.sh"

OUTDIR="/workspace/sweeps/stress-h200-$(date +%Y%m%d-%H%M%S)"
SUMMARY="$OUTDIR/summary.csv"
mkdir -p "$OUTDIR"

# Pre-flight
if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway not running"
    exit 1
fi
if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    echo "ERROR: direct-miner already running. Stop production first."
    exit 1
fi

GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPUS[@]}
echo "Detected $NUM_GPUS GPUs: ${GPUS[*]}"

if [[ $NUM_GPUS -lt 1 ]]; then
    echo "ERROR: no GPUs detected"
    exit 1
fi

echo "cell_id,m,n,k,max_in_flight,gpu_idx,duration_s,total_matmuls,completion_rate_mm_s,tile_rate_per_s,peak_mem_mib,errors,notes" > "$SUMMARY"

# Cell definitions: id|m|n|k|mif
declare -a CELLS=(
    "stress_n128|8192|131072|8192|4"
    "stress_n128_mif8|8192|131072|8192|8"
    "stress_n256|8192|262144|8192|4"
    "stress_m32|32768|16384|8192|4"
    "stress_both|16384|65536|8192|4"
    "stress_max|16384|131072|8192|4"
    "mif16_xl|8192|65536|8192|16"
    "mif16_max|16384|65536|8192|16"
)

SKIP_LIST=""
if [[ -f /tmp/sweep_skip_cells.txt ]]; then
    SKIP_LIST=$(cat /tmp/sweep_skip_cells.txt)
fi

declare -a CELLS_TO_RUN=()
for cell in "${CELLS[@]}"; do
    IFS='|' read -r cell_id _ _ _ _ <<< "$cell"
    if [[ " $SKIP_LIST " == *" $cell_id "* ]]; then
        echo "SKIPPING $cell_id (pre-flight failure)"
        echo "${cell_id},,,,,,,,,,,,skipped_preflight" >> "$SUMMARY"
        continue
    fi
    CELLS_TO_RUN+=("$cell")
done

NUM_CELLS=${#CELLS_TO_RUN[@]}
echo "Running $NUM_CELLS cells in batches of $NUM_GPUS"
echo "Output: $OUTDIR"

DURATION_S=${DURATION_S:-300}

# Memory poller: writes "<gpu>,<MiB>" lines every 5s while it runs.
# Killed when the cell process exits.
poll_memory() {
    local gpu_idx="$1"
    local out_file="$2"
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_idx" 2>/dev/null | head -1 >> "$out_file"
        sleep 5
    done
}

run_cell() {
    local cell_entry="$1"
    local gpu_idx="$2"

    IFS='|' read -r cell_id m n k mif <<< "$cell_entry"
    local log="$OUTDIR/${cell_id}.log"
    local mempoll_file="$OUTDIR/${cell_id}.mem"
    echo "$gpu_idx" > "$OUTDIR/${cell_id}.gpu"

    echo "[GPU $gpu_idx] START $cell_id (m=$m n=$n k=$k mif=$mif)"

    poll_memory "$gpu_idx" "$mempoll_file" &
    local mempoll_pid=$!

    CUDA_VISIBLE_DEVICES="$gpu_idx" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    MINER_DEBUG=true \
    timeout --signal=SIGINT --kill-after=30 "$DURATION_S" \
        uv run direct-miner \
            --m "$m" --n "$n" --k "$k" \
            --max-in-flight "$mif" \
            --enable-b-cache \
            --log-interval 100 \
            --phase-tag "$cell_id" \
            > "$log" 2>&1
    local miner_exit=$?

    kill "$mempoll_pid" 2>/dev/null
    wait "$mempoll_pid" 2>/dev/null || true

    echo "[GPU $gpu_idx] DONE $cell_id (exit=$miner_exit)"
}

batch_num=0
for ((batch_start=0; batch_start < NUM_CELLS; batch_start += NUM_GPUS)); do
    batch_num=$((batch_num + 1))
    echo ""
    echo "================================================"
    echo "  Batch $batch_num"
    echo "================================================"

    declare -a BATCH_PIDS=()

    for ((i=0; i < NUM_GPUS; i++)); do
        idx=$((batch_start + i))
        if [[ $idx -ge $NUM_CELLS ]]; then
            break
        fi
        gpu="${GPUS[$i]}"
        cell="${CELLS_TO_RUN[$idx]}"

        run_cell "$cell" "$gpu" &
        BATCH_PIDS+=($!)
        sleep 3  # stagger startup
    done

    echo "Waiting for ${#BATCH_PIDS[@]} cells..."
    for pid in "${BATCH_PIDS[@]}"; do
        wait "$pid"
    done

    sleep 10  # cooldown — let pinned-memory pool drain

    if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
        echo "WARNING: direct-miner still running after batch; force-killing"
        pkill -9 -f "/\.venv/bin/direct-miner" 2>/dev/null
        sleep 3
    fi

    if ! pgrep -f pearl-gateway > /dev/null; then
        echo "ERROR: pearl-gateway died during batch $batch_num. Aborting."
        break
    fi
done

echo ""
echo "================================================"
echo "  Parsing results"
echo "================================================"

for cell in "${CELLS_TO_RUN[@]}"; do
    IFS='|' read -r cell_id m n k mif <<< "$cell"
    log="$OUTDIR/${cell_id}.log"
    mempoll_file="$OUTDIR/${cell_id}.mem"
    gpu_file="$OUTDIR/${cell_id}.gpu"

    gpu_idx=$(cat "$gpu_file" 2>/dev/null || echo "")

    if [[ ! -f "$log" ]]; then
        echo "${cell_id},${m},${n},${k},${mif},${gpu_idx},,,,,,,,no_log" >> "$SUMMARY"
        continue
    fi

    peak_mem=$(sort -n "$mempoll_file" 2>/dev/null | tail -1)
    peak_mem=${peak_mem:-0}

    final_line=$(grep "DIRECT MINER.*FINAL" "$log" | tail -1)

    errors=$(grep -ciE "error|exception|traceback|cuda error|out of memory" "$log" 2>/dev/null; true)
    errors=$(echo "${errors:-0}" | head -1)
    terminate_noise=$(grep -ci "terminate called without an active exception" "$log" 2>/dev/null; true)
    terminate_noise=$(echo "${terminate_noise:-0}" | head -1)
    errors=$(( ${errors:-0} - ${terminate_noise:-0} ))

    if [[ -z "$final_line" ]]; then
        notes="no_final"
        if grep -qi "out of memory" "$log" 2>/dev/null; then
            notes="oom"
        elif grep -qi "sanity" "$log" 2>/dev/null; then
            notes="sanity_check"
        fi
        echo "${cell_id},${m},${n},${k},${mif},${gpu_idx},${DURATION_S},,,,${peak_mem},${errors},${notes}" >> "$SUMMARY"
        echo "  ${cell_id}: ${notes} (peak ${peak_mem} MiB, gpu ${gpu_idx})"
        continue
    fi

    completed=$(echo "$final_line" | grep -oP "completed=\K[0-9]+")
    elapsed=$(echo "$final_line" | grep -oP "elapsed=\K[0-9.]+")
    comp_rate=$(echo "$final_line" | grep -oP "completion_rate=\K[0-9.]+")
    tile_rate=$(echo "$final_line" | grep -oP "tile_rate=\K[0-9]+")

    echo "${cell_id},${m},${n},${k},${mif},${gpu_idx},${elapsed},${completed},${comp_rate},${tile_rate},${peak_mem},${errors}," >> "$SUMMARY"
    echo "  ${cell_id}: ${tile_rate} tiles/s (${comp_rate} mm/s), peak ${peak_mem} MiB, gpu ${gpu_idx}, errors=${errors}"
done

# Sort by tile rate for readability
{
    head -1 "$SUMMARY"
    tail -n +2 "$SUMMARY" | sort -t, -k10 -n -r
} > "$SUMMARY.sorted"
mv "$SUMMARY.sorted" "$SUMMARY"

echo ""
echo "================================================"
echo "  Sweep complete"
echo "================================================"
echo "Output: $SUMMARY"
