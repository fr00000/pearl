#!/usr/bin/env bash
# H200 parallel sweep: 5 shapes × 3 max_in_flight = 15 cells, distributed
# round-robin across all 4 GPUs (each GPU runs its cells sequentially).
# Each cell: 5 min. Total wallclock: ~20 min (vs. 75 min for serial).
#
# Production mode (diagnostics off by default). The 4 miners share one
# pearl-gateway socket — that's exactly how prod runs, so any gateway
# contention will show up here.

set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/pearl}"
source "${REPO_DIR}/env.sh"

# Pre-flight
if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway is not running"
    exit 1
fi
if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    echo "ERROR: direct-miner already running. Stop it first."
    exit 1
fi

OUTDIR="/workspace/sweeps/h200-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"

echo "shape_id,m,n,k,max_in_flight,gpu,duration_s,total_matmuls,completion_rate_mm_s,raw_outer_tile_rate_per_s,normalized_attempt_rate_per_s,errors,notes" > "$SUMMARY"

# 15 cells: shape_id|m|n|k|mif (round-robin distributed below)
CELLS=(
    "s_baseline|4096|8192|8192|2"
    "s_baseline|4096|8192|8192|4"
    "s_baseline|4096|8192|8192|8"
    "s_winner_h100|4096|32768|8192|2"
    "s_winner_h100|4096|32768|8192|4"
    "s_winner_h100|4096|32768|8192|8"
    "s_max_h100|8192|32768|8192|2"
    "s_max_h100|8192|32768|8192|4"
    "s_max_h100|8192|32768|8192|8"
    "s_xl|8192|65536|8192|2"
    "s_xl|8192|65536|8192|4"
    "s_xl|8192|65536|8192|8"
    "s_xxl|16384|32768|8192|2"
    "s_xxl|16384|32768|8192|4"
    "s_xxl|16384|32768|8192|8"
)

NUM_GPUS=4
DURATION_S=${DURATION_S:-300}
TOTAL_CELLS=${#CELLS[@]}

# Round-robin assignment: cell i → GPU (i % NUM_GPUS)
declare -a GPU0_JOBS GPU1_JOBS GPU2_JOBS GPU3_JOBS
for i in "${!CELLS[@]}"; do
    case $((i % NUM_GPUS)) in
        0) GPU0_JOBS+=("${CELLS[$i]}");;
        1) GPU1_JOBS+=("${CELLS[$i]}");;
        2) GPU2_JOBS+=("${CELLS[$i]}");;
        3) GPU3_JOBS+=("${CELLS[$i]}");;
    esac
done

echo "Sweep output: $OUTDIR"
echo "Per-cell duration: ${DURATION_S}s"
echo "Cells: $TOTAL_CELLS distributed across $NUM_GPUS GPUs"
echo "  GPU 0: ${#GPU0_JOBS[@]} cells"
echo "  GPU 1: ${#GPU1_JOBS[@]} cells"
echo "  GPU 2: ${#GPU2_JOBS[@]} cells"
echo "  GPU 3: ${#GPU3_JOBS[@]} cells"
echo "Expected wallclock: ~$(( ( (TOTAL_CELLS + NUM_GPUS - 1) / NUM_GPUS ) * (DURATION_S + 15) / 60 + 1 )) min"
echo ""

# Atomic CSV writer (flock so 4 parallel workers don't tear lines)
write_summary() {
    flock -x 200
    echo "$1" >> "$SUMMARY"
} 200>"$OUTDIR/.summary.lock"

run_one_cell() {
    local gpu_idx=$1
    local entry=$2
    IFS='|' read -r shape_id m n k mif <<< "$entry"
    local run_id="${shape_id}_mif${mif}"
    local log="$OUTDIR/${run_id}.log"

    echo "[GPU${gpu_idx}] START $run_id (m=$m n=$n k=$k mif=$mif)"

    CUDA_VISIBLE_DEVICES="$gpu_idx" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    MINER_DEBUG=true \
    timeout --signal=SIGINT --kill-after=30 "$DURATION_S" \
        uv run direct-miner \
            --m "$m" --n "$n" --k "$k" \
            --max-in-flight "$mif" \
            --enable-b-cache \
            --log-interval 200 \
            --phase-tag "$run_id" \
            > "$log" 2>&1
    local exit_code=$?

    # Parse FINAL line
    local final_line=$(grep "DIRECT MINER.*FINAL" "$log" | tail -1)
    local note=""
    local completed="" elapsed="" comp_rate="" raw_outer_tile_rate="" normalized_attempt_rate="" errors=0

    if [[ -z "$final_line" ]]; then
        note="no_final"
        grep -qi "out of memory\|CUDA out of memory" "$log" 2>/dev/null && note="oom"
        grep -qi "sanity_check\|assertion" "$log" 2>/dev/null && note="sanity_fail"
        errors=$(grep -ciE "error|exception|traceback|out of memory" "$log" 2>/dev/null; true)
        echo "[GPU${gpu_idx}] DONE  $run_id  -> NO FINAL ($note, errors=$errors)"
    else
        completed=$(echo "$final_line" | grep -oP "completed=\K[0-9]+")
        elapsed=$(echo "$final_line" | grep -oP "elapsed=\K[0-9.]+")
        comp_rate=$(echo "$final_line" | grep -oP "completion_rate=\K[0-9.]+")
        raw_outer_tile_rate=$(echo "$final_line" | grep -oP "raw_outer_tile_rate=\K[0-9]+")
        normalized_attempt_rate=$(echo "$final_line" | grep -oP "normalized_attempt_rate=\K[0-9]+")
        # grep -c always prints a count to stdout (even 0); the trailing `|| echo "0"` pattern
        # produces multi-line "0\n0" output that breaks $(( ... )) arithmetic. Use ; true instead.
        errors=$(grep -ciE "error|exception|traceback|cuda error" "$log" 2>/dev/null; true)
        terminate_noise=$(grep -ci "terminate called without an active exception" "$log" 2>/dev/null; true)
        errors=$(( ${errors:-0} - ${terminate_noise:-0} ))
        echo "[GPU${gpu_idx}] DONE  $run_id  -> ${comp_rate} mm/s, ${normalized_attempt_rate} attempts/s (${raw_outer_tile_rate} raw outer tiles/s), errors=${errors}"
    fi

    write_summary "${shape_id},${m},${n},${k},${mif},${gpu_idx},${elapsed:-${DURATION_S}},${completed},${comp_rate},${raw_outer_tile_rate},${normalized_attempt_rate},${errors},${note}"
}

# Run a worker that processes its job list sequentially.
run_worker() {
    local gpu_idx=$1
    shift
    for entry in "$@"; do
        run_one_cell "$gpu_idx" "$entry"
        # Brief cleanup between cells on the same GPU
        sleep 3
    done
    echo "[GPU${gpu_idx}] worker complete"
}

START=$(date +%s)
echo "Sweep started at $(date)"
echo ""

run_worker 0 "${GPU0_JOBS[@]}" &
PID0=$!
run_worker 1 "${GPU1_JOBS[@]}" &
PID1=$!
run_worker 2 "${GPU2_JOBS[@]}" &
PID2=$!
run_worker 3 "${GPU3_JOBS[@]}" &
PID3=$!

wait $PID0 $PID1 $PID2 $PID3

END=$(date +%s)
echo ""
echo "=========================================="
echo "  H200 sweep complete in $((END - START))s ($(date))"
echo "=========================================="

# Belt and braces — kill anything stuck
if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    pkill -9 -f "/\.venv/bin/direct-miner" 2>/dev/null
    sleep 2
fi

# Sort summary by normalized attempt rate for readability
{
    head -1 "$SUMMARY"
    tail -n +2 "$SUMMARY" | sort -t, -k11 -n -r
} > "$SUMMARY.sorted"
mv "$SUMMARY.sorted" "$SUMMARY"

echo "Summary: $SUMMARY"
echo ""
column -s, -t < "$SUMMARY"

# Final health check
if ! pgrep -f pearl-gateway > /dev/null; then
    echo ""
    echo "WARNING: pearl-gateway died at some point during sweep"
fi
