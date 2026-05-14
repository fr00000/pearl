#!/usr/bin/env bash
# Shape sweep for Phase C production miner.
#
# Runs each shape for $DURATION_S in Phase C production mode
# (--no-diagnostics --enable-b-cache --max-in-flight 4) and emits a
# CSV summary with measured throughput + tile rate.
#
# Tile rate (= matmuls/s × outer_tiles/matmul) is the metric we
# optimize, not raw matmul rate. Each tile is an independent
# winning-hash candidate.
#
# Individual shape failures (OOM, sanity_check, kernel mismatch) are
# expected on the largest candidates; the sweep records them and
# continues. The only abort condition is pearl-gateway dying.

set -uo pipefail

# Pre-flight ---------------------------------------------------------
if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway is not running. Start it and retry."
    exit 1
fi
# Match the venv binary path or the actual launch command, not the
# string "direct-miner" anywhere on the cmdline — otherwise this
# script's own path matches itself.
if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    echo "ERROR: direct-miner already running. Stop it first."
    exit 1
fi

OUTDIR="/workspace/shape-sweep-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "shape_id,m,n,k,outer_tiles_per_matmul,duration_s,total_matmuls,completion_rate_mm_s,tile_rate_per_s,cache_hits,cache_misses,errors,notes" > "$SUMMARY"

# Shape list: id|m|n|k
SHAPES=(
    "baseline|4096|8192|8192"
    "larger_m|8192|8192|8192"
    "wider_n|4096|16384|8192"
    "wider_n2|4096|32768|8192"
    "both_mn|8192|16384|8192"
    "deeper_k|4096|8192|16384"
    "deeper_k2|4096|8192|32768"
    "max_tiles|8192|32768|8192"
)

MAX_IN_FLIGHT=4
DURATION_S=${DURATION_S:-300}

echo "Sweep output: $OUTDIR"
echo "Per-shape duration: ${DURATION_S}s"
echo "max_in_flight: $MAX_IN_FLIGHT"
echo "Total expected runtime: ~$(( (${#SHAPES[@]} * (DURATION_S + 15) + 60) / 60 )) min"
echo ""

for entry in "${SHAPES[@]}"; do
    IFS='|' read -r shape_id m n k <<< "$entry"

    echo ""
    echo "=========================================="
    echo "  Shape: $shape_id (m=$m n=$n k=$k)"
    echo "=========================================="

    log="$OUTDIR/${shape_id}.log"

    # timeout sends SIGINT then SIGKILL after kill-after if the process
    # ignores SIGINT. The miner installs a KeyboardInterrupt handler
    # that drains + logs FINAL stats, so SIGINT is the correct signal.
    timeout --signal=SIGINT --kill-after=30 "$DURATION_S" \
        env MINER_DEBUG=true uv run direct-miner \
            --m "$m" --n "$n" --k "$k" \
            --max-in-flight "$MAX_IN_FLIGHT" \
            --enable-b-cache \
            --no-diagnostics \
            --log-interval 100 \
            --phase-tag "sweep_${shape_id}" \
            > "$log" 2>&1 &

    miner_pid=$!
    wait $miner_pid
    miner_exit=$?

    # 0 = clean exit, 124 = timeout fired, 130 = SIGINT — all expected.
    if [[ $miner_exit -ne 0 && $miner_exit -ne 130 && $miner_exit -ne 124 ]]; then
        echo "WARNING: shape $shape_id exited unexpectedly with code $miner_exit"
    fi

    # Let any in-flight CUDA cleanup finish + pinned-memory release.
    sleep 5

    # Belt-and-braces: kill anything left over so the next shape has a
    # clean GPU.
    if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
        echo "WARNING: direct-miner still running after timeout; SIGKILLing"
        pkill -9 -f "/\.venv/bin/direct-miner" 2>/dev/null
        sleep 3
    fi

    # Confirm gateway alive — if it died, subsequent shapes can't run.
    if ! pgrep -f pearl-gateway > /dev/null; then
        echo "ERROR: pearl-gateway died during $shape_id. Aborting sweep."
        echo "${shape_id},${m},${n},${k},,,,,,,,gateway_died" >> "$SUMMARY"
        break
    fi

    # Parse results from log
    final_line=$(grep "DIRECT MINER\] FINAL" "$log" | tail -1)
    otpm_line=$(grep "outer_tiles_per_matmul=" "$log" | head -1)
    otpm=$(echo "$otpm_line" | grep -oP "outer_tiles_per_matmul=\K[0-9]+")
    bcache_line=$(grep "BCACHE FINAL" "$log" | tail -1)
    # grep -c always prints the count (including 0) to stdout; its exit
    # code is 1 on zero matches. Do not chain `|| echo 0` — that prints
    # a second 0 and corrupts the CSV with embedded newlines.
    errors=$(grep -cE "Traceback|Exception|FATAL|CUDA error|out of memory|invalid proof|rejected|Killed|OOM" "$log")

    if [[ -z "$final_line" ]]; then
        echo "  -> NO FINAL line for $shape_id (likely OOM or sanity_check on startup)"
        notes=""
        if grep -q "out of memory\|OutOfMemoryError" "$log"; then notes="oom"; fi
        if grep -q "AssertionError\|sanity\|must be" "$log"; then notes="${notes:+$notes;}sanity"; fi
        if grep -q "Traceback" "$log" && [[ -z "$notes" ]]; then notes="exception"; fi
        notes="${notes:-no_final}"
        echo "${shape_id},${m},${n},${k},${otpm:-},${DURATION_S},,,,,,${errors},${notes}" >> "$SUMMARY"
        continue
    fi

    completed=$(echo "$final_line" | grep -oP "completed=\K[0-9]+")
    elapsed=$(echo "$final_line" | grep -oP "elapsed=\K[0-9.]+")
    comp_rate=$(echo "$final_line" | grep -oP "completion_rate=\K[0-9.]+")
    tile_rate=$(echo "$final_line" | grep -oP "tile_rate=\K[0-9]+")

    if [[ -n "$bcache_line" ]]; then
        cache_hits=$(echo "$bcache_line" | grep -oP "hits=\K[0-9]+")
        cache_misses=$(echo "$bcache_line" | grep -oP "misses=\K[0-9]+")
    else
        cache_hits=""
        cache_misses=""
    fi

    echo "  -> ${completed} matmuls in ${elapsed}s = ${comp_rate} mm/s, ${tile_rate} tiles/s, errors=${errors}"
    echo "${shape_id},${m},${n},${k},${otpm},${elapsed},${completed},${comp_rate},${tile_rate},${cache_hits},${cache_misses},${errors}," >> "$SUMMARY"
done

echo ""
echo "=========================================="
echo "  Sweep complete"
echo "=========================================="
echo "Summary: $SUMMARY"
echo ""
column -s, -t < "$SUMMARY"
