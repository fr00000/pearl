#!/usr/bin/env bash
# Focused H100/Hopper kernel sweep for direct-miner headless mode.
#
# This sweep intentionally compares normalized_attempt_rate_per_s, not raw
# outer CTA rate. Raw CTA rate is only a launch-grid counter; normalized
# attempts scale each CTA by the number of MMA consumer threads that actually
# check PoW.

set -uo pipefail

direct_miner_pids() {
    ps -eo pid=,comm=,args= | awk '
        $0 ~ /\/\.venv\/bin\/direct-miner/ { print $1; next }
        $2 ~ /python/ && $0 ~ / -m direct_miner/ { print $1; next }
    '
}

REPO_DIR="${REPO_DIR:-/root/pearl}"
if [[ -f "${REPO_DIR}/env.sh" ]]; then
    source "${REPO_DIR}/env.sh"
fi

if ! pgrep -f pearl-gateway > /dev/null; then
    echo "ERROR: pearl-gateway is not running. Start it and retry."
    exit 1
fi

if [[ -n "$(direct_miner_pids)" ]]; then
    echo "ERROR: direct-miner already running. Stop it first."
    exit 1
fi

SHAPE_M="${SHAPE_M:-8192}"
SHAPE_N="${SHAPE_N:-262144}"
SHAPE_K="${SHAPE_K:-32768}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"
GPU_INDEX="${GPU_INDEX:-0}"
DURATION_S="${DURATION_S:-300}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
KERNEL_SWIZZLE="${KERNEL_SWIZZLE:-8}"

OUTDIR="/workspace/sweeps/kernel-h100-$(date +%Y%m%d-%H%M%S)"
SUMMARY="$OUTDIR/summary.csv"
mkdir -p "$OUTDIR"

echo "variant_id,m,n,k,tile_m,tile_n,tile_k,stages,cluster_m,cluster_n,mma_registers,duration_s,total_matmuls,completion_rate_mm_s,raw_outer_tile_rate_per_s,normalized_attempt_rate_per_s,errors,pattern_compatible,notes" > "$SUMMARY"

# variant_id|tile_m|tile_n|tile_k|stages|cluster_m|cluster_n|mma_registers|notes
#
# The current default proof column pattern reaches columns 248/249, so this
# sweep keeps tile_n=256. Non-256 tile_n variants need a separate proof-pattern
# inspection pass before they are meaningful production candidates.
VARIANTS=(
    "prod_default|128|256|128|3|2|1||current_production"
    "prod_regs160|128|256|128|3|2|1|160|register_confirm"
    "prod_regs192|128|256|128|3|2|1|192|register_confirm"
    "prod_regs224|128|256|128|3|2|1|224|register_confirm"
    "prod_s2_c1x1|128|256|128|2|1|1||stage2_probe"
    "prod_s2_c2x1_regs160|128|256|128|2|2|1|160|stage2_probe"
    "prod_s2_c2x1_regs192|128|256|128|2|2|1|192|stage2_probe"
    "s3_c1x1|128|256|128|3|1|1||cluster_stage"
    "s3_c1x2|128|256|128|3|1|2||cluster_stage"
    "s3_c2x2|128|256|128|3|2|2||cluster_stage"
    "s4_c1x1|128|256|128|4|1|1||cluster_stage"
    "s4_c2x1|128|256|128|4|2|1||cluster_stage"
    "s4_c1x2|128|256|128|4|1|2||cluster_stage"
    "s4_c2x2|128|256|128|4|2|2||cluster_stage"
    "s5_c1x1|128|256|128|5|1|1||cluster_stage"
    "k64_s3_c1x1|128|256|64|3|1|1||tile_k_probe"
    "k64_s3_c2x1|128|256|64|3|2|1||tile_k_probe"
    "k64_s4_c2x1|128|256|64|4|2|1||tile_k_probe"
    "k256_s2_c1x1|128|256|256|2|1|1||tile_k_probe"
    "k256_s2_c2x1_regs160|128|256|256|2|2|1|160|tile_k_probe"
    "k256_s2_c2x1_regs192|128|256|256|2|2|1|192|tile_k_probe"
    "m64_s3_c1x1|64|256|128|3|1|1||normalization_guard"
    "m64_s3_c2x1|64|256|128|3|2|1||normalization_guard"
    "m192_s3_c1x1_regs160|192|256|128|3|1|1|160|producer_consumer_geometry"
    "m192_s3_c2x1_regs160|192|256|128|3|2|1|160|producer_consumer_geometry"
)

echo "Kernel sweep output: $OUTDIR"
echo "GPU: $GPU_INDEX"
echo "Shape: m=$SHAPE_M n=$SHAPE_N k=$SHAPE_K"
echo "max_in_flight: $MAX_IN_FLIGHT"
echo "kernel_swizzle: $KERNEL_SWIZZLE"
echo "duration/cell: ${DURATION_S}s"
echo "variants: ${#VARIANTS[@]}"
if [[ -n "$VARIANT_FILTER" ]]; then
    echo "variant filter: $VARIANT_FILTER"
fi
echo ""

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r variant_id tile_m tile_n tile_k stages cluster_m cluster_n mma_registers notes <<< "$entry"
    if [[ -n "$VARIANT_FILTER" ]] && [[ ! "$variant_id" =~ $VARIANT_FILTER ]]; then
        continue
    fi

    log="$OUTDIR/${variant_id}.log"
    pattern_log="$OUTDIR/${variant_id}.pattern.log"

    echo ""
    echo "=========================================="
    echo "  Variant: $variant_id"
    echo "  tile=${tile_m}x${tile_n}x${tile_k} stages=${stages} cluster=${cluster_m}x${cluster_n} regs=${mma_registers:-default}"
    echo "=========================================="

    inspect_cmd=(
        uv run direct-miner-inspect-pattern
        --tile-m "$tile_m"
        --tile-n "$tile_n"
        --tile-k "$tile_k"
        --cluster-m "$cluster_m"
        --cluster-n "$cluster_n"
        --stages "$stages"
        --iterations 2
    )
    if [[ -n "$mma_registers" ]]; then
        inspect_cmd+=(--mma-registers "$mma_registers")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_INDEX" "${inspect_cmd[@]}" > "$pattern_log" 2>&1
    inspect_exit=$?
    pattern_compatible="false"
    if [[ $inspect_exit -eq 0 ]] && grep -q "PATTERN_COMPATIBLE=true" "$pattern_log"; then
        pattern_compatible="true"
    fi

    if [[ "$pattern_compatible" != "true" ]]; then
        echo "  -> pattern incompatible or inspect failed; skipping benchmark"
        echo "${variant_id},${SHAPE_M},${SHAPE_N},${SHAPE_K},${tile_m},${tile_n},${tile_k},${stages},${cluster_m},${cluster_n},${mma_registers:-},${DURATION_S},,,,,0,${pattern_compatible},pattern_check_failed" >> "$SUMMARY"
        continue
    fi

    miner_cmd=(
        timeout --signal=SIGINT --kill-after=30 "$DURATION_S"
        uv run direct-miner
        --m "$SHAPE_M"
        --n "$SHAPE_N"
        --k "$SHAPE_K"
        --max-in-flight "$MAX_IN_FLIGHT"
        --enable-b-cache
        --enable-headless-kernel
        --kernel-tile-m "$tile_m"
        --kernel-tile-n "$tile_n"
        --kernel-tile-k "$tile_k"
        --kernel-stages "$stages"
        --kernel-cluster-m "$cluster_m"
        --kernel-cluster-n "$cluster_n"
        --kernel-swizzle "$KERNEL_SWIZZLE"
        --log-interval 100
        --phase-tag "kernel_${variant_id}"
    )
    if [[ -n "$mma_registers" ]]; then
        miner_cmd+=(--kernel-mma-registers "$mma_registers")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    MINER_DEBUG=true \
    "${miner_cmd[@]}" > "$log" 2>&1
    miner_exit=$?

    if [[ $miner_exit -ne 0 && $miner_exit -ne 124 && $miner_exit -ne 130 ]]; then
        echo "  -> warning: miner exited with code $miner_exit"
    fi

    sleep 5
    leftover_pids="$(direct_miner_pids)"
    if [[ -n "$leftover_pids" ]]; then
        echo "  -> warning: direct-miner still running; force-killing"
        kill -9 $leftover_pids 2>/dev/null
        sleep 3
    fi

    errors=$(grep -ciE "error|exception|traceback|cuda error|out of memory|invalid proof|rejected" "$log" 2>/dev/null; true)
    terminate_noise=$(grep -ci "terminate called without an active exception" "$log" 2>/dev/null; true)
    errors=$(( ${errors:-0} - ${terminate_noise:-0} ))

    final_line=$(grep "DIRECT MINER.*FINAL" "$log" | tail -1)
    if [[ -z "$final_line" ]]; then
        progress_line=$(grep "DIRECT MINER.*completed=" "$log" | tail -1)
        completed=""
        comp_rate=""
        raw_outer_tile_rate=""
        normalized_attempt_rate=""
        if [[ -n "$progress_line" ]]; then
            completed=$(echo "$progress_line" | grep -oP "completed=\K[0-9]+")
            comp_rate=$(echo "$progress_line" | grep -oP "completed=[0-9]+ \(\K[0-9.]+(?=/s avg)")
            raw_outer_tile_rate=$(echo "$progress_line" | grep -oP "outer_tiles=\(\K[0-9]+")
            normalized_attempt_rate=$(echo "$progress_line" | grep -oP "attempts=\(\K[0-9]+")
            echo "  -> NO FINAL line; using last steady log: ${normalized_attempt_rate} attempts/s (${raw_outer_tile_rate} raw outer tiles/s, ${comp_rate} mm/s)"
            echo "${variant_id},${SHAPE_M},${SHAPE_N},${SHAPE_K},${tile_m},${tile_n},${tile_k},${stages},${cluster_m},${cluster_n},${mma_registers:-},${DURATION_S},${completed},${comp_rate},${raw_outer_tile_rate},${normalized_attempt_rate},${errors},${pattern_compatible},no_final_midrun;${notes}" >> "$SUMMARY"
        else
            echo "  -> NO FINAL line"
            echo "${variant_id},${SHAPE_M},${SHAPE_N},${SHAPE_K},${tile_m},${tile_n},${tile_k},${stages},${cluster_m},${cluster_n},${mma_registers:-},${DURATION_S},,,,,${errors},${pattern_compatible},no_final;${notes}" >> "$SUMMARY"
        fi
        continue
    fi

    completed=$(echo "$final_line" | grep -oP "completed=\K[0-9]+")
    elapsed=$(echo "$final_line" | grep -oP "elapsed=\K[0-9.]+")
    comp_rate=$(echo "$final_line" | grep -oP "completion_rate=\K[0-9.]+")
    raw_outer_tile_rate=$(echo "$final_line" | grep -oP "raw_outer_tile_rate=\K[0-9]+")
    normalized_attempt_rate=$(echo "$final_line" | grep -oP "normalized_attempt_rate=\K[0-9]+")

    echo "  -> ${normalized_attempt_rate} attempts/s (${raw_outer_tile_rate} raw outer tiles/s, ${comp_rate} mm/s), errors=${errors}"
    echo "${variant_id},${SHAPE_M},${SHAPE_N},${SHAPE_K},${tile_m},${tile_n},${tile_k},${stages},${cluster_m},${cluster_n},${mma_registers:-},${elapsed},${completed},${comp_rate},${raw_outer_tile_rate},${normalized_attempt_rate},${errors},${pattern_compatible},${notes}" >> "$SUMMARY"
done

{
    head -1 "$SUMMARY"
    tail -n +2 "$SUMMARY" | sort -t, -k16 -n -r
} > "$SUMMARY.sorted"
mv "$SUMMARY.sorted" "$SUMMARY"

echo ""
echo "=========================================="
echo "  Kernel sweep complete"
echo "=========================================="
echo "Summary: $SUMMARY"
echo ""
if command -v column > /dev/null; then
    column -s, -t < "$SUMMARY"
else
    cat "$SUMMARY"
fi
