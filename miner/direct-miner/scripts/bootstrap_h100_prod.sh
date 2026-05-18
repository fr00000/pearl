#!/usr/bin/env bash
# Pearl Mining — H100 SXM Production Bootstrap (Address-Only)
#
# Single-script setup for a fresh Hopper-GPU pod. Takes a mining
# address (not a seed) and launches one direct-miner per detected
# GPU at the verified-winner production shape.
#
# Usage:
#   PEARL_MINING_ADDR="prl1p9lq..." bash bootstrap_h100_prod.sh
#   # OR interactive (prompts):
#   bash bootstrap_h100_prod.sh
#
# Env vars:
#   PEARL_MINING_ADDR     mining address (prompted if not set; cached
#                         at ~/.pearl_mining_address on first run)
#   PEARL_BRANCH          branch to use (default: opt/direct-mining)
#   PEARL_REPO_URL        repo URL (default: fr00000/pearl fork)
#   SKIP_APT_UPDATE       skip apt update/upgrade (default: 0)
#   BUILD_OYSTER          also build the oyster wallet daemon (default: 0;
#                         not needed for address-only mining)
#
# Idempotent: state at ~/.pearl_bootstrap_state. Re-run skips
# completed steps and just relaunches miners.

set -uo pipefail

# ===== Configuration =====
PEARL_REPO_URL="${PEARL_REPO_URL:-https://github.com/fr00000/pearl.git}"
PEARL_BRANCH="${PEARL_BRANCH:-opt/direct-mining}"
REPO_DIR="$HOME/pearl"
LOG_DIR="/workspace/logs"
METRICS_DIR="/workspace/metrics"
PIDFILE_DIR="/tmp/direct-miner-pids"
PEARLD_DATA="$HOME/.pearld/data"
STATE_FILE="$HOME/.pearl_bootstrap_state"

# Verified production shape/kernel — see H100_SXM_VERIFY.md.
SHAPE_M=8192
SHAPE_N=262144
SHAPE_K=32768
MAX_IN_FLIGHT=4
KERNEL_TILE_M=128
KERNEL_TILE_N=256
KERNEL_TILE_K=128
KERNEL_STAGES=3
KERNEL_CLUSTER_M=2
KERNEL_CLUSTER_N=1
KERNEL_MMA_REGISTERS=160
KERNEL_SWIZZLE=8

# Pearl daemon RPC config (pearld is on 44107; oyster — not used here —
# would be on 44207)
PEARLD_RPC_HOST=127.0.0.1
PEARLD_RPC_PORT=44107
PEARLD_RPC_USER=rpcuser
PEARLD_RPC_PASS=rpcpass

SKIP_APT_UPDATE="${SKIP_APT_UPDATE:-0}"
BUILD_OYSTER="${BUILD_OYSTER:-0}"

# ===== Helpers =====
mark_done() { echo "$1" >> "$STATE_FILE"; }
is_done()   { [[ -f "$STATE_FILE" ]] && grep -qx "$1" "$STATE_FILE"; }
log() { echo ""; echo "===> $(date -u +%H:%M:%S) $*"; }
err() { echo "" >&2; echo "!!! $(date -u +%H:%M:%S) ERROR: $*" >&2; }
die() { err "$*"; exit 1; }

# sudo wrapper: passthrough if root, otherwise use sudo if available
sh_sudo() {
    if [[ "$EUID" -eq 0 ]]; then
        "$@"
    elif command -v sudo > /dev/null && sudo -n true 2>/dev/null; then
        sudo "$@"
    else
        "$@"
    fi
}

mkdir -p "$LOG_DIR" "$METRICS_DIR" "$PIDFILE_DIR"

# ===== Step 0: mining address =====
log "Step 0: mining address"

if [[ -z "${PEARL_MINING_ADDR:-}" ]]; then
    if [[ -f "$HOME/.pearl_mining_address" ]]; then
        PEARL_MINING_ADDR=$(cat "$HOME/.pearl_mining_address")
        log "Using cached mining address: $PEARL_MINING_ADDR"
    else
        echo "Enter your Pearl mining address (e.g., prl1p9lq...):"
        read -r PEARL_MINING_ADDR
        echo "$PEARL_MINING_ADDR" > "$HOME/.pearl_mining_address"
        chmod 600 "$HOME/.pearl_mining_address"
        log "Cached at ~/.pearl_mining_address"
    fi
fi

[[ -n "$PEARL_MINING_ADDR" ]] || die "No mining address provided"

# Format check (Pearl bech32m: prl1<bech32m>, e.g. prl1p9lq...)
if [[ ! "$PEARL_MINING_ADDR" =~ ^prl1[0-9a-z]{30,}$ ]]; then
    err "Address '$PEARL_MINING_ADDR' doesn't match expected pattern (prl1<bech32m>)"
    err "Continuing anyway; verify externally before block rewards land."
fi

log "Mining to: $PEARL_MINING_ADDR"

# ===== Step 1: system update =====
log "Step 1: system packages"

if ! is_done "system_update" && [[ "$SKIP_APT_UPDATE" != "1" ]]; then
    log "  apt-get update + upgrade + essential packages..."
    DEBIAN_FRONTEND=noninteractive sh_sudo apt-get update -qq 2>&1 | tail -3

    DEBIAN_FRONTEND=noninteractive sh_sudo apt-get upgrade -y -qq \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        2>&1 | tail -3

    DEBIAN_FRONTEND=noninteractive sh_sudo apt-get install -y -qq \
        build-essential pkg-config libssl-dev libudev-dev \
        cmake ninja-build \
        tmux jq bc netcat-openbsd \
        curl wget git ca-certificates \
        2>&1 | tail -3

    sh_sudo apt-get autoremove -y -qq 2>&1 | tail -2
    sh_sudo apt-get clean

    mark_done "system_update"
    log "System update done"
else
    log "Skipped (already done or SKIP_APT_UPDATE=1)"
fi

# ===== Step 2: toolchain =====
log "Step 2: toolchain (uv, cargo, go)"

if ! is_done "toolchain"; then
    if ! command -v uv > /dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
        log "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env" 2>/dev/null || true

    if ! command -v cargo > /dev/null && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
        log "  Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    fi
    [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v go > /dev/null && [[ ! -x /usr/local/go/bin/go ]]; then
        log "  Installing Go..."
        GO_VERSION=1.22.5
        cd /tmp
        wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
        sh_sudo rm -rf /usr/local/go
        sh_sudo tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
        rm -f "go${GO_VERSION}.linux-amd64.tar.gz"
    fi
    export PATH="$PATH:/usr/local/go/bin:$HOME/go/bin"

    # Install go-task for repo's Taskfile-based builds
    if ! command -v task > /dev/null && [[ ! -x "$HOME/go/bin/task" ]]; then
        log "  Installing go-task..."
        go install github.com/go-task/task/v3/cmd/task@latest
    fi

    # Persist PATH
    if ! grep -q "Pearl bootstrap PATH" "$HOME/.bashrc" 2>/dev/null; then
        cat >> "$HOME/.bashrc" <<'BASHRC_END'
# Pearl bootstrap PATH additions
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/go/bin:$HOME/go/bin:$PATH"
BASHRC_END
    fi

    mark_done "toolchain"
    log "Toolchain ready"
else
    log "Already installed; re-sourcing for this shell"
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env" 2>/dev/null || true
    [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/go/bin:$HOME/go/bin:$PATH"
fi

uv --version > /dev/null 2>&1 || die "uv not available"
cargo --version > /dev/null 2>&1 || die "cargo not available"
go version > /dev/null 2>&1 || die "go not available"

log "  uv:    $(uv --version 2>&1)"
log "  cargo: $(cargo --version 2>&1 | awk '{print $1, $2}')"
log "  go:    $(go version 2>&1 | awk '{print $3}')"
command -v task > /dev/null && log "  task:  $(task --version 2>&1)" || log "  task:  (not installed; fallback to direct go build)"

# ===== Step 3: GPU detection =====
log "Step 3: GPU detection"

command -v nvidia-smi > /dev/null || die "nvidia-smi not found — is this a GPU pod?"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
log "Detected $GPU_COUNT GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv | sed 's/^/   /'

NON_HOPPER=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -cv "^9\.0$" || true)
[[ "$NON_HOPPER" -eq 0 ]] || die "Found $NON_HOPPER non-Hopper GPU(s); kernel is sm_90a only"
[[ "$GPU_COUNT" -ge 1 ]] || die "No GPUs detected"

command -v nvcc > /dev/null || die "nvcc not found — image needs CUDA devel tools to build pearl-gemm; try nvidia/cuda:12.9.0-devel-ubuntu24.04 base"
log "CUDA: $(nvcc --version | grep release | head -1)"

# ===== Step 4: clone =====
log "Step 4: clone repo"

if [[ ! -d "$REPO_DIR/.git" ]]; then
    cd "$HOME"
    git clone "$PEARL_REPO_URL" pearl 2>&1 | tail -3
    cd "$REPO_DIR"
    git checkout "$PEARL_BRANCH"
else
    # Always pull on re-run. Cheap operation; avoids the "I pushed a
    # fix but the pod still runs the old code" footgun.
    cd "$REPO_DIR"
    git fetch origin "$PEARL_BRANCH" 2>&1 | tail -2
    git checkout "$PEARL_BRANCH"
    git pull origin "$PEARL_BRANCH" 2>&1 | tail -2
fi
cd "$REPO_DIR"
log "Repo head: $(git log -1 --oneline)"

# Source the repo's env.sh (sets LD_LIBRARY_PATH for venv CUDA 12.9
# libs vs system CUDA 12.8 — critical to avoid cusparse mismatch)
[[ -f "$REPO_DIR/env.sh" ]] && source "$REPO_DIR/env.sh"

# ===== Step 5: builds =====
log "Step 5: builds (first run: 15-30 min for CUDA kernels)"

if ! is_done "build_python"; then
    log "  Python deps + CUDA kernels via uv sync..."
    uv sync --all-packages 2>&1 | tee "$LOG_DIR/uv-sync.log" | tail -8
    # Re-source env.sh: on a fresh install the nvidia/torch dirs didn't exist
    # before sync, so the earlier source-at-line-225 silently no-op'd. Without
    # LD_LIBRARY_PATH the import below fails with `libc10.so: cannot open`.
    [[ -f "$REPO_DIR/env.sh" ]] && source "$REPO_DIR/env.sh"
    uv run python -c "import direct_miner, pearl_gemm, pearl_gateway; print('Python imports OK')" \
        || die "Python imports failed after uv sync; see $LOG_DIR/uv-sync.log"
    mark_done "build_python"
fi

if ! is_done "build_go"; then
    log "  Go binaries (pearld, prlctl)..."
    mkdir -p "$REPO_DIR/bin"

    if command -v task > /dev/null; then
        (task build:pearl) 2>&1 | tee "$LOG_DIR/build-go.log" | tail -5
        if [[ "$BUILD_OYSTER" == "1" ]]; then
            (task build:oyster) 2>&1 | tee -a "$LOG_DIR/build-go.log" | tail -3
        fi
    else
        # Fallback: build with the same tags Taskfile uses
        log "    (task not available; using direct go build with -tags xmss,zkpow)"
        (cd "$REPO_DIR" && go build -tags xmss,zkpow -o bin/pearld ./node) 2>&1 | tee "$LOG_DIR/build-pearld.log" | tail -3
        (cd "$REPO_DIR" && go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl) 2>&1 | tee "$LOG_DIR/build-prlctl.log" | tail -3
        if [[ "$BUILD_OYSTER" == "1" ]]; then
            (cd "$REPO_DIR" && go build -tags xmss,zkpow -o bin/oyster ./wallet) 2>&1 | tee "$LOG_DIR/build-oyster.log" | tail -3
        fi
    fi

    [[ -x "$REPO_DIR/bin/pearld" ]]  || die "pearld build failed"
    [[ -x "$REPO_DIR/bin/prlctl" ]]  || die "prlctl build failed"
    mark_done "build_go"
fi

log "Binaries:"
ls -la "$REPO_DIR/bin/" | tail -n +2 | sed 's/^/   /'

# ===== Step 6: pearld =====
log "Step 6: pearld"

if ! pgrep -x pearld > /dev/null; then
    mkdir -p "$HOME/.pearld" "$PEARLD_DATA"

    # Ensure session + named window exist. Never rename existing
    # windows — if window 0 was renamed to something else by the user
    # we don't hijack it. ensure_window only creates "pearld" when
    # no window by that name exists yet.
    tmux new-session -d -s pearl -n bootstrap 2>/dev/null || true
    tmux list-windows -t pearl -F '#{window_name}' 2>/dev/null | grep -qx pearld \
        || tmux new-window -t pearl -n pearld

    # pearld --miningaddr is set for symmetry with the canonical pod
    # and as a fallback if pearld's internal miner is ever activated.
    # The gateway (next step) builds its OWN coinbase tx using
    # PEARLD_MINING_ADDRESS — that's the authoritative source for the
    # direct-miner path. Both being the same address is intentional.
    tmux send-keys -t pearl:pearld \
        "$REPO_DIR/bin/pearld \
            --rpcuser=$PEARLD_RPC_USER \
            --rpcpass=$PEARLD_RPC_PASS \
            --rpclisten=${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT} \
            --miningaddr=$PEARL_MINING_ADDR \
            --datadir=$PEARLD_DATA \
            --txindex \
            --notls \
            --debuglevel=info \
            2>&1 | tee $LOG_DIR/pearld.log" C-m

    sleep 10
    pgrep -x pearld > /dev/null || die "pearld failed to start; check $LOG_DIR/pearld.log"
    log "  Started pearld (tmux: pearl:pearld)"
else
    log "  pearld already running (PID $(pgrep -x pearld))"
fi

# ===== Step 7: pearl-gateway =====
log "Step 7: pearl-gateway (Python module via uv)"

if ! pgrep -f "pearl-gateway start" > /dev/null; then
    tmux list-windows -t pearl -F '#{window_name}' 2>/dev/null | grep -qx gateway \
        || tmux new-window -t pearl -n gateway

    tmux send-keys -t pearl:gateway \
        "cd $REPO_DIR && \
         source $REPO_DIR/env.sh && \
         PEARLD_RPC_URL=http://${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT} \
         PEARLD_RPC_USER=$PEARLD_RPC_USER \
         PEARLD_RPC_PASSWORD=$PEARLD_RPC_PASS \
         PEARLD_MINING_ADDRESS=$PEARL_MINING_ADDR \
         MINER_RPC_SOCKET_PATH=/tmp/pearlgw.sock \
         uv run pearl-gateway start 2>&1 | tee $LOG_DIR/gateway.log" C-m

    sleep 12
    pgrep -f "pearl-gateway start" > /dev/null || die "gateway failed to start; check $LOG_DIR/gateway.log"
    [[ -S /tmp/pearlgw.sock ]] || die "Gateway UDS socket not created at /tmp/pearlgw.sock"
    log "  pearl-gateway running, socket at /tmp/pearlgw.sock"
else
    log "  pearl-gateway already running"
fi

# ===== Step 8: wait for sync =====
log "Step 8: pearld sync (cold from genesis: 1-3 h)"

PRLCTL_BASE=( "$REPO_DIR/bin/prlctl" -u "$PEARLD_RPC_USER" -P "$PEARLD_RPC_PASS" -s "${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT}" --notls )

SYNC_START=$(date +%s)
last_logged_blocks=-1
peer_warning_emitted=0
while true; do
    CHAIN_INFO=$("${PRLCTL_BASE[@]}" getblockchaininfo 2>/dev/null || echo "{}")
    BLOCKS=$(echo "$CHAIN_INFO" | jq -r '.blocks // 0' 2>/dev/null || echo "0")
    HEADERS=$(echo "$CHAIN_INFO" | jq -r '.headers // 0' 2>/dev/null || echo "0")
    MEDIAN_TIME=$(echo "$CHAIN_INFO" | jq -r '.mediantime // 0' 2>/dev/null || echo "0")
    PEERS=$("${PRLCTL_BASE[@]}" getpeerinfo 2>/dev/null | jq 'length' 2>/dev/null || echo "0")

    if [[ "$BLOCKS" == "0" ]] && [[ "$HEADERS" == "0" ]]; then
        log "  pearld RPC not yet ready; retrying..."
        sleep 15
        continue
    fi

    NOW=$(date +%s)
    ELAPSED=$((NOW - SYNC_START))
    TIP_AGE=$((NOW - MEDIAN_TIME))

    if [[ "$PEERS" == "0" ]] && (( ELAPSED > 300 )) && [[ "$peer_warning_emitted" == "0" ]]; then
        err "  Peer count still 0 after 5 min — discovery may be failing"
        err "  Try adding '--addpeer=<IP>:44108' to pearld; reachable IPs from this network:"
        err "    34.185.139.226:44108  34.179.180.50:44108  192.222.52.63:44108"
        err "    34.30.61.164:44108    34.169.137.191:44108"
        peer_warning_emitted=1
    fi

    GAP=$((HEADERS - BLOCKS))
    if [[ "$BLOCKS" != "$last_logged_blocks" ]] || (( ELAPSED % 60 < 30 )); then
        log "  blocks=$BLOCKS headers=$HEADERS gap=$GAP peers=$PEERS tip_age=${TIP_AGE}s elapsed=${ELAPSED}s"
        last_logged_blocks=$BLOCKS
    fi

    # Both conditions required: gap<100 alone is a false positive — on cold
    # start the first peer feeds a small batch of headers (e.g. 765) and
    # pearld processes them in seconds, hitting gap=0 before later peers
    # advertise the full chain. tip_age<1800s confirms we're actually near
    # the network tip, not just caught up to one peer's view.
    if (( GAP < 100 )) && [[ "$BLOCKS" != "0" ]] && (( TIP_AGE < 1800 )); then
        log "pearld synced (gap=$GAP blocks, tip_age=${TIP_AGE}s; took ${ELAPSED}s)"
        break
    fi

    sleep 30
done

# ===== Step 9: launch miners =====
log "Step 9: launch miners ($GPU_COUNT GPU(s))"

if pgrep -f "/\.venv/bin/direct-miner" > /dev/null; then
    log "  Stopping existing direct-miner instances..."
    pkill -SIGINT -f "/\.venv/bin/direct-miner" 2>/dev/null
    sleep 10
    pkill -9 -f "/\.venv/bin/direct-miner" 2>/dev/null
    sleep 3
fi
rm -f "$PIDFILE_DIR"/*.pid

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')
LAUNCHED=0
FAILED=0

for gpu_idx in $GPUS; do
    log "  GPU $gpu_idx: launching..."

    log_file="$LOG_DIR/gpu${gpu_idx}-${TIMESTAMP}.log"
    metrics_file="$METRICS_DIR/gpu${gpu_idx}-${TIMESTAMP}.jsonl"
    pidfile="$PIDFILE_DIR/${gpu_idx}.pid"

    cd "$REPO_DIR"
    CUDA_VISIBLE_DEVICES="$gpu_idx" \
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
        --kernel-mma-registers "$KERNEL_MMA_REGISTERS" \
        --kernel-swizzle "$KERNEL_SWIZZLE" \
        --log-interval 200 \
        --phase-tag "prod_gpu${gpu_idx}" \
        --metrics-output "$metrics_file" \
        > "$log_file" 2>&1 &

    pid=$!
    disown $pid 2>/dev/null || true
    echo "$pid" > "$pidfile"

    sleep 5  # stagger to avoid gateway thundering herd at startup

    if kill -0 "$pid" 2>/dev/null; then
        log "    PID $pid (log: $log_file)"
        LAUNCHED=$((LAUNCHED + 1))
    else
        err "    GPU $gpu_idx died within 5s. Tail:"
        tail -20 "$log_file" >&2
        FAILED=$((FAILED + 1))
    fi
done

[[ "$LAUNCHED" -ge 1 ]] || die "No miners successfully launched"

# ===== Done =====
EXPECTED_TILES_PER_GPU=1300000
EXPECTED_TILES=$((LAUNCHED * EXPECTED_TILES_PER_GPU))
EXPECTED_MM=$(awk -v g="$LAUNCHED" 'BEGIN { printf "%.1f", g * 19.9 }')

cat <<EOF

===========================================
  Bootstrap complete
===========================================
  Mining to:         $PEARL_MINING_ADDR
  GPUs detected:     $GPU_COUNT
  Miners launched:   $LAUNCHED
  Miners failed:     $FAILED
  Production shape:  ${SHAPE_M} × ${SHAPE_N} × ${SHAPE_K}, mif=$MAX_IN_FLIGHT
  Kernel:            ${KERNEL_TILE_M}×${KERNEL_TILE_N}×${KERNEL_TILE_K}, stages=$KERNEL_STAGES, cluster=${KERNEL_CLUSTER_M}×${KERNEL_CLUSTER_N}, regs=$KERNEL_MMA_REGISTERS, swizzle=$KERNEL_SWIZZLE, headless
  Expected per-GPU:  ~1.30 M 128-equivalent attempts/s (~19.9 mm/s)
  Expected total:    ~${EXPECTED_TILES} 128-equivalent attempts/s (~${EXPECTED_MM} mm/s aggregate)

Useful commands:
  Per-GPU rate (latest sample):
    for f in $LOG_DIR/gpu*-${TIMESTAMP}.log; do echo "=== \$(basename \$f) ==="; grep 'DIRECT MINER' "\$f" | tail -1; done

  GPU utilization:
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

  pearld block height:
    $REPO_DIR/bin/prlctl -u $PEARLD_RPC_USER -P $PEARLD_RPC_PASS -s ${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT} --notls getblockcount

  pearld peer count:
    $REPO_DIR/bin/prlctl -u $PEARLD_RPC_USER -P $PEARLD_RPC_PASS -s ${PEARLD_RPC_HOST}:${PEARLD_RPC_PORT} --notls getpeerinfo | jq 'length'

  Stop all miners:
    pkill -SIGINT -f /\\.venv/bin/direct-miner

  Tmux session (pearld + gateway windows):
    tmux attach -t pearl

  PIDs:        $PIDFILE_DIR/
  Logs:        $LOG_DIR/
  Metrics:     $METRICS_DIR/   (empty unless --enable-diagnostics; off by default)
  State file:  $STATE_FILE     (re-run safe — completed steps skipped)
  Address:     $HOME/.pearl_mining_address

To force a re-do of a step (e.g., reinstall toolchain), remove its
line from $STATE_FILE and re-run.
EOF
