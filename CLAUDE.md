# Pearl Mining — Production Deployment

This repo is a fork of Pearl with optimized mining code on the `opt/direct-mining`
branch. The headline change: ~16× per-GPU throughput improvement over the
reference `vllm-miner` via a custom direct-miner pipeline with B-cache and
multi-stream A-side overlap.

## Your job as Claude Code

When the user asks you to "deploy a miner" or similar, your goal is to:

1. SSH into the user-provided RunPod pod
2. Run the bootstrap script `miner/direct-miner/scripts/bootstrap_h100_prod.sh`
   with the user's mining address
3. Monitor the bootstrap to completion (~2-4 hours mostly waiting on pearld
   blockchain sync)
4. Verify miners are running healthily
5. Report back with rates, mining address, and useful follow-up commands

This is **deployment work, not engineering work**. Do not modify code. Do not
run sweeps. Do not change the production shape. The script and configuration
are already validated; your job is to execute and verify.

## Required inputs from the user

Before starting, you must have:

- **SSH connection details**: host, port (often non-standard for RunPod), user
  (usually `root`), and either password or SSH key path. RunPod typically gives
  these in the "Connect" panel.
- **Mining address**: a Pearl bech32m address starting with `prl1`. Should be
  ~62 characters. This is where block rewards go.

If either is missing, ask for it. Do NOT proceed with placeholders.

## Production configuration (locked in — do not change)

```
shape:         m=8192, n=262144, k=8192
max_in_flight: 4
flags:         --enable-b-cache (diagnostics off by default)
branch:        opt/direct-mining
expected:      ~2.1M tiles/sec per H100 SXM (or H200 SXM) GPU
```

These values come from extensive sweeps documented in:
- `miner/direct-miner/H100_SXM_VERIFY.md` (production shape verified on H100 SXM)
- `miner/direct-miner/H200_SWEEP_RESULTS.md` (parallel sweep including stress addendum)

Do not edit the bootstrap script's shape constants unless the user explicitly
asks for a different shape.

Single source of truth: the constants `SHAPE_M / SHAPE_N / SHAPE_K /
MAX_IN_FLIGHT` in `bootstrap_h100_prod.sh` are authoritative. This document is
documentation. If the two diverge, the script wins; update this file to match.

## Deployment procedure

### Step 1: Verify SSH connectivity

```bash
ssh -i <key> -p <port> -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    <user>@<host> "echo CONNECTED && nvidia-smi --query-gpu=name,compute_cap --format=csv"
```

Expected: `CONNECTED` plus a list of one or more H100 (compute_cap 9.0) or
H200 (also 9.0) GPUs.

If you see anything other than Hopper GPUs (5090=12.0, B200=10.0, A100=8.0,
etc.), STOP. The kernel is sm_90a only and won't run elsewhere. Tell the user.

If the SSH connection fails, debug it together with the user before continuing.
Common issues: wrong port (RunPod uses non-22 ports), wrong key permissions
(needs 0600), pod still booting.

### Step 2: Verify the base image has what we need

```bash
ssh ... "which nvcc; ls /usr/local/cuda*/bin/nvcc 2>/dev/null; which tmux; df -h | grep -E 'overlay|workspace'"
```

Required:
- `nvcc` present (need CUDA devel tools to build pearl-gemm kernels)
- `tmux` present
- Enough writable disk for the build (see below)

#### Adapting to common RunPod quirks

These came up on multiple pods; handle them in-place rather than re-provisioning.

**nvcc installed but not in PATH.** Several RunPod CUDA images ship `nvcc` at
`/usr/local/cuda/bin/nvcc` without adding it to PATH, so `which nvcc` returns
nothing even though the toolchain is fully present. If you see this, add it
to `~/.bashrc` on the pod before running the bootstrap:

```bash
ssh ... 'grep -q "Pearl CUDA PATH" ~/.bashrc || cat >> ~/.bashrc <<"EOF"
# Pearl CUDA PATH
export PATH=/usr/local/cuda/bin:$PATH
EOF'
```

Also pass `export PATH=/usr/local/cuda/bin:$PATH` inline in the tmux send-keys
command (tmux's new shell may not source .bashrc fast enough for the curl|bash
that follows). If `nvcc` is genuinely missing from disk, re-provision with
`nvidia/cuda:12.9.0-devel-ubuntu22.04` (or 24.04, or 12.8/12.6 devel — any
`cuda:12.x-devel-*` works). Do not try to install CUDA toolkit on top of a
non-devel image; that fight is not worth winning.

**Small `/` overlay (<40 GB).** The bootstrap installs to `/root/pearl` and
its venv ends up at `/root/pearl/.venv` (~5 GB) with uv cache at
`/root/.cache/uv` (~14 GB during builds). On RunPod pods with a 20–30 GB
overlay the build will hit `No space left on device` mid-uv-sync. Redirect
both to `/workspace` (typically 250 GB+, either local NVMe or MooseFS —
both work) by exporting three env vars:

```bash
ssh ... 'grep -q "Pearl UV redirect" ~/.bashrc || cat >> ~/.bashrc <<"EOF"
# Pearl UV redirect (overlay is small, use /workspace)
export UV_CACHE_DIR=/workspace/uv-cache
export UV_PROJECT_ENVIRONMENT=/workspace/pearl-venv
export PEARL_VENV=/workspace/pearl-venv
EOF
mkdir -p /workspace/uv-cache'
```

- `UV_CACHE_DIR` puts the download/build cache on /workspace.
- `UV_PROJECT_ENVIRONMENT` tells `uv sync`/`uv run` to use a venv at that path
  instead of the repo-local `./.venv`.
- `PEARL_VENV` tells `env.sh` to look there for the bundled CUDA libs when
  setting `LD_LIBRARY_PATH` (without this, `import direct_miner` fails with
  `libc10.so: cannot open`).

Pass the same three exports inline in the tmux send-keys command alongside
the PATH fix — the new shell tmux spawns may not finish sourcing `.bashrc`
before the curl|bash starts.

**Missing tmux.** Some images don't ship it: `apt-get install -y tmux`
before Step 3.

### Step 3: Kick off the bootstrap

Two options. Prefer the first when possible.

**Option A: Run inside a tmux session on the pod (recommended)**

This lets the bootstrap run independently of your SSH connection. If SSH drops
during pearld sync (which takes 1-3 hours), the deploy continues.

```bash
ssh ... "tmux new-session -d -s deploy"
ssh ... "tmux send-keys -t deploy 'curl -fsSL https://raw.githubusercontent.com/fr00000/pearl/opt/direct-mining/miner/direct-miner/scripts/bootstrap_h100_prod.sh -o /tmp/bootstrap.sh && chmod +x /tmp/bootstrap.sh && PEARL_MINING_ADDR=\"<ADDRESS>\" bash /tmp/bootstrap.sh 2>&1 | tee /tmp/bootstrap.log' Enter"
```

Note: use the user's actual address; do not commit it anywhere. Construct the
command in-memory only.

**Option B: Stream output directly (only for quick tests)**

```bash
ssh ... "PEARL_MINING_ADDR='<ADDRESS>' bash <(curl -fsSL https://raw.githubusercontent.com/fr00000/pearl/opt/direct-mining/miner/direct-miner/scripts/bootstrap_h100_prod.sh) 2>&1"
```

Will block until done. Don't use this for real deploys; the 1-3 hour pearld
sync will kill your patience and the SSH session.

### Step 4: Monitor progress

After kicking off the bootstrap in Option A, poll progress:

```bash
# Bootstrap log tail
ssh ... "tail -30 /tmp/bootstrap.log"

# Current step (look for "===> " lines in the log)
ssh ... "grep '===> ' /tmp/bootstrap.log | tail -5"

# pearld sync status
ssh ... "~/pearl/bin/prlctl -u rpcuser -P rpcpass -s 127.0.0.1:44107 --notls getblockchaininfo 2>/dev/null | jq '{blocks, headers, peers: (.peers // 0)}'"
```

Expected timeline:
- Steps 1-5 (apt update, toolchain, clone, builds): 20-40 minutes total
- Steps 6-7 (pearld + gateway start): 1-2 minutes
- Step 8 (pearld sync from genesis): 1-3 hours (the long wait)
- Step 9 (launch miners): 1-2 minutes

Poll every 5-15 minutes once the build phase completes. Report progress to the
user every 30 minutes or at every step transition.

### Step 5: Verify successful deployment

When the bootstrap script prints "Bootstrap complete", verify:

```bash
# All miners running?
ssh ... "ls /tmp/direct-miner-pids/ && ps aux | grep direct-miner | grep -v grep | wc -l"

# Per-GPU rates after ~3 minutes of running
ssh ... "for f in /workspace/logs/gpu*-*.log; do echo \"=== \$(basename \$f) ===\"; grep 'DIRECT MINER' \$f | tail -1; done"

# GPU utilization
ssh ... "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"

# Any errors?
ssh ... "grep -iE 'error|exception|cuda error' /workspace/logs/gpu*-*.log | head -10"

# pearld + gateway healthy
ssh ... "pgrep pearld > /dev/null && echo 'pearld OK'; pgrep -f 'pearl-gateway start' > /dev/null && echo 'gateway OK'; ls -la /tmp/pearlgw.sock"
```

Acceptable state:
- One miner PID per GPU (matches `nvidia-smi --list-gpus` count)
- Each miner showing completion_rate 30-32 mm/s and tile_rate ~2.0-2.1M.
  The lower end of the range is expected when the network is small (low
  difficulty, frequent template churn); the high end matches well-warmed
  steady-state on a fully-loaded chain. Don't flag 31 mm/s as a failure.
- GPU utilization 85-100% on each GPU
- No errors in logs
- Gateway socket present, pearld and gateway processes alive

If any check fails, do NOT try aggressive fixes. Report to the user with
specifics.

### Step 6: Hand off

Report to the user:

```
Pod: <host>:<port>
Mining address: <ADDR>
GPUs: <N> × <NAME>
Status: <N>/<N> miners running
Per-GPU rate: ~<X>M tiles/sec
Aggregate: ~<NX>M tiles/sec
Bootstrap log: /tmp/bootstrap.log
Per-miner logs: /workspace/logs/gpu-*.log
Tmux session: tmux attach -t pearl

To check status later:
  ssh ... "ls /tmp/direct-miner-pids/ && for f in /workspace/logs/gpu*-*.log; do tail -1 \$f; done"
```

## Multi-pod deployment

If the user wants the same address on multiple pods, repeat steps 1-6 for each
pod. Each pod is independent — separate pearld sync, separate gateway, separate
miners. They all submit blocks to the same address; no coordination needed.

Run pods one at a time during the build phase (avoid contention on GitHub if
multiple pods cold-pull). The pearld sync can happen in parallel; that's
peer-network limited, not GitHub.

## Common failure modes

### `uv sync` fails partway

Usually a network blip. The bootstrap is idempotent — the `build_python` state
is only marked after `uv sync` succeeds. Re-run the bootstrap; it'll skip
completed steps and retry the failed one.

```bash
ssh ... "PEARL_MINING_ADDR='<ADDRESS>' bash /tmp/bootstrap.sh"
```

If it consistently fails on the same package, check `/workspace/logs/uv-sync.log`
on the pod and report to the user.

### pearld sync stuck at 0 peers after 5 minutes

The bootstrap script warns about this and prints fallback peer IPs. Edit
the pearld invocation to add `--addpeer=<IP>:44108` etc. Note: this
requires killing the running pearld and restarting it with the new args.
Coordinate with the user before doing this — it's a manual intervention.

### Miners die immediately on launch (Step 9)

Common causes:
- Gateway not yet serving templates (pearld sync incomplete)
- Wrong GPU detected (check `nvidia-smi`)
- Out-of-memory (rare with 80GB H100, possible with shared GPU)

Check the miner log for the specific error:

```bash
ssh ... "head -50 /workspace/logs/gpu0-*.log"
```

Don't try to fix with smaller mif or different shape unless the user explicitly
agrees — that diverges from the validated production config.

### SSH disconnects during long operations

Expected behavior if you started the bootstrap inside tmux on the pod
(Option A). The bootstrap continues running independently. Reconnect and
resume polling:

```bash
ssh ... "tmux attach -t deploy"  # to see what's happening
# or
ssh ... "tail -30 /tmp/bootstrap.log"
```

If you used Option B (direct stream), the bootstrap process likely died with
your SSH session. Re-run from scratch using Option A.

## What NOT to do

- **Never commit the user's mining address** to git, files in this repo, or
  logs. It belongs in env vars and `~/.pearl_mining_address` on the pod only.
- **Never ask for or store the user's seed phrase.** The address is sufficient.
  If the user offers a seed, decline politely and remind them only the address
  is needed.
- **Never push to the remote.** This is a deployment operation, not engineering.
  All commits should go through the user's existing dev workflow on their main
  machine.
- **Never modify the production shape** (m=8192, n=262144, k=8192, mif=4)
  without the user's explicit instruction.
- **Never SSH-exec a `pkill -9` against pearld or pearl-gateway** without the
  user's agreement. These are essential infrastructure on the pod.
- **Never launch more than one direct-miner per GPU.** The launcher script
  loops correctly; don't add extras.
- **Never try to "fix" the bootstrap script in-place on the pod** by editing
  `/tmp/bootstrap.sh`. If something's wrong with the script, report it; the
  user fixes it in the repo and you re-deploy.
- **Never re-run the bootstrap with a different mining address** without
  removing `~/.pearl_mining_address` first. The address is cached; changing it
  mid-run could cause confusion.

## When to stop and ask the user

- **Hardware mismatch**: Pod has non-Hopper GPUs or no GPUs
- **Missing nvcc**: Base image isn't CUDA devel
- **Persistent peer discovery failure**: pearld can't find peers after 5+ min
  even with fallback IPs configured
- **`uv sync` fails repeatedly** (3+ retries) on the same package
- **Miners die immediately and consistently** at step 9 across multiple GPUs
- **User-requested mining address differs from cached** `~/.pearl_mining_address`
- **GPU memory shows another process** consuming significant memory before
  miners launch (someone else is using the pod)
- **Any auth or permission errors** that suggest SSH access is restricted in
  unexpected ways

For all of these, stop the deploy and explain to the user what you found.
Don't try to push through; you'll likely make things worse.

## What you can do autonomously

- Run the bootstrap end-to-end on a verified Hopper pod with valid address
- Restart a dead miner via the bootstrap's idempotent re-run path
- Investigate logs to diagnose issues (read-only)
- Report aggregate rates and GPU utilization
- Verify pearld and gateway health
- Repeat the procedure across multiple pods (sequentially or in parallel within
  reason)

## Reference: key files in this repo

- `miner/direct-miner/scripts/bootstrap_h100_prod.sh` — the production bootstrap
  (what you'll execute on each pod)
- `miner/direct-miner/H100_SXM_VERIFY.md` — verified production config
- `miner/direct-miner/H200_SWEEP_RESULTS.md` — sweep data including stress
  addendum (informs the chosen shape)
- `miner/direct-miner/PHASE_C_RESULTS.md` — the per-GPU optimization journey
- `miner/direct-miner/scripts/launch_h200_production.sh` — alternate launcher
  if running outside the bootstrap context (e.g., manual deploy on a pre-built
  pod)

## Expected output for a successful deployment

When everything works, the user should see (after ~2-4 hours):

```
=> Pod xyz.runpod.io:22034 deployed
   GPUs: 4 × H100 80GB HBM3
   Miners: 4/4 running
   Per-GPU rate: 32.1 mm/s, 2,103,706 tiles/sec
   Aggregate: 128.4 mm/s, 8,414,824 tiles/sec
   Mining address: prl1p9lq...
   Block height: 52,180 (synced)
   Status: healthy, no errors
   Logs: /workspace/logs/
   PIDs: /tmp/direct-miner-pids/
   Tmux: tmux attach -t pearl
```

If you can deliver this, the deploy is done.
