# Phase A Results — Direct Synthetic Mining

## Configuration
- Hardware: NVIDIA H100 80GB HBM3 (RunPod, PCIe pod)
- Driver / CUDA: 580.126.09 / 13.0
- Shapes: m=4096, n=8192, k=8192 (outer_tiles_per_matmul = 1024)
- max_in_flight: 2
- Run duration: 3631.9 s (~60 min 32 s)
- Run window: 2026-05-13 21:30:51 → 22:30:51 UTC
- Pearl-gateway transport: UDS (`/tmp/pearlgw.sock`)
- Mining settings: defaults (noise_rank=128, tile_size_m=128, tile_size_n=256, tile_size_k=128)
- `MINER_DEBUG=true`
- Log: `/workspace/direct-miner-phase-a.log`

## Throughput (measured)
- Total completed matmuls: **3,287,105**
- Total launched matmuls:  **3,287,105** (final, after drain)
- Mean completion_rate (full run): **905.1 matmuls/s**
- Mean launch_rate    (full run): **905.1 matmuls/s** (matches; backpressure cap kept the queue tight at ≤2)
- Steady-state instant completion_rate over the run: 900–916 / s (samples in log)
- Tile rate: **926,775 tiles/s** (= 905.1 × 1024)
- Median GPU utilization during run: **90%** (range 87–91%, sampled every 30 s for 30 min — `/workspace/gpu-util-during-run.log`)
- launch − completed delta during steady state: 2 (= max_in_flight); drained to 0 cleanly on SIGINT
- Errors logged during 60-min run: **0**

## Comparison to vLLM baseline
- vLLM baseline (per spec, observed in window 3 prior to this run): ~124 matmuls/s
- direct-miner: 905.1 matmuls/s
- Ratio: **~7.3× more matmul launches per second**
- Caveat: vLLM's matmuls are at the model's natural shapes (varies); direct-miner uses fixed 4096×8192×8192. Tile-rate comparison would require knowing vLLM's effective tile-rate.

## Block production (the Phase A question)
- Initial wallet balance:  **0**
- Final wallet balance:    **0** (`prlctl getbalance`)
- Gateway submission attempts during run: **0**
  - Gateway window 2 (`pearl:2`) showed only `Updating block template` / `Template refreshed` messages.
  - No `Building plain proof`, `Generating ZK proof`, `Received submission`, `Block accepted`, `rejected`, or any error/warn lines.
  - 39 template refreshes during the 60-min window (one every ~93 s on average).
- Pearld blocks added to chain during run: 5 (heights 51578 → 51582)
  - All preceded by `SYNC: Syncing to block height N from peer …`, indicating they were received from network peers, **not produced locally**.
- Network observed block interval during run: ~12 min average (5 blocks / 60 min).

## Anomalies / notes
- Initial implementation used `float16` scales; kernel rejected with `expected dtype: Float but got this dtype: Half`. Fixed in `synthetic_data.py` to use `float32` with magnitude ~1/128 (matching `pearl_gemm/testing/gemm_tensor_generator.py`). Smoke + production runs after the fix produced no errors.
- `in_flight` value: held at 2 (the cap) for the entirety of steady state; momentary drops to 1 at log boundaries are normal. Confirms `wait_for_slot` backpressure is functioning.
- One `WARNING - Pool size exceeded, self._pool_size=128` was logged in the *first* (broken) smoke run; not observed in the post-fix smoke or the production run.

## Phase A conclusion
The minimum viable direct-synthetic-mining loop ran cleanly for 60 minutes at 905 matmuls/s (~7.3× the launch rate of the vLLM-driven baseline) at sustained ~90% GPU. **No blocks were produced and zero candidate submissions were sent to the gateway**, despite ~3.28 M matmuls / ~9.3 × 10⁸ tile evaluations.

Two consistent interpretations:
1. **Variance-consistent.** Network block interval was ~12 min over the same window, implying total network tile-rate is many multiples of ours; finding zero in 60 min is not by itself disqualifying.
2. **Premise-fragile.** The kernel never raised a "winning tile" signal across 9.3 × 10⁸ tiles. Without instrumenting tile-hash-vs-target margins, we can't yet tell if our hash distribution beats the target at all, or only with much lower probability than expected.

Throughput is healthy; the Phase A question ("does direct synthetic mining produce an accepted block?") is **not yet answered** — observed: 0 blocks in 1 hr. A longer run (or chain-side instrumentation) is needed before declaring the thesis validated or invalidated.

## Recommended next step
Two options, in priority order:

1. **Run longer before any optimization.** Extend the same direct-miner invocation to 12–24 hours and re-measure. If still 0 blocks at 24 h with ~7.5 × 10¹⁰ tiles, the gap to the network is large enough to reconsider the premise. If it produces any block, Phase A is validated and Phase B (B-side caching, the spec's biggest expected gain) becomes the right next investment.
2. **Add a tile-hash-margin counter.** Before extended runs, instrument the kernel (or wrap it) to emit how close our best tile-hash got to the pow_target each call. This converts "0 blocks" from a binary into a distribution we can reason about, and would catch any silent kernel/arg mismatch that the protocol-level "no submission" path can't.

Recommendation: do (1) first — it's free engineering effort, the loop is already running cleanly, and it gives a real answer to the headline question rather than a derived one.
