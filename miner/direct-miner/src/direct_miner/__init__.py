"""
Direct synthetic mining for Pearl protocol.

Phase A: minimal viable miner that calls pearl_gemm_noisy() directly with
synthetic A and B matrices, bypassing vLLM. Uses CUDA events for accurate
completion-rate tracking and bounded in-flight work to prevent queue
explosion.

Purpose: answer one question — can synthetic mining produce an accepted
block? No optimization, no caching beyond static B, no multi-stream.
"""

__version__ = "0.1.0"
