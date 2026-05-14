"""Synthetic matrix generation for mining.

Protocol validates matmul correctness and hash difficulty but does not
verify the source of A or that B contains specific weights. Random int8
matrices with valid scale factors are fully protocol-compliant.

Forever-cacheable: the raw B tensor and B_scales (never change).
Never cache: anything derived via hash_key (changes per template).
"""

from typing import Optional

import torch


def make_synthetic_a(
    m: int,
    k: int,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate fresh synthetic A and scales.

    Pearl mining layers use int7 quantization ([-64, 63] range).
    Generate within that range for compatibility with the kernel.

    A must be fresh per iteration because noise factors depend on
    commitment_hash_A which is a hash of A's bytes.
    """
    A = torch.randint(
        -64, 64,
        (m, k),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    # Scales must be fp32 per pearl_gemm_interface docstring
    # (A_scales (m, fp32), B_scales (n, fp32)).
    # Realistic magnitude is ~1/128 (matches gemm_tensor_generator).
    A_scales = (
        torch.rand(m, device=device, dtype=torch.float32, generator=generator)
        / 128.0
    )
    return A, A_scales


def make_synthetic_a_into(
    A: torch.Tensor,
    A_scales: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> None:
    """In-place A generation into pre-allocated buffers.

    Same value semantics as make_synthetic_a (int7 range, fp32 scales
    in [0, 1/128]) but reuses caller-owned buffers — required for the
    Phase C slot pool which pre-allocates A per max_in_flight slot.

    A must be (m, k) int8 and A_scales must be (m,) fp32. Verified at
    runtime; raises on mismatch.
    """
    if A.dtype != torch.int8:
        raise ValueError(f"A must be int8, got {A.dtype}")
    if A_scales.dtype != torch.float32:
        raise ValueError(f"A_scales must be fp32, got {A_scales.dtype}")

    torch.randint(
        -64, 64,
        size=A.shape,
        generator=generator,
        out=A,
    )
    A_scales.uniform_(0.0, 1.0 / 128.0, generator=generator)


def make_synthetic_b(
    n: int,
    k: int,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate B and B_scales — called once per session.

    B itself is forever-cacheable (the int8 bytes never change).
    Note: B-derived artifacts via hash_key (Merkle root, commitment,
    noise factors) ARE NOT cacheable across templates. Phase A doesn't
    cache anything; this distinction matters for future phases only.
    """
    B = torch.randint(
        -64, 64,
        (n, k),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    # Scales must be fp32 per pearl_gemm_interface docstring.
    B_scales = (
        torch.rand(n, device=device, dtype=torch.float32, generator=generator)
        / 128.0
    )
    return B, B_scales


class FixedBPool:
    """Holds the session's static B matrix and scales.

    B itself is the only cross-iteration forever-cacheable state in
    Phase A. Everything else (B Merkle root, commitment_hash_B, BpEB,
    B-side noise factors) is regenerated on every mining call because
    they depend on hash_key, which depends on the current block header.
    """

    def __init__(
        self,
        n: int,
        k: int,
        device: torch.device | str = "cuda",
        seed: int | None = None,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        self.B, self.B_scales = make_synthetic_b(
            n, k, device=device, generator=generator
        )
        self.n = n
        self.k = k
        self.device = device
