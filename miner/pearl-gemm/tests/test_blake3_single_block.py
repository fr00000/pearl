from blake3 import blake3
import numpy as np
import pytest
import torch

from pearl_gemm.test_components import blake3_single_block_keyed


def _u32_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().numpy().astype("<u4", copy=False).tobytes()


@pytest.mark.parametrize("seed", [0, 1, 1234, 987654])
def test_scheduled_single_block_keyed_blake3_matches_python(seed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(seed)
    block_np = rng.integers(0, 2**32, size=16, dtype=np.uint32)
    key_np = rng.integers(0, 2**32, size=8, dtype=np.uint32)

    block = torch.from_numpy(block_np).to(device="cuda", dtype=torch.uint32)
    key = torch.from_numpy(key_np).to(device="cuda", dtype=torch.uint32)

    cuda_hash = blake3_single_block_keyed(block, key)
    expected = blake3(block_np.astype("<u4", copy=False).tobytes(),
                      key=key_np.astype("<u4", copy=False).tobytes()).digest()

    assert _u32_tensor_to_bytes(cuda_hash) == expected
