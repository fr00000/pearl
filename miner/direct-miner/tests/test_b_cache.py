import torch

from direct_miner.b_cache import BSideArtifacts, BSideCache


def _artifacts(value: int = 0) -> BSideArtifacts:
    tensor = torch.tensor([value], dtype=torch.uint8)
    matrix = torch.empty((1, 1), dtype=torch.int8)
    return BSideArtifacts(
        B_tensor_hash=tensor,
        commitment_hash_B=tensor,
        EBR=matrix,
        EBR_fp16=torch.empty((1, 1), dtype=torch.float16),
        EBL_R_major=matrix,
        EBL_K_major=matrix,
        BpEB=matrix,
    )


def test_evict_if_mismatch_drops_old_artifacts():
    cache = BSideCache()
    cache.put(b"old-key", _artifacts())

    assert cache.has_different_key(b"new-key")
    assert cache.evict_if_mismatch(b"new-key") is True

    assert cache.get(b"old-key") is None
    assert cache.stats()["invalidations"] == 1
    assert cache.stats()["current_key_prefix"] is None


def test_evict_if_mismatch_keeps_matching_artifacts():
    cache = BSideCache()
    artifacts = _artifacts()
    cache.put(b"same-key", artifacts)

    assert cache.has_different_key(b"same-key") is False
    assert cache.evict_if_mismatch(b"same-key") is False

    assert cache.get(b"same-key") is artifacts
    assert cache.stats()["invalidations"] == 0
