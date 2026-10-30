"""Fixed seed list (DESIGN §9). All randomness must derive from these."""
import hashlib
import numpy as np

SEEDS = [20261026, 20261027, 20261028, 20261029, 20261030,
         20261101]   # DESIGN_P2 §6 (frozen 2026-10-28): test set v2, seed name derive(20261101, 'testset_v2')
SPLIT_SEED, SCREEN_SEED, TEST_SEED, VALID_SEED, REPLICATE_SEED = SEEDS[:5]
TEST_V2_SEED = SEEDS[5]


def derive(seed: int, *keys) -> int:
    """Deterministic child seed from a base seed and arbitrary keys (e.g. cell id, locus index)."""
    assert seed in SEEDS, f"seed {seed} not in the fixed seed list"
    h = hashlib.sha256(repr((seed,) + tuple(keys)).encode()).digest()
    return int.from_bytes(h[:8], "little")


def rng(seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng(derive(seed, *keys))
