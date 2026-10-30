"""The L0* reference run must use exactly the prior recorded with the set (task p4.famB_l0star_ref)."""
import json
from pathlib import Path

import numpy as np
import pytest

from famB.l0star_ref import check_prior_matches_set
from famB.priors import sim_prior_from_l0star

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "data/processed/sets/l0star_v2.stats.json"


@pytest.mark.skipif(not STATS.exists(), reason="l0star_v2 not available")
def test_prior_matches_l0star_v2(tmp_path):
    pr = sim_prior_from_l0star()
    st = check_prior_matches_set(STATS, pr)
    assert st["eta"] == 1e-3
    # negative control: a perturbed mixture weight must be detected
    bad = json.loads(STATS.read_text())
    w = np.asarray(bad["lambda_mix"]["2"]["w"])
    w[3] += 0.05
    bad["lambda_mix"]["2"]["w"] = w.tolist()
    p = tmp_path / "bad.stats.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        check_prior_matches_set(p, pr)
