"""SharePro_coloc (zhwm/SharePro_coloc @ 059caab, v5.0.0) run in-process.

The official module env/src/SharePro_coloc/src/SharePro/sharepro_coloc.py is imported unchanged (by file path);
`run_sharepro` reproduces its `main()` step by step without the file I/O (same bhat transform, N = max N,
varb = max bhat^2, K = 10, sigma = 1e-5, maxite = 100, eps = 0.1, ubound = 1e6, cthres = 0.95, pthres = 0.5,
and the documented automatic fallback to K = 1 when the K = 10 run does not converge). Equivalence with the CLI
on the shipped example is tested in tests/baselines/test_sharepro_equiv.py.
"""
from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SHAREPRO_PY = PROJECT / "env" / "src" / "SharePro_coloc" / "src" / "SharePro" / "sharepro_coloc.py"
SHAREPRO_COMMIT = "059caab"


@lru_cache(maxsize=1)
def _module():
    import warnings

    spec = importlib.util.spec_from_file_location("sharepro_coloc_059caab", SHAREPRO_PY)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # '\s+' in the upstream file (harmless)
        spec.loader.exec_module(mod)
    logging.getLogger().setLevel(max(logging.getLogger().level, logging.WARNING))
    return mod


def run_sharepro(beta_list, se_list, n_list, ld, K=10, sigma=1e-5, maxite=100, eps=0.1, ubound=1e6,
                 cthres=0.95, pthres=0.5) -> dict:
    """Returns eff (dict k -> variant indices), eff_share, eff_gamma, model, converged flags, K used."""
    m = _module()
    bhatlst = [m.get_bhat(np.asarray(b, float), np.asarray(s, float), np.asarray(n, float))
               for b, s, n in zip(beta_list, se_list, n_list)]
    ld = np.asarray(ld, dtype=np.float64)
    assert not np.isnan(ld).any(), "The LD matrix contains NaN values"
    assert not np.isnan(bhatlst).any(), "summary statistics contain NaN values"
    Nlst = [float(np.max(n)) for n in n_list]
    varb = [float((b ** 2).max()) for b in bhatlst]
    p = len(bhatlst[0])
    model = m.SharePro(p, K, len(bhatlst), sigma, Nlst, varb)
    mc = model.train(bhatlst, ld, maxite, eps, ubound)
    k_used, conv_first = K, bool(mc)
    if mc:
        eff, eff_gamma, eff_share = model.get_effect(ld, cthres, pthres)
    else:
        model = m.SharePro(p, 1, len(bhatlst), sigma, Nlst, varb)
        mc = model.train(bhatlst, ld, maxite, eps, ubound)
        eff, eff_gamma, eff_share = model.get_effect(ld, cthres, pthres)
        k_used = 1
    return dict(eff=eff, eff_gamma=eff_gamma, eff_share=eff_share, model=model, K_used=k_used,
                converged_first=conv_first, converged_final=bool(mc))


def per_trait_pip(model) -> np.ndarray:
    """(num_traits, p): PIP_tj = 1 - prod_k (1 - gamma_jk * delta_t,jk) (variant j causal for trait t)."""
    G = model.gamma
    return np.array([1.0 - np.prod(1.0 - G * sr.delta, axis=1) for sr in model.SR])


def per_trait_effect_prob(model) -> np.ndarray:
    """(num_traits, K): P(effect k active in trait t) = sum_j gamma_jk delta_t,jk (reported in extra only)."""
    G = model.gamma
    return np.array([(G * sr.delta).sum(axis=0) for sr in model.SR])
