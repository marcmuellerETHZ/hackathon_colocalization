"""Tier-1 baseline wrappers (DESIGN §4.1, adapters §4.3, decisions D4). INTERFACES §4 Method API.

Each method: `fit_batch(loci) -> list[dict]` with the INTERFACES §3 columns (call left None for the harness).
All choices not fixed by DESIGN are documented in src/baselines/README.md.
"""
from __future__ import annotations

import json
import math
import resource
import time
import traceback

import numpy as np

from eval import adapters as EA  # the single §4.3 adapter implementation (D6(f))

from .abf import coloc_abf, coloc_abf_grid
from .locus import dense_ld, n_ref_of, rho_hat, scalar_n
from .rworker import RWorkerDied, RWorkerError, RWorkerTimeout, get_worker

NAN = float("nan")

# Proposed per-locus timeouts (s), provisional until the 200-locus validation pilot (DESIGN §4.3). Basis: max
# observed runtime at p <= 5000 (in-sample and n_ref = 150 mismatched LD; src/baselines/benchmarks/runtime_M3.tsv)
# x ~4 for p up to 10k (runtime ~ p^2) x >= 3 safety margin.
DEFAULT_TIMEOUT = {
    "coloc_abf": 60, "lead_snp": 60, "sharepro": 300, "prop_coloc": 300, "coloc_prop_test": 300,
    "colocboost_m1": 120, "coloc_susie": 600, "coloc_susie_mm": 900, "colocboost": 900, "mvsusie": 1800,
}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def _as_list(x) -> list:
    """jsonlite auto_unbox turns length-1 vectors into scalars; normalise to a list (None -> [])."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _clean(o):
    """Replace non-finite floats by None recursively (strict JSON)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _check_inputs(locus) -> None:
    """Reject non-finite or inconsistent inputs (recorded as status 'failed', never silently used)."""
    p = len(locus.z1)
    for k in ("z1", "z2", "beta1", "se1", "beta2", "se2", "n1", "n2"):
        a = np.asarray(getattr(locus, k), dtype=np.float64)
        if a.shape != (p,):
            raise ValueError(f"{k} has shape {a.shape}, expected ({p},)")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{k} contains non-finite values")
    if np.any(np.asarray(locus.se1) <= 0) or np.any(np.asarray(locus.se2) <= 0):
        raise ValueError("non-positive standard errors")
    if p < 2:
        raise ValueError("fewer than 2 variants")


class BaselineMethod:
    name = "base"
    needs_ld = True  # read by eval.harness (False: the locus is read without R / X_ref)
    r_lib: str | None = None  # R library setting, None for pure-Python methods

    def __init__(self, timeout: float | None = None):
        self.timeout = float(timeout if timeout is not None else DEFAULT_TIMEOUT.get(self.name, 600))

    # ---------------------------------------------------------------- API
    def fit_batch(self, loci) -> list[dict]:
        return [self.fit_one(locus) for locus in loci]

    def fit_one(self, locus) -> dict:
        t0 = time.perf_counter()
        extra: dict = {}
        out = dict(s=NAN, P_S=NAN, P_D=NAN, P_W=NAN)
        try:
            _check_inputs(locus)
            res = self._fit(locus)
            status = res.pop("status", "ok")
            extra = res.pop("extra", {})
            out.update(res)
        except RWorkerTimeout:
            status = "timeout"
            extra = {"error": f"timeout after {self.timeout:g} s"}
        except (RWorkerError, RWorkerDied) as e:
            status = "failed"
            extra = {"error": str(e)[:1000]}
        except Exception as e:  # noqa: BLE001 - every failure is recorded, never swallowed silently
            status = "failed"
            extra = {"error": f"{type(e).__name__}: {e}"[:1000], "traceback": traceback.format_exc()[-2000:]}
        if status in ("failed", "timeout"):
            out = dict(s=NAN, P_S=NAN, P_D=NAN, P_W=NAN)
        extra["mem_mb"] = self._mem_mb()
        row = dict(locus_id=str(locus.locus_id), method=self.name, s=out["s"], P_S=out["P_S"], P_D=out["P_D"],
                   P_W=out["P_W"], E1=NAN, E2=NAN, call=None, status=status,
                   runtime_s=time.perf_counter() - t0,
                   extra=json.dumps(_clean(extra), default=_json_default, allow_nan=False))
        return row

    def _fit(self, locus) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _mem_mb(self) -> dict:
        """Peak RSS so far (MB) of this Python process and of its current R worker (VmHWM; running maxima, so
        the per-method maximum over a run's rows is the per-worker peak used to set worker counts)."""
        out = {"py_peak": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
        if self.r_lib is not None:
            out["r_peak"] = round(get_worker(self.r_lib).peak_rss_kb() / 1024, 1)
        return out

    def _r(self, fn, args, arrays):
        return get_worker(self.r_lib).call(fn, args, arrays, timeout=self.timeout)


# =========================================================================================== (1) coloc.abf
class ColocABF(BaselineMethod):
    """coloc.abf via the validated local ABF (src/baselines/abf.py). Extra carries the full W x p12 grid."""

    name = "coloc_abf"
    needs_ld = False  # eval.harness then skips the LD read (coloc.abf uses no LD)

    def __init__(self, W: float = 0.15, p12: float = 5e-6, W_grid=(0.1, 0.15, 0.2), p12_grid=(5e-6, 1e-5),
                 timeout=None, name: str | None = None):
        self.name = name or "coloc_abf"
        super().__init__(timeout if timeout is not None else DEFAULT_TIMEOUT["coloc_abf"])
        self.W, self.p12 = W, p12
        self.W_grid, self.p12_grid = tuple(W_grid), tuple(p12_grid)

    def _fit(self, locus):
        grid = coloc_abf_grid(locus.z1, locus.se1, locus.z2, locus.se2, self.W_grid, self.p12_grid)
        key = f"W={self.W:g},p12={self.p12:g}"
        pp = grid[key] if key in grid else coloc_abf(locus.z1, locus.se1, locus.z2, locus.se2, W=self.W,
                                                     p12=self.p12)
        res = EA.coloc_abf(pp)
        res["extra"] = {"W": self.W, "p12": self.p12, "PP": list(pp), "grid": {k: list(v) for k, v in grid.items()}}
        return res


# =========================================================================================== (2,3) coloc-SuSiE
class ColocSusie(BaselineMethod):
    """susie_rss(z, R, n, L = 10) per trait + coloc.bf_bf on CS effects (as coloc.susie), p12 = 5e-6.

    mismatch=False: susieR 0.14.2 (main library), the standard pipeline.
    mismatch=True : susieR 0.16.6 (env/rlib_susie016, separate R process), R_finite = n_ref, R_mismatch = "eb_mix".
    Falls back to coloc.abf (W = 0.15, p12 = 5e-6) when either trait has no CS.
    """

    name = "coloc_susie"

    def __init__(self, mismatch: bool = False, L: int = 10, p12: float = 5e-6, R_mismatch: str = "eb_mix",
                 max_iter: int | None = None, timeout=None):
        self.mismatch = mismatch
        self.name = "coloc_susie_mm" if mismatch else "coloc_susie"
        self.r_lib = "susie016" if mismatch else "main"
        super().__init__(timeout)
        self.L, self.p12, self.R_mismatch = L, p12, R_mismatch
        self.max_iter = max_iter  # None = susie_rss default (0.14.2: 100; 0.16.6: its own default)

    def _fit(self, locus):
        R = dense_ld(locus)
        args = dict(n1=scalar_n(locus.n1), n2=scalar_n(locus.n2), L=self.L, p1=1e-4, p2=1e-4, p12=self.p12,
                    mismatch=self.mismatch, n_ref=n_ref_of(locus), R_mismatch=self.R_mismatch,
                    max_iter=self.max_iter)
        resp = self._r("susie_coloc", args, {"z1": locus.z1, "z2": locus.z2, "R": R})
        r = resp["result"]
        extra = {"susieR": r.get("susieR"), "t1": r.get("t1"), "t2": r.get("t2"),
                 "r_warnings": resp.get("warnings"), "r_elapsed": resp.get("elapsed")}
        expected = "0.16.6" if self.mismatch else "0.14.2"
        if r.get("susieR") != expected:
            raise RuntimeError(f"wrong susieR version {r.get('susieR')} (expected {expected})")
        conv = bool(r["t1"]["converged"]) and bool(r["t2"]["converged"])
        pairs = r.get("pairs")
        PP = None
        if pairs is not None:
            PP = np.asarray(pairs["PP"], dtype=np.float64).reshape(-1, 5)
            extra["pairs"] = {"PP": PP.tolist(), "idx1": _as_list(pairs["idx1"]), "idx2": _as_list(pairs["idx2"]),
                              "hit1": _as_list(pairs["hit1"]), "hit2": _as_list(pairs["hit2"])}
        fallback = PP is None or not np.all(np.isfinite(PP), axis=1).any()
        fb_pp = None
        if fallback:  # either trait has no CS (or all CS pairs trimmed): coloc.abf, W = 0.15 (coloc default)
            fb_pp = coloc_abf(locus.z1, locus.se1, locus.z2, locus.se2, W=0.15, p12=self.p12)
            extra["fallback_PP"] = list(fb_pp)
        extra["fallback"] = bool(fallback)
        res = EA.coloc_susie(PP, fallback_pp=fb_pp)  # drops non-finite CS-pair rows
        res["extra"] = extra
        res["status"] = "ok" if conv else "nonconverged"
        return res


class ColocSusieMM(ColocSusie):
    """Mismatch-aware coloc-SuSiE (susieR 0.16.6, R_finite = n_ref, R_mismatch = "eb_mix"); class-level name for
    the harness (MethodSpec reads `name` without instantiating)."""

    name = "coloc_susie_mm"

    def __init__(self, **kw):
        kw.pop("mismatch", None)
        super().__init__(mismatch=True, **kw)


# =========================================================================================== (4) SharePro
class SharePro(BaselineMethod):
    """SharePro_coloc 059caab default run (automatic K = 1 fallback = documented mismatch-robust mode, D4).

    P_S = 1 - prod_k (1 - share_k) over reported effect groups (D4).
    P(trait-t signal) = 1 - prod_k (1 - sum_j gamma_jk delta_t,jk) over all K effects: SharePro's per-effect signal
    probability (D6(f), the DESIGN §4.3 "if reported" clause). The PIP-based fallback 1 - prod_j (1 - PIP_tj),
    PIP_tj = 1 - prod_k (1 - gamma_jk delta_t,jk), is kept in extra.P_sig_pip (with its P_D in extra.P_D_pip).
    """

    name = "sharepro"

    def __init__(self, K: int = 10, sigma: float = 1e-5, timeout=None):
        super().__init__(timeout)
        self.K, self.sigma = K, sigma

    def _fit(self, locus):
        from .sharepro import per_trait_effect_prob, per_trait_pip, run_sharepro

        R = dense_ld(locus)
        out = run_sharepro([locus.beta1, locus.beta2], [locus.se1, locus.se2], [locus.n1, locus.n2], R,
                           K=self.K, sigma=self.sigma)
        shares = [float(v) for v in out["eff_share"].values()]
        eff_prob = per_trait_effect_prob(out["model"])  # (2, K): P(effect k active in trait t)
        res = EA.sharepro(shares, eff_prob[0], eff_prob[1])
        sig = [EA.signal_prob(eff_prob[t]) for t in range(2)]
        pip = per_trait_pip(out["model"])  # DESIGN §4.3 PIP fallback, reported only
        sig_pip = [EA.signal_prob(pip[t]) for t in range(2)]
        res_pip = EA.locus_prob_with_signals(EA.signal_prob(shares), pips1=pip[0], pips2=pip[1])
        res["extra"] = {"share": shares, "n_groups": len(shares), "K_used": out["K_used"],
                        "converged_first": out["converged_first"], "converged_final": out["converged_final"],
                        "P_sig": sig, "P_eff": [[round(float(x), 6) for x in eff_prob[t]] for t in range(2)],
                        "P_sig_pip": sig_pip, "P_D_pip": res_pip["P_D"],
                        "max_pip": [float(pip[0].max()), float(pip[1].max())],
                        "groups": {str(k): [int(i) for i in v][:20] for k, v in out["eff"].items()}}
        res["status"] = "ok" if out["converged_final"] else "nonconverged"
        return res


# =========================================================================================== (5) ColocBoost
class ColocBoost(BaselineMethod):
    """colocboost 1.0.10, pairwise, LD given, residual_correlation = [[1, rho_hat], [rho_hat, 1]].

    M=None: default boosting; M=1: the documented LD-mismatch mode ("one iteration with LD", LD_Free vignette).
    P_S = 1 - prod over CoS (1 - cos_npc). P(trait-t signal) = 1 if trait t is in a CoS or has a trait-specific
    uCoS (output_level = 2), else 0 (colocboost reports no per-trait probability).
    """

    name = "colocboost"
    r_lib = "main"

    def __init__(self, M: int | None = None, timeout=None):
        self.M = M
        self.name = "colocboost" if M is None else f"colocboost_m{M}"
        super().__init__(timeout)

    def _fit(self, locus):
        R = dense_ld(locus)
        rho = float(np.clip(rho_hat(locus), -0.99, 0.99))
        args = dict(n1=scalar_n(locus.n1), n2=scalar_n(locus.n2), rho=rho, M=self.M)
        resp = self._r("colocboost", args, {"z1": locus.z1, "z2": locus.z2, "R": R})
        r = resp["result"]
        npc = [float(x) for x in _as_list(r.get("cos_npc"))]
        in_cos = set()
        for o in _as_list(r.get("cos_outcomes")):
            for tok in str(o).replace(" ", "").split(";"):
                if tok.startswith("Y"):
                    in_cos.add(int(tok[1:]))
        ucos = set(int(x) for x in _as_list(r.get("ucos_outcomes")))
        sig = [1.0 if (t in in_cos or t in ucos) else 0.0 for t in (1, 2)]
        res = EA.colocboost(npc, sig[0], sig[1])
        res["extra"] = {"M": self.M, "rho_used": rho, "n_cos": r.get("n_cos"), "cos_npc": npc,
                        "min_npc_outcome": r.get("min_npc_outcome"), "cos_outcomes": r.get("cos_outcomes"),
                        "top_idx": r.get("top_idx"), "ucos_outcomes": sorted(ucos), "P_sig": sig,
                        "max_vcp": r.get("max_vcp"), "cb_patch_uses": r.get("cb_patch_uses"),
                        "r_warnings": resp.get("warnings"),
                        "r_elapsed": resp.get("elapsed")}
        return res


class ColocBoostM1(ColocBoost):
    """ColocBoost in its documented LD-mismatch mode (M = 1); class-level name for the harness."""

    name = "colocboost_m1"

    def __init__(self, **kw):
        kw.pop("M", None)
        super().__init__(M=1, **kw)


# =========================================================================================== (6) mvSuSiE-RSS
class MvSusie(BaselineMethod):
    """mvsusieR 0.3.0 mvsusie_rss with create_mixture_prior(R = 2) (singletons + shared/heterogeneous canonical),
    residual_variance = [[1, rho_hat], [rho_hat, 1]] fixed, L = 10, prior scale estimated.

    Z: per-trait PVE-adjusted z (sqrt((N_t - 1)/(z^2 + N_t - 2)) z, what mvsusie_rss does internally for a common
    N), so that unequal N1, N2 are handled; then mvsusie_rss is called without N.
    Adapter: for effects l that form a CS (others q = 0), q_lt from single_effect_lfsr[l, t], plugged into the
    Family A inclusion-exclusion formulas (eval.adapters.mvsusie). Decision P3-2: rule "hard" (q = 1{lfsr < 0.05},
    DESIGN §4.3, pre-registered) is PRIMARY; rule "soft" (q = 1 - lfsr, D4) is the sensitivity analysis, reported
    in extra["soft"] (the other rule's scores are always in extra, so one run serves both).
    """

    name = "mvsusie"
    r_lib = "mvsusie"

    def __init__(self, L: int = 10, rule: str = "hard", timeout=None):
        super().__init__(timeout)
        if rule not in ("hard", "soft"):
            raise ValueError(f"rule must be 'hard' or 'soft', not {rule!r}")
        self.L, self.rule = L, rule

    @staticmethod
    def adjust_z(z, N):
        z = np.asarray(z, dtype=np.float64)
        return np.sqrt((N - 1.0) / (z * z + N - 2.0)) * z

    def _fit(self, locus):
        R = dense_ld(locus)
        rho = float(np.clip(rho_hat(locus), -0.95, 0.95))
        Z = np.column_stack([self.adjust_z(locus.z1, scalar_n(locus.n1)), self.adjust_z(locus.z2, scalar_n(locus.n2))])
        V = np.array([[1.0, rho], [rho, 1.0]])
        resp = self._r("mvsusie", dict(L=self.L), {"Z": Z, "R": R, "V": V})
        r = resp["result"]
        if r.get("mvsusieR") != "0.3.0":
            raise RuntimeError(f"wrong mvsusieR version {r.get('mvsusieR')}")
        cs = [int(i) for i in _as_list(r.get("cs_index"))]
        lf_raw = np.asarray(r["single_effect_lfsr"] if r.get("single_effect_lfsr") is not None else [],
                            dtype=np.float64)
        if not cs and lf_raw.size < 2:
            # all prior variances estimated 0: mvsusie returns a scalar NA lfsr and no CS -> no active effect
            lf = np.zeros((0, 2))
        else:
            lf = lf_raw.reshape(-1, 2)
        by_rule = {r_: EA.mvsusie(lf, cs, rule=r_) for r_ in ("hard", "soft")}
        res = dict(by_rule[self.rule])
        res["extra"] = {"cs_index": cs, "single_effect_lfsr": lf[[c - 1 for c in cs]].tolist() if cs else [],
                        "rule": self.rule,
                        "soft": {k: by_rule["soft"][k] for k in ("s", "P_S", "P_D", "P_W")},
                        "hard_lfsr005": {k: by_rule["hard"][k] for k in ("s", "P_S", "P_D", "P_W")},
                        "rho_used": rho, "converged": r.get("converged"), "niter": r.get("niter"),
                        "r_warnings": resp.get("warnings"), "r_elapsed": resp.get("elapsed")}
        res["status"] = "ok" if r.get("converged") else "nonconverged"
        return res


# =========================================================================================== (7) proportional tests
class PropColoc(BaselineMethod):
    """prop.coloc 2.0.0 (Patel & Burgess): n = c(N1, N2), tau = rho_hat, alpha = 0.05 (only p_cond depends on it).

    s = p_full (p-value of the proportional = shared null; high = consistent with sharing). P_* undefined (NaN).
    """

    name = "prop_coloc"
    r_lib = "main"

    def _fit(self, locus):
        R = dense_ld(locus)
        tau = float(np.clip(rho_hat(locus), -0.99, 0.99))
        args = dict(n1=scalar_n(locus.n1), n2=scalar_n(locus.n2), tau=tau, alpha=0.05)
        resp = self._r("propcoloc", args, {"b1": locus.beta1, "se1": locus.se1, "b2": locus.beta2,
                                           "se2": locus.se2, "R": R})
        r = resp["result"]
        if r.get("p_full") is None:
            raise RuntimeError("prop.coloc returned no p_full")
        extra = {k: r.get(k) for k in ("p_full", "eta_full", "p_eta", "p_naive", "p_cond", "eta_cond",
                                       "convergence", "top")}
        extra.update(tau=tau, r_elapsed=resp.get("elapsed"))
        res = EA.proportional_test(float(r["p_full"]))
        res["extra"] = extra
        return res


class ColocPropTest(BaselineMethod):
    """colocPropTest 0.9.3 run_proptests (defaults). s = min FDR-adjusted p over tested SNPs (high = consistent
    with proportionality/sharing). No sample-overlap option. s = 0.5 (undefined, flagged) if no test was run."""

    name = "coloc_prop_test"
    r_lib = "main"

    def _fit(self, locus):
        R = dense_ld(locus)
        args = dict(n1=scalar_n(locus.n1), n2=scalar_n(locus.n2))
        resp = self._r("coloctest", args, {"beta1": locus.beta1, "varbeta1": np.square(locus.se1),
                                           "beta2": locus.beta2, "varbeta2": np.square(locus.se2),
                                           "maf1": locus.maf1, "maf2": locus.maf2, "R": R})
        r = resp["result"]
        extra = dict(r, r_elapsed=resp.get("elapsed"), r_warnings=resp.get("warnings"))
        if not r.get("n_tests"):
            extra["undefined"] = True  # no test run: s = 0.5 (ties/undefined rule, via the adapter's NaN -> 0.5)
            res = EA.proportional_test(NAN)
        else:
            res = EA.proportional_test(float(r["min_fdr"]))
        res["extra"] = extra
        return res


# =========================================================================================== (8) lead-SNP rule
class LeadSNP(BaselineMethod):
    """Sanity floor: s = 1 if r^2(lead1, lead2) >= 0.8 in the inference LD (lead = argmax |z|), else 0."""

    name = "lead_snp"

    def __init__(self, r2_threshold: float = 0.8, timeout=None):
        super().__init__(timeout)
        self.r2_threshold = r2_threshold

    def _fit(self, locus):
        j1 = int(np.argmax(np.abs(locus.z1)))
        j2 = int(np.argmax(np.abs(locus.z2)))
        if locus.ld_type == "dense":
            r = float(locus.R[j1, j2])
        else:
            X = np.asarray(locus.X_ref, dtype=np.float64)
            r = float(X[:, j1] @ X[:, j2] / float(locus.n_ref))
        r2 = r * r
        res = EA.lead_snp(j1 == j2 or r2 >= self.r2_threshold)
        res["extra"] = {"lead1": j1, "lead2": j2, "r2": r2, "z1max": float(abs(locus.z1[j1])),
                        "z2max": float(abs(locus.z2[j2]))}
        return res


# =========================================================================================== registry
def all_methods() -> dict:
    """Name -> constructor for every tier-1 baseline wrapper."""
    return {
        "coloc_abf": lambda: ColocABF(),
        "coloc_susie": lambda: ColocSusie(mismatch=False),
        "coloc_susie_mm": lambda: ColocSusieMM(),
        "sharepro": lambda: SharePro(),
        "colocboost": lambda: ColocBoost(M=None),
        "colocboost_m1": lambda: ColocBoostM1(),
        "mvsusie": lambda: MvSusie(),
        "prop_coloc": lambda: PropColoc(),
        "coloc_prop_test": lambda: ColocPropTest(),
        "lead_snp": lambda: LeadSNP(),
    }


def get_method(name: str) -> BaselineMethod:
    return all_methods()[name]()
