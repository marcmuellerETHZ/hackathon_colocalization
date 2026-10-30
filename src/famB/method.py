"""ExactColoc as a Method (INTERFACES Sec. 3/4).

    from famB.method import ExactColoc
    rows = ExactColoc(mode="gaussian").fit_batch(loci)   # loci: list[eval.io.LocusData]

Per locus: screen (Sec. 5 B) -> Rt, V from rho_hat -> priors (mode) -> exact enumeration on GPU (fp64)
-> PP over H0..H4, H34 (exact) and oracle-event posteriors P_S, P_D, P_W, E1, E2 (Monte Carlo).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from common.seeds import SEEDS, derive

from .constants import CAT_NAMES
from .enum import LocusTensors, TermSet, assert_precision_guards, enumerate_batch, get_evaluator
from .events import event_posteriors
from .priors import EBGrid, SimPrior, eb_select, toy_l0star_prior, type_table_exchangeable, type_table_per_snp
from .screen import LDAccess, screen


@dataclass
class ExactColocConfig:
    mode: str = "gaussian"  # "coloc" | "gaussian" | "simulator"
    M: int = 100  # screen size (gaussian / simulator); coloc mode always uses full p
    Kc: int = 2  # max causal per trait (coloc mode forces 1)
    k_cond: int | None = None  # lead-removed candidates per trait (default M // 8)
    collapse: bool = True
    # coloc-mode / per-SNP priors
    W: float = 0.15  # coloc prior sd of beta (sdY = 1 units)
    p1: float = 1e-4
    p2: float = 1e-4
    p12: float = 5e-6
    # gaussian mode
    eb: bool = True
    sigma: tuple = (6.0, 6.0)  # used if eb is False
    eta: float = 1e-3  # used if eb is False (and in simulator mode)
    eta_floor: float = 1e-3  # DESIGN: eta >= 1e-3. Set eta = eta_floor = 0 only when the LD given is already
                             # R_tilde (L0* sets store R_tilde = (1 - 1e-3) R + 1e-3 I)
    eb_grid: EBGrid = field(default_factory=EBGrid)
    rho_b: float = 0.0
    use_rho_hat: bool = True
    rho_clip: float = 0.95
    # simulator mode
    sim_prior_fn: object = None  # callable(locus) -> SimPrior; default toy prior
    # Monte Carlo
    n_mc: int = 16384
    seed: int = 20261029  # must be one of common.seeds.SEEDS; MC draws of locus L use
    seed_key: str = ""    # derive(seed, "famB_mc", seed_key, L.locus_id): independent of batch composition
    # engine
    device: str = "cuda"
    dtype: str = "float64"
    chunk_elems: int = 1 << 22
    batch_elems: int = 1 << 26  # max nB * n_terms per enumeration batch
    compile: bool = False  # torch.compile the term evaluator (3-7x faster after ~10-30 s warm-up)
    # B-mm (DESIGN_P2 §2, famB.mismatch): variance-inflated RSS likelihood, gaussian mode only
    mismatch: bool = False
    mm_iter: int = 2  # plug-in refinements of Delta_hat after the phase-1 fit (pass 0)
    mm_cross_terms: bool = True  # j != j' covariance terms of the reference error (layer E "cross terms")
    mm_n_ref: float | None = None  # override of locus.n_ref (math.inf: Delta = 0, reproduces phase-1 B)


def _rho(locus, cfg) -> float:
    if cfg.mode == "coloc" or not cfg.use_rho_hat:
        return 0.0
    r = float(getattr(locus, "rho_hat", float("nan")))
    if not math.isfinite(r):
        return 0.0
    return float(np.clip(r, -cfg.rho_clip, cfg.rho_clip))


class ExactColoc:
    """Family B. `name` encodes the mode (e.g. exactcoloc_gaussian_M100)."""

    def __init__(self, mode: str = "gaussian", **kw):
        self.cfg = ExactColocConfig(mode=mode, **kw)
        if self.cfg.mode not in ("coloc", "gaussian", "simulator"):
            raise ValueError(self.cfg.mode)
        if self.cfg.mode == "coloc":
            self.cfg.Kc = 1
        self.cfg.eta = max(self.cfg.eta, self.cfg.eta_floor)
        dev = self.cfg.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)
        self.dtype = getattr(torch, self.cfg.dtype)
        if self.cfg.mismatch and self.cfg.mode != "gaussian":
            raise ValueError("B-mm (mismatch=True) is defined for gaussian mode only")
        self.name = (f"exactcoloc_{'mm_' if self.cfg.mismatch else ''}{self.cfg.mode}"
                     + ("" if self.cfg.mode == "coloc" else f"_M{self.cfg.M}"))
        if self.cfg.mismatch:  # primary B-mm = mm_iter 2 with cross terms; other settings are named ablations
            self.name += ("" if self.cfg.mm_iter == 2 else f"_it{self.cfg.mm_iter}") + ("" if self.cfg.mm_cross_terms else "_diag")
        self._ts_cache: dict = {}
        if self.cfg.seed not in SEEDS:
            raise ValueError(f"seed {self.cfg.seed} is not in common.seeds.SEEDS")
        assert_precision_guards()

    # ------------------------------------------------------------------------------------------
    def _termset(self, M, Kc, K):
        key = ("pair", M, Kc, K)
        if key not in self._ts_cache:
            self._ts_cache[key] = TermSet(M, Kc, K=K, device=self.device)
        return self._ts_cache[key]

    def _prepare(self, locus) -> dict:
        """CPU part: screen, LD of the union, per-locus prior parameters."""
        cfg = self.cfg
        t0 = time.perf_counter()
        z1 = np.asarray(locus.z1, np.float64)
        z2 = np.asarray(locus.z2, np.float64)
        p = len(z1)
        ld = LDAccess.from_locus(locus)
        rho = _rho(locus, cfg)
        d = dict(locus_id=locus.locus_id, p=p, rho=rho, ld=ld)
        if cfg.mode == "coloc":
            sel = np.arange(p)
            d.update(sel=sel, mult=np.ones(p), Ruu=None, zmax=float(max(np.abs(z1).max(), np.abs(z2).max())))
            se = np.stack([np.asarray(locus.se1, np.float64), np.asarray(locus.se2, np.float64)])
            d["sd"] = (cfg.W / se)[:, :, None]  # [2, p, 1]
            d["eta"] = 0.0
        else:
            eta_screen = cfg.eta
            sc = screen(z1, z2, ld, M=cfg.M, k_cond=cfg.k_cond, eta=eta_screen,
                        collapse=cfg.collapse and p > cfg.M)
            d.update(sel=sc.sel, mult=sc.mult, zmax=sc.zmax, Ruu=ld.sub(sc.sel), screen=sc)
        d["zu"] = np.stack([z1[d["sel"]], z2[d["sel"]]])
        if cfg.mismatch:
            d["n_ref"] = float(cfg.mm_n_ref) if cfg.mm_n_ref is not None else float(locus.n_ref)
        if cfg.mode == "simulator":
            pr = (cfg.sim_prior_fn or (lambda _l: toy_l0star_prior()))(locus)
            pr.validate()
            d["simprior"] = pr
        d["t_prep"] = time.perf_counter() - t0
        return d

    def _tensors(self, items: list[dict]) -> LocusTensors:
        cfg = self.cfg
        dev, f64 = self.device, torch.float64
        nB = len(items)
        M = len(items[0]["sel"])
        Vinv = np.zeros((nB, 2, 2))
        b = np.zeros((nB, 2, M))
        for i, d in enumerate(items):
            r = d["rho"]
            Vi = np.linalg.inv(np.array([[1.0, r], [r, 1.0]]))
            Vinv[i] = Vi
            b[i] = Vi @ d["zu"]
        logm = np.stack([np.log(d["mult"]) for d in items])
        if cfg.mode == "coloc":
            Rt = None
            sd = np.stack([d["sd"] for d in items])
            logwk = np.zeros((nB, 2, 1))
            lpt = np.stack([type_table_per_snp(cfg.p1, cfg.p2, cfg.p12, 1)] * nB)
            eta = np.zeros(nB)
        else:
            eta = np.array([d["eta"] for d in items])
            Rt = np.stack([(1 - d["eta"]) * d["Ruu"] + d["eta"] * np.eye(M) for d in items])
            if cfg.mode == "gaussian":
                sd = np.zeros((nB, 2, M, 1))
                for i, d in enumerate(items):
                    sd[i, 0] = d["sigma"][0]
                    sd[i, 1] = d["sigma"][1]
                logwk = np.zeros((nB, 2, 1))
                lpt = np.stack([type_table_per_snp(cfg.p1, cfg.p2, cfg.p12, cfg.Kc)] * nB)
            else:
                K = items[0]["simprior"].mix_sd.shape[1]
                sd = np.stack([np.broadcast_to(d["simprior"].mix_sd[:, None, :], (2, M, K)) for d in items])
                with np.errstate(divide="ignore"):  # zero-weight components -> -inf (excluded exactly)
                    logwk = np.stack([np.log(d["simprior"].mix_w) for d in items])
                lpt = np.stack([type_table_exchangeable(d["simprior"].type_prob, M) for d in items])
        tt = lambda a: None if a is None else torch.as_tensor(np.ascontiguousarray(a), dtype=f64, device=dev)
        return LocusTensors(Rt=tt(Rt), b=tt(b), Vinv=tt(Vinv), sd=tt(sd), logwk=tt(logwk),
                            rho_b=torch.full((nB,), cfg.rho_b if cfg.mode == "gaussian" else 0.0, dtype=f64, device=dev),
                            logm=tt(logm), logprior_type=tt(lpt), eta=tt(eta))

    def _r_raw_fn(self, items):
        """Raw-LD lookup between screened variants for the U class when Rt is None (coloc mode)."""
        dev = self.device

        def fn(bidx, vi, vj):
            out = torch.zeros(vi.shape[0], dtype=torch.float64, device=dev)
            same = vi == vj
            for b in torch.unique(bidx).tolist():
                msk = (bidx == b) & ~same
                if not bool(msk.any()):
                    continue
                d = items[b]
                ui = vi[msk].cpu().numpy()
                uj = vj[msk].cpu().numpy()
                uniq, inv = np.unique(np.concatenate([ui, uj]), return_inverse=True)
                rows = d["ld"].rows(d["sel"][uniq])[:, d["sel"][uniq]]
                k = len(ui)
                out[msk] = torch.as_tensor(rows[inv[:k], inv[k:]], dtype=torch.float64, device=dev)
            return torch.where(same, torch.ones_like(out), out)
        return fn

    def _locus_generators(self, items: list[dict]) -> list:
        """One torch.Generator per locus, seeded by derive(seed, "famB_mc", seed_key, locus_id): the Monte Carlo
        draws of a locus do not depend on which other loci share its batch, or on batch order."""
        gens = []
        for d in items:
            g = torch.Generator(device=self.device)
            g.manual_seed(derive(self.cfg.seed, "famB_mc", self.cfg.seed_key, str(d["locus_id"])) % (2 ** 63))
            gens.append(g)
        return gens

    def _mm_fit(self, items: list[dict], T: LocusTensors) -> LocusTensors:
        """B-mm: starting from the phase-1 fit T (pass 0), `mm_iter` times: exact posterior second moments ->
        Delta_hat (famB.mismatch.inflation_blocks) -> EB (sigma, eta) under the inflated likelihood ->
        full-z_M Phi, b. Returns the final tensors (with Phi)."""
        from .mismatch import eb_select_mm, inflation_blocks, logz_and_moments, phi_b
        cfg, dev, f64 = self.cfg, self.device, torch.float64
        M = T.M
        ts = self._termset(M, cfg.Kc, 1)
        Rraw = torch.as_tensor(np.stack([d["Ruu"] for d in items]), dtype=f64, device=dev)
        nref = torch.as_tensor([d["n_ref"] for d in items], dtype=f64, device=dev)
        Vmat = torch.as_tensor(np.stack([[[1.0, d["rho"]], [d["rho"], 1.0]] for d in items]), dtype=f64, device=dev)
        zu = torch.as_tensor(np.stack([d["zu"] for d in items]), dtype=f64, device=dev)
        for d in items:
            d["mm"] = dict(n_ref=d["n_ref"], eta=[float(d["eta"])], sigma=[list(d["sigma"])], delta_mean=[],
                           delta_max=[], delta12_mean=[], n_proj=[], logZ=[])
        for _ in range(cfg.mm_iter):
            logZ, S = logz_and_moments(T, ts)
            res = {"logZ": logZ}
            D, n_proj = inflation_blocks(Rraw, S, nref, cfg.mm_cross_terms)
            Dn = D.cpu().numpy()
            lz = res["logZ"].cpu().numpy()
            for i, d in enumerate(items):
                dd = np.stack([Dn[i, 0, 0], Dn[i, 1, 1]])
                if cfg.eb:
                    s1, s2, eta, _ = eb_select_mm(d["zu"], d["Ruu"], np.log(d["mult"]), (cfg.p1 + cfg.p12, cfg.p2 + cfg.p12),
                                                  cfg.Kc, cfg.eb_grid, dev, dd, termset_cache=self._ts_cache)
                    d["sigma"], d["eta"] = (s1, s2), max(eta, cfg.eta_floor)
                mm = d["mm"]
                mm["logZ"].append(float(lz[i]))
                mm["eta"].append(float(d["eta"]))
                mm["sigma"].append(list(d["sigma"]))
                mm["delta_mean"].append([float(dd[0].mean()), float(dd[1].mean())])
                mm["delta_max"].append([float(dd[0].max()), float(dd[1].max())])
                mm["delta12_mean"].append(float(Dn[i, 0, 1].mean()))
            for d in items:
                d["mm"]["n_proj"].append(int(n_proj))
            T = self._tensors(items)
            Phi, b, _ = phi_b(T.Rt, Vmat, T.Vinv, zu, D)
            T.Phi, T.b = Phi, b
        return T

    # ------------------------------------------------------------------------------------------
    def run_items(self, items: list[dict]) -> list[dict]:
        """Enumerate + events for prepared items sharing M (one GPU batch)."""
        cfg = self.cfg
        M = len(items[0]["sel"])
        if cfg.mode == "gaussian":
            for d in items:
                if cfg.eb:
                    s1, s2, eta, _ = eb_select(d["zu"], d["Ruu"], np.log(d["mult"]),
                                               (cfg.p1 + cfg.p12, cfg.p2 + cfg.p12), cfg.Kc, cfg.eb_grid,
                                               self.device, termset_cache=self._ts_cache)
                    d["sigma"], d["eta"] = (s1, s2), max(eta, cfg.eta_floor)
                else:
                    d["sigma"], d["eta"] = tuple(cfg.sigma), cfg.eta
        elif cfg.mode == "simulator":
            for d in items:
                d["eta"] = cfg.eta
        T = self._tensors(items)
        if cfg.mismatch:
            T = self._mm_fit(items, T)
        ts = self._termset(M, cfg.Kc, T.K)
        gens = self._locus_generators(items)
        res = enumerate_batch(T, ts, n_samples=cfg.n_mc, generator=gens, chunk_elems=cfg.chunk_elems,
                              dtype=self.dtype, evaluator=get_evaluator(cfg.compile))
        ev = event_posteriors(T, ts, res["samples"], generator=gens,
                              r_raw_fn=self._r_raw_fn(items) if T.Rt is None else None)
        out = []
        PP = res["PP"].cpu().numpy()
        pip = res["pip"].cpu().numpy()
        spip = res["spip"].cpu().numpy()
        evn = {k: v.cpu().numpy() for k, v in ev.items()}
        eps = 0.5 / cfg.n_mc
        for i, d in enumerate(items):
            PS, PD = float(evn["S"][i]), float(evn["D"][i])
            pp = {CAT_NAMES[c]: float(PP[i, c]) for c in range(6)}
            num_s = pp["H4"] + pp["H34"]
            den = pp["H3"] + num_s
            top = np.argsort(-spip[i])[:5]
            extra = dict(PP=pp, s_pp=(num_s / den) if den > 0 else 0.5,
                         P_U=float(evn["U"][i]), P_G=float(evn["G"][i]), P_W_strict=float(evn["W"][i]),
                         M=int(M), p=int(d["p"]), n_terms=int(ts.n_terms), n_mc=int(cfg.n_mc),
                         eta=float(d["eta"]), rho_used=float(d["rho"]),
                         sigma=list(d.get("sigma", ())), zmax=float(d["zmax"]),
                         max_pip1=float(pip[i, 0].max()), max_pip2=float(pip[i, 1].max()),
                         top_shared=[[int(d["sel"][j]), float(spip[i, j])] for j in top],
                         t_prep=float(d["t_prep"]))
            if cfg.mismatch:
                extra["mm"] = d["mm"]
            out.append(dict(locus_id=d["locus_id"], method=self.name,
                            s=(PS + eps) / (PS + PD + 2 * eps), P_S=PS, P_D=PD, P_W=max(0.0, 1.0 - PS - PD),
                            E1=float(evn["E1"][i]), E2=float(evn["E2"][i]), call=None, status="ok",
                            runtime_s=float("nan"), extra=extra))
        return out

    def fit_batch(self, loci) -> list[dict]:
        cfg = self.cfg
        results: dict[str, dict] = {}
        prepared = []
        for lo in loci:
            try:
                prepared.append(self._prepare(lo))
            except Exception as e:  # noqa: BLE001 -- reported as status=failed, never silently
                results[lo.locus_id] = dict(locus_id=lo.locus_id, method=self.name, s=float("nan"), P_S=float("nan"),
                                            P_D=float("nan"), P_W=float("nan"), E1=float("nan"), E2=float("nan"),
                                            call=None, status="failed", runtime_s=float("nan"),
                                            extra=json.dumps({"error": repr(e)}))
        groups: dict[int, list] = {}
        for d in prepared:
            groups.setdefault(len(d["sel"]), []).append(d)
        for M, items in groups.items():
            K = 1 if cfg.mode != "simulator" else items[0]["simprior"].mix_sd.shape[1]
            n_terms = self._termset(M, cfg.Kc, K).n_terms
            per = max(1, cfg.batch_elems // max(1, n_terms))
            for s in range(0, len(items), per):
                chunk = items[s:s + per]
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    rows = self.run_items(chunk)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) / len(chunk)
                    for d, r in zip(chunk, rows):
                        r["runtime_s"] = dt + d["t_prep"]
                        r["extra"]["t_gpu_share"] = dt
                        r["extra"] = json.dumps(r["extra"])
                        results[d["locus_id"]] = r
                except Exception as e:  # noqa: BLE001
                    if cfg.mismatch and len(chunk) > 1:  # B-mm: isolate the failing locus (one-by-one retry)
                        for d in chunk:
                            t1 = time.perf_counter()
                            try:
                                r = self.run_items([d])[0]
                                if self.device.type == "cuda":
                                    torch.cuda.synchronize()
                                dt1 = time.perf_counter() - t1
                                r["runtime_s"] = dt1 + d["t_prep"]
                                r["extra"]["t_gpu_share"] = dt1
                                r["extra"] = json.dumps(r["extra"])
                                results[d["locus_id"]] = r
                            except Exception as e1:  # noqa: BLE001
                                results[d["locus_id"]] = dict(
                                    locus_id=d["locus_id"], method=self.name, s=float("nan"), P_S=float("nan"),
                                    P_D=float("nan"), P_W=float("nan"), E1=float("nan"), E2=float("nan"), call=None,
                                    status="failed", runtime_s=float("nan"), extra=json.dumps({"error": repr(e1)}))
                        continue
                    for d in chunk:
                        results[d["locus_id"]] = dict(locus_id=d["locus_id"], method=self.name, s=float("nan"),
                                                      P_S=float("nan"), P_D=float("nan"), P_W=float("nan"),
                                                      E1=float("nan"), E2=float("nan"), call=None, status="failed",
                                                      runtime_s=float("nan"), extra=json.dumps({"error": repr(e)}))
        return [results[lo.locus_id] for lo in loci]
