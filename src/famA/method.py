"""Family A Method (INTERFACES §4): JointSuSiE on GPU with power-aware inclusion–exclusion outputs (DESIGN §5 A)."""
from __future__ import annotations

import json
import math
import time

import numpy as np

import dataclasses

from .joint_susie import TYPE_THRESHOLDS, JointConfig, joint_susie_batch

# Ablation ladder (DESIGN §5 A; P3-5/P3-6). Cumulative rungs; every rung has the P3-5 fixes that are not rungs
# (η floor 1e-3, relative-ELBO/α stopping, own-effect guard on 'power >= 0.2') and π by EB (κ = 4).
LADDER = ("base", "+rho", "+eta", "+E", "+multistart", "+perTraitLD")
_COMPONENTS = ("rho", "eta", "E", "multistart", "perTraitLD")


def ladder_components(rung) -> tuple:
    """Rung name of LADDER (cumulative: '+eta' = rho + eta) or an iterable / '+'-joined string of components from
    ('rho', 'eta', 'E', 'multistart', 'perTraitLD'), e.g. 'rho+eta+multistart+perTraitLD' (ladder without E)."""
    if isinstance(rung, str) and rung in LADDER:
        return _COMPONENTS[:LADDER.index(rung)]
    comps = [c for c in (rung.split("+") if isinstance(rung, str) else rung) if c and c != "base"]
    bad = [c for c in comps if c not in _COMPONENTS]
    if bad:
        raise ValueError(f"unknown ladder components {bad} (rungs {LADDER}, components {_COMPONENTS})")
    return tuple(c for c in _COMPONENTS if c in comps)


def ladder_config(rung, pi_mode: str = "eb", nonweak_own: bool = True, **overrides) -> tuple[JointConfig, bool]:
    """(JointConfig, use_rho) of a ladder rung. pi_mode 'eb' (κ = 4, default, P3-6) or 'fixed' (π = 1/4 each,
    sensitivity); nonweak_own False = μ-only 'power >= 0.2' option. Layer E uses the α stopping rule (its SER
    updates do not maximise the ELBO)."""
    c = ladder_components(rung)
    cfg = JointConfig(estimate_eta="eta" in c, ref_var="E" in c, multistart="multistart" in c,
                      per_trait_ld="perTraitLD" in c, pi_mode=pi_mode, nonweak_own=nonweak_own,
                      convergence="alpha" if "E" in c else "rel")
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg, "rho" in c


def ladder_method(rung, pi_mode: str = "eb", nonweak_own: bool = True, device=None, **overrides):
    cfg, use_rho = ladder_config(rung, pi_mode, nonweak_own, **overrides)
    return JointSusieMethod(cfg, use_rho=use_rho, device=device)


def _ld_input(lc):
    if getattr(lc, "ld_type", "dense") == "factor":
        return "factor", lc.X_ref
    return "dense", lc.R


def attach_ld2(set_path, loci) -> int:
    """Family-A-side loader for per-trait LD (the shared eval reader does not load it): sets lc.R2 / lc.X_ref2
    from the locus set's 'R2' / 'X_ref2' datasets (L0 sets) on the given LocusData objects. Returns the number of
    loci that got a trait-2 LD. Opens only the locus groups of the given (method-input) loci."""
    import h5py
    k = 0
    with h5py.File(set_path, "r") as f:
        for lc in loci:
            g = f[lc.locus_id]
            for name in ("R2", "X_ref2"):
                if name in g:
                    setattr(lc, name, np.asarray(g[name][()], dtype=np.float32))
                    k += 1
    return k


def _ld2_input(lc, kind):
    """Trait-2 LD of the locus (L0: R2 / X_ref2 with attrs ld_type2 / n_ref2) if it has the same kind as trait 1's;
    else None (one LD for both traits)."""
    ea = getattr(lc, "extra_attrs", None) or {}
    k2 = str(ea.get("ld_type2", kind))
    if k2 != kind:
        return None, None
    if kind == "dense":
        R2 = getattr(lc, "R2", None)
        return (R2, None) if R2 is not None else (None, None)
    X2 = getattr(lc, "X_ref2", None)
    if X2 is None:
        return None, None
    nr = ea.get("n_ref2")
    return X2, (int(nr) if nr is not None else X2.shape[0])


def _scalar_n(n) -> float:
    return float(np.median(np.atleast_1d(np.asarray(n, dtype=np.float64))))


def locus_types(lc, default=("gwas", "mol")):
    """(trait1_type, trait2_type) from the locus attributes (sim: extra_attrs trait1_type/trait2_type);
    returns (types, from_attrs). Unknown values raise (never silently mapped)."""
    ea = getattr(lc, "extra_attrs", None) or {}
    t1, t2 = ea.get("trait1_type"), ea.get("trait2_type")
    if t1 is None or t2 is None:
        return tuple(default), False
    t1, t2 = str(t1), str(t2)
    for t in (t1, t2):
        if t not in TYPE_THRESHOLDS:
            raise ValueError(f"{lc.locus_id}: unknown trait type {t!r}")
    return (t1, t2), True


class JointSusieMethod:
    """name encodes the ablation rung (DESIGN §5 A ladder, LADDER above): base (rho = 0, η fixed at the floor) ->
    +rho -> +eta -> +E (ref_var) -> +multistart ('_ms') -> +perTraitLD ('_ptld');
    '_pifix' marks a fixed state prior (default: per-locus EB of π), '_own' the M3 own-effect-only power events,
    '_numu' the μ-only 'power >= 0.2' option, '_floor0' η floor 0 (M3.5 behaviour)."""

    def __init__(self, cfg: JointConfig | None = None, use_rho: bool = True, name: str | None = None, device=None):
        self.cfg = cfg or JointConfig()
        self.use_rho = use_rho
        self.device = device
        c = self.cfg
        self.name = name or ("famA_joint_susie" + ("_rho" if use_rho else "") + ("_eta" if c.estimate_eta else "")
                             + ("_E" if c.ref_var else "") + ("_ms" if c.multistart else "")
                             + ("_ptld" if c.per_trait_ld else "") + ("_pifix" if c.pi_mode == "fixed" else "")
                             + ("" if c.power_marginal else "_own") + ("" if c.nonweak_own else "_numu")
                             + ("_floor0" if c.eta_floor <= 0 else ""))
        # per-trait LD needs lc.R2 / lc.X_ref2 on the input loci; the shared reader does not load them, so use
        # attach_ld2(set_path, loci) before fit_batch (else the fit falls back to one LD, recorded in extra)
        self.needs_ld2 = bool(c.per_trait_ld)

    def fit_batch(self, loci: list) -> list[dict]:
        from eval import adapters
        t0 = time.time()
        z1s, z2s, lds, kinds, n1s, n2s, rhos, nrefs, types, from_attrs, refn, mafs, lds2, nrefs2 = (
            [] for _ in range(14))
        for lc in loci:
            kind, ld = _ld_input(lc)
            l2, n2r = _ld2_input(lc, kind) if self.cfg.per_trait_ld else (None, None)
            lds2.append(l2); nrefs2.append(n2r)
            z1s.append(np.asarray(lc.z1, float)); z2s.append(np.asarray(lc.z2, float))
            lds.append(ld); kinds.append(kind)
            n1s.append(_scalar_n(lc.n1)); n2s.append(_scalar_n(lc.n2))
            rh = float(getattr(lc, "rho_hat", 0.0)) if self.use_rho else 0.0
            rhos.append(rh if math.isfinite(rh) else 0.0)
            nrefs.append(getattr(lc, "n_ref", None) if kind == "factor" else None)
            ty, fa = locus_types(lc, self.cfg.default_types)
            types.append(ty); from_attrs.append(fa)
            nr = getattr(lc, "n_ref", None)
            refn.append(float(nr) if nr is not None and float(nr) > 0 else (float(ld.shape[0]) if kind == "factor" else None))
            mafs.append(getattr(lc, "maf_ref", None))
        try:
            res = joint_susie_batch(z1s, z2s, lds, n1s, n2s, rhos, kinds, self.cfg, device=self.device, n_refs=nrefs,
                                    types=types, ref_sizes=refn if self.cfg.ref_var else None,
                                    mafs=mafs if self.cfg.ref_var else None, lds2=lds2, n_refs2=nrefs2)
            err = None
        except Exception as e:  # pragma: no cover - reported as failures, never silently
            res, err = [None] * len(loci), repr(e)
        rt = (time.time() - t0) / max(len(loci), 1)
        out = []
        for i, (lc, r) in enumerate(zip(loci, res)):
            if r is None:
                out.append(dict(locus_id=lc.locus_id, method=self.name, s=math.nan, P_S=math.nan, P_D=math.nan,
                                P_W=math.nan, E1=math.nan, E2=math.nan, status="failed", runtime_s=rt,
                                extra=json.dumps({"error": err})))
                continue
            sc = adapters.inclusion_exclusion(r["a"], r["w"], r["p1t"], r["p2t"], r["p1t"] + r["p2t"])
            status = "ok" if r["converged"] else "nonconverged"
            if not all(math.isfinite(sc[k]) for k in ("s", "P_S", "P_D")):
                status = "failed"
            extra = dict(eta=r["eta"], niter=r["niter"], n_cs=len(r["sets"]["cs"]),
                         pi=np.round(r["pi"], 6).tolist(), types=list(types[i]), types_from_attrs=from_attrs[i],
                         state_prob=np.round(r["state_prob"], 6).tolist(), s2=np.round(r["s2"], 4).tolist(),
                         a=np.round(r["a"], 6).tolist(), w=np.round(r["w"], 6).tolist(),
                         p1t=np.round(r["p1t"], 6).tolist(), p2t=np.round(r["p2t"], 6).tolist(),
                         rho_used=float(rhos[i]), max_pip1=float(np.max(r["pip1"])), max_pip2=float(np.max(r["pip2"])),
                         start=r.get("start"), elbo_starts=[float(x) for x in r.get("elbo_starts", [])],
                         per_trait_ld=bool(r.get("per_trait_ld")))
            out.append(dict(locus_id=lc.locus_id, method=self.name, s=sc["s"], P_S=sc["P_S"], P_D=sc["P_D"],
                            P_W=sc["P_W"], E1=r["E1"], E2=r["E2"], status=status, runtime_s=rt, extra=json.dumps(extra)))
        return out


class LadderMethod(JointSusieMethod):
    """MethodSpec-buildable ladder rung (harness workers build it from kwargs):
    MethodSpec("famA.method:LadderMethod", {"rung": "rho+eta+multistart+perTraitLD"}, name=...).
    Same fit as ladder_method(rung, pi_mode, nonweak_own, **overrides)."""

    def __init__(self, rung="rho+eta+multistart+perTraitLD", pi_mode: str = "eb", nonweak_own: bool = True,
                 device=None, **overrides):
        cfg, use_rho = ladder_config(rung, pi_mode, nonweak_own, **overrides)
        super().__init__(cfg, use_rho=use_rho, device=device)
        self.rung = rung


def ladder_name(rung, pi_mode: str = "eb", nonweak_own: bool = True, **overrides) -> str:
    """Method name of a ladder rung (no GPU work)."""
    cfg, use_rho = ladder_config(rung, pi_mode, nonweak_own, **overrides)
    return JointSusieMethod(cfg, use_rho=use_rho).name
