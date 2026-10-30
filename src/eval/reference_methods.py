"""Reference Methods (INTERFACES §4): the lead-SNP rule (tier-1 sanity floor) and the strength-only control (D6(e)).

LeadSNPRule (DESIGN §4.1 item 8): i1 = argmax |z1|, i2 = argmax |z2|; s = 1 if r²(i1, i2) >= r2_min in the
inference LD given to the method, else 0. No P_S / P_D (adapter row "Lead-SNP rule: binary").
"""
from __future__ import annotations

import json

import numpy as np

from eval import adapters
from eval.io import LocusData


class LeadSNPRule:
    name = "lead_snp"

    def __init__(self, r2_min: float = 0.8):
        self.r2_min = float(r2_min)

    def fit_one(self, loc: LocusData) -> dict:
        i1 = int(np.argmax(np.abs(loc.z1)))
        i2 = int(np.argmax(np.abs(loc.z2)))
        r = float(loc.ld_rows([i1])[0, i2])
        r2 = r * r
        out = adapters.lead_snp(r2 >= self.r2_min)
        out.update(locus_id=loc.locus_id, method=self.name, status="ok",
                   extra=json.dumps({"lead1": i1, "lead2": i2, "r2_leads": r2,
                                     "max_abs_z1": float(np.abs(loc.z1[i1])), "max_abs_z2": float(np.abs(loc.z2[i2]))}))
        return out

    def fit_batch(self, loci: list[LocusData]) -> list[dict]:
        return [self.fit_one(loc) for loc in loci]


class StrengthOnly:
    """Strength-only control scorer (D6(e)); see src/eval/strength_control.py.

    Sees only the lead |z| of each trait (and the locus pairing type as table stratum); scores come from a table
    fitted on labelled loci by evaluation code (single table, or a K-fold cross-fitted set on a validation
    benchmark). `table`: path to the JSON written by strength_control.dump, or the dict itself.
    needs_ld = False: the harness does not read LD for this method."""

    name = "strength_only"
    needs_ld = False

    def __init__(self, table, name: str | None = None):
        from eval import strength_control as SC
        self.table_path = str(table) if not isinstance(table, dict) else None
        self._table = SC.load(table)
        if name:
            self.name = name

    def fit_batch(self, loci: list[LocusData]) -> list[dict]:
        import pandas as pd

        from eval import strength_control as SC
        rows = []
        for loc in loci:
            a1, a2 = SC.lead_features(loc.z1, loc.z2)
            rows.append((loc.locus_id, a1, a2, str(loc.pairing_type)))
        feats = pd.DataFrame(rows, columns=["locus_id", "lead_abs_z1", "lead_abs_z2", "stratum"])
        pr = SC.predict_any(self._table, feats)
        out = []
        for f, p in zip(feats.itertuples(index=False), pr.itertuples(index=False)):
            sc = adapters.scores(s=p.s, P_S=p.P_S, P_D=p.P_D, P_W=p.P_W)
            sc.update(locus_id=f.locus_id, method=self.name, status="ok",
                      extra=json.dumps({"lead_abs_z1": f.lead_abs_z1, "lead_abs_z2": f.lead_abs_z2,
                                        "table_stratum": p.table_stratum, "fold": int(p.fold)}))
            out.append(sc)
        return out


class RhoOverride:
    """P3-23 real-data sensitivities: runs the Method `target` (module:Class, kwargs) with each locus's rho_hat
    replaced before fit_batch.
      mode "pooled": the per-cohort-pair pooled estimate (locus attribute rho_hat_pair, written by
                     data.real.rho_update); NaN / missing -> 0 (the P3-23(c) fallback chain ends at 0);
      mode "zero":   P3-23(b) rho = 0 for documented donor-disjoint cohort pairs: the locus attribute
                     rho_sens_zero (written by data.real.extract.build_test_arm: 0.0 for donor-disjoint pairs,
                     NaN otherwise) where present -- NaN keeps the primary rho_hat; without the attribute, 0.
    The original value is kept in extra_attrs["rho_hat_primary"]; every output row carries extra.rho_used and
    extra.rho_mode. needs_ld / needs_ld2 are forwarded from the wrapped method."""

    def __init__(self, target: str, kwargs: dict | None = None, mode: str = "pooled", name: str | None = None):
        from eval.harness import MethodSpec
        if mode not in ("pooled", "zero"):
            raise ValueError(f"RhoOverride mode {mode!r}")
        self._m = MethodSpec(target, kwargs or {}).build()
        self.mode = mode
        self.name = name or f"{getattr(self._m, 'name', 'method')}_rho{mode}"
        self.needs_ld = bool(getattr(self._m, "needs_ld", True))
        self.needs_ld2 = bool(getattr(self._m, "needs_ld2", False))

    def _rho(self, loc) -> float:
        ea = loc.extra_attrs or {}
        if self.mode == "zero":
            if "rho_sens_zero" not in ea:
                return 0.0
            try:
                z = float(ea["rho_sens_zero"])
            except (TypeError, ValueError):
                z = float("nan")
            return z if np.isfinite(z) else float(loc.rho_hat)
        r = ea.get("rho_hat_pair")
        try:
            r = float(r)
        except (TypeError, ValueError):
            return 0.0
        return r if np.isfinite(r) else 0.0

    def fit_batch(self, loci: list[LocusData]) -> list[dict]:
        used = {}
        for loc in loci:
            loc.extra_attrs = dict(loc.extra_attrs or {})
            loc.extra_attrs["rho_hat_primary"] = float(loc.rho_hat)
            loc.rho_hat = self._rho(loc)
            used[str(loc.locus_id)] = loc.rho_hat
        rows = self._m.fit_batch(loci)
        out = []
        for r in rows:
            r = dict(r)
            r["method"] = self.name
            ex = r.get("extra") or {}
            if isinstance(ex, str):
                try:
                    ex = json.loads(ex) if ex else {}
                except json.JSONDecodeError:
                    ex = {"method_extra": ex}
            if not isinstance(ex, dict):
                ex = {"method_extra": ex}
            ex.update(rho_used=used.get(str(r.get("locus_id"))), rho_mode=self.mode)
            r["extra"] = json.dumps(ex, default=repr)
            out.append(r)
        return out
