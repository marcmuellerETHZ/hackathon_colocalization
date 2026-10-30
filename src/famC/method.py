"""NeuralColoc Method (INTERFACES §3/§4): `NeuralColoc(checkpoint).fit_batch(list[LocusData]) -> list[dict]`.

Per locus: famC.features.extract on the locus' own inference LD (dense R or factor X_ref), then batched
build + network on the GPU. Outputs (natural-mixture posteriors: famC.train_main.natural_logits for main-task models,
raw softmax for models trained with unweighted CE, `class_w=None`):
  P_S = P(S), P_D = P(D), P_W = 1 - P_S - P_D (absorbs G and U, as Families A and B do),
  s = P_S / (P_S + P_D) (ungated); E1 = E2 = NaN; extra = JSON with the 5-class posteriors and the top
  shared-PIP variant. runtime_s = per-locus share of the batch wall-clock including LD handling.
Trait types come from the locus attributes trait1_type / trait2_type (sim extension; default GWAS x molecular).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from famC.features import DenseLD, FactorLD, StackedLD, build, extract
from famC.model import NeuralColoc as Net
from famC.simadapter import CLASS_ORDER, TYPE_CODE, compact, collate_records
from famC.train_main import natural_logits, strip_ld


def locus_batch_dict(lc, device="cuda", arrays: bool = True) -> tuple[dict, object]:
    """Batch dict (B = 1) and LD access object of one LocusData. arrays=False: scalars and LD only."""
    P = lc.p
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)[None]
    ex = getattr(lc, "extra_attrs", {}) or {}
    rho = float(lc.rho_hat) if np.isfinite(lc.rho_hat) else 0.0
    rse = float(lc.rho_se) if np.isfinite(lc.rho_se) else 1.0
    b = {} if not arrays else dict(
        z1=t(lc.z1), z2=t(lc.z2), se1=t(lc.se1), se2=t(lc.se2), n1=t(lc.n1), n2=t(lc.n2), maf1=t(lc.maf1),
        maf2=t(lc.maf2), maf_ref=t(lc.maf_ref), mask=torch.ones(1, P, dtype=torch.bool, device=device))
    b.update(dict(
             n_ref=torch.tensor([float(lc.n_ref)], dtype=torch.float64, device=device),
             rho_hat=torch.tensor([rho], dtype=torch.float64, device=device),
             rho_se=torch.tensor([rse], dtype=torch.float64, device=device),
             type1=torch.tensor([TYPE_CODE.get(str(ex.get("trait1_type", "gwas")), 0.0)], device=device),
             type2=torch.tensor([TYPE_CODE.get(str(ex.get("trait2_type", "mol")), 1.0)], device=device)))
    if lc.ld_type == "dense":
        ld = DenseLD(torch.as_tensor(np.asarray(lc.R), dtype=torch.float32, device=device)[None])
    else:
        ld = FactorLD(torch.as_tensor(np.asarray(lc.X_ref), dtype=torch.float32, device=device)[None], 1,
                      n=float(lc.n_ref))
    return b, ld


class NeuralColoc:
    name = "neuralcoloc"

    def __init__(self, checkpoint: str | Path, variant: str = "full", d=128, layers=4, K=16, M=128, device="cuda",
                 batch: int = 32, class_w: str | None = "main"):
        """class_w: "main" = the main-task logit correction (logits - log CLASS_W; models trained with
        class-weighted CE, README C-T2); None = no correction (models trained with unweighted CE on the natural
        mixture, e.g. famC.c1)."""
        assert class_w in ("main", None), class_w
        self.device, self.M, self.variant, self.batch, self.class_w = device, M, variant, batch, class_w
        self.net = Net(d=d, layers=layers, K=K, use_ld=variant != "nold", use_all=variant != "screen").to(device)
        self.net.load_state_dict(torch.load(checkpoint, map_location=device))
        self.net.eval()

    @torch.no_grad()
    def fit_batch(self, loci: list) -> list[dict]:
        out = []
        for i in range(0, len(loci), self.batch):
            out += self._fit(loci[i:i + self.batch])
        return out

    def _batched_record(self, loci) -> dict:
        """One extract() over a batch of loci of different p (padded), each with its own LD (StackedLD)."""
        dev = self.device
        B = len(loci)
        P = max(lc.p for lc in loci)
        cols = ("z1", "z2", "se1", "se2", "n1", "n2", "maf1", "maf2", "maf_ref")
        arr = {k: np.zeros((B, P)) for k in cols}
        for k in ("se1", "se2", "n1", "n2"):
            arr[k][:] = 1.0
        mask = np.zeros((B, P), bool)
        lds, sc = [], {k: np.zeros(B) for k in ("n_ref", "rho_hat", "rho_se", "type1", "type2")}
        for i, lc in enumerate(loci):
            p = lc.p
            for k in cols:
                arr[k][i, :p] = np.asarray(getattr(lc, k), dtype=np.float64)
            mask[i, :p] = True
            b1, ld = locus_batch_dict(lc, dev, arrays=False)
            lds.append(ld)
            for k in sc:
                sc[k][i] = float(b1[k][0])
        b = {k: torch.as_tensor(v, device=dev) for k, v in arr.items()}
        b["mask"] = torch.as_tensor(mask, device=dev)
        b.update({k: torch.as_tensor(v, device=dev) for k, v in sc.items()})
        return extract(b, StackedLD(lds), M=self.M)

    def _fit(self, loci):
        torch.cuda.synchronize()
        t0 = time.time()
        ok, err = [], {}
        for j, lc in enumerate(loci):
            try:
                lc.validate()
                for k in ("z1", "z2", "se1", "se2", "maf1", "maf2", "maf_ref"):
                    if not np.all(np.isfinite(np.asarray(getattr(lc, k), dtype=np.float64))):
                        raise ValueError(f"non-finite {k}")
                ok.append(j)
            except Exception as e:  # noqa: BLE001  (reported as status=failed, never swallowed silently)
                err[j] = f"{type(e).__name__}: {e}"
        rec = None
        if ok:
            try:
                rec = self._batched_record([loci[j] for j in ok])
            except Exception:  # noqa: BLE001  isolate the failing locus with the per-locus path
                recs, ok2 = [], []
                for j in ok:
                    try:
                        b, ld = locus_batch_dict(loci[j], self.device)
                        recs.append(compact(extract(b, ld, M=self.M)))
                        ok2.append(j)
                    except Exception as e:  # noqa: BLE001
                        err[j] = f"{type(e).__name__}: {e}"
                ok = ok2
                rec = collate_records(recs, self.device) if recs else None
        probs = np.zeros((len(loci), 5))
        top = {}
        if rec is not None:
            f = build(rec)
            if self.variant == "nold":
                f = strip_ld(f)
            o = self.net(f)
            lg = o["logits"].double()
            probs[ok] = torch.softmax(natural_logits(lg) if self.class_w == "main" else lg, -1).cpu().numpy()
            pk = o["pip_logits"].argmax(-1).cpu().numpy()
            pp = torch.softmax(o["pip_logits"].double(), -1).cpu().numpy()
            sel = rec["sel"].cpu().numpy()
            for k, j in enumerate(ok):
                if pk[k] < self.M:
                    top[j] = (int(sel[k, pk[k]]), float(pp[k, pk[k]]))
        torch.cuda.synchronize()
        rt = (time.time() - t0) / max(len(loci), 1)
        rows = []
        for j, lc in enumerate(loci):
            if j in err:
                rows.append(dict(locus_id=lc.locus_id, method=self.name, s=math.nan, P_S=math.nan, P_D=math.nan,
                                 P_W=math.nan, E1=math.nan, E2=math.nan, call=None, status="failed", runtime_s=rt,
                                 extra=json.dumps({"error": err[j]})))
                continue
            PS, PD = float(probs[j, 0]), float(probs[j, 1])
            s = (PS + 1e-12) / (PS + PD + 2e-12)
            ex = {"P5": dict(zip(CLASS_ORDER, probs[j].round(8).tolist())), "variant": self.variant}
            if j in top:
                ex["top_shared_idx"], ex["top_shared_p"] = top[j]
            rows.append(dict(locus_id=lc.locus_id, method=self.name, s=s, P_S=PS, P_D=PD, P_W=max(0.0, 1 - PS - PD),
                             E1=math.nan, E2=math.nan, call=None, status="ok", runtime_s=rt, extra=json.dumps(ex)))
        return rows
