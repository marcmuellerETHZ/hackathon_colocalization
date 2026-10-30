"""LightGBM floor as a harness Method (INTERFACES §3/§4): `FloorColoc(model_dir).fit_batch(list[LocusData])`.

Per locus: the same record as NeuralColoc (famC.features.extract on the locus' own inference LD, compacted exactly
as the training records: fp16 per-variant columns and screen LD), then famC.floor.features_batch (leads,
conditional z, coloc.abf, SuSiE summaries on the screen) and the trained booster. Class probabilities are
corrected to the natural mixture as in famC.floor (divide by CLASS_W, renormalise). Outputs as NeuralColoc:
P_S, P_D, P_W = 1 - P_S - P_D, s = P_S / (P_S + P_D), E1 = E2 = NaN, extra = 5-class posteriors.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from famC.features import extract
from famC.method import locus_batch_dict
from famC.simadapter import CLASS_ORDER, compact


class FloorColoc:
    name = "famC_floor"

    def __init__(self, model_dir: str | Path, device: str = "cpu", M: int = 128, batch: int = 64, threads: int = 1):
        import lightgbm as lgb
        from famC.train_main import CLASS_W
        self.model_dir = Path(model_dir)
        self.bst = lgb.Booster(model_file=str(self.model_dir / "model.txt"))
        cfg = json.loads((self.model_dir / "metrics.json").read_text())
        self.best_iter = int(cfg["best_iter"])
        self.susie = bool(cfg.get("susie", True))
        self.cw = CLASS_W.numpy()
        self.device, self.M, self.batch, self.threads = device, M, batch, threads
        if device == "cpu":
            torch.set_num_threads(threads)

    @torch.no_grad()
    def fit_batch(self, loci: list) -> list[dict]:
        out = []
        for i in range(0, len(loci), self.batch):
            out += self._fit(loci[i:i + self.batch])
        return out

    def _fit(self, loci):
        from famC.floor import features_batch
        t0 = time.time()
        recs, ok, err = [], [], {}
        for j, lc in enumerate(loci):
            try:
                lc.validate()
                for k in ("z1", "z2", "se1", "se2", "maf1", "maf2", "maf_ref"):
                    if not np.all(np.isfinite(np.asarray(getattr(lc, k), dtype=np.float64))):
                        raise ValueError(f"non-finite {k}")
                b, ld = locus_batch_dict(lc, self.device)
                recs.append(compact(extract(b, ld, M=self.M)))
                ok.append(j)
            except Exception as e:  # noqa: BLE001  (reported as status=failed)
                err[j] = f"{type(e).__name__}: {e}"
        probs = np.zeros((len(loci), 5))
        if recs:
            try:
                X = features_batch(recs, susie=self.susie, device=self.device)
                P = self.bst.predict(X, num_iteration=self.best_iter, num_threads=self.threads)
                P = P / self.cw[None]
                probs[ok] = P / P.sum(1, keepdims=True)
            except Exception as e:  # noqa: BLE001
                for j in ok:
                    err[j] = f"{type(e).__name__}: {e}"
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
            rows.append(dict(locus_id=lc.locus_id, method=self.name, s=s, P_S=PS, P_D=PD, P_W=max(0.0, 1 - PS - PD),
                             E1=math.nan, E2=math.nan, call=None, status="ok", runtime_s=rt,
                             extra=json.dumps({"P5": dict(zip(CLASS_ORDER, probs[j].round(8).tolist()))})))
        return rows
