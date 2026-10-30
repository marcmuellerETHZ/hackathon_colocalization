"""P3-5 validation-only smoke: runtime and non-convergence of the Family A ladder rungs on screen_v2 validation
cells OTHER than the kill-test A1 cell (L0|P1|r_p=0.0, never opened here). No truth file is read.

python -m famA.validation.smoke_p3 --per-cell 60 --out results/famA/p3_fixes_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.io import LocusSetReader
from famA.method import attach_ld2, ladder_method

SETS = Path("data/processed/sets/screen_v2")
A1_CELL = "L0|P1|r_p=0.0"
CELLS = ["L0_P3_rho=0.5", "L2_P1_r_p=0.0", "L2_P3_rho=0.5", "L4_P1_r_p=0.0", "L4_P3_rho=0.5"]
RUNGS = ["+eta", "rho+eta+multistart", "rho+eta+multistart+perTraitLD"]   # no layer E until table v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default=",".join(RUNGS))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    loci = []
    for cell in CELLS:
        path = SETS / f"{cell}.h5"
        with LocusSetReader(path) as rd:
            ids = sorted(rd.ids())
            pick = sorted(rng.choice(len(ids), size=min(a.per_cell, len(ids)), replace=False).tolist())
            ls = [rd.read(ids[i]) for i in pick]
        for lc in ls:
            assert lc.cell_id != A1_CELL and lc.split == "valid", (lc.locus_id, lc.cell_id, lc.split)
        attach_ld2(path, ls)
        loci += ls
    print(f"{len(loci)} loci; p range {min(lc.p for lc in loci)}-{max(lc.p for lc in loci)}", flush=True)
    rows, summ = [], []
    # warm-up (Triton compilation, CUDA context) on 4 loci, not timed
    ladder_method("+eta").fit_batch(loci[:4])
    for rung in a.rungs.split(","):
        m = ladder_method(rung)
        torch.cuda.synchronize()
        t0 = time.time()
        res = m.fit_batch(loci)
        torch.cuda.synchronize()
        wall = time.time() - t0
        for lc, r in zip(loci, res):
            ex = json.loads(r["extra"])
            rows.append(dict(rung=rung, method=m.name, locus_id=lc.locus_id, cell=lc.cell_id, p=lc.p,
                             status=r["status"], niter=ex.get("niter"), eta=ex.get("eta"), start=ex.get("start"),
                             per_trait_ld=ex.get("per_trait_ld"), elbo_default=(ex.get("elbo_starts") or [np.nan])[0],
                             elbo_noshare=(ex.get("elbo_starts") + [np.nan])[1] if ex.get("elbo_starts") else np.nan,
                             P_S=r["P_S"], P_D=r["P_D"]))
        d = pd.DataFrame([x for x in rows if x["rung"] == rung])
        s = dict(rung=rung, method=m.name, n=len(d), wall_s=round(wall, 2), loci_per_s=round(len(d) / wall, 3),
                 failed=int((d.status == "failed").sum()), nonconverged=int((d.status == "nonconverged").sum()),
                 median_niter=float(d.niter.median()), max_niter=int(d.niter.max()),
                 noshare_chosen=int((d.start == "noshare").sum()), per_trait_ld=int(d.per_trait_ld.fillna(False).sum()))
        for cell, dc in d.groupby("cell"):
            s[f"nonconv[{cell}]"] = int((dc.status == "nonconverged").sum())
            s[f"noshare[{cell}]"] = int((dc.start == "noshare").sum())
        summ.append(s)
        print(json.dumps(s), flush=True)
    pd.DataFrame(rows).to_csv(out / "per_locus.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(out / "summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
