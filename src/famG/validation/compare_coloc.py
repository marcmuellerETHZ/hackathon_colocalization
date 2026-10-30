"""End-to-end check of the G Method (GPU SuSiE-RSS x2 + GPU bf_bf + adapter) vs R coloc-SuSiE
(susieR 0.14.2 + coloc.susie, coloc.abf fallback), same adapter applied to both.

Usage: python -m famG.validation.compare_coloc --work DIR --rjson-dir DIR/coloc014 --out OUTDIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from famG.locus import LocusData
from famG.method import GPUColocSusie, coloc_susie_adapter
from famG.validation.make_loci import LocusStore


def store_loci(store: LocusStore) -> list[LocusData]:
    loci = []
    for i in range(len(store)):
        r = store.row(i)
        kind, ld = store.ld_input(i)
        p = int(r.p)
        maf = store.maf(i)
        loci.append(LocusData(locus_id=r.locus_id, z1=store.z(i, 1), z2=store.z(i, 2), n1=np.full(p, float(r.n)),
                              n2=np.full(p, float(r.n2)), ld_type=kind, R=ld if kind == "dense" else None,
                              X_ref=ld if kind == "factor" else None, n_ref=int(r.n_ref), maf1=maf, maf2=maf))
    return loci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rjson-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    store = LocusStore(Path(a.work))
    loci = store_loci(store)
    m = GPUColocSusie()
    m.fit_batch(loci[:2])  # warm-up
    torch.cuda.synchronize(); t0 = time.time()
    preds = m.fit_batch(loci)
    torch.cuda.synchronize(); t_gpu = time.time() - t0
    rj = {}
    for f in sorted(Path(a.rjson_dir).glob("c*.json")):
        rj.update(json.loads(f.read_text()))
    rows = []
    for pr in preds:
        q = rj.get(pr["locus_id"])
        row = dict(locus_id=pr["locus_id"], s=pr["s"], P_S=pr["P_S"], P_D=pr["P_D"], status=pr["status"],
                   fallback=json.loads(pr["extra"])["fallback_abf"])
        if q is None or "error" in q:
            row["r_status"] = "missing" if q is None else "error: " + str(q.get("error"))
        else:
            pp = np.asarray(q["pp"], dtype=float).reshape(-1, 5)
            s, P_S, P_D, P_W, fb = coloc_susie_adapter(None if q["fallback"] else pp, pp if q["fallback"] else None)
            row.update(r_status="ok", r_s=s, r_P_S=P_S, r_P_D=P_D, r_fallback=bool(q["fallback"]), r_time=float(q["time"]),
                       d_s=abs(pr["s"] - s), d_PS=abs(pr["P_S"] - P_S), d_PD=abs(pr["P_D"] - P_D))
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df[df.r_status == "ok"]
    summ = dict(n=len(df), n_compared=len(ok), fallback_agree=float((ok.fallback == ok.r_fallback).mean()),
                max_d_s=float(ok.d_s.max()), max_d_PS=float(ok.d_PS.max()), max_d_PD=float(ok.d_PD.max()),
                n_fallback=int(ok.r_fallback.sum()), gpu_s=t_gpu, r_s_single_thread=float(ok.r_time.sum()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "coloc_per_locus.tsv", sep="\t", index=False)
    (out / "coloc_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
