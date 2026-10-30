"""Smoke run of Family A (+ G) on a few validation screening loci (not an evaluation).
Usage: PYTHONPATH=src python reports/famAG_M3.5/scripts/smoke_screen.py data/processed/sets/screen_v1.h5 4 reports/famAG_M3.5/smoke_screen_v1.tsv"""
import json, sys, time
import numpy as np, pandas as pd
from eval.io import read_locus_attrs, LocusSetReader
from eval.truth import read_truth
path = sys.argv[1]; per_cell = int(sys.argv[2]); out = sys.argv[3]
attrs = read_locus_attrs(path)
assert attrs["split"].isin(["valid", "validation"]).all()
rng = np.random.default_rng(20261029)
attrs["bucket"] = attrs["locus_id"].str.split("__").str[2]
pick = attrs.groupby(["cell_id", "bucket"], group_keys=False).apply(lambda d: d.sample(min(per_cell, len(d)), random_state=1))
ids = pick["locus_id"].tolist() if "locus_id" in pick else pick.index.tolist()
with LocusSetReader(path) as rd:
    loci = [rd.read(i) for i in ids]
print("loci", len(loci), "kinds", pd.Series([l.ld_type for l in loci]).value_counts().to_dict(),
      "p", int(np.median([l.p for l in loci])), "types", pd.Series([str(l.extra_attrs.get('trait1_type'))+'/'+str(l.extra_attrs.get('trait2_type')) for l in loci]).value_counts().to_dict(), flush=True)
import torch
from famA.method import JointSusieMethod
from famA.joint_susie import JointConfig
from famG.method import GPUColocSusie
rows = []
for name, m in [("A_rho_eta", JointSusieMethod(JointConfig(estimate_eta=True))),
                ("A_rho_eta_pifix", JointSusieMethod(JointConfig(estimate_eta=True, pi_mode="fixed"))),
                ("A_rho_eta_E", JointSusieMethod(JointConfig(estimate_eta=True, ref_var=True, convergence="alpha"))),
                ("G", GPUColocSusie())]:
    torch.cuda.synchronize(); t0 = time.time()
    res = m.fit_batch(loci)
    torch.cuda.synchronize(); dt = time.time() - t0
    for r in res:
        r["run"] = name
    rows += res
    st = pd.Series([r["status"] for r in res]).value_counts().to_dict()
    print(name, f"{len(loci)/dt:.2f} loci/s", st, flush=True)
df = pd.DataFrame(rows)
tr = read_truth(path.replace(".h5", ".truth.parquet"))
tr = tr[tr.locus_id.isin(ids)][["locus_id", "oracle_class", "mismatch_level", "cell_id"]]
df = df.merge(tr, on="locus_id")
summ = df.groupby(["run", "oracle_class"])[["P_S", "P_D", "P_W", "s"]].median().round(3)
print(summ.to_string())
df.drop(columns=["extra"]).to_csv(out, sep="\t", index=False)
