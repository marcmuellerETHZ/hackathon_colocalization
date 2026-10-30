#!/usr/bin/env python
"""p5.verify_infra check (d): coloc.abf posteriors from R coloc 5.2.3 for 5 screen_v2 pilot loci vs the stored
baseline-wrapper output (extra.grid at W = 0.15, p12 = 5e-6 and 1e-5) of the pilot200 coloc_abf run.
Usage: python coloc_r_check.py <project root> <scratch dir>"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root / "src"))
from eval.io import read_locus  # noqa: E402

runs = json.loads((root / "src/baselines/benchmarks/pilot200_screen_v2_runs.json").read_text())["runs"]
pred = pd.read_parquet(root / runs["coloc_abf"]["run_dir"] / "predictions.parquet")
g = np.random.default_rng(20261029)
pick = pred.iloc[g.choice(len(pred), 5, replace=False)]
scratch.mkdir(parents=True, exist_ok=True)
for f in scratch.glob("locus_*.tsv"):
    f.unlink()
for k, r in enumerate(pick.itertuples()):
    L = read_locus(root / "data/processed/sets/screen_v2.h5", r.locus_id)
    pd.DataFrame(dict(snp=L.variant_id, beta1=L.beta1, se1=L.se1, beta2=L.beta2, se2=L.se2, n1=L.n1, n2=L.n2,
                      maf1=L.maf1, maf2=L.maf2)).to_csv(scratch / f"locus_{k}.tsv", sep="\t", index=False)
subprocess.run([str(root / "env/renv/bin/Rscript"), str(Path(__file__).with_name("coloc_r_check.R")), str(scratch)],
               check=True, capture_output=True)
R = pd.read_csv(scratch / "r_pp.tsv", sep="\t")
rows, worst = [], 0.0
for k, r in enumerate(pick.itertuples()):
    e = json.loads(r.extra)
    for p12 in (5e-6, 1e-5):
        py = np.array(e["grid"][f"W=0.15,p12={p12:g}"])
        rr = R[(R.file == f"locus_{k}.tsv") & np.isclose(R.p12, p12)].iloc[0]
        rp = rr[["PP0", "PP1", "PP2", "PP3", "PP4"]].to_numpy(float)
        d = float(np.max(np.abs(py - rp)))
        worst = max(worst, d)
        rows.append(dict(locus_id=r.locus_id, p12=p12, nsnps=int(rr.nsnps), max_abs_diff=d, R_PP3=rp[3], R_PP4=rp[4],
                         wrapper_PP3=py[3], wrapper_PP4=py[4], coloc=rr.coloc_version))
    # the stored s/P_S/P_D columns are the W=0.15, p12=5e-6 row
    assert np.allclose([r.P_S, r.P_D], [e["PP"][4], e["PP"][3]])
print(json.dumps({"n_loci": 5, "max_abs_diff_all_PP": worst, "rows": rows}, indent=1))
sys.exit(0 if worst < 1e-6 else 1)
