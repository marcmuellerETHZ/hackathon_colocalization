#!/usr/bin/env python
"""Regenerate the pipeline example inputs from the project's simulated VALIDATION locus set.

Needs the original project data (not shipped): data/processed/sets/valid_grid_v1.h5 and its truth parquet,
produced by the project's simulator (src/sim) on validation-split LD blocks. Usage, from the repository root:

    PYTHONPATH=src python mvp/example/make_example.py [--data-root <project root>] [--out mvp/example]

Writes, per locus, <name>.sumstats.tsv.gz (snp, beta1, se1, n1, maf1, beta2, se2, n2, maf2) and
<name>.ld_factor.txt.gz (n_ref x p standardised reference genotypes, header = SNP ids; R = X'X / n_ref), plus
<name>.meta.json (locus id, cell, oracle class, rho_hat, n_ref). Numbers are written with 17 significant digits so
that the pipeline sees the same doubles the benchmark used.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LOCI = {
    "L2_S_locus": "valid_grid_v1__L2_P3_rho=0.8__S__0009230",          # credible sets in both traits
    "L2_W_fallback_locus": "valid_grid_v1__L2_P3_rho=0.8__W__0009204",  # no CS in one trait -> coloc.abf fallback
}


def g17(x) -> str:
    return repr(float(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("COLOC_BENCH_ROOT", "."))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    from eval.io import LocusSetReader  # project reader (refuses test-split loci)

    root = Path(a.data_root)
    out = Path(a.out)
    truth = pd.read_parquet(root / "data/processed/sets/valid_grid_v1.truth.parquet").set_index("locus_id")
    with LocusSetReader(root / "data/processed/sets/valid_grid_v1.h5") as rd:
        for name, lid in LOCI.items():
            L = rd.read(lid)
            assert L.split == "valid" and L.ld_type == "factor", (L.split, L.ld_type)
            snp = [str(v) for v in L.variant_id]
            with gzip.open(out / f"{name}.sumstats.tsv.gz", "wt") as fh:
                fh.write("snp\tbeta1\tse1\tn1\tmaf1\tbeta2\tse2\tn2\tmaf2\n")
                for i in range(len(snp)):
                    fh.write("\t".join([snp[i]] + [g17(v[i]) for v in (L.beta1, L.se1, L.n1, L.maf1,
                                                                        L.beta2, L.se2, L.n2, L.maf2)]) + "\n")
            X = np.asarray(L.X_ref, dtype=np.float64)
            with gzip.open(out / f"{name}.ld_factor.txt.gz", "wt") as fh:
                fh.write("\t".join(snp) + "\n")
                for row in X:
                    fh.write("\t".join(g17(v) for v in row) + "\n")
            t = truth.loc[lid]
            meta = {"locus_id": lid, "cell_id": L.cell_id, "split": L.split, "oracle_class": str(t["oracle_class"]),
                    "mismatch_level": L.mismatch_level, "pairing_type": L.pairing_type,
                    "rho_hat": float(L.rho_hat), "rho_true": float(t["rho"]), "n_ref": int(L.n_ref),
                    "n_variants": len(snp), "n1": float(np.median(L.n1)), "n2": float(np.median(L.n2)),
                    "window": f"{L.window_chr}:{L.window_start}-{L.window_end}"}
            (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(name, meta)


if __name__ == "__main__":
    main()
