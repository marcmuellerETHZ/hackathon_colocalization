"""Identity test G: batched GPU SuSiE-RSS vs susieR 0.14.2 susie_rss on the same loci (DESIGN §5 G, §7.1 item 6).

Usage: python -m famG.validation.compare_identity --work DIR --rdir DIR/r014 --out reports/famAG_M3/identity_G \
           [--dtype float32|float64] [--traits 1,2] [--no-n]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from famG.susie_rss import SusieRSSConfig, susie_rss_batch
from famG.validation.make_loci import LocusStore


def load_r_results(rdir: Path, store: LocusStore, trait: int, prefix="t"):
    out = {}
    for jf in sorted(rdir.glob(f"{prefix}{trait}_c*.json")):
        js = json.loads(jf.read_text())
        bf = np.fromfile(jf.with_suffix(".bin"), dtype=np.float64)
        off = 0
        for lid, rec in js.items():
            p = int(rec["p"])
            L = int(rec.get("L", 10))
            pip = bf[off:off + p]; off += p
            alpha = bf[off:off + L * p].reshape(L, p); off += L * p
            lbfv = bf[off:off + L * p].reshape(L, p); off += L * p
            rec = dict(rec)
            rec.update(pip=pip, alpha=alpha, lbf_variable=lbfv)
            out[lid] = rec
    return out


def _sets(cs):
    return sorted(tuple(sorted(np.atleast_1d(np.asarray(c, dtype=int)).tolist())) for c in cs)


def cs_equal_up_to_ties(g, q, rel=1e-8):
    """CS lists equal, except for members whose posterior alpha is tied (to `rel`) with the swapped member
    (variables in perfect LD have mathematically identical alpha; floating-point rounding breaks the tie)."""
    a = {int(l): set(c) for l, c in zip(g["sets"]["cs_index"], g["sets"]["cs"])}
    b = {int(l): set(np.atleast_1d(np.asarray(c, dtype=int)).tolist()) for l, c in zip(np.atleast_1d(q["cs_index"]), q["cs"])}
    if set(a) != set(b):
        return False
    for l in a:
        dif = sorted(a[l] ^ b[l])
        if not dif:
            continue
        vals = q["alpha"][l, dif]
        if vals.max() - vals.min() > rel * vals.max():
            return False
    return True


def compare(store: LocusStore, gpu: list, rres: dict, trait: int):
    rows = []
    for i in range(len(store)):
        r = store.row(i)
        lid = r.locus_id
        g = gpu[i]
        q = rres.get(lid)
        row = dict(locus_id=lid, trait=trait, p=int(r.p), ld_kind=r.ld_kind, scenario=r.scenario, n=float(r.n if trait == 1 else r.n2))
        if q is None:
            row["status"] = "missing_R"
            rows.append(row)
            continue
        if "error" in q:
            row.update(status="R_error", r_error=q["error"], gpu_status=g["status"])
            rows.append(row)
            continue
        Vr = np.atleast_1d(np.asarray(q["V"], dtype=float))
        row.update(
            status="ok", gpu_status=g["status"],
            max_dpip=float(np.max(np.abs(g["pip"] - q["pip"]))),
            max_dalpha=float(np.max(np.abs(g["alpha"] - q["alpha"]))),
            max_dlbf=float(np.nanmax(np.abs(g["lbf_variable"] - q["lbf_variable"]))) if np.isfinite(q["lbf_variable"]).any() else np.nan,
            max_dV_rel=float(np.max(np.abs(g["V"] - Vr) / np.maximum(np.abs(Vr), 1e-12))),
            niter_gpu=g["niter"], niter_r=int(q["niter"]),
            conv_gpu=g["converged"], conv_r=bool(q["converged"]),
            elbo_gpu=float(g["elbo"][-1]) if len(g["elbo"]) else np.nan,
            elbo_r=float(np.atleast_1d(q["elbo"])[-1]),
            n_cs_gpu=len(g["sets"]["cs"]), n_cs_r=len(q["cs"]),
            cs_equal=_sets(g["sets"]["cs"]) == _sets(q["cs"]),
            cs_equal_ties=cs_equal_up_to_ties(g, q),
            max_cs_size_r=max([len(np.atleast_1d(c)) for c in q["cs"]], default=0),
            r_time=float(q["time"]), r_warnings=";".join(np.atleast_1d(q.get("warnings", [])).astype(str)) if q.get("warnings") else "",
        )
        row["delbo"] = row["elbo_gpu"] - row["elbo_r"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_gpu(store: LocusStore, trait: int, cfg: SusieRSSConfig, use_n=True):
    zs, lds, ns, kinds = [], [], [], []
    for i in range(len(store)):
        r = store.row(i)
        kind, ld = store.ld_input(i, dense_dtype=np.float32 if cfg.ld_dtype == torch.float32 else np.float64)
        zs.append(store.z(i, trait)); lds.append(ld); kinds.append(kind)
        ns.append(float(r.n if trait == 1 else r.n2) if use_n else None)
    torch.cuda.synchronize()
    t0 = time.time()
    res = susie_rss_batch(zs, lds, ns, kinds, cfg)
    torch.cuda.synchronize()
    return res, time.time() - t0


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df.status == "ok"]
    s = {
        "n_loci_traits": int(len(df)), "n_compared": int(len(ok)),
        "n_R_error": int((df.status == "R_error").sum()),
        "max_dpip": float(ok.max_dpip.max()), "q99_dpip": float(ok.max_dpip.quantile(0.99)),
        "median_dpip": float(ok.max_dpip.median()),
        "n_dpip_ge_1e-3": int((ok.max_dpip >= 1e-3).sum()),
        "max_dalpha": float(ok.max_dalpha.max()), "max_dlbf": float(ok.max_dlbf.max()),
        "niter_equal_frac": float((ok.niter_gpu == ok.niter_r).mean()),
        "converged_equal_frac": float((ok.conv_gpu == ok.conv_r).mean()),
        "cs_equal_frac": float(ok.cs_equal.mean()),
        "cs_equal_up_to_alpha_ties_frac": float(ok.cs_equal_ties.mean()),
        "max_abs_delbo": float(ok.delbo.abs().max()),
    }
    for scen in ["consistent", "mismatch"]:
        sub = ok[ok.scenario == scen]
        if len(sub):
            s[f"max_dpip_{scen}"] = float(sub.max_dpip.max())
            s[f"n_dpip_ge_1e-3_{scen}"] = int((sub.max_dpip >= 1e-3).sum())
    for kind in ["dense", "factor"]:
        sub = ok[ok.ld_kind == kind]
        if len(sub):
            s[f"max_dpip_{kind}"] = float(sub.max_dpip.max())
    if s["n_R_error"]:
        e = df[df.status == "R_error"]
        s["R_errors_gpu_status"] = e.gpu_status.value_counts().to_dict()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--rdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--traits", default="1,2")
    ap.add_argument("--rprefix", default="t")
    ap.add_argument("--no-n", action="store_true")
    a = ap.parse_args()
    store = LocusStore(Path(a.work))
    cfg = SusieRSSConfig(ld_dtype=getattr(torch, a.dtype))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dfs, times = [], {}
    for t in [int(x) for x in a.traits.split(",")]:
        gpu, el = run_gpu(store, t, cfg, use_n=not a.no_n)
        times[f"trait{t}_gpu_s"] = el
        rres = load_r_results(Path(a.rdir), store, t, a.rprefix)
        dfs.append(compare(store, gpu, rres, t))
    df = pd.concat(dfs)
    tag = f"{a.dtype}{'_non' if a.no_n else ''}"
    df.to_csv(out / f"per_locus_{tag}.tsv", sep="\t", index=False)
    s = summarize(df)
    s.update(times)
    s["gpu_total_s"] = sum(times.values())
    s["r_total_s_single_thread"] = float(df.r_time.sum())
    (out / f"summary_{tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
