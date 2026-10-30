"""Family B simulator-prior reference posteriors on an L0* set (task p4.famB_l0star_ref, DESIGN v4 Sec. 5 B/C).

    python -m famB.l0star_ref --set data/processed/sets/l0star_v2.h5 --out results/famB/l0star_ref \
        --n-mc 32768 --reps 4 --compile

B runs in simulator-prior mode with the exact `src/sim/l0star.py` prior (`priors.sim_prior_from_l0star`) and
eta = eta_floor = 0, because L0* sets already store R_tilde = (1 - 1e-3) R + 1e-3 I (the matrix the data were
generated with). p <= M, so there is no screen: the posterior is exact up to Monte Carlo error of the oracle-class
events. The MC error is measured, not assumed: `reps` independent replicate runs, each with `n_mc` stratified draws;
replicate r of locus L draws from derive(20261029, "famB_mc", f"l0star_ref_rep{r}", L) (per-locus keyed, so
results do not depend on batching); the reported posterior is the
replicate mean and its SE is the replicate standard deviation / sqrt(reps).

Output `<out>/posteriors.parquet`, one row per locus:
    locus_id, P_S, P_U, P_D, P_W, P_G      5-class oracle posteriors (natural L0* prior; sum to 1; P_W is the
                                           strict W class, not 1 - P_S - P_D)
    se_S ... se_G                          replicate MC standard errors of the means
    PP_H0 ... PP_H34                       exact hypothesis posteriors (identical across replicates)
    n_mc                                   total draws per locus (reps * n_mc), status
plus `<out>/run.json` (config, prior, timings, MC-error summary). Oracle labels are never read here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("S", "U", "D", "W", "G")
PP_NAMES = ("H0", "H1", "H2", "H3", "H4", "H34")


def check_prior_matches_set(stats_path: Path, prior) -> dict:
    """The set's recorded generator prior (stats.json) must equal the prior B uses."""
    from sim import l0star

    st = json.loads(Path(stats_path).read_text())
    for t in (1, 2):
        w_set = np.asarray(st["lambda_mix"][str(t)]["w"], float)
        s_set = np.asarray(st["lambda_mix"][str(t)]["sigma"], float)
        w_code, s_code = l0star.LAMBDA_MIX[t]
        assert np.allclose(w_set / w_set.sum(), w_code, atol=1e-12), f"lambda mixture weights differ (trait {t})"
        assert np.allclose(s_set, s_code), f"lambda mixture sds differ (trait {t})"
        keep = s_code[(l0star.LAMBDA_MIX[1][0] > 0) | (l0star.LAMBDA_MIX[2][0] > 0)]  # components B keeps
        assert np.allclose(prior.mix_sd[t - 1], keep)
    for k, v in st["type_prior"].items():
        assert abs(v - l0star.TYPE_PRIOR[k]) < 1e-12, f"type prior differs ({k})"
    assert abs(float(st["eta"]) - l0star.ETA) < 1e-15
    assert list(st["kinds"]) == ["gwas", "mol"], "B's thresholds assume trait 1 GWAS-like, trait 2 molecular"
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="data/processed/sets/l0star_v2.h5")
    ap.add_argument("--out", default="results/famB/l0star_ref")
    ap.add_argument("--n-mc", type=int, default=32768)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=200, help="loci per fit_batch call (progress logging)")
    ap.add_argument("--limit", type=int, default=0, help="first N loci only (smoke test)")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args(argv)

    import torch

    from eval.io import LocusSetReader
    from famB.method import ExactColoc
    from famB.priors import sim_prior_from_l0star

    set_path = Path(a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prior = sim_prior_from_l0star()
    prior.validate()
    stats = check_prior_matches_set(set_path.parent / f"{set_path.stem}.stats.json", prior)

    t0 = time.time()
    with LocusSetReader(set_path) as rd:
        ids = rd.ids()
        if a.limit:
            ids = ids[:a.limit]
        loci = [rd.read(i) for i in ids]
    for lo in loci:
        assert lo.ld_type == "dense" and lo.p <= 100, "L0* loci must be dense and unscreened"
        assert float(lo.extra_attrs.get("eta", math.nan)) == stats["eta"]
    print(f"[l0star_ref] {len(loci)} loci read in {time.time() - t0:.1f}s; p = {sorted({lo.p for lo in loci})}",
          flush=True)

    reps = []
    timings = []
    for r in range(a.reps):
        m = ExactColoc(mode="simulator", eta=0.0, eta_floor=0.0, sim_prior_fn=lambda _l: prior, n_mc=a.n_mc,
                       seed=20261029, seed_key=f"l0star_ref_rep{r}", compile=a.compile, M=100)
        t1 = time.time()
        rows = []
        for s in range(0, len(loci), a.chunk):
            rows += m.fit_batch(loci[s:s + a.chunk])
            print(f"[l0star_ref] rep {r}: {min(s + a.chunk, len(loci))}/{len(loci)} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        timings.append(time.time() - t1)
        bad = [x["locus_id"] for x in rows if x["status"] != "ok"]
        if bad:
            raise RuntimeError(f"rep {r}: {len(bad)} loci failed, e.g. {bad[:3]}: "
                               f"{[x['extra'] for x in rows if x['status'] != 'ok'][:1]}")
        df = []
        for x in rows:
            e = json.loads(x["extra"])
            assert e["eta"] == 0.0 and e["M"] == e["p"], (e["eta"], e["M"], e["p"])
            df.append(dict(locus_id=x["locus_id"], S=x["P_S"], U=e["P_U"], D=x["P_D"], W=e["P_W_strict"],
                           G=e["P_G"], **{f"PP_{h}": e["PP"][h] for h in PP_NAMES}, rho_used=e["rho_used"],
                           n_terms=e["n_terms"]))
        reps.append(pd.DataFrame(df).set_index("locus_id").loc[ids])
        torch.cuda.empty_cache()

    P = np.stack([rp[list(CLASSES)].to_numpy() for rp in reps])  # [R, n, 5]
    assert np.allclose(P.sum(-1), 1.0, atol=1e-9), "class posteriors must sum to 1"
    PPs = np.stack([rp[[f"PP_{h}" for h in PP_NAMES]].to_numpy() for rp in reps])
    pp_dev = float(np.abs(PPs - PPs[:1]).max())
    mean = P.mean(0)
    se = P.std(0, ddof=1) / math.sqrt(a.reps) if a.reps > 1 else np.full_like(mean, np.nan)
    n_tot = a.reps * a.n_mc
    se_binom = np.sqrt(mean * (1 - mean) / n_tot)  # i.i.d.-sampling bound (stratified sampling is <= this)

    res = pd.DataFrame({"locus_id": ids})
    for k, c in enumerate(CLASSES):
        res[f"P_{c}"] = mean[:, k]
    for k, c in enumerate(CLASSES):
        res[f"se_{c}"] = se[:, k]
    for k, h in enumerate(PP_NAMES):
        res[f"PP_{h}"] = PPs[0, :, k]
    res["rho_used"] = reps[0]["rho_used"].to_numpy()
    res["n_mc"] = n_tot
    res["status"] = "ok"
    res.to_parquet(out / "posteriors.parquet", index=False)

    mc = {
        "replicates": a.reps, "n_mc_per_replicate": a.n_mc, "n_mc_total": n_tot,
        "se_max": float(np.nanmax(se)), "se_mean": float(np.nanmean(se)),
        "se_p99": float(np.nanpercentile(se, 99)),
        "se_max_per_class": {c: float(np.nanmax(se[:, k])) for k, c in enumerate(CLASSES)},
        "se_mean_per_class": {c: float(np.nanmean(se[:, k])) for k, c in enumerate(CLASSES)},
        "binomial_bound_se_max": float(se_binom.max()),
        # ratio of the empirical replicate variance to the iid binomial variance (pooled; < 1 = stratification gain)
        "variance_ratio_vs_iid": float(np.nansum(se ** 2) / np.sum(se_binom ** 2)),
        "max_abs_diff_exact_PP_across_reps": pp_dev,
    }
    run = {"set": str(set_path), "n_loci": len(ids), "eta": 0.0, "eta_floor": 0.0, "mode": "simulator",
           "prior": {"type_prob_nonzero": {f"{i}{j}{k}": float(prior.type_prob[i, j, k])
                                          for i in range(3) for j in range(3) for k in range(3)
                                          if prior.type_prob[i, j, k] > 0},
                     "mix_w": prior.mix_w.tolist(), "mix_sd": prior.mix_sd.tolist()},
           "n_terms": int(reps[0]["n_terms"].iloc[0]), "compile": a.compile, "replicate_wall_s": timings,
           "mc_error": mc, "commit": os.environ.get("ORCH_COMMIT"), "job": os.environ.get("ORCH_JOB_ID"),
           "seeds": {"seed": 20261029, "per_locus": 'derive(20261029, "famB_mc", f"l0star_ref_rep{r}", locus_id)'},
           "class_mean_posterior": {c: float(mean[:, k].mean()) for k, c in enumerate(CLASSES)}}
    (out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run["mc_error"], indent=1), flush=True)
    print(f"[l0star_ref] wrote {out / 'posteriors.parquet'} ({len(res)} rows) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
