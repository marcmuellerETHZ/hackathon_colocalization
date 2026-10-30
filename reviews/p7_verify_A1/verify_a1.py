"""Independent recomputation of Family A kill test A1 (p7-verify-A1).

Does NOT import src/eval/metrics.py or src/famA. Implements DESIGN v4 metric definitions:
  * S-vs-D AUPRC on score `s` (§4.3 / §8.1), step-wise average precision with ties as one threshold;
  * power matching (§2.4, §8.6): S and D reweighted to equal joint distributions of the
    class-determining (mu1, mu2) in 5x5 trait-specific quantile bins (quantiles on pooled S u D of
    the cell); weighted S prevalence 0.5;
  * failures (status failed/timeout or NaN s) least favourable: s = 0 for S, s = 1 for D;
    non-converged runs with output are used.
Usage: python verify_a1.py <A_pred.parquet> <baseline_pred.parquet> <truth.parquet>
"""
import sys, json
import numpy as np, pandas as pd

A_NAME = "famA_joint_susie_rho_eta_ms_ptld"
BASE = ["coloc_susie", "sharepro", "mvsusie"]


def ap_weighted(y, s, w):
    """Weighted step-wise AP; tied scores form one threshold."""
    m = w > 0
    y, s, w = y[m], s[m], w[m]
    order = np.argsort(-s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * y); fp = np.cumsum(w * (1 - y))
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # end of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def qbin(x, q=5):
    edges = np.quantile(x, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def pm_weights(y, mu1, mu2, target="pooled"):
    """Power-matching weights. target: mass per bin = pooled (nS+nD), 'S' (nS), 'min'."""
    b = qbin(mu1) * 5 + qbin(mu2)
    nS = np.bincount(b[y == 1], minlength=25).astype(float)
    nD = np.bincount(b[y == 0], minlength=25).astype(float)
    ok = (nS > 0) & (nD > 0)
    mass = {"pooled": nS + nD, "S": nS, "D": nD, "min": np.minimum(nS, nD)}[target] * ok
    with np.errstate(divide="ignore", invalid="ignore"):
        wS = np.where(ok, mass / nS, 0.0); wD = np.where(ok, mass / nD, 0.0)
    w = np.where(y == 1, wS[b], wD[b])
    w[y == 1] /= w[y == 1].sum(); w[y == 0] /= w[y == 0].sum()   # prevalence 0.5
    return w


def scores(pred, ids, y, method, mode="lf"):
    p = pred[pred.method == method].set_index("locus_id").reindex(ids)  # outer join on expected loci
    s = p["s"].to_numpy(float).copy()
    fail = p["status"].isin(["failed", "timeout"]).to_numpy() | np.isnan(s) | p["status"].isna().to_numpy()
    if mode == "lf":
        s[fail] = np.where(y[fail] == 1, 0.0, 1.0)
    return s, fail


def main(a_path, b_path, t_path, n_boot=2000, seed=20261102):
    t = pd.read_parquet(t_path)
    t = t[t.oracle_class.isin(["S", "D"])].reset_index(drop=True)
    ids = t.locus_id.to_numpy(); y = (t.oracle_class == "S").to_numpy(int)
    mu1 = t.mu1.abs().to_numpy(); mu2 = t.mu2.abs().to_numpy()
    pred = pd.concat([pd.read_parquet(a_path), pd.read_parquet(b_path)], ignore_index=True)
    out = {"n_S": int(y.sum()), "n_D": int((1 - y).sum())}
    S = {}
    for m in [A_NAME] + BASE:
        s, fail = scores(pred, ids, y, m); S[m] = s
        r = {"n_fail": int(fail.sum()), "unweighted": ap_weighted(y, s, np.ones(len(y)))}
        for tg in ["pooled", "S", "D", "min"]:
            r["pm_" + tg] = ap_weighted(y, s, pm_weights(y, mu1, mu2, tg))
        keep = ~fail
        r["pm_pooled_excl"] = ap_weighted(y[keep], s[keep], pm_weights(y[keep], mu1[keep], mu2[keep]))
        out[m] = r
    # paired locus bootstrap (weights recomputed per replicate)
    rng = np.random.default_rng(seed); n = len(y)
    reps = {"pm_pooled": [], "unweighted": []}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi = y[i]; w = pm_weights(yi, mu1[i], mu2[i]); u = np.ones(n)
        reps["pm_pooled"].append(ap_weighted(yi, S[A_NAME][i], w) - ap_weighted(yi, S["coloc_susie"][i], w))
        reps["unweighted"].append(ap_weighted(yi, S[A_NAME][i], u) - ap_weighted(yi, S["coloc_susie"][i], u))
    for k, v in reps.items():
        v = np.array(v); d = out[A_NAME][k] - out["coloc_susie"][k]; se = float(v.std(ddof=1))
        best = max(out[m][k] for m in BASE)
        out["diff_" + k] = {"diff": d, "se": se, "ci95_pct": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                            "verdict": "PASS" if out[A_NAME][k] >= best - se else "FAIL"}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
