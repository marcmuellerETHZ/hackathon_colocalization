"""Independent reviewer checks for the simulator commits 3416f2e / 850218b / fb426c4 (task p2.review_sim).

CPU only; never touches test-split data (no window of any split is materialised; the test-lock checks only call
the access predicates, never a generator with split="test" and access granted).

  python reviews/p2_review_sim_checks.py            # prints a JSON summary; exit 1 if any check fails

Checks
  A  oracle_class: own re-implementation of the DESIGN v4 §1 precedence rules (written from the text, per-rule
     predicates evaluated independently), 200,000 random draws near the thresholds, both trait types; every
     draw gets exactly one class, labels and class-determining (mu1, mu2, pair, d) agree.
  B  OLS statistics (D6(a)): sim.generator.noisy_stats on a CPU toy Study vs statsmodels OLS per SNP
     (intercept; intercept + 2 covariates as at L3); factor-form transform vs the exact OLS t of a toy
     individual-level regression with N = 50,000.
  C  rho-hat: genome-wide mode bias/SE at rho in {0, .3, .5, .8, -.5} (own seeds); per-window mode on a toy
     block-AR(1) LD with exact LD, null and with a concordant shared signal, with/without residualisation.
  D  test lock: generate(split="test") refused; _check_access refused outside scripts/eval_test.py even with
     access granted; behaviour of the call-stack check inside a spawned worker (the test run's default path).
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

RESULTS: dict = {}
FAIL: list = []


def record(name, ok, **info):
    RESULTS[name] = dict(ok=bool(ok), **info)
    if not ok:
        FAIL.append(name)


# =========================================================================================== A oracle
def my_oracle(c1, lam1, mu1, c2, lam2, mu2, r2, k1, k2):
    """Written from DESIGN v4 §1 only."""
    from scipy.stats import norm
    alpha = {"gwas": 5e-8, "mol": 1e-5}
    tau = {"gwas": 6.29, "mol": 5.26}
    weak = {"gwas": 4.61, "mol": 3.58}

    def pw(mu, a):
        za = norm.isf(a / 2)
        return norm.sf(za - abs(mu)) + norm.cdf(-za - abs(mu))

    T1 = {j: (abs(l), abs(m)) for j, l, m in zip(c1, lam1, mu1)}
    T2 = {j: (abs(l), abs(m)) for j, l, m in zip(c2, lam2, mu2)}
    powered1 = [j for j in T1 if T1[j][1] >= tau[k1] and T1[j][0] >= tau[k1]]
    powered2 = [j for j in T2 if T2[j][1] >= tau[k2] and T2[j][0] >= tau[k2]]
    nonweak1 = [j for j in T1 if T1[j][1] >= weak[k1]]
    nonweak2 = [j for j in T2 if T2[j][1] >= weak[k2]]
    shared = [j for j in T1 if j in T2]
    shared_powered = [j for j in shared if j in powered1 and j in powered2]
    shared_nonweak = [j for j in shared if j in nonweak1 and j in nonweak2]
    pair = None
    if powered1 and powered2:
        cands = []
        for a in powered1:
            for b in powered2:
                if a == b:
                    continue
                rr = min(max(r2(a, b), 0.0), 1.0)
                d = min(T1[a][0], T2[b][0]) * math.sqrt(1 - rr)
                cands.append((-rr, d, a, b))
        if cands:
            pair = min(cands)
    # each class's own rule (not precedence-resolved)
    rule = {
        "S": len(shared_powered) > 0,
        "U": bool(powered1 and powered2 and not shared_powered and pair is not None and pair[1] < 2.0),
        "D": bool(powered1 and powered2 and not shared_nonweak),
        "W": (not nonweak1) or (not nonweak2),
    }
    cls = next((c for c in ("S", "U", "D", "W") if rule[c]), "G")
    if cls == "S":
        j = max(sorted(shared_powered), key=lambda k: min(pw(T1[k][1], alpha[k1]), pw(T2[k][1], alpha[k2])))
        cd = (T1[j][1], T2[j][1], j, j)
    elif cls in ("U", "D"):
        cd = (T1[pair[2]][1], T2[pair[3]][1], pair[2], pair[3])
    else:
        m1 = max((T1[j][1] for j in T1), default=0.0)
        m2 = max((T2[j][1] for j in T2), default=0.0)
        cd = (m1, m2, None, None)
    return cls, rule, cd, (None if pair is None else (-pair[0], pair[1]))


def check_oracle(n_draws=200_000):
    from sim.oracle import CLASSES, oracle_class, power
    from sim.params import TAU, WEAK
    from scipy.stats import norm
    # thresholds = 0.8 / 0.2 power, two-sided (DESIGN §1 values)
    thr = {k: (float(power(TAU[k], a)), float(power(WEAK[k], a))) for k, a in (("gwas", 5e-8), ("mol", 1e-5))}
    ok_thr = (TAU == {"gwas": 6.29, "mol": 5.26} and WEAK == {"gwas": 4.61, "mol": 3.58}
              and all(abs(p8 - 0.8) < 0.01 and abs(p2 - 0.2) < 0.01 for p8, p2 in thr.values()))
    record("A0_thresholds", ok_thr, tau=TAU, weak=WEAK, power_at_tau_weak=thr)

    g = np.random.default_rng(777)
    kinds = [("gwas", "mol"), ("mol", "mol"), ("gwas", "gwas"), ("mol", "gwas")]
    counts = {c: 0 for c in CLASSES}
    n_mismatch, n_multi_ok, n_cd_mismatch = 0, 0, 0
    examples = []
    for it in range(n_draws):
        k1, k2 = kinds[it % 4]
        p = 6
        R2 = g.uniform(0, 1, (p, p)) ** g.choice([0.3, 1, 3])
        R2 = (R2 + R2.T) / 2
        np.fill_diagonal(R2, 1.0)
        if g.random() < 0.1:   # exact ties in r²
            R2[:] = np.round(R2, 1)
            np.fill_diagonal(R2, 1.0)
        n1, n2 = g.integers(0, 4), g.integers(0, 4)
        c1 = list(g.choice(p, n1, replace=False))
        c2 = list(g.choice(p, n2, replace=False))
        if c1 and g.random() < 0.4 and c1[0] not in c2:
            c2 = [c1[0]] + c2[: max(0, n2 - 1)]

        def eff(n, k):
            # values concentrated around the weak / powered thresholds, some exactly on them
            base = g.choice([0.5, 3.0, WEAK[k], 5.0, TAU[k], 7.0, 15.0, 40.0], size=n)
            jit = np.where(g.random(n) < 0.3, 0.0, g.normal(0, 0.5, n))
            return (base + jit) * g.choice([-1, 1], n)

        lam1, lam2 = eff(len(c1), k1), eff(len(c2), k2)
        mu1 = np.where(g.random(len(c1)) < 0.5, lam1, eff(len(c1), k1))
        mu2 = np.where(g.random(len(c2)) < 0.5, lam2, eff(len(c2), k2))
        r2f = lambda a, b: float(R2[a, b])
        o = oracle_class(c1, lam1, mu1, c2, lam2, mu2, r2f, k1, k2)
        cls, rule, cd, pr = my_oracle(c1, lam1, mu1, c2, lam2, mu2, r2f, k1, k2)
        counts[o.cls] += 1
        if o.cls != cls:
            n_mismatch += 1
            if len(examples) < 5:
                examples.append(dict(c1=c1, c2=c2, lam1=lam1.tolist(), mu1=mu1.tolist(), lam2=lam2.tolist(),
                                     mu2=mu2.tolist(), kinds=(k1, k2), theirs=o.cls, mine=cls))
            continue
        same_cd = (abs(o.mu1 - cd[0]) < 1e-12 and abs(o.mu2 - cd[1]) < 1e-12)
        if cls in ("S", "D", "U"):
            same_cd &= (o.cd1, o.cd2) == (cd[2], cd[3])
        if cls in ("D", "U"):
            same_cd &= abs(o.r2 - pr[0]) < 1e-12 and abs(o.d - pr[1]) < 1e-12
        n_cd_mismatch += not same_cd
        n_multi_ok += sum(rule.values()) >= 1 or cls == "G"
    record("A1_oracle_partition", n_mismatch == 0 and n_cd_mismatch == 0 and sum(counts.values()) == n_draws,
           draws=n_draws, class_counts=counts, label_mismatches=n_mismatch, class_determining_mismatches=n_cd_mismatch,
           examples=examples)
    # kind arguments are required
    try:
        oracle_class([1], [7], [7], [1], [7], [7], lambda a, b: 1.0)   # type: ignore[call-arg]
        req = False
    except TypeError:
        req = True
    try:
        oracle_class([1], [7], [7], [1], [7], [7], lambda a, b: 1.0, "gwas", "eqtl")
        bad = False
    except ValueError:
        bad = True
    record("A2_trait_types_required", req and bad)
    # power saturation: S with two shared powered variants, both power == 1.0 in fp64 -> the choice is by id
    o = oracle_class([1, 2], [50, 9], [50, 9], [1, 2], [9, 50], [9, 50], lambda a, b: 0.0, "gwas", "mol")
    o2 = oracle_class([1, 2], [12, 40], [12, 40], [1, 2], [40, 12], [40, 12], lambda a, b: 0.0, "gwas", "mol")
    record("A3_S_cd_tie_note", True, note="informational", case1=(o.cd1, o.mu1, o.mu2),
           case2=(o2.cd1, o2.mu1, o2.mu2), power_9_gwas=float(power(9.0, 5e-8)),
           power_12_gwas=float(power(12.0, 5e-8)))


# =========================================================================================== B OLS
def check_ols():
    import statsmodels.api as sm
    import torch
    from sim.generator import Study, TraitSpec, noisy_stats

    g = np.random.default_rng(31)
    N, p = 400, 25
    f = g.uniform(0.05, 0.5, p)
    G = g.binomial(2, f, size=(N, p)).astype(np.float64)
    pcs = g.normal(size=(N, 2)) + 0.3 * (G[:, :2] - G[:, :2].mean(0))   # covariates correlated with genotypes
    worst = {}
    for label, C in (("intercept", None), ("intercept+2PC", pcs)):
        Gt = torch.as_tensor(G, dtype=torch.float32)
        X = Gt - Gt.mean(0)
        sd = X.std(0, unbiased=False)
        X = X / sd
        Q, Xr = None, X
        if C is not None:
            Cm = torch.cat([torch.ones(N, 1, dtype=torch.float64), torch.as_tensor(C)], 1)
            Q, _ = torch.linalg.qr(Cm)
            Q = Q.to(torch.float32)
            Xr = X - Q @ (Q.T @ X)
        st = Study(TraitSpec("mol", N, "indiv", "NFE"), rows=[], X=X, Xr=Xr, af=G.mean(0) / 2, Q=Q, gsd=sd)
        st.xnorm = torch.linalg.vector_norm(Xr, dim=0)
        beta = torch.zeros(p)
        beta[3], beta[7] = 0.3, -0.2
        out = noisy_stats(st, beta, torch.zeros(p), np.random.default_rng(5))
        s = (X @ beta).double()
        var_s = float((s ** 2).mean() - s.mean() ** 2)
        y = (X @ beta).numpy().astype(np.float64) + math.sqrt(1 - var_s) * np.asarray(
            torch.as_tensor(out["eps"], dtype=torch.float32).numpy(), dtype=np.float64)
        rel = {"beta": 0.0, "se": 0.0, "z": 0.0}
        for j in range(p):
            D = np.column_stack([np.ones(N), G[:, j]] + ([] if C is None else [C]))
            fit = sm.OLS(y, D).fit()
            ref = {"beta": fit.params[1], "se": fit.bse[1], "z": fit.tvalues[1]}
            for k in rel:
                got = float(out[k][j])
                rel[k] = max(rel[k], abs(got - ref[k]) / max(abs(ref[k]), 1e-3))
        worst[label] = rel
    ok = all(v < 1e-4 for r in worst.values() for v in r.values())
    record("B1_indiv_OLS_vs_statsmodels", ok, max_rel_err=worst, N=N, p=p)

    # factor-form transform: z = zg / sqrt(1 - zg²/N) with zg = sqrt(N) r vs the exact OLS t = r sqrt(N-2)/sqrt(1-r²)
    N = 50_000
    x = g.binomial(2, 0.3, N).astype(np.float64)
    xs = (x - x.mean()) / x.std()
    rows = []
    for b in (0.0, 0.02, 0.1, 0.3):
        yv = b * xs + math.sqrt(1 - b * b) * g.standard_normal(N)
        fit = sm.OLS(yv, np.column_stack([np.ones(N), x])).fit()
        r = np.corrcoef(xs, yv)[0, 1]
        zg = math.sqrt(N) * r
        z_sim = zg / math.sqrt(1 - zg * zg / N)
        rows.append({"b": b, "t_ols": float(fit.tvalues[1]), "z_sim": z_sim,
                     "rel": abs(z_sim - fit.tvalues[1]) / max(abs(fit.tvalues[1]), 1e-6)})
    record("B2_factor_transform_vs_OLS_t", all(r["rel"] < 1e-4 or abs(r["t_ols"]) < 3 for r in rows)
           and all(abs(r["z_sim"] - r["t_ols"]) < 1e-3 * max(1, abs(r["t_ols"])) for r in rows), rows=rows)


# =========================================================================================== C rho-hat
def check_rho():
    from stats.rho import rho_hat_genomewide, rho_hat_window
    out = {}
    ok = True
    for rho in (0.0, 0.3, 0.5, 0.8, -0.5):
        g = np.random.default_rng([4242, int(round(rho * 100)) + 100])
        est = [rho_hat_genomewide(rho, g) for _ in range(300)]
        r = np.array([e["rho_hat"] for e in est])
        se = np.array([e["rho_se"] for e in est])
        bias, mcse = r.mean() - rho, r.std() / math.sqrt(len(r))
        out[f"gw_rho={rho}"] = {"bias": bias, "mc_se": mcse, "sd": r.std(), "mean_se": se.mean(),
                                "se_ratio": se.mean() / r.std()}
        ok &= abs(bias) < 0.01 and 0.8 < se.mean() / r.std() < 1.25
    # per-window toy: 1,500 SNPs in blocks of 50 with AR(1) 0.9 LD, exact R given
    p, bs = 1500, 50
    A = 0.9 ** np.abs(np.subtract.outer(np.arange(bs), np.arange(bs)))
    R = np.kron(np.eye(p // bs), A)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(p))
    for rho in (0.0, 0.3):
        for signal in (False, True):
            for resid in (False, True):
                g = np.random.default_rng([99, int(rho * 10), int(signal), int(resid)])
                ests, ses = [], []
                for _ in range(200):
                    e1 = L @ g.standard_normal(p)
                    e2 = rho * e1 + math.sqrt(1 - rho * rho) * (L @ g.standard_normal(p))
                    lam1, lam2 = np.zeros(p), np.zeros(p)
                    if signal:   # shared concordant signal, D14 leakage case
                        j = 25 + bs * int(g.integers(p // bs))
                        lam1[j], lam2[j] = 8.0, 8.0
                    z1, z2 = R @ lam1 + e1, R @ lam2 + e2
                    o = rho_hat_window(z1, z2, R, order=g.permutation(p), residualize_signal=resid)
                    ests.append(o["rho_hat"])
                    ses.append(o["rho_se"])
                ests, ses = np.array(ests), np.array(ses)
                k = f"win_rho={rho}_signal={signal}_resid={resid}"
                out[k] = {"bias": ests.mean() - rho, "mc_se": ests.std() / math.sqrt(len(ests)), "sd": ests.std(),
                          "mean_se": np.nanmean(ses), "se_ratio": np.nanmean(ses) / ests.std()}
                if resid or not signal:
                    ok &= abs(ests.mean() - rho) < max(3 * ests.std() / math.sqrt(len(ests)), 0.01) + 0.005
    record("C_rho_hat", ok, **out)


def check_rho_window_paired(reps=1000):
    """C2: residualised vs plain per-window rho-hat on the SAME z (paired), exact LD, null / shared signal."""
    from stats.rho import rho_hat_window
    p, bs = 1500, 50
    A = 0.9 ** np.abs(np.subtract.outer(np.arange(bs), np.arange(bs)))
    R = np.kron(np.eye(p // bs), A)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(p))
    out, ok = {}, True
    for rho in (0.0, 0.3):
        for signal in (False, True):
            g = np.random.default_rng([2026, int(rho * 10), int(signal)])
            est = {False: [], True: []}
            for _ in range(reps):
                e1 = L @ g.standard_normal(p)
                e2 = rho * e1 + math.sqrt(1 - rho * rho) * (L @ g.standard_normal(p))
                lam1, lam2 = np.zeros(p), np.zeros(p)
                if signal:
                    j = 25 + bs * int(g.integers(p // bs))
                    lam1[j], lam2[j] = 8.0, 8.0
                z1, z2 = R @ lam1 + e1, R @ lam2 + e2
                order = g.permutation(p)
                for resid in (False, True):
                    est[resid].append(rho_hat_window(z1, z2, R, order=order, residualize_signal=resid)["rho_hat"])
            a, b = np.array(est[False]), np.array(est[True])
            dlt = b - a
            k = f"rho={rho}_signal={signal}"
            out[k] = {"bias_plain": a.mean() - rho, "bias_resid": b.mean() - rho, "mc_se": a.std() / math.sqrt(reps),
                      "paired_diff": dlt.mean(), "paired_diff_se": dlt.std() / math.sqrt(reps), "reps": reps}
            ok &= abs(b.mean() - rho) < 4 * b.std() / math.sqrt(reps) + 0.005
    record("C2_rho_window_paired", ok, **out)


# =========================================================================================== D test lock
HELPER = textwrap.dedent('''
    def child(_):
        from sim.testset import _called_from_eval_test
        return _called_from_eval_test()
''')
FAKE_EVAL_TEST = textwrap.dedent('''
    import sys, json, multiprocessing as mp
    sys.path[:0] = [{src!r}, {helper_dir!r}]
    from p2rs_helper import child     # pool target in a module, as sim.testset._worker

    if __name__ == "__main__":
        from sim.testset import _called_from_eval_test
        with mp.get_context("spawn").Pool(2) as pool:
            kids = pool.map(child, [0, 1])
        print(json.dumps({{"parent": _called_from_eval_test(), "spawned_children": kids}}))
''')


def check_test_lock():
    import eval.io as eio
    from sim import testset
    from sim.testset import _check_access, generate
    from eval.io import TestSplitAccessError

    res = {}
    os.environ.pop("COLOC_TEST_ACCESS", None)
    try:
        generate({"main_grid": {}}, Path(tempfile.mkdtemp(prefix="claude-p2rs-")) / "never", split="test")
        res["generate_test_refused"] = False
    except TestSplitAccessError:
        res["generate_test_refused"] = True
    orig = eio.has_test_access
    try:
        eio.has_test_access = lambda *a, **k: True        # access granted, but caller is not scripts/eval_test.py
        try:
            _check_access("test", False)
            res["refused_outside_eval_test_with_access"] = False
        except TestSplitAccessError:
            res["refused_outside_eval_test_with_access"] = True
    finally:
        eio.has_test_access = orig
    for split, kw in (("train", {}),):
        from sim.train_api import TrainSampler
        try:
            TrainSampler(split="test", device="cpu")
            res["train_api_test_refused"] = False
        except ValueError:
            res["train_api_test_refused"] = True
    from sim.windows import split_windows
    try:
        split_windows("test")
        res["split_windows_test_refused"] = False
    except PermissionError:
        res["split_windows_test_refused"] = True
    # call-stack check in a spawned worker (generate(..., workers=4) is the default eval_test.py path)
    d = Path(tempfile.mkdtemp(prefix="claude-p2rs-")) / "scripts"
    d.mkdir()
    (d.parent / "p2rs_helper.py").write_text(HELPER)
    (d / "eval_test.py").write_text(FAKE_EVAL_TEST.format(src=str(ROOT / "src"), helper_dir=str(d.parent)))
    p = subprocess.run([sys.executable, str(d / "eval_test.py")], capture_output=True, text=True, timeout=600)
    try:
        res["stack_check"] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        res["stack_check"] = {"stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    has_fix = "in_worker" in testset._check_access.__code__.co_varnames
    res["worker_check_fixed"] = has_fix
    ok = (res["generate_test_refused"] and res["refused_outside_eval_test_with_access"]
          and res["train_api_test_refused"] and res["split_windows_test_refused"])
    record("D_test_lock", ok, **res)


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C", "D"]
    if "A" in which:
        check_oracle()
    if "B" in which:
        check_ols()
    if "C" in which:
        check_rho()
    if "D" in which:
        check_test_lock()
    if "C2" in which:
        check_rho_window_paired()
    print(json.dumps(RESULTS, indent=1, default=float))
    print("FAILED:" if FAIL else "ALL OK", FAIL)
    sys.exit(1 if FAIL else 0)
