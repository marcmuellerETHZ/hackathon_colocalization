"""Calibration and bias of the D6(d) rho-hat estimator on realistic simulated loci (validation split).

For each configuration (pairing x mismatch level) and locus type (N = null, S, D, W), loci are generated with
the main generator (`draw_candidate` + `finalize`) on validation windows, and rho-hat is recomputed on the
locus' z-scores and given LD with:
  * variants: old = M3 estimator (greedy pruning r² < 0.1, unit variances, observed-information SE);
    p_m = pruned + D6(d) signal mask (r² > 0.1 around max|z| > 4); n_m = mask without pruning; p_nm = pruned,
    no mask; 1p = unit marginal variances, 3p = free marginal variances; SE: info (independent SNPs),
    jkB (delete-one-block jackknife, B contiguous blocks), ld (3p info SE x sqrt(LD design effect D),
    D = mean over used SNPs of sum_k r²_adj(j, k) in the given LD); attr = the value stored by the generator.
Reported per group: bias = mean(rho_hat - rho) (+ MC SE), SD(rho_hat - rho), RMS reported SE, and the ratio
SD / RMS SE (calibrated if within [0.85, 1.15]).

  python -m sim.validate_rho --per-group 400 --out src/sim/validation/rho_calibration
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import VALID_SEED, rng
from sim import params
from sim.generator import CellSpec, WindowContext, assert_no_tf32, draw_candidate, finalize
from sim.windows import split_windows
from stats import rho as R

CONFIGS = [("P1_nested_rp0.6", "P1", 0.6, "nested", None), ("P1_disjoint", "P1", 0.0, "disjoint", None),
           ("P3_eqtl_rho0.5", "P3", 0.5, None, "eqtl")]
LEVELS = ("L0", "L2")
TYPES = ("N", "S", "Sc", "D", "W")
BLOCKS = (10, 20, 40)
FAST = False


def _r2_adj(Rsub: np.ndarray, n_ref: float) -> np.ndarray:
    r2 = Rsub * Rsub
    return r2 - (1 - r2) / max(n_ref - 2, 1.0) if n_ref < 1e6 else r2


def variants(z1, z2, Rd, n_ref, g) -> dict:
    """rho-hat variants on one locus: (rho, se) per estimator name (see module docstring)."""
    out = {}
    pr = R.prune(Rd, 0.1, g.permutation(len(z1)))
    pm = np.zeros(len(z1), bool)
    pm[pr] = True
    mask = R.signal_mask(z1, z2, Rd)
    small = (np.abs(z1) < 2) & (np.abs(z2) < 2)
    sets = {"old": pm & small, "p_m": pm & ~mask & small, "n_m": ~mask & small, "p_nm": pm & small}
    if FAST:
        sets = {"old": sets["old"], "p_m": sets["p_m"]}
    # residualised variant: project the strong signals (|z| > 4) out of both traits, then mask + prune
    sig = np.where(np.maximum(np.abs(z1), np.abs(z2)) > 4)[0]
    r1z, r2z, kC = R.residualize(z1, z2, Rd, sig)
    rs = pm & ~mask & (np.abs(r1z) < 2) & (np.abs(r2z) < 2)
    if rs.sum() >= 10:
        a, b = r1z[rs], r2z[rs]
        rr = float(R.mle_from_stats(int(rs.sum()), float(a @ a + b @ b), float(a @ b)))
        out["r_p_m_1p_info"] = (rr, R._info_se(rr, int(rs.sum()), float(a @ a + b @ b), float(a @ b), 2.0))
        n, sqq, s12 = R.block_stats(r1z, r2z, rs, 10)
        out["r_p_m_1p_jk10"] = R.jackknife_from_blocks(n, sqq, s12)
        if not FAST:
            bs = R.block_stats4(r1z, r2z, rs, 10)
            rj, sej, _, _ = R.jackknife3(*bs)
            out["r_p_m_3p_jk10"] = (rj, sej)
        out["_kC"] = (kC, int(rs.sum()))
    for name, use in sets.items():
        n_u = int(use.sum())
        if n_u < 10:
            continue
        a, b = z1[use], z2[use]
        S11, S22, S12 = float(a @ a), float(b @ b), float(a @ b)
        # 1-parameter (unit variances)
        r1 = float(R.mle_from_stats(n_u, S11 + S22, S12))
        if name in ("old", "p_m"):
            out[f"{name}_1p_info"] = (r1, R._info_se(r1, n_u, S11 + S22, S12, 2.0))
        if name == "p_m":
            n, sqq, s12 = R.block_stats(z1, z2, use, 10)
            out["p_m_1p_jk10"] = R.jackknife_from_blocks(n, sqq, s12)
        if FAST:
            continue
        # 3-parameter
        est = R.mle3_from_stats(n_u, S11, S22, S12)
        if name in ("p_m", "p_nm"):
            se3 = R.info_se3(n_u, S11, S22, S12, est=est)
            out[f"{name}_3p_info"] = (est[0], se3)
        if name in ("p_m", "n_m", "p_nm"):
            for B in (10, 20):
                if name != "p_m" and B == 20:
                    continue
                bs = R.block_stats4(z1, z2, use, B)
                rj, sej, _, _ = R.jackknife3(*bs)
                out[f"{name}_3p_jk{B}"] = (rj, sej)
        if name == "p_m":
            idx = np.where(use)[0]
            Ds = _r2_adj(Rd[np.ix_(idx, idx)], n_ref)
            np.fill_diagonal(Ds, 1.0)
            D = float(Ds.sum() / len(idx))
            out["p_m_3p_ld"] = (est[0], se3 * np.sqrt(max(D, 1.0)))
            out["_D"] = (D, n_u)
    return out


def run(per_group: int, loci_per_window: int, out: Path, log=print):
    assert_no_tf32()
    wins = split_windows("valid")
    order = list(rng(VALID_SEED, "validate_rho", "windows").permutation(len(wins)))
    rows = []
    need = {(c[0], L, t): per_group for c in CONFIGS for L in LEVELS for t in TYPES}
    t0 = time.time()
    v = 0
    while any(n > 0 for n in need.values()):
        w = wins.iloc[order[v % len(order)]]
        v += 1
        ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split="valid",
                            block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]), device="cuda")
        if ctx.p < 200:
            continue
        for (name, pairing, ov, p1m, p3m) in CONFIGS:
            for lvl in LEVELS:
                for t in TYPES:
                    key = (name, lvl, t)
                    for k in range(min(loci_per_window, need[key])):
                        cell = CellSpec(f"{lvl}|{name}", lvl, pairing, ov, p1_mode=p1m, p3_mode=p3m,
                                        ltype_weights=((t, 1.0),))
                        g = rng(VALID_SEED, "validate_rho", name, lvl, t, v, k)
                        c = draw_candidate(ctx, cell, g)
                        if c is None:
                            continue
                        loc, tr, _ = finalize(ctx, c, g, f"x{v}_{k}")
                        Rd = loc.ld()
                        row = {"config": name, "level": lvl, "type": t, "oracle": tr["oracle_class"],
                               "rho": tr["rho"], "p": loc.p, "lead1": tr["lead_abs_z1"], "lead2": tr["lead_abs_z2"],
                               "N1": tr["N1"], "N2": tr["N2"]}
                        for k_, (rv, sv) in variants(loc.z1, loc.z2, Rd, loc.n_ref, g).items():
                            row[f"rho_{k_}"], row[f"se_{k_}"] = rv, sv
                        row["rho_attr"], row["se_attr"] = loc.rho_hat, loc.rho_se
                        rows.append(row)
                        need[key] -= 1
        ctx.release()
        del ctx
        if v % 10 == 0:
            log(f"[{time.time()-t0:6.0f}s] windows {v}, loci {len(rows)}, remaining {sum(need.values())}")
    import pandas as pd
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out) + ".parquet")
    summ = summarize(df)
    Path(str(out) + ".json").write_text(json.dumps({"per_group": per_group, "runtime_s": time.time() - t0,
                                                    "summary": summ}, indent=1))
    log(json.dumps(summ, indent=1))
    return df


def summarize(df):
    out = []
    names = sorted({c[4:] for c in df.columns if c.startswith("rho_") and not c.startswith("rho__")} - {"attr"}) + ["attr"]
    for keys, sub in df.groupby(["config", "level", "type"]):
        rec = {"config": keys[0], "level": keys[1], "type": keys[2], "n": int(len(sub))}
        for tag in names:
            if f"rho_{tag}" not in sub:
                continue
            d = (sub[f"rho_{tag}"] - sub["rho"]).to_numpy(float)
            se = sub[f"se_{tag}"].to_numpy(float)
            ok = np.isfinite(d) & np.isfinite(se)
            d, se = d[ok], se[ok]
            if len(d) < 5:
                continue
            rec[tag] = {"bias": float(d.mean()), "bias_mc_se": float(d.std() / np.sqrt(len(d))),
                        "sd": float(d.std()), "rms_se": float(np.sqrt((se ** 2).mean())),
                        "ratio_sd_over_se": float(d.std() / np.sqrt((se ** 2).mean())), "n_ok": int(ok.sum())}
        rec["oracle_share"] = sub["oracle"].value_counts(normalize=True).round(3).to_dict()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=400)
    ap.add_argument("--loci-per-window", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "validation/rho_calibration"))
    ap.add_argument("--fast", action="store_true", help="only old, p_m and residualised variants")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=None)
    a = ap.parse_args()
    global FAST, CONFIGS, TYPES, LEVELS
    FAST = a.fast
    if a.configs:
        CONFIGS = [c for c in CONFIGS if c[0] in a.configs]
    if a.types:
        TYPES = tuple(a.types)
    if a.levels:
        LEVELS = tuple(a.levels)
    run(a.per_group, a.loci_per_window, Path(a.out), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
