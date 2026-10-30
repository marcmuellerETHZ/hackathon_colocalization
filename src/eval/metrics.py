"""The SINGLE metrics implementation (DESIGN §7.2, §8). Hash-protected via configs/FROZEN_SHA256.

Layout
  * Elementary metrics on arrays: `average_precision`, `roc_auc`, `ece`, `brier`, ...
  * `EvalFrame`: truth + one or more methods' predictions aligned on the expected locus list (outer join,
    DESIGN §4.3 failure scoring; modes "least_favourable" and "excluded").
  * Weights: power matching (S vs D) and cell pooling (DESIGN §2.4, §8.1, §8.6), recomputed from whatever
    rows a frame holds, so a bootstrap replicate recomputes them automatically.
  * Endpoint functions taking (frame, method, setting).
  * Uncertainty: cluster bootstrap with BCa intervals over LD blocks, paired differences, sign test.
  * `compute_run_metrics`: what the harness writes to metrics.json and scripts/verify_run.py recomputes.

All decisions not fixed by DESIGN are listed in src/eval/README.md ("Decisions / assumptions").
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

SCORE_COLS = ("s", "P_S", "P_D", "P_W", "E1", "E2")
FAIL_STATUSES = ("failed", "timeout")
REALISTIC_LEVELS = ("L1", "L2", "L3", "L4", "L5", "L6")  # L5 deferred; included if ever present
IDEAL_LEVEL = "L0"
SHIFTED_LEVELS = ("L1swap", "L2swap")  # shifted test arm (DESIGN v4 §8.5/§8.6), pool-swap world
DEFAULT_PAIRING_WEIGHTS = {"P1": 0.5, "P2": 0.25, "P3": 0.25}
# r² / d strata of D loci (DESIGN v4 §8.6, D6(e)): used for STRATIFIED REPORTING only, not for power matching.
# Bins are half-open [lo, hi); D loci outside the edges (r² >= 0.95) are reported in a separate "out" bin.
DEFAULT_R2_EDGES = (0.0, 0.1, 0.3, 0.5, 0.8, 0.95)
DEFAULT_D_EDGES = (2.0, 4.0, 8.0, 16.0, math.inf)
N_QUANTILE_BINS = 5  # power matching: 5 x 5 trait-specific quantile bins of class-determining (|mu1|, |mu2|)
NI_MARGIN = 0.01  # C3 non-inferiority margin (DESIGN §0)
W_LIMIT = 0.025  # C3: confident-S on W <= 2.5% and confident-D on W <= 2.5%
E_GATE = 0.95
FIXED_THRESHOLD = 0.8

# least favourable outcome of a failed locus, by true class (DESIGN §4.3)
_LF = {
    "S": dict(s=0.0, P_S=0.0, P_D=1.0, P_W=0.0, E1=1.0, E2=1.0),  # ranked as D, called D
    "D": dict(s=1.0, P_S=1.0, P_D=0.0, P_W=1.0, E1=0.0, E2=0.0),  # ranked as S, called S, lowest P_D
    "W": dict(s=0.5, P_S=1.0, P_D=1.0, P_W=0.0, E1=1.0, E2=1.0),  # confident call on W
}
_LF["G"] = _LF["W"]
_LF["U"] = _LF["W"]


# ==================================================================================================
# Elementary metrics
# ==================================================================================================
def _prep(y, score, w):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    keep = w > 0
    return y[keep], score[keep], w[keep]


def _grouped_counts(y, score, w):
    """Cumulative weighted TP/FP at each distinct score (descending); ties form ONE threshold."""
    order = np.argsort(-score, kind="mergesort")
    s, yy, ww = score[order], y[order], w[order]
    tp = np.cumsum(ww * yy)
    fp = np.cumsum(ww * ~yy)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    return tp[last], fp[last]


def average_precision(y, score, w=None) -> float:
    """Step-wise average precision: sum_k (R_k - R_{k-1}) P_k over distinct thresholds (ties = one threshold).
    Equals sklearn.metrics.average_precision_score (with sample_weight). NaN if no positive or no row."""
    y, score, w = _prep(y, score, w)
    if len(y) == 0 or not (w * y).sum() > 0:
        return float("nan")
    if np.isnan(score).any():
        raise ValueError("NaN scores reach average_precision; failures must be resolved by EvalFrame")
    tp, fp = _grouped_counts(y, score, w)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y, score, w=None) -> float:
    """Weighted AUROC with ties counted 1/2 (trapezoid on the grouped ROC curve). NaN if a class is empty."""
    y, score, w = _prep(y, score, w)
    if len(y) == 0 or not ((w * y).sum() > 0 and (w * ~y).sum() > 0):
        return float("nan")
    if np.isnan(score).any():
        raise ValueError("NaN scores reach roc_auc")
    tp, fp = _grouped_counts(y, score, w)
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def auc_contributions(y, score, w=None) -> np.ndarray:
    """Per-row additive decomposition of the weighted AUROC: sum(contrib) == roc_auc(y, score, w).

    positive i: w_i * P_w(s_neg < s_i) [+1/2 ties] / (2 W_pos); negative j: w_j * P_w(s_pos > s_j) [+1/2] / (2 W_neg).
    Used by the block-level sign test."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    out = np.zeros(len(y))
    Wp, Wn = (w * y).sum(), (w * ~y).sum()
    if not (Wp > 0 and Wn > 0):
        return out * np.nan

    def frac_below(ref_s, ref_w, query):  # weighted fraction of ref strictly below query + 1/2 equal
        o = np.argsort(ref_s, kind="mergesort")
        rs, cw = ref_s[o], np.r_[0.0, np.cumsum(ref_w[o])]
        lo = np.searchsorted(rs, query, side="left")
        hi = np.searchsorted(rs, query, side="right")
        return (cw[lo] + 0.5 * (cw[hi] - cw[lo])) / cw[-1]

    pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
    out[pos] = w[pos] * frac_below(s[neg], w[neg], s[pos]) / (2 * Wp)
    # 1 - (strictly below + 1/2 equal) = strictly above + 1/2 equal
    out[neg] = w[neg] * (1.0 - frac_below(s[pos], w[pos], s[neg])) / (2 * Wn)
    return out


def weighted_rate(mask, w=None) -> float:
    mask = np.asarray(mask, dtype=bool)
    w = np.ones(len(mask)) if w is None else np.asarray(w, dtype=np.float64)
    tot = w.sum()
    return float((w * mask).sum() / tot) if tot > 0 else float("nan")


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0,1]: sum_b n_b/N |mean(p_b) - mean(y_b)|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def apply_logit_intercept(p, a: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(_logit(p) + a)))


def fit_logit_intercept(p, y) -> float:
    """MLE of a in sigmoid(logit(p) + a) (the common one-parameter recalibration, DESIGN §8.2)."""
    x, y = _logit(p), np.asarray(y, float)
    if len(x) == 0 or y.min() == y.max():
        return float("nan")
    a = 0.0
    for _ in range(100):
        q = 1.0 / (1.0 + np.exp(-(x + a)))
        g = np.sum(y - q)
        h = np.sum(q * (1 - q))
        if h <= 0:
            break
        step = g / h
        a += step
        if abs(step) < 1e-12:
            break
    return float(a)


def sign_test(deltas) -> dict:
    """Two-sided exact binomial sign test on non-zero deltas (|delta| < 1e-12 counted as ties)."""
    d = np.asarray(deltas, float)
    d = d[~np.isnan(d)]
    pos, neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    n = pos + neg
    p = float(stats.binomtest(pos, n, 0.5).pvalue) if n > 0 else float("nan")
    return {"n_pos": pos, "n_neg": neg, "n_tie": int(len(d) - n), "p_value": p}


def matched_threshold(score_w, w=None, alpha: float = W_LIMIT) -> dict:
    """Smallest threshold t (call if score >= t) with weighted rate on W loci <= alpha (DESIGN §8.1/C3).

    Scores that can never be called (gated out) should be passed as -inf. Returns {t, rate}. The threshold is
    placed just above a W score value (np.nextafter), so the rule `score >= t` excludes that value."""
    s = np.asarray(score_w, float)
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    keep = w > 0
    s, w = s[keep], w[keep]
    if len(s) == 0:
        return {"t": float("nan"), "rate": float("nan")}
    if np.isnan(s).any():
        raise ValueError("NaN in W scores for matched threshold")
    tot = w.sum()
    vals = np.unique(s[np.isfinite(s)])
    # candidate thresholds: lowest finite score itself, then just above each distinct value
    cands = np.r_[vals[:1], np.nextafter(vals, np.inf)] if len(vals) else np.array([np.inf])
    for t in cands:
        r = w[s >= t].sum() / tot
        if r <= alpha + 1e-12:
            return {"t": float(t), "rate": float(r)}
    return {"t": float("inf"), "rate": 0.0}


# ==================================================================================================
# EvalFrame: truth + predictions on the expected locus list
# ==================================================================================================
@dataclass
class EvalFrame:
    locus_id: np.ndarray
    cls: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    r2: np.ndarray
    d: np.ndarray
    rho: np.ndarray
    block: np.ndarray
    cell: np.ndarray
    level: np.ndarray
    pairing: np.ndarray
    overlap: np.ndarray
    control: np.ndarray
    scores: dict = field(default_factory=dict)  # method -> {col -> array}
    failed: dict = field(default_factory=dict)  # method -> bool array (any failure, incl. per-column)
    status: dict = field(default_factory=dict)  # method -> status array ("missing" for no row)
    mode: str = "least_favourable"

    _ROW = ("locus_id", "cls", "mu1", "mu2", "r2", "d", "rho", "block", "cell", "level", "pairing",
            "overlap", "control")

    def __len__(self):
        return len(self.locus_id)

    @property
    def methods(self):
        return list(self.scores)

    def take(self, idx) -> "EvalFrame":
        idx = np.asarray(idx)
        kw = {k: getattr(self, k)[idx] for k in self._ROW}
        return EvalFrame(**kw, scores={m: {c: a[idx] for c, a in sc.items()} for m, sc in self.scores.items()},
                         failed={m: a[idx] for m, a in self.failed.items()},
                         status={m: a[idx] for m, a in self.status.items()}, mode=self.mode)

    def mask(self, m) -> "EvalFrame":
        return self.take(np.nonzero(np.asarray(m, bool))[0])


def _col(df, name, default, dtype=None):
    if name in df.columns:
        a = df[name].to_numpy()
    else:
        a = np.full(len(df), default)
    return a.astype(dtype) if dtype is not None else a


def build_eval_frame(truth: pd.DataFrame, preds: dict[str, pd.DataFrame] | pd.DataFrame,
                     mode: str = "least_favourable") -> EvalFrame:
    """Outer join of predictions onto the expected locus list (truth rows). DESIGN §4.3 failure scoring.

    Failure of method m at locus i: no row, status in {failed, timeout}, s NaN, or NaN in P_S / P_D when that
    column is applicable for m (non-NaN for some non-failed locus). `nonconverged` rows with output are used.
    mode="least_favourable": failed values replaced by the class's least favourable outcome.
    mode="excluded": loci failed by ANY method in `preds` are dropped (paired comparisons stay paired).
    mode="raw": no replacement (NaNs remain; for diagnostics only)."""
    if isinstance(preds, pd.DataFrame):
        if preds["method"].nunique() != 1:
            raise ValueError("pass a dict {method: df} for several methods")
        preds = {str(preds["method"].iloc[0]): preds}
    if mode not in ("least_favourable", "excluded", "raw"):
        raise ValueError(mode)
    t = truth.reset_index(drop=True)
    ids = t["locus_id"].astype(str).to_numpy()
    pos = {lid: i for i, lid in enumerate(ids)}
    n = len(ids)
    cls = t["oracle_class"].astype(str).to_numpy()
    fr = EvalFrame(
        locus_id=ids, cls=cls, mu1=np.abs(_col(t, "mu1", np.nan, float)), mu2=np.abs(_col(t, "mu2", np.nan, float)),
        r2=_col(t, "r2", np.nan, float), d=_col(t, "d", np.nan, float), rho=_col(t, "rho", np.nan, float),
        block=_col(t, "block_id", "", str), cell=_col(t, "cell_id", "", str), level=_col(t, "mismatch_level", "", str),
        pairing=_col(t, "pairing_type", "", str), overlap=_col(t, "overlap", np.nan, float),
        control=np.array(["" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)
                          for x in _col(t, "control", "")], dtype=object),
        mode=mode)
    any_failed = np.zeros(n, bool)
    for m, df in preds.items():
        df = df.reset_index(drop=True)
        lid = df["locus_id"].astype(str).to_numpy()
        if pd.Series(lid).duplicated().any():
            raise ValueError(f"{m}: duplicate locus_id in predictions")
        unknown = [x for x in lid if x not in pos]
        if unknown:
            raise ValueError(f"{m}: {len(unknown)} predicted loci not in the expected list, e.g. {unknown[:3]}")
        rows = np.array([pos[x] for x in lid], dtype=np.int64)
        sc = {}
        for c in SCORE_COLS:
            a = np.full(n, np.nan)
            if c in df.columns:
                a[rows] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            sc[c] = a
        st = np.full(n, "missing", dtype=object)
        st[rows] = df["status"].astype(str).to_numpy() if "status" in df.columns else "ok"
        row_failed = (st == "missing") | np.isin(st, FAIL_STATUSES) | np.isnan(sc["s"])
        applicable = {c: bool((~np.isnan(sc[c]) & ~row_failed).any()) for c in SCORE_COLS}
        bad = {}
        for c in SCORE_COLS:
            if c == "s":
                bad[c] = row_failed
            elif not applicable[c]:
                sc[c][:] = np.nan  # column not produced by this method: NaN everywhere, never a failure
                bad[c] = np.zeros(n, bool)
            elif c in ("P_S", "P_D"):
                bad[c] = row_failed | np.isnan(sc[c])
            else:  # P_W, E1, E2: NaN on a non-failed locus is legitimate (E: gate not applied)
                bad[c] = row_failed
        failed = bad["s"] | bad["P_S"] | bad["P_D"]
        if mode == "least_favourable":
            for c in SCORE_COLS:
                if bad[c].any():
                    sc[c][bad[c]] = np.array([_LF[k][c] for k in cls[bad[c]]])
        fr.scores[m] = sc
        fr.failed[m] = failed
        fr.status[m] = st
        any_failed |= failed
    if mode == "excluded":
        fr = fr.mask(~any_failed)
    return fr


# ==================================================================================================
# Weights
# ==================================================================================================
def bin_index(x, edges) -> np.ndarray:
    """Half-open bin index k with edges[k] <= x < edges[k+1]; -1 for NaN or x outside [edges[0], edges[-1])."""
    x = np.asarray(x, float)
    e = np.asarray(edges, float)
    k = np.searchsorted(e, x, side="right") - 1
    k[(k < 0) | (k >= len(e) - 1) | np.isnan(x)] = -1
    return k


def bin_labels(edges) -> list[str]:
    e = list(edges)
    return [f"[{e[i]:g},{e[i + 1]:g})" for i in range(len(e) - 1)]


def _qbins(x, n_q=N_QUANTILE_BINS):
    edges = np.quantile(x, np.linspace(0, 1, n_q + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def power_match_weights(cls, mu1, mu2, stratum, group, n_q: int = N_QUANTILE_BINS) -> np.ndarray:
    """Power-matching weights for S vs D (DESIGN §2.4, §8.6).

    The benchmark calls this with ONE stratum per cell (DESIGN v4 §8.6: quantiles on the pooled S ∪ D loci of
    the cell; r²/d strata are for reporting, not matching). The general form is kept for stratified reports:
    within each group (evaluation cell) and each D stratum b: the joint
    distribution of (|mu1|, |mu2|) in n_q x n_q trait-specific quantile bins (quantiles over S ∪ D_b) is
    equalised between S and D_b: in every 2-D bin k holding both classes, each class gets total mass
    m_k = (n_S,k + n_D,k)/2 spread evenly over its loci; bins holding only one class get weight 0. The
    stratum is then rescaled so the D_b mass equals |D_b|; S loci accumulate weight over strata. Hence in
    each group the S mass equals the D mass (weighted S prevalence 0.5 in S ∪ D). Each group's S ∪ D mass
    is normalised to 1 (pooling weights are applied separately). Rows not S/D get weight 0."""
    cls = np.asarray(cls)
    isS, isD = cls == "S", cls == "D"
    mu1, mu2, stratum = np.asarray(mu1, float), np.asarray(mu2, float), np.asarray(stratum)
    w = np.zeros(len(cls))
    gcode = pd.factorize(np.asarray(group))[0]
    order = np.argsort(gcode, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(gcode))] if len(gcode) else np.r_[0]
    nb = n_q * n_q
    for gi in range(len(bounds) - 1):
        rows = order[bounds[gi]:bounds[gi + 1]]
        S = rows[isS[rows]]
        Dg = rows[isD[rows]]
        if len(S) == 0 or len(Dg) == 0:
            continue
        wS = np.zeros(len(S))
        for b in np.unique(stratum[Dg]):
            D = Dg[stratum[Dg] == b]
            U = np.r_[S, D]
            k = _qbins(mu1[U], n_q) * n_q + _qbins(mu2[U], n_q)
            kS, kD = k[: len(S)], k[len(S):]
            nS = np.bincount(kS, minlength=nb).astype(float)
            nD = np.bincount(kD, minlength=nb).astype(float)
            m = np.where((nS > 0) & (nD > 0), (nS + nD) / 2.0, 0.0)
            tot = m.sum()
            if tot <= 0:
                continue
            scale = len(D) / tot
            with np.errstate(divide="ignore", invalid="ignore"):
                perS = np.where(nS > 0, m / nS, 0.0) * scale
                perD = np.where(nD > 0, m / nD, 0.0) * scale
            wS += perS[kS]
            w[D] = perD[kD]
        w[S] = wS
        mass = w[rows].sum()
        if mass > 0:
            w[rows] /= mass
    return w


def cell_pooling_weights(frame: EvalFrame, setting: str,
                         pairing_weights: dict | None = None) -> dict:
    """Cell -> pooling weight for a setting (DESIGN §8.1, §8.6), computed over the cells present.

    realistic: levels L1–L6 present, equal weight per level; pairing types P1 .5 / P2 .25 / P3 .25 (renormalised
               over types present), equal within type over the overlap values present.
    ideal:     level L0 and overlap == 0; pairing weights as above.
    shifted:   levels L1swap/L2swap (shifted arm), same rule as realistic (equal over levels and over cells).
    all:       every cell equal weight.
    level:<L>: cells of level L, pairing weights as realistic.  cell:<id>: that cell only.
    Without level/pairing metadata (empty strings) every cell gets equal weight."""
    pw = DEFAULT_PAIRING_WEIGHTS if pairing_weights is None else pairing_weights
    _, first = np.unique(frame.cell, return_index=True)
    first = np.sort(first)
    cells = pd.DataFrame({"cell": frame.cell[first], "level": frame.level[first], "pairing": frame.pairing[first],
                          "overlap": frame.overlap[first]})
    if setting.startswith("cell:"):
        cid = setting[5:]
        return {cid: 1.0} if cid in set(cells["cell"]) else {}
    if setting == "all" or (cells["level"] == "").all():
        if setting not in ("all", "realistic"):
            return {}
        return {c: 1.0 / len(cells) for c in cells["cell"]}
    if setting == "realistic":
        sel = cells[cells["level"].isin(REALISTIC_LEVELS)]
    elif setting == "shifted":
        sel = cells[cells["level"].isin(SHIFTED_LEVELS)]
    elif setting == "ideal":
        sel = cells[(cells["level"] == IDEAL_LEVEL) & (np.nan_to_num(cells["overlap"].astype(float), nan=1) == 0)]
    elif setting.startswith("level:"):
        sel = cells[cells["level"] == setting[6:]]
    else:
        raise ValueError(f"unknown setting {setting!r}")
    if sel.empty:
        return {}
    levels = sorted(sel["level"].unique())
    out = {}
    for L in levels:
        sl = sel[sel["level"] == L]
        types = sorted(sl["pairing"].unique())
        tw = {p: pw.get(p, 0.0) for p in types}
        tot = sum(tw.values())
        if tot <= 0:
            tw = {p: 1.0 for p in types}
            tot = float(len(types))
        for p in types:
            sp = sl[sl["pairing"] == p]
            for c in sp["cell"]:
                out[c] = (1.0 / len(levels)) * (tw[p] / tot) / len(sp)
    return out


def _map_cells(cell, cw: dict) -> np.ndarray:
    return pd.Series(cell).map(cw).fillna(0.0).to_numpy(dtype=float)


def sd_weights(frame: EvalFrame, setting: str, pairing_weights: dict | None = None) -> np.ndarray:
    """Final S-vs-D weights: power matching within cell (5x5 trait-specific quantile bins of the class-determining
    (|mu1|, |mu2|), quantiles over the pooled S ∪ D loci of the cell; DESIGN v4 §8.6) x cell pooling weight.
    0 outside the setting / S∪D. Only cells present in the setting are matched (the rest get weight 0)."""
    cw = cell_pooling_weights(frame, setting, pairing_weights)
    cwa = _map_cells(frame.cell, cw)
    use = cwa > 0
    pm = np.zeros(len(frame))
    if use.any():
        idx = np.nonzero(use)[0]
        pm[idx] = power_match_weights(frame.cls[idx], frame.mu1[idx], frame.mu2[idx], np.zeros(len(idx), int),
                                      frame.cell[idx])
    return pm * cwa


def dw_weights(frame: EvalFrame, setting: str, pairing_weights: dict | None = None) -> np.ndarray:
    """D-vs-W weights: within each cell D and W each get half the cell's pooling weight, uniform within class."""
    cw = cell_pooling_weights(frame, setting, pairing_weights)
    cwa = _map_cells(frame.cell, cw)
    w = np.zeros(len(frame))
    for k in ("D", "W"):
        m = (frame.cls == k) & (cwa > 0)
        if m.any():
            cnt = pd.Series(frame.cell[m]).value_counts()
            w[m] = cwa[m] * 0.5 / pd.Series(frame.cell[m]).map(cnt).to_numpy(dtype=float)
    return w


# ==================================================================================================
# Endpoints
# ==================================================================================================
@dataclass
class MetricConfig:
    """Metric parameters recorded in every manifest. r2_edges / d_edges: strata of D loci for STRATIFIED
    REPORTING (DESIGN v4 §8.6); power matching never uses them. t_S / t_D: thresholds of the three-way call
    (the C3 thresholds from a validation run when evaluating a later run; 0.8 by default)."""
    r2_edges: tuple = DEFAULT_R2_EDGES
    d_edges: tuple = DEFAULT_D_EDGES
    pairing_weights: dict = field(default_factory=lambda: dict(DEFAULT_PAIRING_WEIGHTS))
    t_S: float = FIXED_THRESHOLD
    t_D: float = FIXED_THRESHOLD
    recal_a_S: float | None = None  # logit-intercept fitted on validation (None: not applied)
    recal_a_D: float | None = None
    w_limit: float = W_LIMIT
    e_gate: float = E_GATE
    shuffle_seed: int = 20261029

    @classmethod
    def from_dict(cls, d: dict | None) -> "MetricConfig":
        d = dict(d or {})
        for k in ("r2_edges", "d_edges"):
            if d.get(k) is not None:
                d[k] = tuple(float(x) for x in d[k])
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def auprc_sd(frame: EvalFrame, method: str, setting: str = "realistic", cfg: MetricConfig = MetricConfig()) -> float:
    w = sd_weights(frame, setting, cfg.pairing_weights)
    m = w > 0
    return average_precision(frame.cls[m] == "S", frame.scores[method]["s"][m], w[m])


def auroc_sd(frame: EvalFrame, method: str, setting: str = "realistic", cfg: MetricConfig = MetricConfig()) -> float:
    w = sd_weights(frame, setting, cfg.pairing_weights)
    m = w > 0
    return roc_auc(frame.cls[m] == "S", frame.scores[method]["s"][m], w[m])


def auroc_dw(frame: EvalFrame, method: str, setting: str = "realistic", cfg: MetricConfig = MetricConfig()) -> float:
    pd_ = frame.scores[method]["P_D"]
    if np.isnan(pd_).all():
        return float("nan")  # not applicable (e.g. proportional test)
    w = dw_weights(frame, setting, cfg.pairing_weights)
    m = w > 0
    return roc_auc(frame.cls[m] == "D", pd_[m], w[m])


def _gate(frame: EvalFrame, method: str, e_gate: float) -> np.ndarray:
    sc = frame.scores[method]
    g = np.ones(len(frame), bool)
    for c in ("E1", "E2"):
        e = sc[c]
        g &= np.isnan(e) | (e >= e_gate)
    return g


def calls_from_arrays(PS, PD, E1, E2, t_S: float, t_D: float, e_gate: float = E_GATE, failed=None) -> np.ndarray:
    """DESIGN §1: S if P_S >= t_S; D if P_D >= t_D and E1,E2 >= e_gate (where available, NaN = not applied);
    else insufficient. 'NA' if the method defines neither P_S nor P_D, or for failed loci (`failed` mask)."""
    PS, PD, E1, E2 = (np.asarray(x, float) for x in (PS, PD, E1, E2))
    n = len(PS)
    if np.isnan(PS).all() and np.isnan(PD).all():
        return np.full(n, "NA", dtype=object)
    gate = (np.isnan(E1) | (E1 >= e_gate)) & (np.isnan(E2) | (E2 >= e_gate))
    callS = np.nan_to_num(PS, nan=-np.inf) >= t_S
    callD = ~callS & (np.nan_to_num(PD, nan=-np.inf) >= t_D) & gate
    out = np.where(callS, "S", np.where(callD, "D", "insufficient")).astype(object)
    if failed is not None:
        out[np.asarray(failed, bool)] = "NA"
    return out


def three_way_calls(frame: EvalFrame, method: str, t_S: float, t_D: float, e_gate: float = E_GATE) -> np.ndarray:
    """Three-way calls on a frame (failed loci already carry least-favourable values in that mode)."""
    sc = frame.scores[method]
    return calls_from_arrays(sc["P_S"], sc["P_D"], sc["E1"], sc["E2"], t_S, t_D, e_gate)


def matched_w_thresholds(frame: EvalFrame, method: str, setting: str = "realistic",
                         cfg: MetricConfig = MetricConfig()) -> dict:
    """C3 thresholds (DESIGN v4 §0, §8.2), set on a validation run:
      t_S = the smallest threshold with confident-S rate on W <= w_limit (2.5%);
      then, with t_S fixed and S precedence (a locus with P_S >= t_S is called S, never D), t_D = the smallest
      threshold with confident-D rate on W <= w_limit, where a W locus is called D iff P_S < t_S, P_D >= t_D and
      the E gate passes (E1, E2 >= e_gate where reported).
    W rates are weighted with the D-vs-W cell pooling weights restricted to W. A method without P_S gets only
    t_D (no S calls); without P_D only t_S."""
    w = dw_weights(frame, setting, cfg.pairing_weights)
    isW = (frame.cls == "W") & (w > 0)
    sc = frame.scores[method]
    out = {}
    callS = np.zeros(len(frame), bool)
    if not np.isnan(sc["P_S"]).all():
        r = matched_threshold(sc["P_S"][isW], w[isW], cfg.w_limit)
        out.update(t_S=r["t"], rate_S_on_W=r["rate"])
        callS = np.nan_to_num(sc["P_S"], nan=-np.inf) >= r["t"]
    if not np.isnan(sc["P_D"]).all():
        eff = np.where(_gate(frame, method, cfg.e_gate) & ~callS, sc["P_D"], -np.inf)
        eff = np.where(np.isnan(eff), -np.inf, eff)
        r = matched_threshold(eff[isW], w[isW], cfg.w_limit)
        out.update(t_D=r["t"], rate_D_on_W=r["rate"])
    return out


def recall_at(frame: EvalFrame, method: str, t_S: float, t_D: float, setting: str = "realistic",
              cfg: MetricConfig = MetricConfig()) -> dict:
    """Recall of S (call S | S) and of D (call D | D), and confident-call rates on W, cell-pooled weights."""
    calls = three_way_calls(frame, method, t_S, t_D, cfg.e_gate)
    w = dw_weights(frame, setting, cfg.pairing_weights)
    cwmap = cell_pooling_weights(frame, setting, cfg.pairing_weights)
    out = {}
    # S recall uses cell pooling with S uniform within cell
    wS = np.zeros(len(frame))
    for c, wc in cwmap.items():
        m = (frame.cell == c) & (frame.cls == "S")
        if m.any():
            wS[m] = wc / m.sum()
    iS = wS > 0
    iD = (frame.cls == "D") & (w > 0)
    iW = (frame.cls == "W") & (w > 0)
    out["recall_S"] = weighted_rate(calls[iS] == "S", wS[iS]) if iS.any() else float("nan")
    out["recall_D"] = weighted_rate(calls[iD] == "D", w[iD]) if iD.any() else float("nan")
    out["W_called_S"] = weighted_rate(calls[iW] == "S", w[iW]) if iW.any() else float("nan")
    out["W_called_D"] = weighted_rate(calls[iW] == "D", w[iW]) if iW.any() else float("nan")
    return out


def confusion(frame: EvalFrame, method: str, t_S: float, t_D: float, e_gate: float = E_GATE) -> dict:
    """Unweighted three-way confusion counts: {true class: {call: n}}."""
    calls = three_way_calls(frame, method, t_S, t_D, e_gate)
    out = {}
    for k in ("S", "D", "W", "G", "U"):
        m = frame.cls == k
        if m.any():
            vals, cnt = np.unique(calls[m].astype(str), return_counts=True)
            out[k] = {str(v): int(c) for v, c in zip(vals, cnt)}
    return out


def calibration(frame: EvalFrame, method: str, cfg: MetricConfig = MetricConfig(), n_bins: int = 10) -> dict:
    """ECE/Brier of P_S for event {class S} and of P_D for event {class D}, on the UNWEIGHTED distribution of
    all loci in the frame (DESIGN §8.2); raw and, if cfg carries validation-fitted intercepts, recalibrated.
    Also the intercept fitted on THIS frame (use it from a validation run only)."""
    out = {}
    for ev, col, a in (("S", "P_S", cfg.recal_a_S), ("D", "P_D", cfg.recal_a_D)):
        p = frame.scores[method][col]
        if np.isnan(p).all():
            continue
        ok = ~np.isnan(p)
        y = (frame.cls[ok] == ev).astype(float)
        pp = p[ok]
        out[f"ece_{ev}"] = ece(pp, y, n_bins)
        out[f"brier_{ev}"] = brier(pp, y)
        out[f"recal_a_{ev}_fitted_here"] = fit_logit_intercept(pp, y)
        if a is not None:
            q = apply_logit_intercept(pp, a)
            out[f"ece_{ev}_recal"] = ece(q, y, n_bins)
            out[f"brier_{ev}_recal"] = brier(q, y)
    return out


def realized_fdr(frame: EvalFrame, method: str, t_S: float, t_D: float, e_gate: float = E_GATE) -> dict:
    """Among calls S (D): realized FDR = fraction not of class S (D), unweighted; nominal = mean(1 - P) over calls."""
    calls = three_way_calls(frame, method, t_S, t_D, e_gate)
    out = {}
    for k, col in (("S", "P_S"), ("D", "P_D")):
        m = calls == k
        out[f"n_called_{k}"] = int(m.sum())
        if m.any():
            out[f"fdr_{k}_realized"] = float(np.mean(frame.cls[m] != k))
            out[f"fdr_{k}_nominal"] = float(np.mean(1.0 - frame.scores[method][col][m]))
        else:
            out[f"fdr_{k}_realized"] = float("nan")
            out[f"fdr_{k}_nominal"] = float("nan")
    return out


def false_s_slope(frame: EvalFrame, method: str, true_class: str, t_S: float, use_abs: bool = False) -> float:
    """OLS slope of the false-S indicator (P_S >= t_S) on rho among loci of `true_class` (D or W),
    expressed per 0.1 rho (C4). NaN if rho has no spread."""
    m = frame.cls == true_class
    if m.sum() < 3:
        return float("nan")
    x = frame.rho[m]
    x = np.abs(x) if use_abs else x
    y = (np.nan_to_num(frame.scores[method]["P_S"][m], nan=-np.inf) >= t_S).astype(float)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    vx = np.var(x)
    if not vx > 1e-12:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / vx * 0.1)


def false_s_by_bin(frame: EvalFrame, method: str, t_S: float, by: str = "r2", edges=DEFAULT_R2_EDGES) -> dict:
    """Unweighted false-S rate (P_S >= t_S) on D loci by half-open bin of r² or d; D loci outside the edges are
    reported under "out"."""
    m = frame.cls == "D"
    x = frame.r2 if by == "r2" else frame.d
    PS = np.nan_to_num(frame.scores[method]["P_S"], nan=-np.inf)
    k = bin_index(x, edges)
    out = {}
    for i, lab in enumerate(bin_labels(edges) + ["out"]):
        b = m & (k == (i if lab != "out" else -1))
        out[lab] = {"n": int(b.sum()), "false_S": float(np.mean(PS[b] >= t_S)) if b.any() else float("nan")}
    return out


def stratified_report(frame: EvalFrame, method: str, by: str, edges, t_S: float, t_D: float,
                      setting: str = "realistic", cfg: MetricConfig = MetricConfig()) -> dict:
    """Stratified reporting over D strata (DESIGN v4 §8.6: r² edges [0,.1,.3,.5,.8,.95), d edges [2,4,8,16,inf)).

    For each stratum b: the frame is restricted to all non-D loci plus the D loci in b, and then
      auprc_sd / auroc_sd   S vs D_b, power matching recomputed within cell on S ∪ D_b (§8.6 rule unchanged);
      auroc_dw              D_b vs W;
      false_S, recall_D     unweighted rates on D_b at (t_S, t_D) with S precedence and the E gate;
      n                     number of D loci in b.
    D loci outside the edges (or with NaN) form the stratum "out" (rates only)."""
    x = frame.r2 if by == "r2" else frame.d
    k = bin_index(x, edges)
    isD = frame.cls == "D"
    calls = three_way_calls(frame, method, t_S, t_D, cfg.e_gate)
    out = {}
    for i, lab in enumerate(bin_labels(edges) + ["out"]):
        inb = isD & (k == (i if lab != "out" else -1))
        r = {"n": int(inb.sum()),
             "false_S": float(np.mean(calls[inb] == "S")) if inb.any() else float("nan"),
             "recall_D": float(np.mean(calls[inb] == "D")) if inb.any() else float("nan")}
        if lab != "out":
            sub = frame.mask(~isD | inb)
            if inb.any():
                r.update(auprc_sd=auprc_sd(sub, method, setting, cfg), auroc_sd=auroc_sd(sub, method, setting, cfg),
                         auroc_dw=auroc_dw(sub, method, setting, cfg))
            else:
                r.update(auprc_sd=float("nan"), auroc_sd=float("nan"), auroc_dw=float("nan"))
        out[lab] = r
    return out


def tie_shares(frame: EvalFrame, method: str) -> dict:
    s = frame.scores[method]["s"]
    m = np.isin(frame.cls, ("S", "D"))
    s = s[m]
    if len(s) == 0:
        return {"share_s_eq_0.5": float("nan"), "share_in_tie_groups": float("nan")}
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return {"share_s_eq_0.5": float(np.mean(s == 0.5)), "share_in_tie_groups": float(np.mean(cnt[inv] > 1))}


# ==================================================================================================
# Bootstrap
# ==================================================================================================
def _cluster_index(clusters):
    codes, uniq = pd.factorize(np.asarray(clusters), sort=True)
    order = np.argsort(codes, kind="mergesort")
    bounds = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(uniq)))]
    groups = [order[bounds[k]:bounds[k + 1]] for k in range(len(uniq))]
    return groups


_BOOT: dict = {}  # fork-shared state for parallel bootstrap workers


def _boot_eval(job):
    kind, payload = job
    stat, groups = _BOOT["stat"], _BOOT["groups"]
    if kind == "boot":
        return [np.atleast_1d(np.asarray(stat(np.concatenate([groups[k] for k in pick])), float)) for pick in payload]
    K = len(groups)
    return [np.atleast_1d(np.asarray(stat(np.concatenate([groups[j] for j in range(K) if j != k])), float))
            for k in payload]


def cluster_bootstrap(stat: Callable[[np.ndarray], float | np.ndarray], clusters, n_boot: int = 2000,
                      rng: np.random.Generator | None = None, level: float = 0.95, bca: bool = True,
                      n_jobs: int = 1, return_replicates: bool = False) -> dict:
    """Cluster (LD-block) bootstrap of stat(row_index_array). Clusters are resampled with replacement; the
    replicate passes the concatenated row indices (duplicates allowed) to `stat`, which must recompute any
    weights from those rows. BCa interval (acceleration from the leave-one-cluster-out jackknife) or
    percentile. Replicates returning NaN are dropped and counted. `stat` may return a vector.
    All resampling draws are made up front from `rng`, so results do not depend on n_jobs (fork workers).
    return_replicates=True adds "replicates" (n_boot x k) and "jackknife" (K x k), needed by
    `bootstrap_p_value`."""
    if rng is None:
        raise ValueError("pass an explicit numpy Generator (seeded from src/common/seeds.py)")
    groups = _cluster_index(clusters)
    K = len(groups)
    all_idx = np.concatenate(groups)
    theta = np.atleast_1d(np.asarray(stat(all_idx), float))
    picks = rng.integers(0, K, size=(n_boot, K))
    jobs_b = [("boot", picks[i::max(n_jobs, 1) * 4]) for i in range(max(n_jobs, 1) * 4)]
    jobs_j = [("jack", list(range(K))[i::max(n_jobs, 1) * 4]) for i in range(max(n_jobs, 1) * 4)] if bca else []
    _BOOT.update(stat=stat, groups=groups)
    try:
        if n_jobs > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(n_jobs) as pool:
                rb = pool.map(_boot_eval, jobs_b)
                rj = pool.map(_boot_eval, jobs_j) if bca else []
        else:
            rb = [_boot_eval(j) for j in jobs_b]
            rj = [_boot_eval(j) for j in jobs_j]
    finally:
        _BOOT.clear()
    # undo the strided split so replicate b corresponds to picks[b]
    boots = np.empty((n_boot, len(theta)))
    n_chunks = len(jobs_b)
    for i, chunk in enumerate(rb):
        boots[i::n_chunks] = np.array(chunk).reshape(-1, len(theta))
    if bca:
        jack = np.empty((K, len(theta)))
        for i, chunk in enumerate(rj):
            if len(chunk):
                jack[i::len(jobs_j)] = np.array(chunk).reshape(-1, len(theta))
    res = interval_from_replicates(theta, boots, jack if bca else None, level)
    res.update(n_boot=n_boot, n_clusters=K)
    if return_replicates:
        res["replicates"] = boots
        res["jackknife"] = jack if bca else None
    if len(theta) == 1:
        for k in ("estimate", "ci_low", "ci_high", "se", "z0", "acceleration"):
            res[k] = float(res[k][0])
    return res


def interval_from_replicates(theta, boots, jack=None, level: float = 0.95) -> dict:
    """BCa interval (jack given: acceleration from the leave-one-cluster-out jackknife) or percentile interval
    (jack None) per component of theta, from bootstrap replicates (n_boot x k). NaN replicates are dropped per
    component. Shared by cluster_bootstrap and `contrast`."""
    theta = np.atleast_1d(np.asarray(theta, float))
    boots = np.asarray(boots, float).reshape(-1, len(theta))
    bca = jack is not None
    if bca:
        jack = np.asarray(jack, float).reshape(-1, len(theta))
    alpha = (1 - level) / 2
    lo, hi = np.full(len(theta), np.nan), np.full(len(theta), np.nan)
    z0s, accs = np.full(len(theta), np.nan), np.full(len(theta), np.nan)
    for j in range(len(theta)):
        bj = boots[:, j]
        bj = bj[~np.isnan(bj)]
        if len(bj) < 10 or np.isnan(theta[j]):
            continue
        if not bca:
            lo[j], hi[j] = np.quantile(bj, [alpha, 1 - alpha])
            continue
        prop = (np.sum(bj < theta[j]) + 0.5 * np.sum(bj == theta[j])) / len(bj)
        prop = min(max(prop, 1.0 / (2 * len(bj))), 1 - 1.0 / (2 * len(bj)))
        z0 = stats.norm.ppf(prop)
        jj = jack[:, j]
        jj = jj[~np.isnan(jj)]
        dlt = jj.mean() - jj
        den = 6.0 * (np.sum(dlt ** 2)) ** 1.5
        a = float(np.sum(dlt ** 3) / den) if den > 0 else 0.0
        qs = []
        for zq in (stats.norm.ppf(alpha), stats.norm.ppf(1 - alpha)):
            qs.append(stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))
        lo[j], hi[j] = np.quantile(bj, qs)
        z0s[j], accs[j] = z0, a
    ok = (~np.isnan(boots)).sum(axis=0) >= 2
    se = np.full(len(theta), np.nan)
    if ok.any():
        se[ok] = np.nanstd(boots[:, ok], axis=0, ddof=1)
    return {"estimate": theta, "ci_low": lo, "ci_high": hi, "se": se,
            "n_boot_nan": int(np.isnan(boots).any(axis=1).sum()), "z0": z0s, "acceleration": accs,
            "method": "bca" if bca else "percentile", "level": level}


def contrast(res: dict, ia: int, ib: int) -> dict:
    """Paired difference component ia - ib of a vector bootstrap (cluster_bootstrap(..., return_replicates=True)):
    estimate, BCa (or percentile) CI from the SAME replicates and jackknife, and two-sided-usable replicates for
    `bootstrap_p_value`. Pairing holds because every component was computed on the same resampled clusters."""
    est = np.atleast_1d(np.asarray(res["estimate"], float))
    b = np.asarray(res["replicates"], float)
    jk = res.get("jackknife")
    d_b = b[:, ia] - b[:, ib]
    d_j = None if jk is None else np.asarray(jk, float)[:, ia] - np.asarray(jk, float)[:, ib]
    out = interval_from_replicates(est[ia] - est[ib], d_b[:, None], None if d_j is None else d_j[:, None],
                                   res.get("level", 0.95))
    for k in ("estimate", "ci_low", "ci_high", "se", "z0", "acceleration"):
        out[k] = float(np.atleast_1d(out[k])[0])
    out["replicates"], out["jackknife"] = d_b, d_j
    out["n_boot"], out["n_clusters"] = res.get("n_boot"), res.get("n_clusters")
    return out


def endpoint_vector(frame: EvalFrame, methods: list[str], settings: list[str],
                    cfg: MetricConfig = MetricConfig()) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    """Co-primary endpoints of several methods at once (weights are computed once per setting and shared, which
    is exactly what auprc_sd / auroc_sd / auroc_dw compute per method): returns the vector and its labels
    (method, setting, endpoint) with endpoint in {auprc_sd, auroc_sd, auroc_dw}. For joint (paired) bootstraps."""
    vals, labels = [], []
    for st in settings:
        w = sd_weights(frame, st, cfg.pairing_weights)
        m = w > 0
        y = frame.cls[m] == "S"
        wd = dw_weights(frame, st, cfg.pairing_weights)
        md = wd > 0
        yd = frame.cls[md] == "D"
        for meth in methods:
            sc = frame.scores[meth]
            vals.append(average_precision(y, sc["s"][m], w[m]))
            labels.append((meth, st, "auprc_sd"))
            vals.append(roc_auc(y, sc["s"][m], w[m]))
            labels.append((meth, st, "auroc_sd"))
            vals.append(float("nan") if np.isnan(sc["P_D"]).all() else roc_auc(yd, sc["P_D"][md], wd[md]))
            labels.append((meth, st, "auroc_dw"))
    return np.asarray(vals, float), labels


def bootstrap_p_value(res: dict, null: float, j: int = 0, alternative: str = "greater") -> float:
    """One-sided bootstrap p-value of H0: theta <= null vs H1: theta > null ("greater"; "less" mirrors it),
    obtained by inverting the interval of `cluster_bootstrap(..., return_replicates=True)`: the p-value is the
    one-sided level alpha at which the lower BCa (or percentile) bound equals `null`, so "p < alpha" is
    equivalent to "the one-sided (1 - alpha) lower bound exceeds null".
    BCa: with F = share of replicates <= null (clipped to [1/(2B), 1 - 1/(2B)]), u = Phi^-1(F) - z0,
    p = Phi(u / (1 + a u) - z0); with a = z0 = 0 this is the percentile p-value F."""
    if alternative not in ("greater", "less"):
        raise ValueError(alternative)
    b = np.asarray(res["replicates"], float)
    b = b[:, j] if b.ndim == 2 else b
    b = b[~np.isnan(b)]
    if len(b) < 10:
        return float("nan")
    sgn = 1.0 if alternative == "greater" else -1.0
    bb, nul = sgn * b, sgn * null
    est = np.atleast_1d(np.asarray(res["estimate"], float))[j] * sgn
    if np.isnan(est):
        return float("nan")
    B = len(bb)
    F = (np.sum(bb < nul) + 0.5 * np.sum(bb == nul)) / B
    F = min(max(F, 1.0 / (2 * B)), 1 - 1.0 / (2 * B))
    if res.get("method") != "bca" or res.get("jackknife") is None:
        return float(F)
    prop = (np.sum(bb < est) + 0.5 * np.sum(bb == est)) / B
    prop = min(max(prop, 1.0 / (2 * B)), 1 - 1.0 / (2 * B))
    z0 = stats.norm.ppf(prop)
    jk = np.asarray(res["jackknife"], float)
    jk = sgn * (jk[:, j] if jk.ndim == 2 else jk)
    jk = jk[~np.isnan(jk)]
    dlt = jk.mean() - jk
    den = 6.0 * (np.sum(dlt ** 2)) ** 1.5
    a = float(np.sum(dlt ** 3) / den) if den > 0 else 0.0
    u = stats.norm.ppf(F) - z0
    if 1 + a * u <= 0:  # outside the BCa transformation's range: fall back to the percentile p-value
        return float(F)
    return float(stats.norm.cdf(u / (1 + a * u) - z0))


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="mergesort")
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def paired_bootstrap(frame: EvalFrame, metric: Callable[[EvalFrame, str], float], method_a: str, method_b: str,
                     n_boot: int = 2000, rng: np.random.Generator | None = None, level: float = 0.95,
                     n_jobs: int = 1, unit: str = "block", return_replicates: bool = False) -> dict:
    """Paired cluster bootstrap (BCa) over LD blocks of metric(A) - metric(B); weights are recomputed per
    replicate because `metric` recomputes them from the resampled frame. Also returns the block sign test on
    the AUROC-additive per-block contributions when `metric` is auroc-like (see block_sign_test)."""
    def stat(idx):
        f = frame.take(idx)
        return metric(f, method_a) - metric(f, method_b)
    return cluster_bootstrap(stat, resampling_units(frame, unit), n_boot=n_boot, rng=rng, level=level, bca=True,
                             n_jobs=n_jobs, return_replicates=return_replicates)


def resampling_units(frame: EvalFrame, unit: str = "block") -> np.ndarray:
    """Bootstrap unit: "block" (the LD block, DESIGN §3/§8.3: claims) or "locus" (screening kill rules, §3)."""
    if unit == "block":
        return frame.block
    if unit == "locus":
        return frame.locus_id
    raise ValueError(f"unknown bootstrap unit {unit!r}")


def block_sign_test(frame: EvalFrame, method_a: str, method_b: str, endpoint: str = "sd",
                    setting: str = "realistic", cfg: MetricConfig = MetricConfig()) -> dict:
    """Sign test over LD blocks (DESIGN §3, §8.3). Per block: difference between A and B of the block's share of
    the weighted AUROC (S vs D on s, or D vs W on P_D); shares sum to the AUROC exactly (auc_contributions)."""
    if endpoint == "sd":
        w = sd_weights(frame, setting, cfg.pairing_weights)
        y, col = frame.cls == "S", "s"
    else:
        w = dw_weights(frame, setting, cfg.pairing_weights)
        y, col = frame.cls == "D", "P_D"
    m = w > 0
    ca = auc_contributions(y[m], frame.scores[method_a][col][m], w[m])
    cb = auc_contributions(y[m], frame.scores[method_b][col][m], w[m])
    blk = frame.block[m]
    df = pd.DataFrame({"b": blk, "d": ca - cb}).groupby("b")["d"].sum()
    res = sign_test(df.to_numpy())
    res["n_blocks"] = int(len(df))
    return res


def _recall_weights(frame: EvalFrame, setting: str, cfg: MetricConfig) -> tuple[np.ndarray, np.ndarray]:
    """(wS, wDW): S-recall weights (uniform within cell x pooling weight) and the D-vs-W weights (as recall_at)."""
    w = dw_weights(frame, setting, cfg.pairing_weights)
    cwa = _map_cells(frame.cell, cell_pooling_weights(frame, setting, cfg.pairing_weights))
    wS = np.zeros(len(frame))
    isS = (frame.cls == "S") & (cwa > 0)
    if isS.any():
        cnt = pd.Series(frame.cell[isS]).value_counts()
        wS[isS] = cwa[isS] / pd.Series(frame.cell[isS]).map(cnt).to_numpy(dtype=float)
    return wS, w


def c3_test(frame: EvalFrame, candidate: str, baselines: list[str], thresholds: dict, setting: str = "realistic",
            cfg: MetricConfig = MetricConfig(), margin: float = NI_MARGIN, alpha: float = 0.05, n_boot: int = 2000,
            rng: np.random.Generator | None = None, n_jobs: int = 1, unit: str = "block") -> dict:
    """C3 test (DESIGN v4 §0). `thresholds` = {method: {"t_S": .., "t_D": ..}} fixed on VALIDATION by
    `matched_w_thresholds` (a missing t_S / t_D means the method makes no such call).

    At these thresholds, recall of S and of D (recall_at weights; S precedence; E gate) of the candidate vs:
      * non-inferiority to EVERY eligible baseline: one-sided paired cluster bootstrap, H0: diff <= -margin
        (margin 0.01), p < alpha for both recalls and every baseline (intersection-union, no adjustment);
      * superiority to the BEST eligible baseline (highest point recall on this frame, per endpoint) on at least
        one of the two recalls: one-sided H0: diff <= 0, Holm-adjusted over the two, min adjusted p < alpha.
    Eligible = baselines with a defined P_D in this frame (the proportional test and the lead-SNP rule have none;
    callers should not list them). All differences come from ONE paired bootstrap (same resamples)."""
    sc_ok = [b for b in baselines if b != candidate and not np.isnan(frame.scores[b]["P_D"]).all()]
    inel = [b for b in baselines if b != candidate and b not in sc_ok]
    if not sc_ok:
        raise ValueError("no eligible baseline (needs a defined P_D)")
    meths = [candidate] + sc_ok

    def thr(m):
        t = thresholds.get(m, {})
        return float(t.get("t_S", np.inf)), float(t.get("t_D", np.inf))

    def recalls(f: EvalFrame) -> dict:
        wS, w = _recall_weights(f, setting, cfg)
        iS, iD = wS > 0, (f.cls == "D") & (w > 0)
        out = {}
        for m in meths:
            calls = three_way_calls(f, m, *thr(m), cfg.e_gate)
            out[m] = (weighted_rate(calls[iS] == "S", wS[iS]) if iS.any() else np.nan,
                      weighted_rate(calls[iD] == "D", w[iD]) if iD.any() else np.nan)
        return out

    def stat(idx):
        r = recalls(frame.take(idx))
        return np.array([r[candidate][e] - r[b][e] for b in sc_ok for e in (0, 1)])

    point = recalls(frame)
    res = cluster_bootstrap(stat, resampling_units(frame, unit), n_boot=n_boot, rng=rng, bca=True, n_jobs=n_jobs,
                            return_replicates=True)
    est = np.atleast_1d(res["estimate"])
    lo, hi = np.atleast_1d(res["ci_low"]), np.atleast_1d(res["ci_high"])
    names = ("recall_S", "recall_D")
    out = {"candidate": candidate, "eligible": sc_ok, "ineligible": inel, "setting": setting, "unit": unit,
           "margin": margin, "alpha": alpha, "n_boot": n_boot, "n_clusters": res["n_clusters"],
           "thresholds": {m: dict(zip(("t_S", "t_D"), thr(m))) for m in meths},
           "recall": {m: dict(zip(names, point[m])) for m in meths}, "vs": {}}
    ni_all = True
    for i, b in enumerate(sc_ok):
        d = {}
        for e, nm in enumerate(names):
            j = 2 * i + e
            p_ni = bootstrap_p_value(res, -margin, j, "greater")
            p_sup = bootstrap_p_value(res, 0.0, j, "greater")
            d[nm] = {"diff": float(est[j]), "ci95_low": float(lo[j]), "ci95_high": float(hi[j]),
                     "p_noninferiority": p_ni, "p_superiority_unadjusted": p_sup}
            ni_all &= bool(p_ni < alpha)
        out["vs"][b] = d
    best = {nm: max(sc_ok, key=lambda b, e=e: -np.inf if np.isnan(point[b][e]) else point[b][e])
            for e, nm in enumerate(names)}
    p_sup = [out["vs"][best[nm]][nm]["p_superiority_unadjusted"] for nm in names]
    p_adj = holm(p_sup)
    out["superiority"] = {nm: {"best_baseline": best[nm], "p": p_sup[e], "p_holm": p_adj[e]}
                          for e, nm in enumerate(names)}
    out["noninferior_all"] = ni_all
    out["superior_one"] = bool(min(p_adj) < alpha)
    out["pass"] = bool(ni_all and out["superior_one"])
    return out


# ==================================================================================================
# Run-level metrics (metrics.json)
# ==================================================================================================
def _clean(x):
    """JSON-safe: NaN -> None, +-inf -> "inf"/"-inf", numpy scalars/arrays -> python."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return x


def _settings_present(frame: EvalFrame, cfg: MetricConfig) -> list[str]:
    out = []
    for s in ("realistic", "ideal", "shifted", "all"):
        if cell_pooling_weights(frame, s, cfg.pairing_weights):
            out.append(s)
    return out


def _shuffled_auprc(frame: EvalFrame, method: str, setting: str, cfg: MetricConfig) -> dict:
    """Negative control (DESIGN §7.1-9): AUPRC after permuting the scores within each cell (labels and weights
    fixed) equals the weighted prevalence (0.5) up to sampling noise."""
    rng = np.random.default_rng(cfg.shuffle_seed)
    w = sd_weights(frame, setting, cfg.pairing_weights)
    m = w > 0
    if not m.any():
        return {}
    y = frame.cls[m] == "S"
    s = frame.scores[method]["s"][m].copy()
    cells = frame.cell[m]
    for c in np.unique(cells):
        ic = np.nonzero(cells == c)[0]
        s[ic] = s[rng.permutation(ic)]
    return {"auprc_shuffled": average_precision(y, s, w[m]), "weighted_prevalence": weighted_rate(y, w[m])}


def _controls(frame: EvalFrame, method: str) -> dict:
    out = {}
    for ctl, col, thr in (("pos_shared", "s", 0.9), ("null", "P_W", None)):
        m = frame.control == ctl
        if not m.any():
            continue
        v = frame.scores[method][col][m]
        if np.isnan(v).all():
            out[ctl] = {"n": int(m.sum()), "note": f"{col} not available"}
            continue
        d = {"n": int(m.sum()), f"min_{col}": float(np.nanmin(v)), f"mean_{col}": float(np.nanmean(v))}
        if thr is not None:
            d[f"share_{col}_gt_{thr}"] = float(np.mean(v > thr))
        out[ctl] = d
    return out


def method_metrics(frame: EvalFrame, method: str, cfg: MetricConfig) -> dict:
    res = {}
    settings = _settings_present(frame, cfg)
    for st in settings:
        res[st] = {"auprc_sd": auprc_sd(frame, method, st, cfg), "auroc_sd": auroc_sd(frame, method, st, cfg),
                   "auroc_dw": auroc_dw(frame, method, st, cfg)}
    per_cell = {}
    for c in sorted(np.unique(frame.cell)):
        st = f"cell:{c}"
        ic = frame.cell == c
        per_cell[c] = {"auprc_sd": auprc_sd(frame, method, st, cfg), "auroc_sd": auroc_sd(frame, method, st, cfg),
                       "auroc_dw": auroc_dw(frame, method, st, cfg),
                       "n": {k: int((ic & (frame.cls == k)).sum()) for k in ("S", "D", "W", "G", "U")}}
        # calibration per cell (= mismatch level x pairing x overlap; DESIGN §8.2 "per mismatch level x rho")
        cal = calibration(frame.mask(ic), method, cfg)
        per_cell[c]["calibration"] = {k: v for k, v in cal.items() if not k.startswith("recal_a_")}
    res["per_cell"] = per_cell
    for key in ("auprc_sd", "auroc_sd", "auroc_dw"):
        vals = [v[key] for v in per_cell.values() if v[key] is not None and not np.isnan(v[key])]
        res.setdefault("macro", {})[key] = float(np.mean(vals)) if vals else float("nan")
    levels = sorted(set(frame.level) - {""})
    res["per_level"] = {L: {k: f(frame, method, f"level:{L}", cfg)
                            for k, f in (("auprc_sd", auprc_sd), ("auroc_sd", auroc_sd), ("auroc_dw", auroc_dw))}
                        for L in levels}
    res["ties"] = tie_shares(frame, method)
    main = "realistic" if "realistic" in settings else (settings[0] if settings else None)
    if main is not None:
        res["matched_w"] = mw = matched_w_thresholds(frame, method, main, cfg)
        res["matched_w"]["setting"] = main
        res["at_fixed_0.8"] = recall_at(frame, method, FIXED_THRESHOLD, FIXED_THRESHOLD, main, cfg)
        res["at_config_thresholds"] = recall_at(frame, method, cfg.t_S, cfg.t_D, main, cfg)
        if "t_S" in mw or "t_D" in mw:
            res["at_matched_w_here"] = recall_at(frame, method, mw.get("t_S", np.inf), mw.get("t_D", np.inf), main, cfg)
        res["shuffled_label_control"] = _shuffled_auprc(frame, method, main, cfg)
        res["by_r2_stratum_config_thresholds"] = stratified_report(frame, method, "r2", cfg.r2_edges, cfg.t_S,
                                                                   cfg.t_D, main, cfg)
        res["by_d_stratum_config_thresholds"] = stratified_report(frame, method, "d", cfg.d_edges, cfg.t_S,
                                                                  cfg.t_D, main, cfg)
    res["confusion_config_thresholds"] = confusion(frame, method, cfg.t_S, cfg.t_D, cfg.e_gate)
    res["calibration"] = calibration(frame, method, cfg)
    res["fdr_config_thresholds"] = realized_fdr(frame, method, cfg.t_S, cfg.t_D, cfg.e_gate)
    res["false_S_slope_per_0.1rho"] = {k: false_s_slope(frame, method, k, cfg.t_S) for k in ("D", "W")}
    res["controls"] = _controls(frame, method)
    return res


def failure_summary(frame: EvalFrame, method: str) -> dict:
    st = frame.status[method].astype(str)
    out = {"status_counts": {k: int(v) for k, v in zip(*np.unique(st, return_counts=True))},
           "n_failed": int(frame.failed[method].sum()), "n_expected": int(len(frame))}
    per_cell = {}
    for c in sorted(np.unique(frame.cell)):
        ic = frame.cell == c
        nf = int(frame.failed[method][ic].sum())
        per_cell[c] = {"n": int(ic.sum()), "n_failed": nf, "share": nf / max(int(ic.sum()), 1),
                       "n_nonconverged": int((st[ic] == "nonconverged").sum())}
    out["per_cell"] = per_cell
    out["max_cell_failure_share"] = max((v["share"] for v in per_cell.values()), default=0.0)
    out["blocks_claims_gt_1pct"] = bool(out["max_cell_failure_share"] > 0.01)
    return out


def compute_run_metrics(truth: pd.DataFrame, predictions: pd.DataFrame, cfg: MetricConfig | dict | None = None) -> dict:
    """metrics.json content for one run (one or more methods in `predictions`). Deterministic."""
    cfg = cfg if isinstance(cfg, MetricConfig) else MetricConfig.from_dict(cfg)
    out = {"metric_config": cfg.to_dict(), "methods": {}}
    for m, df in predictions.groupby("method", sort=True):
        lf = build_eval_frame(truth, {m: df}, "least_favourable")
        ex = build_eval_frame(truth, {m: df}, "excluded")
        out["methods"][m] = {"failures": failure_summary(lf, m),
                             "least_favourable": method_metrics(lf, m, cfg),
                             "excluded": method_metrics(ex, m, cfg) if len(ex) else {}}
    return _clean(out)


def dumps(metrics: dict) -> str:
    return json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False)
