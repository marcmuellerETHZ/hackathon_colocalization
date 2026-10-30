"""OP2 operating point (DESIGN_P2 §3, frozen 2026-10-28): a phase-2 amendment next to the matched-W C3 operating
point of eval.metrics (matched_w_thresholds), which is unchanged. Kept in its own module so that the hash-protected
src/eval/metrics.py (datasets/configs/FROZEN_SHA256) stays byte-identical; it uses only metrics' public building blocks.
"""
from __future__ import annotations

import numpy as np

from eval.metrics import (MetricConfig, EvalFrame, _gate, dw_weights, matched_threshold, sd_weights,
                          weighted_rate)

OP2_D_LIMIT = 0.10  # DESIGN_P2 §3 (frozen 2026-10-28): unweighted confident-S on power-matched D loci <= 10%


def _min_feasible(score, w, alpha) -> float:
    """matched_threshold's t, with 'no callable score' (all -inf / no loci) meaning 'no constraint' (-inf)."""
    s = np.asarray(score, float)
    if not np.isfinite(s).any():
        return -np.inf
    return float(matched_threshold(s, w, alpha)["t"])


def op2_thresholds(frame: EvalFrame, method: str, setting: str = "realistic",
                   cfg: MetricConfig = MetricConfig(), d_limit: float = OP2_D_LIMIT) -> dict:
    """OP2 (DESIGN_P2 §3, phase-2 amendment; eval.metrics.matched_w_thresholds (OP1) is unchanged):
      t_S = the smallest threshold with weighted confident-S on W <= w_limit (2.5%; the OP1 W weights) AND
            unweighted confident-S on D <= d_limit (10%), the D share taken over all power-matched D loci of the
            fit set (class D with S-vs-D weight > 0 in `setting`). Both rates fall monotonically in t, so the
            smallest feasible t is the max of the two one-sided matched thresholds. A threshold is a probability:
            if the smallest feasible t exceeds 1 (only "no S calls" satisfies both), t_S = 1 and op2_feasible =
            False; S calls are then NOT made (t_S_call = inf), and consumers must treat t_S = 1 as "no S calls".
      t_D = as OP1, after S precedence at the OP2 S rule: the smallest threshold with weighted confident-D on W
            <= w_limit, a W locus being called D iff not called S, P_D >= t_D and the E gate passes.
    NaN scores (excluded/raw frames) are never called, as in calls_from_arrays."""
    w = dw_weights(frame, setting, cfg.pairing_weights)
    isW = (frame.cls == "W") & (w > 0)
    isD = (frame.cls == "D") & (sd_weights(frame, setting, cfg.pairing_weights) > 0)
    sc = frame.scores[method]
    out = {"setting": setting, "w_limit": cfg.w_limit, "d_limit": d_limit, "n_W": int(isW.sum()),
           "n_D_matched": int(isD.sum())}
    callS = np.zeros(len(frame), bool)
    if not np.isnan(sc["P_S"]).all():
        ps = np.nan_to_num(sc["P_S"], nan=-np.inf)
        tW = _min_feasible(ps[isW], w[isW], cfg.w_limit)
        tD = _min_feasible(ps[isD], None, d_limit)
        t = max(tW, tD)
        feasible = bool(t <= 1.0)
        t_call = t if feasible else np.inf
        callS = ps >= t_call
        out.update(t_S=float(t) if feasible else 1.0, t_S_call=float(t_call), op2_feasible=feasible,
                   t_S_W_side=tW, t_S_D_side=tD,
                   rate_S_on_W=weighted_rate(callS[isW], w[isW]) if isW.any() else float("nan"),
                   rate_S_on_D=float(callS[isD].mean()) if isD.any() else float("nan"))
    if not np.isnan(sc["P_D"]).all():
        eff = np.where(_gate(frame, method, cfg.e_gate) & ~callS, sc["P_D"], -np.inf)
        eff = np.where(np.isnan(eff), -np.inf, eff)
        r = matched_threshold(eff[isW], w[isW], cfg.w_limit)
        out.update(t_D=r["t"], rate_D_on_W=r["rate"])
    return out
