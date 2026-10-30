"""The SINGLE implementation of the pre-registered score adapters (DESIGN v4 §4.3, §5; decisions D4, D6(f)).

Hash-protected via configs/FROZEN_SHA256. Every method wrapper (src/baselines/, src/fam*/) builds its output
scores with these helpers, so each §4.3 formula exists in exactly one place (D6(f)). All helpers return a dict
with the INTERFACES §3 score keys s, P_S, P_D, P_W, E1, E2 (NaN where undefined for the method); wrappers add
locus_id / method / status / extra themselves.

Ties and undefined values: s = 0.5 (an adapter never emits NaN s for a locus it ran on; NaN s is scored as a
failure by the metrics). A NaN P_S / P_D on a locus where the method normally reports it is also a failure.

Index of helpers (method -> helper):
  coloc.abf                        coloc_abf(pp)                      alias coloc_abf_scores(pp)
  coloc-SuSiE (0.14.2 and 0.16.x)  coloc_susie(pp_pairs, fallback_pp)  alias coloc_susie_scores(pp_rows)
  SharePro                         sharepro(shares, effect_prob1, effect_prob2)        [D4 + D6(f)]
  ColocBoost (both modes)          colocboost(cos_npc, signal1, signal2)                [D4, D6(f)]
  generic locus-probability row    locus_prob_with_signals(P_S, p_signal1, p_signal2 | pips1, pips2)
                                   alias locus_prob_scores(P_S, P_sig1, P_sig2)
  mvSuSiE-RSS                      mvsusie(single_effect_lfsr, cs_index, rule="soft")    [D4]
  activity-probability form        inclusion_exclusion_q(q1, q2)
  Family A (and anything with the  inclusion_exclusion(a, w, p1t, p2t, p12t)             [§5]
  full per-effect events)
  proportional test                proportional_test(p_prop)
  lead-SNP rule                    lead_snp(shared)
  P(trait has a signal)            signal_prob(probs) == one_minus_prod(probs) = 1 - prod(1 - x)
"""
from __future__ import annotations

import math

import numpy as np

NAN = float("nan")


def _f(x) -> float:
    return float(x) if x is not None else NAN


def ratio_or_half(num: float, other: float) -> float:
    """num / (num + other); 0.5 when undefined (both 0 or non-finite)."""
    num, other = _f(num), _f(other)
    den = num + other
    if not math.isfinite(den) or den <= 0 or not math.isfinite(num):
        return 0.5
    return num / den


def scores(s=NAN, P_S=NAN, P_D=NAN, P_W=NAN, E1=NAN, E2=NAN) -> dict:
    return {"s": _f(s), "P_S": _f(P_S), "P_D": _f(P_D), "P_W": _f(P_W), "E1": _f(E1), "E2": _f(E2)}


def _clip01(x: float) -> float:
    """Clip a finite probability to [0, 1]; NaN stays NaN (it is then scored as a failure)."""
    x = _f(x)
    return min(max(x, 0.0), 1.0) if math.isfinite(x) else x


# ---------------------------------------------------------------------------------------------- coloc
def coloc_abf(pp) -> dict:
    """coloc.abf posteriors pp = (PP0..PP4): s = PP4/(PP3+PP4), P_S = PP4, P_D = PP3 (ungated),
    P_W = PP0 + PP1 + PP2."""
    pp = np.asarray(pp, float)
    if pp.shape != (5,):
        raise ValueError("pp must be (PP0, PP1, PP2, PP3, PP4)")
    return scores(s=ratio_or_half(pp[4], pp[3]), P_S=pp[4], P_D=pp[3], P_W=pp[0] + pp[1] + pp[2])


coloc_abf_scores = coloc_abf


def coloc_susie(pp_pairs, fallback_pp=None) -> dict:
    """coloc-SuSiE (both susieR versions). pp_pairs: array (k, 5) of PP0..PP4 per credible-set pair.
    Rows with a non-finite PP (coloc.bf_bf trimming) are dropped. If no finite row remains (or pp_pairs is
    None/empty, i.e. a trait has no CS), the coloc.abf row on `fallback_pp` is returned; without fallback a
    ValueError is raised (the caller must fall back).
    s = max_k PP4/(PP3+PP4); P_S = max_k PP4; P_D = (1 - max_k PP4) * max_k PP3; P_W = 1 - P_S - P_D (>= 0)."""
    pp = None
    if pp_pairs is not None and np.asarray(pp_pairs).size > 0:
        pp = np.asarray(pp_pairs, float).reshape(-1, 5)
        pp = pp[np.all(np.isfinite(pp), axis=1)]
        if pp.shape[0] == 0:
            pp = None
    if pp is None:
        if fallback_pp is None:
            raise ValueError("no finite CS-pair rows and no fallback coloc.abf posteriors")
        return coloc_abf(fallback_pp)
    s = max(ratio_or_half(r[4], r[3]) for r in pp)
    PS = float(pp[:, 4].max())
    PD = (1.0 - PS) * float(pp[:, 3].max())
    return scores(s=s, P_S=PS, P_D=PD, P_W=max(0.0, 1.0 - PS - PD))


def coloc_susie_scores(pp_rows) -> dict:
    """Alias with the src/baselines signature: raises ValueError when no finite CS-pair row exists."""
    return coloc_susie(pp_rows, None)


# ---------------------------------------------------------------------------------------------- signal probabilities
def signal_prob(probs) -> float:
    """1 - prod_j (1 - x_j) with x clipped to [0, 1]; 0 for an empty list. Used for P(trait has a signal) from
    PIPs (DESIGN §4.3 fallback) or from per-effect activity probabilities (D6(f)), and for locus P_S from
    per-effect/per-CoS sharing probabilities (D4)."""
    x = np.clip(np.asarray(probs if probs is not None else [], float).ravel(), 0.0, 1.0)
    if len(x) == 0:
        return 0.0
    if np.isnan(x).any():
        return NAN
    if (x >= 1.0).any():
        return 1.0
    return float(-np.expm1(np.sum(np.log1p(-x))))


one_minus_prod = signal_prob


def locus_prob_with_signals(P_S: float, p_signal1: float | None = None, p_signal2: float | None = None,
                            pips1=None, pips2=None) -> dict:
    """§4.3 SharePro / ColocBoost row: s = P_S = the method's locus colocalisation probability;
    P_D = (1 - P_S) * P(trait-1 signal) * P(trait-2 signal), with P(trait-t signal) the method's per-trait signal
    probability if reported, else 1 - prod_j(1 - PIP_tj) over its per-trait PIPs; P_W = 1 - P_S - P_D (>= 0).
    P_S is clipped to [0, 1]; a NaN P_S gives NaN P_S/P_D (scored as a failure) and s = 0.5."""
    p1 = _f(p_signal1) if p_signal1 is not None else signal_prob(pips1)
    p2 = _f(p_signal2) if p_signal2 is not None else signal_prob(pips2)
    PS = _clip01(P_S)
    s = PS if math.isfinite(PS) else 0.5
    PD = (1.0 - PS) * p1 * p2
    PW = max(0.0, 1.0 - PS - PD) if math.isfinite(PD) else NAN
    return scores(s=s, P_S=PS, P_D=PD, P_W=PW)


def locus_prob_scores(P_S: float, P_sig1: float, P_sig2: float) -> dict:
    """Alias with the src/baselines signature (per-trait signal probabilities given)."""
    return locus_prob_with_signals(P_S, p_signal1=P_sig1, p_signal2=P_sig2)


def sharepro(shares, effect_prob1, effect_prob2) -> dict:
    """SharePro (D4, D6(f)). shares: per reported effect group k, its colocalisation probability share_k;
    P_S = 1 - prod_k (1 - share_k).  effect_prob_t: per effect k, SharePro's probability that effect k is active
    in trait t, sum_j gamma_jk delta_t,jk (its per-effect signal probability, the "if reported" clause of §4.3);
    P(trait-t signal) = 1 - prod_k (1 - effect_prob_t,k).  Then the §4.3 locus-probability row."""
    return locus_prob_with_signals(signal_prob(shares), p_signal1=signal_prob(effect_prob1),
                                   p_signal2=signal_prob(effect_prob2))


def colocboost(cos_npc, signal1: bool | float, signal2: bool | float) -> dict:
    """ColocBoost (D4, D6(f)). P_S = 1 - prod over colocalisation sets (1 - cos_npc). ColocBoost reports no
    per-trait probability, so P(trait-t signal) = 1 if trait t is in a CoS or a trait-specific uCoS else 0
    (binary; the resulting P_D ties are reported, not fixed: D6(f))."""
    return locus_prob_with_signals(signal_prob(cos_npc), p_signal1=float(signal1), p_signal2=float(signal2))


# ---------------------------------------------------------------------------------------------- inclusion-exclusion
def inclusion_exclusion(a, w, p1t, p2t, p12t) -> dict:
    """Family A locus outputs (DESIGN §5, mean-field independent effects l = 1..L):
    a_l   = P(l shared and powered in both)            -> P_S = 1 - prod(1 - a_l)
    w_l   = P(l shared with power >= 0.2 in both)       (blocks D)
    p1t_l = P(l powered in trait 1 and not blocking), p2t_l likewise, p12t_l = P(powered in 1 or 2, not blocking)
    P_D = prod(1-w) - prod(1-w-p1t) - prod(1-w-p2t) + prod(1-w-p12t);  P_W = 1 - P_S - P_D;  s = P_S/(P_S+P_D)."""
    a, w, p1t, p2t, p12t = (np.asarray(x, float) for x in (a, w, p1t, p2t, p12t))
    PS = 1.0 - np.prod(1.0 - a)
    PD = np.prod(1 - w) - np.prod(1 - w - p1t) - np.prod(1 - w - p2t) + np.prod(1 - w - p12t)
    PD = float(min(max(PD, 0.0), 1.0))
    return scores(s=ratio_or_half(PS, PD), P_S=PS, P_D=PD, P_W=max(0.0, 1.0 - PS - PD))


def inclusion_exclusion_q(q1, q2) -> dict:
    """Inclusion-exclusion from per-effect activity probabilities q_lt = P(effect l active in trait t), the two
    traits' activity within an effect treated as independent (D4, mvSuSiE):
      a_l = w_l = q_l1 q_l2;  p~1_l = q_l1 (1 - q_l2);  p~2_l = q_l2 (1 - q_l1);  p~12_l = q_l1 + q_l2 - 2 q_l1 q_l2
    and then `inclusion_exclusion`. No effects: s = 0.5, P_S = P_D = 0, P_W = 1."""
    q1 = np.clip(np.asarray(q1, float).ravel(), 0.0, 1.0)
    q2 = np.clip(np.asarray(q2, float).ravel(), 0.0, 1.0)
    if q1.shape != q2.shape:
        raise ValueError("q1 and q2 must have the same length (one entry per effect)")
    if q1.size == 0:
        return scores(s=0.5, P_S=0.0, P_D=0.0, P_W=1.0)
    w = q1 * q2
    return inclusion_exclusion(w, w, q1 * (1 - q2), q2 * (1 - q1), q1 + q2 - 2 * q1 * q2)


def mvsusie(single_effect_lfsr, cs_index, rule: str = "soft") -> dict:
    """mvSuSiE-RSS (D4). single_effect_lfsr: (L, 2) per-effect lfsr per trait; cs_index: 1-based indices of the
    effects that form a credible set (others get q = 0).
    rule="soft" (primary, D4): q_lt = 1 - lfsr_lt.   rule="hard" (DESIGN §4.3 wording "lfsr < 0.05 defines
    active"): q_lt = 1{lfsr_lt < 0.05}. Then `inclusion_exclusion_q`."""
    lf = np.asarray(single_effect_lfsr, float).reshape(-1, 2)
    q = np.zeros_like(lf)
    for l in cs_index if cs_index is not None else []:
        l = int(l)
        if not 1 <= l <= lf.shape[0]:
            raise ValueError(f"cs_index {l} outside 1..{lf.shape[0]}")
        if rule == "soft":
            q[l - 1] = 1.0 - lf[l - 1]
        elif rule == "hard":
            q[l - 1] = (lf[l - 1] < 0.05).astype(float)
        else:
            raise ValueError(rule)
    return inclusion_exclusion_q(q[:, 0], q[:, 1])


# ---------------------------------------------------------------------------------------------- S-vs-D-only rows
def proportional_test(p_prop: float) -> dict:
    """Proportional test (prop.coloc / colocPropTest): s = p_prop (high = consistent with sharing).
    P_S, P_D undefined (S-vs-D only)."""
    p = _f(p_prop)
    return scores(s=p if math.isfinite(p) else 0.5)


def lead_snp(shared: bool) -> dict:
    """Lead-SNP rule: binary s; no probabilities."""
    return scores(s=1.0 if shared else 0.0)
