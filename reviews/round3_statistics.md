# Review round 3 — statistics (final, light)

Reviewer: independent statistics agent, 2026-10-26. Scope: DESIGN.md v3 §1, §2.2, §2.4, §4.3, §5 A/B, C3, §8.6, checked against reviews/round2_statistics.md. Throwaway scripts: /tmp/claude-190744/r3stat/ (pd.py, bf.py, rho.py).

**Verdict: not yet ready to freeze. Six short must-fix items, all text-only.** The round-2 core fixes are correct: inclusion-exclusion P_D, BF form, p_prop direction, pairing-type grid and the nested-ρ mapping.

## Round-2 must-change items

| R2 item | v3 status |
|---|---|
| M1 infeasible ρ grid | **Mostly fixed.** The P1 grid and the nested mapping c = r_p·√(N_syn/N1) are correct (derived: Cov = c·√(N2/N_syn)·R, which equals r_p·√(N2/N1)·R; c ≤ 1 because N1 ≥ N_syn). The P2 grid is still partly infeasible (MF1). Overlap cells at L3 and L6 are incoherent (MF2). |
| M2 Family A P_D | **Fixed.** Monte Carlo (4e5 draws) matches the formula in every case to ≤ 1e-4: ambiguous single signal 0/0; L = 1 split 0/0; two distinct signals 0.9998/0.9998; mixed grey 0.0658/0.0659; shared weak-in-2 + distinct 0.7141/0.7141. The p̃^{1∨2} form equals R2's a/b/c form, because a non-blocking effect cannot be powered in both traits. Layer D text was not updated (MF4). |
| M3 proportional-test sign | **Fixed** (s = p_prop). The known-direction control I proposed in R2 is wrong for this method (MF3). |
| M4 C1 world | **Mostly fixed** (L0\*). Background, true ρ and the generating R are not pinned (MF5). |
| M5 C3 thresholds | **Mostly fixed** in §0. §8.2 still has the old rule, and the threshold choice is not pinned (MF6). |
| S1 BF formula | **Fixed.** Re-checked against the full-form Gaussian marginal (p = 25, ρ ∈ {0, 0.4, −0.7}; D-type, rank-1 shared, 2+2). Relative error ≤ 5e-14 throughout. |

## Must-fix

**MF1. The P2 ρ grid is still infeasible for some (N1, N2) pairs (§2.4, §8.6).**
- The bound is ρ ≤ r_p·√(N2/N1).
- At (N2 = 20k, N1 = 400k), ρ = 0.3 needs r_p = 1.34, which is impossible, and ρ = 0.2 needs r_p = 0.89.
- At (50k, 400k), ρ = 0.3 needs r_p = 0.85.

Fix:
- Parameterise P2 by r_p (nested, N_s = N2) and record the realised ρ; or
- pin the ρ = 0.3 cell to (N1 = 100k, N2 = 50k), where r_p = 0.42.

Either way, the generator must assert r_p = ρ·√(N1/N2) ≤ 0.9.

**MF2. Overlap cells at L3 and L6 are physically incoherent (§2.3 × §8.6).**
- At L6, trait 1 is drawn from mosaic-FIN and trait 2 from mosaic-NFE. Yet §8.6 crosses L6 with nested P1 (r_p = 0.6), P2 (ρ = 0.3) and P3 (same donors, ρ = 0.5 or 0.8). Shared donors cannot belong to two populations.
- L3 × P3 has the same problem: the trait-2 study is admixed while trait 1 is "NFE", but P3 means the same donors.

Fix: in overlapping cells, the shared rows come from one population.
- L6: either only P1 r_p = 0 and P2/P3 ρ = 0, or both traits FIN with an NFE reference.
- L3 × P3: both traits use the admixed study.

Write the rule in §2.3 and §8.6.

**MF3. The §7.1.9 control "λ = 30 shared at r² = 1 gives s > 0.9" fails for the proportional test by construction.**
- Under exact proportionality, p_prop ~ U(0,1), so s > 0.9 holds only about 10% of the time.
- Fix: exempt the proportional test from that control and use a method-specific pair of controls instead:
  - on S, p_prop ≥ 0.05 in about 95% of replicates (calibration);
  - on D at r² < 0.1 with λ = 30 in both traits, p_prop < 1e-3.

  This was my own R2 error.

**MF4. Layer D still defines E₂ with τ₂ (§5 "Cross-cutting layers").**
- §5 A says the exclusion probabilities use the 0.2-power thresholds.
- Fix: E₂ = P(|λ_{2,j}| < w₂ ∀ j ∈ CS₁), with w₁ = 4.61 and w₂ = 3.58 (and symmetrically for E₁).
- As written, a grey shared effect (e.g. λ₂ = 4.5) passes the exclusion check, and the D call disagrees with oracle D/G.

**MF5. Pin the L0\* world fully, or C1 can kill C spuriously (§5 B).** B is exact only if L0\* has all of the following:
- no polygenic background, no M type, and ≤ 2 causal variants per trait;
- the true ρ (or ρ = 0) given to both B and C;
- z generated with the same R̃ that B uses, i.e. η in the generator as well;
- an MC error for the class integrals reported well below the KL tolerance of 0.02 (e.g. ≥ 512 draws per configuration with mass > 1e-6).

Also fix the PVE definition, which is inconsistent:
- §1 defines λ = √N·β_std, but §2.2 still uses λ = √(Nq²/(1−q²)) and "λ up to about 150" (the correct figure is about 90).
- Fix: use one formula, set the residual variance to 1 − βᵀR_tβ, and cap the total trait PVE at ≤ 0.5 (resample otherwise).
- Without the cap, M loci (3 × 0.3) and the shifted arm (4 × 0.3) give a negative residual variance for trait 2.

**MF6. C3 wording (§8.2, §1).**
- §8.2 still says "a confident-call rate on W of 5%", and §1 points to §8.1 for the rule. Both should state the §0 rule.
- Pin the threshold choice: for each of t_S and t_D, take the *smallest* threshold whose achieved W rate is ≤ 2.5%. Set t_S first; S takes precedence. Report the achieved rates.
- The validation W set uses the same cells, weights and natural W sub-type mix as the headline.

## Should (not blocking)

- **Per-window ρ̂ SE is larger than stated.** Simulated SD at ρ = 0.3 is 0.15, 0.11 and 0.08 for 50, 100 and 200 pruned SNPs, against the stated 0.06–0.1. State it as "about 0.08–0.15".
  - Report the per-window ρ̂ bias by oracle class at ρ = 0. The |z| < 2 filter leaves LD-tagged signal in S loci.
  - R2 S8 (estimate both marginal variances) is still open.
- **R2 items S4–S7 are still open:**
  - power matching for S has no r²/d bin;
  - the ECE event and its weighting are undefined;
  - the per-block sign-test statistic is undefined;
  - class-determining pair: argmin d rather than argmax r².

  Freeze each as one sentence.
- **mvSuSiE adapter:** define a_l, w_l and p̃ per effect as Σ_j α_lj·1[lfsr condition] so that the inclusion-exclusion inputs are explicit.

## Checked and fine

- §1 partition and G wording (exhaustive and consistent with D; the weak shared variant does not block).
- Power thresholds 4.61/3.58/6.29/5.26.
- P_S/P_D coherence of all §4.3 adapters.
- Nested-ρ mapping.
- The §8.6 cell count (6 × 7 = 42).
