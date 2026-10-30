# p9 verify C2: independent recompute of the famC C2 kill test

**Verdict: CONFIRMED.** C2 fails on (iii): the network does not beat its no-LD ablation, so C is demoted to a fast pre-filter (DESIGN §5 C).

Scripts are in `reviews/p9_verify_C2/` (`recompute_c2.py`, `leakage_checks.py`, each with its `.out`). Neither imports src/eval or src/famC.

## Provenance
- **Checksums.** My own recompute of the manifest sha256 matches the registry for famC_C2@1, famC_C2_model@1, famC_nold_600k@1, famC_floor_600k@1, famB_m4_valid@1 and famA_m4_valid@1. The prediction sha256 for net, nold and floor matches each runs.json.
- **Checkpoint.** The net is `p9_main_600k/model.pt`; its log ends `EVAL step 18750 … DONE`, with 18750 = 8·600k/256. The r2 job is an unchanged resubmit after an OOM at step 968. No checkpoint was selected on validation.
- **Pilot first.** The timing pilot (11:17) projected 0.98 h at 600k, and a note fixed 600k at 11:21. Training started at 14:07.
- **Same records.** The net and nold configs differ only in run, variant, start and job (same train_shards and schedule; 4.8M loci seen each). The floor used `--max-train-shards 296`, so n_train = 600000 (90% fit, 10% early-stop). The `n_train: 200000` in its config.json is an unused default.

## Recompute (M4 main, 12,170 loci; realistic S-vs-D, §8.6 weights)
Weights: L1–L6 equal; P1/P2/P3 at 0.5/0.25/0.25; 5×5 power matching per cell. Failures are scored least-favourably; there were none on realistic S/D loci.

| Method | AUPRC (mine) | Claimed | Unweighted |
|---|---|---|---|
| net | 0.964092 | 0.9641 | 0.9701 |
| no-LD | 0.965374 | 0.9654 | 0.9702 |
| floor | 0.958795 | 0.9588 | 0.9618 |
| B | 0.833917 | 0.8339 | 0.8164 |
| A | 0.675014 | 0.6750 | 0.6765 |

Agreement is to 1e-15; sklearn's weighted AP agrees too.

Paired bootstrap: 80 LD blocks, 2,000 replicates, weights recomputed per replicate, own seed.

| Contrast | Difference | SE (mine / claimed) | Beats by > 1 SE? |
|---|---|---|---|
| net − no-LD | −0.0013 | 0.0023 / 0.0023 | **No** |
| net − floor | +0.0053 | 0.0033 / 0.0034 | Yes |
| net − B | +0.130 | 0.0075 / 0.0073 | Yes |

## Does the FAIL hold?
Yes. DESIGN_P2 §4 requires beating no-LD by more than 1 paired SE. The difference is negative, with 95% interval [−0.0053, +0.0035]. Conditions (i) and (ii) pass, but that cannot rescue the result. I took the ECE and region gap from c2.json without recomputing them.

## Leakage sanity check (context only)
- **Split overlap: none.** All 83 blocks touched by M4 windows are split = valid. All 773 blocks touched by train windows are split = train. No block is shared, and no window overlaps a train window at base-pair level.
- **Design cues: none.** Single-variable weighted AUPRCs are 0.50–0.52 for N1, N2, N_s, n_ref, p, ρ̂, ρ̂-SE, trait types and lead |z|; random scores 0.499.
- **A trivial LD-free score does not explain 0.96.** corr(|z1|, |z2|) over SNPs with |z| ≥ 3 scores 0.61; coloc.abf scores 0.855.
- **Floor gain.** The top features are susie_fallback (19%), coloc PP3 (16%) and susie_PP4 (10%). Metadata features total 4.3%.

Reading: I found no split or design-cue leakage. The gap between learned (~0.96) and classical (≤ 0.90) methods looks like in-simulator amortisation. The learners share M4's generator, priors and class definitions; only the windows differ. The floor reaches 0.959 just by recombining coloc and SuSiE outputs. M4 therefore tests transfer across regions, not across simulators. Held-out mechanism L6 stays at 0.918–0.926.

Not checked: the network's 30 global inputs (`glob_raw`) for label-derived quantities.
