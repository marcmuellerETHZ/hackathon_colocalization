# Manuscript

- `manuscript.md`: main text in PLOS Computational Biology structure. Figures are referenced from `figures/`.
- `S1_Text_failed_approaches.md`: S1 Text, the approaches that did not work.
- `S2_Text_agent_process.md`: S2 Text, the agent process record.
- `figures/`: Fig1–Fig4 as PNG for viewing and RGB TIFF (LZW, 300 dpi, ≤ 7.5 in wide) for submission.
  - `make_figures.py` regenerates them. It is kept as it ran, so it reads the original project layout. In this
    repository the same files (byte-identical) are:

    | Path in `make_figures.py` | File in this repository |
    |---|---|
    | `results/eval_test/test_frozen-v1_ab94c5a1/benchmarks/sim/summary.md` | `results/test_v1/benchmarks/sim/summary.md` |
    | `results/eval_test/test_frozen-v1_ab94c5a1/runtime.json` | `results/test_v1/runtime.json` |
    | `results/famC/c2_v1/c2.json` | `results/phase2_validation/famC_C2/c2.json` |
    | `results/famB/p9_m4_eval/m4_Bmm_v2.json` | `results/phase2_validation/famB_mm_M4/m4_Bmm_v2.json` |
    | `results/baselines_m4/m4_valid_v3/m4summary.md` (cited in a comment) | `results/phase2_validation/m4_valid_baselines_v3/m4summary.md` |

  - `Fig2_mismatch_data.json` and `Fig3_speed_accuracy_data.json` hold the plotted values.

The manuscript text refers to the current repository layout.


The manuscript was fact-checked by an independent agent against the project record and Crossref. The fixes are applied.
