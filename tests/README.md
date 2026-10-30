# Tests

| Path | What it is | Runs from a fresh clone |
|---|---|---|
| `smoke/` | release smoke tests: every module imports, the recompute script reproduces the shipped benchmark on a subset, the README's headline numbers are those in `results/` | yes (numpy, scipy, pandas, pyarrow; h5py and pyyaml for some imports) |
| `<module>/` (`baselines`, `data`, `data_real`, `eval`, `famA`, `famB`, `famC`, `famG`, `orch`, `sim`, `stats`) | the project's original test suite, one folder per `src/` module | partly: many tests need the project's data (`data/processed/...`, 1000 Genomes genotypes), the R environments or the GPU, and skip or fail without them |

The mvp tests are in `../mvp/tests/`. From the repository root, `python -m pytest` runs `tests/smoke` and
`mvp/tests` (see `pytest.ini`). The pipeline tests skip unless Rscript with susieR 0.16.6 and coloc 5.2.3 is
available; point to it with `COLOC_BENCH_RSCRIPT` and `COLOC_BENCH_RLIB_SUSIE016`.

The original suite runs from a full replica of the environment, one module at a time:

```bash
source env/activate.sh
PYTHONPATH=src python -m pytest -q tests/eval
```

Changes from the project's `tests/`: the committed split and config files are read from `datasets/splits/` and
`datasets/configs/`; `orch/test_orch.py` no longer hard-codes the project location. Tests that build a
throw-away repository (`eval/test_eval_testrun.py`, `eval/test_testset_v2_lock.py`, `eval/test_eval_test_lock.py`)
still lay it out with `configs/` at its root, because the hash-frozen test-run code they exercise expects that.
