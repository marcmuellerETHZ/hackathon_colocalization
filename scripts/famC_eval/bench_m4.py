"""Benchmark Family C's per-cell harness runs on a fixed id list (evaluation side; reads truth).

scripts/run_benchmark.py with its expected ids restricted to --ids-file within each given locus set (per-cell
sets of valid_grid_v1, where each set holds only part of the list; scripts/baselines_p4.py bench requires every id
in every set). Everything else (metrics, strength-only control, bootstrap) is run_benchmark's.

    python scripts/famC_eval/bench_m4.py --ids-file results/baselines_m4/m4_subsample_v1/main.txt \
        --runs-json results/famC/m4_provisional_v1/runs.json -- --label famC_m4_provisional --unit block
--runs-json (written by scripts/famC_eval/run_harness.py) adds one --locus-set and one --reuse NAME=<run_dir> per
run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark as RB  # noqa: E402


def main():
    import json
    argv = sys.argv[1:]
    assert argv[0] == "--ids-file" and argv[2] == "--runs-json" and argv[4] == "--", \
        "usage: --ids-file F --runs-json J -- <run_benchmark args>"
    want = set(Path(argv[1]).read_text().split())
    rj = json.loads(Path(argv[3]).read_text())
    extra = []
    for r in rj["runs"]:
        extra += ["--locus-set", r["set"], "--reuse", f"{rj['method']}={r['run_dir']}"]
    orig = RB.expected_ids

    def expected_ids(locus_set, per_cell):
        assert not per_cell
        return [i for i in orig(locus_set, None) if i in want]   # orig applies the test-split guard

    RB.expected_ids = expected_ids
    return RB.main(extra + argv[5:])


if __name__ == "__main__":
    sys.exit(main())
