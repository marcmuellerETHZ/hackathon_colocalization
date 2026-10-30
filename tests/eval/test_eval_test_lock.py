"""Test lock (DESIGN §7.3) and hash-protected definitions (§7.4), exercised in a throw-away git repo."""
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

import check_frozen
import eval.io as eio
import eval_test

ROOT = Path(__file__).resolve().parents[2]

FAKE_GEN = '''
from pathlib import Path
import numpy as np, pandas as pd
from eval.io import write_locus_set
from eval.truth import write_truth
from eval_toy import make_locus

def generate(testset, out_dir, seed):
    rng = np.random.default_rng(seed)
    locs, truths = [], []
    for i, cls in enumerate(["S", "D", "W"] * 4):
        loc, t = make_locus(f"t{i}", cls, rng, split="test", block=f"toyb{i % 3}")
        locs.append(loc); truths.append(t)
    f = Path(out_dir) / "test_sim.h5"
    write_locus_set(f, locs)
    write_truth(pd.DataFrame(truths), Path(out_dir) / "test_sim.truth.parquet")
    return [f]
'''


def git(repo, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    (r / "src" / "eval").mkdir(parents=True)
    shutil.copy(ROOT / "src" / "eval" / "metrics.py", r / "src" / "eval" / "metrics.py")
    ts = yaml.safe_load((ROOT / "datasets" / "configs" / "testset_v1.yaml").read_text())
    ts["generator"]["entry"] = "fake_gen:generate"
    (r / "configs" / "testset_v1.yaml").write_text(yaml.safe_dump(ts))
    (r / "configs" / "FROZEN_SHA256").write_text("# test\nPENDING  src/eval/metrics.py\nPENDING  configs/testset_v1.yaml\n")
    check_frozen.update(r)
    (r / "configs" / "test_methods_v1.yaml").write_text(yaml.safe_dump(
        {"methods": [{"target": "eval.reference_methods:LeadSNPRule", "kwargs": {"r2_min": 0.8}, "timeout_s": 60,
                      "n_workers": 2}]}))
    (r / ".gitignore").write_text("data/\n")
    git(r, "init", "-q")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    gen = tmp_path / "genmod"
    gen.mkdir()
    (gen / "fake_gen.py").write_text(FAKE_GEN)
    monkeypatch.syspath_prepend(str(gen))
    monkeypatch.setattr(eio, "PROJECT_ROOT", r)
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    return r


def test_check_frozen_detects_changes(repo):
    assert check_frozen.check(repo) == []
    f = repo / "src" / "eval" / "metrics.py"
    f.write_text(f.read_text() + "\n# drift\n")
    assert check_frozen.check(repo) == ["CHANGED  src/eval/metrics.py"]
    (repo / "configs" / "testset_v1.yaml").unlink()
    assert any(p.startswith("MISSING") for p in check_frozen.check(repo))


def test_design_section_hash_only_covers_that_section(tmp_path):
    md = "# T\n## 1. One\na\n## 2. Two\nb\n## 8. Eight\nc\n## 9. Nine\nd\n"
    assert check_frozen.section_text(md, "1") == "## 1. One\na\n"
    assert check_frozen.section_text(md, "8") == "## 8. Eight\nc\n"
    (tmp_path / "D.md").write_text(md)
    h8 = check_frozen.entry_hash(tmp_path, "D.md#§8")
    (tmp_path / "D.md").write_text(md.replace("d\n", "changed\n"))  # other section edited
    assert check_frozen.entry_hash(tmp_path, "D.md#§8") == h8
    (tmp_path / "D.md").write_text(md.replace("c\n", "changed\n"))
    assert check_frozen.entry_hash(tmp_path, "D.md#§8") != h8


def test_real_design_sections_exist():
    md = (ROOT / "DESIGN.md").read_text()
    assert check_frozen.section_text(md, "1").startswith("## 1. Estimand")
    assert check_frozen.section_text(md, "8").startswith("## 8. Metrics")

# The lifecycle tests of the single test run (started / interrupted / resumed / completed, refusals, truth
# unreadable after completion) moved to test_eval_testrun.py (p6.dryrun_testpipeline: the pipeline changed).
