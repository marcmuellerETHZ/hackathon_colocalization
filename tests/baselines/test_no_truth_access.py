"""Baseline wrappers must never open truth files (INTERFACES §2)."""
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "baselines"


# The only allowed mention: passing `truth_path=None` to eval.harness.run_method, which DISABLES the harness's
# metric computation (baselines.run never lets the harness read oracle labels).
ALLOWED = ("truth_path=None",)


def test_no_truth_reference_in_wrappers():
    for f in list(SRC.rglob("*.py")) + list(SRC.rglob("*.R")):
        txt = f.read_text()
        for a in ALLOWED:
            txt = txt.replace(a, "")
        assert "truth" not in txt.lower(), f"{f} mentions truth files"


def test_allowed_token_only_disables(tmp_path):
    """Positive control: any other mention is caught; the allowed token is used only as harness kwarg."""
    assert "truth" in "x = pd.read_parquet('a.truth.parquet')".replace(ALLOWED[0], "")
    txt = (SRC / "run.py").read_text()
    assert txt.count("truth_path=None") == 1 and "truth_path=None, log=log)" in txt
