"""INTERFACES §2: no method module may open truth files. Static scan of src/fam*/ and src/baselines/ (Python and R),
plus a runtime check in eval.truth.read_truth. Each check has a positive control (a planted violation is caught)."""
import ast
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
METHOD_DIRS = ["src/baselines"] + [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src").glob("fam*"))]
_TRUTH_STR = re.compile(r"truth\.parquet|\.truth\b|read_truth|eval\.truth")


def violations(py_or_r_file: Path) -> list[str]:
    text = py_or_r_file.read_text(errors="replace")
    out = []
    if py_or_r_file.suffix == ".py":
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out += [a.name for a in node.names if a.name == "eval.truth" or a.name.endswith(".truth")]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "eval.truth" or mod.endswith(".truth") or (mod == "eval" and any(a.name == "truth" for a in node.names)):
                    out.append(f"from {mod} import ...")
                if any(a.name in ("read_truth",) for a in node.names):
                    out.append("read_truth")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _TRUTH_STR.search(node.value):
                out.append(f"string {node.value!r}")
    else:
        out += [m.group(0) for m in _TRUTH_STR.finditer(text)]
    return out


def method_files():
    for d in METHOD_DIRS:
        for f in (ROOT / d).rglob("*"):
            if f.suffix in (".py", ".R", ".r") and f.is_file():
                yield f


def test_method_code_never_touches_truth():
    bad = {str(f.relative_to(ROOT)): v for f in method_files() if (v := violations(f))}
    assert not bad, f"method code references truth files: {bad}"


@pytest.mark.parametrize("code", [
    "from eval.truth import read_truth\n",
    "import eval.truth\n",
    "from eval import truth\n",
    "import pandas as pd\nx = pd.read_parquet('data/x.truth.parquet')\n",
])
def test_scanner_positive_control(tmp_path, code):
    f = tmp_path / "m.py"
    f.write_text(code)
    assert violations(f)


def test_scanner_negative_control(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("from eval.io import read_locus\nfrom eval import adapters\n")
    assert not violations(f)
    r = tmp_path / "m.R"
    r.write_text("x <- arrow::read_parquet('a.truth.parquet')\n")
    assert violations(r)


def test_runtime_guard_blocks_method_packages(tmp_path):
    from eval import truth as T
    mod = types.ModuleType("famZ.fake")
    code = compile("def go(p):\n    return read_truth(p)\n", "famZ/fake.py", "exec")
    mod.__dict__["read_truth"] = T.read_truth
    mod.__dict__["__name__"] = "famZ.fake"
    exec(code, mod.__dict__)
    sys.modules["famZ.fake"] = mod
    try:
        with pytest.raises(PermissionError):
            mod.go(tmp_path / "nonexistent.truth.parquet")
    finally:
        del sys.modules["famZ.fake"]
