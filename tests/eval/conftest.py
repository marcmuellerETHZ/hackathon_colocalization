"""pytest setup for eval tests: import path = src/, temp files only under /tmp/claude-<uid>/ (project rule)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_TMP = Path(f"/tmp/claude-{os.getuid()}/pytest-eval")
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_TMP)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def pytest_configure(config):
    if not config.option.basetemp:
        config.option.basetemp = str(_TMP / f"run-{os.getpid()}")
