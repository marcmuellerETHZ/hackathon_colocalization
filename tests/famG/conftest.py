import os
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
for _p in (PROJECT / "src", PROJECT / "tests" / "famG"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
RSCRIPT = PROJECT / "env" / "renv" / "bin" / "Rscript"


@pytest.fixture(scope="session")
def has_r():
    if not RSCRIPT.exists():
        if os.environ.get("ORCH_JOB_ID"):       # inside an orch job the identity gates must run (review p2)
            pytest.fail(f"R not available at {RSCRIPT} inside orch job {os.environ['ORCH_JOB_ID']}")
        pytest.skip("R not available")
    return True


@pytest.fixture(scope="session")
def cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
