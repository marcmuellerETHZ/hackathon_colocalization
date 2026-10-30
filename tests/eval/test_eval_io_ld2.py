"""famA request (p6.dryrun_testpipeline): opt-in trait-2 LD (R2 / X_ref2) through the shared reader and the
harness (methods declaring needs_ld2 = True)."""
import json

import numpy as np

import eval.io as eio
from eval import harness as H
from eval.io import write_locus_set
from eval_toy import make_locus


def test_trait2_ld_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(eio, "_splits_file", lambda: None)
    monkeypatch.delenv(eio.ACCESS_ENV, raising=False)
    rng = np.random.default_rng(1)
    a, _ = make_locus("withR2", "S", rng, p=12)
    b, _ = make_locus("noR2", "S", rng, p=12)
    a.R2 = (0.5 * a.R).astype(np.float32)
    a.extra_attrs = {"ld_type2": "dense", "n_ref2": 300}
    path = tmp_path / "ld2.h5"
    write_locus_set(path, [a, b])
    with eio.LocusSetReader(path) as r:
        assert r.read("withR2").R2 is None                       # default: not read
        got = r.read("withR2", load_ld2=True)
        assert np.array_equal(got.R2, a.R2) and got.X_ref2 is None and got.extra_attrs["ld_type2"] == "dense"
        assert r.read("noR2", load_ld2=True).R2 is None           # locus without trait-2 LD
    for flag, want in ((True, {"withR2": True, "noR2": False}), (False, {"withR2": False, "noR2": False})):
        rows = H.execute(H.MethodSpec("eval_dummy_methods:LD2Check", {"needs_ld2": flag}, "ld2_check"), path,
                         ["withR2", "noR2"], "ld2_check", n_workers=1, timeout_s=60, log=lambda s: None)
        assert {r["locus_id"]: json.loads(r["extra"])["has_R2"] for r in rows} == want
    rows = H.execute(H.MethodSpec("eval_dummy_methods:LD2Check", {"needs_ld2": True}, "ld2_check"), path,
                     ["withR2"], "ld2_check", n_workers=1, timeout_s=60, log=lambda s: None)
    assert abs(json.loads(rows[0]["extra"])["R2_sum"] - float(a.R2.astype(float).sum())) < 1e-3
    from famA.method import JointSusieMethod
    m = JointSusieMethod()
    assert hasattr(m, "needs_ld2")                                # famA declares it (True when per_trait_ld)
