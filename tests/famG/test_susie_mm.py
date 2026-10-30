"""G-mm (famG.susie_mm): known-answer identity vs susieR 0.16.6 eb_mix on toy loci, Triton vs torch, padding,
factor vs dense, fp32 vs fp64 and the Method contract."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from famag_testutils import PROJECT, run_r, toy_locus
from famG.susie_mm import MMConfig, susie_mm_batch
from famG.validation.mm_loci import export

RLIB = str(PROJECT / "env" / "rlib_susie016")
RUNNER = PROJECT / "src" / "famG" / "validation" / "run_susie016_mm.R"


def _toys(seed=7):
    """Loci with a finite reference: z from in-sample genotypes (n_ind), LD from an independent reference of size
    n_ref drawn from the same AR process (mismatch), so lambda > 0 occurs and the eb_mix gate is exercised."""
    rng = np.random.default_rng(seed)
    out = []
    specs = [(120, 200, (10,), (8.0,), (10,), (8.0,)),          # shared
             (150, 300, (20,), (7.0,), (90,), (7.0,)),          # distinct
             (100, 150, (), (), (), ()),                         # null
             (160, 120, (5, 70), (9.0, -6.0), (70,), (12.0,)),   # two signals
             (140, 150, (30,), (30.0,), (30,), (25.0,))]         # very strong (gate prior odds overflow branch)
    for p, n_ref, c1, l1, c2, l2 in specs:
        X, R, z1, z2 = toy_locus(rng, 2000, p, c1, l1, c2, l2)
        Xr, Rr, _, _ = toy_locus(rng, n_ref, p)
        out.append({"id": f"toy_p{p}_{len(out)}", "z": [z1, z2], "n": [2000.0, 1500.0], "n_ref": n_ref, "R": Rr,
                    "X": Xr})
    return out


@pytest.fixture(scope="module")
def r_ref(has_r):
    loci = _toys()
    tmp = Path(tempfile.mkdtemp(prefix="claude-famGmm-"))
    for d in loci:
        export({k: v for k, v in d.items() if k != "X"}, tmp / "loci")
    ids = [d["id"] for d in loci]
    run_r(f'commandArgs <- function(trailingOnly = TRUE) c("{tmp}/loci", "{tmp}/r", {", ".join(repr(i).replace(chr(39), chr(34)) for i in ids)}); '
          f'source("{RUNNER}")', r_libs=RLIB)
    return loci, {i: json.load(open(tmp / "r" / f"{i}.json")) for i in ids}


def _check(loci, R, fits, tol_pip):
    lam_pos = 0
    for d, f in zip(loci, fits):
        for t in (0, 1):
            r = R[d["id"]][f"t{t + 1}"]
            g = f[t]
            assert (r.get("error") is not None) == (g["status"] != "ok")
            if g["status"] != "ok":
                continue
            assert np.abs(g["pip"] - np.asarray(r["pip"])).max() < tol_pip
            assert g["niter"] == r["niter"] and g["converged"] == r["converged"]
            assert abs(g["lambda_bias"] - r["lambda_bias"]) < 1e-5
            lam_pos += r["lambda_bias"] > 0
    return lam_pos


def test_identity_vs_susieR_0166_cpu_fp64(r_ref):
    loci, R = r_ref
    fits = susie_mm_batch([{k: v for k, v in d.items() if k != "X"} for d in loci],
                          MMConfig(ld_dtype=torch.float64, use_triton=False), device=torch.device("cpu"))
    assert _check(loci, R, fits, 1e-6) >= 1          # the eb_mix gate (lambda > 0) is exercised


def test_identity_factor_ld_cpu(r_ref):
    """Factor LD X (R = X'X/n_ref, n_ref < p/2 for some loci) equals the dense path."""
    loci, R = r_ref
    cfg = MMConfig(ld_dtype=torch.float64, use_triton=False, factor_ratio=10.0)   # force factor everywhere
    ins = []
    for d in loci:
        X = d["X"]
        ins.append({"z": d["z"], "n": d["n"], "n_ref": d["n_ref"], "X": X})
    fits = susie_mm_batch(ins, cfg, device=torch.device("cpu"))
    # the toy R in r_ref is the reference R (= X'X/n_ref of the standardised X)
    _check(loci, R, fits, 1e-6)


def test_identity_gpu_triton_fp64_and_fp32(r_ref, cuda):
    loci, R = r_ref
    ins = [{k: v for k, v in d.items() if k != "X"} for d in loci]
    f64 = susie_mm_batch(ins, MMConfig(ld_dtype=torch.float64, use_triton=True), device=cuda)
    _check(loci, R, f64, 1e-6)
    f64t = susie_mm_batch(ins, MMConfig(ld_dtype=torch.float64, use_triton=False), device=cuda)
    for a, b in zip(f64, f64t):
        for t in (0, 1):
            assert np.abs(a[t]["pip"] - b[t]["pip"]).max() < 1e-7
    f32 = susie_mm_batch(ins, MMConfig(ld_dtype=torch.float32, use_triton=True), device=cuda)
    for a, b in zip(f64, f32):
        for t in (0, 1):
            assert np.abs(a[t]["pip"] - b[t]["pip"]).max() < 1e-3


def test_padding_and_batch_invariance(r_ref):
    """Fitting loci together (padded to a common width) or one at a time gives the same answer (up to the rounding
    of a different reduction width, amplified over up to 50 iterations of non-converging fits: < 1e-6)."""
    loci, _ = r_ref
    ins = [{k: v for k, v in d.items() if k != "X"} for d in loci]
    cfg = MMConfig(ld_dtype=torch.float64, use_triton=False, bucket_ratio=10.0)
    together = susie_mm_batch(ins, cfg, device=torch.device("cpu"))
    for d, f in zip(ins, together):
        one = susie_mm_batch([d], cfg, device=torch.device("cpu"))[0]
        for t in (0, 1):
            assert np.abs(one[t]["pip"] - f[t]["pip"]).max() < 1e-6
            assert one[t]["niter"] == f[t]["niter"]


def test_method_contract(cuda):
    from famG.method_mm import GPUColocSusieMM
    from famG.locus import LocusData
    rng = np.random.default_rng(3)
    p = 200
    cases = {"shared": ((50,), (9.0,), (50,), (9.0,)), "distinct": ((20,), (9.0,), (150,), (9.0,)),
             "null": ((), (), (), ())}
    loci = []
    for name, (c1, l1, c2, l2) in cases.items():
        X, R, z1, z2 = toy_locus(rng, 3000, p, c1, l1, c2, l2)
        n = np.full(p, 3000.0)
        maf = np.full(p, 0.3)
        se = 1 / np.sqrt(2 * n * maf * 0.7)
        loci.append(LocusData(locus_id=name, z1=z1, z2=z2, n1=n, n2=n, R=R.astype(np.float32), ld_type="dense",
                              n_ref=3000, beta1=z1 * se, se1=se, beta2=z2 * se, se2=se, maf1=maf, maf2=maf))
    rows = {r["locus_id"]: r for r in GPUColocSusieMM(device=cuda).fit_batch(loci)}
    assert rows["shared"]["P_S"] > 0.9
    assert rows["distinct"]["P_D"] > 0.9
    assert rows["null"]["P_W"] > 0.9


def test_gemv_kernels(cuda):
    import famG.mm_kernels as K
    g = torch.Generator(device="cpu").manual_seed(0)
    for (B, Kk, I, J) in [(3, 2, 300, 300), (2, 1, 150, 517), (4, 5, 77, 1000)]:
        M = torch.randn(B, I, J, generator=g, dtype=torch.float64).to(cuda)
        v = torch.randn(B, Kk, I, generator=g, dtype=torch.float64).to(cuda)
        w = torch.randn(B, Kk, J, generator=g, dtype=torch.float64).to(cuda)
        assert (K.vtm(v, M) - torch.bmm(v, M)).abs().max().item() < 1e-10
        assert (K.mv(w, M) - torch.bmm(w, M.transpose(1, 2))).abs().max().item() < 1e-10
