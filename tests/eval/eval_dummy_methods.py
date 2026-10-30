"""Dummy Methods for harness tests (importable by spawned workers via the inherited sys.path)."""
import hashlib
import json
import os
import subprocess
import time

import numpy as np


class SlowMethod:
    """Hangs (and spawns a grandchild `sleep`) on loci in `slow_ids`; writes the grandchild pid to `pid_file`."""
    name = "slow"

    def __init__(self, slow_ids=(), pid_file=None, sleep_s=60):
        self.slow_ids, self.pid_file, self.sleep_s = set(slow_ids), pid_file, sleep_s

    def fit_batch(self, loci):
        out = []
        for loc in loci:
            if loc.locus_id in self.slow_ids:
                p = subprocess.Popen(["sleep", str(self.sleep_s)])
                if self.pid_file:
                    with open(self.pid_file, "w") as fh:
                        fh.write(str(p.pid))
                time.sleep(self.sleep_s)
            out.append({"locus_id": loc.locus_id, "s": 0.5, "status": "ok"})
        return out


class FlakyMethod:
    """Raises on `fail_ids`, returns nonconverged on `nc_ids`, crashes the process on `crash_ids`,
    omits `missing_ids` from its output, reports an invalid status on `bad_status_ids`."""
    name = "flaky"

    def __init__(self, fail_ids=(), nc_ids=(), crash_ids=(), missing_ids=(), bad_status_ids=()):
        self.fail_ids, self.nc_ids, self.crash_ids = set(fail_ids), set(nc_ids), set(crash_ids)
        self.missing_ids, self.bad_status_ids = set(missing_ids), set(bad_status_ids)

    def fit_batch(self, loci):
        out = []
        for loc in loci:
            if loc.locus_id in self.crash_ids:
                os._exit(3)
            if loc.locus_id in self.fail_ids:
                raise RuntimeError(f"boom at {loc.locus_id}")
            if loc.locus_id in self.missing_ids:
                continue
            st = "nonconverged" if loc.locus_id in self.nc_ids else ("weird" if loc.locus_id in self.bad_status_ids else "ok")
            out.append({"locus_id": loc.locus_id, "s": float(np.tanh(abs(loc.z2).max())), "P_S": 0.3, "P_D": 0.3,
                        "status": st, "runtime_s": 0.001, "extra": {"k": 1}})
        return out


class BatchMethod:
    """Records batch sizes and p per batch; optionally touches the GPU."""
    name = "batchy"

    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

    def fit_batch(self, loci):
        ps = sorted({loc.p for loc in loci})
        dev = "cpu"
        if self.use_gpu:
            import torch
            if torch.cuda.is_available():
                x = torch.tensor(np.stack([loc.z1 for loc in loci if loc.p == loci[0].p]), device="cuda")
                _ = (x @ x.T).sum().item()
                dev = "cuda"
        return [{"locus_id": loc.locus_id, "s": 0.5, "status": "ok",
                 "extra": json.dumps({"batch_size": len(loci), "ps": ps, "device": dev})} for loc in loci]


class ToyColoc:
    """Informative toy scorer with known behaviour on eval_toy loci (AR(1) LD, S share a variant, D far apart):
    r2 = r²(lead1, lead2) in the given LD, strength q_t = sigmoid(max|z_t| - 5);
    P_S = r2 q1 q2, P_D = (1 - r2) q1 q2, s = r2 (ungated), plus optional N(0, noise) on r2 (seeded per locus)."""
    name = "toy_coloc"

    def __init__(self, noise: float = 0.0, name: str | None = None):
        self.noise = float(noise)
        if name:
            self.name = name

    def fit_batch(self, loci):
        out = []
        for loc in loci:
            i1, i2 = int(np.argmax(np.abs(loc.z1))), int(np.argmax(np.abs(loc.z2)))
            r2 = float(loc.ld_rows([i1])[0, i2] ** 2)
            if self.noise > 0:
                g = np.random.default_rng(int(hashlib.sha256(loc.locus_id.encode()).hexdigest()[:8], 16))
                r2 = float(np.clip(r2 + g.normal(0, self.noise), 0, 1))
            q1 = 1 / (1 + np.exp(-(np.abs(loc.z1).max() - 5)))
            q2 = 1 / (1 + np.exp(-(np.abs(loc.z2).max() - 5)))
            out.append({"locus_id": loc.locus_id, "s": r2, "P_S": r2 * q1 * q2, "P_D": (1 - r2) * q1 * q2,
                        "P_W": 1 - q1 * q2, "status": "ok"})
        return out


class NoLDCheck:
    """Asserts that the harness did not load LD (needs_ld = False)."""
    name = "no_ld_check"
    needs_ld = False

    def fit_batch(self, loci):
        for loc in loci:
            assert loc.R is None and loc.X_ref is None, "LD was loaded"
        return [{"locus_id": loc.locus_id, "s": 0.5, "status": "ok"} for loc in loci]


class RecordingColoc(ToyColoc):
    """ToyColoc that appends every locus id it fits to `log_file` (checkpoint/resume tests)."""
    name = "recording"

    def __init__(self, log_file, **kw):
        super().__init__(**kw)
        self.log_file = log_file

    def fit_batch(self, loci):
        with open(self.log_file, "a") as fh:
            for loc in loci:
                fh.write(loc.locus_id + "\n")
        return super().fit_batch(loci)


class BatchOOM(ToyColoc):
    """Raises a fake CUDA OOM on any batch of more than one locus; always fails the loci in `fail_ids`."""
    name = "batch_oom"

    def __init__(self, fail_ids=(), **kw):
        super().__init__(**kw)
        self.fail_ids = set(fail_ids)

    def fit_batch(self, loci):
        if len(loci) > 1:
            raise RuntimeError("CUDA out of memory (fake)")
        if loci[0].locus_id in self.fail_ids:
            raise RuntimeError("deterministic failure")
        return super().fit_batch(loci)


class LD2Check:
    """Reports whether trait-2 LD (R2 / X_ref2) reached fit_batch; `needs_ld2` set from the constructor."""
    name = "ld2_check"

    def __init__(self, needs_ld2=True):
        self.needs_ld2 = bool(needs_ld2)

    def fit_batch(self, loci):
        out = []
        for loc in loci:
            r2 = getattr(loc, "R2", None)
            out.append({"locus_id": loc.locus_id, "s": 0.5, "status": "ok",
                        "extra": json.dumps({"has_R2": r2 is not None,
                                             "R2_sum": None if r2 is None else float(np.asarray(r2, float).sum())})})
        return out


class RhoEcho:
    """Returns the rho_hat it was given in s (RhoOverride tests)."""
    name = "rho_echo"
    needs_ld = False

    def fit_batch(self, loci):
        return [{"locus_id": loc.locus_id, "s": 0.5, "P_S": 0.5, "P_D": 0.25, "status": "ok",
                 "extra": json.dumps({"rho_seen": float(loc.rho_hat)})} for loc in loci]
