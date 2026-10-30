#!/usr/bin/env python
"""The ONLY entry point that may produce or read test-split data (DESIGN §7.3). One test run only.

Test run (refuses unless ALL hold):
  1. the commit of the running code carries a PI-created git tag `frozen-*` (HEAD; inside an orch job the
     snapshot commit $ORCH_COMMIT, eval.io.code_identity);
  2. results/test_access.log as committed AT THAT COMMIT has no `completed` entry;
  3. the working-tree results/test_access.log has no entries beyond the committed ones (fresh start), or
     (--resume NONCE) only entries of that one open nonce, whose `started` commit is the running commit;
  4. the code is clean (outside orch: no modified tracked files, no untracked files under src/, scripts/,
     configs/; inside orch: the snapshot is a committed tree by construction);
  5. scripts/check_frozen.py passes (metrics, adapters, strength control, oracle, splits, DESIGN.md, configs);
  6. configs/testset_v1.yaml and the methods file exist; the methods file is frozen and filled in (status:
     frozen, methods with target and timeout_s, a primary candidate in the list, validation-fixed t_S/t_D for
     every method with P_S/P_D, the strength-only control with its frozen table, every target importable,
     the real-arm builder importable when real arms are configured).
Then appends `started` (fresh nonce) to results/test_access.log, exports it as $COLOC_TEST_ACCESS, and runs
the pipeline of src/eval/testrun.py (real arms, generation, every method, evaluation, verification). On any
error or signal it appends `interrupted` (the nonce stays open); `--resume NONCE` continues the SAME run
(finished stages, finished units and every finished locus are kept: the resumed run is the single test run,
DESIGN §11 M7). `completed` is appended when the bundle is written and verified. `--abandon NONCE` closes an
open nonce with `aborted` (PI decision). The PI commits results/test_access.log after the run.

Per-testset lock (DESIGN_P2 §6, M8; src/eval/testset_lock.py). `--testset configs/testset_v2.yaml` runs every
step above for test set v2 with v2's own lock: its access log (results/test_access_v2.log; the v1 log is never
read for the precondition nor written), its frozen-tag prefix (`frozen-v2`), its hash list
(configs/FROZEN_P2_SHA256) and its output directory. The default `--testset configs/testset_v1.yaml` has no
`lock:` block and keeps the phase-1 behaviour exactly (results/test_access.log, any `frozen-*` tag,
configs/FROZEN_SHA256). The v2 run checks in-run that no generation key overlaps v1 (eval.testrun).

Validation dry run (`--dry-run-valid`): the identical pipeline on validation blocks with every per-cell size
multiplied by --scale; no test access, no log entry; operating points / recalibration / strength table are
DRY placeholders fitted on the dry set. Re-running with the same --label resumes it.

Log line format (TSV): utc  event  commit  nonce  tag  detail

Usage:
  python scripts/eval_test.py --check                                   # preconditions only, writes nothing
  python scripts/eval_test.py --methods configs/test_methods_v1.yaml    # THE test run
  python scripts/eval_test.py --resume <nonce>                          # continue an interrupted test run
  python scripts/eval_test.py --dry-run-valid --methods configs/proposals/test_methods_dry_v1.yaml \\
      --scale 0.05 --label dry_v1 [--cores 32 --ram-gb 64]
  python scripts/eval_test.py --testset configs/testset_v2.yaml --methods configs/test_methods_v2.yaml  # v2 run
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG_REL = "results/test_access.log"
TESTSET_REL = "configs/testset_v1.yaml"
OPEN_EVENTS = ("started", "interrupted", "resumed")
CLOSE_EVENTS = ("completed", "aborted")


def _git(repo: Path, *args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


def _events(text: str) -> list[list[str]]:
    return [l.split("\t") for l in text.splitlines() if l.strip() and not l.startswith("#")]


def _identity(repo: Path) -> tuple[Path, str, bool]:
    """(git repo, commit of the running code, inside an orch snapshot)."""
    from eval.io import code_identity, in_orch_snapshot
    snap = in_orch_snapshot(repo)
    git_repo, commit = code_identity(repo)
    return Path(git_repo), commit, snap


def _frozen_tags(git_repo: Path, commit: str, prefix: str = "frozen-") -> list[str]:
    return [t for t in _git(git_repo, "tag", "--points-at", commit).stdout.split() if t.startswith(prefix)]


def _lock(repo: Path, testset_rel: str = TESTSET_REL):
    """The test-set config's lock (eval.testset_lock; v1 defaults without a `lock:` block)."""
    import yaml
    from eval.testset_lock import lock_for
    p = repo / testset_rel
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {"seed": 0}
    return lock_for(cfg, testset_rel)


def check_preconditions(repo: Path = ROOT, methods_file: Path | None = None, check_frozen_files: bool = True,
                        resume: str | None = None, testset_rel: str = TESTSET_REL) -> list[str]:
    """List of reasons to refuse (empty = allowed). testset_rel selects the test set and its lock."""
    refusals = []
    lk = _lock(repo, testset_rel)
    log_rel = lk.access_log
    if lk.testset_id != "v1":
        from eval.testset_lock import lock_problems
        refusals += lock_problems(lk)
    git_repo, commit, snap = _identity(repo)
    if not _frozen_tags(git_repo, commit, lk.frozen_tag_prefix):
        refusals.append(f"the running commit {commit[:10]} carries no {lk.frozen_tag_prefix}* tag")
    committed = _git(git_repo, "show", f"{commit}:{log_rel}", check=False)
    committed_ev = _events(committed.stdout if committed.returncode == 0 else "")
    if any(len(e) > 1 and e[1] == "completed" for e in committed_ev):
        refusals.append(f"committed {log_rel} already has a completed test run (one test run only)")
    wt = repo / log_rel
    wt_ev = _events(wt.read_text() if wt.exists() else "")
    if resume is None:
        if wt_ev != committed_ev:
            refusals.append(f"{log_rel} has uncommitted entries; the PI must commit it first "
                            "(or resume the open run with --resume NONCE)")
    else:
        if wt_ev[:len(committed_ev)] != committed_ev:
            refusals.append(f"{log_rel} differs from its committed version")
        mine = [e for e in wt_ev if len(e) > 3 and e[3] == resume]
        others = [e for e in wt_ev[len(committed_ev):] if len(e) <= 3 or e[3] != resume]
        if not any(e[1] == "started" for e in mine):
            refusals.append(f"nonce {resume} has no started entry")
        if any(e[1] in CLOSE_EVENTS for e in mine):
            refusals.append(f"nonce {resume} is closed (completed/aborted); it cannot be resumed")
        if others:
            refusals.append(f"uncommitted log entries of other nonces: {len(others)}")
        st = [e for e in mine if e[1] == "started"]
        if st and st[0][2] != commit:
            refusals.append(f"nonce {resume} was started at {st[0][2][:10]}, the running code is {commit[:10]}")
    if not snap:
        tracked = [l for l in _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.splitlines() if l.strip()]
        tracked = [l for l in tracked if not l.endswith(log_rel)]
        untracked = [l for l in _git(repo, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts",
                                     "configs").stdout.splitlines() if l.startswith("??")]
        if tracked or untracked:
            refusals.append(f"working tree not clean ({len(tracked)} modified tracked, {len(untracked)} untracked code files)")
    if check_frozen_files:
        import check_frozen
        probs = check_frozen.check(repo, repo / lk.frozen_hashes)
        if probs:
            refusals.append(f"check_frozen failed: {probs[:5]}")
    if not (repo / testset_rel).exists():
        refusals.append(f"{testset_rel} missing")
    if methods_file is not None and not Path(methods_file).exists():
        refusals.append(f"methods file {methods_file} missing")
    elif methods_file is not None:
        refusals += check_methods_file(Path(methods_file))
        refusals += plumbing_problems(repo, Path(methods_file), "test", testset_rel)
    return refusals


def check_methods_file(path: Path) -> list[str]:
    """The methods file must be filled in and frozen by the PI (configs/test_methods_v1.yaml template)."""
    import yaml
    out = []
    try:
        d = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"methods file unreadable: {e}"]
    if d.get("status", "frozen") != "frozen":
        out.append(f"methods file status is {d.get('status')!r}, not 'frozen'")
    ms = d.get("methods") or []
    if not ms:
        out.append("methods file lists no methods")
    names = []
    for i, m in enumerate(ms):
        if not isinstance(m, dict) or "target" not in m or "timeout_s" not in m:
            out.append(f"methods[{i}] needs at least target and timeout_s")
            continue
        names.append(m.get("name") or m["target"])
    cands = d.get("candidates") or {}
    prim = cands.get("primary", "__unset__")
    if prim is None and not cands.get("decision"):
        # D16 (MVR route) allows primary: null, but only with the decision note named (candidates.decision), so an
        # unfilled template is still refused
        out.append("no primary candidate named (candidates.primary) and no decision note (candidates.decision)")
    elif prim not in ("__unset__", None) and prim not in names:
        out.append(f"primary candidate {prim!r} is not in the methods list")
    return out


def plumbing_problems(repo: Path, methods_file: Path, mode: str, testset_rel: str = TESTSET_REL) -> list[str]:
    import yaml
    from eval.testrun import check_methods_runnable
    try:
        doc = yaml.safe_load(Path(methods_file).read_text()) or {}
    except Exception as e:  # noqa: BLE001
        return [f"methods file unreadable: {e}"]
    if not (doc.get("methods") or []):
        return []
    testset = yaml.safe_load((repo / testset_rel).read_text()) if (repo / testset_rel).exists() else None
    return check_methods_runnable(doc, repo, mode, testset)


def append_log(repo: Path, event: str, commit: str, nonce: str, tag: str, detail: str = "",
               log_rel: str = LOG_REL) -> None:
    p = repo / log_rel
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a") as fh:
        if new:
            fh.write("# utc\tevent\tcommit\tnonce\ttag\tdetail\n")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fh.write("\t".join([ts, event, commit, nonce, tag, detail.replace("\t", " ").replace("\n", " ")[:2000]]) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _entry(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _sigterm(signum, frame):  # orch cancel / scheduler timeout: unwind the main thread so the log is written
    raise KeyboardInterrupt(f"signal {signum}")


def _pipeline(repo: Path, methods_file: Path, mode: str, bundle: Path, data_dir: Path, label: str, opts: dict,
              extra: dict, testset_rel: str = TESTSET_REL):
    import yaml
    from eval.testrun import Pipeline, RunConfig
    testset = yaml.safe_load((repo / testset_rel).read_text())
    doc = yaml.safe_load(Path(methods_file).read_text())
    cfg = RunConfig(mode=mode, bundle=bundle, data_dir=data_dir, label=label, **opts)
    return Pipeline(cfg, testset, doc, repo=repo, extra_manifest=extra), testset


def run_test(repo: Path, methods_file: Path, out_dir: Path, *, resume: str | None = None, opts: dict | None = None,
             bundle: Path | None = None, testset_rel: str = TESTSET_REL) -> dict:
    """THE test run (or its resumption) of the test set `testset_rel` (its own lock). Writes started/resumed, then
    completed or interrupted, to that test set's access log."""
    lk = _lock(repo, testset_rel)
    git_repo, commit, _ = _identity(repo)
    tag = _frozen_tags(git_repo, commit, lk.frozen_tag_prefix)[0]
    nonce = resume or uuid.uuid4().hex
    label = f"test_{tag}_{nonce[:8]}"
    bundle = bundle or (repo / "results" / "eval_test" / label)
    detail = f"host={socket.gethostname()} methods={methods_file} sha256={_sha256(methods_file)} bundle={bundle}"
    if lk.testset_id != "v1":
        detail += f" testset={lk.testset_id} config={testset_rel} sha256={_sha256(repo / testset_rel)}"

    def log_event(event, d):
        append_log(repo, event, commit, nonce, tag, d, log_rel=lk.access_log)
    log_event("resumed" if resume else "started", detail)
    from eval.io import ACCESS_ENV
    os.environ[ACCESS_ENV] = nonce
    old = signal.signal(signal.SIGTERM, _sigterm)
    try:
        pipe, testset = _pipeline(repo, methods_file, "test", bundle, out_dir / label, label, opts or {},
                                  {"test_run": True, "tag": tag, "nonce": nonce, "commit": commit,
                                   "methods_file": str(methods_file), "methods_file_sha256": _sha256(methods_file),
                                   "testset_id": lk.testset_id, "testset_config": testset_rel,
                                   "access_log": lk.access_log},
                                  testset_rel)
        gen = _entry(testset["generator"]["entry"])
        res = pipe.run(gen)
        if res.get("verify") and not res["verify"]["ok"]:
            raise RuntimeError(f"bundle verification failed: {json.dumps(res['verify']['problems'])[:1500]}")
        log_event("completed", f"bundle={bundle}")
        return res
    except BaseException as e:
        log_event("interrupted", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"\nTEST RUN INTERRUPTED; resume with: python scripts/eval_test.py --testset {testset_rel} "
              f"--resume {nonce} --methods {methods_file}", file=sys.stderr)
        raise
    finally:
        signal.signal(signal.SIGTERM, old)
        os.environ.pop(ACCESS_ENV, None)


def run_dry(repo: Path, methods_file: Path, label: str, scale: float, opts: dict, bundle: Path | None = None,
            data_dir: Path | None = None, testset_rel: str = TESTSET_REL) -> dict:
    """Validation dry run of the identical pipeline (no test access, no log entry)."""
    import yaml
    from eval.io import ACCESS_ENV
    os.environ.pop(ACCESS_ENV, None)
    if "__" in label:
        raise SystemExit("label must not contain '__'")
    lock = check_preconditions(repo, methods_file, check_frozen_files=True, testset_rel=testset_rel)
    plumbing = plumbing_problems(repo, methods_file, "dry", testset_rel)
    if plumbing:
        raise SystemExit("DRY RUN REFUSED (plumbing):\n  - " + "\n  - ".join(plumbing))
    bundle = bundle or (repo / "results" / "eval_test_dry" / label)
    data_dir = data_dir or (repo / "data" / "processed" / "eval_test_dry" / label)
    git_repo, commit, snap = _identity(repo)
    extra = {"DRY": True, "commit": commit, "orch_snapshot": snap, "methods_file": str(methods_file),
             "testset_config": testset_rel, "testset_sha256": _sha256(repo / testset_rel),
             "methods_file_sha256": _sha256(methods_file),
             "test_mode_lock_refusals_observed": lock,
             "note": "validation blocks only; operating points / recalibration / strength table are DRY placeholders"}
    print("test-mode lock checks (informational in a dry run):")
    for r in lock:
        print(f"  - would refuse: {r}")
    pipe, testset = _pipeline(repo, methods_file, "dry", bundle, data_dir, label, dict(opts, scale=scale), extra,
                              testset_rel)
    old = signal.signal(signal.SIGTERM, _sigterm)
    try:
        from sim.testset import generate
        return pipe.run(generate)
    finally:
        signal.signal(signal.SIGTERM, old)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check preconditions only")
    ap.add_argument("--methods", type=Path, default=ROOT / "configs" / "test_methods_v1.yaml")
    ap.add_argument("--testset", default=TESTSET_REL,
                    help="test-set config (relative to the repo); selects the test set and its lock (DESIGN_P2 §6)")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for generated test locus sets (default: the test set's lock.default_out; "
                         "v1: data/processed/testset_v1)")
    ap.add_argument("--resume", default=None, metavar="NONCE", help="continue the interrupted test run NONCE")
    ap.add_argument("--abandon", default=None, metavar="NONCE", help="close an open nonce with `aborted` (PI)")
    ap.add_argument("--dry-run-valid", action="store_true", help="identical pipeline on validation blocks")
    ap.add_argument("--scale", type=float, default=0.05, help="dry run: multiplier of every per-cell size")
    ap.add_argument("--label", default=None, help="dry run: bundle label (re-use it to resume)")
    ap.add_argument("--bundle", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--cores", type=int, default=int(os.environ.get("ORCH_CORES", 64)))
    ap.add_argument("--ram-gb", type=float, default=float(os.environ.get("ORCH_RAM_GB", 120)))
    ap.add_argument("--gen-workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--accept-failed-units", action="store_true",
                    help="finish the bundle although some (method x set) units failed twice (reported as not run)")
    ap.add_argument("--methods-only", nargs="*", default=None, help="dry run only: restrict to these methods")
    ap.add_argument("--dry-real-builder", action="store_true",
                    help="dry run: build the real arms with real_data_arms.builder in its validation mode")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="dry run only: subset of real sim methods evaluate verify (e.g. `--stages real`)")
    a = ap.parse_args(argv)
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        print("source env/activate.sh first", file=sys.stderr)
        return 2
    opts = {"cores": a.cores, "ram_gb": a.ram_gb, "gen_workers": a.gen_workers, "n_boot": a.n_boot,
            "n_jobs": a.n_jobs, "accept_failed_units": a.accept_failed_units}
    if a.dry_run_valid:
        if a.resume or a.abandon:
            ap.error("--resume/--abandon are test-run options; re-run a dry run with the same --label")
        label = a.label or f"dry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        if a.methods_only:
            opts["methods_only"] = tuple(a.methods_only)
        if a.dry_real_builder:
            opts["dry_real_builder"] = True
        if a.stages:
            opts["stages"] = tuple(a.stages)
        res = run_dry(ROOT, a.methods, label, a.scale, opts, bundle=a.bundle, data_dir=a.data_dir,
                      testset_rel=a.testset)
        ok = (res.get("verify") or {}).get("ok")
        print(f"DRY RUN {'OK' if ok else 'FINISHED WITH PROBLEMS'}: {sorted(res['evaluate']['benchmark']['parts'])} "
              f"bundle {a.bundle or ROOT / 'results' / 'eval_test_dry' / label}")
        return 0 if ok else 1
    if a.methods_only or a.stages or a.dry_real_builder:
        ap.error("--methods-only / --stages / --dry-real-builder are dry-run options")
    lk = _lock(ROOT, a.testset)
    out_dir = a.out if a.out is not None else ROOT / lk.default_out
    if a.abandon:
        git_repo, commit, _ = _identity(ROOT)
        tags = _frozen_tags(git_repo, commit, lk.frozen_tag_prefix)
        append_log(ROOT, "aborted", commit, a.abandon, tags[0] if tags else "", "abandoned by the PI (--abandon)",
                   log_rel=lk.access_log)
        print(f"nonce {a.abandon} closed; the PI must commit {lk.access_log}")
        return 0
    refusals = check_preconditions(ROOT, a.methods, resume=a.resume, testset_rel=a.testset)
    if refusals:
        print("REFUSED:")
        for r in refusals:
            print(f"  - {r}")
        return 1
    print("preconditions OK")
    if a.check:
        return 0
    run_test(ROOT, a.methods, out_dir, resume=a.resume, opts=opts, bundle=a.bundle, testset_rel=a.testset)
    print(f"test run completed; the PI must now commit {lk.access_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
