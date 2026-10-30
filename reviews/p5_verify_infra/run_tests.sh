#!/bin/bash
# p5.verify_infra: full test suite, one pytest run per module, junit xml per module.
# Run inside an orch job (needs $ORCH_OUT). TMPDIR is redirected into $ORCH_OUT so pytest's
# tmp_path does not land in /tmp/pytest-of-$USER.
set -uo pipefail
export TMPDIR=$ORCH_OUT/tmp; mkdir -p "$TMPDIR"
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
mods=${MODS:-"baselines data data_real eval famA famB famC famG orch sim stats"}
echo "commit $(cat $ORCH_CODE/.commit 2>/dev/null || echo ${ORCH_COMMIT:-?})"
overall=0
for m in $mods; do
  echo "=================== MODULE $m"
  t0=$(date +%s)
  python -m pytest tests/$m -q -rfEsx -p no:cacheprovider --basetemp="$TMPDIR/pt_$m" \
      --junitxml="$ORCH_OUT/junit_$m.xml" 2>&1 | tail -60
  rc=${PIPESTATUS[0]}
  echo "MODULE $m rc=$rc seconds=$(( $(date +%s) - t0 ))"
  [ $rc -ne 0 ] && overall=1
done
python - <<'PY'
import glob, os, xml.etree.ElementTree as ET
out = os.environ["ORCH_OUT"]
print("\n=== SUMMARY (module: tests passed failed errors skipped)")
for f in sorted(glob.glob(f"{out}/junit_*.xml")):
    r = ET.parse(f).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    n, fl, er, sk = (int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    print(f"{os.path.basename(f)[6:-4]:10s} {n:4d} {n-fl-er-sk:4d} {fl:3d} {er:3d} {sk:3d}")
    for tc in s.iter("testcase"):
        for kind in ("failure", "error"):
            e = tc.find(kind)
            if e is not None:
                print(f"   {kind.upper()}: {tc.get('classname')}::{tc.get('name')} :: {(e.get('message') or '')[:200]}")
PY
exit $overall
