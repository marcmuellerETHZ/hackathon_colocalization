"""Recompute full-mode manifest checksums (sha256 over 'relpath\\tsha256(file)\\n', os.walk sorted, followlinks)."""
import hashlib, os, sys, yaml
from pathlib import Path
HOME = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas")
def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for c in iter(lambda: fh.read(8 << 20), b""): h.update(c)
    return h.hexdigest()
def manifest(root):
    files = []
    for dp, dns, fns in os.walk(root, followlinks=True):
        dns.sort(); files += [Path(dp) / f for f in sorted(fns)]
    h = hashlib.sha256()
    for f in files: h.update(f"{f.relative_to(root)}\t{sha(f)}\n".encode())
    return h.hexdigest(), len(files)
arts = yaml.safe_load(open(HOME / "ARTIFACTS.yaml"))["artifacts"]
for name in ["famG_mm_identity", "famG_mm_timing", "famG_mm_slice2000"]:
    rec = [a for a in arts if a["name"] == name and str(a["version"]) == "1"][-1]
    ck, n = manifest(HOME / rec["path"])
    print(name, "n_files", n, "/", rec["n_files"], "match" if ck == rec["checksum"] else f"MISMATCH {ck}")
