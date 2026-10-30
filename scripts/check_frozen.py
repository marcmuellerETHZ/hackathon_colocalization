#!/usr/bin/env python
"""Hash-protected definitions (DESIGN §7.4).

configs/FROZEN_SHA256 has one entry per line:   <sha256 or PENDING>  <path>[#§<n>]
A `#§<n>` suffix hashes only section n of a Markdown file: from the line starting with "## <n>." up to (not
including) the next line starting with "## ". Lines starting with '#' are comments.

  python scripts/check_frozen.py            # verify; exit 1 on any mismatch, missing file or PENDING entry
  python scripts/check_frozen.py --update   # PI only: rewrite the hashes of all listed entries
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The hash lists name paths in the original project layout (configs/, splits/, DESIGN_P2.md at the root). In the
# released layout these live under datasets/ and docs/; an entry is looked up at its listed path first, then here.
RELOCATED = {"configs/": "datasets/configs/", "splits/": "datasets/splits/", "DESIGN_P2.md": "docs/DESIGN_P2.md"}


def locate(root: Path, rel: str) -> Path:
    """root/rel if it exists, else its released location (RELOCATED), else root/rel."""
    p = root / rel
    if p.exists():
        return p
    for old, new in RELOCATED.items():
        if rel == old or (old.endswith("/") and rel.startswith(old)):
            q = root / (new + rel[len(old):])
            if q.exists():
                return q
    return p


def section_text(md: str, n: str) -> str:
    lines = md.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(rf"^## {re.escape(n)}\.(\s|$)", l)), None)
    if start is None:
        raise KeyError(f"section {n} not found")
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    return "".join(lines[start:end])


def entry_hash(root: Path, entry: str) -> str:
    path, _, sec = entry.partition("#§")
    p = locate(root, path)
    if not p.exists():
        raise FileNotFoundError(path)
    if sec:
        return hashlib.sha256(section_text(p.read_text(), sec).encode()).hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse(frozen: Path) -> list[tuple[str, str, str]]:
    """[(hash, entry, raw line)]; comments kept as ('', '', line)."""
    out = []
    for line in frozen.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(("", "", line))
            continue
        h, entry = line.split(None, 1)
        out.append((h, entry.strip(), line))
    return out


def check(root: Path = ROOT, frozen: Path | None = None) -> list[str]:
    frozen = frozen or locate(root, "configs/FROZEN_SHA256")
    problems = []
    for h, entry, _ in parse(frozen):
        if not entry:
            continue
        if h == "PENDING":
            problems.append(f"PENDING  {entry}")
            continue
        try:
            got = entry_hash(root, entry)
        except (FileNotFoundError, KeyError) as e:
            problems.append(f"MISSING  {entry} ({e})")
            continue
        if got != h:
            problems.append(f"CHANGED  {entry}")
    return problems


def update(root: Path = ROOT, frozen: Path | None = None) -> list[str]:
    frozen = frozen or locate(root, "configs/FROZEN_SHA256")
    lines, pending = [], []
    for h, entry, raw in parse(frozen):
        if not entry:
            lines.append(raw)
            continue
        try:
            lines.append(f"{entry_hash(root, entry)}  {entry}")
        except (FileNotFoundError, KeyError):
            lines.append(f"PENDING  {entry}")
            pending.append(entry)
    frozen.write_text("\n".join(lines) + "\n")
    return pending


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rewrite hashes (PI only, before tagging frozen-*)")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args(argv)
    if a.update:
        pending = update(a.root)
        print(f"updated; {len(pending)} entries still PENDING (file missing): {pending}")
        return 0
    probs = check(a.root)
    for p in probs:
        print(p)
    print("FROZEN OK" if not probs else f"FROZEN CHECK FAILED ({len(probs)} problems)")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
