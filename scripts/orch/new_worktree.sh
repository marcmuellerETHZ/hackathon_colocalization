#!/bin/bash
# Create an agent worktree + branch inside the project and link the shared heavy dirs.
#   scripts/orch/new_worktree.sh <name> [base-branch=master]
# Result: .worktrees/<name> on branch <name>, with symlinks data -> <main>/data and
# .venv -> <main>/.venv (both git-ignored via .git/info/exclude, never tracked).
# env/activate.sh uses absolute paths, so env/ needs no links.
set -euo pipefail
name=${1:?usage: new_worktree.sh <name> [base-branch]}
base=${2:-master}
MAIN=$(git rev-parse --path-format=absolute --git-common-dir)
MAIN=${MAIN%/.git}
wt="$MAIN/.worktrees/$name"
[ -e "$wt" ] && { echo "exists: $wt" >&2; exit 1; }
git -C "$MAIN" worktree add "$wt" -b "$name" "$base"
ex="$MAIN/.git/info/exclude"
mkdir -p "$(dirname "$ex")"
for pat in /data /.venv /runs/ /.orch/; do
  grep -qxF "$pat" "$ex" 2>/dev/null || echo "$pat" >> "$ex"
done
[ -e "$wt/data" ] || ln -s "$MAIN/data" "$wt/data"
[ -e "$wt/.venv" ] || ln -s "$MAIN/.venv" "$wt/.venv"
echo "worktree $wt (branch $name from $base); data -> $MAIN/data"
git -C "$wt" status --short | head -5
