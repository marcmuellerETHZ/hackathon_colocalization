#!/usr/bin/env bash
# Shared, resumable, verified download helper for the real-data arms (owner: real-data engineer).
# Source this file; then call:  fetch <url> <dest> [md5]   (md5 as hex or GCS base64; empty = none upstream)
# Behaviour:
#   * resumable: downloads to <dest>.part with `curl -C -`, renamed only after verification;
#   * skips a file that already exists with the server's Content-Length and a recorded sha256;
#   * verifies: byte size == Content-Length; md5 == upstream md5 when given; full gzip/BGZF
#     decompression test for *.gz/*.bgz; then appends "sha256  file" to <dir>/SHA256SUMS.
#   * every failure is fatal (set -euo pipefail; no stderr suppression).
set -euo pipefail

remote_size() {  # prints Content-Length of the final response
  curl -fsSIL --retry 5 --retry-delay 5 "$1" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}'
}

md5_to_hex() {   # accepts 32-hex or base64 (GCS md5Hash)
  local m=$1
  if [[ ${#m} -eq 32 ]]; then echo "$m"; else echo "$m" | base64 -d | od -An -tx1 | tr -d ' \n'; echo; fi
}

record_sha256() {  # record_sha256 <file>; idempotent
  local f=$1 d b s
  d=$(dirname "$f"); b=$(basename "$f")
  s=$(sha256sum "$f" | cut -d' ' -f1)
  touch "$d/SHA256SUMS"
  if grep -q "  $b\$" "$d/SHA256SUMS"; then
    if ! grep -q "^$s  $b\$" "$d/SHA256SUMS"; then echo "ERROR sha256 changed for $f" >&2; return 1; fi
  else
    echo "$s  $b" >> "$d/SHA256SUMS"
  fi
}

fetch() {
  local url=$1 dest=$2 md5=${3:-}
  mkdir -p "$(dirname "$dest")"
  local rs; rs=$(remote_size "$url")
  if [[ -z "$rs" ]]; then echo "ERROR no Content-Length for $url" >&2; return 1; fi
  local b; b=$(basename "$dest")
  if [[ -f "$dest" ]] && [[ $(stat -c %s "$dest") == "$rs" ]] && grep -q "  $b\$" "$(dirname "$dest")/SHA256SUMS" 2>/dev/null; then
    echo "[skip] $b (verified earlier)"; return 0
  fi
  if [[ -f "$dest" ]]; then mv "$dest" "$dest.part"; fi   # unverified complete file: re-verify via .part path
  local ps=0; [[ -f "$dest.part" ]] && ps=$(stat -c %s "$dest.part")
  if (( ps > rs )); then echo "WARN $b.part larger than remote; restarting" >&2; rm -f "$dest.part"; ps=0; fi
  if (( ps < rs )); then
    echo "[get ] $b ($((rs/1000000)) MB, resume at $((ps/1000000)) MB)"
    curl -fsS --retry 10 --retry-delay 15 --retry-all-errors -C - -o "$dest.part" "$url"
  fi
  local ls; ls=$(stat -c %s "$dest.part")
  if [[ "$ls" != "$rs" ]]; then echo "ERROR size mismatch $b: local $ls remote $rs" >&2; return 1; fi
  if [[ -n "$md5" ]]; then
    local want got; want=$(md5_to_hex "$md5"); got=$(md5sum "$dest.part" | cut -d' ' -f1)
    if [[ "$want" != "$got" ]]; then echo "ERROR md5 mismatch $b: want $want got $got" >&2; return 1; fi
    echo "[md5 ] $b ok"
  fi
  case "$b" in
    *.gz|*.bgz) gzip -t "$dest.part"; echo "[gzip] $b ok";;
  esac
  mv "$dest.part" "$dest"
  record_sha256 "$dest"
  echo "[done] $b"
}
