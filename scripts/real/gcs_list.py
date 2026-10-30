"""List objects (name, size, md5Hash) of a public GCS bucket prefix via the JSON API (stdlib only).
Usage: python -I gcs_list.py <bucket> <prefix>   -> TSV on stdout."""
import json
import sys
import urllib.parse
import urllib.request


def main(bucket: str, prefix: str) -> None:
    tok = None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": "1000"}
        if tok:
            q["pageToken"] = tok
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        for it in d.get("items", []):
            print(it["name"], it["size"], it.get("md5Hash", ""), sep="\t")
        tok = d.get("nextPageToken")
        if not tok:
            break


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
