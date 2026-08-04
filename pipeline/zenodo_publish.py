#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zenodo_publish.py -- publish a new VERSION of an existing Zenodo record,
preserving its concept DOI.

WHY NOT THE GITHUB INTEGRATION
------------------------------
Zenodo's one-click GitHub integration mints a NEW concept DOI the first time it
sees a release. Record 17751741 was deposited manually, so letting the
integration take over would strand v1 under a different concept DOI and split the
citation record in two. The REST API can add versions to the existing concept
instead, which is what this script does:

    POST /api/deposit/depositions/{id}/actions/newversion
    -> new draft, same concept DOI, fresh version DOI

The concept DOI (`10.5281/zenodo.<concept>`) always resolves to the newest
version and is what the paper should cite. Each version also gets its own DOI for
exact reproducibility.

VERSIONING POLICY (see --version)
---------------------------------
    vMAJOR.MINOR.PATCH

    MAJOR  the gate changed in a way that moves ontologies between kept and
           dropped, or a dataset view changed definition. Results computed on
           the previous MAJOR are not comparable.
    MINOR  new ontologies or pairs from a harvest; gate unchanged. Previous
           results remain valid on their subset.
    PATCH  corrections that do not change membership -- repaired files,
           metadata, documentation.

The quarterly harvest proposes MINOR bumps. A gate change is always MAJOR and is
made by a human.

Usage
-----
    export ZENODO_TOKEN=...
    python zenodo_publish.py --concept 17751740 --version v1.1.0 \\
        --files release.zip pipeline-src.zip --metadata release/zenodo.json

    python zenodo_publish.py ... --dry-run     # validate without depositing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("pip install requests")

SANDBOX = "https://sandbox.zenodo.org"
PROD = "https://zenodo.org"


def api(base: str, path: str) -> str:
    return f"{base}/api{path}"


def latest_version_id(base: str, token: str, concept: int) -> int:
    """Resolve a concept id to the id of its most recent version."""
    r = requests.get(api(base, f"/records/{concept}"),
                     params={"access_token": token}, timeout=60)
    r.raise_for_status()
    d = r.json()
    latest = d.get("links", {}).get("latest", "")
    if latest:
        return int(latest.rstrip("/").rsplit("/", 1)[-1])
    return int(d["id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, type=int,
                    help="concept record id, e.g. 17751740")
    ap.add_argument("--version", required=True, help="vMAJOR.MINOR.PATCH")
    ap.add_argument("--files", nargs="+", required=True, type=Path)
    ap.add_argument("--metadata", type=Path,
                    help="zenodo.json with title/creators/description/keywords")
    ap.add_argument("--sandbox", action="store_true",
                    help="deposit to sandbox.zenodo.org (test first!)")
    ap.add_argument("--publish", action="store_true",
                    help="publish immediately; otherwise leave an editable draft")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = SANDBOX if args.sandbox else PROD
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token and not args.dry_run:
        sys.exit("ZENODO_TOKEN is not set")

    missing = [f for f in args.files if not f.is_file()]
    if missing:
        sys.exit(f"missing files: {missing}")
    total = sum(f.stat().st_size for f in args.files)
    print(f"{len(args.files)} file(s), {total / 1e6:.1f} MB, version {args.version}")
    for f in args.files:
        print(f"   {f.name:44s} {f.stat().st_size / 1e6:9.1f} MB")

    meta = {}
    if args.metadata and args.metadata.is_file():
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    meta["version"] = args.version.lstrip("v")
    meta.setdefault("upload_type", "dataset")
    meta.setdefault("access_right", "open")
    meta.setdefault("license", "cc-by-4.0")

    if args.dry_run:
        print("\n[dry run] metadata that would be sent:")
        print(json.dumps(meta, indent=2, ensure_ascii=False)[:1200])
        return

    # 1. newest version of the concept
    latest = latest_version_id(base, token, args.concept)
    print(f"latest version of concept {args.concept}: record {latest}")

    # 2. open a new-version draft (same concept DOI)
    r = requests.post(
        api(base, f"/deposit/depositions/{latest}/actions/newversion"),
        params={"access_token": token}, timeout=120)
    r.raise_for_status()
    draft_url = r.json()["links"]["latest_draft"]
    draft_id = int(draft_url.rstrip("/").rsplit("/", 1)[-1])
    print(f"draft created: {draft_id}")

    # 3. a new version starts as a copy -- clear inherited files
    r = requests.get(api(base, f"/deposit/depositions/{draft_id}"),
                     params={"access_token": token}, timeout=60)
    r.raise_for_status()
    draft = r.json()
    for f in draft.get("files", []):
        requests.delete(
            api(base, f"/deposit/depositions/{draft_id}/files/{f['id']}"),
            params={"access_token": token}, timeout=120).raise_for_status()
    print(f"cleared {len(draft.get('files', []))} inherited file(s)")

    # 4. upload via the bucket API (no 100 MB form limit)
    bucket = draft["links"]["bucket"]
    for f in args.files:
        print(f"uploading {f.name} ...", flush=True)
        with f.open("rb") as fh:
            requests.put(f"{bucket}/{f.name}", data=fh,
                         params={"access_token": token},
                         timeout=None).raise_for_status()

    # 5. metadata
    requests.put(api(base, f"/deposit/depositions/{draft_id}"),
                 params={"access_token": token},
                 json={"metadata": meta}, timeout=60).raise_for_status()
    print("metadata set")

    if args.publish:
        r = requests.post(
            api(base, f"/deposit/depositions/{draft_id}/actions/publish"),
            params={"access_token": token}, timeout=300)
        r.raise_for_status()
        d = r.json()
        print(f"\nPUBLISHED  version DOI: {d['doi']}")
        print(f"           concept DOI: {d.get('conceptdoi', '(unchanged)')}")
    else:
        print(f"\nDRAFT ready for review: {base}/deposit/{draft_id}")
        print("re-run with --publish, or publish from the web UI")


if __name__ == "__main__":
    main()
