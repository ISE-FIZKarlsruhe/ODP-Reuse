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

The monthly harvest publishes MINOR bumps (publish-release.yml). A gate change
is always MAJOR and is made by a human.

STEPS
-----
A new version starts as a copy of the previous one. Its metadata is KEPT
(communities, related identifiers and everything else zenodo.json does not
name); only the fields zenodo.json names, plus version, publication date and
notes, are replaced. Its inherited files are cleared. The unattended workflow
runs the steps one at a time, so the version's DOI is known (the draft reserves
it) before anything is public, and the draft is discarded if a later step fails:

    open      new-version draft, metadata set, DOI reserved  (--draft-json)
    upload    files into the draft's bucket
    publish   irreversible: the version and its DOI go live
    discard   delete an unpublished draft
    all       open + upload (+ publish with --publish); the default

Usage
-----
    export ZENODO_TOKEN=...
    python zenodo_publish.py --concept 17751740 --version v1.3.0 \\
        --files corpus.zip corpus-hashed.zip --metadata release-iri/zenodo.json

    python zenodo_publish.py --step open --concept 17751740 --version v1.3.0 \\
        --metadata release-iri/zenodo.json --notes-file notes.txt --draft-json draft.json
    python zenodo_publish.py --step upload  --draft-id 99 --files corpus.zip ...
    python zenodo_publish.py --step publish --draft-id 99

    python zenodo_publish.py ... --dry-run     # validate without depositing
"""

from __future__ import annotations

import argparse
import datetime as dt
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
    """Resolve a concept id to the id of its most recent version.

    Zenodo answers /records/<concept> with the newest version (a redirect,
    which requests follows), so that record's ``id`` is the answer.
    ``links.latest`` is not: it ends in ``/versions/latest``.
    """
    r = requests.get(api(base, f"/records/{concept}"), timeout=60)
    r.raise_for_status()
    return int(r.json()["id"])


def merged_metadata(inherited: dict, overrides: dict, version: str,
                    notes: str = "", today: str = "") -> dict:
    """The new version's metadata: the previous version's, with ``overrides``
    and the version fields replaced. The previous version's DOI is dropped;
    the draft reserves its own."""
    meta = {k: v for k, v in inherited.items() if k not in ("doi", "prereserve_doi")}
    meta.update(overrides)
    meta["version"] = version.lstrip("v")
    meta["publication_date"] = today or dt.date.today().isoformat()
    if notes:
        meta["notes"] = notes
    meta.setdefault("upload_type", "dataset")
    meta.setdefault("access_right", "open")
    meta.setdefault("license", "cc-by-4.0")
    meta["prereserve_doi"] = True
    return meta


def open_draft(base: str, token: str, concept: int, overrides: dict, version: str,
               notes: str = "", today: str = "") -> dict:
    """Open a new-version draft with its metadata set and no files."""
    tok = {"access_token": token}
    latest = latest_version_id(base, token, concept)
    r = requests.post(api(base, f"/deposit/depositions/{latest}/actions/newversion"),
                      params=tok, timeout=120)
    r.raise_for_status()
    draft_id = int(r.json()["links"]["latest_draft"].rstrip("/").rsplit("/", 1)[-1])

    r = requests.get(api(base, f"/deposit/depositions/{draft_id}"), params=tok, timeout=60)
    r.raise_for_status()
    draft = r.json()
    for f in draft.get("files", []):
        requests.delete(api(base, f"/deposit/depositions/{draft_id}/files/{f['id']}"),
                        params=tok, timeout=120).raise_for_status()

    meta = merged_metadata(draft.get("metadata", {}), overrides, version, notes, today)
    r = requests.put(api(base, f"/deposit/depositions/{draft_id}"), params=tok,
                     json={"metadata": meta}, timeout=60)
    r.raise_for_status()
    reserved = r.json().get("metadata", {}).get("prereserve_doi") or {}
    doi = reserved.get("doi") if isinstance(reserved, dict) else None
    return {"draft_id": draft_id,
            # Zenodo names a version's DOI after its record id.
            "doi": doi or f"10.5281/zenodo.{draft_id}",
            "url": draft.get("links", {}).get("html", f"{base}/deposit/{draft_id}"),
            "concept": concept}


def upload(base: str, token: str, draft_id: int, files: list[Path]) -> None:
    """Upload through the bucket API (no 100 MB form limit)."""
    tok = {"access_token": token}
    r = requests.get(api(base, f"/deposit/depositions/{draft_id}"), params=tok, timeout=60)
    r.raise_for_status()
    bucket = r.json()["links"]["bucket"]
    for f in files:
        print(f"uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB) ...", flush=True)
        with f.open("rb") as fh:
            requests.put(f"{bucket}/{f.name}", data=fh, params=tok,
                         timeout=None).raise_for_status()


def publish(base: str, token: str, draft_id: int) -> str:
    """Publish the draft -- irreversible. Returns the version DOI."""
    r = requests.post(api(base, f"/deposit/depositions/{draft_id}/actions/publish"),
                      params={"access_token": token}, timeout=300)
    r.raise_for_status()
    return r.json()["doi"]


def discard(base: str, token: str, draft_id: int) -> None:
    """Delete an unpublished draft."""
    requests.delete(api(base, f"/deposit/depositions/{draft_id}"),
                    params={"access_token": token}, timeout=120).raise_for_status()


NEEDS = {"all": ("concept", "version", "files"), "open": ("concept", "version"),
         "upload": ("draft_id", "files"), "publish": ("draft_id",),
         "discard": ("draft_id",)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default="all", choices=tuple(NEEDS))
    ap.add_argument("--concept", type=int, help="concept record id, e.g. 17751740")
    ap.add_argument("--version", help="vMAJOR.MINOR.PATCH")
    ap.add_argument("--files", nargs="+", type=Path, default=[])
    ap.add_argument("--metadata", type=Path,
                    help="zenodo.json with title/creators/description/keywords")
    ap.add_argument("--notes-file", type=Path, help="version notes (Zenodo 'notes')")
    ap.add_argument("--draft-id", type=int, help="the draft, for upload/publish/discard")
    ap.add_argument("--draft-json", type=Path,
                    help="open/all: write {draft_id, doi, url, concept} here")
    ap.add_argument("--sandbox", action="store_true",
                    help="deposit to sandbox.zenodo.org (test first!)")
    ap.add_argument("--publish", action="store_true",
                    help="step all: publish immediately; otherwise leave a draft")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    base = SANDBOX if args.sandbox else PROD
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token and not args.dry_run:
        sys.exit("ZENODO_TOKEN is not set")
    missing = [f"--{n.replace('_', '-')}" for n in NEEDS[args.step] if not getattr(args, n)]
    if missing:
        sys.exit(f"--step {args.step} needs {' '.join(missing)}")
    absent = [f for f in args.files if not f.is_file()]
    if absent:
        sys.exit(f"missing files: {absent}")
    for f in args.files:
        print(f"   {f.name:44s} {f.stat().st_size / 1e6:9.1f} MB")

    overrides = (json.loads(args.metadata.read_text(encoding="utf-8"))
                 if args.metadata and args.metadata.is_file() else {})
    notes = (args.notes_file.read_text(encoding="utf-8").strip()
             if args.notes_file and args.notes_file.is_file() else "")

    if args.dry_run:
        print("\n[dry run] metadata that would be sent:")
        print(json.dumps(merged_metadata({}, overrides, args.version or "v0", notes),
                         indent=2, ensure_ascii=False)[:1200])
        return

    if args.step in ("all", "open"):
        draft = open_draft(base, token, args.concept, overrides, args.version, notes)
        print(f"draft {draft['draft_id']} opened, DOI reserved: {draft['doi']}")
        if args.draft_json:
            args.draft_json.write_text(json.dumps(draft, indent=1) + "\n", encoding="utf-8")
        args.draft_id = draft["draft_id"]
    if args.step in ("all", "upload"):
        upload(base, token, args.draft_id, args.files)
    if args.step == "publish" or (args.step == "all" and args.publish):
        print(f"\nPUBLISHED  version DOI: {publish(base, token, args.draft_id)}")
    elif args.step == "discard":
        discard(base, token, args.draft_id)
        print(f"draft {args.draft_id} discarded")
    elif args.step == "all":
        print(f"\nDRAFT ready for review: {base}/deposit/{args.draft_id}")
        print("publish it with --step publish --draft-id, or from the web UI")


if __name__ == "__main__":
    main()
