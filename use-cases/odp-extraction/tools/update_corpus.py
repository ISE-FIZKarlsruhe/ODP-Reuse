#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_corpus.py -- replace corpus files with newer upstream releases, safely.

A corpus that will be published must not be updated on trust. Every replacement
here has to clear four checks before it touches the working copy, and every
replacement that happens is recorded with what changed:

  1. the download must succeed and not be an HTML page, a Git LFS pointer or an
     error body;
  2. it must PARSE as RDF;
  3. it must declare at least one entity under the same WIDE definition the rest
     of the pipeline uses -- a file that parses but is empty is a regression;
  4. it must not lose more than ``--max-shrink`` of the existing entity count
     unless ``--allow-shrink`` is given. An ontology that halves overnight is
     more likely a wrong URL than a real release, and this catches the case where
     a repository's default-branch file is a stub while the release is a tag.

Anything failing a check is REJECTED, the original file is left untouched, and
the reason is written to the report. Nothing is silent.

The original corpus is copied to ``<corpus>.bak-<timestamp>`` before the first
replacement, so a bad batch can be reverted wholesale.

Usage
-----
    # see what would happen -- this is the default
    python tools/update_corpus.py --corpus corpora/mseo \\
        --updates stats/mseo/updates.json --out stats/mseo

    # actually do it
    python tools/update_corpus.py ... --apply

``updates.json`` is a list of objects with at least ``file`` and
``download_url``; ``latest_version``, ``local_version``, ``upstream_repo`` and
``release_date`` are recorded when present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))

from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402

UA = "Mozilla/5.0 (compatible; odp-reuse-corpus-updater/1.0)"
HTML_MARKERS = (b"<!doctype html", b"<html", b"<HTML")
LFS_MARKER = b"version https://git-lfs"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(p: Path) -> dict:
    """Version, triple and entity counts of a local RDF file."""
    from rdflib import OWL, RDF, URIRef
    g, err = parse_graph(p)
    if g is None:
        return {"parsed": False, "error": err[:150], "n_triples": 0,
                "n_entities": 0, "version": ""}
    gs, _ = strip_annotations(g)
    ents = collect_entities(gs, wide=True)
    onts = [s for s in g.subjects(RDF.type, OWL.Ontology) if isinstance(s, URIRef)]
    ver = ""
    if onts:
        vs = ([str(o) for o in g.objects(onts[0], OWL.versionInfo)]
              + [str(o) for o in g.objects(onts[0], OWL.versionIRI)])
        ver = vs[0] if vs else ""
    return {"parsed": True, "error": "", "n_triples": len(g),
            "n_entities": len(ents), "version": ver[:120],
            "ontology_iri": str(onts[0]) if onts else ""}


def download(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        return False, f"download failed: {type(exc).__name__}: {exc}"
    if not data:
        return False, "empty response"
    head = data[:512].lstrip()
    if any(head.lower().startswith(m.lower()) for m in HTML_MARKERS) \
            or b"<html" in head[:200].lower():
        return False, "response is an HTML page, not RDF"
    if data.startswith(LFS_MARKER):
        return False, "response is a Git LFS pointer, not the file"
    dest.write_bytes(data)
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--updates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="actually replace files; without it nothing is written")
    ap.add_argument("--max-shrink", type=float, default=0.5,
                    help="reject an update whose entity count falls below this "
                         "fraction of the current one (default 0.5)")
    ap.add_argument("--allow-shrink", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    args.corpus = args.corpus.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    updates = json.loads(args.updates.read_text(encoding="utf-8"))
    if isinstance(updates, dict):
        updates = updates.get("updates") or updates.get("results") or []
    updates = [u for u in updates if u.get("download_url")]
    print(f"{len(updates)} candidate updates; apply={args.apply}", flush=True)

    tmp = args.out / "_download"
    tmp.mkdir(exist_ok=True)
    backup = None
    rows: list[dict] = []

    for i, u in enumerate(updates, 1):
        target = args.corpus / u["file"]
        row = {"file": u["file"], "upstream_repo": u.get("upstream_repo", ""),
               "release_date": u.get("release_date", ""),
               "claimed_version": u.get("latest_version", ""),
               "download_url": u["download_url"], "action": "", "reason": ""}
        if not target.is_file():
            row["action"] = "REJECTED"
            row["reason"] = "no such file in the corpus"
            rows.append(row)
            continue

        before = describe(target)
        row.update({f"old_{k}": v for k, v in before.items()
                    if k in ("version", "n_triples", "n_entities")})
        row["old_sha256"] = sha256(target)[:16]

        dl = tmp / (u["file"].replace("/", "__"))
        ok, err = download(u["download_url"], dl, args.timeout)
        if not ok:
            row["action"], row["reason"] = "REJECTED", err
            rows.append(row)
            print(f"  [{i}/{len(updates)}] REJECT {u['file']}: {err}", flush=True)
            continue

        after = describe(dl)
        row.update({f"new_{k}": v for k, v in after.items()
                    if k in ("version", "n_triples", "n_entities")})
        row["new_sha256"] = sha256(dl)[:16]

        if not after["parsed"]:
            row["action"] = "REJECTED"
            row["reason"] = f"downloaded file does not parse: {after['error']}"
        elif after["n_entities"] == 0:
            row["action"] = "REJECTED"
            row["reason"] = "downloaded file declares no entities"
        elif (before["n_entities"] and not args.allow_shrink
              and after["n_entities"] < args.max_shrink * before["n_entities"]):
            row["action"] = "REJECTED"
            row["reason"] = (f"entity count would fall from {before['n_entities']} "
                             f"to {after['n_entities']}; likely the wrong artefact")
        elif row["old_sha256"] == row["new_sha256"]:
            row["action"] = "SKIPPED"
            row["reason"] = "byte-identical to the local copy; already current"
        else:
            row["action"] = "UPDATED" if args.apply else "WOULD_UPDATE"
            if args.apply:
                if backup is None:
                    backup = args.corpus.with_name(
                        args.corpus.name + f".bak-{time.strftime('%Y%m%dT%H%M%S')}")
                    shutil.copytree(args.corpus, backup)
                    print(f"  [backup] {backup}", flush=True)
                shutil.copyfile(dl, target)

        rows.append(row)
        print(f"  [{i}/{len(updates)}] {row['action']:12s} {u['file']:28s} "
              f"{before.get('version','')[:20]!r} -> {after.get('version','')[:20]!r} "
              f"ents {before['n_entities']}->{after['n_entities']} {row['reason']}",
              flush=True)

    fields = ["file", "action", "reason", "old_version", "new_version",
              "claimed_version", "old_n_entities", "new_n_entities",
              "old_n_triples", "new_n_triples", "old_sha256", "new_sha256",
              "upstream_repo", "release_date", "download_url"]
    rp = args.out / "corpus_updates.csv"
    with rp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    tally = Counter(r["action"] for r in rows)
    print(f"\n[OK] {rp}")
    for k, v in tally.most_common():
        print(f"  {k}: {v}")
    if backup:
        print(f"  original corpus preserved at {backup}")
    if not args.apply and tally.get("WOULD_UPDATE"):
        print("\nNothing was written. Re-run with --apply to perform the update.")


if __name__ == "__main__":
    main()
