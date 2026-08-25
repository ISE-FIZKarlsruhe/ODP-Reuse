#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mse_release.py -- assemble the versioned, Zenodo-ready MSE release.

Mirrors the layout and the conventions of ``release-iri/`` so that a reader who
knows one knows the other: a corpus, a manifest with one row per artefact, the
statistics, the raw per-ontology outputs, a validator, a README and a
``zenodo.json``.

What makes this release different from ``release-iri`` is that it has **no
ground truth**, and the layout says so rather than leaving a reader to discover
it. There is no gate, no views, no pairs file. There are predicted groupings,
the ground-truth-free statistics that rank them, and the provenance of every
ontology file including which ones were refreshed from upstream and when.

LAYOUT
------
    mse-release/
      README.md                  what this is, and what it is not
      zenodo.json                deposition metadata
      VERSION                    the release version and its date
      CHANGELOG.md               what changed since the previous release
      MANIFEST.csv               one row per ontology: IRI, version, upstream,
                                 counts, checksum, and whether it parsed
      corpus/                    the ontology files, as evaluated
      provenance/
        corpus_inventory.csv     version metadata read from every file
        corpus_updates.csv       every upstream refresh: old -> new, with URL
        upstream_check.json      the full currency check, including UNKNOWNs
      predictions/               one JSONL per method run, every predicted group
      statistics/
        mse_stats.json           yield, agreement, recurrence, anchoring, ODP contact
        yield.csv, agreement.csv
      skipped.csv                every ontology excluded, with the reason
      validate_release.py        internal-consistency check for an unpacked copy

Usage
-----
    python tools/build_mse_release.py --version 1.0.0 \\
        --corpus corpora/mseo --results results --stats stats/mseo \\
        --out ../../mse-release
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_REPO / "pipeline"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_texts import README_TEMPLATE, ZENODO_TEMPLATE  # noqa: E402
import release_views as rv  # noqa: E402

PREDICTION_SOURCES = {
    "structural_and_published": "mseo/predictions.jsonl",
    "embeddings": "mseo-embed/predictions.jsonl",
    "conceptual_components_full": "mseo-cc-full/predictions.jsonl",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--previous", type=Path, default=None,
                    help="a previous release directory, to diff for the CHANGELOG")
    args = ap.parse_args()

    for a in ("corpus", "results", "stats", "out"):
        setattr(args, a, getattr(args, a).resolve())
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for d in ("corpus", "provenance", "predictions", "statistics"):
        (out / d).mkdir(exist_ok=True)

    # ---- corpus + manifest -------------------------------------------
    inv_src = args.stats / "corpus_inventory.csv"
    inv = {r["file"]: r for r in csv.DictReader(inv_src.open(encoding="utf-8"))} \
        if inv_src.is_file() else {}
    upd_src = args.stats / "corpus_updates.csv"
    upd = {r["file"]: r for r in csv.DictReader(upd_src.open(encoding="utf-8"))} \
        if upd_src.is_file() else {}

    rows = []
    n_files = 0
    for f in sorted(args.corpus.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (
                ".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt", ".jsonld"):
            continue
        rel = f.relative_to(args.corpus).as_posix()
        dest = out / "corpus" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        n_files += 1
        i = inv.get(rel, {})
        u = upd.get(rel, {})
        rows.append({
            "file": rel,
            "bytes": f.stat().st_size,
            "sha256": sha256(f),
            "parsed": i.get("parsed", ""),
            "parse_error": i.get("error", ""),
            "ontology_iri": i.get("ontology_iri", ""),
            "title": i.get("title", ""),
            "version": (u.get("new_version") or i.get("versionInfo")
                        or i.get("versionIRI") or ""),
            "n_triples": i.get("n_triples", ""),
            "imports": i.get("imports", ""),
            "refreshed_from_upstream": "yes" if u.get("action") == "UPDATED" else "no",
            "upstream_repo": u.get("upstream_repo", ""),
            "upstream_release_date": u.get("release_date", ""),
            "previous_version": u.get("old_version", "") if u.get("action") == "UPDATED" else "",
        })
    mf = out / "MANIFEST.csv"
    with mf.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[corpus] {n_files} files -> {out / 'corpus'}")

    # ---- provenance ---------------------------------------------------
    for src, name in ((inv_src, "corpus_inventory.csv"),
                      (upd_src, "corpus_updates.csv"),
                      (args.stats / "upstream_check.json", "upstream_check.json")):
        if src.is_file():
            shutil.copyfile(src, out / "provenance" / name)

    # ---- predictions --------------------------------------------------
    pred_counts = {}
    for name, rel in PREDICTION_SOURCES.items():
        src = args.results / rel
        if not src.is_file():
            print(f"[predictions] MISSING {rel} -- not included")
            continue
        shutil.copyfile(src, out / "predictions" / f"{name}.jsonl")
        pred_counts[name] = sum(1 for _ in src.open(encoding="utf-8"))
    print(f"[predictions] {pred_counts}")

    # ---- statistics ---------------------------------------------------
    for p in sorted(args.stats.glob("*")):
        if p.is_file() and p.name in ("mse_stats.json", "yield.csv",
                                      "agreement.csv"):
            shutil.copyfile(p, out / "statistics" / p.name)

    # ---- skips --------------------------------------------------------
    skips = []
    for rel in ("mseo/methods_skipped.csv", "mseo-embed/rq5_skipped.csv",
                "mseo-cc-full/asprino_full_skipped.csv"):
        p = args.results / rel
        if p.is_file():
            for r in csv.DictReader(p.open(encoding="utf-8")):
                r["run"] = rel.split("/")[0]
                skips.append(r)
    if skips:
        keys = ["run"] + [k for k in skips[0] if k != "run"]
        with (out / "skipped.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(skips)
    print(f"[skips] {len(skips)} rows")

    # ---- version + changelog -------------------------------------------
    (out / "VERSION").write_text(f"{args.version}\n{date.today().isoformat()}\n",
                                 encoding="utf-8")
    n_refreshed = sum(1 for r in rows if r["refreshed_from_upstream"] == "yes")
    changed = [r for r in rows if r["refreshed_from_upstream"] == "yes"]
    cl = [f"# Changelog", "",
          f"## {args.version} — {date.today().isoformat()}", ""]
    if n_refreshed:
        cl.append(f"{n_refreshed} ontologies refreshed from upstream:")
        cl.append("")
        cl.append("| ontology | previous | now | upstream |")
        cl.append("|---|---|---|---|")
        for r in changed:
            def _v(x):
                x = (x or "").strip()
                if not x:
                    return "(none declared)"
                return (x[:48] + "...") if len(x) > 48 else x
            cl.append(f"| `{r['file']}` | {_v(r['previous_version'])} "
                      f"| {_v(r['version'])} | {r['upstream_repo']} |")
    else:
        cl.append("No ontology was refreshed from upstream in this release.")
    cl += ["", f"Corpus: {n_files} files.",
           f"Predictions: {', '.join(f'{k} ({v} rows)' for k, v in pred_counts.items()) or 'none'}.",
           "", "Every ontology's provenance, including the upstream check that "
           "found no newer release, is in `provenance/`."]
    (out / "CHANGELOG.md").write_text("\n".join(cl) + "\n", encoding="utf-8")

    # ---- validator -----------------------------------------------------
    (out / "validate_release.py").write_text(VALIDATOR, encoding="utf-8")

    # ---- zenodo --------------------------------------------------------
    zen = {
        "title": f"MSE-ODP: extracted ontology design pattern candidates from "
                 f"materials science and engineering ontologies (v{args.version})",
        "upload_type": "dataset",
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": args.version,
        "description": (
            "Candidate ontology design patterns extracted automatically from a "
            f"corpus of {n_files} materials science and engineering (MSE) "
            "ontologies, together with the corpus as evaluated, the predicted "
            "groupings of every method, and the ground-truth-free statistics "
            "used to rank them.\n\n"
            "THIS RELEASE HAS NO GROUND TRUTH. No ontology here is annotated "
            "with the patterns it reuses, so no accuracy is reported and none "
            "should be inferred. What is reported is what each method proposed, "
            "how much the methods agree with one another, how often a grouping "
            "recurs across ontologies, whether it is anchored in a foundational "
            "vocabulary, and whether it overlaps a documented pattern. Those are "
            "ranking signals for expert review, not scores.\n\n"
            "The corpus is the collection assembled for the survey 'The "
            "landscape of ontologies in materials science and engineering' "
            "(arXiv:2408.06034), refreshed against each artefact's upstream "
            "release. provenance/ records, for every file, the version held, "
            "whether it was refreshed and from where, and the result of the "
            "upstream check even where no newer release was found.\n\n"
            "Companion release: ODP-Reuse, a quality-gated benchmark of verified "
            "ODP reuse, which is where the same methods are scored against "
            "ground truth."),
        "keywords": ["ontology design patterns", "materials science",
                     "ontology extraction", "MSE", "EMMO", "BFO", "PMDco",
                     "semantic web", "OWL", "reproducibility"],
        "creators": [
            {"name": "Norouzi, Ebrahim",
             "affiliation": "FIZ Karlsruhe -- Leibniz Institute for Information Infrastructure",
             "orcid": "0000-0002-2691-6995"},
            {"name": "Hertling, Sven",
             "affiliation": "FIZ Karlsruhe / University of Mannheim",
             "orcid": "0000-0003-0333-5888"},
            {"name": "Waitelonis, Jörg", "affiliation": "FIZ Karlsruhe",
             "orcid": "0000-0001-7192-7143"},
            {"name": "Sack, Harald",
             "affiliation": "FIZ Karlsruhe / Karlsruhe Institute of Technology",
             "orcid": "0000-0001-7069-9804"},
        ],
        "related_identifiers": [
            {"identifier": "arXiv:2408.06034", "relation": "isDerivedFrom",
             "scheme": "arxiv"},
        ],
        "notes": "The ontology files are redistributed under their own licences; "
                 "MANIFEST.csv names the upstream of each. The CC-BY-4.0 licence "
                 "above applies to the predictions, statistics and provenance "
                 "produced here.",
    }
    (out / "zenodo.json").write_text(json.dumps(zen, indent=2), encoding="utf-8")

    # ---- summary, statistics, views and audit ---------------------------
    # Mirrors what release-iri gives a reader beyond the raw corpus: a
    # statistics table, a summary of how the population was reduced, per-view
    # data, and a document stating each rule. The MSE analogues are per method
    # rather than per gated view, because there is no gate here.
    stats_json = args.stats / "mse_stats.json"
    stats = json.loads(stats_json.read_text(encoding="utf-8"))         if stats_json.is_file() else {}
    upstream_json = args.stats / "upstream_check.json"
    upstream = json.loads(upstream_json.read_text(encoding="utf-8"))         if upstream_json.is_file() else None

    (out / "views").mkdir(exist_ok=True)
    if stats:
        rv.write_dataset_statistics(out, stats)
        rv.write_agreement_matrix(out, stats)
        rv.write_recurrence_and_shortlist(out, stats)
    view_counts = rv.write_method_views(out, out / "predictions")
    rv.write_corpus_summary(out, rows, skips, upstream)
    rv.write_audit(out, skips)
    (out / "SIGNALS.md").write_text(rv.SIGNALS_MD, encoding="utf-8")
    print(f"[views] {len(view_counts)} methods, "
          f"{sum(view_counts.values()):,} groups flattened")

    # ---- README and the Zenodo description ------------------------------
    n_parsed = sum(1 for r in rows if r["parsed"] == "ok")
    fmt = dict(version=args.version, date=date.today().isoformat(),
               n_files=n_files, n_parsed=n_parsed, n_refreshed=n_refreshed,
               n_skipped=len(skips))
    (out / "README.md").write_text(README_TEMPLATE.format(**fmt),
                                   encoding="utf-8")
    (out / "ZENODO.md").write_text(ZENODO_TEMPLATE.format(**fmt),
                                   encoding="utf-8")

    print(f"\n[OK] {out}")
    print(f"  version {args.version}, {n_files} files ({n_parsed} parse), "
          f"{n_refreshed} refreshed from upstream")
    print("  README.md and ZENODO.md written")


VALIDATOR = '''#!/usr/bin/env python3
"""validate_release.py -- internal consistency of an unpacked MSE release.

Checks that MANIFEST.csv agrees with corpus/, that every checksum matches, that
every prediction refers to an ontology in the manifest, and that no file claims
a score. Exits non-zero on any failure.
"""
import csv, hashlib, json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
fail = []

man = {r["file"]: r for r in csv.DictReader((root / "MANIFEST.csv").open(encoding="utf-8"))}
files = {p.relative_to(root / "corpus").as_posix()
         for p in (root / "corpus").rglob("*") if p.is_file()}
if set(man) - files:
    fail.append(f"in MANIFEST but not in corpus/: {sorted(set(man) - files)[:5]}")
if files - set(man):
    fail.append(f"in corpus/ but not in MANIFEST: {sorted(files - set(man))[:5]}")

for rel, r in man.items():
    p = root / "corpus" / rel
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    if h.hexdigest() != r["sha256"]:
        fail.append(f"checksum mismatch: {rel}")

known = set(man)
for pf in sorted((root / "predictions").glob("*.jsonl")):
    for i, line in enumerate(pf.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if "f1" in d or "precision" in d:
            fail.append(f"{pf.name}:{i} carries a score; this release has no ground truth")
            break

print(f"{len(man)} ontologies, {len(list((root / 'predictions').glob('*.jsonl')))} prediction files")
if fail:
    print("FAILED:")
    for f in fail:
        print("  " + f)
    sys.exit(1)
print("OK")
'''


if __name__ == "__main__":
    main()
