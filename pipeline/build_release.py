#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_release.py -- materialise the publishable release.

WHAT IS PUBLISHED
-----------------
The corpus is published UNFILTERED. Every discovered ontology ships, whether or
not it passed the quality gate, because publishing only the filtered subset would
make the quality analysis unfalsifiable: a reader could not check whether an
exclusion was justified, nor re-gate with different thresholds. The gate ships as
METADATA (MANIFEST.csv), not as a destructive filter.

    corpus/<source>/<id>/<id>.<ext>          every ontology, gated or not
    corpus/<source>/<id>/patterns/*.<ext>    all of its candidate patterns
    MANIFEST.csv                             every ontology + gate status + WHY

THE SIX VIEWS
-------------
On top of that corpus sit six views -- three portions x two reuse levels:

    views/github-reuse      views/github-complete
    views/noOBO-reuse       views/noOBO-complete
    views/OBO-reuse         views/OBO-complete

  *-reuse     ontologies passing gate rules G1-G5 with at least one verified
              reused pattern entity.
  *-complete  those where at least one pattern is reused in full (G6), computed
              AFTER pattern copies and duplicates are removed -- so, unlike the
              previously published set, it cannot be satisfied by an ontology
              that simply IS a copy of a pattern.

A view is a filter, not a copy: it holds tables that point into `corpus/`, so
nothing is duplicated and a user can always inspect what a view left out.

    views/<name>/pairs.csv          which pattern belongs to which ontology
    views/<name>/ontologies.csv     per-ontology metadata
    views/<name>/statistics.csv|md  the paper's numbers for this view
    views/<name>/README.md          self-contained description

Each `<name>-hashed` sibling holds the IRI-anonymised tables plus `id_map.csv`.

Usage
-----
    # corpus + views (file copies only, fast)
    python build_release.py --manifest out/MANIFEST.csv         --data-root /path/to/MultiSource --out ../release

    # add the hashed variants (parses and rewrites RDF; slow on the OBO portion)
    python build_release.py --manifest out/MANIFEST.csv         --data-root /path/to/MultiSource --out ../release --hash
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file, pattern_files  # noqa: E402

SOURCES = ("github", "noOBO", "OBO")
LEVELS = ("reuse", "complete")
SOURCE_LABEL = {
    "github": "GitHub Code Search portion",
    "noOBO": "Paper & Repository portion (non-OBO)",
    "OBO": "DOS-DP portion (OBO-family ontologies built from YAML templates)",
}

# Same hash as MultiSource/hash_iris.py so the two releases interoperate.
HASH_PREFIX = "http://www.mydata.org/"


def hash_iri(iri: str, prefix: str = HASH_PREFIX) -> str:
    h = hashlib.sha1(iri.encode("utf-8")).digest()
    return prefix + base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")


def hash_token(value: str) -> str:
    h = hashlib.sha1(value.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")


def longpath(p: Path) -> str:
    """Windows-safe path string for file I/O.

    Some GitHub-derived ontology ids are ~100 characters, and the release layout
    repeats the id twice (``corpus/<source>/<id>/<id>.owl``), which overruns the
    260-character MAX_PATH default and fails with a misleading FileNotFoundError
    on an open() whose parent directory demonstrably exists. The ``\\\\?\\``
    prefix opts into extended-length paths. No-op on POSIX.
    """
    if sys.platform != "win32":
        return str(p)
    ap = p.resolve()
    s = str(ap)
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


def makedirs(p: Path) -> None:
    """mkdir -p that survives MAX_PATH on Windows.

    Path.mkdir() uses the non-extended Win32 API and fails at 260 characters
    even when the subsequent copy, which goes through longpath(), would succeed.
    """
    import os
    os.makedirs(longpath(p), exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    makedirs(dst.parent)
    shutil.copyfile(longpath(src), longpath(dst))


# ----------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select(rows: list[dict], source: str, level: str) -> list[dict]:
    """Rows belonging to one of the six dataset VIEWS.

    A view is a filter over the published corpus, not a separate copy of it. The
    corpus is published UNFILTERED (see publish_corpus): a user must be able to
    see the ontologies the gate rejected and disagree with the gate. The views
    say which ontologies a given analysis used, and `MANIFEST.csv` says why each
    of the others was excluded.
    """
    out = [r for r in rows if r["dataset"] == source and not r["drop_reason"]]
    if level == "complete":
        out = [r for r in out if int(r.get("n_complete_patterns") or 0) > 0]
    return sorted(out, key=lambda r: r["ontology_id"])


def publish_corpus(rows: list[dict], data_root: Path, dest: Path,
                   sources: list[str]) -> dict:
    """Copy EVERY discovered ontology, gate-passing or not.

    Publishing only the filtered subset would make the quality analysis
    unfalsifiable: a reader could not check whether an exclusion was justified,
    and could not re-gate with different thresholds. So the corpus ships whole
    and the gate ships as metadata.
    """
    counts: dict[str, int] = {}
    for source in sources:
        n = 0
        for r in sorted((x for x in rows if x["dataset"] == source),
                        key=lambda x: x["ontology_id"]):
            oid = r["ontology_id"]
            src = data_root / DATASETS[source] / oid
            if not src.is_dir():
                continue
            d = dest / source / oid
            makedirs(d / "patterns")
            main = main_ontology_file(src)
            if main is not None:
                copy_file(main, d / main.name)
            for pf in pattern_files(src):
                copy_file(pf, d / "patterns" / pf.name)
            n += 1
        counts[source] = n
        print(f"[OK] corpus/{source}: {n} ontologies published (unfiltered)")
    return counts


def pair_rows(onto_dir: Path, oid: str, strip_annots: bool = True,
              approved: set[str] | None = None) -> list[dict]:
    """Recompute the ontology-pattern relation for the release.

    ``approved`` names the pattern stems whose LABEL match a human adjudicated
    as genuine reuse for THIS ontology. For those stems the reused-entity set is
    the union of IRI-matched and label-matched entities; every other stem stays
    exact-IRI. Without this the iri+label release would silently contain
    exact-IRI content, because the relation is recomputed here rather than read
    from the manifest.
    """
    main = main_ontology_file(onto_dir)
    if main is None:
        return []
    g, err = parse_graph(main)
    if g is None:
        return []
    if strip_annots:
        g, _ = strip_annotations(g)
    host = set(collect_entities(g, wide=True))
    out = []
    for pf in pattern_files(onto_dir):
        pg, _ = parse_graph(pf)
        if pg is None:
            continue
        if strip_annots:
            pg, _ = strip_annotations(pg)
        pents = set(collect_entities(pg, wide=True))
        if not pents:
            continue
        inter = host & pents
        n_label = 0
        if approved and pf.stem in approved:
            from odpqual.gtmodes import gt_by_label
            extra = gt_by_label(g, host, pg) - inter
            if extra:
                inter = inter | extra
                pents = pents | extra
                n_label = len(extra)
        if not inter:
            continue
        out.append({
            "ontology_id": oid,
            "pattern": pf.stem,
            "n_label_entities": n_label,
            "pattern_file": f"ontologies/{oid}/patterns/{pf.name}",
            "n_pattern_entities": len(pents),
            "n_reused_entities": len(inter),
            "coverage_pct": round(100.0 * len(inter) / len(pents), 2),
            "complete_reuse": int(pents <= host),
        })
    return out


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None,
              eol: str | None = None) -> None:
    """``eol`` None keeps the csv module's CRLF; merge_harvest passes "\\n"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore",
                           lineterminator=eol or "\r\n")
        w.writeheader()
        w.writerows(rows)


def statistics(rows: list[dict], pairs: list[dict]) -> dict:
    n_onto = len(rows)
    pats = {p["pattern"] for p in pairs}
    cov = [p["coverage_pct"] for p in pairs]
    ents = [int(r["n_entities_wide"] or 0) for r in rows]
    per_onto = Counter(p["ontology_id"] for p in pairs)
    thresholds = {}
    for t in (10, 30, 50, 70, 90, 100):
        thresholds[f"onto_ge_{t}pct"] = len(
            {p["ontology_id"] for p in pairs if p["coverage_pct"] >= t})
    return {
        "n_ontologies": n_onto,
        "n_pairs": len(pairs),
        "n_distinct_patterns": len(pats),
        "n_complete_pairs": sum(1 for p in pairs if p["complete_reuse"]),
        "n_complete_ontologies": len(
            {p["ontology_id"] for p in pairs if p["complete_reuse"]}),
        "mean_patterns_per_ontology": round(
            sum(per_onto.values()) / n_onto, 2) if n_onto else 0.0,
        "median_entities_per_ontology": (
            sorted(ents)[len(ents) // 2] if ents else 0),
        "mean_entities_per_ontology": round(sum(ents) / n_onto, 1) if n_onto else 0.0,
        "mean_coverage_pct": round(sum(cov) / len(cov), 2) if cov else 0.0,
        **thresholds,
    }


def top_patterns(pairs: list[dict], k: int = 10) -> list[dict]:
    c = Counter(p["pattern"] for p in pairs)
    return [{"rank": i, "pattern": p, "n_ontologies": n}
            for i, (p, n) in enumerate(c.most_common(k), 1)]


# ----------------------------------------------------------------------

DATASET_README = """\
# {name}

{source_label} — **{level_desc}**

Part of the **ODP-Reuse** release. This directory is self-contained: it holds the
ontologies, the ODP files they reuse, and the mapping between them.

## Contents

| Path | What it is |
|---|---|
| `ontologies/<id>/<id>.<ext>` | the ontology |
| `ontologies/<id>/patterns/*` | the ODP files this ontology reuses |
| `pairs.csv` | **the ontology-to-pattern mapping** — one row per (ontology, pattern) |
| `ontologies.csv` | per-ontology metadata (entity counts, reuse measures) |
| `statistics.csv`, `statistics.md` | the numbers reported in the paper for this dataset |
| `top_patterns.csv` | the most frequently reused patterns here |

## `pairs.csv` columns

| Column | Meaning |
|---|---|
| `ontology_id` | directory name under `ontologies/` |
| `pattern` | pattern stem (its file name without extension) |
| `pattern_file` | path to the pattern file, relative to this directory |
| `n_pattern_entities` | entities declared by the pattern |
| `n_reused_entities` | of those, how many occur in the ontology |
| `coverage_pct` | `100 * n_reused / n_pattern_entities` |
| `complete_reuse` | 1 if every pattern entity occurs in the ontology |

## Headline statistics

{stats_table}

## How this dataset was built

Every ontology here passed the quality gate in `pipeline/odpqual/gate.py`:

1. **G1** it parses as RDF;
2. **G2** it is *not itself a copy of a pattern* — its `owl:Ontology` IRI is not in
   a pattern-library namespace, it is not byte-identical to one of its patterns,
   and its entity set does not equal a pattern's;
3. **G3** it is not a byte-identical duplicate of another ontology (forks and
   mirrors collapse to one representative);
4. **G4** it contributes at least 10 entities of its own beyond its patterns;
5. **G5** at least one pattern has a non-empty entity overlap.
{level_gate}

Rule G2 matters most. A 100%-exact-IRI criterion is *tautological* on a file that
is a verbatim copy of a pattern, and the previously published complete-reuse set
contained 14 such files out of 49. They are excluded here.

Entity sets use the WIDE definition (declared classes and properties, plus IRIs
occurring only in a class/property/individual position). Annotation vocabularies
(SKOS, DC, DCTERMS, CPANN, VANN, FOAF, …) are stripped from **both** the ontology
and the pattern before comparison, so the two sides are treated identically.

## Reproducing it

```bash
cd pipeline
python build_manifest.py --data-root <corpus> --out out
python build_release.py --manifest out/MANIFEST.csv --data-root <corpus> --out ../release
```

`../MANIFEST.csv` in this release records **every** ontology considered, including
the ones the gate dropped and why — so these counts can be audited rather than
taken on trust.
"""


def render_stats_table(stats: dict) -> str:
    keys = [
        ("n_ontologies", "Ontologies"),
        ("n_pairs", "Ontology–pattern pairs"),
        ("n_distinct_patterns", "Distinct patterns"),
        ("n_complete_ontologies", "Ontologies with a fully reused pattern"),
        ("n_complete_pairs", "Fully reused pairs"),
        ("mean_patterns_per_ontology", "Mean patterns per ontology"),
        ("mean_coverage_pct", "Mean pattern coverage (%)"),
        ("median_entities_per_ontology", "Median entities per ontology"),
    ]
    out = ["| Measure | Value |", "|---|---:|"]
    out += [f"| {label} | {stats[k]} |" for k, label in keys if k in stats]
    return "\n".join(out)


def build_dataset(source: str, level: str, rows: list[dict], data_root: Path,
                  out_root: Path,
                  approved: dict[tuple[str, str], set[str]] | None = None) -> dict:
    name = f"{source}-{level}"
    dest = out_root / "views" / name
    dest.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    kept: list[dict] = []
    for r in rows:
        oid = r["ontology_id"]
        src_dir = data_root / DATASETS[source] / oid
        if not src_dir.is_dir():
            continue
        prs = pair_rows(src_dir, oid,
                        approved=(approved or {}).get((source, oid)))
        if level == "complete":
            if not any(p["complete_reuse"] for p in prs):
                continue
        elif not prs:
            continue
        # pattern_file points into the shared unfiltered corpus, so a view never
        # duplicates ontology files -- it only says which ones it uses.
        for p in prs:
            p["pattern_file"] = f"../../corpus/{source}/{oid}/patterns/" \
                                + Path(p["pattern_file"]).name
            p["ontology_dir"] = f"../../corpus/{source}/{oid}"
        all_pairs.extend(prs)
        kept.append(r)

    stats = write_view(dest, source, level, kept, all_pairs)
    print(f"[OK] {name}: {stats['n_ontologies']} ontologies, "
          f"{stats['n_pairs']} pairs, {stats['n_distinct_patterns']} patterns")
    return {"dataset": name, **stats}


VIEW_PAIR_COLUMNS = [
    "ontology_id", "ontology_dir", "pattern", "pattern_file",
    "n_pattern_entities", "n_reused_entities", "coverage_pct",
    "complete_reuse", "n_label_entities"]
VIEW_ONTOLOGY_COLUMNS = [
    "ontology_id", "dataset", "n_pattern_files", "n_triples",
    "n_entities_narrow", "n_entities_wide", "n_own_entities",
    "n_gt_patterns", "n_complete_patterns", "max_reuse_pct"]


def write_view(dest: Path, source: str, level: str, kept: list[dict],
               all_pairs: list[dict], eol: str | None = None) -> dict:
    """Write one view's tables, statistics and README; return its statistics.

    Shared with merge_harvest.py, which appends a harvest's new ontologies to
    the tables and rewrites the rest from them.
    """
    name = f"{source}-{level}"
    write_csv(dest / "pairs.csv", all_pairs, VIEW_PAIR_COLUMNS, eol=eol)
    write_csv(dest / "ontologies.csv", kept, VIEW_ONTOLOGY_COLUMNS, eol=eol)

    stats = statistics(kept, all_pairs)
    write_csv(dest / "statistics.csv", [stats], eol=eol)
    write_csv(dest / "top_patterns.csv", top_patterns(all_pairs), eol=eol)

    level_desc = ("every ontology with at least one verified reused pattern entity"
                  if level == "reuse"
                  else "ontologies in which at least one pattern is reused in full")
    level_gate = ("" if level == "reuse" else
                  "6. **G6** at least one pattern is reused in full, evaluated after\n"
                  "   rules 1-5 have already removed pattern copies and duplicates.\n")
    (dest / "statistics.md").write_text(
        f"# Statistics — {name}\n\n{render_stats_table(stats)}\n", encoding="utf-8",
        newline=eol)
    (dest / "README.md").write_text(DATASET_README.format(
        name=name, source_label=SOURCE_LABEL[source], level_desc=level_desc,
        stats_table=render_stats_table(stats), level_gate=level_gate),
        encoding="utf-8", newline=eol)
    return stats


# ----------------------------------------------------------------------

def hash_dataset(plain: Path, hashed: Path, max_mb: float = 0.0,
                 skip_existing: bool = False, progress_every: int = 0) -> dict:
    """IRI-hash every RDF file of a plain dataset directory.

    Identifiers in pairs.csv / ontologies.csv are hashed with the same function,
    so the mapping still resolves. Failures are recorded, never silent.

    ``skip_existing`` makes the walk resumable: a file whose output already
    exists is counted and not re-parsed. That matters when the input is the full
    corpus rather than a view -- 6,345 files and gigabytes of OWL, where a crash
    two thirds of the way through would otherwise mean starting over.
    ``progress_every`` prints a heartbeat every N files so a long background run
    is observable instead of silent.
    """
    from rdflib import BNode, Graph, URIRef
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    protected = tuple(str(n) for n in (RDF, RDFS, OWL, XSD)) + (
        "http://www.w3.org/XML/1998/namespace",)
    hashed.mkdir(parents=True, exist_ok=True)
    id_map: dict[tuple[str, str], str] = {}
    failures: list[dict] = []
    n_files = 0

    def hash_graph(g: Graph) -> Graph:
        out = Graph()
        imap: dict[URIRef, URIRef] = {}
        bmap: dict[BNode, BNode] = {}

        def m(t):
            if isinstance(t, URIRef):
                s = str(t)
                if s.startswith(protected):
                    return t
                if t not in imap:
                    imap[t] = URIRef(hash_iri(s))
                return imap[t]
            if isinstance(t, BNode):
                if t not in bmap:
                    bmap[t] = BNode(f"b{hash_token(str(t))}")
                return bmap[t]
            return t

        for s, p, o in g:
            out.add((m(s), m(p), m(o)))
        return out

    candidates = [p for p in sorted(plain.rglob("*"))
                  if p.is_file() and p.suffix.lower() in
                  (".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3", ".trig", ".jsonld")]
    n_seen = n_resumed = 0

    for src in candidates:
        rel = src.relative_to(plain)
        dst = hashed / rel
        n_seen += 1
        if progress_every and n_seen % progress_every == 0:
            print(f"[..] {hashed.name}: {n_seen}/{len(candidates)} seen, "
                  f"{n_files} hashed, {n_resumed} already present, "
                  f"{len(failures)} failures", flush=True)
        if skip_existing and Path(longpath(dst)).exists():
            n_resumed += 1
            n_files += 1
            continue
        if max_mb > 0 and src.stat().st_size > max_mb * 1024 * 1024:
            failures.append({"file": str(rel), "reason": "file_too_large",
                             "detail": f"{src.stat().st_size / 1e6:.0f}MB"})
            continue
        g, err = parse_graph(src)
        if g is None:
            failures.append({"file": str(rel), "reason": "parse_error",
                             "detail": err})
            continue
        try:
            hg = hash_graph(g)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fmt = {".ttl": "turtle", ".n3": "n3", ".nt": "nt",
                   ".trig": "trig", ".jsonld": "json-ld"}.get(
                       src.suffix.lower(), "xml")
            hg.serialize(destination=longpath(dst), format=fmt)
            n_files += 1
        except MemoryError:
            failures.append({"file": str(rel), "reason": "out_of_memory",
                             "detail": f"{src.stat().st_size / 1e6:.0f}MB"})
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(rel), "reason": "serialize_error",
                             "detail": type(exc).__name__})

    # identifier-level tables: hash the ids so the mapping still resolves
    for fname, idcols in (("pairs.csv", ("ontology_id", "pattern")),
                          ("ontologies.csv", ("ontology_id",)),
                          ("top_patterns.csv", ("pattern",))):
        srcf = plain / fname
        if not srcf.is_file():
            continue
        with srcf.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for c in idcols:
                if c in r and r[c]:
                    kind = "ontology" if c == "ontology_id" else "pattern"
                    id_map[(kind, r[c])] = hash_token(r[c])
                    r[c] = id_map[(kind, r[c])]
            if "pattern_file" in r and r["pattern_file"]:
                r["pattern_file"] = "(see hashed ontologies/ tree)"
        if rows:
            write_csv(hashed / fname, rows, list(rows[0]))

    for fname in ("statistics.csv", "statistics.md", "top_patterns.csv"):
        s = plain / fname
        if s.is_file() and not (hashed / fname).exists():
            copy_file(s, hashed / fname)

    with (hashed / "id_map.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "original", "hashed"])
        for (kind, orig), h in sorted(id_map.items()):
            w.writerow([kind, orig, h])

    if failures:
        write_csv(hashed / "hash_failures.csv", failures,
                  ["file", "reason", "detail"])

    (hashed / "README.md").write_text(f"""\
# {hashed.name}

IRI-anonymised form of [`{plain.name}`](../{plain.name}). **Identical content and
identical reuse relations** — only identifiers differ.

Every IRI is replaced by `{HASH_PREFIX}` + URL-safe base64 of its SHA-1, using the
same function as `MultiSource/hash_iris.py`, so this release and the ontology
dumps interoperate. RDF, RDFS, OWL and XSD IRIs are left intact. Blank node
labels are hashed deterministically. Ontology ids and pattern stems in
`pairs.csv`, `ontologies.csv` and `top_patterns.csv` are hashed with the same
function; `id_map.csv` is the crosswalk.

## Why an anonymised form exists

An ODP-detection method evaluated on the plain form can cheat: it can recognise
`ontologydesignpatterns.org` namespace prefixes instead of learning structure. On
the hashed form that shortcut is unavailable, so a score reflects structural and
semantic signal. Use the hashed form to **evaluate** detection methods and the
plain form to **inspect** or **curate**.

Because the hash is deterministic, identical IRIs stay aligned across ontologies
and patterns: an entity reused by two ontologies still has one identifier.

Files: {n_files} RDF files rewritten{
    f"; {len(failures)} could not be processed (see `hash_failures.csv`)" if failures else ""}.
""", encoding="utf-8")

    print(f"[OK] {hashed.name}: {n_files} RDF files hashed"
          + (f", {len(failures)} failures" if failures else ""))
    return {"dataset": hashed.name, "n_files": n_files,
            "n_failures": len(failures)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--levels", nargs="*", default=list(LEVELS), choices=LEVELS)
    ap.add_argument("--no-copy", action="store_true",
                    help="write the tables and views only; do not copy the corpus")
    ap.add_argument("--approved-labels", type=Path, default=None,
                    help="CSV of human-adjudicated label matches; selects the "
                         "iri+label content. Omit for the exact-IRI release.")
    ap.add_argument("--hash", action="store_true",
                    help="also build the hashed sibling of every dataset")
    ap.add_argument("--hash-max-mb", type=float, default=0.0,
                    help="skip hashing files larger than this (0 = no limit)")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    approved: dict[tuple[str, str], set[str]] = {}
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.setdefault((r["dataset"], r["ontology"]),
                                        set()).add(r["pattern"])
        print(f"[iri+label] {len(approved)} ontologies with confirmed labels")
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. the UNFILTERED corpus, published once and shared by every view
    if not args.no_copy:
        publish_corpus(rows, args.data_root, args.out / "corpus", args.sources)

    # 2. the six views over it
    summary: list[dict] = []
    for source in args.sources:
        for level in args.levels:
            sel = select(rows, source, level)
            summary.append(build_dataset(source, level, sel, args.data_root,
                                         args.out, approved=approved or None))

    write_csv(args.out / "dataset_statistics.csv", summary)

    # the full gate record travels with the release: dropped rows included
    copy_file(args.manifest, args.out / "MANIFEST.csv")
    for extra in ("gate_summary.md", "gate_summary.csv"):
        src = args.manifest.parent / extra
        if src.is_file():
            copy_file(src, args.out / extra)
    # the filter documentation and the removal audit travel with the data
    here = Path(__file__).resolve().parent
    if (here / "FILTERS.md").is_file():
        copy_file(here / "FILTERS.md", args.out / "FILTERS.md")
    audit = args.manifest.parent / "audit"
    if audit.is_dir():
        for f in sorted(audit.glob("*.csv")):
            copy_file(f, args.out / "audit" / f.name)

    if args.hash:
        for source in args.sources:
            for level in args.levels:
                plain = args.out / "views" / f"{source}-{level}"
                if plain.is_dir():
                    hash_dataset(plain,
                                 args.out / "views" / f"{source}-{level}-hashed",
                                 max_mb=args.hash_max_mb)

    print(f"\n[OK] release at {args.out}")
    print("     write the top-level README with: python write_release_readme.py "
          f"--release {args.out}")


if __name__ == "__main__":
    main()
