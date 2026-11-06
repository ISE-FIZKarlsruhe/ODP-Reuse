#!/usr/bin/env python3
"""
pattern_iris.py -- decide which pattern IRIs the harvester searches for.

The released corpus searched the IRIs that ``run_multiple_source.sh`` extracted
from a clone of odpa/patterns-repository (MultiSource/report/retrieved_patterns.csv,
144 IRIs). This module holds that extraction, so the release, a local re-run and
the quarterly CI harvest all use the same rules.

Two commands:

  extract   the original extraction, same positional arguments and outputs as
            the heredoc it replaces in run_multiple_source.sh; prints the
            unique IRIs on stdout.

                python pattern_iris.py extract <host-filter> <report.csv> \\
                    <dirs_summary.csv> <dirs_without_iri.txt> <files_without_iri.txt> FILE...

  sets      the search set for a harvest: every IRI the release searched,
            labelled ``release``, plus every IRI in the pattern repository today
            that the release did not search, labelled ``new``. Keeping the
            release IRIs even when they have left the repository makes the
            harvest like-for-like with the release, and the label keeps "GitHub
            changed" apart from "the ODPA added patterns".

                python pattern_iris.py sets --release report/retrieved_patterns.csv \\
                    --repo patterns-repository --out report/pattern_sets.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RX_XML_BASE = re.compile(r'xml:base\s*=\s*"([^"]+)"', re.I)
RX_XMLNS_BASE = re.compile(r'xmlns:base\s*=\s*"([^"]+)"', re.I)   # non-standard but used
RX_XMLNS_DEFAULT = re.compile(r'xmlns\s*=\s*"([^"]+)"', re.I)       # default namespace (strip '#')
RX_ONTO_ATTR = re.compile(r'<owl:Ontology[^>]*?(?:IRI|rdf:about)\s*=\s*"([^"]+)"', re.I)
RX_OWLXML_ONTOIRI = re.compile(r'<Ontology[^>]*?\sontologyIRI\s*=\s*"([^"]+)"', re.I)  # OWL/XML
RX_TTL_AT_BASE = re.compile(r'@base\s*<([^>]+)>', re.I)
RX_TTL_BASE = re.compile(r'^\s*BASE\s*<([^>]+)>\s*$', re.I | re.M)               # Turtle 1.1
RX_TTL_EXPL_SUBJ = re.compile(r'^\s*<([^>]+)>\s+a\s+owl:Ontology\b', re.I | re.M)
RX_TTL_VER_IRI = re.compile(r'owl:versionIRI\s*<([^>]+)>', re.I)
RX_TTL_PREFIX_COLON = re.compile(r'@prefix\s*:\s*<([^>]+)>\s*\.', re.I)
RX_TTL_COLON_IS_ONTO = re.compile(r'^\s*:\s+a\s+owl:Ontology\b', re.I | re.M)

RX_GENERIC_TEMPLATE = re.compile(r'/template(/.*)?$', re.I)
RX_GENERIC_HOMEPAGE = re.compile(r'^https?://(www\.)?ontologydesignpatterns\.org/?$', re.I)

ONTOLOGY_SUFFIXES = (".owl", ".ttl")


def extract_iris(text: str, host_filter: str = "") -> set[tuple[str, str]]:
    """Return {(iri, method)} for every ontology IRI the text declares."""
    matches: set[tuple[str, str]] = set()

    def add(iri: str, method: str) -> None:
        iri = iri.strip()
        if iri and (host_filter == "" or host_filter.lower() in iri.lower()):
            matches.add((iri, method))

    for m in RX_XML_BASE.findall(text):
        add(m, "xml:base")
    for m in RX_XMLNS_BASE.findall(text):
        add(m, "xmlns:base (namespace)")
    for m in RX_XMLNS_DEFAULT.findall(text):
        add(m.strip().rstrip('#'), "xmlns (default namespace)")
    for m in RX_ONTO_ATTR.findall(text):
        add(m, "owl:Ontology IRI/rdf:about")
    for m in RX_OWLXML_ONTOIRI.findall(text):
        add(m, "OWL/XML ontologyIRI")
    for m in RX_TTL_AT_BASE.findall(text):
        add(m, "Turtle @base")
    for m in RX_TTL_BASE.findall(text):
        add(m, "Turtle BASE")
    for m in RX_TTL_EXPL_SUBJ.findall(text):
        add(m, "Turtle <IRI> a owl:Ontology")
    for m in RX_TTL_VER_IRI.findall(text):
        add(m, "owl:versionIRI")
    # Heuristic: default ':' prefix + ': a owl:Ontology'
    m = RX_TTL_PREFIX_COLON.search(text)
    if m and RX_TTL_COLON_IS_ONTO.search(text):
        add(m.group(1).strip(), "Turtle @prefix : <> + : a owl:Ontology")
    return matches


def is_generic(iri: str) -> bool:
    """Template and homepage IRIs say nothing about which pattern is reused."""
    return bool(RX_GENERIC_TEMPLATE.search(iri) or RX_GENERIC_HOMEPAGE.match(iri))


# Language namespaces that pattern files declare as their default namespace
# (xmlns="..."). The extraction picks them up, but they are not patterns: a code
# search for the OWL namespace returns any OWL file on GitHub.
LANGUAGE_NAMESPACES = {
    "http://www.w3.org/2002/07/owl",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns",
    "http://www.w3.org/2000/01/rdf-schema",
    "http://www.w3.org/2001/XMLSchema",
    "http://www.w3.org/1999/xhtml",
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/XML/1998/namespace",
}


def is_language_namespace(iri: str) -> bool:
    return iri.strip().rstrip("#/") in LANGUAGE_NAMESPACES


def read_text(p) -> str:
    try:
        with io.open(p, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def repo_files(repo: Path) -> list[str]:
    return sorted(str(p) for p in repo.rglob("*")
                  if p.is_file() and p.suffix.lower() in ONTOLOGY_SUFFIXES
                  and ".git" not in p.parts)


def unique_iris(files, host_filter: str = "") -> list[str]:
    found: set[str] = set()
    for f in files:
        found |= {iri for iri, _ in extract_iris(read_text(f), host_filter)}
    return sorted(found)


def build_pattern_sets(release_iris, current_iris) -> list[dict[str, str]]:
    release = list(dict.fromkeys(i for i in release_iris if i and not is_language_namespace(i)))
    seen = set(release)
    new = sorted(i for i in set(current_iris)
                 if i not in seen and not is_generic(i) and not is_language_namespace(i))
    return ([{"pattern_iri": i, "pattern_set": "release"} for i in release]
            + [{"pattern_iri": i, "pattern_set": "new"} for i in new])


def read_iri_list(path: Path) -> list[str]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines[1:] if lines and lines[0] == "pattern_iri" else lines


# ---------------------------------------------------------------- commands

def cmd_extract(host_filter: str, extract_report: str, dirs_summary: str,
                dirs_no_iri: str, files_no_iri: str, files: list[str]) -> None:
    rows = []
    dir_counts = defaultdict(lambda: {"total": 0, "with": 0, "without": 0, "iris": set()})
    files_without = []

    for p in files:
        d = os.path.dirname(p)
        dir_counts[d]["total"] += 1
        matches = extract_iris(read_text(p), host_filter)
        if matches:
            dir_counts[d]["with"] += 1
            for iri, method in sorted(matches):
                rows.append({"file": p, "dir": d, "iri": iri, "method": method})
                dir_counts[d]["iris"].add(iri)
        else:
            dir_counts[d]["without"] += 1
            files_without.append(p)

    with open(extract_report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "dir", "iri", "method"])
        w.writeheader()
        w.writerows(rows)

    with open(dirs_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dir", "total_files", "files_with_iri",
                                          "files_without_iri", "unique_iris"])
        w.writeheader()
        for d in sorted(dir_counts):
            w.writerow({"dir": d, "total_files": dir_counts[d]["total"],
                        "files_with_iri": dir_counts[d]["with"],
                        "files_without_iri": dir_counts[d]["without"],
                        "unique_iris": len(dir_counts[d]["iris"])})

    with open(dirs_no_iri, "w", encoding="utf-8") as f:
        for d in sorted(dir_counts):
            if dir_counts[d]["with"] == 0:
                f.write(d + "\n")
    with open(files_no_iri, "w", encoding="utf-8") as f:
        for p in files_without:
            f.write(p + "\n")

    for iri in sorted({r["iri"] for r in rows}):
        print(iri)


def cmd_sets(release: Path, repo: Path, out: Path) -> None:
    release_iris = read_iri_list(release)
    dropped = [i for i in release_iris if is_language_namespace(i)]
    if dropped:
        print(f"[pattern_iris] not searched (language namespaces, not patterns): "
              f"{', '.join(dropped)}", file=sys.stderr)
    rows = build_pattern_sets(release_iris, unique_iris(repo_files(repo)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pattern_iri", "pattern_set"])
        w.writeheader()
        w.writerows(rows)
    n_new = sum(r["pattern_set"] == "new" for r in rows)
    print(f"[pattern_iris] {len(rows) - n_new} release + {n_new} new pattern IRIs -> {out}",
          file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="extract IRIs from files (run_multiple_source.sh)")
    ex.add_argument("host_filter")
    ex.add_argument("extract_report")
    ex.add_argument("dirs_summary")
    ex.add_argument("dirs_no_iri")
    ex.add_argument("files_no_iri")
    ex.add_argument("files", nargs="*")

    st = sub.add_parser("sets", help="write the release/new search set for a harvest")
    st.add_argument("--release", required=True, type=Path,
                    help="IRIs the release searched (report/retrieved_patterns.csv)")
    st.add_argument("--repo", required=True, type=Path, help="clone of odpa/patterns-repository")
    st.add_argument("--out", required=True, type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        cmd_extract(a.host_filter, a.extract_report, a.dirs_summary, a.dirs_no_iri,
                    a.files_no_iri, a.files)
    else:
        cmd_sets(a.release, a.repo, a.out)


if __name__ == "__main__":
    main()
