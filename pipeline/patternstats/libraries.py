"""The four pattern libraries: where they are, and which files count as patterns.

Library directories live under ``--data-root`` (the same root
``build_manifest.py`` takes, i.e. ``C:/Users/eno/Documents/PhD/MultiSource_server``).

A *pattern file record* is one RDF file inside a library, ``.git`` excluded.
Three counting units are distinguished throughout the package because the
libraries are organised differently and no single unit is honest for all four:

FILE
    one RDF file.  ODPA ships each submitted pattern as a directory that also
    contains the ontologies the pattern imports (``DUL.owl``, ``COS.owl``,
    ``cpannotationschema.owl`` ...), so FILE over-counts ODPA.
CONTENT
    unique ``sha256`` within a library.  The unit used as the default
    denominator: it removes the 26 physical copies of ``DUL.owl`` but keeps the
    genuinely distinct variants.
STEM
    unique case-folded basename within a library.  What the reuse evidence
    (``pairs.csv``) joins on, because a pattern is identified there by stem.

Additionally each ODPA file is tagged ``role = primary | shipped_dependency``.
``shipped_dependency`` is derived from the data, not from a hand list: a stem
that occurs in at least ``DEP_MIN_DIRS`` distinct pattern directories is a
carried-along import rather than the pattern that directory submits.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# library definitions
# --------------------------------------------------------------------------

#: short code -> (directory name under --data-root, display name used in papers)
LIBRARIES: dict[str, tuple[str, str]] = {
    "ODPA": ("patterns-repository", "OntologyDesignPatterns.org"),
    "MANCHESTER": ("manchester", "Manchester ODP Public Catalog"),
    "DOSDP": ("dos-dp", "DOS-DP"),
    "ISODP": ("Industrial-Standard-Ontology-Design-Patterns", "IS-ODP"),
}

#: order used in every table
LIB_ORDER: tuple[str, ...] = ("ODPA", "MANCHESTER", "DOSDP", "ISODP")

RDF_SUFFIXES: tuple[str, ...] = (
    ".owl", ".ttl", ".rdf", ".n3", ".nt", ".trig", ".jsonld", ".omn", ".ofn",
)

#: a stem seen in >= this many distinct ODPA pattern directories is a shipped
#: dependency, not the pattern that directory contributes.
DEP_MIN_DIRS = 3

SKIP_DIR_NAMES = {".git", ".github", "__pycache__", ".ipynb_checkpoints"}


@dataclass
class PatternFile:
    """One RDF file in one library."""

    library: str                  # short code, e.g. "ODPA"
    rel_path: str                 # posix path relative to the library directory
    abs_path: str
    stem: str                     # basename without suffix
    suffix: str
    n_bytes: int
    sha256: str
    group: str                    # ODPA: top-level directory; else ""
    role: str = "primary"         # primary | shipped_dependency
    content_kind: str = ""        # see sniff()
    key: str = field(default="")  # stable id: "<library>/<rel_path>"

    def __post_init__(self) -> None:
        if not self.key:
            self.key = f"{self.library}/{self.rel_path}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


#: bytes of each file read for ``sniff``
SNIFF_BYTES = 4096


def sniff(path: Path, n_bytes: int) -> str:
    """What the file actually contains, from its first ``SNIFF_BYTES``.

    An ``.owl`` extension is a claim, not a fact.  Nine of the twenty-six
    ``DUL.owl`` files in ODPA are a parked-domain HTML page from an Italian
    registrar, saved when the DOLCE URL they were fetched from had lapsed, and
    the only way to see that without a parser is to look at the bytes.

    Returns one of: ``empty``, ``html``, ``http_error_text``, ``rdf/xml``,
    ``turtle``, ``owl/xml``, ``owl/functional``, ``xml``, ``json``, ``unknown``,
    ``unreadable``.

    ``owl/xml`` and ``owl/functional`` matter because rdflib parses neither: they
    are valid OWL 2 in a syntax no RDF parser reads, so they show up as parse
    failures that are not corrupt files.  Ten ODPA files and six DOS-DP files are
    in this position.
    """
    if n_bytes == 0:
        return "empty"
    try:
        with path.open("rb") as fh:
            raw = fh.read(SNIFF_BYTES)
    except OSError:
        return "unreadable"
    if not raw.strip():
        return "empty"
    text = raw.decode("latin-1", errors="replace")
    low = text.lstrip("﻿ \t\r\n").lower()
    head = low[:1024]
    if low.startswith("<html") or low.startswith("<!doctype html") or "<html" in head:
        return "html"
    if n_bytes < 200 and ("not found" in low or "404" in low or "403" in low
                          or "forbidden" in low):
        return "http_error_text"
    if "<rdf:rdf" in low or "<owl:ontology" in low or "<rdf:description" in low:
        return "rdf/xml"
    if low.startswith("prefix(") or "\nontology(" in low:
        return "owl/functional"
    if "<!doctype ontology" in head or "<ontology" in head:
        return "owl/xml"
    if low.startswith("@prefix") or low.startswith("@base") or "\n@prefix" in low:
        return "turtle"
    if low.startswith("{") or low.startswith("["):
        return "json"
    if low.startswith("<?xml") or low.startswith("<"):
        return "xml"
    return "unknown"


def _walk_rdf(root: Path):
    """Yield RDF files under ``root``, skipping VCS/junk directories.

    ``os.walk`` is used rather than ``Path.rglob`` because ODPA contains a
    *directory* literally named ``Airline.owl``; a glob on ``*.owl`` matches it
    and a naive count then reports 273 files where 272 exist.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for fn in filenames:
            if Path(fn).suffix.lower() in RDF_SUFFIXES:
                yield Path(dirpath) / fn


def discover(data_root: Path, libraries: list[str] | None = None) -> list[PatternFile]:
    """Inventory every RDF file in every library, with hashes and roles."""
    libs = list(libraries or LIB_ORDER)
    out: list[PatternFile] = []

    for code in libs:
        dirname, _ = LIBRARIES[code]
        root = (data_root / dirname).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"library {code}: {root} is not a directory")
        for p in sorted(_walk_rdf(root)):
            rel = p.relative_to(root).as_posix()
            group = rel.split("/")[0] if "/" in rel else ""
            n_bytes = p.stat().st_size
            out.append(
                PatternFile(
                    library=code,
                    rel_path=rel,
                    abs_path=str(p),
                    stem=p.stem,
                    suffix=p.suffix.lower(),
                    n_bytes=n_bytes,
                    sha256=sha256_file(p),
                    group=group,
                    content_kind=sniff(p, n_bytes),
                )
            )

    _assign_roles(out)
    return out


def _norm(s: str) -> str:
    return "".join(c for c in s.casefold() if c.isalnum())


def _assign_roles(files: list[PatternFile]) -> None:
    """Tag shipped dependencies.  Data-driven, per library.

    Rules, in order:
      1. a file whose stem names its own directory is the pattern -- always
         ``primary``.  Without this, ``Situation/situation.owl`` would be
         discarded merely because 5 other patterns ship a copy of it;
      2. otherwise a stem occurring in ``DEP_MIN_DIRS`` or more distinct
         directories is a ``shipped_dependency``.  ``patterns-repository/EEP/``
         contains nothing but ``DUL.owl`` and its documentation, so under this
         rule EEP contributes no pattern file -- which is the honest answer;
      3. otherwise, ``primary``.
    """
    dirs_by_stem: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in files:
        if f.group:
            dirs_by_stem[(f.library, f.stem.casefold())].add(f.group)
    for f in files:
        if not f.group:
            continue
        if _norm(f.stem) == _norm(f.group):
            continue
        if len(dirs_by_stem[(f.library, f.stem.casefold())]) >= DEP_MIN_DIRS:
            f.role = "shipped_dependency"


def dependency_stems(files: list[PatternFile]) -> dict[str, list[tuple[str, int]]]:
    """Per library, the stems tagged ``shipped_dependency`` and their fan-out."""
    counts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for f in files:
        if f.role == "shipped_dependency":
            counts[f.library][f.stem].add(f.group)
    return {
        lib: sorted(((s, len(d)) for s, d in m.items()), key=lambda t: (-t[1], t[0]))
        for lib, m in counts.items()
    }
