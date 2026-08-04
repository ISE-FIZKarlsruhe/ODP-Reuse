"""
The quality gate.

Each rule below has a stated justification, because the gate is the paper's
methodological contribution and not merely data hygiene. The gate is applied in
order; the first failing rule is recorded as ``drop_reason`` and later rules are
not evaluated (so counts partition cleanly).

  G1  PARSEABLE
      The ontology file must parse as RDF.
      Justification: an unparseable file cannot be analysed at all. 234 of 1,784
      audited GitHub files fail here. Excluding them is forced, not a choice.

  G2  NOT A PATTERN COPY  (the decisive rule)
      The ontology must not itself be a pattern. It fails if
        (a) any IRI it declares as owl:Ontology lies in a pattern-library
            namespace, or
        (b) its file is byte-identical to one of its own pattern files, or
        (c) its WIDE entity set equals a pattern's WIDE entity set.
      Justification: a 100%-exact-IRI-match criterion is tautological on such
      files -- the "reuse" is the file being a copy of the pattern. 14 of the 49
      ontologies in the previously published complete-reuse set fail (a) and 6
      fail (b), 16 distinct once combined with G3 duplicates. Without this rule
      the benchmark measures file duplication, not reuse.

      Measured strictly: (a) tests the IRI the file declares as owl:Ontology, not
      any IRI it mentions. Testing "any rdf:about containing a pattern namespace"
      over-counts to 27/49, because it also flags ontologies that legitimately
      REFERENCE pattern classes -- which is the reuse being studied.

  G3  NOT A DUPLICATE
      Among files with identical content hashes, one representative is kept
      (lexicographically smallest id); the rest are dropped as duplicates.
      Justification: 233 of 1,862 GitHub files (12.5%) are byte-identical
      copies from forks and mirrors. Duplicates inflate every count and leak
      across any train/test split.

  G4  SUBSTANTIVE
      The ontology must contribute at least ``min_own_entities`` entities that
      are not supplied by its patterns.
      Justification: 626 of 1,784 GitHub files have (near-)empty entity sets --
      configs, metadata stubs, alignment files. They are not ontologies in the
      sense the study is about, and they cannot exhibit reuse.

  G5  HAS GROUND TRUTH
      At least one pattern must have a non-empty entity intersection with the
      ontology.
      Justification: required for any evaluation; an ontology with no pattern
      overlap carries no reuse signal and no ground truth to score against.

  G6  COMPLETE REUSE  (only for the complete-reuse variant)
      At least one pattern must have *all* of its entities present in the
      ontology, evaluated only on pairs that already passed G1-G5.
      Justification: this is the original complete-reuse criterion, but now
      computed after pattern copies and duplicates have been removed, so it can
      no longer be satisfied by a file being a copy of the pattern.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .entities import (
    collect_entities,
    declared_ontology_iris,
    parse_graph,
    strip_annotations,
)

# Namespaces of the four integrated pattern libraries. An ontology declaring its
# own IRI inside one of these is a copy of a pattern, not a consumer of one.
PATTERN_LIBRARY_NAMESPACES: tuple[str, ...] = (
    "http://www.ontologydesignpatterns.org/",
    "http://ontologydesignpatterns.org/",
    "https://www.ontologydesignpatterns.org/",
    "https://ontologydesignpatterns.org/",
    "http://www.gong.manchester.ac.uk/odp/",
    "http://purl.obolibrary.org/obo/dosdp/",
    "http://www.hsu-hh.de/aut/ontologies/",
)

RDF_SUFFIXES = (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt", ".trig", ".jsonld")

# Rules, in evaluation order. Used to keep report ordering stable.
RULES = ("G1_parse", "G2_pattern_copy", "G3_duplicate", "G4_substantive",
         "G5_has_gt", "G6_complete_reuse")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_pattern_library(iri: str) -> bool:
    return any(iri.startswith(ns) for ns in PATTERN_LIBRARY_NAMESPACES)


@dataclass
class OntologyRecord:
    """One row of MANIFEST.csv."""
    ontology_id: str
    dataset: str
    path: str
    content_sha256: str = ""
    n_pattern_files: int = 0

    parsed: str = ""                     # "ok" or the exception TYPE (short)
    error_detail: str = ""               # the exact exception message, verbatim
    n_triples: int = 0
    n_triples_annotation: int = 0        # removed by strip_annotations
    n_entities_narrow: int = 0
    n_entities_wide: int = 0
    declared_iris: str = ""

    self_match_reason: str = ""          # "" | declared_iri | identical_file | identical_entities
    duplicate_of: str = ""

    n_own_entities: int = 0              # WIDE entities not supplied by any pattern
    n_gt_patterns: int = 0               # patterns with non-empty overlap
    n_complete_patterns: int = 0         # patterns fully contained
    max_reuse_pct: float = 0.0           # best per-pattern coverage, 0-100

    n_label_confirmed: int = 0        # patterns extended by adjudicated labels
    n_label_entities: int = 0         # entities added by those labels

    stage_reached: str = "discovered"
    drop_reason: str = ""

    pattern_errors: list[str] = field(default_factory=list, repr=False)

    patterns: dict[str, set[str]] = field(default_factory=dict, repr=False)
    entities: set[str] = field(default_factory=set, repr=False)

    def csv_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("patterns", "entities", "pattern_errors")}
        # pattern-level parse failures, verbatim, semicolon separated
        d["n_pattern_errors"] = len(self.pattern_errors)
        d["pattern_errors"] = " ; ".join(self.pattern_errors)[:1000]
        return d


def main_ontology_file(onto_dir: Path) -> Path | None:
    """The ontology's own file: prefer <dirname>.<ext>, else the largest RDF file."""
    cands = [p for p in onto_dir.iterdir()
             if p.is_file() and p.suffix.lower() in RDF_SUFFIXES]
    if not cands:
        return None
    exact = [p for p in cands if p.stem == onto_dir.name]
    if exact:
        return sorted(exact, key=lambda p: RDF_SUFFIXES.index(p.suffix.lower()))[0]
    return max(cands, key=lambda p: p.stat().st_size)


def pattern_files(onto_dir: Path) -> list[Path]:
    pdir = onto_dir / "patterns"
    if not pdir.is_dir():
        return []
    return sorted(p for p in pdir.iterdir()
                  if p.is_file() and p.suffix.lower() in RDF_SUFFIXES)


def analyse_ontology(onto_dir: Path, dataset: str,
                     strip_annots: bool = True,
                     max_mb: float = 0.0,
                     approved_labels: set[tuple[str, str]] | None = None
                     ) -> OntologyRecord:
    """Parse one ontology and its patterns; fill everything except G3.

    ``approved_labels`` holds (dataset, ontology, pattern) triples whose
    label-only matches a human adjudicated as genuine reuse. When supplied, the
    ground truth for those pairs is the union of IRI-matched and label-matched
    entities; every other pair stays exact-IRI. Unadjudicated label matches are
    never admitted -- only 21.5% of them proved genuine, so accepting them
    wholesale would corrupt the ground truth rather than extend it.
    """
    rec = OntologyRecord(ontology_id=onto_dir.name, dataset=dataset,
                         path=str(onto_dir))

    main = main_ontology_file(onto_dir)
    if main is None:
        rec.parsed = "NoOntologyFile"
        rec.drop_reason = "G1_parse:NoOntologyFile"
        return rec

    rec.content_sha256 = sha256(main)
    pats = pattern_files(onto_dir)
    rec.n_pattern_files = len(pats)

    if max_mb > 0 and main.stat().st_size > max_mb * 1024 * 1024:
        rec.parsed = "SizeLimitExceeded"
        rec.drop_reason = f"G1_parse:SizeLimitExceeded(>{max_mb}MB)"
        return rec

    # ---- G1 parse. The exact exception message is kept in error_detail; the
    # short type goes in parsed/drop_reason so the summary tallies stay readable.
    g, err = parse_graph(main)
    if g is None:
        short = err.split(":", 1)[0]
        rec.parsed = short
        rec.error_detail = err
        rec.drop_reason = f"G1_parse:{short}"
        return rec
    rec.parsed = "ok"
    rec.n_triples = len(g)
    if strip_annots:
        g, removed = strip_annotations(g)
        rec.n_triples_annotation = removed

    rec.n_entities_narrow = len(collect_entities(g, wide=False))
    ents = set(collect_entities(g, wide=True))
    rec.entities = ents
    rec.n_entities_wide = len(ents)

    iris = declared_ontology_iris(g)
    rec.declared_iris = ";".join(iris[:5])

    # ---- G2 (a) the ontology declares itself inside a pattern library
    if any(in_pattern_library(i) for i in iris):
        rec.self_match_reason = "declared_iri"
        rec.drop_reason = "G2_pattern_copy:declared_iri"
        return rec

    # ---- parse patterns
    main_hash = rec.content_sha256
    for pf in pats:
        if sha256(pf) == main_hash:
            rec.self_match_reason = "identical_file"
            rec.drop_reason = "G2_pattern_copy:identical_file"
            return rec
        pg, _perr = parse_graph(pf)
        if pg is None:
            # An unparseable PATTERN file is not fatal for the ontology, but it
            # must not vanish: it means one candidate pattern could not be
            # checked, which affects G5/G6. Recorded verbatim.
            rec.pattern_errors.append(f"{pf.name}: {_perr}")
            continue
        if strip_annots:
            pg, _ = strip_annotations(pg)
        pents = set(collect_entities(pg, wide=True))
        if pents:
            rec.patterns[pf.stem] = pents

    # ---- G2 (c) entity sets identical to a pattern
    for stem, pents in rec.patterns.items():
        if pents == ents and ents:
            rec.self_match_reason = "identical_entities"
            rec.drop_reason = f"G2_pattern_copy:identical_entities({stem})"
            return rec

    # ---- reuse measures (needed by G4/G5/G6)
    supplied = set().union(*rec.patterns.values()) if rec.patterns else set()
    rec.n_own_entities = len(ents - supplied)

    # Human-confirmed label matches extend the ground truth for those pairs only.
    label_extra: dict[str, set[str]] = {}
    if approved_labels:
        from .gtmodes import gt_by_label
        for stem in rec.patterns:
            if (dataset, onto_dir.name, stem) not in approved_labels:
                continue
            pf = next((f for f in pats if f.stem == stem), None)
            if pf is None:
                continue
            pg, _ = parse_graph(pf)
            if pg is None:
                continue
            extra = gt_by_label(g, ents, pg) - (ents & rec.patterns[stem])
            if extra:
                label_extra[stem] = extra
                rec.patterns[stem] = rec.patterns[stem] | extra
        rec.n_label_confirmed = len(label_extra)
        rec.n_label_entities = sum(len(v) for v in label_extra.values())

    overlaps = {s: len(ents & p) for s, p in rec.patterns.items()}
    rec.n_gt_patterns = sum(1 for v in overlaps.values() if v > 0)
    rec.n_complete_patterns = sum(
        1 for s, p in rec.patterns.items() if p and p <= ents
    )
    rec.max_reuse_pct = round(max(
        (100.0 * len(ents & p) / len(p) for p in rec.patterns.values() if p),
        default=0.0), 2)
    return rec


def apply_gate(records: list[OntologyRecord],
               min_own_entities: int = 1) -> list[OntologyRecord]:
    """Apply G3-G6 across the whole collection and set stage_reached."""
    # ---- G3 duplicates: group by content hash, keep the smallest id
    by_hash: dict[str, list[OntologyRecord]] = {}
    for r in records:
        if r.drop_reason or not r.content_sha256:
            continue
        by_hash.setdefault(r.content_sha256, []).append(r)
    for _h, group in by_hash.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r.ontology_id)
        keeper = group[0]
        for dup in group[1:]:
            dup.duplicate_of = keeper.ontology_id
            dup.drop_reason = f"G3_duplicate:of({keeper.ontology_id})"

    for r in records:
        # stage: how far did it get?
        if r.drop_reason.startswith("G1"):
            r.stage_reached = "discovered"
            continue
        r.stage_reached = "parsed"
        if r.drop_reason.startswith(("G2", "G3")):
            continue

        # ---- G4 substantive
        if r.n_own_entities < min_own_entities:
            r.drop_reason = (f"G4_substantive:own_entities="
                             f"{r.n_own_entities}<{min_own_entities}")
            continue
        r.stage_reached = "substantive"

        # ---- G5 has ground truth
        if r.n_gt_patterns == 0:
            r.drop_reason = "G5_has_gt:no_pattern_overlap"
            continue
        r.stage_reached = "has_gt"

        # ---- G6 complete reuse (variant membership, not a drop for the full set)
        if r.n_complete_patterns > 0:
            r.stage_reached = "complete_reuse"

    return records


def gate_summary(records: list[OntologyRecord]) -> dict:
    """Per-rule attrition counts, in rule order."""
    total = len(records)
    counts = {r: 0 for r in RULES}
    for rec in records:
        if rec.drop_reason:
            rule = rec.drop_reason.split(":", 1)[0]
            if rule in counts:
                counts[rule] += 1
    kept = sum(1 for r in records if not r.drop_reason)
    complete = sum(1 for r in records
                   if not r.drop_reason and r.n_complete_patterns > 0)
    return {
        "discovered": total,
        **{f"dropped_{k}": v for k, v in counts.items()},
        "kept_reuse": kept,
        "kept_complete_reuse": complete,
    }
