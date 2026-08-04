"""Analysis 1 -- alignment of each pattern to a foundational / top-level ontology.

Two independent signals, reported separately and never merged:

IMPORTS
    the objects of ``owl:imports`` in the pattern file.  A declared commitment:
    the pattern's author said "this pattern is built on that".  Strong evidence,
    but it undercounts, because many ODP files reference DUL terms while shipping
    DUL as a sibling file instead of importing it.

REFERENCES
    every distinct IRI occurring anywhere in the file (subject, predicate or
    object position) that is *not* in the pattern's own namespace, mapped to a
    vocabulary by longest-prefix match against ``vocab.NAMESPACES``.  Broad
    evidence, but it over-counts, because a single ``rdfs:seeAlso`` to a DOLCE
    term counts the same as a subclass axiom.  ``--min-ref-iris`` therefore
    gates a reference-based alignment on a minimum number of distinct IRIs from
    that vocabulary (default 1, i.e. ungated; the CSV carries the raw counts so
    any threshold can be applied afterwards).

TRANSITIVE IMPORTS
    ``owl:imports`` alone badly understates ODPA, and for a structural reason:
    ``AgentRole/agentrole.owl`` imports ``objectrole.owl``, which is where the
    DOLCE commitment lives.  So a third figure resolves the import graph *within
    the libraries* -- an import IRI is followed when some library file declares
    it as its ``owl:Ontology`` -- and reports the vocabularies reachable from a
    pattern.  This is a lower bound: imports pointing outside the four libraries
    are counted (``n_unresolved_imports``) but not fetched, and no network access
    happens anywhere in this package.

Both signals are restricted to vocabularies whose tier is ``foundational`` or
``cross_domain``.  ``odp_infra`` (cpannotationschema, the ODPA submission
namespaces, ``urn:unnamed:ontology`` for DOS-DP), ``domain`` (UBERON, GO, OPC-UA
...) and ``rdf_builtin`` are counted and reported but are not "top-level
alignment".  The tier assignment is a judgement and is listed in full in
``vocab.py`` so a reader can disagree with any single line of it.

Outputs
-------
tl_per_pattern.csv        one row per pattern: which vocabularies, by which signal
tl_by_library.csv         library x vocabulary x signal counts
tl_summary.csv            per library: aligned / not aligned, both signals
tl_unrecognised_ns.csv    namespaces no catalogue entry matched, with fan-out
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import units, vocab
from .libraries import LIBRARIES, LIB_ORDER


def resolve(rec: dict) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """``(reference vocab counts, unrecognised signature counts, import vocabs)``.

    Resolution happens here, not in ``extract.py``, so that a change to
    ``vocab.NAMESPACES`` takes effect without reparsing the libraries.
    """
    known, unknown = vocab.map_signature(rec.get("ns_signature", {}))
    imp: dict[str, int] = {}
    for key in rec.get("import_signature", []):
        hit = vocab.lookup(key)
        name = hit[0] if hit else "UNRECOGNISED"
        imp[name] = imp.get(name, 0) + 1
    return known, unknown, imp


def _norm_iri(iri: str) -> str:
    return iri.rstrip("#/")


def build_import_graph(records: list[dict]) -> tuple[dict[str, list[str]], dict[str, int]]:
    """``(key -> keys it imports, key -> number of imports that did not resolve)``.

    An import resolves when some *parsed* library file declares the imported IRI
    as its ``owl:Ontology``.  Nothing is fetched: an import of a URL that is not
    in the four libraries stays unresolved and is counted, not guessed at.
    """
    iri_to_key: dict[str, str] = {}
    for r in records:
        if not r.get("parsed"):
            continue
        for oi in r.get("ontology_iris", []):
            iri_to_key.setdefault(_norm_iri(oi), r["key"])
        own = r.get("own_ns", "")
        if own:
            iri_to_key.setdefault(_norm_iri(own), r["key"])

    edges: dict[str, list[str]] = {}
    unresolved: dict[str, int] = {}
    for r in records:
        if not r.get("parsed"):
            continue
        out, miss = [], 0
        for imp in r.get("imports", []):
            tgt = iri_to_key.get(_norm_iri(imp))
            if tgt is None:
                miss += 1
            elif tgt != r["key"]:
                out.append(tgt)
        edges[r["key"]] = out
        unresolved[r["key"]] = miss
    return edges, unresolved


def reachable(edges: dict[str, list[str]], start: str) -> set[str]:
    """Keys reachable from ``start`` (excluding ``start``), cycle-safe."""
    seen: set[str] = set()
    stack = list(edges.get(start, ()))
    while stack:
        k = stack.pop()
        if k in seen or k == start:
            continue
        seen.add(k)
        stack.extend(edges.get(k, ()))
    return seen


def run(records: list[dict], out_dir: Path, unit: str = units.DEFAULT_UNIT,
        min_ref_iris: int = 1) -> dict:
    recs = units.select(records, unit)
    edges, unresolved = build_import_graph(records)
    by_key = {r["key"]: r for r in records if r.get("parsed")}

    name_tier: dict[str, str] = {}
    for _p, nm, tier in vocab.NAMESPACES:
        name_tier.setdefault(nm, tier)
    name_tier["UNRECOGNISED"] = "unknown"

    per_pattern: list[dict] = []
    # library -> signal -> vocab -> n patterns
    by_lib: dict[tuple[str, str, str], int] = Counter()
    unknown_ns: dict[str, set[str]] = defaultdict(set)   # namespace -> pattern keys
    unknown_ns_lib: dict[str, Counter] = defaultdict(Counter)

    for r in recs:
        ref_counts, unknown, imp_vocab = resolve(r)

        imp_names = {n for n in imp_vocab if n != "UNRECOGNISED"}
        imp_align = {n for n in imp_names if name_tier.get(n) in vocab.ALIGNMENT_TIERS}
        imp_found = {n for n in imp_names if name_tier.get(n) == "foundational"}

        ref_names = {n for n, c in ref_counts.items() if c >= min_ref_iris}
        ref_align = {n for n in ref_names if name_tier.get(n) in vocab.ALIGNMENT_TIERS}
        ref_found = {n for n in ref_names if name_tier.get(n) == "foundational"}
        ref_artefact = {n for n in ref_names if name_tier.get(n) == "artefact"}

        # transitive: this pattern's own imports plus everything reachable
        reach = reachable(edges, r["key"])
        tr_names = set(imp_names)
        for k in reach:
            other = by_key.get(k)
            if other is None:
                continue
            _rc, _un, oimp = resolve(other)
            tr_names |= {n for n in oimp if n != "UNRECOGNISED"}
            # a file reached through imports also *contributes* its own
            # references: reaching DUL.owl is a DOLCE commitment even though
            # DUL.owl imports nothing
            tr_names |= {n for n, c in _rc.items() if c >= min_ref_iris}
            for oi in other.get("ontology_iris", []):
                hit = vocab.lookup(oi)
                if hit:
                    tr_names.add(hit[0])
        tr_align = {n for n in tr_names if name_tier.get(n) in vocab.ALIGNMENT_TIERS}
        tr_found = {n for n in tr_names if name_tier.get(n) == "foundational"}

        for ns in unknown:
            unknown_ns[ns].add(r["key"])
            unknown_ns_lib[ns][r["library"]] += 1

        per_pattern.append({
            "unit": unit,
            "library": r["library"],
            "key": r["key"],
            "stem": r["stem"],
            "group": r["group"],
            "own_ns": r.get("own_ns", ""),
            "n_imports": len(r.get("imports", [])),
            "imports": "|".join(r.get("imports", [])),
            "import_vocabs": "|".join(sorted(imp_names)),
            "import_foundational": "|".join(sorted(imp_found)),
            "import_alignment": "|".join(sorted(imp_align)),
            "aligned_by_imports": int(bool(imp_align)),
            "foundational_by_imports": int(bool(imp_found)),
            "n_imports_resolved_in_libraries": len(edges.get(r["key"], ())),
            "n_imports_unresolved": unresolved.get(r["key"], 0),
            "n_reachable_via_imports": len(reach),
            "transitive_vocabs": "|".join(sorted(tr_names)),
            "transitive_foundational": "|".join(sorted(tr_found)),
            "aligned_by_transitive_imports": int(bool(tr_align)),
            "foundational_by_transitive_imports": int(bool(tr_found)),
            "ref_vocabs": "|".join(f"{n}:{ref_counts[n]}" for n in sorted(ref_names)),
            "ref_foundational": "|".join(sorted(ref_found)),
            "ref_alignment": "|".join(sorted(ref_align)),
            "aligned_by_refs": int(bool(ref_align)),
            "foundational_by_refs": int(bool(ref_found)),
            "artefact_vocabs": "|".join(sorted(ref_artefact)),
            "n_unrecognised_ns": len(unknown),
            "unrecognised_ns": "|".join(sorted(unknown)[:6]),
        })

        for n in imp_names:
            by_lib[(r["library"], "imports", n)] += 1
        for n in ref_names:
            by_lib[(r["library"], "references", n)] += 1
        for n in tr_names:
            by_lib[(r["library"], "transitive_imports", n)] += 1

    units.write_csv(out_dir / "tl_per_pattern.csv", per_pattern)

    # ---- library x vocabulary x signal --------------------------------
    lib_totals = Counter(r["library"] for r in recs)
    rows = []
    for (lib, sig, name), n in sorted(by_lib.items(),
                                      key=lambda kv: (LIB_ORDER.index(kv[0][0]),
                                                      kv[0][1], -kv[1], kv[0][2])):
        rows.append({
            "library": lib,
            "library_name": LIBRARIES[lib][1],
            "signal": sig,
            "vocabulary": name,
            "tier": name_tier.get(name, "unknown"),
            "n_patterns": n,
            "n_patterns_in_library": lib_totals[lib],
            "pct_of_library": units.pct(n, lib_totals[lib]),
        })
    units.write_csv(out_dir / "tl_by_library.csv", rows)

    # ---- per-library summary -----------------------------------------
    summary = []
    for lib in LIB_ORDER:
        sub = [p for p in per_pattern if p["library"] == lib]
        if not sub:
            continue
        n = len(sub)
        imp_any = sum(1 for p in sub if p["n_imports"] > 0)
        a_imp = sum(p["aligned_by_imports"] for p in sub)
        f_imp = sum(p["foundational_by_imports"] for p in sub)
        a_ref = sum(p["aligned_by_refs"] for p in sub)
        f_ref = sum(p["foundational_by_refs"] for p in sub)
        a_tr = sum(p["aligned_by_transitive_imports"] for p in sub)
        f_tr = sum(p["foundational_by_transitive_imports"] for p in sub)
        artefact = sum(1 for p in sub if p["artefact_vocabs"])
        both = sum(1 for p in sub if p["aligned_by_imports"] and p["aligned_by_refs"])
        either = sum(1 for p in sub
                     if p["aligned_by_imports"] or p["aligned_by_refs"]
                     or p["aligned_by_transitive_imports"])
        summary.append({
            "library": lib,
            "library_name": LIBRARIES[lib][1],
            "unit": unit,
            "n_patterns": n,
            "n_with_any_import": imp_any,
            "pct_with_any_import": units.pct(imp_any, n),
            "n_aligned_imports": a_imp,
            "pct_aligned_imports": units.pct(a_imp, n),
            "n_foundational_imports": f_imp,
            "pct_foundational_imports": units.pct(f_imp, n),
            "n_aligned_refs": a_ref,
            "pct_aligned_refs": units.pct(a_ref, n),
            "n_foundational_refs": f_ref,
            "pct_foundational_refs": units.pct(f_ref, n),
            "n_aligned_transitive": a_tr,
            "pct_aligned_transitive": units.pct(a_tr, n),
            "n_foundational_transitive": f_tr,
            "pct_foundational_transitive": units.pct(f_tr, n),
            "n_imports_unresolved_total": sum(p["n_imports_unresolved"] for p in sub),
            "n_patterns_with_unresolved_imports": sum(
                1 for p in sub if p["n_imports_unresolved"] > 0),
            "n_aligned_both_signals": both,
            "n_aligned_either_signal": either,
            "n_aligned_none": n - either,
            "pct_aligned_none": units.pct(n - either, n),
            "n_with_artefact_iris": artefact,
            "pct_with_artefact_iris": units.pct(artefact, n),
        })
    units.write_csv(out_dir / "tl_summary.csv", summary)

    # ---- unrecognised namespaces -------------------------------------
    unk_rows = []
    for ns, keys in sorted(unknown_ns.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        unk_rows.append({
            "namespace": ns,
            "n_patterns": len(keys),
            "libraries": "|".join(f"{k}:{v}" for k, v in sorted(unknown_ns_lib[ns].items())),
        })
    units.write_csv(out_dir / "tl_unrecognised_ns.csv", unk_rows)

    return {
        "per_pattern": per_pattern,
        "by_library": rows,
        "summary": summary,
        "unrecognised": unk_rows,
    }
