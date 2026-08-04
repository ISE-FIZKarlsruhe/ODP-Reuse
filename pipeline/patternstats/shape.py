"""Analysis 4 -- size and shape per library, extending the pattern-library table.

Declaration counts reproduce ``MultiSource/odp_stats.py::collect_defined_entities``
exactly (``owl:Class`` union ``rdfs:Class``; ``owl:ObjectProperty``;
``owl:DatatypeProperty``; URIRef subjects only), so the ``Avg C / OP / DP``
columns of the manuscript's Table "Pattern libraries" can be checked line for
line against ``sh_by_library.csv`` at ``unit=file``.

AXIOMS.  rdflib gives triples, not OWL axioms, and no OWL reasoner is used
anywhere in this repository, so no true axiom count is available.  Three proxies
are reported instead, all exactly defined:

    n_triples            all triples in the file as parsed
    n_triples_logical    after ``odpqual.entities.strip_annotations`` -- the
                         annotation vocabularies, literal-valued triples and
                         ``owl:Ontology`` header triples removed.  This is the
                         same graph the reuse pipeline measures, so it is the
                         proxy to quote alongside any RQ number.
    n_axiom_shapes       the sum of the individually counted constructs
                         (subClassOf, subPropertyOf, equivalentClass,
                         disjointWith, someValuesFrom, allValuesFrom,
                         cardinality forms, inverseOf, domain, range,
                         intersectionOf, unionOf, propertyChainAxiom).  Closest
                         to "logical axioms", still not identical to one.

CONNECTEDNESS.  Three variants, because the answer depends entirely on which
entities are asked about (see ``extract._connectivity`` for the graph):

    connected_core       own-namespace classes and object properties only.
                         The headline number.  The ODP submission template
                         declares its metadata fields (``aim``,
                         ``also_known_as``, ``additional_information``, ...) as
                         ``owl:DatatypeProperty``, and those are isolated by
                         construction; including them would report nearly every
                         ODPA pattern as disconnected for a reason that has
                         nothing to do with its modelling.
    connected_own        all own-namespace declarations.
    connected_declared   every declaration in the file, imports included.

Outputs
-------
sh_per_pattern.csv   one row per pattern, every count
sh_by_library.csv    per library: n, mean/median/min/max for each measure
sh_units.csv         the four counting units side by side, per library
sh_parse_failures.csv every file that did not parse, with the parser message
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from pathlib import Path

from . import units
from .libraries import LIBRARIES, LIB_ORDER

AXIOM_SHAPE_FIELDS = (
    "n_subclassof", "n_subpropertyof", "n_equivalentclass", "n_disjointwith",
    "n_somevaluesfrom", "n_allvaluesfrom", "n_cardinality", "n_inverseof",
    "n_domain", "n_range", "n_intersection", "n_union", "n_propertychain",
)

MEASURES = (
    "n_classes", "n_class_expressions", "n_anon_class_expressions",
    "n_object_properties", "n_datatype_properties",
    "n_annotation_properties", "n_individuals",
    "n_triples", "n_triples_logical", "n_axiom_shapes",
    "n_entities_wide", "n_entities_narrow",
    "n_own_declared", "n_own_core", "n_bytes",
    "n_subclassof", "n_equivalentclass", "n_disjointwith", "n_restrictions",
    "n_somevaluesfrom", "n_allvaluesfrom", "n_cardinality", "n_inverseof",
    "n_domain", "n_range", "n_intersection", "n_union", "n_propertychain",
    "n_components_core", "n_components_own",
)


def _row(r: dict, unit: str) -> dict:
    row = {
        "unit": unit,
        "library": r["library"],
        "key": r["key"],
        "stem": r["stem"],
        "group": r["group"],
        "role": r["role"],
        "suffix": r["suffix"],
        "sha256": r["sha256"],
        "own_ns": r.get("own_ns", ""),
        "n_ontology_iris": len(r.get("ontology_iris", [])),
    }
    for f in MEASURES:
        if f == "n_axiom_shapes":
            continue
        row[f] = r.get(f, 0)
    row["n_axiom_shapes"] = sum(r.get(f, 0) for f in AXIOM_SHAPE_FIELDS)
    for f in ("connected_core", "connected_own", "connected_declared"):
        row[f] = int(bool(r.get(f)))
    for f in ("n_isolated_core", "largest_component_core", "n_struct_nodes",
              "n_struct_edges", "n_imports", "n_distinct_iris"):
        row[f] = r.get(f, 0) if f != "n_imports" else len(r.get("imports", []))
    return row


def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "sum": 0}
    return {
        "mean": round(st.mean(vals), 2),
        "median": round(st.median(vals), 2),
        "min": min(vals),
        "max": max(vals),
        "sum": round(sum(vals), 2),
    }


def run(records: list[dict], out_dir: Path, unit: str = units.DEFAULT_UNIT,
        also_units: tuple[str, ...] = units.UNITS) -> dict:
    parsed = units.parsed_only(records)

    per = [_row(r, unit) for r in units.select(records, unit)]
    units.write_csv(out_dir / "sh_per_pattern.csv", per)

    # ---- per-library aggregates, for every unit -----------------------
    rows = []
    for u in also_units:
        sub_all = units.select(records, u)
        for lib in LIB_ORDER:
            sub = [_row(r, u) for r in sub_all if r["library"] == lib]
            if not sub:
                continue
            base = {
                "unit": u,
                "library": lib,
                "library_name": LIBRARIES[lib][1],
                "n_patterns": len(sub),
                "n_connected_core": sum(r["connected_core"] for r in sub),
                "pct_connected_core": units.pct(sum(r["connected_core"] for r in sub), len(sub)),
                # connectivity is only a question where there are >= 2 nodes to
                # connect.  DOS-DP declares nothing in a namespace of its own, so
                # its core is empty and "connected" there is vacuously true;
                # these two columns are the ones to quote.
                "n_core_defined": sum(1 for r in sub if r["n_own_core"] >= 2),
                "n_connected_core_where_defined": sum(
                    r["connected_core"] for r in sub if r["n_own_core"] >= 2),
                # blank, not 0.0, when no pattern in the library has a core of
                # >= 2 nodes: connectivity is then undefined, not false
                "pct_connected_core_where_defined": (
                    units.pct(sum(r["connected_core"] for r in sub if r["n_own_core"] >= 2),
                              sum(1 for r in sub if r["n_own_core"] >= 2))
                    if any(r["n_own_core"] >= 2 for r in sub) else ""),
                "n_connected_own": sum(r["connected_own"] for r in sub),
                "pct_connected_own": units.pct(sum(r["connected_own"] for r in sub), len(sub)),
                "n_connected_declared": sum(r["connected_declared"] for r in sub),
                "pct_connected_declared": units.pct(sum(r["connected_declared"] for r in sub), len(sub)),
                "n_declared_defined": sum(
                    1 for r in sub
                    if r["n_classes"] + r["n_object_properties"] + r["n_datatype_properties"] >= 2),
                "pct_connected_declared_where_defined": units.pct(
                    sum(r["connected_declared"] for r in sub
                        if r["n_classes"] + r["n_object_properties"] + r["n_datatype_properties"] >= 2),
                    sum(1 for r in sub
                        if r["n_classes"] + r["n_object_properties"] + r["n_datatype_properties"] >= 2)),
                "n_with_zero_own_core": sum(1 for r in sub if r["n_own_core"] == 0),
                "n_with_zero_classes": sum(1 for r in sub if r["n_classes"] == 0),
                "n_with_zero_triples_logical": sum(1 for r in sub if r["n_triples_logical"] == 0),
                "n_with_zero_entities_wide": sum(1 for r in sub if r["n_entities_wide"] == 0),
                "pct_with_zero_entities_wide": units.pct(
                    sum(1 for r in sub if r["n_entities_wide"] == 0), len(sub)),
            }
            for m in MEASURES:
                a = _agg([r[m] for r in sub])
                base[f"{m}__mean"] = a["mean"]
                base[f"{m}__median"] = a["median"]
                base[f"{m}__min"] = a["min"]
                base[f"{m}__max"] = a["max"]
                base[f"{m}__sum"] = a["sum"]
            rows.append(base)
    units.write_csv(out_dir / "sh_by_library.csv", rows)

    # ---- inventory: the four counting units, side by side --------------
    inv = []
    all_files = records
    for lib in LIB_ORDER:
        lf = [r for r in all_files if r["library"] == lib]
        lp = [r for r in lf if r.get("parsed")]
        inv.append({
            "library": lib,
            "library_name": LIBRARIES[lib][1],
            "directory": LIBRARIES[lib][0],
            "n_rdf_files": len(lf),
            "n_parsed": len(lp),
            "n_parse_failures": len(lf) - len(lp),
            "n_unique_content": len({r["sha256"] for r in lp}),
            "n_unique_stem": len({r["stem"].casefold() for r in lp}),
            "n_primary": len([r for r in lp if r["role"] == "primary"]),
            "n_shipped_dependency": len([r for r in lp if r["role"] != "primary"]),
            "n_primary_unique_content": len({r["sha256"] for r in lp
                                             if r["role"] == "primary"}),
            "n_groups": len({r["group"] for r in lf if r["group"]}),
            "total_mb": round(sum(r["n_bytes"] for r in lf) / 1e6, 2),
            "suffixes": "|".join(f"{k}:{v}" for k, v in
                                 sorted(Counter(r["suffix"] for r in lf).items())),
            "content_kinds": "|".join(f"{k}:{v}" for k, v in
                                      sorted(Counter(r.get("content_kind", "")
                                                     for r in lf).items())),
            "n_html_masquerading_as_rdf": sum(1 for r in lf
                                              if r.get("content_kind") == "html"),
            "n_empty_files": sum(1 for r in lf if r.get("content_kind") == "empty"),
        })
    tot = {
        "library": "TOTAL", "library_name": "all four libraries", "directory": "",
        "n_rdf_files": sum(r["n_rdf_files"] for r in inv),
        "n_parsed": sum(r["n_parsed"] for r in inv),
        "n_parse_failures": sum(r["n_parse_failures"] for r in inv),
        "n_unique_content": len({r["sha256"] for r in parsed}),
        "n_unique_stem": len({r["stem"].casefold() for r in parsed}),
        "n_primary": sum(r["n_primary"] for r in inv),
        "n_shipped_dependency": sum(r["n_shipped_dependency"] for r in inv),
        "n_primary_unique_content": len({r["sha256"] for r in parsed
                                         if r["role"] == "primary"}),
        "n_groups": sum(r["n_groups"] for r in inv),
        "total_mb": round(sum(r["total_mb"] for r in inv), 2),
        "suffixes": "",
        "content_kinds": "|".join(f"{k}:{v}" for k, v in
                                  sorted(Counter(r.get("content_kind", "")
                                                 for r in records).items())),
        "n_html_masquerading_as_rdf": sum(1 for r in records
                                          if r.get("content_kind") == "html"),
        "n_empty_files": sum(1 for r in records if r.get("content_kind") == "empty"),
    }
    inv.append(tot)
    units.write_csv(out_dir / "sh_units.csv", inv)

    # ---- parse failures ------------------------------------------------
    fails = [{
        "library": r["library"], "key": r["key"], "stem": r["stem"],
        "role": r["role"], "n_bytes": r["n_bytes"],
        "content_kind": r.get("content_kind", ""),
        "error_type": (r.get("error", "") or "").split(":", 1)[0],
        "error": r.get("error", ""),
    } for r in records if not r.get("parsed")]
    fails.sort(key=lambda r: (r["library"], r["content_kind"], r["key"]))
    units.write_csv(out_dir / "sh_parse_failures.csv", fails)

    # failure taxonomy: what the file actually was, by library
    ftax = []
    for lib in LIB_ORDER:
        lf = [r for r in records if r["library"] == lib]
        if not lf:
            continue
        for kind in sorted({r.get("content_kind", "") for r in lf}):
            sub = [r for r in lf if r.get("content_kind") == kind]
            ok = sum(1 for r in sub if r.get("parsed"))
            ftax.append({
                "library": lib, "library_name": LIBRARIES[lib][1],
                "content_kind": kind, "n_files": len(sub),
                "n_parsed": ok, "n_failed": len(sub) - ok,
                "pct_failed": units.pct(len(sub) - ok, len(sub)),
            })
    units.write_csv(out_dir / "sh_content_kinds.csv", ftax)

    return {"per_pattern": per, "by_library": rows, "inventory": inv,
            "failures": fails, "content_kinds": ftax}
