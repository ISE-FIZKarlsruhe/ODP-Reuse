"""Render the CSVs in an output directory as markdown tables on stdout.

    python -m pipeline.patternstats.report --out pipeline/out/patternstats

Nothing is computed here -- every figure is read back from the CSVs that
``python -m pipeline.patternstats`` wrote, so the printed tables and the
committed data cannot drift apart.  Redirect to a file if you want one; this
module deliberately does not write anything.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def table(rows: list[dict], cols: list[str], headers: list[str] | None = None,
          align_right: set[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    headers = headers or cols
    ar = align_right or {c for c in cols if c not in ("library", "library_name",
                                                      "domain", "vocabulary",
                                                      "signal", "tier", "stem")}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---:" if c in ar else "---" for c in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def section(title: str) -> str:
    return f"\n## {title}\n\n"


#: The published Table "Pattern libraries" as it stands in
#: paper-iri/tables/pattern_libraries.tex -- quoted, not recomputed, so that the
#: comparison below is against what the manuscript actually says.
PUBLISHED = {
    "ODPA": (117, 21.73, 23.61, 1.43),
    "MANCHESTER": (17, 18.76, 3.06, 1.12),
    "DOSDP": (471, 379.26, 1.77, 0.00),
    "ISODP": (10, 55.50, 17.60, 9.70),
}


def paper_table(d: Path) -> str:
    """Published Table 1 against the recomputation, at the unit that matches it.

    The published Avg C counts every ``rdf:type owl:Class`` triple, anonymous
    class expressions included; the matching column here is
    ``n_class_expressions``.  The unit that matches is ``file`` -- one row per
    RDF file -- except that the published ODPA *denominator* is not the number
    of ODPA files.
    """
    rows = {r["library"]: r for r in read(d / "sh_by_library.csv")
            if r["unit"] == "file"}
    inv = {r["library"]: r for r in read(d / "sh_units.csv")}
    out = ["| Library | Published n | Published C / OP / DP | Recomputed n (files) "
           "| Recomputed C_expr / OP / DP | Named C |",
           "|---|---:|---|---:|---|---:|"]
    for lib, (pn, pc, pop, pdp) in PUBLISHED.items():
        r = rows.get(lib)
        if not r:
            continue
        out.append(
            f"| {r['library_name']} | {pn} | {pc} / {pop} / {pdp} "
            f"| {r['n_patterns']} "
            f"| {r['n_class_expressions__mean']} / {r['n_object_properties__mean']}"
            f" / {r['n_datatype_properties__mean']} "
            f"| {r['n_classes__mean']} |")
    out.append("")
    out.append("Inventory for reference (files / parsed / unique content / unique stem):")
    out.append("")
    out.append(table([inv[k] for k in PUBLISHED if k in inv],
                     ["library_name", "n_rdf_files", "n_parsed", "n_unique_content",
                      "n_unique_stem", "n_primary_unique_content", "n_groups"]))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("pipeline/out/patternstats"))
    ap.add_argument("--unit", default="primary_content")
    a = ap.parse_args()
    d = a.out

    print("# Pattern-library descriptive statistics\n")

    print(section("Inventory (sh_units.csv)"))
    print(table(read(d / "sh_units.csv"),
                ["library_name", "n_rdf_files", "n_parsed", "n_parse_failures",
                 "n_unique_content", "n_unique_stem", "n_primary",
                 "n_shipped_dependency", "n_primary_unique_content", "total_mb"]))

    print(section("Published Table 1 vs recomputation"))
    print(paper_table(d))

    print(section("Size and shape (sh_by_library.csv)"))
    rows = [r for r in read(d / "sh_by_library.csv") if r["unit"] == a.unit]
    print(table(rows, ["library_name", "n_patterns", "n_classes__mean",
                       "n_object_properties__mean", "n_datatype_properties__mean",
                       "n_triples__mean", "n_triples_logical__mean",
                       "n_axiom_shapes__mean", "pct_connected_core"]))

    print(section("Top-level alignment (tl_summary.csv)"))
    print(table(read(d / "tl_summary.csv"),
                ["library_name", "n_patterns", "pct_with_any_import",
                 "n_foundational_imports", "n_aligned_imports",
                 "n_foundational_transitive", "n_aligned_transitive",
                 "n_foundational_refs", "n_aligned_refs",
                 "n_aligned_either_signal", "n_aligned_none",
                 "n_patterns_with_unresolved_imports",
                 "n_imports_unresolved_total"]))

    print(section("Alignment targets (tl_by_library.csv, alignment tiers only)"))
    rows = [r for r in read(d / "tl_by_library.csv")
            if r["tier"] in ("foundational", "cross_domain")]
    print(table(rows, ["library_name", "signal", "vocabulary", "tier",
                       "n_patterns", "pct_of_library"]))

    print(section("Inter-library overlap"))
    print("### Cross-library stem collisions (ov_cross_library.csv)\n")
    print(table(read(d / "ov_cross_library.csv"),
                ["stem", "libraries", "n_files", "n_distinct_contents",
                 "identical_across_libraries"]))
    print("\n### Jaccard, cross-library (ov_jaccard_summary.csv)\n")
    print(table(read(d / "ov_jaccard_summary.csv"),
                ["library_a", "library_b", "n_pairs_total", "n_pairs_jaccard_gt0",
                 "pct_pairs_jaccard_gt0", "max_jaccard",
                 "mean_jaccard_over_nonzero"]))
    print("\n### Cross-library IRI references (ov_cross_refs.csv)\n")
    print(table(read(d / "ov_cross_refs.csv"),
                ["referencing_library", "referenced_library", "same_library",
                 "n_referencing_patterns", "n_distinct_referenced_iris"]))

    print(section("Domains (dm_by_library.csv)"))
    print(table([r for r in read(d / "dm_by_library.csv") if r["n_patterns"] != "0"],
                ["library_name", "domain", "n_patterns", "pct_of_library"]))

    print(section("Domain classifier uncertainty (dm_uncertainty.csv)"))
    print(table(read(d / "dm_uncertainty.csv"),
                ["library_name", "n_patterns", "pct_unclassified", "pct_ambiguous",
                 "mean_margin", "n_with_vocab_hint", "pct_vocab_hint_agrees",
                 "pct_stem_agrees_with_combined",
                 "pct_entity_agrees_with_combined"]))

    print(section("Domain x reuse (dm_reuse_crosstab.csv)"))
    print(table(read(d / "dm_reuse_crosstab.csv"),
                ["domain", "n_patterns", "n_patterns_reused", "pct_patterns_reused",
                 "n_reuse_pairs", "pct_of_all_pairs", "pairs_per_pattern",
                 "n_distinct_reusing_ontologies", "n_complete_pairs"]))

    print(section("Reuse-join coverage (dm_reuse_coverage.csv)"))
    print(table(read(d / "dm_reuse_coverage.csv"),
                ["unit", "n_reused_stems", "n_reused_stems_matched",
                 "n_reused_stems_unmatched", "n_pairs_total", "n_pairs_in_crosstab",
                 "pct_pairs_in_crosstab", "n_pairs_dropped_no_library_file",
                 "n_pairs_dropped_by_unit", "n_ambiguous_stems"]))

    print(section("Content kinds — what the files actually are (sh_content_kinds.csv)"))
    print(table(read(d / "sh_content_kinds.csv"),
                ["library_name", "content_kind", "n_files", "n_parsed", "n_failed"]))

    print(section("Connectivity (sh_by_library.csv)"))
    rows = [r for r in read(d / "sh_by_library.csv") if r["unit"] == a.unit]
    print(table(rows, ["library_name", "n_patterns", "n_core_defined",
                       "pct_connected_core_where_defined", "n_declared_defined",
                       "pct_connected_declared_where_defined",
                       "n_with_zero_own_core", "n_with_zero_entities_wide"]))


if __name__ == "__main__":
    main()
